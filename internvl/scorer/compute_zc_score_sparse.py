# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import os
import sys
import warnings
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
import transformers
from pandas import DataFrame
from peft import PeftModel
from PIL import Image, ImageFile, PngImagePlugin
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)

from internvl.dist_utils import init_dist, rank0_print
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
from internvl.patch import (
    concat_pad_data_collator,
    replace_internlm2_attention_class,
    replace_llama_attention_class,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_phi3_attention_class,
    replace_qwen2_attention_class,
    replace_train_dataloader,
    replace_train_sampler,
)
from internvl.train.arguments import DataTrainingArguments, ModelArguments
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from internvl.train.dataset import build_datasets
from internvl.train.dataset_packed import packed_collate_fn
from internvl.train.trainer import CustomTrainer

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)



def compute_sparselora_attention_importance(model, train_dataloader, max_batch_num, device):
    """
    针对注意力层的SparseLoRA重要性计算
    
    对于QKV层：使用Q和K的L2范数乘积
    对于输出投影层：使用激活值的L2范数
    """
    
    # 存储不同类型的激活值
    qkv_importance = {}      # 存储QK乘积分数
    output_importance = {}   # 存储输出投影的L2范数
    
    def get_qkv_importance_hook(name):
        """处理QKV层：计算Q和K的L2范数乘积"""
        def hook(module, input, output):
            # output shape: [batch, seq_len, 3*hidden_dim]
            # 假设hidden_dim = output.shape[-1] // 3
            with torch.no_grad():
                hidden_dim = output.shape[-1] // 3
                
                # 分离Q和K
                q = output[..., :hidden_dim]      # Query投影
                k = output[..., hidden_dim:2*hidden_dim]  # Key投影
                
                # 计算L2范数（沿着特征维度）
                q_norm = torch.norm(q, p=2, dim=-1).mean().item()
                k_norm = torch.norm(k, p=2, dim=-1).mean().item()
                
                # SparseLoRA核心：QK范数乘积
                importance_score = q_norm * k_norm
                
                # 更新重要性分数（移动平均）
                if name not in qkv_importance:
                    qkv_importance[name] = importance_score
                else:
                    qkv_importance[name] = (qkv_importance[name] + importance_score) / 2
        return hook
    
    def get_output_importance_hook(name):
        """处理输出投影层：使用激活值的L2范数"""
        def hook(module, input, output):
            # output shape: [batch, seq_len, hidden_dim]
            with torch.no_grad():
                # 计算激活值的L2范数
                l2_norm = torch.norm(output, p=2, dim=-1).mean().item()
                
                if name not in output_importance:
                    output_importance[name] = l2_norm
                else:
                    output_importance[name] = (output_importance[name] + l2_norm) / 2
        return hook
    
    # 注册hooks
    hooks = []
    target_qkv_modules = ["attention.wqkv", "attn.qkv"]
    target_output_modules = ["attention.wo", "attn.proj"]
    
    for name, module in model.named_modules():
        # 处理QKV层
        if any(name.endswith(target) for target in target_qkv_modules):
            hooks.append(module.register_forward_hook(get_qkv_importance_hook(name)))
            print(f"Registered QKV hook for: {name}")
        
        # 处理输出投影层
        elif any(name.endswith(target) for target in target_output_modules):
            hooks.append(module.register_forward_hook(get_output_importance_hook(name)))
            print(f"Registered output hook for: {name}")
    
    # 执行前向传播（不需要梯度）
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(train_dataloader):
            if i >= max_batch_num:
                break
            
            # 准备输入
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # 前向传播
            _ = model(**inputs)
            
            # 可选：定期打印进度
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{max_batch_num} batches")
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 合并所有重要性分数
    all_importance = {}
    all_importance.update(qkv_importance)
    all_importance.update(output_importance)
    
    return all_importance, qkv_importance, output_importance


def compute_attention_sparsity_config(importance_scores, sparsity_ratio=0.5):
    """
    根据SparseLoRA的重要性分数，为注意力层配置稀疏率
    """
    if not importance_scores:
        return {}
    
    # 按重要性排序
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    total_layers = len(sorted_layers)
    
    sparsity_config = {}
    
    for idx, (layer_name, score) in enumerate(sorted_layers):
        percentile = idx / total_layers
        
        # SparseLoRA的动态稀疏率分配
        if percentile < 0.3:      # 前30%最重要的层
            sparsity = 0.1        # 保持稠密
            priority = "high"
        elif percentile < 0.7:    # 中间40%
            sparsity = 0.3        # 中等稀疏
            priority = "medium"
        else:                      # 后30%最不重要的层
            sparsity = 0.6        # 高稀疏
            priority = "low"
        
        # 区分QKV层和输出投影层
        layer_type = "qkv" if any(x in layer_name for x in ["wqkv", "qkv"]) else "output"
        
        sparsity_config[layer_name] = {
            'sparsity': sparsity,
            'importance_score': score,
            'rank': idx,
            'priority': priority,
            'type': layer_type
        }
    
    return sparsity_config


def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from liger_kernel.transformers import (
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_qwen2,
        )
        from internvl.patch import apply_liger_kernel_to_internvit

        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()

    # Load or initialize model
    if model_args.model_name_or_path is not None:
        logger.info("Loading InternVLChatModel...")
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for InternLM")
        else:
            config.llm_config._attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for LLaMA")
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config
        )
    else:
        logger.info("Loading ViT-6B...")
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config
        )
        logger.info("Loading LLaMA...")
        llm_config = AutoConfig.from_pretrained(
            model_args.llm_path, trust_remote_code=True
        )
        if llm_config.model_type == "internlm2":
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for InternLM")
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for LLaMA")
        llm = model_type.from_pretrained(
            model_args.llm_path,
            torch_dtype=torch.bfloat16,
            config=llm_config,
            trust_remote_code=True,
        )
        logger.info("Building InternVLChatConfig...")
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
        )
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info("Building InternVLChatModel...")
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)

    model.img_context_token_id = img_context_token_id
    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info("Loading pretrained MLP projector...")
        state_dict = torch.load(model_args.mlp_path, map_location="cpu")
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)

    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f"Resizing position embedding from {model.config.vision_config.image_size} "
            f"to {data_args.force_image_size}..."
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio ** 2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    # Build dataset and trainer
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone and not isinstance(model.vision_model, PeftModel):
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm and not isinstance(model.language_model, PeftModel):
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

    # Log trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()

    # Compute SparseLoRA importance for attention layers
    logger.info("Computing SparseLoRA importance for attention layers...")
    portion = model_args.zc_proxy_score_portion
    max_batch_num = int(len(train_dataloader) * portion + 0.5)
    logger.info(f"Processing {max_batch_num} batches for importance computation")

    importance_scores, qkv_scores, output_scores = compute_sparselora_attention_importance(
        model=model,
        train_dataloader=train_dataloader,
        max_batch_num=max_batch_num,
        device=training_args.device
    )

    # Distributed synchronization
    if dist.is_initialized():
        dist.barrier()
        for name in importance_scores:
            score_tensor = torch.tensor([importance_scores[name]], device=model.device)
            dist.all_reduce(score_tensor, op=dist.ReduceOp.AVG)
            importance_scores[name] = score_tensor.item()

    # Compute sparsity configuration (optional)
    sparsity_ratio = getattr(model_args, 'sparsity_ratio', 0.5)
    sparsity_config = compute_attention_sparsity_config(importance_scores, sparsity_ratio)

    # Save results
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(model_args.zc_proxy_score_save_path), exist_ok=True)

        # Save all importance scores
        score_list = [(name, score) for name, score in importance_scores.items()]
        score_list_sorted = sorted(score_list, key=lambda x: x[1], reverse=True)
        df = DataFrame(score_list_sorted, columns=["layer", "importance_score"])
        df.to_csv(model_args.zc_proxy_score_save_path, index=False)

        # Save QKV and output scores separately
        # if qkv_scores:
        #     qkv_df = DataFrame(sorted(qkv_scores.items(), key=lambda x: x[1], reverse=True),
        #                        columns=["layer", "qkv_importance"])
        #     qkv_df.to_csv(model_args.zc_proxy_score_save_path.replace('.csv', '_qkv.csv'), index=False)

        # if output_scores:
        #     output_df = DataFrame(sorted(output_scores.items(), key=lambda x: x[1], reverse=True),
        #                           columns=["layer", "output_importance"])
        #     output_df.to_csv(model_args.zc_proxy_score_save_path.replace('.csv', '_output.csv'), index=False)

        # Save sparsity configuration
        # if sparsity_config:
        #     config_list = [
        #         [name, config['sparsity'], config['importance_score'],
        #          config['rank'], config['priority'], config['type']]
        #         for name, config in sparsity_config.items()
        #     ]
        #     config_df = DataFrame(config_list,
        #                           columns=["layer", "sparsity", "importance_score",
        #                                    "rank", "priority", "type"])
        #     config_df.to_csv(model_args.zc_proxy_score_save_path.replace('.csv', '_sparsity.csv'), index=False)

        logger.info(f"SparseLoRA importance scores saved to {model_args.zc_proxy_score_save_path}")

        # Print top-10 most important layers
        logger.info("Top-10 most important attention layers:")
        for i, (layer, score) in enumerate(score_list_sorted[:10]):
            logger.info(f"  {i+1}. {layer}: {score:.6f}")

    logger.info("SparseLoRA importance computation completed.")


if __name__ == "__main__":
    main()