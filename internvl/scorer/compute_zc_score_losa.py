# --------------------------------------------------------
# InternVL - 使用LoSA重要性评估方法
# Copyright (c) 2024 OpenGVLab
# --------------------------------------------------------

import logging
import os
import sys
import warnings
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def compute_rmi_importance_attention_layers(model, dataloader, device, max_batch_num):
    """
    简化版RMI计算：只计算同一类型内维度相同的层之间的相关性
    """
    
    layer_groups = {
        'qkv': ['attention.wqkv', 'attn.qkv'],
        'output': ['attention.wo', 'attn.proj']
    }
    
    layer_activations = defaultdict(list)
    hooks = []
    
    def activation_hook(name, layer_type):
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    output = output[0]
                
                act = output.detach().float()
                
                if layer_type == 'qkv':
                    hidden_dim = act.shape[-1] // 3
                    act = act[..., :hidden_dim]
                
                act = act.mean(dim=1)
                layer_activations[f"{layer_type}:{name}"].append(act.cpu())
        return hook
    
    model.eval()
    for name, module in model.named_modules():
        for layer_type, target_patterns in layer_groups.items():
            if any(name.endswith(target) for target in target_patterns):
                hooks.append(module.register_forward_hook(activation_hook(name, layer_type)))
                break
    
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(dataloader, total=max_batch_num)):
            if i >= max_batch_num:
                break
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            _ = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    all_importance_scores = {}
    all_layer_correlations = {}
    
    for layer_type in layer_groups.keys():
        type_layer_names = [name for name in layer_activations.keys() if name.startswith(f"{layer_type}:")]
        
        if len(type_layer_names) <= 1:
            for full_name in type_layer_names:
                original_name = full_name.split(':', 1)[1]
                all_importance_scores[original_name] = 0.5
            continue
        
        # ⭐ 按维度分组处理
        dim_groups = {}
        for full_name in type_layer_names:
            acts = torch.cat(layer_activations[full_name], dim=0)
            dim = acts.shape[-1]
            if dim not in dim_groups:
                dim_groups[dim] = []
            dim_groups[dim].append(full_name)
        
        # 分别处理每个维度组
        for dim, names_in_group in dim_groups.items():
            if len(names_in_group) <= 1:
                for full_name in names_in_group:
                    original_name = full_name.split(':', 1)[1]
                    all_importance_scores[original_name] = 0.5
                continue
            
            logger.info(f"Processing {layer_type} layers with dim={dim}: {len(names_in_group)} layers")
            
            # 收集并归一化激活值
            processed = {}
            for full_name in names_in_group:
                acts = torch.cat(layer_activations[full_name], dim=0)
                acts_norm = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
                processed[full_name] = acts_norm
            
            # 计算相关性矩阵
            n = len(names_in_group)
            corr_matrix = torch.zeros(n, n)
            
            for i, name_i in enumerate(names_in_group):
                for j, name_j in enumerate(names_in_group):
                    act_i = processed[name_i]
                    act_j = processed[name_j]
                    min_samples = min(act_i.shape[0], act_j.shape[0])
                    similarity = (act_i[:min_samples] * act_j[:min_samples]).sum(dim=1).mean()
                    corr_matrix[i, j] = similarity
            
            # 计算重要性
            avg_corr = corr_matrix.mean(dim=1)
            for i, full_name in enumerate(names_in_group):
                original_name = full_name.split(':', 1)[1]
                importance = max(0, 1 - avg_corr[i].item())
                all_importance_scores[original_name] = importance
                all_layer_correlations[original_name] = avg_corr[i].item()
                
                logger.info(f"  {original_name}: Corr={avg_corr[i].item():.4f}, Importance={importance:.4f}")
    
    return all_importance_scores, all_layer_correlations, {}


def compute_enhanced_rmi_importance(model, dataloader, device, max_batch_num):
    """
    增强版RMI计算：结合HSIC (Hilbert-Schmidt Independence Criterion)
    更精确地衡量层间独立性
    """
    
    layer_activations = defaultdict(list)
    target_modules = ["attention.wqkv", "attention.wo", "attn.qkv", "attn.proj"]
    
    # 注册hook
    hooks = []
    
    def rbf_kernel(x, sigma=None):
        """RBF核函数，用于HSIC计算"""
        n = x.size(0)
        pairwise_dist = torch.cdist(x, x, p=2).pow(2)
        if sigma is None:
            sigma = pairwise_dist.median()
        kernel = torch.exp(-pairwise_dist / (2 * sigma**2 + 1e-8))
        return kernel
    
    def hsic(x, y):
        """计算HSIC (Hilbert-Schmidt Independence Criterion)"""
        n = x.size(0)
        K = rbf_kernel(x)
        L = rbf_kernel(y)
        
        H = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n
        KH = K @ H
        LH = L @ H
        
        hsic_val = torch.trace(KH @ LH) / (n - 1)**2
        return hsic_val
    
    def activation_hook(name):
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    output = output[0]
                
                act = output.detach().float()
                # 降维：取序列维度平均
                act = act.mean(dim=1)  # [batch, hidden_dim]
                
                # 可选：PCA降维到更低维度（加速HSIC计算）
                # 这里简单取前256维
                if act.size(-1) > 256:
                    act = act[..., :256]
                
                layer_activations[name].append(act.cpu())
        return hook
    
    model.eval()
    for name, module in model.named_modules():
        if any(name.endswith(target) for target in target_modules):
            hooks.append(module.register_forward_hook(activation_hook(name)))
    
    # 收集激活值
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(dataloader, total=max_batch_num)):
            if i >= max_batch_num:
                break
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            _ = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    # 处理激活值
    layer_names = list(layer_activations.keys())
    processed_acts = {}
    for name in layer_names:
        acts = torch.cat(layer_activations[name], dim=0)
        # 标准化
        acts = (acts - acts.mean(dim=0, keepdim=True)) / (acts.std(dim=0, keepdim=True) + 1e-8)
        processed_acts[name] = acts
    
    # 使用HSIC计算层间独立性
    # HSIC值越大表示越相关，所以重要性 = 1 - normalized_HSIC
    importance_scores = {}
    
    for i, name_i in enumerate(layer_names):
        total_hsic = 0
        for j, name_j in enumerate(layer_names):
            if i != j:
                # 计算HSIC
                hsic_val = hsic(processed_acts[name_i], processed_acts[name_j])
                total_hsic += hsic_val.item()
        
        # 平均HSIC并归一化
        avg_hsic = total_hsic / (len(layer_names) - 1)
        # 归一化到[0,1]区间
        norm_hsic = 1 / (1 + avg_hsic)  # HSIC越大，归一化值越小
        importance_scores[name_i] = norm_hsic
        
        logger.debug(f"Layer: {name_i}, HSIC: {avg_hsic:.6f}, Importance: {norm_hsic:.4f}")
    
    return importance_scores


def compute_layer_sparsity_from_rmi(importance_scores, global_sparsity=0.5, temperature=1.0):
    """
    根据RMI重要性分数分配每层稀疏率
    
    LoSA策略：重要性高的层获得更低的稀疏率（保留更多参数）
    
    Args:
        importance_scores: 层名 -> 重要性分数
        global_sparsity: 全局目标稀疏率
        temperature: 温度参数，控制分配的非均匀程度
    
    Returns:
        layer_sparsity: 层名 -> 稀疏率
    """
    if not importance_scores:
        return {}
    
    # 重要性分数归一化
    scores = torch.tensor(list(importance_scores.values()))
    weights = F.softmax(scores / temperature, dim=0)
    
    # 重要性高的层获得更低的稀疏率
    # 稀疏率分布：weight_i 越小，稀疏率越大
    # sparsity_i = global_sparsity * (1 - weight_i * (1 - min_sparsity/global_sparsity))
    min_sparsity = global_sparsity * 0.3  # 最低稀疏率为全局的30%
    
    layer_sparsity = {}
    layer_names = list(importance_scores.keys())
    
    for i, name in enumerate(layer_names):
        # 重要性权重越低，稀疏率越高
        sparsity = global_sparsity * (1 - weights[i].item() * 0.7)
        sparsity = max(min_sparsity, min(global_sparsity, sparsity))
        layer_sparsity[name] = sparsity
    
    return layer_sparsity


def compute_adaptive_rank_allocation(reconstruction_errors, base_rank=8):
    """
    基于重构误差动态分配LoRA秩（LoSA策略）
    
    Args:
        reconstruction_errors: 层名 -> 重构误差
        base_rank: 基础秩
    
    Returns:
        layer_ranks: 层名 -> 分配的秩
    """
    if not reconstruction_errors:
        return {}
    
    errors = list(reconstruction_errors.values())
    avg_error = np.mean(errors)
    
    layer_ranks = {}
    for name, err in reconstruction_errors.items():
        # 重构误差大的层获得更高的秩
        rank_scale = 1 + (err - avg_error) / (avg_error + 1e-8)
        rank = max(1, min(base_rank * 2, round(base_rank * rank_scale)))
        layer_ranks[name] = rank
    
    return layer_ranks


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
    logger.info("Finished")

    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(
        f"model.config.vision_config.image_size: {model.config.vision_config.image_size}"
    )
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f"Resizing position embedding from "
            f"{model.config.vision_config.image_size} "
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
        * (data_args.down_sample_ratio**2)
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
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

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

    rank0_print("Model trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()

    # ============================================================
    # ⭐ 使用LoSA的RMI方法计算注意力层重要性
    # ============================================================
    logger.info("=" * 60)
    logger.info("Computing LoSA RMI (Representation Mutual Information) importance...")
    logger.info("=" * 60)

    portion = model_args.zc_proxy_score_portion
    max_batch_num = int(len(train_dataloader) * portion + 0.5)

    logger.info(f"Using {max_batch_num} batches for RMI computation")

    model = trainer._wrap_model(model).to(trainer.args.device)

    # 选择使用标准RMI还是增强版HSIC-RMI
    use_hsic = getattr(model_args, 'use_hsic_rmi', False)
    
    if use_hsic:
        logger.info("Using Enhanced HSIC-RMI method...")
        importance_scores = compute_enhanced_rmi_importance(
            model, train_dataloader, trainer.args.device, max_batch_num
        )
    else:
        logger.info("Using Standard RMI method (Cosine Similarity based)...")
        importance_scores, layer_correlations, activations = compute_rmi_importance_attention_layers(
            model, train_dataloader, trainer.args.device, max_batch_num
        )
    
    # 分布式同步
    # if dist.is_initialized():
    #     dist.barrier()
        
    #     # 收集所有进程的重要性分数
    #     all_scores = []
    #     if dist.get_rank() == 0:
    #         # 主进程收集所有分数
    #         for name in importance_scores:
    #             score_tensor = torch.tensor([importance_scores[name]], device=model.device)
    #             all_scores.append(score_tensor)
    #     else:
    #         # 其他进程发送分数
    #         for name in importance_scores:
    #             score_tensor = torch.tensor([importance_scores[name]], device=model.device)
    #             dist.send(tensor=score_tensor, dst=0)

    # Distributed synchronization
    if dist.is_initialized():
        dist.barrier()
        for name in importance_scores:
            score_tensor = torch.tensor([importance_scores[name]], device=model.device)
            dist.all_reduce(score_tensor, op=dist.ReduceOp.AVG)
            importance_scores[name] = score_tensor.item()
    
    # 计算稀疏配置（可选）
    # global_sparsity = getattr(model_args, 'global_sparsity', 0.5)
    # layer_sparsity = compute_layer_sparsity_from_rmi(
    #     importance_scores, global_sparsity, temperature=1.0
    # )
    
    # 保存结果
    
    if dist.get_rank() == 0:
        # 按重要性排序
        importance_list = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Successfully computed RMI scores for {len(importance_list)} layers")
        logger.info("\n Top-10 most important layers (lowest correlation):")
        for i, (layer, score) in enumerate(importance_list[:10]):
            logger.info(f"  {i+1}. {layer}: RMI={score:.6f}")
        
        logger.info("\n Bottom-10 least important layers (highest correlation):")
        for i, (layer, score) in enumerate(importance_list[-10:]):
            logger.info(f"  {i+1}. {layer}: RMI={score:.6f}")
        
        # 保存重要性分数
        os.makedirs(os.path.dirname(model_args.zc_proxy_score_save_path), exist_ok=True)
        df = DataFrame(importance_list, columns=["layer", "rmi_importance"])
        df.to_csv(model_args.zc_proxy_score_save_path, index=False)
        logger.info(f"RMI importance scores saved to {model_args.zc_proxy_score_save_path}")
   
        # # 保存稀疏配置
        # if layer_sparsity:
        #     sparsity_list = [(name, layer_sparsity[name]) for name in importance_scores.keys()]
        #     sparsity_list_sorted = sorted(sparsity_list, key=lambda x: x[1], reverse=True)
        #     df_sparsity = DataFrame(sparsity_list_sorted, columns=["layer", "sparsity_rate"])
        #     sparsity_path = model_args.zc_proxy_score_save_path.replace('.csv', '_sparsity.csv')
        #     df_sparsity.to_csv(sparsity_path, index=False)
        #     logger.info(f"Sparsity configuration saved to {sparsity_path}")
            
        #     # 打印稀疏配置统计
        #     sparsities = [s for _, s in sparsity_list]
        #     logger.info(f"\n Sparsity Statistics:")
        #     logger.info(f"  Min: {min(sparsities):.3f}")
        #     logger.info(f"  Max: {max(sparsities):.3f}")
        #     logger.info(f"  Mean: {sum(sparsities)/len(sparsities):.3f}")
        
        # # 保存层间相关性矩阵（如果可用）
        # if not use_hsic and 'layer_correlations' in dir():
        #     corr_list = [(name, layer_correlations[name]) for name in importance_scores.keys()]
        #     corr_list_sorted = sorted(corr_list, key=lambda x: x[1], reverse=True)
        #     df_corr = DataFrame(corr_list_sorted, columns=["layer", "avg_correlation"])
        #     corr_path = model_args.zc_proxy_score_save_path.replace('.csv', '_correlation.csv')
        #     df_corr.to_csv(corr_path, index=False)
        #     logger.info(f"Layer correlation matrix saved to {corr_path}")
    
    logger.info("LoSA RMI importance computation completed!")



if __name__ == "__main__":
    main()