# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import os
import sys
import warnings
from functools import partial

import torch
import torch.distributed as dist
import transformers
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

from internvl.dist_utils import init_dist
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
from internvl.train.dataset import build_datasets, process_inputs
from internvl.train.dataset_packed import packed_collate_fn
from internvl.train.trainer import CustomTrainer

# Set constants for image processing and logging
IGNORE_INDEX = -100  # 用于标记需要忽略的token索引（在计算损失时忽略）
Image.MAX_IMAGE_PIXELS = None   # 取消PIL图像像素限制，允许处理大图像
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载被截断的图像文件
MaximumDecompressedSize = 1024  # 最大解压缩大小（MB）
MegaByte = 2**20    # 1MB的字节数
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte   # 设置PNG文本块最大大小

warnings.filterwarnings("ignore")   # 忽略警告信息
logger = logging.getLogger(__name__)    # 获取logger实例

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 启用tokenizer并行处理


def len2weight(x, loss_reduction):
    # 根据序列长度计算权重，支持不同的损失缩减策略
    # - "token": 每个token权重相同
    # - "sample": 按样本数平均
    # - "square": 按长度的平方根缩放
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def main():
    """
       InternVL模型训练和特征提取的主函数

       主要流程:
       1. 应用模型和训练优化补丁
       2. 初始化分布式训练环境
       3. 解析命令行参数和配置文件
       4. 设置日志和随机种子
       5. 加载tokenizer和模型
       6. 配置模型参数和训练设置
       7. 构建训练数据集
       8. 设置参数冻结策略
       9. 创建训练器和数据加载器
       10. 提取序列特征表示并保存
    """
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()  # 用融合的RMSNorm替换原始实现，提升训练速度
    replace_train_sampler()  # 获取启动器类型，默认为slurm
    replace_train_dataloader()  # 替换数据加载器以优化数据加载流程

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")

    # 创建HuggingFace参数解析器，支持三种参数类型
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    # 支持从JSON配置文件或命令行参数解析
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        print("Load from json config! ")
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # 否则从命令行参数解析
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
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
    tokenizer.tokenizer_path = tokenizer_path  # 保存tokenizer路径
    tokenizer.model_max_length = data_args.max_seq_length  # 设置最大序列长度
    # 定义需要添加的特殊token列表，用于多模态任务
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
    # 添加特殊token到tokenizer
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # 如果使用打包数据集，应用相应的注意力机制补丁
    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    # 如果使用liger内核优化，应用相应的优化补丁
    if model_args.use_liger:
        from liger_kernel.transformers import (
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_qwen2,
        )

        from internvl.patch import apply_liger_kernel_to_internvit

        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info("Loading InternVLChatModel...")
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate

        # 根据模型类型配置attn_implementation
        if config.llm_config.model_type == "internlm2":
            # config.llm_config.attn_implementation = "sdpa"  # for InternLM
            # logger.info("Using sdpa for InternLM")
            config.llm_config._attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for LLaMA")
        else:
            # config.llm_config.attn_implementation = "sdpa"  # for LLaMA
            # logger.info("Using sdpa for Model")
            config.llm_config._attn_implementation = "flash_attention_2"
            logger.info("Using flash_attention_2 for LLaMA")

        # 设置模型配置参数
        config.template = data_args.conv_style   # 对话模板风格
        config.select_layer = model_args.vision_select_layer    # 视觉层选择
        config.dynamic_image_size = data_args.dynamic_image_size    # 是否使用动态图像尺寸
        config.use_thumbnail = data_args.use_thumbnail   # 是否使用缩略图
        config.ps_version = model_args.ps_version   # 位置编码版本
        config.min_dynamic_patch = data_args.min_dynamic_patch  # 最小动态patch数
        config.max_dynamic_patch = data_args.max_dynamic_patch  # 最大动态patch数
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
            llm_config.attn_implementation = "flash_attention_2"  # for InternLM
            logger.info("Using flash_attention_2 for InternLM")
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
            logger.info("Using flash_attention_2 for LLaMA")

        # # 根据模型类型配置attn_implementation
        # if llm_config.model_type == "internlm2":
        #     model_type = InternLM2ForCausalLM
        #     llm_config.attn_implementation = "sdpa"  # for InternLM
        #     logger.info("Using sdpa for InternLM")
        # else:
        #     model_type = AutoModelForCausalLM
        #     llm_config._attn_implementation = "sdpa"  # for LLaMA
        #     logger.info("Using sdpa for Model")

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

    model.img_context_token_id = img_context_token_id   # 设置图像上下文token ID

    # 确保模型的下采样比例与数据参数一致
    assert model.config.downsample_ratio == data_args.down_sample_ratio

    # 如果提供了MLP投影器路径，加载预训练的MLP权重
    if model_args.mlp_path is not None:
        logger.info("Loading pretrained MLP projector...")
        state_dict = torch.load(model_args.mlp_path, map_location="cpu")
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info("Finished")

    # 处理图像位置编码和token数量计算
    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(
        f"model.config.vision_config.image_size: {model.config.vision_config.image_size}"
    )

    # 如果模型图像尺寸与强制图像尺寸不一致，调整位置编码
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

    # 设置强制图像尺寸
    model.config.force_image_size = data_args.force_image_size

    # 计算图像token数量：(图像尺寸/patch大小)^2 * 下采样比例^2
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio**2)
    )

    # 如果添加了新token，调整语言模型的词嵌入层
    if num_new_tokens > 0:
        # 调整词嵌入大小
        model.language_model.resize_token_embeddings(len(tokenizer))

        # 用已有token的均值初始化新添加的token嵌入
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # 更新词汇表大小配置
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    # 构建训练数据集
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
        # 冻结模块的所有参数
        for param in module.parameters():
            param.requires_grad = False

    # 应用参数冻结策略
    if model_args.freeze_backbone and not isinstance(model.vision_model, PeftModel):
        model.vision_model = model.vision_model.eval()   # 设置视觉模型为评估模式
        _freeze_params(model.vision_model)  # 冻结视觉模型参数

    if model_args.freeze_llm and not isinstance(model.language_model, PeftModel):
        model.language_model = model.language_model.eval()  # 设置语言模型为评估模式
        _freeze_params(model.language_model)    # 冻结语言模型参数

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True    # 解冻语言模型头部

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)  # 冻结MLP投影器

    # 部分解冻ViT层
    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]    # 获取需要解冻的层
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True  # 解冻参数

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # 选择数据整理器 collator
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

    # 创建自定义训练器
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_dataloader = trainer.get_train_dataloader()

    filename = os.path.splitext(os.path.basename(data_args.meta_path))[0]
    save_dir = "./embeddings/" + filename
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Computing sequence representation...")

    model = trainer._wrap_model(model).to(trainer.args.device).eval()

    sep_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # 只在主进程显示进度条
    if dist.get_rank() == 0:
        dataloader = tqdm(train_dataloader, total=len(train_dataloader))
    else:
        dataloader = train_dataloader

    embeds_list = []
    cnt = 0

    # for inputs in tqdm(train_dataloader, total=len(train_dataloader)):
    for inputs in dataloader:
        inputs = trainer._prepare_inputs(inputs)

        new_inputs = process_inputs(inputs, model, tokenizer)

        embeds = model.extract_sequence_feature(**new_inputs).detach().cpu()

        embeds_list.append(embeds)
        cnt += 1
        if cnt > 10:
            break

    # torch.distributed.barrier()
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])

    world_size = torch.distributed.get_world_size()
    merged_embeds = [None] * world_size
    torch.distributed.all_gather_object(merged_embeds, embeds_list)

    merged_embeds = [x for sublist in merged_embeds for x in sublist]
    merged_embeds = torch.cat(merged_embeds, dim=0)

    if dist.get_rank() == 0:
        logger.info("Finished computing sequence representation.")
        torch.save(merged_embeds, os.path.join(save_dir, "embeddings.pt"))

    # if torch.distributed.get_rank() == 0:
    #     torch.save(merged_embeds, os.path.join(save_dir, "embeddings.pt"))


if __name__ == "__main__":
    main()
