# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# v1
# --------------------------------------------------------

import json
import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from peft import PeftModel, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import GenerationConfig, LlamaForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from internvl.conversation import get_conv_template
from internvl.dist_utils import rank0_print
from internvl.dmole import DMoLELinear, MoELoraConfig, hijack_peft_mappings     # 核心 ->  DMoLELinear

from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM

from internvl.model.autoencoder import AutoEncoder

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig       # 使用自定义配置类
    main_input_name = "pixel_values"        # 主要输入是图像
    base_model_prefix = "language_model"
    _no_split_modules = [                   # 不分割的模块（保持完整性）
        "InternVisionModel",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
    ]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: InternVLChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        # 计算图像token数量
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        # 存储LLM架构名称（用于LoRA配置）
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        # 初始化视觉和语言模型
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Phi3ForCausalLM":
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        self.autoencoders = nn.ModuleList()
        self.thresholds = []

        dmole_arch = llm_dmole_arch = vision_dmole_arch = None
        if config.use_dmole:
            with open(config.dmole_arch_path, "r") as f:
                dmole_arch = json.load(f)
            vision_dmole_arch = {
                ".".join(k.split(".")[1:]): v
                for k, v in dmole_arch.items()
                if "vision" in k
            }
            llm_dmole_arch = {
                ".".join(k.split(".")[1:]): v
                for k, v in dmole_arch.items()
                if "language_model" in k
            }
            rank0_print("dmole_arch:", dmole_arch, "path:", config.dmole_arch_path)
            rank0_print("vision_dmole_arch:", vision_dmole_arch)
            rank0_print("llm_dmole_arch:", llm_dmole_arch)
            if config.autoencoder_path is None:
                config.autoencoder_path = "autoencoder_models"
            self.load_autoencoder(config.autoencoder_path)
            rank0_print("suceesfully loaded autoencoder from ", config.autoencoder_path)

        if config.use_backbone_lora:
            self.wrap_backbone_lora(
                r=config.use_backbone_lora,
                lora_alpha=2 * config.use_backbone_lora,
                dmole_arch=vision_dmole_arch,
            )

        if config.use_llm_lora:
            self.wrap_llm_lora(
                r=config.use_llm_lora,
                lora_alpha=2 * config.use_llm_lora,
                dmole_arch=llm_dmole_arch,
            )

    def wrap_backbone_lora(
        self,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        adapter_name="default",
        dmole_arch=None,            # D-MoLE架构配置，包含哪些层应该添加LoRA
    ):
        """
        将视觉编码器（backbone）包装为LoRA适配器
        动态层间专家分配器（Dynamic Layer-Wise Expert Allocator）

        如果是D-MoLE模式，会根据dmole_arch只对特定层添加LoRA
        否则对所有层添加LoRA（普通LoRA模式）
        """
        # 创建MoELoraConfig配置
        # MoELoraConfig是自定义的配置类，支持D-MoLE的专家分配
        lora_config = MoELoraConfig(
            r=r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            dmole_arch=dmole_arch,
        )

        # 如果是D-MoLE模式，注册自定义模块
        # DMoLELinear是自定义的线性层，支持动态专家路由
        if dmole_arch is not None:
            # 将标准的nn.Linear替换为DMoLELinear
            # 在创建LoRA适配器时会使用支持专家路由的线性层
            lora_config._register_custom_module({nn.Linear: DMoLELinear})
            # 对应公式(2)：f^t_l(x) = W^0_l x + Σ_{k=1}^{t-1} I^k_l · g^k_l(x) · ΔW^k_l x + I^t_l · ΔW^t_l x


        """
        @contextmanager
        def hijack_peft_mappings():
            try:
                # 1. 劫持配置映射：将标准的LoraConfig替换为MoELoraConfig
                PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = MoELoraConfig
                # 2. 劫持模型映射：将标准的LoraModel替换为MoELoraModel
                PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = MoELoraModel
                yield
            finally:
                PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = MoELoraConfig
                PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = MoELoraModel
        """
        # 劫持PEFT库的映射关系，将标准的LoRA配置和模型替换为自定义版本 进入时执行try->退出时执行finally
        with hijack_peft_mappings():
            self.vision_model = get_peft_model(
                self.vision_model, lora_config, adapter_name=adapter_name
            )

        # 启用输入梯度要求（用于梯度检查点等）
        self.vision_model.enable_input_require_grads()

        # 打印可训练参数信息
        rank0_print(f"Wrapping backbone lora: ", end="")
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(
        self,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        adapter_name="default",
        dmole_arch=None,
    ):
        """
        将语言模型（LLM）包装为LoRA适配器
        对应论文中的动态层间专家分配器

        根据不同的LLM架构选择不同的目标模块
        """
        if self.llm_arch_name == "InternLM2ForCausalLM":
            target_modules = [
                "attention.wqkv",
                "attention.wo",
                "feed_forward.w1",
                "feed_forward.w2",
                "feed_forward.w3",
            ]
        elif self.llm_arch_name == "Phi3ForCausalLM":
            target_modules = [
                "mlp.down_proj",
                "mlp.gate_up_proj",
                "self_attn.o_proj",
                "self_attn.qkv_proj",
            ]
        elif self.llm_arch_name in ["Qwen2ForCausalLM", "LlamaForCausalLM"]:
            target_modules = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ]
        else:
            raise NotImplemented

        # 创建MoELoraConfig配置，根据不同llm_arch_name选取不同的target_modules
        lora_config = MoELoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            dmole_arch=dmole_arch,
        )
        if dmole_arch is not None:
            lora_config._register_custom_module({nn.Linear: DMoLELinear})

        with hijack_peft_mappings():
            self.language_model = get_peft_model(
                self.language_model, lora_config, adapter_name=adapter_name
            )
        self.language_model.enable_input_require_grads()
        rank0_print(f"Wrapping llm lora: ", end="")
        self.language_model.print_trainable_parameters()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 步骤1：参数初始化与预处理
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # 去除多余的维度
        image_flags = image_flags.squeeze(-1)
        # 获取文本的词嵌入表示
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # 步骤2：视觉特征提取
        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        # 步骤3：维度重塑与统计信息
        # B=batch_size, N=序列长度, C=隐藏维度
        B, N, C = input_embeds.shape
        # 展平：[B, N, C] → [B*N, C]
        input_embeds = input_embeds.reshape(B * N, C)

        # 分布式训练中的统计信息打印
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            rank0_print(
                f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = (
                    statistics.tolist()
                )
                self.num_samples += num_samples
                rank0_print(
                    f"total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}"
                )

        # 步骤4：图像特征插入
        # 展平token IDs
        input_ids = input_ids.reshape(B * N)
        # 找到图像token位置 -> 创建布尔掩码，标记哪些位置是<IMG_CONTEXT> token
        selected = input_ids == self.img_context_token_id
        try:
            # 关键：将图像特征插入到文本嵌入中 -> 将图像占位符的嵌入替换为实际视觉特征
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
            ignore_flag = False
        except Exception as e:
            # 异常处理：形状不匹配时的容错机制
            vit_embeds = vit_embeds.reshape(-1, C)
            rank0_print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True
        # 步骤5：恢复形状并调用语言模型
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,         # 融合后的多模态嵌入
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits  # 预测logits [B, N, vocab_size]

        """
        logits: 预测每个位置的下一个token
        labels: 每个位置的实际下一个token
        
        所以需要：
        shift_logits = logits[:, :-1, :]  # 去掉最后一个预测
        shift_labels = labels[:, 1:]      # 去掉第一个token
        """

        # 步骤6：损失计算（训练时）
        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(
                loss_weight, dtype=torch.float32, device=labels.device
            )
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        # 步骤7：返回结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def extract_sequence_feature(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        vit_embeds_max = vit_embeds.max(dim=1).values

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        if not isinstance(self.language_model, PeftModel):
            outputs = self.language_model.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            llm_embeds = outputs.last_hidden_state
        else:
            outputs = self.language_model.model.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            llm_embeds = outputs.last_hidden_state

        llm_embeds_max = llm_embeds.max(dim=1).values

        return torch.cat([vit_embeds_max, llm_embeds_max], dim=1)

    @torch.no_grad()
    def get_last_hidden_states(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        rank0_print(
            f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
        )

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            rank0_print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        last_hidden_states = self.language_model.get_last_hidden_states(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        return last_hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        lora_weights_path = os.path.join(
            pretrained_model_name_or_path, "lora_weights.bin"
        )

        if os.path.exists(lora_weights_path):
            rank0_print("LoRA weights found. Loading LoRA model...")

            # Load the config, either from kwargs or from pretrained path
            config = kwargs.pop("config", None) or InternVLChatConfig.from_pretrained(
                pretrained_model_name_or_path
            )

            # Load the base model
            rank0_print(f"Loading base model from {config.name_or_path}")
            base_model = super().from_pretrained(
                config.name_or_path, *model_args, **kwargs
            )
            rank0_print("Base model loaded successfully.")

            rank0_print("config:", config)

            # Initialize model with the base components
            model = cls(config, base_model.vision_model, base_model.language_model)
            # mlp1 should be loaded from the base model
            model.mlp1 = base_model.mlp1

            # Load LoRA weights
            try:
                model.load_lora_weights(lora_weights_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA weights: {e}")

            # Optionally set torch dtype
            torch_dtype = kwargs.pop("torch_dtype", None)
            if torch_dtype is not None:
                model = model.to(torch_dtype)

            return model
        else:
            rank0_print(
                "LoRA weights not found. Loading base model without LoRA weights."
            )
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

    def load_lora_weights(self, weights_path):
        lora_state_dict = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(lora_state_dict, strict=False)
        rank0_print(f"Successfully loaded LoRA weights from {weights_path}")

    def load_autoencoder(self, base_autoencoder_path):
        """
        加载历史任务的自编码器和阈值，用于D-MoLE的任务识别

        对应：
        - 4.1节：Autoencoder Router and Gating Function
        - 附录A：Implementation Details
        - 附录F：Analysis of Autoencoder Router

        每个任务的自编码器负责：
        1. 提取任务特定的特征分布
        2. 计算重构误差以衡量输入与任务的相似性
        3. 在get_task_ids()中用于专家路由决策

        使用轻量级自编码器作为"任务指纹提取器"：
        1. 每个任务训练一个独立的自编码器
        2. 自编码器学习该任务数据分布的特征表示
        3. 通过重构误差判断输入属于哪个任务
        4. 基于任务相似性激活相关专家进行知识迁移
        """
        # =========== 1. 任务列表定义 ===========
        TASKS = [
            "vizwiz_caption",
            "skvg",
            "textcaps",
            "iconqa",
            "ocrvqa",
            "flickr30k",
            "vizwiz",
            "kvqa",
            "pmcvqa",
        ]
        # =========== 2. 遍历并加载任务 ===========
        # 只加载当前任务之前的历史任务
        # Iterate through the tasks and load corresponding autoencoder and threshold
        for task in TASKS[: self.config.task_id]:
            """
            self.config.task_id: 当前任务ID
            示例：如果task_id=3，则加载前2个任务的自编码器
            
            对应论文算法1的Step 4：
            - 为每个任务训练一个独立的轻量级自编码器
            """
            autoencoder_file = os.path.join(
                base_autoencoder_path, task, "autoencoder.pt"
            )
            threshold_file = os.path.join(base_autoencoder_path, task, "threshold.txt")

            rank0_print(
                f"Loading autoencoder and threshold for task {task} from {autoencoder_file} and {threshold_file}"
            )

            try:
                # =========== 3. 加载状态字典以推断维度 ===========
                # 论文附录A：输入维度3072，隐藏维度128
                # Load the state dict to infer dimensions
                state_dict = torch.load(autoencoder_file, map_location=self.device)

                """
                自编码器结构：
                - 2层MLP：编码器+解码器
                - 输入/输出维度：3072（视觉特征1024 + 文本特征2048）
                - 隐藏层维度：128
                - 论文附录A：Each autoencoder is trained for 100 epochs with a learning rate of 1e-3
                """
                # Auto-set input_dim and hidden_dim based on state_dict
                # Example: assuming 'encoder_layer.weight' holds the relevant dimension
                for key, value in state_dict.items():
                    if (
                        "weight" in key
                    ):  # Find a key with 'weight', usually containing the layer shape
                        input_dim = value.size(1)  # Number of input features
                        hidden_dim = value.size(0)  # Number of output features (hidden layer size)
                        break  # Found the dimensions, break the loop

                rank0_print(
                    f"Inferred dimensions for autoencoder - input_dim: {input_dim}, hidden_dim: {hidden_dim}"
                )

                # =========== 4. 创建并加载自编码器 ===========
                """
                论文第5.1节：autoencoder用于捕获每个任务特定的数据分布
                
                重构误差 ℒ_rec^t(z) = ℒ_MSE(z, ẑ^t) 用于：
                1. 衡量输入与任务t的相似性
                2. 低于阈值τ_t → 输入可能属于任务t
                3. 论文附录F：t-SNE可视化显示不同任务的特征明显分离
                """
                # Create and load the autoencoder with inferred dimensions
                autoencoder = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
                autoencoder.load_state_dict(state_dict)

                self.autoencoders.append(autoencoder)

                # =========== 5. 加载阈值 ===========
                """
                论文第4.1节和附录G：阈值τ_t的确定
                - 基于每个任务训练集的重构误差分布
                - 设置略高于观察到的损失范围
                - 容纳轻微分布偏移
                - 功能：
                    1. 防止不相关专家被选择
                    2. 拒绝未知任务样本
                
                论文附录G Table 7：阈值敏感性分析显示方法对阈值不敏感
                """
                # Load threshold
                with open(threshold_file, "r") as f:
                    threshold = float(f.read().strip())
                self.thresholds.append(threshold)

                """
                阈值使用（见get_task_ids）：
                below_threshold_indices = np.where(mse_scores[i, :] < thresholds)[0]
                
                对应论文公式10：ℛ = {t | ℒ_rec^t(z) ≤ τ_t}
                """

            except FileNotFoundError as e:
                rank0_print(
                    f"File not found: {e.filename}, please check the file path."
                )
                exit(0)
            except Exception as e:
                rank0_print(
                    f"Error occurred while loading autoencoder or threshold for task {task}: {str(e)}"
                )
                exit(0)

        self.autoencoders = self.autoencoders.to(self.device)
        # 冻结自编码器参数
        self.autoencoders.requires_grad_(False)
        rank0_print("Autoencoders and thresholds have been loaded.")

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """
        从原始图像像素提取视觉特征，并转换为语言模型兼容的格式

        输入: pixel_values [batch_size, channels, height, width]
        输出: vit_embeds [batch_size, num_tokens, hidden_dim]

        处理流程:
        1. 精度转换 → 2. ViT编码 → 3. 移除CLS token → 4. 重塑为2D网格
        5. 像素重排 → 6. 重塑为序列 → 7. MLP投影
        """
        pixel_values = pixel_values.to(torch.bfloat16)
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def process_inputs(
        self,
        tokenizer,
        input_ids,
        pixel_values,
        num_patches_list,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    ):
        # last pixel value of each patch is the thumbnail, extract it with num_patches_list
        thumb_pixel_values = []
        idx_shift = 0
        for num_patches in num_patches_list:
            thumb_pixel_values.append(pixel_values[idx_shift + num_patches - 1])
            idx_shift += num_patches
        thumb_pixel_values = torch.stack(thumb_pixel_values)

        img_tokens = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN]
        img_token_ids = tokenizer.convert_tokens_to_ids(img_tokens)
        img_token_ids = torch.tensor(img_token_ids, device=input_ids.device)

        sep_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        text_input_ids = []
        for i in range(input_ids.size(0)):
            cur_input_ids = input_ids[i]
            sep_positions = (cur_input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            start_idx = sep_positions[0] + 1
            end_idx = sep_positions[1] + 1

            truncated_input_ids = cur_input_ids[start_idx:end_idx]

            mask = ~torch.isin(truncated_input_ids, img_token_ids)
            truncated_input_ids = truncated_input_ids[mask]
            text_input_ids.append(truncated_input_ids)
        pad_sequence = torch.nn.utils.rnn.pad_sequence
        text_input_ids = pad_sequence(text_input_ids, batch_first=True, padding_value=0)
        text_attention_mask = text_input_ids != 0

        return thumb_pixel_values, text_input_ids, text_attention_mask

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        """
        在 batch_chat 内部会自动处理D-MoLE逻辑：

        任务识别：通过自编码器确定输入任务
        专家激活：设置对应的专家掩码
        多专家推理：使用激活的专家生成描述
        """
        if history is not None or return_history:
            rank0_print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            rank0_print(
                "Warning: `image_counts` is deprecated. Please use `num_patches_list` instead."
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            rank0_print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        device = torch.device(
            self.language_model.device if torch.cuda.is_available() else "cpu"
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id

        if self.config.use_dmole:
            self.set_expert_masks([])

            thumb_pixel_values, text_input_ids, text_attention_mask = (
                self.process_inputs(
                    tokenizer, input_ids, pixel_values, num_patches_list
                )
            )

            seq_rep = self.extract_sequence_feature(
                pixel_values=thumb_pixel_values,
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            # 关键步骤，选择最优的专家
            best_task_ids = self.get_task_ids(seq_rep)
            self.set_expert_masks(best_task_ids)

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [
            response.split(template.sep.strip())[0].strip() for response in responses
        ]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            rank0_print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        device = torch.device(
            self.language_model.device if torch.cuda.is_available() else "cpu"
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        generation_config["eos_token_id"] = eos_token_id

        if self.config.use_dmole:
            self.set_expert_masks([])

            thumb_pixel_values, text_input_ids, text_attention_mask = (
                self.process_inputs(
                    tokenizer, input_ids, pixel_values, num_patches_list
                )
            )

            seq_rep = self.extract_sequence_feature(
                pixel_values=thumb_pixel_values,
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            best_task_ids = self.get_task_ids(seq_rep)
            self.set_expert_masks(best_task_ids)

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                rank0_print(query_to_print, response)
            return response

    @torch.no_grad()
    def get_task_ids(
        self,
        seq_reps,
        mode="evaluation",
        current_task_id=None,
        second_task_threshold=0.3,
    ):
        """
        D-MoLE核心方法：基于自编码器重构误差识别输入样本所属的任务

        对应算法：
        1. 特征提取：seq_reps = concat(v_pooled, w_pooled)
        2. 重构误差计算：ℒ_rec^t(z) = ℒ_MSE(z, ẑ^t)
        3. 任务选择：基于阈值和投票机制

        返回：需要激活的专家对应的任务ID列表
        """
        batch_size = seq_reps.size(0)

        # =========== 1. 初始化检查 ===========
        # 如果没有自编码器，则：评估模式：返回空列表（无历史任务可迁移） / 训练模式：返回当前任务ID（只训练当前任务）
        if len(self.autoencoders) == 0:
            return [] if mode == "evaluation" else [current_task_id]

        # =========== 2. 模式特定的预处理 ===========
        # 训练模式：已知当前任务ID
        if mode == "training":
            if current_task_id is None:
                raise ValueError("current_task_id is required for training mode")
            # 第一个任务不需要历史任务（没有历史可迁移）
            else:
                return [current_task_id]
            # 只考虑当前任务之前的自编码器（历史任务）
            # 设置前提：只能从已学习的任务迁移知识
            # available_autoencoders = min(current_task_id - 1, len(self.autoencoders))
            # thresholds = np.array(self.thresholds[:available_autoencoders])
        # 评估模式：未知任务ID，考虑所有自编码器
        else:
            return [self.config.task_id]
        # else:  # evaluation mode
        #     available_autoencoders = len(self.autoencoders)
        #     thresholds = np.array(self.thresholds)
        # # 如果没有可用的自编码器，返回基本结果
        # if available_autoencoders == 0:
        #     return [] if mode == "evaluation" else [current_task_id]
        #
        # # =========== 3. 特征归一化 ===========
        # # 归一化特征，使MSE计算更稳定
        # seq_reps = F.normalize(seq_reps, p=2, dim=1)
        #
        # # =========== 4. 计算重构误差 ===========
        # # 存储每个样本对每个任务的重构误差
        # mse_scores = np.zeros((batch_size, available_autoencoders))
        # mse_loss = nn.MSELoss(reduction="none")
        #
        # # 遍历所有可用的自编码器 - MSE scores
        # for idx in range(available_autoencoders):
        #     autoencoder = self.autoencoders[idx].to(seq_reps.device)
        #     # 重构特征
        #     reconstructions = autoencoder(seq_reps).bfloat16()
        #     # 计算逐元素MSE，然后求样本平均
        #     per_element_loss = mse_loss(reconstructions, seq_reps)
        #     per_sample_mse = per_element_loss.mean(dim=1)
        #     # 保存到数组（转换为CPU numpy便于后续处理）
        #     mse_scores[:, idx] = per_sample_mse.detach().cpu().float().numpy()
        #
        # # =========== 5. 基于阈值筛选候选任务 ===========
        # # 为每个样本选择最佳任务（基于阈值）
        # task_ids = np.full(batch_size, -1)  # -1表示未识别到任务
        #
        # for i in range(batch_size):
        #     # 找出重构误差低于阈值的任务
        #     below_threshold_indices = np.where(mse_scores[i, :] < thresholds)[0]
        #
        #     if len(below_threshold_indices) > 0:
        #         # 在低于阈值的任务中选择重构误差最小的
        #         min_mse_idx = below_threshold_indices[
        #             np.argmin(mse_scores[i, below_threshold_indices])
        #         ]
        #         # 转换为1-based索引（任务ID从1开始）
        #         task_ids[i] = min_mse_idx + 1
        #
        # # =========== 6. 投票机制整合批次结果 ===========
        # """
        # # 过滤无效的识别结果（-1表示未识别）
        # # 假设batch_size=4，识别结果：
        # task_ids = [1, -1, 2, 1]  # 样本0:任务1, 样本1:未识别, 样本2:任务2, 样本3:任务1
        # # 过滤后：
        # valid_task_ids = [1, 2, 1]  # 只保留有效识别
        # """
        # valid_task_ids = task_ids[task_ids != -1]
        #
        # if len(valid_task_ids) == 0:
        #     # 如果没有样本识别到任何任务
        #     return [] if mode == "evaluation" else [current_task_id]
        #
        # """
        # # 统计每个任务的投票数
        # # 假设 valid_task_ids = [1, 2, 1, 1, 3, 2, 1]
        # # vote_counts = [0, 4, 2, 1]
        # # 解释：
        # # 索引0: 任务0出现0次（任务ID从1开始，所以索引0总是0）
        # # 索引1: 任务1出现4次
        # # 索引2: 任务2出现2次
        # # 索引3: 任务3出现1次
        # """
        # vote_counts = np.bincount(valid_task_ids)
        # primary_task_id = np.argmax(vote_counts)    # 得票最多的任务
        # primary_votes = vote_counts[primary_task_id]    # 得票最多的任务的票数
        #
        # # =========== 7. 根据模式选择最终任务 ===========
        # if mode == "training":
        #     """
        #     训练模式策略（算法1训练阶段）：
        #     - 总是包含当前任务：确保学习新任务知识
        #     - 可选包含一个历史任务：促进知识迁移
        #
        #     返回格式：[current_task_id, (optional) historical_task_id]
        #     """
        #     # Training mode: always include current task + optionally one historical task 总是包含当前任务
        #     result_tasks = [current_task_id]
        #     # 如果识别出的主要任务不是当前任务，考虑作为历史任务
        #     if primary_task_id != current_task_id:
        #         historical_task_id = primary_task_id
        #         support_rate = primary_votes / len(valid_task_ids)
        #
        #         if support_rate >= second_task_threshold:
        #             result_tasks.append(historical_task_id)
        # else:
        #     """
        #     评估模式策略（论文算法1评估阶段）：
        #     - 选择1-2个最相关的任务
        #     - 基于投票支持率阈值
        #
        #     返回格式：[primary_task_id, (optional) secondary_task_id]
        #     """
        #     # Evaluation mode: original logic for 0-2 tasks
        #     result_tasks = [primary_task_id]
        #
        #     # 检查是否需要第二个任务
        #     # 条件1：有多个自编码器可用
        #     # 条件2：样本识别出了多个不同任务
        #     if available_autoencoders > 1 and len(np.unique(valid_task_ids)) > 1:
        #         remaining_votes = vote_counts.copy()
        #         remaining_votes[primary_task_id] = 0
        #
        #         if np.max(remaining_votes) > 0:
        #             # 找到得票第二多的任务
        #             secondary_task_id = np.argmax(remaining_votes)
        #             secondary_votes = remaining_votes[secondary_task_id]
        #             # 计算第二任务的支持率
        #             secondary_support_rate = secondary_votes / len(valid_task_ids)
        #             # 只有支持率足够高时才添加第二任务
        #             if secondary_support_rate >= second_task_threshold:
        #                 result_tasks.append(secondary_task_id)
        #
        # return result_tasks

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def set_expert_masks(self, task_id):
        if self.config.use_backbone_lora:
            self.vision_model.set_expert_masks(task_id)
        if self.config.use_llm_lora:
            self.language_model.set_expert_masks(task_id)

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
