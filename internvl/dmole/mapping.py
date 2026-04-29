from contextlib import contextmanager

from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING

from .config import MoELoraConfig
from .model import MoELoraModel


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
