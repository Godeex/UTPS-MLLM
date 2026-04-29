# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

"""
# 图像相关Token
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'  # 图像上下文标记
IMG_START_TOKEN = '<img>'            # 图像开始标记
IMG_END_TOKEN = '</img>'             # 图像结束标记

# 视觉定位Token（空间理解）
QUAD_START_TOKEN = '<quad>'          # 四边形区域开始
QUAD_END_TOKEN = '</quad>'           # 四边形区域结束
REF_START_TOKEN = '<ref>'            # 引用区域开始  
REF_END_TOKEN = '</ref>'             # 引用区域结束
BOX_START_TOKEN = '<box>'            # 边界框开始
BOX_END_TOKEN = '</box>'             # 边界框结束

# 图像归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # ImageNet均值
IMAGENET_STD = (0.229, 0.224, 0.225)   # ImageNet标准差

# CLIP模型归一化
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)  # CLIP均值
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)    # CLIP标准差

# SigLIP模型归一化
SIGLIP_MEAN = (0.5, 0.5, 0.5)    # SigLIP均值（简单归一化）
SIGLIP_STD = (0.5, 0.5, 0.5)     # SigLIP标准差

"""