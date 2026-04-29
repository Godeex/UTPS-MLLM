# UTPS-MLLM



## Installation

```bash
# Clone the repository
conda create -n mllm python=3.10 -y
conda activate mllm

# Install flash-attention (pre-compiled wheel for CUDA 12 + PyTorch 2.6)
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


# Install dependencies
uv pip install opencv-python imageio decord pycocoevalcap wandb datasets
conda install openjdk=8 -y  # for pycocoevalcap
```


## Data Preparation

# 1. Download datasets and put in document "data"

[Download](https://drive.google.com/file/d/1NyUCf0jA0B-xfHXH7c5VZZDeEpL_TdzV/view?usp=sharing)

# 2. Download pretrained model and put in document "pretrained"


