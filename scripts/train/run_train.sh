#!/bin/bash
set -e  # 出错时退出

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34221
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


GPUS=$(echo -n $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
GPUS=$((GPUS + 1))  # 根据实际GPU数量计算

# Basic configuration
BASE_MODEL="InternVL2-2B"
INITIAL_MODEL_PATH="pretrained/${BASE_MODEL}"
CONTINUAL_BASE_DIR='checkpoints/v1'
AUTOENCODER_PATH="autoencoder_models"
ENABLE_EVALUATION=${ENABLE_EVALUATION:-false}

# Create output directory
mkdir -p "$CONTINUAL_BASE_DIR"

# Select task index: id = index+1
TASK_INDEX=1

# Task configurations: name:metapath:batch_size:epochs
TASK_CONFIGS=(
  "vizwiz_caption:./shell/dmole/vizwiz_caption.json:10:1"
  "skvg:./shell/dmole/skvg.json:10:5"
  "textcaps:./shell/dmole/textcaps.json:10:1"
  "iconqa:./shell/dmole/iconqa.json:10:1"
  "ocrvqa:./shell/dmole/ocrvqa.json:8:1"
  "flickr30k:./shell/dmole/flickr30k.json:10:1"
  "vizwiz:./shell/dmole/vizwiz.json:10:5"
  "kvqa:./shell/dmole/kvqa.json:10:1"
  "pmcvqa:./shell/dmole/pmcvqa.json:10:1"
)

# Evaluation datasets
EVALUATION_DATASETS=(
  "caption-vizwiz-val"
  "grouding-skvg-test"
  "caption-textcaps-val"
  "vqa-iconqa-test"
  "vqa-ocrvqa-val"
  "caption-flickr30k"
  "vqa-vizwiz-val"
  "vqa-kvqa-test"
  "vqa-pmcvqa-test-clean"
)

# Function to parse task configuration
parse_task_config() {
  local config=$1
  IFS=":" read -r TASK_NAME META_PATH BATCH_SIZE EPOCHS <<< "$config"
}

# Main training
PREVIOUS_MODEL=$INITIAL_MODEL_PATH

if [ $TASK_INDEX -lt ${#TASK_CONFIGS[@]} ]; then
  # Parse current task configuration
  parse_task_config "${TASK_CONFIGS[$TASK_INDEX]}"
  EVAL_DATASET="${EVALUATION_DATASETS[$TASK_INDEX]}"
else
  echo "Error: Task index $TASK_INDEX out of range. Max index: $((${#TASK_CONFIGS[@]} - 1))"
  exit 1
fi

# Set output directory and DMOLE arch path
OUTPUT_DIR="${CONTINUAL_BASE_DIR}/$((TASK_INDEX+1))_${BASE_MODEL}-${TASK_NAME}"  # 使用 TASK_INDEX
DMOLE_ARCH_PATH="lora_arch/dmole/$((TASK_INDEX+1))_${BASE_MODEL}_${TASK_NAME}_arch.json"  
mkdir -p "$OUTPUT_DIR"


echo "=========================================="
echo "任务配置信息:"
echo "任务索引: $TASK_INDEX"
echo "任务名称: $TASK_NAME"
echo "元数据路径: $META_PATH"
echo "批大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "输出目录: $OUTPUT_DIR"
echo "GPU数量: $GPUS"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "$META_PATH" ]; then
  echo "错误: 配置文件不存在: $META_PATH"
  exit 1
fi

if [ ! -d "$PREVIOUS_MODEL" ] && [ ! -f "${PREVIOUS_MODEL}/pytorch_model.bin" ]; then
  echo "警告: 模型路径可能不存在: $PREVIOUS_MODEL"
fi

# 生成日志文件名（包含时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"
PID_FILE="${OUTPUT_DIR}/training.pid"

echo "开始后台训练..."
echo "日志文件: $LOG_FILE"
echo "进程ID文件: $PID_FILE"

# 使用 nohup 后台运行 torchrun
nohup torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${PREVIOUS_MODEL}" \
  --conv_style "internlm2-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 8 \
  --use_backbone_lora 8 \
  --use_dmole True \
  --dmole_arch_path "${DMOLE_ARCH_PATH}" \
  --autoencoder_path "${AUTOENCODER_PATH}" \
  --task_id $((TASK_INDEX+1)) \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "wandb" \
  > "${LOG_FILE}" 2>&1 &

