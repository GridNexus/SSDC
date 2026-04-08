#!/bin/bash

# 1. 清理环境（可选，谨慎使用）
# echo "🧹 Cleaning up user processes..."
# pkill -9 -u yzf python
# find /dev/shm -user yzf -delete 2>/dev/null

# 2. 核心设置
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export VLLM_PORT=$((20000 + RANDOM % 10000))

# 🔥 关键：单卡模式，强制只使用 GPU 6
export CUDA_VISIBLE_DEVICES=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 3. 路径配置
embed_model_path=/data/yzf/pro_ject/ARR1/CMP/checkpoint
model_dir=/data/yzf/pro_ject/model/Qwen3-VL-8B-sft
root_dir=/data/yzf/pro_ject/data

# 4. 超参数
xi=0.1
lambda=0.4
source=PAB
target=PAB

# 5. 运行脚本（移除了 CUDA_VISIBLE_DEVICES，因为已经在上面设置了）
python vllm_infer_SSDC.py \
    --xi ${xi} \
    --lambda ${lambda} \
    --embed_model_path ${embed_model_path} \
    --source ${source} \
    --target ${target} \
    --ssdc_checkpoint /data/yzf/pro_ject/ARR1/CMP/checkpoint/best.pth \
    --ssdc_config /data/yzf/pro_ject/ARR1/CMP/checkpoint/configs.yaml \
    --model_dir ${model_dir} \
    --base_model RDE \
    --root_dir ${root_dir} \
    --tag test \
    2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log

# conda create --name myenv python=3.10
# pip install vllm easydict ftfy prettytable nltk qwen_vl_utils