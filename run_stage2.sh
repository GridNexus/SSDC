#!/bin/bash

# 1. Optional environment cleanup
# echo "🧹 Cleaning up user processes..."
# pkill -9 -u yzf python
# find /dev/shm -user yzf -delete 2>/dev/null

# 2. Core settings
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export VLLM_PORT=$((20000 + RANDOM % 10000))

# Force single-GPU execution
export CUDA_VISIBLE_DEVICES=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 3. Path configuration
embed_model_path=/*/checkpoint
model_dir=/*/Qwen3-VL-8B-sft
root_dir=/*/data

# 4. Hyperparameters
xi=0.1
lambda=0.4
source=PAB
target=PAB

# 5. Run the script (CUDA_VISIBLE_DEVICES is already set above)
python vllm_infer_SSDC.py \
    --xi ${xi} \
    --lambda ${lambda} \
    --embed_model_path ${embed_model_path} \
    --source ${source} \
    --target ${target} \
    --ssdc_checkpoint /data/*/best.pth \
    --ssdc_config /data/*/configs.yaml \
    --model_dir ${model_dir} \
    --base_model RDE \
    --root_dir ${root_dir} \
    --tag test \
    2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log

# conda create --name myenv python=3.10
# pip install vllm easydict ftfy prettytable nltk qwen_vl_utils
