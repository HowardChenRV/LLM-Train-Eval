#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export OMP_NUM_THREADS=16
export TORCH_DIST_INIT_BARRIER=0
export NCCL_MAX_NRINGS=1
export ASCEND_LAUNCH_BLOCKING=1
export MAX_JOBS=32

# Distributed training variables
export NNODES=${WORLD_SIZE:-1}
# export GPUS_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export GPUS_PER_NODE=8
export GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12453}

# wandb offline
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENTITY=${WANDB_ENTITY:-infini-perf}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://aurora-wandb.1nfini.cc}
export WANDB_API_KEY=${WANDB_API_KEY:-local-7d8f64d4173580c0ce08d87b32c01f36b92fbc56}

export DATA_CLIENT_MODE=${DATA_CLIENT_MODE:-offline}

# Training library path
export BASE_PATH=${BASE_PATH:-/mnt/public/chenyonghua}
export PYTHONPATH=${BASE_PATH}/LLM-Train-Eval/pretrain:${BASE_PATH}/Megatron-LM:$PYTHONPATH
