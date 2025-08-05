#!/bin/bash

# Setting the environment variables
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_EXECUTION_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MUSA_PRINT_ENV=1
export MCCL_ALGOS=1
export MUSA_EXECUTE_COUNT=1
export MCCL_BUFFSIZE=20480000
export MCCL_NET_SHARED_BUFFERS=0

# Distributed training variables
export NNODES=${WORLD_SIZE:-1}
export GPUS_PER_NODE=8
export GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12453}

# wandb offline
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENTITY=${WANDB_ENTITY:-infini-perf}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://aurora-wandb.infini-ai.com}
export WANDB_API_KEY=${WANDB_API_KEY:-local-7d8f64d4173580c0ce08d87b32c01f36b92fbc56}

export DATA_CLIENT_MODE=${DATA_CLIENT_MODE:-offline}

# Training library path
export BASE_PATH=${BASE_PATH:-/data/chenyonghua}
export PATCH_HOME=/data/mt_experience_test/llama_bk/megatron_train/megatron-lm-musa-patch
export MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=/data/chenyonghua/LLM-Train-Eval/pretrain:${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH
