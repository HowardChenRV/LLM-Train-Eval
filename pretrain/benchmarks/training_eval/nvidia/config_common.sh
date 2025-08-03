#!/bin/bash

# Setting the environment variables
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Distributed training variables
export NNODES=${WORLD_SIZE:-1}
export GPUS_PER_NODE=8
export GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12453}


# Training library path
export BASE_PATH=${BASE_PATH:-/mnt/public/chenyonghua}
export PYTHONPATH=${BASE_PATH}/infini-tbench/Megatron-LM:$PYTHONPATH
