#!/bin/bash
set -ex
_THIS_DIR=$(dirname "$0")

# Setting the environment variables
export OMP_NUM_THREADS=1

# Distributed training variables
export NNODES=${WORLD_SIZE:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12453}
export PYTHONPATH=$PYTHONPATH:${BASE_PATH}/Infini-Eval

SRC_PATH=${BASE_PATH}/Infini-Eval/sys_eval/bench/nccl_bad_nodes.py
LOG_PATH=${BASE_PATH}/log/base_eval
mkdir -p ${LOG_PATH}


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

if [ $NODE_RANK == 0 ]; then
    ln -s $CUDA_PATH/bin/cucc $CUDA_PATH/bin/nvcc
    pushd ${BASE_PATH}/Infini-Eval/sys_eval/csrc && make build
    popd
fi
pip install tabulate

torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    2>&1 | tee ${LOG_PATH}/$(hostname).log