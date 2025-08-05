#!/bin/bash
set -ex
_THIS_DIR=$(dirname "$0")

source $_THIS_DIR/config_common.sh

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

# if [ $NODE_RANK == 0 ]; then
#     ln -s $CUDA_PATH/bin/cucc $CUDA_PATH/bin/nvcc
#     pushd ${BASE_PATH}/Infini-Eval/sys_eval/csrc && make build
#     popd
# fi
# pip install tabulate

torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    2>&1 | tee ${LOG_PATH}/$(hostname).log