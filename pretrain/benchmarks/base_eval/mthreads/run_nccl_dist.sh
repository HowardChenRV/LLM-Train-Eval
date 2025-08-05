#!/bin/bash
set -exo pipefail

_THIS_DIR=$(dirname "$0")

source $_THIS_DIR/config_common.sh

SRC_PATH=$_THIS_DIR/sys_common_bench.py
LOG_PATH=${BASE_PATH}/log/base_eval
mkdir -p ${LOG_PATH}


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

AURORA_DATA_CLIENT_ARGS=(
    --aurora-test-type nccl/bandwidth
    --aurora-tester chenyonghua
    --aurora-save-dir ${LOG_PATH}/nccl_bandwidth
    --aurora-hardware-name ${HRADWARE_NAME:-A100-SXM4-80GB}
    --aurora-platform-provider ${PLATFORM_PROVIDER:-cloud-infini-ai} 
)

# if [ $NODE_RANK == 0 ]; then
#     pushd ${BASE_PATH}/LLM-Train-Eval/pretrain/aurora/sys_eval/csrc && make build
#     popd
# fi

torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${AURORA_DATA_CLIENT_ARGS[@]} \
    2>&1 | tee ${LOG_PATH}/$(hostname).log
