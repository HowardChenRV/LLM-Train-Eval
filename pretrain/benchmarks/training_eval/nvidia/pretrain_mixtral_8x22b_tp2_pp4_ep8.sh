#!/bin/bash
set -exo pipefail
_THIS_DIR=$(dirname "$0")

source $_THIS_DIR/config_common.sh

# Runs Mixtral 8x7B model

TRAIN_ITERS=${TRAIN_ITERS:-100}

TP=${TP:-2}
PP=${PP:-4}
EP=${EP:-8}
DP=$((${GPU_NUM}/${TP}/${PP}/${EP}))

MODEL_SIZE="8x22"

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

BASE_PATH=${BASE_PATH:-/mnt/sctest/chenyonghua}
SRC_PATH=$_THIS_DIR/pretrain_gpt.py

LOG_NAME=mixtral-${MODEL_SIZE}b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}_EP${EP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME} && chmod 777 -R ${BASE_PATH}/log/

TOKENIZER_MODEL=/workspace/Mixtral-8x7B-v0.1/tokenizer.model
DATA_PATH=${DATA_PATH:-/workspace/datasets/wudao_mistralbpe_content_document}
DATA_CACHE_PATH=${BASE_PATH}/data_cache/${LOG_NAME}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 65536
    --num-layers 56
    --hidden-size 6144
    --ffn-hidden-size 16384
    --num-attention-heads 48
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path ${DATA_PATH}
    --data-cache-path ${DATA_CACHE_PATH}
    --split 1
    --num-workers 0
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --exit-interval ${TRAIN_ITERS}
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --use-flash-attn
    --transformer-impl transformer_engine
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --num-layers-per-virtual-pipeline-stage 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 500
    --eval-interval 500000
    --eval-iters 0
    --no-load-optim
    --no-load-rng
    --wandb-project ${HRADWARE_NAME:-A100-SXM4-80GB}-megatron_r0.10.0-mixtral_${MODEL_SIZE}b
    --wandb-exp-name pretrain-WS${WORLD_SIZE}-tp${TP}-pp${PP}-ep${EP}-gbs${GLOBAL_BATCH_SIZE}-mbs${MICRO_BATCH_SIZE}-seqlen${MAX_SEQ_LEN}
    --wandb-save-dir ${LOG_NAME}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
)

torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee ${LOG_PATH}
