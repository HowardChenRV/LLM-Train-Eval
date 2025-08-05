#!/bin/bash
set -exo pipefail
_THIS_DIR=$(dirname "$0")

source $_THIS_DIR/config_common.sh

# Runs the "70B" parameter model

TRAIN_ITERS=${TRAIN_ITERS:-100}

TP=${TP:-1}
PP=${PP:-4}
DP=$((${GPU_NUM}/${TP}/${PP}))

LR=3e-5
MIN_LR=3e-6

# Network size variables
MODEL_SIZE=${MODEL_SIZE:-3}

if   [ ${MODEL_SIZE} == 3 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=16; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 7 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 13 ];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 70 ];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 130 ];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_QUERY_GROUP=8;  NUM_LAYERS=88; FFN_HIDDEN_SIZE=31232; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == "tiny" ]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((64*${DP}))}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}

SRC_PATH=${_THIS_DIR}/pretrain_gpt.py
LOG_NAME=llama2-${MODEL_SIZE}b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME} && chmod 777 -R ${BASE_PATH}/log/

TOKENIZER_MODEL=/workspace/Llama-2-7b-hf/tokenizer.model
DATA_PATH=${DATA_PATH:-/workspace/datasets/wudao_mistralbpe_content_document}
DATA_CACHE_PATH=${BASE_PATH}/data_cache/${LOG_NAME}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_HEAD}
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUP}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --norm-epsilon ${NORM_EPS}
    --seq-length ${MAX_SEQ_LEN}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --no-masked-softmax-fusion
    --no-rope-fusion
    --position-embedding-type rope
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE} 
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --exit-interval ${TRAIN_ITERS}
    --train-iters 5000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr ${LR}
    --lr-decay-style cosine 
    --min-lr ${MIN_LR}
    --lr-warmup-fraction .001
    --lr-decay-iters 430000 
    --use-flash-attn
    --transformer-impl transformer_engine
)

MIXED_PRECISION_ARGS=(
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP} 
    --pipeline-model-parallel-size ${PP} 
    --use-distributed-optimizer
    --sequence-parallel
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path ${DATA_PATH}
    --data-cache-path ${DATA_CACHE_PATH}
    --split 1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 10000 
    --eval-interval 10000
    --eval-iters 0
    --wandb-project ${HRADWARE_NAME:-4090-24GB}-megatron_r0.10.0-llama2_${MODEL_SIZE}b
    --wandb-exp-name pretrain-WS${WORLD_SIZE}-tp${TP}-pp${PP}-gbs${GLOBAL_BATCH_SIZE}-mbs${MICRO_BATCH_SIZE}-seqlen${MAX_SEQ_LEN}
    --wandb-save-dir ${LOG_NAME}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
)

AURORA_DATA_CLIENT_ARGS=(
    --aurora-test-type training/pretrain
    --aurora-tester chenyonghua
    --aurora-hardware-name ${HRADWARE_NAME:-4090-24GB}
    --aurora-platform-provider cloud-infini-ai
    --aurora-model-serial llama2
    --aurora-model-size ${MODEL_SIZE}
    --aurora-framework-version 0.10.0
)

torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${AURORA_DATA_CLIENT_ARGS[@]} \
    2>&1 | tee ${LOG_PATH}