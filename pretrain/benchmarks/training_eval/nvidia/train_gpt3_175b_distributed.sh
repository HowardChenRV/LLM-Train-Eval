#!/bin/bash
set -o pipefail
set -ex
_THIS_DIR=$(dirname "$0")

source $_THIS_DIR/config_common.sh


# Runs the "175B" parameter model
TRAIN_ITERS=${TRAIN_ITERS:-10}

TP=${TP:-4}
PP=${PP:-8}
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables
MODEL_SIZE="175"

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

BASE_PATH=${BASE_PATH:-/mnt/sctest/chenyonghua}
SRC_PATH=$_THIS_DIR/pretrain_gpt.py

LOG_NAME=gpt3-${MODEL_SIZE}b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME} && chmod 777 -R ${BASE_PATH}/log/

VOCAB_FILE=${BASE_PATH}/gpt2/vocab.json
MERGE_FILE=${BASE_PATH}/gpt2/merges.txt
DATA_PATH=$_THIS_DIR/../datasets/wudao_mistralbpe_content_document
DATA_CACHE_PATH=${BASE_PATH}/data_cache/${LOG_NAME}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 96
    --hidden-size 12288 
    --num-attention-heads 96 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE} 
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --use-flash-attn
    --transformer-impl transformer_engine
)
#     --rampup-batch-size 16 16 5859375 

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP}
	--pipeline-model-parallel-size ${PP}
)

DATA_ARGS=(
    --data-path ${DATA_PATH} 
    --vocab-file ${VOCAB_FILE} 
    --merge-file ${MERGE_FILE} 
    --tokenizer-type GPT2BPETokenizer
    --data-cache-path ${DATA_CACHE_PATH}
    --split 1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 0
    --wandb-project nvidia_h100-megatron_r0.8.0-gpt3_${MODEL_SIZE}b
    --wandb-exp-name pretrain-WS${WORLD_SIZE}-tp${TP}-pp${PP}-gbs${GLOBAL_BATCH_SIZE}-mbs${MICRO_BATCH_SIZE}-seqlen${MAX_SEQ_LEN}
    --wandb-save-dir ${LOG_NAME}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
)

torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee ${LOG_PATH}
