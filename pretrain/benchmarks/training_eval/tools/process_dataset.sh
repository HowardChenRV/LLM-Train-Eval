#!/bin/bash

set -e

: "${BASE_PATH:=/mnt/sctest/chenyonghua}"
: "${MEGATRON_PATH:=${BASE_PATH}/Megatron-LM}"
: "${TOKENIZER_PATH:=${BASE_PATH}/Llama-2-70b-hf/tokenizer.model}"
: "${DATASET_PATH:=${BASE_PATH}/RedPajama-Data-1T-Sample}"
: "${SAVE_PATH:=${BASE_PATH}/RedPajama-Data-Llama-datasets}"


array=(
 arxiv_sample
 book_sample
 c4_sample
 cc_2019-30_sample
 cc_2020-05_sample
 cc_2021-04_sample
 cc_2022-05_sample
 cc_2023-06_sample
 github_sample
 stackexchange_sample
 wikipedia_sample
)

mkdir -p RedPajama-Data-splica

for file_name in ${array[@]}
do
    python ${MEGATRON_PATH}/tools/preprocess_data.py\
       --input ${DATASET_PATH}/${file_name}.jsonl \
       --output-prefix RedPajama-Data-splica/redpajama_${file_name}  \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH}\
       --append-eod \
       --workers 64
done

python ${MEGATRON_PATH}/tools/merge_datasets.py --input ./RedPajama-Data-splica/ --output-prefix ./RedPajama-Data-Llama_text_document

rm -rf RedPajama-Data-splica

mkdir -p ${SAVE_PATH} && mv ./RedPajama-Data-Llama* ${SAVE_PATH}/
