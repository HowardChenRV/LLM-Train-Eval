# 测试说明

## 常用环境变量

- export NCCL_IB_QPS_PER_CONNECTION=8

- export NCCL_MIN_NCHANNELS=32

- export NCCL_PXN_DISABLE=0

- export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

以上是常用的NCCL参数，根据集群特性选取。

- export TP=4

按向量切分模型并行数。

- export PP=4

按层切分模型并行数。

- export EP=8

按专家切分模型并行数。

- export MODEL_SIZE=70

模型尺寸，llama2 支持 7, 13, 70, tiny等尺寸.

- export TRAIN_ITERS=1000

性能测试时，建议轮数设置为100；稳定性测试时需要根据测试时长 / 单轮耗时计算所需轮数。

- export GLOBAL_BATCH_SIZE=1024

需要设置为DP的倍数，建议 DP * 2^n, `n`取决于单轮训练耗时，建议单轮耗时10~30s为佳。

- export BASE_PATH=/mnt/public/chenyonghua

日志、数据缓存、测试数据等文件存储位置，当采用多机分布式训练时，BASE_PATH建议设置为共享存储上。


## 测试准备

### 1. 预热环境镜像

训练任务需要特殊的组件，需要使用包含组件的镜像。公司内部可选：

- harbor.infini-ai.com/nxdx/training_eval_nvidia:25.04
这是公司内部编译的镜像，包含必要的组件，数据集和nccl-tests相关组件。内置训练组件为megatron-lm-0.10.0。

### 2. 下载tokenizer

评测镜像内置tokenizer:

- [/workspace/llama2-7b-hf/tokenizer.model](https://modelscope.cn/models/shakechen/Llama-2-7b-hf/resolve/master/tokenizer.model)

### 3. 下载数据集

评测镜像内置数据集：

- /workspace/datasets/wudao_mistralbpe_content_document
这是pai-megatron-patch使用的小数据集，文件小，适用于小型集群测试

```bash
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
```

## 训练测试任务

在云平台上启动测试任务代码如下：

```
#!/bin/bash
## 拷贝测试代码
scp xxx:/mnt/public/LLM-Train-Eval/pretrain.tar.gz /workspace/
tar -zxf /workspace/LLM-Train-Eval/pretrain.tar.gz

## 运行测试
bash /workspace/LLM-Train-Eval/pretrain/benchmarks/training_eval/nvidia/pretrain_llama2_7b_tp1_pp1.sh
```
