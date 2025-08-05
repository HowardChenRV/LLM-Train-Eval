# 测试说明

## 测试准备

### 1. 预热环境镜像

阿里云PPU使用的镜像推荐为：ppu-training:1.4.3-pytorch2.5.1-ppu-py310-cu126-ubuntu22.04


### 2. 下载Megatron-LM源码

```bash
# 这里推荐使用megatron 0.8.0
git clone https://github.com/NVIDIA/Megatron-LM -b core_r0.8.0 --depth=1
```

### 3. 下载tokenizer

- [llama2-7b-hf/tokenizer.model](https://modelscope.cn/models/shakechen/Llama-2-7b-hf/resolve/master/tokenizer.model)

### 4. 下载数据集

如果是正常的训练任务需要构建数据集，测试可以采用构建好的数据集，可以选择：

- mistral-datasets/wudao_mistralbpe_content_document
这是pai-megatron-patch使用的小数据集，文件小，适用于小型集群测试

```bash
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
```

- RedPajama-Data-1T-Sample-datasets/RedPajama-Data-Llama
这是已经预处理过的数据集，文件大小大约2GB,，需要从OSS下载

```bash
pip install awscli
aws configure  # 帐号密码联系李文/周拓

aws  --endpoint-url https://infini-testing.oss-cn-beijing.aliyuncs.com s3 cp  s3://infini-testing/RedPajama-Data-1T-Sample-datasets/RedPajama-Data-Llama.bin  RedPajama-Data-1T-Sample-datasets/
aws  --endpoint-url https://infini-testing.oss-cn-beijing.aliyuncs.com s3 cp  s3://infini-testing/RedPajama-Data-1T-Sample-datasets/RedPajama-Data-Llama.idx  RedPajama-Data-1T-Sample-datasets/
```

### 5. 下载测试代码 

```bash
git clone https://gitlab.infini-ai.com/qa/LLM-Train-Eval/pretrain -b main --depth=1
```

## 训练测试

```bash
bash /mnt/data/chenyonghua/LLM-Train-Eval/pretrain/benchmarks/training_eval/alippu/pretrain_llama2_70b_tp4_pp4.sh
```
