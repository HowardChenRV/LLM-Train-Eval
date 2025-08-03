# 测试说明

## 测试准备

### 1. 预热环境镜像

训练任务需要特殊的组件，需要使用包含组件的镜像。公司内部可选：

- cr.infini-ai.com/infini-ai/llm-demo:pytorch-24.03-py3
这是公司内部编译的镜像，包含必要的组件，数据集和nccl-tests相关组件。内置训练组件为megatron-lm-0.4.0。

- dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.01
这是阿里云Pai-Megatron-Patch的官方镜像，组件更新维护

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
git clone https://gitlab.infini-ai.com/qa/infini-tbench -b main --depth=1
```

## 训练测试

