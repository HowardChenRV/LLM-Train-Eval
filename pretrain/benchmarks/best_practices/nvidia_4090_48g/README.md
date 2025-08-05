# NVIDIA 4090 / 4090D 训练最佳实践

---

- 初版 | 更新日期： 20250801

---

测试推荐镜像为 `harbor.infini-ai.com/magatron_infinigence_images/megatron-infinigence:25.04_r14`

## 测试模型

### Llama2系列

4090-48G 此处性能测试采用 `Llama-2-7B` 模型

```txt
Number of parameters in transformer layers in billions:  6.48
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 6.74
Number of parameters in most loaded shard in billions: 1.7502
Number of parameters in other shards in billions: 1.6191
Theoretical memory footprints: weight and optimizer=20029.52 MB
```

## 推荐使用

参考任务： https://aurora-wandb.infini-ai.com/infini-perf/4090-48GB-megatron_r0.10.0-llama2_7b

### 单机测试

单机测试推荐采用 TP=1 PP=4 GLOBAL_BATCH_SIZE=128 的配置

4090-48G 单机训练性能结果如下：

```log
[before the start of training step] datetime: 2025-08-01 09:31:11 
Number of parameters in transformer layers in billions:  6.48
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 6.74
Number of parameters in most loaded shard in billions: 1.7502
Number of parameters in other shards in billions: 1.6191
Theoretical memory footprints: weight and optimizer=20029.52 MB
[Rank 4] (after 1 iterations) memory (MB) | allocated: 18598.7529296875 | max allocated: 20931.01220703125 | reserved: 21534.0 | max reserved: 21534.0[Rank 0] (after 1 iterations) memory (MB) | allocated: 20034.7529296875 | max allocated: 30227.64599609375 | reserved: 31086.0 | max reserved: 31086.0

[Rank 2] (after 1 iterations) memory (MB) | allocated: 18598.7529296875 | max allocated: 25079.2666015625 | reserved: 25890.0 | max reserved: 25890.0
[Rank 6] (after 1 iterations) memory (MB) | allocated: 20115.05078125 | max allocated: 20115.08251953125 | reserved: 20258.0 | max reserved: 20258.0
 [2025-08-01 09:31:44] iteration        1/    5000 | consumed samples:          128 | elapsed time per iteration (ms): 33186.0 | throughput per GPU (TFLOP/s/GPU): 91.0 | learning rate: 6.976744E-08 | global batch size:   128 | lm loss: 1.045013E+01 | loss scale: 1.0 | grad norm: 35.543 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:32:10] iteration        2/    5000 | consumed samples:          256 | elapsed time per iteration (ms): 25879.2 | throughput per GPU (TFLOP/s/GPU): 116.7 | learning rate: 1.395349E-07 | global batch size:   128 | lm loss: 1.044560E+01 | loss scale: 1.0 | grad norm: 35.570 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:32:36] iteration        3/    5000 | consumed samples:          384 | elapsed time per iteration (ms): 25801.3 | throughput per GPU (TFLOP/s/GPU): 117.1 | learning rate: 2.093023E-07 | global batch size:   128 | lm loss: 1.044709E+01 | loss scale: 1.0 | grad norm: 34.729 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:33:02] iteration        4/    5000 | consumed samples:          512 | elapsed time per iteration (ms): 25840.7 | throughput per GPU (TFLOP/s/GPU): 116.9 | learning rate: 2.790698E-07 | global batch size:   128 | lm loss: 1.044305E+01 | loss scale: 1.0 | grad norm: 34.803 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:33:28] iteration        5/    5000 | consumed samples:          640 | elapsed time per iteration (ms): 25837.1 | throughput per GPU (TFLOP/s/GPU): 116.9 | learning rate: 3.488372E-07 | global batch size:   128 | lm loss: 1.042942E+01 | loss scale: 1.0 | grad norm: 33.243 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:33:54] iteration        6/    5000 | consumed samples:          768 | elapsed time per iteration (ms): 25829.7 | throughput per GPU (TFLOP/s/GPU): 116.9 | learning rate: 4.186047E-07 | global batch size:   128 | lm loss: 1.034294E+01 | loss scale: 1.0 | grad norm: 37.255 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:34:19] iteration        7/    5000 | consumed samples:          896 | elapsed time per iteration (ms): 25828.6 | throughput per GPU (TFLOP/s/GPU): 116.9 | learning rate: 4.883721E-07 | global batch size:   128 | lm loss: 1.017383E+01 | loss scale: 1.0 | grad norm: 34.413 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:34:45] iteration        8/    5000 | consumed samples:         1024 | elapsed time per iteration (ms): 25834.0 | throughput per GPU (TFLOP/s/GPU): 116.9 | learning rate: 5.581395E-07 | global batch size:   128 | lm loss: 1.008204E+01 | loss scale: 1.0 | grad norm: 33.950 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:35:11] iteration        9/    5000 | consumed samples:         1152 | elapsed time per iteration (ms): 25824.2 | throughput per GPU (TFLOP/s/GPU): 117.0 | learning rate: 6.279070E-07 | global batch size:   128 | lm loss: 9.877135E+00 | loss scale: 1.0 | grad norm: 31.556 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 09:35:37] iteration       10/    5000 | consumed samples:         1280 | elapsed time per iteration (ms): 25836.8 | throughput per GPU (TFLOP/s/GPU): 116.9 | learning rate: 6.976744E-07 | global batch size:   128 | lm loss: 9.619979E+00 | loss scale: 1.0 | grad norm: 26.519 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

### 多机测试

4090卡本身不具备RDMA高性能网卡，节点间通信通过以太网卡组网。常见配置为 25Gb/s * 2 的IB网卡聚合为带宽 50Gb/s 的网卡，如果网卡配置不同，可能严重影响多机测试性能结果。

2节点采用单 25Gb/s 网卡性能测试结果如下：

```log
 [2025-08-01 10:26:37] iteration        1/    5000 | consumed samples:          256 | elapsed time per iteration (ms): 34294.9 | throughput per GPU (TFLOP/s/GPU): 88.1 | learning rate: 6.976744E-08 | global batch size:   256 | lm loss: 1.044845E+01 | loss scale: 1.0 | grad norm: 34.157 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:27:03] iteration        2/    5000 | consumed samples:          512 | elapsed time per iteration (ms): 26496.5 | throughput per GPU (TFLOP/s/GPU): 114.0 | learning rate: 1.395349E-07 | global batch size:   256 | lm loss: 1.044817E+01 | loss scale: 1.0 | grad norm: 35.813 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:27:30] iteration        3/    5000 | consumed samples:          768 | elapsed time per iteration (ms): 26493.2 | throughput per GPU (TFLOP/s/GPU): 114.0 | learning rate: 2.093023E-07 | global batch size:   256 | lm loss: 1.044805E+01 | loss scale: 1.0 | grad norm: 34.794 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:27:56] iteration        4/    5000 | consumed samples:         1024 | elapsed time per iteration (ms): 26561.6 | throughput per GPU (TFLOP/s/GPU): 113.7 | learning rate: 2.790698E-07 | global batch size:   256 | lm loss: 1.044099E+01 | loss scale: 1.0 | grad norm: 33.940 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:28:23] iteration        5/    5000 | consumed samples:         1280 | elapsed time per iteration (ms): 26610.4 | throughput per GPU (TFLOP/s/GPU): 113.5 | learning rate: 3.488372E-07 | global batch size:   256 | lm loss: 1.041553E+01 | loss scale: 1.0 | grad norm: 36.017 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:28:50] iteration        6/    5000 | consumed samples:         1536 | elapsed time per iteration (ms): 26742.1 | throughput per GPU (TFLOP/s/GPU): 112.9 | learning rate: 4.186047E-07 | global batch size:   256 | lm loss: 1.034509E+01 | loss scale: 1.0 | grad norm: 34.448 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:29:16] iteration        7/    5000 | consumed samples:         1792 | elapsed time per iteration (ms): 26718.7 | throughput per GPU (TFLOP/s/GPU): 113.0 | learning rate: 4.883721E-07 | global batch size:   256 | lm loss: 1.015612E+01 | loss scale: 1.0 | grad norm: 32.529 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:29:43] iteration        8/    5000 | consumed samples:         2048 | elapsed time per iteration (ms): 26654.2 | throughput per GPU (TFLOP/s/GPU): 113.3 | learning rate: 5.581395E-07 | global batch size:   256 | lm loss: 1.006977E+01 | loss scale: 1.0 | grad norm: 32.856 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:30:10] iteration        9/    5000 | consumed samples:         2304 | elapsed time per iteration (ms): 27066.6 | throughput per GPU (TFLOP/s/GPU): 111.6 | learning rate: 6.279070E-07 | global batch size:   256 | lm loss: 9.789261E+00 | loss scale: 1.0 | grad norm: 29.892 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 10:30:37] iteration       10/    5000 | consumed samples:         2560 | elapsed time per iteration (ms): 26821.5 | throughput per GPU (TFLOP/s/GPU): 112.6 | learning rate: 6.976744E-07 | global batch size:   256 | lm loss: 9.607471E+00 | loss scale: 1.0 | grad norm: 24.659 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

注意使用时根据节点数量调整 GLOBAL_BATCH_SIZE 值，将单轮训练耗时保持在 10 ～ 30 s 内为佳。
