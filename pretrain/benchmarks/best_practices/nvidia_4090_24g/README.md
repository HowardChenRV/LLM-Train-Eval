# NVIDIA 4090 / 4090D 训练最佳实践

---

- 初版 | 更新日期： 20250801

---

测试推荐镜像为 `harbor.infini-ai.com/magatron_infinigence_images/megatron-infinigence:25.04_r14`

## 测试模型

### Llama2系列

4090 单卡只有24GB显存，因此此处性能测试采用 `Llama-2-3.5B` 模型(层数为7b模型的一半)

```txt
Number of parameters in transformer layers in billions:  3.24
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 3.50
Number of parameters in most loaded shard in billions: 0.9406
Number of parameters in other shards in billions: 0.8096
Theoretical memory footprints: weight and optimizer=10764.77 MB
```

## 推荐使用

参考任务： https://aurora-wandb.infini-ai.com/infini-perf/4090-24GB-megatron_r0.10.0-llama2_3b

### 单机测试

单机测试推荐采用 TP=1 PP=4 GLOBAL_BATCH_SIZE=128 的配置

4090 单机训练性能结果如下：

```log
[before the start of training step] datetime: 2025-08-01 07:19:18 
Number of parameters in transformer layers in billions:  3.24
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 3.50
Number of parameters in most loaded shard in billions: 0.9406
Number of parameters in other shards in billions: 0.8096
Theoretical memory footprints: weight and optimizer=10764.77 MB
[Rank 4] (after 1 iterations) memory (MB) | allocated: 9334.3779296875 | max allocated: 10606.50830078125 | reserved: 10998.0 | max reserved: 10998.0
[Rank 2] (after 1 iterations) memory (MB) | allocated: 9334.3779296875 | max allocated: 12680.6357421875 | reserved: 13236.0 | max reserved: 13236.0[Rank 6] (after 1 iterations) memory (MB) | allocated: 10850.67578125 | max allocated: 10850.70751953125 | reserved: 11000.0 | max reserved: 11000.0

[Rank 0] (after 1 iterations) memory (MB) | allocated: 10770.3779296875 | max allocated: 15754.88818359375 | reserved: 16334.0 | max reserved: 16334.0
 [2025-08-01 07:19:37] iteration        1/    5000 | consumed samples:          128 | elapsed time per iteration (ms): 19138.8 | throughput per GPU (TFLOP/s/GPU): 80.2 | learning rate: 6.976744E-08 | global batch size:   128 | lm loss: 1.043863E+01 | loss scale: 1.0 | grad norm: 26.203 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:19:50] iteration        2/    5000 | consumed samples:          256 | elapsed time per iteration (ms): 12936.1 | throughput per GPU (TFLOP/s/GPU): 118.7 | learning rate: 1.395349E-07 | global batch size:   128 | lm loss: 1.044022E+01 | loss scale: 1.0 | grad norm: 26.305 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:20:03] iteration        3/    5000 | consumed samples:          384 | elapsed time per iteration (ms): 12978.1 | throughput per GPU (TFLOP/s/GPU): 118.3 | learning rate: 2.093023E-07 | global batch size:   128 | lm loss: 1.044375E+01 | loss scale: 1.0 | grad norm: 25.930 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:20:16] iteration        4/    5000 | consumed samples:          512 | elapsed time per iteration (ms): 13014.1 | throughput per GPU (TFLOP/s/GPU): 118.0 | learning rate: 2.790698E-07 | global batch size:   128 | lm loss: 1.044319E+01 | loss scale: 1.0 | grad norm: 26.038 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:20:29] iteration        5/    5000 | consumed samples:          640 | elapsed time per iteration (ms): 13076.0 | throughput per GPU (TFLOP/s/GPU): 117.5 | learning rate: 3.488372E-07 | global batch size:   128 | lm loss: 1.043448E+01 | loss scale: 1.0 | grad norm: 24.564 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:20:42] iteration        6/    5000 | consumed samples:          768 | elapsed time per iteration (ms): 13074.7 | throughput per GPU (TFLOP/s/GPU): 117.5 | learning rate: 4.186047E-07 | global batch size:   128 | lm loss: 1.039646E+01 | loss scale: 1.0 | grad norm: 28.022 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:20:55] iteration        7/    5000 | consumed samples:          896 | elapsed time per iteration (ms): 13081.8 | throughput per GPU (TFLOP/s/GPU): 117.4 | learning rate: 4.883721E-07 | global batch size:   128 | lm loss: 1.031364E+01 | loss scale: 1.0 | grad norm: 25.976 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:21:09] iteration        8/    5000 | consumed samples:         1024 | elapsed time per iteration (ms): 13068.7 | throughput per GPU (TFLOP/s/GPU): 117.5 | learning rate: 5.581395E-07 | global batch size:   128 | lm loss: 1.027362E+01 | loss scale: 1.0 | grad norm: 25.762 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:21:22] iteration        9/    5000 | consumed samples:         1152 | elapsed time per iteration (ms): 13071.2 | throughput per GPU (TFLOP/s/GPU): 117.5 | learning rate: 6.279070E-07 | global batch size:   128 | lm loss: 1.011453E+01 | loss scale: 1.0 | grad norm: 24.505 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:21:35] iteration       10/    5000 | consumed samples:         1280 | elapsed time per iteration (ms): 13065.4 | throughput per GPU (TFLOP/s/GPU): 117.6 | learning rate: 6.976744E-07 | global batch size:   128 | lm loss: 9.947481E+00 | loss scale: 1.0 | grad norm: 23.632 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

### 多机测试

4090卡本身不具备RDMA高性能网卡，节点间通信通过以太网卡组网。常见配置为 25Gb/s * 2 的IB网卡聚合为带宽 50Gb/s 的网卡，如果网卡配置不同，将严重影响多机测试性能结果。

2节点采用单 25Gb/s 网卡性能测试结果如下：

```log
 [2025-08-01 07:35:40] iteration        1/    5000 | consumed samples:          256 | elapsed time per iteration (ms): 20170.9 | throughput per GPU (TFLOP/s/GPU): 76.1 | learning rate: 6.976744E-08 | global batch size:   256 | lm loss: 1.044431E+01 | loss scale: 1.0 | grad norm: 25.369 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:35:53] iteration        2/    5000 | consumed samples:          512 | elapsed time per iteration (ms): 13419.9 | throughput per GPU (TFLOP/s/GPU): 114.4 | learning rate: 1.395349E-07 | global batch size:   256 | lm loss: 1.043809E+01 | loss scale: 1.0 | grad norm: 26.766 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:36:06] iteration        3/    5000 | consumed samples:          768 | elapsed time per iteration (ms): 13453.4 | throughput per GPU (TFLOP/s/GPU): 114.2 | learning rate: 2.093023E-07 | global batch size:   256 | lm loss: 1.044171E+01 | loss scale: 1.0 | grad norm: 25.963 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:36:20] iteration        4/    5000 | consumed samples:         1024 | elapsed time per iteration (ms): 13565.0 | throughput per GPU (TFLOP/s/GPU): 113.2 | learning rate: 2.790698E-07 | global batch size:   256 | lm loss: 1.044017E+01 | loss scale: 1.0 | grad norm: 25.419 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:36:34] iteration        5/    5000 | consumed samples:         1280 | elapsed time per iteration (ms): 13546.1 | throughput per GPU (TFLOP/s/GPU): 113.4 | learning rate: 3.488372E-07 | global batch size:   256 | lm loss: 1.043097E+01 | loss scale: 1.0 | grad norm: 26.623 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:36:47] iteration        6/    5000 | consumed samples:         1536 | elapsed time per iteration (ms): 13577.0 | throughput per GPU (TFLOP/s/GPU): 113.1 | learning rate: 4.186047E-07 | global batch size:   256 | lm loss: 1.039971E+01 | loss scale: 1.0 | grad norm: 25.934 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:37:01] iteration        7/    5000 | consumed samples:         1792 | elapsed time per iteration (ms): 13604.3 | throughput per GPU (TFLOP/s/GPU): 112.9 | learning rate: 4.883721E-07 | global batch size:   256 | lm loss: 1.030662E+01 | loss scale: 1.0 | grad norm: 24.639 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:37:14] iteration        8/    5000 | consumed samples:         2048 | elapsed time per iteration (ms): 13613.5 | throughput per GPU (TFLOP/s/GPU): 112.8 | learning rate: 5.581395E-07 | global batch size:   256 | lm loss: 1.026286E+01 | loss scale: 1.0 | grad norm: 25.277 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:37:28] iteration        9/    5000 | consumed samples:         2304 | elapsed time per iteration (ms): 13622.2 | throughput per GPU (TFLOP/s/GPU): 112.7 | learning rate: 6.279070E-07 | global batch size:   256 | lm loss: 1.005576E+01 | loss scale: 1.0 | grad norm: 23.829 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-08-01 07:37:42] iteration       10/    5000 | consumed samples:         2560 | elapsed time per iteration (ms): 13623.2 | throughput per GPU (TFLOP/s/GPU): 112.7 | learning rate: 6.976744E-07 | global batch size:   256 | lm loss: 9.945539E+00 | loss scale: 1.0 | grad norm: 22.506 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

注意使用时根据节点数量调整 GLOBAL_BATCH_SIZE 值，将单轮训练耗时保持在 10 ～ 30 s 内为佳。
