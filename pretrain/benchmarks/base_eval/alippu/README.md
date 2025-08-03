# 集群基础性能评测

## 设备支持

推荐镜像：ppu-training:1.4.3-pytorch2.5.1-ppu-py310-cu126-ubuntu22.04

## 运行方式

```
bash /mnt/data/chenyonghua/infini-tbench/benchmarks/base_eval/alippu/run_alippu_dist.sh
```

## 测试结果

PPU卡间采用立方体互联架构，没有switch, 因此测试结果中p2p性能可能浮动较大。