import os
import sys
import torch
import torch.distributed as dist
import time
import argparse
import pytest
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
try:
    from torch import OutOfMemoryError
except:
    from torch.cuda import OutOfMemoryError
from sys_eval.utils.timer import bench_timer
from sys_eval.bench.bench_core import BenchmarkBase
from sys_eval.utils.runtime_driver import CudaDriver


# 思路：节点两两组合，然后错位两两组合，最后汇总所有节点的两次结果
#      执行 256MB 的 ALLREDUCE 操作

class NCCLBadNodesBench(BenchmarkBase):
    def __init__(self, device="cuda"):
        self.name = "NCCLBadNodes"
        self.device = device
        self.run_args = []
        self.rank = dist.get_rank()
        if device == "cuda":
            active_driver = CudaDriver()
            self.device_count = active_driver.get_device_count()
            self.di = active_driver.get_device_interface()
        else:
            raise NotImplementedError("device not configed: {}".format(device))
        self.node_count = dist.get_world_size() // self.device_count
        self.node_rank = self.rank // self.device_count
        self.di.set_device(self.rank % self.device_count)
        self.results = torch.empty(2, device=self.device)
        self.gather_result = None

    def finish(self):
        self.results = None
        self.gather_result = None
        dist.barrier()

    def run(self):
        world_size = dist.get_world_size()
        tensor_length = 256 * 1024 ** 2   # 1GB
        tensor_size_GB = tensor_length * 4 / 1024 ** 3  # GB
        test_tensor = torch.randn(tensor_length, device=self.device)

        # test all
        dist.all_reduce(test_tensor[0 : tensor_length - 1], op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        # 两两组合 例如: 0-1, 2-3, 4-5, ...
        # 如果为节点总数为单数，最后一个节点不参与组合
        group_ranks = []
        for i in range(0, self.node_count - 1, 2):
            group_ranks.append([i * self.device_count + j for j in range(self.device_count * 2)])

        group = None
        for group_rank in group_ranks:
            if self.rank in group_rank:
                group = dist.new_group(group_rank, use_local_synchronization=True)
                break

        if group is not None:
            def fn():
                dist.all_reduce(test_tensor[0: tensor_length - 1], 
                                op=dist.ReduceOp.SUM, group=group)
                torch.cuda.synchronize()
                dist.barrier(group=group)

            with torch.inference_mode():
                op_times = bench_timer(fn, device=self.device, warmups=3, 
                                       repeats=10, return_mode=["mean"],
                                       timeout=240)
            
            self.results[0] = tensor_size_GB / op_times[0] * 1000  # GB/s

        dist.barrier()
        test_tensor = torch.randn(tensor_length, device=self.device)
        # 进行第二次测试
        # 两两错位组合 例如: 1-2, 3-4, ..., n-0
        group_ranks = []
        for i in range(1, self.node_count - 1, 2):
            group_ranks.append([i * self.device_count + j for j in range(self.device_count * 2)])

        if self.node_count % 2 == 0:
            group_ranks.append([i * self.device_count + j for j in range(self.device_count) 
                                for i in [0, self.node_count - 1]])

        group = None
        for group_rank in group_ranks:
            if self.rank in group_rank:
                group = dist.new_group(group_rank, use_local_synchronization=True)

        if group is not None:
            def fn():
                dist.all_reduce(test_tensor[0: tensor_length - 1], 
                                op=dist.ReduceOp.SUM, group=group)
                torch.cuda.synchronize()
                dist.barrier(group=group)

            with torch.inference_mode():
                op_times = bench_timer(fn, device=self.device, warmups=3, 
                                       repeats=10, return_mode=["mean"],
                                       timeout=240)
            
            self.results[1] = tensor_size_GB / op_times[0] * 1000  # GB/s

        dist.barrier()

    def print_header(self):
        if self.rank == 0:
            print(f"\n## {self.name} benchmark:", flush=True)
            print("\tnodes: {}".format(self.node_count), flush=True)
            print("Results: (GB/s)", flush=True)
    
    def print_result(self):
        from tabulate import tabulate

        # 集合每个节点local_rank0的结果
        world_size = dist.get_world_size()
        group_rank = [i for i in range(world_size) if i % self.device_count == 0]
        group = dist.new_group(group_rank)

        if self.rank == 0:
            self.gather_result = [torch.zeros_like(self.results) for _ in group_rank]
        else:
            self.gather_result = None
        dist.barrier()

        if self.rank % self.device_count == 0:
            dist.gather(self.results, self.gather_result, dst=0, group=group)
        dist.barrier()

        if self.rank == 0:
            table_name = ["ALLREDUCE-1", "ALLREDUCE-2"]
            table = [result.tolist() for result in self.gather_result]
            print(tabulate(table, headers=table_name, tablefmt="github", 
                           floatfmt=".2f", showindex="always"), flush=True)


if __name__ == "__main__":
    dist.init_process_group("nccl")

    bench = NCCLBadNodesBench() 
    bench.run()
    bench.print_header()
    bench.print_result()
    bench.finish()
    dist.destroy_process_group()