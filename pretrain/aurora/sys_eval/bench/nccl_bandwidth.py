import os
import sys
import torch
import torch.distributed as dist
import time
import argparse
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
try:
    from torch import OutOfMemoryError
except:
    from torch.cuda import OutOfMemoryError
from aurora.sys_eval.utils.timer import bench_timer
from aurora.sys_eval.bench.bench_core import BenchmarkBase
from aurora.sys_eval.utils.runtime_driver import CudaDriver, NpuDriver, MusaDriver


class OperationType(Enum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    ALL_TO_ALL = 2


@dataclass
class NCCLBandwidthArgs:
    operation: OperationType
    size_in_mb: int
    warmups: int = 3
    repeats: int = 10
    return_mode: list = field(default_factory=lambda: ["mean"])
    timeout: int = 600


class NCCLBandwidthBench(BenchmarkBase):
    def __init__(self, device="cuda"):
        self.name = "NCCLBandwidth"
        self.device = device
        self.run_args = []
        self.rank = dist.get_rank()
        if device == "cuda":
            active_driver = CudaDriver()
            self.device_count = active_driver.get_device_count()
            self.di = active_driver.get_device_interface()
        elif device == "npu":
            active_driver = NpuDriver()
            self.device_count = active_driver.get_device_count()
            self.di = active_driver.get_device_interface()
        elif device == "musa":
            active_driver = MusaDriver()
            self.device_count = active_driver.get_device_count()
            self.di = active_driver.get_device_interface()
        else:
            raise NotImplementedError("device not configed: {}".format(device))
        self.di.set_device(self.rank % self.device_count)
        self.results = torch.empty(0, device=self.device)
        self.gather_result = None

    def finish(self):
        self.results = None
        self.gather_result = None
        dist.barrier()

    def run(self, args: NCCLBandwidthArgs):
        world_size = dist.get_world_size()
        tensor_length = int(args.size_in_mb * 1024 ** 2 / 4.0)

        if args.operation == OperationType.ALL_REDUCE:
            # 1024 is MB to GB
            # 2 考虑双向传输
            # (world_size - 1) 实际传输量修正因子，自拷贝也会带来开销，但是这里不考虑
            trans_size_GB = 2 * args.size_in_mb * ((world_size - 1)) / world_size / 1024
            test_tensor = torch.randn(tensor_length, device=self.device)

            def fn():
                dist.all_reduce(test_tensor[0 : tensor_length - 1], op=dist.ReduceOp.SUM)
                dist.barrier()

        elif args.operation == OperationType.ALL_GATHER:
            local_tensor_length = int(tensor_length / world_size)
            tensor_length = local_tensor_length * world_size  # Ensure tensor_length is divisible by world_size
            output_tensor = torch.rand(tensor_length, device=self.device)
            input_tensor = torch.rand(local_tensor_length, device=self.device)
            # TODO: maybe not equal to tensor_length， same issue with all_to_all
            trans_size_GB = args.size_in_mb * (world_size - 1) / world_size / 1024

            def fn():
                dist.all_gather_into_tensor(output_tensor, input_tensor)
                dist.barrier()

        elif args.operation == OperationType.ALL_TO_ALL:
            # tensor_length must be divisible by world_size
            local_tensor_length = int(tensor_length / world_size)
            input_tensor = torch.randn(world_size, local_tensor_length, device=self.device)
            output_tensor = torch.empty_like(input_tensor)
            trans_size_GB = args.size_in_mb * (world_size - 1) / world_size / 1024
        
            def fn():
                dist.all_to_all_single(output_tensor, input_tensor)
                dist.barrier()
        else:
            raise ValueError(f"Unsupported operation: {args.operation}")

        with torch.inference_mode():
            op_times = bench_timer(fn, device=self.device, warmups=args.warmups, 
                                   repeats=args.repeats, return_mode=args.return_mode,
                                   timeout=args.timeout)

        op_bw = trans_size_GB / op_times[0] * 1000  # GB/s

        self.run_args.append(args)
        self.results = torch.cat([self.results, torch.tensor([op_bw], device=self.device)])

    def print_header(self):
        if self.rank == 0:
            print(f"\n## {self.name} benchmark:", flush=True)
            print("\tgpus: {}".format(dist.get_world_size()))
            for args in self.run_args:
                print("\t{}".format(args), flush=True)
            print("Results: (GB/s)", flush=True)
    
    def print_result(self):
        from tabulate import tabulate

        if self.rank == 0:
            self.gather_result = [self.results.tolist()]
            header = []
            for args in self.run_args:
                header.append(f"{args.operation.name}-{args.size_in_mb}MB")
            print(tabulate(self.gather_result, headers=header, tablefmt="github"), flush=True)
            print("", flush=True)  # Add a new line


if __name__ == "__main__":
    dist.init_process_group()
    all_args = [
        NCCLBandwidthArgs(OperationType.ALL_REDUCE, 1024),
        NCCLBandwidthArgs(OperationType.ALL_GATHER, 1024),
        NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 1024),
    ]
    bench = NCCLBandwidthBench("npu")
    for args in all_args:
        bench.run(args)
    bench.print_header()
    bench.print_result()
    bench.finish()

    dist.destroy_process_group()
