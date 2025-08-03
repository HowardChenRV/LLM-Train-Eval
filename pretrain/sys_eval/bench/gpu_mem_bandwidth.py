import time
import torch
from dataclasses import dataclass, field
import torch.distributed as dist
from sys_eval.utils.ctypes_lib import MemcpyBandwidth
from sys_eval.bench.bench_core import BenchmarkBase
from sys_eval.utils.runtime_driver import CudaDriver
from sys_eval.utils.timer import timeout_handler


@dataclass
class MemcpyBandwidthArgs:
    size_in_mb: int
    warmups: int = 3
    repeats: int = 10
    return_mode: list = field(default_factory=lambda: ["mean"])  # not used
    timeout: int = 240


class MemcpyBandwidthBench(BenchmarkBase):
    def __init__(self, device="cuda"):
        self.name = "MemcpyBandwidth"
        self.device = device
        self.run_args = []
        self.rank = dist.get_rank()
        if device == "cuda":
            active_driver = CudaDriver()
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

    def run(self, args: MemcpyBandwidthArgs):
        test_bandwidth = MemcpyBandwidth()
        mem_mode = 0  # 0: page-locked memory, 1: pageable memory
        test_size = args.size_in_mb * 1024 * 1024
        local_rank = self.rank % self.device_count

        bw = [0, 0, 0]
        def run_fn():
            for idx in range(self.device_count):
                if idx == local_rank:  # one by one to avoid memory bw conflict
                    bw[0] = test_bandwidth.d2h_transfer(test_size, mem_mode, False, args.warmups, args.repeats)
                    bw[1] = test_bandwidth.h2d_transfer(test_size, mem_mode, False, args.warmups, args.repeats)
                    bw[2] = test_bandwidth.d2d_transfer(test_size, args.warmups, args.repeats)
                dist.barrier()
        
        if args.timeout > 0:
            with timeout_handler(args.timeout):
                run_fn()
        else:
            run_fn()

        self.run_args.append(args)
        self.results = torch.cat([self.results, torch.tensor(bw, device=self.device)])
        dist.barrier()

    def print_header(self):
        if self.rank == 0:
            print(f"\n## {self.name} benchmark", flush=True)
            print("\tgpus: {}".format(dist.get_world_size()), flush=True)
            for args in self.run_args:
                print("\t{}".format(args), flush=True)
            print("Results: (GB/s)", flush=True)

    def print_result(self):
        from tabulate import tabulate

        if self.rank == 0:
            self.gather_result = [torch.zeros_like(self.results) for _ in range(dist.get_world_size())]
        else:
            self.gather_result = None
        dist.barrier()

        dist.gather(self.results, self.gather_result, dst=0)
        dist.barrier()

        if self.rank == 0:
            header = []
            for args in self.run_args:
                header.append(f"d2h-{args.size_in_mb}MB")
                header.append(f"h2d-{args.size_in_mb}MB")
                header.append(f"d2d-{args.size_in_mb}MB")  
            
            table = [result.tolist() for result in self.gather_result]
            print(tabulate(table, headers=header, tablefmt="github", showindex="always"), flush=True)
            print("", flush=True)  # Add a new line


if __name__ == "__main__":
    dist.init_process_group("nccl")

    all_args = [
        MemcpyBandwidthArgs(128),
        MemcpyBandwidthArgs(512),
        MemcpyBandwidthArgs(1024),
    ]

    bench = MemcpyBandwidthBench()
    for args in all_args:
        bench.run(args)
    bench.print_header()
    bench.print_result()
    bench.finish()

    dist.barrier()
    dist.destroy_process_group()
