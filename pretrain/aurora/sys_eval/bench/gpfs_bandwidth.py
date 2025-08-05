import os
import argparse
import time
import torch
import torch.distributed as dist
from pathlib import Path
from dataclasses import dataclass, field
from aurora.sys_eval.utils.timer import bench_timer
from aurora.sys_eval.bench.bench_core import BenchmarkBase
from aurora.sys_eval.utils.runtime_driver import CudaDriver, NpuDriver


@dataclass
class GPFSBandwidthArgs:
    size_in_mb: int
    warmups: int = 3
    repeats: int = 5
    return_mode: list = field(default_factory=lambda: ["mean"])
    test_path: Path = None
    timeout: int = 120


class GPFSBandwidthBench(BenchmarkBase):
    def __init__(self, device="cuda"):
        self.name = "GPFSBandwidth"
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
        else:
            raise NotImplementedError("device not configed: {}".format(device))
        self.di.set_device(self.rank % self.device_count)
        self.results = torch.empty(0, device=self.device)
        self.gather_result = None
        self.save_path = None

    def finish(self):
        if self.save_path is not None:
            os.remove(self.save_path)
        self.results = None
        self.gather_result = None
        dist.barrier()

    def run(self, args: GPFSBandwidthArgs):
        if args.test_path is None:
            base_path = os.environ.get('BASE_PATH', '/mnt/public')
            test_path = Path(base_path) / 'gpfs_eval'
        else:
            test_path = Path(args.test_path)
        os.makedirs(test_path, exist_ok=True)

        self.save_path = test_path / f'rank_{self.rank}.pt'
        # 错位加载，防止系统缓存影响
        if self.rank + self.device_count < dist.get_world_size():
            self.load_path = test_path / f'rank_{self.rank + self.device_count}.pt'
        else:
            self.load_path = test_path / f'rank_{self.rank % self.device_count}.pt'

        data = torch.randn(1024, 512, args.size_in_mb, dtype=torch.float16, device=self.device)
        iterval = 5  # Interval between benchmarks

        def write_data():
            torch.save(data, self.save_path)
            os.system("sync")
            dist.barrier()

        def read_data():
            torch.load(self.load_path)
            dist.barrier()

        with torch.inference_mode():
            write_times = bench_timer(write_data, device=self.device, warmups=args.warmups, 
                                      repeats=args.repeats, return_mode=args.return_mode,
                                      timeout=args.timeout)
            time.sleep(iterval)
            self.di.empty_cache()
            read_times = bench_timer(read_data, device=self.device, warmups=args.warmups, 
                                     repeats=args.repeats, return_mode=args.return_mode,
                                     timeout=args.timeout)

        self.run_args.append(args)
        total_size_GB = args.size_in_mb / 1024 * dist.get_world_size()
        write_speed = total_size_GB / write_times[0] * 1000
        read_speed = total_size_GB / read_times[0] * 1000
        self.results = torch.cat([self.results, torch.tensor([write_speed, read_speed], device=self.device)])

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
                header.append(f"write-{args.size_in_mb}MB")
                header.append(f"read-{args.size_in_mb}MB")
            print(tabulate(self.gather_result, headers=header, tablefmt="github"), flush=True)
            print("", flush=True)  # Add a new line


if __name__ == "__main__":
    dist.init_process_group(backend='nccl')

    all_args = [
        GPFSBandwidthArgs(128, test_path="/mnt/chenyonghua"),
        GPFSBandwidthArgs(512, test_path="/mnt/chenyonghua"),
        GPFSBandwidthArgs(1024, test_path="/mnt/chenyonghua"),
    ]
    bench = GPFSBandwidthBench()
    for args in all_args:
        bench.run(args)
    bench.print_header()
    bench.print_result()
    bench.finish()

    dist.destroy_process_group()
