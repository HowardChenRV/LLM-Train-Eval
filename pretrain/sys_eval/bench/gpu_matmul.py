import time
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from sys_eval.utils.timer import bench_timer
from sys_eval.utils.runtime_driver import CudaDriver, NpuDriver
from sys_eval.bench.bench_core import BenchmarkBase


@dataclass
class GEMMBenchArgs:
    m: int
    n: int
    k: int
    warmups: int = 3
    repeats: int = 10
    return_mode: list = field(default_factory=lambda: ["mean"])
    timeout: int = 120


class GEMMBench(BenchmarkBase):
    def __init__(self, device="cuda"):
        self.name = "GEMMBench"
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

    def finish(self):
        self.results = None
        self.gather_result = None
        dist.barrier()

    def run(self, args: GEMMBenchArgs):
        torch.manual_seed(1234)  # Set seed for reproducibility
        dtype_bf16 = torch.bfloat16
        dtype_fp16 = torch.float16
        iterval = 5  # Interval between benchmarks

        nFLOPS = 2 * args.m * args.n * args.k  # FLOPS
        bf16_a = torch.randn(args.m, args.k, device=self.device, dtype=dtype_bf16)
        bf16_b = torch.randn(args.n, args.k, device=self.device, dtype=dtype_bf16).transpose(-1, -2)
        fp16_a = torch.randn(args.m, args.k, device=self.device, dtype=dtype_fp16)
        fp16_b = torch.randn(args.n, args.k, device=self.device, dtype=dtype_fp16).transpose(-1, -2)

        with torch.inference_mode():
            ms_bf16 = bench_timer(lambda: torch.matmul(bf16_a, bf16_b), warmups=args.warmups, repeats=args.repeats, 
                                  device=self.device, return_mode=args.return_mode, timeout=args.timeout)
            time.sleep(iterval)
            ms_fp16 = bench_timer(lambda: torch.matmul(fp16_a, fp16_b), warmups=args.warmups, repeats=args.repeats, 
                                  device=self.device, return_mode=args.return_mode, timeout=args.timeout)
            time.sleep(iterval)
        tflops_bf16 = nFLOPS / ms_bf16[0] * 1e-9
        tflops_fp16 = nFLOPS / ms_fp16[0] * 1e-9

        self.run_args.append(args)
        self.results = torch.cat([self.results, torch.tensor([tflops_bf16, tflops_fp16], device=self.device)])
        dist.barrier()

    def print_header(self):
        if self.rank == 0:
            print(f"\n## {self.name} benchmark:", flush=True)
            print("\tgpus: {}".format(dist.get_world_size()))
            for args in self.run_args:
                print("\t{}".format(args), flush=True)
            print("Results: (TFLOPS)", flush=True)

    def print_result(self):
        from tabulate import tabulate

        if self.rank == 0:
            self.gather_result = [torch.zeros_like(self.results) for _ in range(dist.get_world_size())]
        else:
            self.gather_result = None
        dist.barrier()

        if self.device == "npu":
            # huawei hccl not support gather yet.
            self.gather_result = [torch.zeros_like(self.results) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(self.gather_result, self.results)
        else:
            dist.gather(self.results, self.gather_result, dst=0)
            dist.barrier()

        if self.rank == 0:
            header = []
            for args in self.run_args:
                header.append(f"{args.m}x{args.n}x{args.k}-bf16")
                header.append(f"{args.m}x{args.n}x{args.k}-fp16")

            table = [result.tolist() for result in self.gather_result]
            print(tabulate(table, headers=header, tablefmt="github", showindex="always"), flush=True)
            print("", flush=True)  # Add a new line


if __name__ == "__main__":
    from sys_eval.utils.timer import timeout_handler
    dist.init_process_group()

    all_args = [
        GEMMBenchArgs(16384, 8192, 1280),
        GEMMBenchArgs(16384, 1024, 8192),
        GEMMBenchArgs(8192, 8192, 8192),
    ]

    bench = GEMMBench()
    for args in all_args:
        bench.run(args)
    bench.print_header()
    bench.print_result()
    bench.finish()

    dist.barrier()
    dist.destroy_process_group()
