import time
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from sys_eval.utils.ctypes_lib import P2PBandwidthLatency
from sys_eval.bench.bench_core import BenchmarkBase
from sys_eval.utils.runtime_driver import CudaDriver
from sys_eval.utils.timer import timeout_handler


@dataclass
class P2PBandwidthLatencyArgs:
    size_in_mb: int
    warmups: int = 1  # no use
    repeats: int = 3  # no use
    return_mode: list = field(default_factory=lambda: ["mean"])
    timeout: int = 120


class P2PBandwidthLatencyBench(BenchmarkBase):
    def __init__(self, device="cuda"):
        self.name = "P2PBandwidthLatency"
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

    def run(self, args: P2PBandwidthLatencyArgs):
        if self.rank % self.device_count == 0:
            p2p_bw_latency = P2PBandwidthLatency()
            num_elems = args.size_in_mb * 1024 * 1024
            p2p_method = 0  # 0: P2P_WRITE, 1: P2P_READ
            p2p_enable = True

            matrixs = [0, 0, 0]
            def run_fn():
                matrixs[0] = p2p_bw_latency.unidirectional_bandwidth_matrix(
                    num_elems, self.device_count, p2p_enable, p2p_method, args.warmups, args.repeats)
                matrixs[1] = p2p_bw_latency.bidirectional_bandwidth_matrix(
                    num_elems, self.device_count, p2p_enable, args.warmups, args.repeats)
                matrixs[2] = p2p_bw_latency.latency_matrix(
                    self.device_count, p2p_enable, p2p_method, args.warmups, args.repeats)
        
            if args.timeout > 0:
                with timeout_handler(args.timeout):
                    run_fn()
            else:
                run_fn()

            self.run_args.append(args)
            self.di.set_device(self.rank % self.device_count)
            result = torch.tensor(matrixs,
                                  device=self.device)
            self.results = torch.cat([self.results, result])
        
        dist.barrier()
    
    def print_header(self):
        if dist.get_rank() == 0:
            print(f"\n## {self.name} benchmark", flush=True)
            print("\tgpus: {}".format(dist.get_world_size()))
            for args in self.run_args:
                print("\t{}".format(args), flush=True)
            print("Results:", flush=True)

    def print_result(self):
        from tabulate import tabulate

        group = dist.new_group([i for i in range(dist.get_world_size()) 
                                if i % self.device_count == 0])

        if self.rank == 0:  # group_rank == 0
            self.gather_result = [torch.zeros_like(self.results) for _ in 
                                  range(dist.get_world_size() // self.device_count)]
        else:
            self.gather_result = None
        dist.barrier()

        if self.rank % self.device_count == 0:
            dist.gather(self.results, self.gather_result, dst=0, group=group)
        dist.barrier()

        if self.rank == 0:
            for idx, args in enumerate(self.run_args):
                print("Unidirectional Bandwidth Matrix (GB/s): size={}MB".format(args.size_in_mb), flush=True)
                bw_table = [result[idx * 3 + 0].reshape((self.device_count, self.device_count)) 
                            for result in self.gather_result]
                bw_table = torch.cat(bw_table, dim=0)
                table_name = ["gpu:{}".format(i) for i in range(self.device_count)]
                print(tabulate(bw_table.tolist(), headers=table_name, 
                               tablefmt="github", showindex="always"), flush=True)
                print("", flush=True)  # Add a new line

                print("Bidirectional Bandwidth Matrix (GB/s): size={}MB".format(args.size_in_mb), flush=True)
                bw_table = [result[idx * 3 + 1].reshape((self.device_count, self.device_count)) 
                            for result in self.gather_result]
                bw_table = torch.cat(bw_table, dim=0)
                table_name = ["gpu:{}".format(i) for i in range(self.device_count)]
                print(tabulate(bw_table.tolist(), headers=table_name, 
                               tablefmt="github", showindex="always"), flush=True)
                print("", flush=True)  # Add a new line

                print("Latency Matrix (us): size={}MB".format(args.size_in_mb), flush=True)
                latency_table = [result[idx * 3 + 2].reshape((self.device_count, self.device_count)) 
                                 for result in self.gather_result]
                latency_table = torch.cat(latency_table, dim=0)
                table_name = ["gpu:{}".format(i) for i in range(self.device_count)]
                print(tabulate(latency_table.tolist(), headers=table_name, 
                               tablefmt="github", showindex="always"), flush=True)
                print("", flush=True)  # Add a new line

    
if __name__ == "__main__":
    dist.init_process_group("nccl")
    bench = P2PBandwidthLatencyBench()
    all_args = [
        P2PBandwidthLatencyArgs(32),
        P2PBandwidthLatencyArgs(64),
    ]
    for args in all_args:
        bench.run(args)
    bench.print_header()
    bench.print_result()
    bench.finish()

    dist.barrier()
    dist.destroy_process_group()