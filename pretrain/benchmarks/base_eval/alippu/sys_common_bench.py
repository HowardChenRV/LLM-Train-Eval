import os
import torch.distributed as dist
from sys_eval.bench.gpu_matmul import GEMMBenchArgs, GEMMBench
from sys_eval.bench.gpu_mem_bandwidth import MemcpyBandwidthArgs, MemcpyBandwidthBench
from sys_eval.bench.gpu_p2p_bandwidth import P2PBandwidthLatencyArgs, P2PBandwidthLatencyBench
from sys_eval.bench.nccl_bandwidth import OperationType, NCCLBandwidthArgs, NCCLBandwidthBench
from sys_eval.bench.gpfs_bandwidth import GPFSBandwidthArgs, GPFSBandwidthBench


if __name__ == "__main__":
    dist.init_process_group("nccl")

    gemm_args = [
        GEMMBenchArgs(16384, 8192, 1280),
        GEMMBenchArgs(16384, 1024, 8192),
        GEMMBenchArgs(8192, 8192, 8192),
    ]
    bench = GEMMBench()
    for args in gemm_args:
        bench.run(args)
    bench.print_header()
    bench.print_result()
    bench.finish()

    memcpy_args = [
        MemcpyBandwidthArgs(128),
        MemcpyBandwidthArgs(512),
        MemcpyBandwidthArgs(1024),
    ]
    bench2 = MemcpyBandwidthBench()
    for args in memcpy_args:
        bench2.run(args)
    bench2.print_header()
    bench2.print_result()
    bench2.finish()

    p2p_args = [
        P2PBandwidthLatencyArgs(128),  # not support for metax when p2p bench multi-run
    ]
    bench3 = P2PBandwidthLatencyBench()
    for args in p2p_args:
        bench3.run(args)
    bench3.print_header()
    bench3.print_result()
    bench3.finish()

    nccl_args = [
        NCCLBandwidthArgs(OperationType.ALL_REDUCE, 1024),
        NCCLBandwidthArgs(OperationType.ALL_REDUCE, 2048),
        NCCLBandwidthArgs(OperationType.ALL_REDUCE, 4096),
        NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 1024),
        NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 2048),
        NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 4096),
    ]
    bench4 = NCCLBandwidthBench()
    for args in nccl_args:
        bench4.run(args)
    bench4.print_header()
    bench4.print_result()
    bench4.finish()

    base_path = os.environ.get("BASE_PATH", "/mnt/data/sctest")
    gpfs_args = [
        GPFSBandwidthArgs(512, test_path=base_path, timeout=300),
        GPFSBandwidthArgs(1024, test_path=base_path, timeout=600),
    ]
    bench5 = GPFSBandwidthBench()
    for args in gpfs_args:
        bench5.run(args)
    bench5.print_header()
    bench5.print_result()
    bench5.finish()

    dist.barrier()
    dist.destroy_process_group()
