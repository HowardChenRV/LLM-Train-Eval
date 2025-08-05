import os
from argparse import ArgumentParser
import torch
if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    import musa_patch
else:
    pass
import torch.distributed as dist
from aurora.sys_eval.bench.gpu_matmul import GEMMBenchArgs, GEMMBench
from aurora.sys_eval.bench.gpu_mem_bandwidth import MemcpyBandwidthArgs, MemcpyBandwidthBench
from aurora.sys_eval.bench.gpu_p2p_bandwidth import P2PBandwidthLatencyArgs, P2PBandwidthLatencyBench
from aurora.sys_eval.bench.nccl_bandwidth import OperationType, NCCLBandwidthArgs, NCCLBandwidthBench
from aurora.sys_eval.bench.gpfs_bandwidth import GPFSBandwidthArgs, GPFSBandwidthBench
from aurora.common.data_store.data_client import DataTopic, DataClient
from aurora.common.metrics.gemm_op_metrics import GemmOperatorPerformanceMetrics, Precision, Operator
from aurora.common.metrics.memcpy_bandwidth_metrics import MemcpyBandwidthPerformanceMetrics
from aurora.common.metrics.nccl_bandwidth_metrics import NCCLBandwidthPerformanceMetrics


def add_aurora_args(parser):
    group = parser.add_argument_group(title='Data-Client')

    group.add_argument('--aurora-test-type', default=None,
                       choices=["gemm/matmul", "nccl/bandwidth", "gpfs/bandwidth", "p2p/bandwidth", "memcpy/bandwidth"],
                       help='Test type for data client',)

    group.add_argument('--aurora-save-dir', default=None,
                       help='local data store path for data client',)

    group.add_argument('--aurora-tester', default='tester',
                       help='Tester for data client',)
    
    group.add_argument('--aurora-hardware-name', default=None,
                       help='Hardware name for data client',)
    
    group.add_argument('--aurora-platform-provider', default='cloud',
                        help='Platform provider for data client',)
    
    group.add_argument('--aurora-warmup-times', default=3, type=int,
                        help='Warmup times for test case',)

    group.add_argument('--aurora-repeat-times', default=10, type=int,
                        help='Repeat times for test case',)

    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_aurora_args(parser)
    args = parser.parse_args()
    if args.aurora_test_type is None:
        raise ValueError("Please specify the test type using --aurora-test-type")

    dist.init_process_group()
    if dist.get_rank() == 0:
        os.makedirs(args.aurora_save_dir, exist_ok=True)
        data_client = DataClient(args.aurora_save_dir, args)
    else:
        data_client = None

    if args.aurora_test_type == "gemm/matmul":
        gemm_args = [
            GEMMBenchArgs(16384, 8192, 1280, args.aurora_warmup_times, args.aurora_repeat_times),
            GEMMBenchArgs(16384, 1024, 8192, args.aurora_warmup_times, args.aurora_repeat_times),
            GEMMBenchArgs(8192, 8192, 8192, args.aurora_warmup_times, args.aurora_repeat_times),
        ]
        bench = GEMMBench(device="musa")
        for case_inputs in gemm_args:
            bench.run(case_inputs)
        bench.print_header()
        bench.print_result()

        if data_client is not None:
            all_res = [result.tolist() for result in bench.gather_result]
            for idx, res in enumerate(all_res):
                bf16_metrics = GemmOperatorPerformanceMetrics(
                    rank=idx,
                    precision=Precision.FP16.value,
                    operator=Operator.MATMUL.value,
                    warmup_times=args.aurora_warmup_times,
                    repeat_times=args.aurora_repeat_times,
                    shape1_tflops=res[0],
                    shape2_tflops=res[2],
                    shape3_tflops=res[4],
                )
                data_client.send_data(DataTopic.GEMM_OP_PERFORMANCE, bf16_metrics)
                fp16_metrics = GemmOperatorPerformanceMetrics(
                    rank=idx,
                    precision=Precision.FP16.value,
                    operator=Operator.MATMUL.value,
                    warmup_times=args.aurora_warmup_times,
                    repeat_times=args.aurora_repeat_times,
                    shape1_tflops=res[1],
                    shape2_tflops=res[3],
                    shape3_tflops=res[5],
                )
                data_client.send_data(DataTopic.GEMM_OP_PERFORMANCE, fp16_metrics)

            data_client.finish()
        dist.barrier()
        bench.finish()
    elif args.aurora_test_type == "memcpy/bandwidth":
        memcpy_args = [
            MemcpyBandwidthArgs(128, args.aurora_warmup_times, args.aurora_repeat_times),
            MemcpyBandwidthArgs(512, args.aurora_warmup_times, args.aurora_repeat_times),
            MemcpyBandwidthArgs(1024, args.aurora_warmup_times, args.aurora_repeat_times),
        ]
        bench = MemcpyBandwidthBench()
        for case_inputs in memcpy_args:
            bench.run(case_inputs)
        bench.print_header()
        bench.print_result()

        if data_client is not None:
            all_res = [result.tolist() for result in bench.gather_result]
            for idx, res in enumerate(all_res):
                for size_index in range(len(memcpy_args)):
                    memcpy_metrics = MemcpyBandwidthPerformanceMetrics(
                        rank=idx,
                        warmup_times=args.aurora_warmup_times,
                        repeat_times=args.aurora_repeat_times,
                        size_in_mb=memcpy_args[size_index].size_in_mb,
                        d2h_bw=res[size_index * 3 + 0],
                        h2d_bw=res[size_index * 3 + 1],
                        d2d_bw=res[size_index * 3 + 2],
                    )
                    data_client.send_data(DataTopic.MEMCPY_BANDWIDTH_PERFORMANCE, memcpy_metrics)

            data_client.finish()
        dist.barrier()
        bench.finish()
    elif args.aurora_test_type == "nccl/bandwidth":
        nccl_args = [
            NCCLBandwidthArgs(OperationType.ALL_REDUCE, 1024, args.aurora_warmup_times, args.aurora_repeat_times),
            NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 1024, args.aurora_warmup_times, args.aurora_repeat_times),
            NCCLBandwidthArgs(OperationType.ALL_REDUCE, 2048, args.aurora_warmup_times, args.aurora_repeat_times),
            NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 2048, args.aurora_warmup_times, args.aurora_repeat_times),
            NCCLBandwidthArgs(OperationType.ALL_REDUCE, 4096, args.aurora_warmup_times, args.aurora_repeat_times),
            NCCLBandwidthArgs(OperationType.ALL_TO_ALL, 4096, args.aurora_warmup_times, args.aurora_repeat_times),
        ]
        bench = NCCLBandwidthBench(device="musa")
        for case_inputs in nccl_args:
            bench.run(case_inputs)
        bench.print_header()
        bench.print_result()

        if data_client is not None:
            all_res = [bench.results.tolist()]
            for idx, res in enumerate(all_res):
                for size_index in range(len(nccl_args) // 2):
                    nccl_metrics = NCCLBandwidthPerformanceMetrics(
                        rank=idx,
                        warmup_times=args.aurora_warmup_times,
                        repeat_times=args.aurora_repeat_times,
                        size_in_mb=nccl_args[size_index * 2 + 0].size_in_mb,
                        all_reduce_bw=res[size_index * 2 + 0],
                        all_to_all_bw=res[size_index * 2 + 1],
                    )
                    data_client.send_data(DataTopic.NCCL_BANDWIDTH_PERFORMANCE, nccl_metrics)

            data_client.finish()
        dist.barrier()
        bench.finish()

    elif args.aurora_test_type == "gpfs/bandwidth":
        base_path = os.environ.get("BASE_PATH", "/mnt/public/sctest")
        gpfs_args = [
            GPFSBandwidthArgs(128, test_path=base_path, timeout=300),
            GPFSBandwidthArgs(256, test_path=base_path, timeout=600),
        ]
        bench = GPFSBandwidthBench()
        for args in gpfs_args:
            bench.run(args)
        bench.print_header()
        bench.print_result()
        bench.finish()

    else:
        raise ValueError("Unsupported test type: {}".format(args.aurora_test_type))

    dist.barrier()
    dist.destroy_process_group()
