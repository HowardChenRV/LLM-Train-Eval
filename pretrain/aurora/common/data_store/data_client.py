import os
import json
import uuid
import pytz
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Union, Dict
from pathlib import Path
from kafka.producer.future import FutureRecordMetadata
from pydantic import validate_call
import torch
import torch.distributed as dist
from dataclasses import asdict
from aurora.common.data_store.kafka_helper import get_producer
from aurora.common.logger import logger

shanghai_tz = pytz.timezone('Asia/Shanghai')  # 统一使用上海时区


def get_timestamp_in_ms():
    """Helper function to get timestamp in ms

    Returns:
        [int]: [timestamp in ms]
    """
    return round(time.time() * 1000.0)


class TestType(Enum):
    """
    测试类型枚举
    """
    
    STATIC_INFERENCE_PERFORMANCE = 11  # 推理性能 - 静态
    SERVING_INFERENCE_PERFORMANCE = 12  # 推理性能 - serving
    STATIC_INFERENCE_CORRECTNESS = 13  # 推理正确性 - 静态
    SERVING_INFERENCE_CORRECTNESS = 14  # 推理正确性 - serving

    TRAINING_MEGATRON_PRETRAIN_PERFORMANCE = 21  # 训练性能


class DataType(Enum):
    """
    数据类型枚举
    """
    TEST_STATISTIC = 1  # 本次测试结果的统计数据
    TEST_PROCESS = 2  # 本次测试的过程记录数据，需要时存储


class DataTopic(Enum):
    INFER_MODEL_INFO         = "infer_model_info"
    INFER_HARDWARE_INFO      = "infer_hardware_info"
    INFER_TEST_META          = "infer_test_meta"
    INFER_STOP_META          = "infer_stop_meta"
    INFER_STATIC_PERFORMANCE = "infer_static_performance"
    TRAINING_TEST_META                     = "training_test_meta"
    TRAINING_MEGATRON_PRETRAIN_PERFORMANCE = "training_megatron_pretrain_performance"
    TRAINING_STOP_META                     = "training_stop_meta"
    GEMM_TEST_META                         = "gemm_test_meta"
    GEMM_OP_PERFORMANCE                    = "gemm_op_performance"
    GEMM_STOP_META                         = "gemm_stop_meta"
    MEMCPY_BANDWIDTH_TEST_META             = "memcpy_bandwidth_test_meta"
    MEMCPY_BANDWIDTH_PERFORMANCE           = "memcpy_bandwidth_performance"
    MEMCPY_BANDWIDTH_STOP_META             = "memcpy_bandwidth_stop_meta"
    NCCL_BANDWIDTH_TEST_META             = "nccl_bandwidth_test_meta"
    NCCL_BANDWIDTH_PERFORMANCE           = "nccl_bandwidth_performance"
    NCCL_BANDWIDTH_STOP_META             = "nccl_bandwidth_stop_meta"
    P2P_BANDWIDTH_TEST_META                 = "p2p_bandwidth_test_meta"
    P2P_BANDWIDTH_PERFORMANCE               = "p2p_bandwidth_performance"
    P2P_BANDWIDTH_STOP_META                 = "p2p_bandwidth_stop_meta"
    GPFS_TEST_META                         = "gpfs_test_meta"
    GPFS_BANDWIDTH_PERFORMANCE             = "gpfs_bandwidth_performance"
    GPFS_STOP_META                         = "gpfs_stop_meta"


class CustomProducer:
    def __init__(self, save_dir: Path, task_id: str = None):
        self.file_path = save_dir / f'data_client_{task_id}.log'

    def send(self, topic: str, data: dict):
        with open(self.file_path, 'a') as f:
            f.write(json.dumps({"topic": topic, "data": data}) + '\n')


def convert_objects_to_str(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_objects_to_str(value)
        elif isinstance(value, torch.dtype):  # Check for torch.dtype specifically
            d[key] = str(value)
        else:
            try:
                from megatron.core.enums import ModelType
                if isinstance(value, ModelType):
                    d[key] = str(value)
            except ImportError:
                pass
        # Add other specific type checks if needed
    return d


class DataClient:
    def __init__(self, save_dir: str, args, **kwargs):
        self.task_id = generate_task_id()
        self._custom_producer = CustomProducer(Path(save_dir), task_id=self.task_id)
        self._producer = get_producer()
        self.test_type = args.aurora_test_type
        logger.info("aurora task id: {}".format(self.task_id))
        # send test meta
        if self.test_type == "training/pretrain":
            from aurora.common.metrics.megatron_training_performance import MegatronPretrainPerformanceMeta

            gpu_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))

            test_meta_metrics = MegatronPretrainPerformanceMeta(
                tester = args.aurora_tester,
                test_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
                hardware_name = args.aurora_hardware_name,
                world_size = int(dist.get_world_size() // gpu_per_node),
                gpu_num = dist.get_world_size(),
                gpu_per_node = gpu_per_node,
                platform_provider = args.aurora_platform_provider,
                model_serial = args.aurora_model_serial,
                model_size = args.aurora_model_size,
                framework_name = args.aurora_framework_name,
                framework_version = args.aurora_framework_version,
                precision = "fp16" if args.fp16 else "bf16",
                seq_length = args.seq_length,
                global_batch_size = args.global_batch_size,
                micro_batch_size = args.micro_batch_size,
                optimizer = args.optimizer,
                dp = args.data_parallel_size,
                mp = (args.pipeline_model_parallel_size * args.tensor_model_parallel_size),
                tp = args.tensor_model_parallel_size,
                pp = args.pipeline_model_parallel_size,
                sp = 1 if args.sequence_parallel else 0,
                cp = args.context_parallel_size if args.context_parallel_size else 0,
                vpp = args.virtual_pipeline_model_parallel_size if args.virtual_pipeline_model_parallel_size else 0,
                ep = args.export_model_parallel_size if hasattr(args, "export_model_parallel_size") else 0,
                use_flash_attn = 1 if args.use_flash_attn else 0,
                use_te = 1 if (args.transformer_impl == "transformer_engine") else 0,
                training_args = vars(args)
            )

            # 发布元数据
            self.send_data(DataTopic.TRAINING_TEST_META, 
                           test_meta_metrics)
            
        elif self.test_type == "gemm/matmul":
            from aurora.common.metrics.gemm_op_metrics import GemmOperatorPerformanceMeta
            gpu_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))

            test_meta_metrics = GemmOperatorPerformanceMeta(
                tester = args.aurora_tester,
                test_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
                hardware_name = args.aurora_hardware_name,
                world_size = int(dist.get_world_size() // gpu_per_node),
                gpu_num = dist.get_world_size(),
                gpu_per_node = gpu_per_node,
                platform_provider = args.aurora_platform_provider,
            )

            # 发布元数据
            self.send_data(DataTopic.GEMM_TEST_META,
                            test_meta_metrics)

        elif self.test_type == "memcpy/bandwidth":
            from aurora.common.metrics.memcpy_bandwidth_metrics import MemcpyBandwidthMeta
            gpu_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))

            test_meta_metrics = MemcpyBandwidthMeta(
                tester = args.aurora_tester,
                test_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
                hardware_name = args.aurora_hardware_name,
                world_size = int(dist.get_world_size() // gpu_per_node),
                gpu_num = dist.get_world_size(),
                gpu_per_node = gpu_per_node,
                platform_provider = args.aurora_platform_provider,
            )

            # 发布元数据
            self.send_data(DataTopic.MEMCPY_BANDWIDTH_TEST_META,
                            test_meta_metrics)

        elif self.test_type == "nccl/bandwidth":
            from aurora.common.metrics.nccl_bandwidth_metrics import NCCLBandwidthMeta
            gpu_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))

            test_meta_metrics = NCCLBandwidthMeta(
                tester = args.aurora_tester,
                test_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
                hardware_name = args.aurora_hardware_name,
                world_size = int(dist.get_world_size() // gpu_per_node),
                gpu_num = dist.get_world_size(),
                gpu_per_node = gpu_per_node,
                platform_provider = args.aurora_platform_provider,
            )

            # 发布元数据
            self.send_data(DataTopic.NCCL_BANDWIDTH_TEST_META,
                            test_meta_metrics)

        elif self.test_type == "p2p/bandwidth":
            from aurora.common.metrics.p2p_bandwidth_metrics import P2PBandwidthMeta
            gpu_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))

            test_meta_metrics = P2PBandwidthMeta(
                tester = args.aurora_tester,
                test_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
                hardware_name = args.aurora_hardware_name,
                world_size = int(dist.get_world_size() // gpu_per_node),
                gpu_num = dist.get_world_size(),
                gpu_per_node = gpu_per_node,
                platform_provider = args.aurora_platform_provider,
            )

            self.send_data(DataTopic.P2P_BANDWIDTH_TEST_META, test_meta_metrics)

        elif self.test_type == "gpfs/bandwidth":
            from aurora.common.metrics.gpfs_bandwidth_metrics import GPFSBandwidthMeta
            gpu_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))

            test_meta_metrics = GPFSBandwidthMeta(
                tester = args.aurora_tester,
                test_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
                hardware_name = args.aurora_hardware_name,
                world_size = int(dist.get_world_size() // gpu_per_node),
                gpu_num = dist.get_world_size(),
                gpu_per_node = gpu_per_node,
                platform_provider = args.aurora_platform_provider,
            )

            self.send_data(DataTopic.GPFS_TEST_META, test_meta_metrics)

        elif self.test_type == "upload_record":
            print("upload offline record, file path: {}".format(args.log_file))
        else:
            raise NotImplementedError("test type {} is not supported".format(self.test_type))

    def _send_data(self, topic: str, data: object, callback: Callable = None,
                   errback: Callable = None) -> FutureRecordMetadata:
        # 如果producer未初始化，则直接返回
        if self._producer is None:
            return
        
        future = self._producer.send(topic, data)
        if callback is not None:
            future.add_callback(callback)
        if errback is not None:
            future.add_errback(errback)
        return future

    @validate_call
    def send_data(self, data_topic: DataTopic, data: object, **kwargs) -> FutureRecordMetadata:
        """
        发送测试元数据
        :param task_id: 任务ID
        :param test_type: 测试类型
        :param meta_data: 元数据对象
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return: 异步操作对象
        """

        # @TODO: 此处应避免直接向test_meta和stop_meta频道直接发送数据
        data = {'task_id': self.task_id, 'timestamp': get_timestamp_in_ms(), 'data': convert_objects_to_str(asdict(data))}
        self._custom_producer.send(data_topic.value, data)
        return self._send_data(data_topic.value, data, kwargs.get('callback', None), kwargs.get('errback', None))

    def upload_file(self, file_path: Path, **kwargs) -> FutureRecordMetadata:
        """
        上传本地记录
        :param file_path: 文件路径
        :param topic: 上传的topic
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return: 异步操作对象
        """
        with open(file_path, 'rb') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    self._send_data(record['topic'], record['data'], kwargs.get('callback', None), kwargs.get('errback', None))
                    print(record['data'])
                except Exception as e:
                    print("Failed to parse line: {}".format(line))
                    raise e
            print("Upload file {} successfully".format(file_path))

        if self._producer:
            self._producer.flush()

    def finish(self, **kwargs):
        from aurora.common.metrics.general_metrics import DataClientFinishMetrics

        finish_summary_metrics = DataClientFinishMetrics(
            stop_time = datetime.now().astimezone(shanghai_tz).replace(tzinfo=None).isoformat(),
            summary = kwargs,
        )

        if self.test_type == "training/pretrain":
            self.send_data(DataTopic.TRAINING_STOP_META, finish_summary_metrics)
        elif self.test_type == "gemm/matmul":
            self.send_data(DataTopic.GEMM_STOP_META, finish_summary_metrics)
        elif self.test_type == "memcpy/bandwidth":
            self.send_data(DataTopic.MEMCPY_BANDWIDTH_STOP_META, finish_summary_metrics)
        elif self.test_type == "nccl/bandwidth":
            self.send_data(DataTopic.NCCL_BANDWIDTH_STOP_META, finish_summary_metrics)
        elif self.test_type == "p2p/bandwidth":
            self.send_data(DataTopic.P2P_BANDWIDTH_STOP_META, finish_summary_metrics)
        elif self.test_type == "gpfs/bandwidth":
            self.send_data(DataTopic.GPFS_STOP_META, finish_summary_metrics)
        else:
            raise NotImplementedError("test type {} is not supported".format(self.test_type))
        
        # stop producer
        if self._producer:
            self._producer.flush()


def generate_task_id() -> str:
    """
    生成任务ID（暂定为UUID v4）
    :param test_type: 测试类型
    :return: 任务ID字符串
    """
    return str(uuid.uuid4())


if __name__ == "__main__":
    # 加入本地导入命令行，参数传入导入文件，然后执行导入文件
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True, help="log file path")
    parser.add_argument("--aurora-test-type", type=str, default="upload_record", help="aurora test type")

    args = parser.parse_args()
    data_client = DataClient("data_client", args)

    if args.log_file:
        data_client.upload_file(args.log_file)
