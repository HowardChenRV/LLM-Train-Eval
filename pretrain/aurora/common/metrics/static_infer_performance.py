from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple
from datetime import datetime


class TestSource(Enum):
    TEST = 1            # 日常业务测试
    CI = 2              # CI
    REGRESSION = 3      # 集成回归
    SCRIPT = 4          # 日常脚本调试、刷数
    EXTERNAL = 5        # 外部数据源导入    


class FrameworkType(Enum):
    VLLM = "vllm"
    LMDEPLOY = "lmdeploy"
    TRT_LLM = "trt_llm"


@dataclass
class InferenceStaticPerformanceMeta:
    tester: str
    test_time: str                                      # 测试开始时间，采用 ISO 8601 格式，YYYY-MM-DDTHH:MM:SS
    stop_time: str                                      # 预留结束时间，测试结束时更新
    hardware_name: str
    hardware_num: int = field(default=-1)
    platform_name: str
    model_name: str
    framework_name: str = field(default=FrameworkType.VLLM)
    framework_version: str
    test_source: int = field(default=TestSource.TEST)
    labels: Dict[str, str] = field(default=None)        # 预留个标签，可以传需求单或者需求文档


@dataclass
class InferenceStaticPerformanceMetrics:
    tp_size: int                            # tensor parallelism size
    batch_size: str
    input_length: str
    output_length: str
    warmup_times: str
    repeat_times: str
    avg_total_latency_ms: float             # latency (ms)
    min_total_latency_ms: float
    max_total_latency_ms: float
    avg_ttft_ms: float                      # ttft (ms)
    min_ttft_ms: float
    max_ttft_ms: float
    avg_tpot_ms: float                      # tpot (ms)
    min_tpot_ms: float
    max_tpot_ms: float
    avg_e2e_throughput: float               # e2e_throughput (token/s)
    min_e2e_throughput: float
    max_e2e_throughput: float
    avg_incremental_throughput: float       # incremental_throughput (token/s)
    min_incremental_throughput: float
    max_incremental_throughput: float
    avg_memory_usage: float                 # memory_usage (percent)
    min_memory_usage: float
    max_memory_usage: float
    avg_power_usage: float                  # power_usage (W)
    min_power_usage: float
    max_power_usage: float
    avg_utilization_usage: float            # device utilization busy (percent)
    min_utilization_usage: float
    max_utilization_usage: float