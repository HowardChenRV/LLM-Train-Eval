from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

class Precision(Enum):
    FP16 = 'fp16'
    BF16 = 'bf16'
    FP8 = 'fp8'
    INT8 = 'int8'
    FP32 = 'fp32'
    FP64 = 'fp64'


class Operator(Enum):
    MATMUL = 'matmul'
    CONV2D = 'conv2d'
    CONV3D = 'conv3d'


@dataclass
class GemmOperatorPerformanceMeta:
    tester: str
    test_time: str
    hardware_name: str
    world_size: int = -1
    gpu_num: int = -1
    gpu_per_node: int = -1
    platform_provider: str = 'cloud'


@dataclass
class GemmOperatorPerformanceMetrics:
    rank: int
    precision: Precision
    operator: Operator
    warmup_times: int
    repeat_times: int
    shape1_tflops: float = None
    shape2_tflops: float = None
    shape3_tflops: float = None
    shape4_tflops: float = None
    shape5_tflops: float = None
    shape6_tflops: float = None
