from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


@dataclass
class NCCLBandwidthMeta:
    tester: str
    test_time: str
    hardware_name: str
    world_size: int = -1
    gpu_num: int = -1
    gpu_per_node: int = -1
    platform_provider: str = 'cloud'


@dataclass
class NCCLBandwidthPerformanceMetrics:
    rank: int
    warmup_times: int
    repeat_times: int
    size_in_mb: int
    all_reduce_bw: float = None
    all_to_all_bw: float = None
    