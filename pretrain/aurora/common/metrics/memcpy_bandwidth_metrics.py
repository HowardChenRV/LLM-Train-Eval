from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


@dataclass
class MemcpyBandwidthMeta:
    tester: str
    test_time: str
    hardware_name: str
    world_size: int = -1
    gpu_num: int = -1
    gpu_per_node: int = -1
    platform_provider: str = 'cloud'


@dataclass
class MemcpyBandwidthPerformanceMetrics:
    rank: int
    warmup_times: int
    repeat_times: int
    size_in_mb: int
    d2h_bw: float = None
    h2d_bw: float = None
    d2d_bw: float = None
