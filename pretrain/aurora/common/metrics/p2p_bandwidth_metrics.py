from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

class P2PMetrics(Enum):
    unidirectional_bandwidth = 'unidirectional_bandwidth'
    bidirectional_bandwidth = 'bidirectional_bandwidth'
    latency = 'latency'

@dataclass
class P2PBandwidthMeta:
    tester: str
    test_time: str
    hardware_name: str
    world_size: int = -1
    gpu_num: int = -1
    gpu_per_node: int = -1
    platform_provider: str = 'cloud'

@dataclass
class P2PBandwidthPerformanceMetrics:
    rank: int
    warmup_times: int
    repeat_times: int
    size_in_mb: int
    p2p_metric: P2PMetrics
    gpu_0: float = None
    gpu_1: float = None
    gpu_2: float = None
    gpu_3: float = None
    gpu_4: float = None
    gpu_5: float = None
    gpu_6: float = None
    gpu_7: float = None
