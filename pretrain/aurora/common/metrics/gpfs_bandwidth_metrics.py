from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


@dataclass
class GPFSBandwidthMeta:
    tester: str
    test_time: str
    hardware_name: str
    world_size: int = -1
    gpu_num: int = -1
    gpu_per_node: int = -1
    platform_provider: str = 'cloud'


@dataclass
class GPFSBandwidthMetrics:
    repeats: int
    base_path: str
    size_in_mb: int
    write_time: float
    read_time: float
    write_bandwidth: float
    read_bandwidth: float
