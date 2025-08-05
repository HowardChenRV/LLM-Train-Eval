from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


@dataclass
class DataClientFinishMetrics:
    stop_time: str
    summary: Dict
