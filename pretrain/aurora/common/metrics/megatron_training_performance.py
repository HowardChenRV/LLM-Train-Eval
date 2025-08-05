from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class FrameworkType(Enum):
    MEGATRON_LM = "megatron_lm"
    MEGATRON_DEEPSPEED = "megatron_deepspeed"
    MEGATRON_INFINIGENCE = "megatron_infinigence"
    DIFFUSERS = "diffusers"


class MODEL_TYPE(Enum):
    GPT3 = "gpt3"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    QWEN2 = "qwen2"
    MISTRAL = "mistral"
    STABLE_DIFFUSION2 = "stable_diffusion2"


class TestSource(Enum):
    TEST = 1            # 日常业务测试
    CI = 2              # CI
    REGRESSION = 3      # 集成回归
    SCRIPT = 4          # 日常脚本调试、刷数
    EXTERNAL = 5        # 外部数据源导入    


# @TODO: 混训性能测试数据结构暂不支持
@dataclass
class MegatronPretrainPerformanceMeta:
    tester: str
    test_time: str                                      # 测试开始时间，采用 ISO 8601 格式，YYYY-MM-DDTHH:MM:SS
    hardware_name: str
    world_size: int = field(default=-1)                 # 分布式训练任务的节点数量
    gpu_num: int = field(default=-1)                    # 所有节点的 GPU 总数量
    gpu_per_node: int = field(default=-1)               # 每个节点的 GPU 数量
    platform_provider: str = field(default='cloud')
    model_serial: str = field(default=MODEL_TYPE.GPT3.value)  # 模型系列名称
    model_size: float = field(default=-1)               # 模型大小，单位 B
    framework_name: str = field(default=FrameworkType.MEGATRON_LM.value)
    framework_version: str = field(default='0.8.0')
    precision: str = field(default='fp16')
    seq_length: int = field(default=-1)                  # 序列长度
    global_batch_size: int = field(default=-1)           # global batch size
    micro_batch_size: int = field(default=-1)            # micro batch size
    optimizer: str = field(default='adam')
    dp: int = field(default=-1)                          # 数据并行度
    mp: int = field(default=-1)                          # 模型并行度
    tp: int = field(default=-1)                          # 测试并行度
    pp: int = field(default=-1)                          # 参数并行度
    sp: int = field(default=-1)                          # 序列并行度
    cp: int = field(default=-1)                          # 控制并行度
    vpp: int = field(default=-1)                         # 虚拟流水线并行度
    ep: int = field(default=-1)                          # 专家并行
    use_flash_attn: bool = field(default=False)          # 是否使用 Flash Attention
    use_te: bool = field(default=False)                  # 是否使用 Transformer Engine
    training_args: Dict = field(default_factory=dict)    # 测试参数


@dataclass
class MegatronPretrainPerformanceMetrics:
    iteration: int
    elapsed_time_per_iteration: float
    samples_per_sec: float
    tokens_per_sec: float
    tokens_per_sec_per_replica: float
    tokens_per_gpu_per_second: float
    tokens_per_gpu_per_second_per_replica: float
    total_tflops: float
    tflops_per_gpu: float
    loss: float
