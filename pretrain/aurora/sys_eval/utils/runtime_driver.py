from abc import ABCMeta, abstractmethod
from typing import Callable, List, Protocol, Sequence


class DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def get_active_torch_device(self):
        pass

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args


class HIPDriver(GPUDriver):

    def __init__(self):
        super().__init__()

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        import torch
        return torch.version.hip is not None

    def get_active_torch_device(self):
        import torch
        # when using hip devices, the device string in pytorch is "cuda"
        return torch.device("cuda", self.get_current_device())

    def get_empty_cache_for_benchmark(self):
        import torch

        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')


    def get_empty_cache_for_benchmark(self):
        import torch

        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
    

class CudaDriver(GPUDriver):

    def __init__(self):
        super().__init__()

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    def get_device_count(self):
        return self.get_device_interface().device_count()

    @staticmethod
    def is_active():
        import torch
        return torch.cuda.is_available() and (torch.version.hip is None)

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
        


class NpuDriver(GPUDriver):

    def __init__(self):
        super().__init__()

    def get_active_torch_device(self):
        import torch
        import torch_npu
        return torch.device("npu", self.get_current_device())

    def get_device_interface(self):
        import torch
        import torch_npu
        return torch_npu.npu

    def get_device_count(self):
        return self.get_device_interface().device_count()

    @staticmethod
    def is_active():
        import torch
        import torch_npu
        return torch_npu.npu.is_available() and (torch.version.hip is None)

    def get_empty_cache_for_benchmark(self):
        import torch
        import torch_npu

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='npu')
        

class MusaDriver(GPUDriver):

    def __init__(self):
        super().__init__()

    def get_active_torch_device(self):
        import torch
        import musa_patch
        return torch.device("musa", self.get_current_device())

    def get_device_interface(self):
        import torch
        import musa_patch
        return torch.cuda

    def get_device_count(self):
        return self.get_device_interface().device_count()

    @staticmethod
    def is_active():
        import torch
        import musa_patch
        return torch.cuda.is_available() and (torch.version.hip is None)

    def get_empty_cache_for_benchmark(self):
        import torch
        import musa_patch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='musa')