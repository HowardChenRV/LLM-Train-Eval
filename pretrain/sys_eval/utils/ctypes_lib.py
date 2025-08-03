import ctypes
import pytest
from typing import Optional, Any, List, Dict
from dataclasses import dataclass

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class MemcpyBandwidth:
    """
    extern "C" float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
        bool wc, unsigned int nWarmups, unsigned int nRepeats);
    extern "C" float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode,
        bool wc, unsigned int nWarmups, unsigned int nRepeats);
    extern "C" float testDeviceToDeviceTransfer(unsigned int memSize, 
        unsigned int nWarmups, unsigned int nRepeats);
    """
    exported_functions = [
        Function("testDeviceToHostTransfer", ctypes.c_float, 
                    [ctypes.c_uint, ctypes.c_int, ctypes.c_bool, ctypes.c_uint, ctypes.c_uint]),
        Function("testHostToDeviceTransfer", ctypes.c_float,
                    [ctypes.c_uint, ctypes.c_int, ctypes.c_bool, ctypes.c_uint, ctypes.c_uint]),
        Function("testDeviceToDeviceTransfer", ctypes.c_float,
                    [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint])
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = f'{_THIS_DIR}/../csrc/lib_memcpy_bandwidth.so'):
        if so_file not in MemcpyBandwidth.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            MemcpyBandwidth.path_to_library_cache[so_file] = lib
        self.lib = MemcpyBandwidth.path_to_library_cache[so_file]

        if so_file not in MemcpyBandwidth.path_to_dict_mapping:
            _funcs = {}
            for func in MemcpyBandwidth.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            MemcpyBandwidth.path_to_dict_mapping[so_file] = _funcs
        self.funcs = MemcpyBandwidth.path_to_dict_mapping[so_file]

    def d2h_transfer(self, mem_size: int, mem_mode: int, wc: bool, warmups: int, repeats: int) -> float:
        return self.funcs["testDeviceToHostTransfer"](mem_size, mem_mode, wc, warmups, repeats)

    def h2d_transfer(self, mem_size: int, mem_mode: int, wc: bool, warmups: int, repeats: int) -> float:
        return self.funcs["testHostToDeviceTransfer"](mem_size, mem_mode, wc, warmups, repeats)
    
    def d2d_transfer(self, mem_size: int, warmups: int, repeats: int) -> float:
        return self.funcs["testDeviceToDeviceTransfer"](mem_size, warmups, repeats)
        # TODO: 如何捕获OOM异常


class P2PBandwidthLatency:
    """
    extern "C" double* testBandwidthMatrix(int numElems, int numGPUs, bool p2p, 
        P2PDataTransfer p2p_method, unsigned int nWarmups, unsigned int nRepeats);
    extern "C" double* testBidirectionalBandwidthMatrix(int numElems, int numGPUs, bool p2p,
        unsigned int nWarmups, unsigned int nRepeats);
    extern "C" double* testLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method,
        unsigned int nWarmups, unsigned int nRepeats);
    """
    exported_functions = [
        Function("testBandwidthMatrix", ctypes.POINTER(ctypes.c_double),
                    [ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]),
        Function("testBidirectionalBandwidthMatrix", ctypes.POINTER(ctypes.c_double),
                    [ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_uint, ctypes.c_uint]),
        Function("testLatencyMatrix", ctypes.POINTER(ctypes.c_double),
                    [ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_uint, ctypes.c_uint])
    ]

    path_to_library_cache: Dict[str, Any] = {}
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = f'{_THIS_DIR}/../csrc/lib_p2p_bandwidth_latency.so'):
        if so_file not in P2PBandwidthLatency.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            P2PBandwidthLatency.path_to_library_cache[so_file] = lib
        self.lib = P2PBandwidthLatency.path_to_library_cache[so_file]

        if so_file not in P2PBandwidthLatency.path_to_dict_mapping:
            _funcs = {}
            for func in P2PBandwidthLatency.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            P2PBandwidthLatency.path_to_dict_mapping[so_file] = _funcs
        self.funcs = P2PBandwidthLatency.path_to_dict_mapping[so_file]

    def unidirectional_bandwidth_matrix(self, num_elems: int, num_gpus: int, p2p: bool, p2p_method: int,
                                        warmups: int, repeats: int) -> List[float]:
        result_ptr = self.funcs["testBandwidthMatrix"](num_elems, num_gpus, p2p, p2p_method, warmups, repeats)
        return [result_ptr[i] for i in range(num_gpus * num_gpus)]
    
    def bidirectional_bandwidth_matrix(self, num_elems: int, num_gpus: int, p2p: bool, warmups: int,
                                       repeats: int) -> List[float]:
        result_ptr = self.funcs["testBidirectionalBandwidthMatrix"](num_elems, num_gpus, p2p, warmups, repeats)
        return [result_ptr[i] for i in range(num_gpus * num_gpus)]
    
    def latency_matrix(self, num_gpus: int, p2p: bool, p2p_method: int, warmups: int, 
                       repeats: int) -> List[float]:
        result_ptr = self.funcs["testLatencyMatrix"](num_gpus, p2p, p2p_method, warmups, repeats)
        return [result_ptr[i] for i in range(num_gpus * num_gpus)]
    

############################# test ########################################

def test_memcpy_bw_shared_api():
    import torch
    torch.cuda.set_device(0)  # TODO: 是否需要使用CUDA设置device
    lib = MemcpyBandwidth()
    test_size = 1024 * 1024 * 1024  # 1GB
    mem_mode = 0  # 0: page-locked memory, 1: pageable memory
    warmups = 5
    repeats = 20
    d2h_bw = lib.d2h_transfer(test_size, mem_mode, False, warmups, repeats)
    h2d_bw = lib.h2d_transfer(test_size, mem_mode, False, warmups, repeats)
    warmups = 50
    repeats = 200
    d2d_bw = lib.d2d_transfer(test_size, warmups=warmups, repeats=repeats)
    assert d2h_bw > 0 and h2d_bw > 0 and d2d_bw > 0
    print(f"\nDevice to Host Bandwidth: {d2h_bw} GB/s")
    print(f"Host to Device Bandwidth: {h2d_bw} GB/s")
    print(f"Device to Device Bandwidth: {d2d_bw} GB/s")


def test_p2p_bw_latency_shared_api():
    import torch
    gpu_count = torch.cuda.device_count()
    lib = P2PBandwidthLatency()
    num_elems = 100 * 1024 * 1024  # 100MB
    p2p_method = 0  # 0: P2P_WRITE, 1: P2P_READ
    p2p_enable = True  # P2P_ENABLE, P2P_DISABLE
    warmups = 5
    repeats = 20
    bw_matrix = lib.unidirectional_bandwidth_matrix(num_elems, gpu_count, p2p_enable, p2p_method, warmups, repeats)
    bidirectional_bw_matrix = lib.bidirectional_bandwidth_matrix(num_elems, gpu_count, p2p_enable, warmups, repeats)
    warmups = 20
    repeats = 200
    latency_matrix = lib.latency_matrix(gpu_count, p2p_enable, p2p_method, warmups, repeats)

    assert len(bw_matrix) == gpu_count * gpu_count
    assert len(bidirectional_bw_matrix) == gpu_count * gpu_count
    assert len(latency_matrix) == gpu_count * gpu_count

    print("\nUnidirectional p2p Bandwidth Matrix: GB/s")
    for i in range(gpu_count):
        print(bw_matrix[i*gpu_count: (i+1)*gpu_count])

    print("\nBidirectional p2p Bandwidth Matrix: GB/s")
    for i in range(gpu_count):
        print(bidirectional_bw_matrix[i*gpu_count: (i+1)*gpu_count])

    print("\np2p Latency Matrix: ns")
    for i in range(gpu_count):
        print(latency_matrix[i*gpu_count: (i+1)*gpu_count])
