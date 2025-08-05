import time
import signal
import torch
from contextlib import contextmanager
from typing import Any, Dict, List, Callable
from . import runtime_driver


@contextmanager
def timeout_handler(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timeout of {seconds} seconds reached")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


RETURN_MODES = ("min", "max", "mean", "median", "all")


def bench_timer(fn: Callable, device: str="cuda", 
                warmups: int=-1, repeats: int=-1, 
                return_mode: list[str]=["mean"],
                timeout: int=0) -> float|List[float]:
    assert set(return_mode).issubset(RETURN_MODES)
    # 如果返回 all 则不可以包含其他返回值
    if "all" in return_mode:
        assert len(return_mode) == 1
    
    if warmups == -1:
        warmups = 3
    if repeats == -1:
        repeats = 10

    if device == "cuda":
        active_driver = runtime_driver.CudaDriver()
        di = active_driver.get_device_interface()
    elif device == "npu":
        active_driver = runtime_driver.NpuDriver()
        di = active_driver.get_device_interface()
    elif device == "musa":
        active_driver = runtime_driver.MusaDriver()
        di = active_driver.get_device_interface()
    else:
        raise NotImplementedError("device not configed: {}".format(device))

    cache = active_driver.get_empty_cache_for_benchmark()

    start_event = [di.Event(enable_timing=True) for i in range(repeats)]
    end_event = [di.Event(enable_timing=True) for i in range(repeats)]
    

    def run_fn(): 
        # Warmup
        for _ in range(warmups):
            fn()
        di.synchronize()

        # Benchmark
        for i in range(repeats):
            # clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            start_event[i].record()
            fn()
            end_event[i].record()
    
    if timeout > 0:
        with timeout_handler(timeout):
            run_fn()
    else:
        run_fn()

    # Record clocks
    di.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    
    if return_mode == "all":
        return times.tolist()
    else:
        return [getattr(torch, mode)(times).item() for mode in return_mode]
