
from abc import ABC, abstractmethod


class BenchmarkBase(ABC):
    def __init__(self, args):
        self.name = "BenchmarkBase"

    @abstractmethod
    def finish(self):
        print("BenchmarkBase.stop()")

    def run(self):
        print("BenchmarkBase.run()")

    @abstractmethod
    def print_header(self):
        print(f"Running {self.name} benchmark")

    @abstractmethod
    def print_result(self):
        print(f"{self.name} benchmark result")

    def upload_test_meta(self):
        raise NotImplementedError("upload_test_meta not implemented")

    def upload_perf_result(self):
        raise NotImplementedError("upload_perf_result not implemented")

    def upload_summary(self):
        raise NotImplementedError("upload_summary not implemented")

    
