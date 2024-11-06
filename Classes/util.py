import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import torch


def moving_average(x: np.ndarray, window=200):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def is_debugging() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def n_processes(n_processes_per_gpu: int = 3):
    try:
        # If we have GPUs, then start as many runs as there are GPUs
        cmd = "nvidia-smi --list-gpus"
        output = subprocess.check_output(cmd, shell=True).decode()
        # The driver exists but no GPU is available (for instance, the eGPU is disconnected)
        if "failed" in output:
            return 1
        n_gpus = int(len(output.splitlines()))
        if n_gpus > 0:
            return n_gpus * n_processes_per_gpu
    except subprocess.CalledProcessError:
        pass
    # Otherwise, start only one run at a time on the cpu
    return 1


@dataclass
class GPU:
    index: int
    total_memory: int
    """Total memory (MB)"""
    used_memory: int
    """Used memory (MB)"""
    free_memory: int
    """Free memory (MB)"""
    memory_usage: float
    """Memory usage between 0 and 1"""
    utilization: float
    """Utilization between 0 and 1"""

    def __init__(self, index: int, total_memory: int, used_memory: int, free_memory: int, utilization: int):
        self.index = index
        self.total_memory = total_memory
        self.used_memory = used_memory
        self.free_memory = free_memory
        self.memory_usage = used_memory / total_memory
        self.utilization = utilization / 100


def list_gpus() -> list[GPU]:
    """List all available GPU devices"""
    try:
        cmd = "nvidia-smi  --format=csv,noheader,nounits --query-gpu=memory.total,memory.used,memory.free,utilization.gpu"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        return []
    res = []
    for i, line in enumerate(csv.split("\n")):
        total_memory, used_memory, free_memory, utilization = map(int, line.split(","))
        res.append(
            GPU(
                index=i,
                total_memory=total_memory,
                used_memory=used_memory,
                free_memory=free_memory,
                utilization=utilization,
            )
        )
    return res


def get_gpu_processes():
    try:
        cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
        return set(map(int, csv.split("\n")))
    except subprocess.CalledProcessError:
        # No GPU available
        return set[int]()
    except ValueError:
        # No processes
        return set[int]()


def get_max_gpu_usage(pids: set[int]):
    try:
        cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
        max_memory = 0
        for line in csv.split("\n"):
            pid, used_memory = map(int, line.split(","))
            if pid in pids:
                max_memory = max(max_memory, used_memory)
        return max_memory
    except subprocess.CalledProcessError:
        return 0


def get_device():
    # Cuda seems to use the pid to check whether to free the GPU memory.
    # Therefore, assigning each process to its pid % the number of GPUs seemt to avoid memory leaks
    n_devices = torch.cuda.device_count()
    if n_devices == 0:
        return torch.device("cpu")
    device_num = os.getpid() % n_devices
    return torch.device(f"cuda:{device_num}")
