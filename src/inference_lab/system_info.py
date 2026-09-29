"""Best-effort, dependency-light hardware/software inventory."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

import numpy as np


def collect_system_info() -> dict[str, object]:
    cpu_model = platform.processor() or "not reported by Python"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpu_file:
            for line in cpu_file:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as memory_file:
            cgroup_memory_limit = memory_file.read().strip()
    except OSError:
        cgroup_memory_limit = "not reported"
    info: dict[str, object] = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "cgroup_memory_limit_bytes": cgroup_memory_limit,
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "gpu": None,
        "gpu_probe": "nvidia-smi not available" if shutil.which("nvidia-smi") is None else "nvidia-smi present",
    }
    if shutil.which("nvidia-smi"):
        try:
            completed = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=5, check=True)
            info["gpu"] = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            info["gpu_probe"] = f"nvidia-smi query failed: {type(exc).__name__}"
    return info
