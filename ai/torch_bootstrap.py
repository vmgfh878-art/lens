from __future__ import annotations

import os
import sys
from typing import Any


_BOOTSTRAPPED_TORCH: Any | None = None


def configure_torch_environment(*, cpu_only: bool = False, nvml_cuda_check: bool = True) -> None:
    """Windows torch DLL 로드 전에 필요한 환경변수를 설정한다."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if nvml_cuda_check:
        os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def bootstrap_torch(*, cpu_only: bool = False, nvml_cuda_check: bool = True):
    """pandas/numpy보다 먼저 torch를 로드하기 위한 공통 진입점."""
    global _BOOTSTRAPPED_TORCH
    configure_torch_environment(cpu_only=cpu_only, nvml_cuda_check=nvml_cuda_check)
    if _BOOTSTRAPPED_TORCH is None:
        import torch

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        _BOOTSTRAPPED_TORCH = torch
    return _BOOTSTRAPPED_TORCH


def torch_is_loaded() -> bool:
    return "torch" in sys.modules
