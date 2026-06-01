"""CP216 — cluster (ticker) + block (time, two lengths) bootstrap CI.

GPU 가속: 전체 resample 을 (n_iter, n_unit) tensor 로 잡고 mean axis=1 한 번에.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BootstrapCI:
    n_iter: int
    ci_lower: float
    ci_upper: float
    point: float
    method: str


def _quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def _resample_means_numpy(arr: np.ndarray, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    idx = rng.integers(0, n, size=(n_iter, n))
    sampled = arr[idx]
    return sampled.mean(axis=1)


def _resample_means_torch(arr: np.ndarray, n_iter: int, device: str, seed: int) -> np.ndarray:
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    n = len(arr)
    arr_t = torch.tensor(arr, dtype=torch.float32, device=device)
    # 메모리 보호: 너무 큰 (n_iter*n) 이면 chunk 로
    max_cells = 50_000_000  # ~200MB float32
    chunk = max(1, min(n_iter, max_cells // max(n, 1)))
    out: list[float] = []
    remaining = n_iter
    while remaining > 0:
        cur = min(chunk, remaining)
        idx = torch.randint(0, n, (cur, n), generator=g, device=device)
        sampled = arr_t[idx]
        means = sampled.mean(dim=1)
        out.append(means.detach().cpu().numpy())
        remaining -= cur
    return np.concatenate(out, axis=0)


# ----------------------------- cluster bootstrap (ticker) -----------------------------


def cluster_bootstrap_ci(
    diff_per_unit: np.ndarray,
    n_iter: int = 1000,
    ci_level: float = 0.95,
    device: str = "cpu",
    seed: int = 0,
) -> BootstrapCI:
    """ticker (cluster) 단위 mean 차이의 부트스트랩 CI.

    diff_per_unit[i] = ticker i 의 평균 (L_A - L_B).
    복원추출 → 1000 resample → 백분위 CI.
    """
    arr = np.asarray(diff_per_unit, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 5:
        return BootstrapCI(
            n_iter=0,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            point=float(arr.mean()) if n else float("nan"),
            method="cluster_ticker",
        )
    point = float(arr.mean())
    if device == "cuda":
        means = _resample_means_torch(arr, n_iter, device, seed)
    else:
        means = _resample_means_numpy(arr, n_iter, np.random.default_rng(seed))
    alpha = (1.0 - ci_level) / 2.0
    return BootstrapCI(
        n_iter=n_iter,
        ci_lower=_quantile(means, alpha),
        ci_upper=_quantile(means, 1.0 - alpha),
        point=point,
        method="cluster_ticker",
    )


# ----------------------------- block bootstrap (time) -----------------------------


def _moving_block_indices_numpy(n: int, block: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Moving Block Bootstrap. n 길이 시계열에서 n 개를 block 길이로 채움."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_iter, n_blocks))
    offsets = np.arange(block)
    # idx[iter, b, off] = starts[iter, b] + off
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_iter, -1)[:, :n]
    return idx


def _resample_means_block_numpy(arr: np.ndarray, block: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    idx = _moving_block_indices_numpy(n, block, n_iter, rng)
    return arr[idx].mean(axis=1)


def _resample_means_block_torch(arr: np.ndarray, block: int, n_iter: int, device: str, seed: int) -> np.ndarray:
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    n = len(arr)
    n_blocks = int(np.ceil(n / block))
    arr_t = torch.tensor(arr, dtype=torch.float32, device=device)
    offsets = torch.arange(block, device=device)
    max_cells = 50_000_000
    cells_per_iter = n_blocks * block
    chunk = max(1, min(n_iter, max_cells // max(cells_per_iter, 1)))
    out: list[float] = []
    remaining = n_iter
    while remaining > 0:
        cur = min(chunk, remaining)
        starts = torch.randint(0, n - block + 1, (cur, n_blocks), generator=g, device=device)
        # broadcast 후 truncate
        idx = starts[:, :, None] + offsets[None, None, :]
        idx = idx.reshape(cur, -1)[:, :n]
        sampled = arr_t[idx]
        means = sampled.mean(dim=1)
        out.append(means.detach().cpu().numpy())
        remaining -= cur
    return np.concatenate(out, axis=0)


def block_bootstrap_ci(
    time_series_diff: np.ndarray,
    block: int,
    n_iter: int = 1000,
    ci_level: float = 0.95,
    device: str = "cpu",
    seed: int = 0,
) -> BootstrapCI:
    arr = np.asarray(time_series_diff, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < max(10, block * 2):
        return BootstrapCI(
            n_iter=0,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            point=float(arr.mean()) if n else float("nan"),
            method=f"block_{block}",
        )
    block = max(1, min(block, n))
    point = float(arr.mean())
    if device == "cuda":
        means = _resample_means_block_torch(arr, block, n_iter, device, seed)
    else:
        means = _resample_means_block_numpy(arr, block, n_iter, np.random.default_rng(seed))
    alpha = (1.0 - ci_level) / 2.0
    return BootstrapCI(
        n_iter=n_iter,
        ci_lower=_quantile(means, alpha),
        ci_upper=_quantile(means, 1.0 - alpha),
        point=point,
        method=f"block_{block}",
    )


def block_size_sqrt_t(n: int) -> int:
    return max(2, int(np.floor(np.sqrt(n))))
