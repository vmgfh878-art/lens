"""CP216 — IC 시계열 / pinball loss 계산.

라인 metric: 각 asof_date 의 ticker 간 Spearman rank correlation = IC_t
밴드 metric: per (ticker, asof, horizon) pinball_low + pinball_high
GPU 가속 pinball: torch + cuda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ----------------------------- 라인 IC -----------------------------


def daily_ic_series(merged_panel: pd.DataFrame, score_col: str, actual_col: str = "actual") -> pd.Series:
    """각 asof_date 에서 ticker 간 Spearman rank correlation 의 시계열.

    Returns: index=asof_date, values=IC_t
    """
    out: list[tuple[str, float]] = []
    for asof, g in merged_panel.groupby("asof_date", sort=True):
        a = g[score_col].astype(float).values
        b = g[actual_col].astype(float).values
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(a[mask], b[mask])
        if np.isnan(rho):
            continue
        out.append((asof, float(rho)))
    s = pd.Series(dict(out)).sort_index()
    s.name = "ic"
    return s


def line_ic_series_pair(
    merged_panel: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """ops vs baseline 한 쌍의 IC 시계열을 동시 반환 (같은 asof 위에서)."""
    ic_a = daily_ic_series(merged_panel, "score_a", "actual")
    ic_b = daily_ic_series(merged_panel, "score_b", "actual")
    common = ic_a.index.intersection(ic_b.index)
    return ic_a.reindex(common), ic_b.reindex(common)


# ----------------------------- 밴드 pinball -----------------------------


def pinball(y: np.ndarray, q_hat: np.ndarray, q: float) -> np.ndarray:
    diff = y - q_hat
    return np.where(diff >= 0, diff * q, -diff * (1.0 - q))


def pinball_total(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_low: float,
    q_high: float,
) -> np.ndarray:
    """row-wise pinball_low(actual, lower) + pinball_high(actual, upper)."""
    pl = pinball(actual, lower, q_low)
    pu = pinball(actual, upper, q_high)
    return pl + pu


def band_pinball_panel(
    merged_panel: pd.DataFrame,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    """per-row pinball for ops (lower_a/upper_a) and baseline (lower_b/upper_b)."""
    actual = merged_panel["actual"].astype(float).values
    mask = (
        np.isfinite(actual)
        & np.isfinite(merged_panel["lower_a"].astype(float).values)
        & np.isfinite(merged_panel["upper_a"].astype(float).values)
        & np.isfinite(merged_panel["lower_b"].astype(float).values)
        & np.isfinite(merged_panel["upper_b"].astype(float).values)
    )
    out = merged_panel.loc[mask].copy()
    out["pinball_a"] = pinball_total(
        out["actual"].astype(float).values,
        out["lower_a"].astype(float).values,
        out["upper_a"].astype(float).values,
        q_low,
        q_high,
    )
    out["pinball_b"] = pinball_total(
        out["actual"].astype(float).values,
        out["lower_b"].astype(float).values,
        out["upper_b"].astype(float).values,
        q_low,
        q_high,
    )
    return out


def band_pinball_daily(panel_with_pinball: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """각 asof_date 의 평균 pinball 시계열 (ops, baseline)."""
    grp = panel_with_pinball.groupby("asof_date", sort=True)
    pa = grp["pinball_a"].mean()
    pb = grp["pinball_b"].mean()
    common = pa.index.intersection(pb.index)
    pa = pa.reindex(common)
    pb = pb.reindex(common)
    pa.name = "pinball_a"
    pb.name = "pinball_b"
    return pa, pb


# ----------------------------- GPU pinball -----------------------------


def pinball_total_torch(actual, lower, upper, q_low: float, q_high: float, device: str = "cuda"):
    """torch 텐서 입력 또는 numpy 입력 모두 받음. 디바이스 옮겨 계산 후 numpy 반환."""
    import torch

    if not isinstance(actual, torch.Tensor):
        actual_t = torch.tensor(actual, dtype=torch.float32, device=device)
        lower_t = torch.tensor(lower, dtype=torch.float32, device=device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=device)
    else:
        actual_t = actual.to(device)
        lower_t = lower.to(device)
        upper_t = upper.to(device)
    dl = actual_t - lower_t
    du = actual_t - upper_t
    pl = torch.where(dl >= 0, dl * q_low, -dl * (1.0 - q_low))
    pu = torch.where(du >= 0, du * q_high, -du * (1.0 - q_high))
    return (pl + pu).detach().cpu().numpy()
