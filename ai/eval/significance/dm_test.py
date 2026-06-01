"""CP216 — Diebold-Mariano test with HAC (Newey-West) variance.

d_t = L_A_t - L_B_t (A=ops, B=baseline). A 가 더 좋으면 mean(d) < 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class DMResult:
    n: int
    mean_diff: float
    dm_stat: float
    hac_lags: int
    p_value_two_sided: float
    p_value_a_better: float  # H1: A < B (mean_diff < 0)
    p_value_b_better: float


def _auto_lags(n: int) -> int:
    """Newey-West 권장: floor(4 * (n/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _newey_west_var(d: np.ndarray, lags: int) -> float:
    """HAC variance of mean(d_t). Bartlett kernel."""
    n = len(d)
    d_centered = d - d.mean()
    gamma0 = float(np.dot(d_centered, d_centered) / n)
    s = gamma0
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(np.dot(d_centered[l:], d_centered[:-l]) / n)
        s += 2.0 * w * cov
    return max(s / n, 1e-18)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lags: int | None = None) -> DMResult:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    d = a - b
    n = len(d)
    if n < 5:
        return DMResult(
            n=n,
            mean_diff=float("nan"),
            dm_stat=float("nan"),
            hac_lags=0,
            p_value_two_sided=float("nan"),
            p_value_a_better=float("nan"),
            p_value_b_better=float("nan"),
        )
    if lags is None:
        lags = _auto_lags(n)
    var_dbar = _newey_west_var(d, lags)
    sd_dbar = float(np.sqrt(var_dbar))
    mean_d = float(d.mean())
    dm_stat = mean_d / sd_dbar
    p_two = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))
    p_a = float(norm.cdf(dm_stat))      # H1: mean < 0 (A 더 좋음)
    p_b = float(1.0 - norm.cdf(dm_stat))  # H1: mean > 0
    return DMResult(
        n=n,
        mean_diff=mean_d,
        dm_stat=float(dm_stat),
        hac_lags=int(lags),
        p_value_two_sided=p_two,
        p_value_a_better=p_a,
        p_value_b_better=p_b,
    )
