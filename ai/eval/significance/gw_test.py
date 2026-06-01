"""CP216 — Giacomini-White conditional predictive ability test.

z_t = L_A_t - L_B_t (= DM 의 d_t).
test info set h_t = (1, regime_t).
회귀: z_t = α + β · regime_t + ε_t.
Wald 통계 (HAC): H_0 : α = β = 0  →  chi2(2).
보조: β 단독 t-stat (regime 효과 conditional 유의성).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import chi2, norm

from .dm_test import _auto_lags


@dataclass
class GWResult:
    n: int
    n_regime_on: int
    alpha: float
    beta: float
    wald_stat: float
    wald_p_value: float
    beta_t_stat: float
    beta_p_value_two_sided: float
    mean_d_regime_on: float
    mean_d_regime_off: float
    hac_lags: int


def _newey_west_cov(residual_x: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance estimator for S = E[h_t h_t' ε_t^2] with serial correlation."""
    n = residual_x.shape[0]
    S = residual_x.T @ residual_x / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = residual_x[l:].T @ residual_x[:-l] / n
        S += w * (cov + cov.T)
    return S


def gw_test(loss_a: np.ndarray, loss_b: np.ndarray, regime: np.ndarray, lags: int | None = None) -> GWResult:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    h = np.asarray(regime, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(h)
    a = a[mask]
    b = b[mask]
    h = h[mask]
    z = a - b
    n = len(z)
    if n < 20 or h.sum() < 3 or (n - h.sum()) < 3:
        return GWResult(
            n=n,
            n_regime_on=int(h.sum()),
            alpha=float("nan"),
            beta=float("nan"),
            wald_stat=float("nan"),
            wald_p_value=float("nan"),
            beta_t_stat=float("nan"),
            beta_p_value_two_sided=float("nan"),
            mean_d_regime_on=float("nan"),
            mean_d_regime_off=float("nan"),
            hac_lags=0,
        )
    if lags is None:
        lags = _auto_lags(n)
    X = np.column_stack([np.ones(n), h])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ z
    resid = z - X @ beta_hat
    # HAC: V_beta = (X'X)^-1 · X'·Ω·X · (X'X)^-1, with Newey-West on residual_x = (h_t * resid_t)
    rx = X * resid[:, None]
    S = _newey_west_cov(rx, lags)
    V_beta = XtX_inv @ (S * n) @ XtX_inv  # scale back
    V_beta = V_beta / n
    # Wald for joint H_0: beta = 0
    try:
        V_inv = np.linalg.inv(V_beta)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V_beta)
    wald = float(beta_hat @ V_inv @ beta_hat)
    p_wald = float(1.0 - chi2.cdf(wald, df=2))
    se_beta1 = float(np.sqrt(max(V_beta[1, 1], 1e-18)))
    t_beta1 = float(beta_hat[1] / se_beta1)
    p_beta1 = float(2.0 * (1.0 - norm.cdf(abs(t_beta1))))
    mean_on = float(z[h > 0.5].mean()) if (h > 0.5).any() else float("nan")
    mean_off = float(z[h < 0.5].mean()) if (h < 0.5).any() else float("nan")
    return GWResult(
        n=n,
        n_regime_on=int(h.sum()),
        alpha=float(beta_hat[0]),
        beta=float(beta_hat[1]),
        wald_stat=wald,
        wald_p_value=p_wald,
        beta_t_stat=t_beta1,
        beta_p_value_two_sided=p_beta1,
        mean_d_regime_on=mean_on,
        mean_d_regime_off=mean_off,
        hac_lags=int(lags),
    )
