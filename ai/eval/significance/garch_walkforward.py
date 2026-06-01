"""CP216.2 — GARCH(1,1) walk-forward 베이스라인.

운영 모델 (CP153 / CP178) 의 stage5T 3-fold 와 1:1 일치.
fold 마다:
  1. train 구간 log_return 으로 GARCH(1,1) Normal innovations fit
  2. 각 asof_date ∈ val ∪ test 에 대해:
     - asof_date 까지 conditional variance σ²_t 추적 (실측 ε_t = log_r_t - μ 로 reweight)
     - h-step ahead cumulative variance: Σ_{i=1..H} σ²_{t+i}, 미래 σ² 는 ω + (α+β)·σ²_{t+i-1} 점화
     - quantile = μ·h ± z_q · √cum_var
  3. (ticker, asof_date, horizon, band_lower, band_upper) row 생성

운영 모델 이 fold 마다 1회 fresh fit + 그 후 forward only inference 와 동치.
in-sample leakage 없음.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .folds import CP153_FOLDS, CP178_FOLDS, Fold

LOG = logging.getLogger(__name__)


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float


def _fit_garch(log_returns: np.ndarray) -> Optional[GarchParams]:
    if len(log_returns) < 100 or not np.isfinite(log_returns).all():
        return None
    try:
        from arch import arch_model

        scale = 100.0
        am = arch_model(
            log_returns * scale,
            mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)
        p = res.params
        return GarchParams(
            mu=float(p.get("mu", 0.0)) / scale,
            omega=float(p.get("omega", 0.0)) / (scale ** 2),
            alpha=float(p.get("alpha[1]", 0.0)),
            beta=float(p.get("beta[1]", 0.0)),
        )
    except Exception as exc:
        LOG.debug("GARCH fit failed: %s", exc)
        return None


def _walkforward_conditional_variance(
    params: GarchParams,
    log_returns: np.ndarray,    # train end 직후 ~ eval end 까지 (train 이후 영역)
    sigma2_start: float,
) -> np.ndarray:
    """각 timestep t 의 conditional variance σ²_t (= forecast variance for t).

    σ²_{t+1} = ω + α·ε_t² + β·σ²_t,  ε_t = log_r_t - μ
    sigma2_start = train 마지막 시점의 σ²_train_end
    """
    n = len(log_returns)
    sigma2 = np.empty(n + 1, dtype=float)
    sigma2[0] = sigma2_start
    eps = log_returns - params.mu
    for t in range(n):
        sigma2[t + 1] = params.omega + params.alpha * (eps[t] ** 2) + params.beta * sigma2[t]
    return sigma2  # length n+1


def _train_terminal_sigma2(params: GarchParams, train_log_returns: np.ndarray) -> float:
    """train 끝에서의 conditional variance."""
    if len(train_log_returns) == 0:
        return params.omega / max(1.0 - params.alpha - params.beta, 1e-12)
    eps = train_log_returns - params.mu
    sigma2 = params.omega / max(1.0 - params.alpha - params.beta, 1e-6)
    for t in range(len(train_log_returns)):
        sigma2 = params.omega + params.alpha * (eps[t] ** 2) + params.beta * sigma2
    return float(sigma2)


def _h_step_cumulative_var(params: GarchParams, sigma2_t: float, horizon: int) -> float:
    """t 시점 이후 horizon h 거래일 동안 누적 분산.

    σ²_{t+1} = ω + α·ε_t² + β·σ²_t  →  미래는 ε 미관측이므로 E[ε²] = σ² 가정.
    그러면 σ²_{t+1} = ω + (α+β)·σ²_t 점화.
    누적 var = Σ_{i=1..h} σ²_{t+i}.
    """
    persistence = params.alpha + params.beta
    cum_var = 0.0
    s2 = sigma2_t
    for _ in range(horizon):
        s2 = params.omega + persistence * s2
        cum_var += s2
    return cum_var


def _build_per_ticker(
    ticker: str,
    price_sub: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    folds: list[Fold],
    horizons_per_asof: dict,   # asof_date → list[int]
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    """ticker 한 개 의 walk-forward GARCH 예측 생성."""
    price_sub = price_sub.sort_values("date").reset_index(drop=True)
    closes = price_sub["close"].astype(float).values
    dates = price_sub["date"].values  # str dates
    if len(closes) < 200:
        return pd.DataFrame()
    log_r_full = np.diff(np.log(np.clip(closes, 1e-9, None)))
    log_r_dates = dates[1:]  # log_r[i] 는 date[i+1] 거래일 수익률
    log_r_series = pd.Series(log_r_full, index=log_r_dates)

    rows = []
    for fold in folds:
        train_mask = (log_r_series.index >= fold.train_start) & (log_r_series.index < fold.train_end)
        eval_mask = (log_r_series.index >= fold.val_start) & (log_r_series.index < fold.test_end)
        train_lr = log_r_series[train_mask].values
        eval_lr = log_r_series[eval_mask].values
        eval_dates = log_r_series.index[eval_mask]
        if len(train_lr) < 100 or len(eval_lr) == 0:
            continue
        params = _fit_garch(train_lr)
        if params is None:
            continue
        sigma2_train_end = _train_terminal_sigma2(params, train_lr)
        # eval 영역 σ² walk-forward (실측 ε)
        sigma2_eval = _walkforward_conditional_variance(params, eval_lr, sigma2_train_end)
        # sigma2_eval[i] = σ² at end of eval_lr[i-1] (date eval_dates[i-1])
        # asof_date = eval_dates[i-1] 시점에서 σ²_t = sigma2_eval[i] 사용 (next-step 예측에 들어가는 conditional var).
        # 각 asof_date 에 대해 horizon 별 cumulative var 계산
        for i, asof in enumerate(eval_dates):
            asof_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
            if asof_str not in horizons_per_asof:
                continue
            sigma2_t = float(sigma2_eval[i + 1])  # 1-step ahead 시작값
            for h in horizons_per_asof[asof_str]:
                cum_var = _h_step_cumulative_var(params, sigma2_t, int(h))
                cum_sd = float(np.sqrt(max(cum_var, 1e-18)))
                mu_h = params.mu * float(h)
                lower = mu_h + norm.ppf(q_low) * cum_sd
                upper = mu_h + norm.ppf(q_high) * cum_sd
                rows.append((asof_str, int(h), lower, upper, fold.fold_id))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, columns=["asof_date", "horizon_step", "band_lower", "band_upper", "fold_id"])
    out.insert(0, "ticker", ticker)
    # actual_return 조인
    keys = ["ticker", "asof_date", "horizon_step"]
    out = out.merge(anchor_sub[keys + ["actual_return"]], on=keys, how="left")
    out["model_id"] = f"garch_1_1_normal_wf_q{int(q_low*100)}_q{int(q_high*100)}"
    out["source_cp"] = "CP216_2_BASELINE_WF"
    return out


def build_band_garch_walkforward(
    ops_band: pd.DataFrame,
    price_df: pd.DataFrame,
    timeframe: str,
    q_low: float,
    q_high: float,
    folds: Optional[list[Fold]] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """운영 panel (ticker, asof_date, horizon_step) 에 정렬된 GARCH walk-forward forecast."""
    if folds is None:
        folds = CP153_FOLDS if timeframe == "1D" else CP178_FOLDS
    if ops_band.empty:
        return ops_band
    # asof_date string 정규화
    anchor = ops_band[["ticker", "asof_date", "horizon_step", "actual_return"]].drop_duplicates().reset_index(drop=True)
    # ticker 별 anchor 의 (asof_date → horizons) 맵
    horizons_per_asof_by_ticker: dict[str, dict[str, list[int]]] = {}
    for ticker, sub in anchor.groupby("ticker", sort=False):
        by_asof = sub.groupby("asof_date")["horizon_step"].apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
        horizons_per_asof_by_ticker[ticker] = by_asof
    # price dict
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
    price_by_ticker = {t: g for t, g in price_df.groupby("ticker", sort=False)}

    tickers = sorted(anchor["ticker"].unique())
    iterator = tickers
    if progress:
        from tqdm import tqdm
        iterator = tqdm(tickers, desc=f"GARCH-WF {timeframe}")
    out_chunks = []
    for ticker in iterator:
        if ticker not in price_by_ticker or ticker not in horizons_per_asof_by_ticker:
            continue
        rows = _build_per_ticker(
            ticker=ticker,
            price_sub=price_by_ticker[ticker],
            anchor_sub=anchor[anchor["ticker"] == ticker],
            folds=folds,
            horizons_per_asof=horizons_per_asof_by_ticker[ticker],
            q_low=q_low,
            q_high=q_high,
        )
        if not rows.empty:
            out_chunks.append(rows)
    if not out_chunks:
        return pd.DataFrame()
    return pd.concat(out_chunks, axis=0, ignore_index=True)
