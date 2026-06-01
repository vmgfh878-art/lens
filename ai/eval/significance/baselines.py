"""CP216 — 베이스라인 prediction 생성 / 정렬.

라인 (운영 CP210 대비):
  - naive_zero        : score=0
  - historical_mean   : 각 ticker 의 학습 윈도우 log_return 평균 → 모든 미래 예측에 적용
  - cp175_beta5       : 운영 frozen backup parquet 그대로 (검정 대상은 교집합 panel)

밴드 (운영 CP153/CP178 대비):
  - bollinger          : 20일 이동평균 ± 2σ → quantile 가격 → log_return space
  - historical_quantile: 학습 윈도우 200일 log_return 분위수 q10/q90
  - garch_p_q_1_1      : per-ticker GARCH(1,1), 정규 가정으로 quantile forecast

운영 panel 의 (ticker, asof_date[, horizon_step]) 를 anchor 로 jonin.
결과 컬럼은 운영 parquet 와 동일 형태로 맞춰서 비교 join 단순화.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

LOG = logging.getLogger(__name__)


# ----------------------------- 공통 유틸 -----------------------------


def load_ops_line() -> pd.DataFrame:
    df = pd.read_parquet(config.OPS_LINE_1D)
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date.astype(str)
    return df


def load_ops_band(timeframe: str) -> pd.DataFrame:
    path = config.OPS_BAND_1D if timeframe == "1D" else config.OPS_BAND_1W
    df = pd.read_parquet(path)
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date.astype(str)
    return df


def load_prices_1d() -> pd.DataFrame:
    df = pd.read_parquet(config.PRICES_1D)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def load_prices_1w() -> pd.DataFrame:
    df = pd.read_parquet(config.PRICES_1W)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def load_indicators_1d() -> pd.DataFrame:
    df = pd.read_parquet(config.INDICATORS_1D)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def _ticker_log_return_series(price_df: pd.DataFrame, ticker: str) -> pd.Series:
    sub = price_df[price_df["ticker"] == ticker].sort_values("date")
    closes = sub["close"].astype(float).values
    if len(closes) < 2:
        return pd.Series(dtype=float)
    log_r = np.diff(np.log(np.clip(closes, 1e-9, None)))
    return pd.Series(log_r, index=sub["date"].values[1:])


# ----------------------------- 라인 베이스라인 -----------------------------


def build_line_naive_zero(ops_line: pd.DataFrame) -> pd.DataFrame:
    """모든 (ticker, asof_date) 에 score=0."""
    out = ops_line[["ticker", "asof_date", "actual_h5_return"]].copy()
    out["line_score"] = 0.0
    out["safe_line_score"] = 0.0
    out["model_id"] = "naive_zero"
    out["source_cp"] = "CP216_BASELINE"
    return out


def build_line_historical_mean(
    ops_line: pd.DataFrame,
    price_df: pd.DataFrame,
    window: int = config.HISTORICAL_WINDOW,
) -> pd.DataFrame:
    """각 (ticker, asof_date) 의 학습 윈도우 log_return 평균.

    asof_date 직전 `window` 거래일의 평균을 점추정 score 로 사용 (5일 누적 X — line metric 은
    daily IC 이므로 1일 단위 평균이면 충분).
    """
    out_rows = []
    grouped = price_df.groupby("ticker", sort=False)
    price_by_ticker: Dict[str, pd.DataFrame] = {t: g.sort_values("date") for t, g in grouped}

    for ticker, sub in ops_line.groupby("ticker", sort=False):
        if ticker not in price_by_ticker:
            continue
        price = price_by_ticker[ticker]
        closes = price["close"].astype(float).values
        dates = price["date"].values
        log_r = np.empty_like(closes, dtype=float)
        log_r[:] = np.nan
        if len(closes) >= 2:
            log_r[1:] = np.diff(np.log(np.clip(closes, 1e-9, None)))
        rolling_mean = pd.Series(log_r, index=dates).rolling(window, min_periods=window // 2).mean()
        sub_sorted = sub.sort_values("asof_date").copy()
        mapped = rolling_mean.reindex(sub_sorted["asof_date"].values).values
        sub_sorted["line_score"] = mapped
        sub_sorted["safe_line_score"] = mapped
        sub_sorted["model_id"] = "historical_mean"
        sub_sorted["source_cp"] = "CP216_BASELINE"
        out_rows.append(sub_sorted[["ticker", "asof_date", "actual_h5_return",
                                     "line_score", "safe_line_score", "model_id", "source_cp"]])
    return pd.concat(out_rows, axis=0, ignore_index=True) if out_rows else pd.DataFrame()


def load_line_cp175_beta5() -> pd.DataFrame:
    """frozen backup parquet 그대로."""
    df = pd.read_parquet(config.CP175_FROZEN)
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date.astype(str)
    return df


# ----------------------------- 밴드 베이스라인 -----------------------------


def _band_anchor(ops_band: pd.DataFrame) -> pd.DataFrame:
    """(ticker, asof_date, horizon_step) anchor 만 추출."""
    cols = ["ticker", "asof_date", "horizon_step", "actual_return"]
    return ops_band[cols].drop_duplicates().reset_index(drop=True)


def convert_band_price_to_return(ops_band: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """1D ops 밴드 (CP153) 의 band_lower/upper 가 가격 단위로 저장되어 있음.

    baseline 들은 수익률 단위 → 통일 필요. asof_close 기준으로
    `band_return = band_price / asof_close - 1` 환산.

    median absolute 값 > 1 이면 가격으로 판단해서 환산. 그 외 (이미 수익률) 통과.
    """
    if ops_band.empty:
        return ops_band
    median_abs = float(ops_band[["band_lower", "band_upper"]].abs().median().median())
    if median_abs < 1.0:
        return ops_band  # 이미 수익률 단위
    price = (
        price_df[["ticker", "date", "close"]]
        .rename(columns={"date": "asof_date", "close": "asof_close"})
        .drop_duplicates(subset=["ticker", "asof_date"])
    )
    merged = ops_band.merge(price, on=["ticker", "asof_date"], how="left")
    valid = merged["asof_close"].astype(float) > 0
    merged.loc[valid, "band_lower"] = (
        merged.loc[valid, "band_lower"].astype(float) / merged.loc[valid, "asof_close"].astype(float) - 1.0
    )
    merged.loc[valid, "band_upper"] = (
        merged.loc[valid, "band_upper"].astype(float) / merged.loc[valid, "asof_close"].astype(float) - 1.0
    )
    merged.loc[~valid, ["band_lower", "band_upper"]] = float("nan")
    return merged.drop(columns=["asof_close"])


def build_band_bollinger(
    ops_band: pd.DataFrame,
    price_df: pd.DataFrame,
    timeframe: str,
    window: int = config.BOLLINGER_WINDOW,
    n_sigma: float = config.BOLLINGER_SIGMA,
) -> pd.DataFrame:
    """이동평균 ± n_sigma · σ 를 가격 밴드로 잡고, asof_close 대비 log_return space 로 변환.

    band_lower/upper 는 모든 horizon 동일하게 (스칼라) 추정 — Bollinger 는 본질적으로
    단일 점추정. horizon 별 차별화는 의도된 한계.
    """
    anchor = _band_anchor(ops_band)
    rows = []
    grouped = price_df.groupby("ticker", sort=False)
    price_by_ticker: Dict[str, pd.DataFrame] = {t: g.sort_values("date") for t, g in grouped}

    for ticker, sub in anchor.groupby("ticker", sort=False):
        if ticker not in price_by_ticker:
            continue
        price = price_by_ticker[ticker]
        closes = price["close"].astype(float)
        dates = price["date"].values
        log_close = np.log(np.clip(closes.values, 1e-9, None))
        log_r = np.empty_like(log_close)
        log_r[:] = np.nan
        log_r[1:] = np.diff(log_close)
        # log return 의 rolling std 와 평균 (정상화에 더 가까움)
        ma = pd.Series(log_r, index=dates).rolling(window, min_periods=window // 2).mean()
        sd = pd.Series(log_r, index=dates).rolling(window, min_periods=window // 2).std()
        ma_arr = ma.reindex(sub["asof_date"].values).values
        sd_arr = sd.reindex(sub["asof_date"].values).values
        # 1일 표준편차를 horizon h 거래일에 √h 로 스케일
        h = sub["horizon_step"].astype(int).values
        scale = np.sqrt(np.where(h > 0, h, 1))
        lower = ma_arr * h - n_sigma * sd_arr * scale
        upper = ma_arr * h + n_sigma * sd_arr * scale
        out = sub.copy()
        out["band_lower"] = lower
        out["band_upper"] = upper
        out["model_id"] = f"bollinger_w{window}_s{n_sigma}"
        out["source_cp"] = "CP216_BASELINE"
        rows.append(out)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def build_band_historical_quantile(
    ops_band: pd.DataFrame,
    price_df: pd.DataFrame,
    timeframe: str,
    window: int = config.HISTORICAL_WINDOW,
    q_low: float = config.Q_LOW,
    q_high: float = config.Q_HIGH,
) -> pd.DataFrame:
    """학습 윈도우의 단일 일별(or 주별) log_return 분위수 → h 거래일 누적 분위수.

    단순화: 윈도우 분위수 q_low/q_high 를 그대로 점추정으로 사용하고
    horizon 누적은 √h 스케일링.
    """
    anchor = _band_anchor(ops_band)
    rows = []
    grouped = price_df.groupby("ticker", sort=False)
    price_by_ticker: Dict[str, pd.DataFrame] = {t: g.sort_values("date") for t, g in grouped}

    for ticker, sub in anchor.groupby("ticker", sort=False):
        if ticker not in price_by_ticker:
            continue
        price = price_by_ticker[ticker]
        closes = price["close"].astype(float).values
        dates = price["date"].values
        log_r = np.empty_like(closes, dtype=float)
        log_r[:] = np.nan
        if len(closes) >= 2:
            log_r[1:] = np.diff(np.log(np.clip(closes, 1e-9, None)))
        s = pd.Series(log_r, index=dates)
        q_lo = s.rolling(window, min_periods=window // 2).quantile(q_low)
        q_hi = s.rolling(window, min_periods=window // 2).quantile(q_high)
        ma = s.rolling(window, min_periods=window // 2).mean()
        lo_arr = q_lo.reindex(sub["asof_date"].values).values
        hi_arr = q_hi.reindex(sub["asof_date"].values).values
        ma_arr = ma.reindex(sub["asof_date"].values).values
        h = sub["horizon_step"].astype(int).values
        scale = np.sqrt(np.where(h > 0, h, 1))
        # (qx - mean) 을 표준편차 단위 deviation 으로 가정 → √h 스케일
        lower = ma_arr * h + (lo_arr - ma_arr) * scale
        upper = ma_arr * h + (hi_arr - ma_arr) * scale
        out = sub.copy()
        out["band_lower"] = lower
        out["band_upper"] = upper
        out["model_id"] = f"historical_quantile_w{window}_q{int(q_low*100)}_q{int(q_high*100)}"
        out["source_cp"] = "CP216_BASELINE"
        rows.append(out)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------- GARCH(1,1) -----------------------------


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    sigma2_last: float


def _fit_garch_per_ticker(log_returns: np.ndarray) -> Optional[GarchFit]:
    """`arch` 의 ConstantMean + GARCH(1,1) + Normal innovations.

    실패 시 None. 호출자가 NaN 처리.
    """
    if len(log_returns) < 60 or not np.isfinite(log_returns).any():
        return None
    try:
        from arch import arch_model

        # arch 는 percentage 단위 권장 (수치 안정). log_return * 100 으로 fit.
        scale = 100.0
        am = arch_model(
            log_returns * scale,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)
        params = res.params
        cv = np.asarray(res.conditional_volatility, dtype=float)
        if len(cv) == 0 or not np.isfinite(cv[-1]):
            return None
        sigma2_last = float(cv[-1] ** 2)
        mu = float(params.get("mu", 0.0)) / scale
        omega = float(params.get("omega", 0.0)) / (scale ** 2)
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        sigma2_last = sigma2_last / (scale ** 2)
        return GarchFit(mu=mu, omega=omega, alpha=alpha, beta=beta, sigma2_last=sigma2_last)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("GARCH fit failed: %s", exc)
        return None


def _garch_forecast_quantiles(
    fit: GarchFit,
    horizon: int,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    """h-step ahead 누적 수익률의 정규 분위수 forecast.

    GARCH(1,1) 정상상태 분산을 사용 (h>1 에서 수렴 가정).
    누적 분산 ≈ Σ_{i=1..h} σ_i^2. 간단화: σ^2_t+i ≈ σ_uncond.
    """
    persistence = fit.alpha + fit.beta
    if persistence < 1.0:
        sigma2_uncond = fit.omega / (1.0 - persistence)
    else:
        sigma2_uncond = fit.sigma2_last  # IGARCH 비슷한 경우 last 사용
    # 1-step: σ_1^2 = ω + α·ε_t^2 + β·σ_t^2 ≈ ω + (α+β)·σ_t^2
    sigma2_steps = []
    s2 = fit.sigma2_last
    for _ in range(horizon):
        s2 = fit.omega + persistence * s2
        sigma2_steps.append(s2)
    cum_var = float(np.sum(sigma2_steps))
    cum_mu = fit.mu * horizon
    cum_sd = np.sqrt(max(cum_var, 1e-18))
    lower = cum_mu + norm.ppf(q_low) * cum_sd
    upper = cum_mu + norm.ppf(q_high) * cum_sd
    return lower, upper


def build_band_garch(
    ops_band: pd.DataFrame,
    price_df: pd.DataFrame,
    timeframe: str,
    q_low: float = config.Q_LOW,
    q_high: float = config.Q_HIGH,
    progress: bool = False,
) -> pd.DataFrame:
    """per-ticker GARCH(1,1) fit → asof 마다 forecast quantile.

    효율: ticker 마다 한 번 fit (전체 시계열). asof 별 conditional variance 는
    fit 결과를 walk-forward 로 다시 적용해야 정확하나, 비용 문제로 한 번 fit
    후 마지막 σ^2 기준 (정상상태) 으로 horizon forecast 사용.

    이는 베이스라인이므로 정확도보다 재현성 우선. 결과 limitation 은 보고서에 명시.
    """
    anchor = _band_anchor(ops_band)
    rows = []
    grouped = price_df.groupby("ticker", sort=False)
    price_by_ticker: Dict[str, pd.DataFrame] = {t: g.sort_values("date") for t, g in grouped}
    tickers = list(anchor["ticker"].unique())
    iterator = tickers
    if progress:
        from tqdm import tqdm
        iterator = tqdm(tickers, desc=f"GARCH {timeframe}")
    for ticker in iterator:
        if ticker not in price_by_ticker:
            continue
        price = price_by_ticker[ticker]
        closes = price["close"].astype(float).values
        dates = price["date"].values
        log_r = np.diff(np.log(np.clip(closes, 1e-9, None)))
        # asof 마다 walk-forward fit 은 비용 너무 큼 → 전체 시계열 fit 1회.
        fit = _fit_garch_per_ticker(log_r)
        sub = anchor[anchor["ticker"] == ticker].copy()
        if fit is None:
            sub["band_lower"] = np.nan
            sub["band_upper"] = np.nan
        else:
            h = sub["horizon_step"].astype(int).values
            lower = np.empty(len(sub), dtype=float)
            upper = np.empty(len(sub), dtype=float)
            for i, hi in enumerate(h):
                lo, hi_v = _garch_forecast_quantiles(fit, int(hi), q_low, q_high)
                lower[i] = lo
                upper[i] = hi_v
            sub["band_lower"] = lower
            sub["band_upper"] = upper
        sub["model_id"] = f"garch_1_1_normal_q{int(q_low*100)}_q{int(q_high*100)}"
        sub["source_cp"] = "CP216_BASELINE"
        rows.append(sub)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------- panel 정렬 -----------------------------


def intersect_line_panel(ops: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """(ticker, asof_date) 교집합 join — 같은 panel 에서 score / actual_h5_return 비교."""
    keys = ["ticker", "asof_date"]
    a = ops[keys + ["safe_line_score", "actual_h5_return"]].rename(
        columns={"safe_line_score": "score_a", "actual_h5_return": "actual_a"}
    )
    b = baseline[keys + ["safe_line_score", "actual_h5_return"]].rename(
        columns={"safe_line_score": "score_b", "actual_h5_return": "actual_b"}
    )
    merged = a.merge(b, on=keys, how="inner")
    # actual 은 운영/베이스라인 둘 다 같은 사실 — 운영 actual_a 를 사용
    merged = merged.rename(columns={"actual_a": "actual"})
    merged = merged.drop(columns=["actual_b"])
    return merged


def intersect_band_panel(ops: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    keys = ["ticker", "asof_date", "horizon_step"]
    a = ops[keys + ["band_lower", "band_upper", "actual_return"]].rename(
        columns={"band_lower": "lower_a", "band_upper": "upper_a", "actual_return": "actual_a"}
    )
    b = baseline[keys + ["band_lower", "band_upper", "actual_return"]].rename(
        columns={"band_lower": "lower_b", "band_upper": "upper_b", "actual_return": "actual_b"}
    )
    merged = a.merge(b, on=keys, how="inner")
    merged = merged.rename(columns={"actual_a": "actual"}).drop(columns=["actual_b"])
    return merged
