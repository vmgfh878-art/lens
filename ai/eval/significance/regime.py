"""CP216 — regime 라벨 (VIX>30, drawdown ≤ -10%, OR 결합).

가격은 시장 전체 평균을 가정하면 정의가 모호 — 여기서는 SPY/IVV 같은 단일
프록시 ticker 를 쓰지 않고 모든 ticker 의 close 평균 log-cumulative 곡선을
시장 프록시로 사용. drawdown 도 그 곡선의 200d rolling max 대비.

단순화이지만 베이스라인 regime 라벨 정의로는 충분. 자세한 정의는 보고서.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_vix_label(indicators_1d: pd.DataFrame, threshold: float = config.VIX_THRESHOLD) -> pd.Series:
    """asof_date → bool (VIX > threshold)."""
    df = indicators_1d[["date", "vix_close"]].drop_duplicates(subset=["date"]).copy()
    df = df.sort_values("date")
    df["vix_label"] = (df["vix_close"].astype(float) > threshold).astype(int)
    return df.set_index("date")["vix_label"]


def build_market_drawdown(price_df: pd.DataFrame, window: int = config.DD_WINDOW) -> pd.Series:
    """모든 ticker close 의 일별 평균 → log 누적 → 200d rolling max 대비 drawdown.

    Returns: index=date, values=drawdown (음수)
    """
    daily = price_df.groupby("date")["close"].mean().sort_index()
    log_idx = np.log(np.clip(daily.values, 1e-9, None))
    cum = log_idx - log_idx[0]  # 0 기준 누적 log-return
    rolling_max = pd.Series(cum, index=daily.index).rolling(window, min_periods=window // 4).max()
    dd = pd.Series(cum, index=daily.index) - rolling_max
    return dd  # log space; -0.10 ≈ -10% drop


def build_drawdown_label(
    price_df: pd.DataFrame,
    window: int = config.DD_WINDOW,
    threshold: float = config.DD_THRESHOLD,
) -> pd.Series:
    dd = build_market_drawdown(price_df, window=window)
    return (dd <= threshold).astype(int)


def build_regime_labels(
    indicators_1d: pd.DataFrame,
    price_df: pd.DataFrame,
    vix_threshold: float = config.VIX_THRESHOLD,
    dd_window: int = config.DD_WINDOW,
    dd_threshold: float = config.DD_THRESHOLD,
) -> pd.DataFrame:
    """DataFrame[date, vix_high, drawdown_low, combined]"""
    vix = build_vix_label(indicators_1d, vix_threshold)
    dd = build_drawdown_label(price_df, dd_window, dd_threshold)
    common = vix.index.intersection(dd.index)
    out = pd.DataFrame(
        {
            "vix_high": vix.reindex(common).fillna(0).astype(int),
            "drawdown_low": dd.reindex(common).fillna(0).astype(int),
        }
    )
    out["combined"] = ((out["vix_high"] + out["drawdown_low"]) > 0).astype(int)
    out.index.name = "date"
    return out
