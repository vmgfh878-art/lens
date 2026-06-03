"""
전략 데이터 로딩 + 캐시 + 집계.

CP226 분리 산출. parquet I/O, lru_cache 보존, 티커별 시그널 + aggregate 산출.
의존: strategy_indicators (align_date_dtype / normalize_rsi / compute_signal_frame),
strategy_backtest_engine (ticker_metrics), strategy_rules (STRATEGIES/StrategyRule),
parquet_store.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.services import parquet_store
from app.services.strategy_backtest_engine import _ticker_metrics
from app.services.strategy_indicators import (
    _align_date_dtype,
    _compute_signal_frame,
    _normalize_rsi,
)
from app.strategies.strategy_rules import STRATEGIES
from fastapi import HTTPException

MIN_EVAL_DAYS = 120


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "v1"


@lru_cache(maxsize=1)
def _load_frame() -> pd.DataFrame:
    base = _data_dir()
    price = pd.read_parquet(base / "market_prices_1d.parquet")
    close_column = "adjusted_close" if "adjusted_close" in price.columns else "close"
    price = price[["ticker", "date", "open", "high", "low", close_column, "volume"]].rename(
        columns={close_column: "close"}
    )
    price["ticker"] = price["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price = price.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])
    price = price.drop_duplicates(["ticker", "date"], keep="last")
    price["daily_return"] = price.groupby("ticker")["close"].pct_change().fillna(0.0)
    price["previous_close"] = price.groupby("ticker")["close"].shift(1)
    true_range = pd.concat(
        [
            (price["high"] - price["low"]).abs(),
            (price["high"] - price["previous_close"]).abs(),
            (price["low"] - price["previous_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    price["atr_ratio_calc"] = (
        true_range.groupby(price["ticker"]).transform(
            lambda values: values.rolling(14, min_periods=5).mean()
        )
        / price["close"]
    )

    indicators = pd.read_parquet(base / "market_indicators_1d.parquet")
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"])
    indicators = indicators.sort_values(["ticker", "date"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    )
    for column in [
        "ma_5_ratio",
        "ma_20_ratio",
        "ma_60_ratio",
        "macd_ratio",
        "bb_position",
        "vol_change",
    ]:
        if column in indicators.columns:
            indicators[column] = pd.to_numeric(indicators[column], errors="coerce")
    indicators["rsi_norm"] = (
        _normalize_rsi(indicators["rsi"]) if "rsi" in indicators.columns else np.nan
    )

    # Use shared parquet_store to avoid loading a second copy of these files
    # (predictions.py already holds them; store ensures only one in-process copy).
    _raw_line = parquet_store.get_raw("line_1d")
    if _raw_line is None:
        raise FileNotFoundError("predictions_line_1d.parquet not found in parquet_store")
    line = _raw_line.copy()
    line["ticker"] = line["ticker"].astype(str).str.upper()
    line["date"] = pd.to_datetime(line["asof_date"])
    for column in ["line_score", "safe_line_score", "line_rank_by_date", "safe_line_rank_by_date"]:
        if column in line.columns:
            line[column] = pd.to_numeric(line[column], errors="coerce")
    line = line.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    _raw_band = parquet_store.get_raw("band_1d")
    if _raw_band is None:
        raise FileNotFoundError("predictions_band_1d.parquet not found in parquet_store")
    band = _raw_band[pd.to_numeric(_raw_band["horizon_step"], errors="coerce") == 5].copy()
    band["ticker"] = band["ticker"].astype(str).str.upper()
    band["date"] = pd.to_datetime(band["asof_date"])
    for column in ["band_lower", "band_upper"]:
        band[column] = pd.to_numeric(band[column], errors="coerce")
    band = band.groupby(["ticker", "date"], as_index=False, observed=True).agg(
        band_lower=("band_lower", "min"), band_upper=("band_upper", "max")
    )

    # CP214 — 머지 직전 date dtype 평탄화 (방어). 근본 fix 는 parquet_store 에서 했지만
    # 향후 다른 source 가 추가돼도 머지가 깨지지 않게 idempotent helper 적용.
    price = _align_date_dtype(price)
    indicators = _align_date_dtype(indicators)
    line = _align_date_dtype(line)
    band = _align_date_dtype(band)

    # CP227 — 머지 직전 dtype 계약 강제. _align_date_dtype 이후 date 는 datetime64 여야
    # 머지 키가 호환되며, 어긋나면 silent 머지 실패 (CP214 사고) 가 다시 들어온다.
    for _nm, _f in (("price", price), ("indicators", indicators), ("line", line), ("band", band)):
        assert pd.api.types.is_datetime64_any_dtype(
            _f["date"]
        ), f"{_nm}.date not datetime before merge"

    frame = price.merge(
        indicators[
            [
                "ticker",
                "date",
                "ma_5_ratio",
                "ma_20_ratio",
                "ma_60_ratio",
                "macd_ratio",
                "bb_position",
                "vol_change",
                "rsi_norm",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    )
    frame = frame.merge(
        line[
            [
                "ticker",
                "date",
                "line_score",
                "safe_line_score",
                "line_rank_by_date",
                "safe_line_rank_by_date",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    )
    frame = frame.merge(band, on=["ticker", "date"], how="left")

    frame["ma_20_ratio"] = frame["ma_20_ratio"].fillna(
        frame.groupby("ticker")["close"].transform(
            lambda values: values / values.rolling(20, min_periods=15).mean() - 1.0
        )
    )
    frame["ma_60_ratio"] = frame["ma_60_ratio"].fillna(
        frame.groupby("ticker")["close"].transform(
            lambda values: values / values.rolling(60, min_periods=40).mean() - 1.0
        )
    )
    frame["band_lower_return"] = frame["band_lower"] / frame["close"] - 1.0
    frame["band_upper_return"] = frame["band_upper"] / frame["close"] - 1.0
    frame["band_width_return"] = frame["band_upper_return"] - frame["band_lower_return"]
    width_reference = frame.groupby("ticker")["band_width_return"].transform(
        lambda values: values.rolling(60, min_periods=20).median().shift(1)
    )
    frame["band_width_expansion"] = (frame["band_width_return"] / width_reference).replace(
        [np.inf, -np.inf], np.nan
    )
    frame["band_width_expansion"] = frame["band_width_expansion"].fillna(1.0)
    frame["band_width_percentile"] = frame.groupby("ticker")["band_width_return"].rank(pct=True)
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _sector_map() -> dict[str, str]:
    try:
        stock_info = pd.read_parquet(_data_dir() / "market_stock_info.parquet")
    except Exception:
        return {}
    stock_info["ticker"] = stock_info["ticker"].astype(str).str.upper()
    return {
        str(row["ticker"]).upper(): str(row["sector"]) if pd.notna(row.get("sector")) else "Unknown"
        for _, row in stock_info.iterrows()
    }


@lru_cache(maxsize=16)
def _strategy_results(strategy_id: str) -> dict[str, Any]:
    if strategy_id not in STRATEGIES:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 전략입니다: {strategy_id}")
    rule = STRATEGIES[strategy_id]
    frame = _load_frame()
    if rule.uses_line:
        frame = frame[frame["line_score"].notna()].copy()
    if rule.uses_band:
        frame = frame[
            frame["band_lower_return"].notna() & frame["band_width_return"].notna()
        ].copy()
    if frame.empty:
        raise HTTPException(
            status_code=404, detail=f"{rule.label}에 사용할 로컬 데이터가 없습니다."
        )

    end_date = frame["date"].max()
    start_date = end_date - pd.Timedelta(days=365)
    frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()

    by_ticker: dict[str, pd.DataFrame] = {}
    metrics_rows = []
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        if len(ticker_frame) < MIN_EVAL_DAYS:
            continue
        signal_frame = _compute_signal_frame(ticker_frame, rule)
        by_ticker[str(ticker)] = signal_frame
        metrics = _ticker_metrics(signal_frame)
        metrics_rows.append({"ticker": str(ticker), **metrics})

    if not metrics_rows:
        raise HTTPException(
            status_code=404, detail=f"{rule.label}에 필요한 평가 가능 티커가 없습니다."
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    pass_mask = (
        (metrics_frame["strategyReturnPct"] >= metrics_frame["buyHoldReturnPct"])
        & (metrics_frame["maxDrawdownPct"] >= metrics_frame["buyHoldMaxDrawdownPct"])
        & (metrics_frame["marketParticipationRate"].between(0.2, 0.95))
    )
    aggregate = {
        "strategyReturnPct": float(metrics_frame["strategyReturnPct"].mean()),
        "buyHoldReturnPct": float(metrics_frame["buyHoldReturnPct"].mean()),
        "excessReturnPct": float(metrics_frame["excessReturnPct"].mean()),
        "maxDrawdownPct": float(metrics_frame["maxDrawdownPct"].mean()),
        "buyHoldMaxDrawdownPct": float(metrics_frame["buyHoldMaxDrawdownPct"].mean()),
        "maxDrawdownImprovementPct": float(metrics_frame["maxDrawdownImprovementPct"].mean()),
        "feeAdjustedSharpe": float(metrics_frame["feeAdjustedSharpe"].mean()),
        "buyHoldSharpe": float(metrics_frame["buyHoldSharpe"].mean()),
        "strategySortino": float(metrics_frame["strategySortino"].mean()),
        "buyHoldSortino": float(metrics_frame["buyHoldSortino"].mean()),
        "marketParticipationRate": float(metrics_frame["marketParticipationRate"].mean()),
        "avgSelectedCount": None,
        "avgTradeCount": float(metrics_frame["tradeCount"].mean()),
        "largeLossAvoidanceRate": float(metrics_frame["largeLossAvoidanceRate"].dropna().mean()),
        "passTickerRatio": float(pass_mask.mean()),
    }
    return {
        "rule": rule,
        "start_date": start_date,
        "end_date": end_date,
        "by_ticker": by_ticker,
        "metrics": metrics_frame,
        "aggregate": aggregate,
    }


__all__ = [
    "MIN_EVAL_DAYS",
    "_data_dir",
    "_load_frame",
    "_sector_map",
    "_strategy_results",
]
