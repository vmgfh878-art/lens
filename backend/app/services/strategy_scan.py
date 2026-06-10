"""
전략 데이터 로딩 + 캐시 + 집계.

CP226 분리 산출. CP244 메모리 재설계:
  - `_load_base_frame` : 가격 + 지표 (line/band 없이) — 모든 전략 공통 base.
  - `_load_frame(needs_line, needs_band)` : base + 전략이 필요로 하는 line/band 만 머지.
    band 를 안 쓰는 전략(indicator_balance_v2)이 band(97MB) 를 로드하지 않게 한다.
  - `_scan_results` : scan 용. 종목별 최신 신호 row 만 보관 (by_ticker full 75MB 미상주).
  - `_backtest_signal_frame` : backtest 용. 종목 1개 full signal_frame.

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
def _load_base_frame() -> pd.DataFrame:
    """가격 + 지표 (line/band 없이). 모든 전략 공통 base.

    band/line 은 _load_frame(needs_line, needs_band) 에서 전략이 실제로 필요로 할
    때만 머지한다 — band 를 안 쓰는 전략이 band parquet(97MB)을 로드하지 않게.
    """
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

    price = _align_date_dtype(price)
    indicators = _align_date_dtype(indicators)
    for _nm, _f in (("price", price), ("indicators", indicators)):
        assert pd.api.types.is_datetime64_any_dtype(
            _f["date"]
        ), f"{_nm}.date not datetime before merge"

    frame = price.merge(
        indicators[
            [
                "ticker",
                "date",
                "ma_5_ratio",  # CP253 — 발굴 momentum archetype 용
                "ma_20_ratio",
                "ma_60_ratio",
                "macd_ratio",
                "bb_position",
                "rsi_norm",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    )
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
    # CP245 — scan/backtest 에서 안 쓰는 원천 컬럼 drop (메모리). atr_ratio_calc 계산 후
    # OHLCV·previous_close 는 불필요. daily_return 은 _ticker_metrics/_points 가 쓰므로 유지.
    frame = frame.drop(
        columns=[
            c for c in ["open", "high", "low", "volume", "previous_close"] if c in frame.columns
        ]
    )
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    # CP253 — 발굴 전략(momentum) 파생. strategy_ablation_v2 와 동일(groupby ticker, 당일까지
    # shift = 미래 누수 0). 동일 프레임에서 _raw_target 이 v2_raw 와 byte-identical 재현.
    grp = frame.groupby("ticker")
    frame["roc_20"] = grp["close"].transform(lambda s: s / s.shift(20) - 1.0)
    frame["macd_accel"] = grp["macd_ratio"].transform(lambda s: s - s.shift(3))
    return frame


def _merge_line(frame: pd.DataFrame) -> pd.DataFrame:
    raw_line = parquet_store.get_raw("line_1d")
    if raw_line is None:
        raise FileNotFoundError("predictions_line_1d.parquet not found in parquet_store")
    line = raw_line.copy()
    line["ticker"] = line["ticker"].astype(str).str.upper()
    line["date"] = pd.to_datetime(line["asof_date"])
    for column in ["line_score", "safe_line_score", "line_rank_by_date", "safe_line_rank_by_date"]:
        if column in line.columns:
            line[column] = pd.to_numeric(line[column], errors="coerce")
    line = line.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    line = _align_date_dtype(line)
    assert pd.api.types.is_datetime64_any_dtype(line["date"]), "line.date not datetime before merge"
    # CP245 — scan/backtest 는 line_score 만 쓴다 (safe_line_score / rank 미사용 → 메모리 절감).
    return frame.merge(
        line[["ticker", "date", "line_score"]],
        on=["ticker", "date"],
        how="left",
    )


def _merge_band(frame: pd.DataFrame) -> pd.DataFrame:
    raw_band = parquet_store.get_raw("band_1d")
    if raw_band is None:
        raise FileNotFoundError("predictions_band_1d.parquet not found in parquet_store")
    band = raw_band[pd.to_numeric(raw_band["horizon_step"], errors="coerce") == 5].copy()
    band["ticker"] = band["ticker"].astype(str).str.upper()
    band["date"] = pd.to_datetime(band["asof_date"])
    for column in ["band_lower", "band_upper"]:
        band[column] = pd.to_numeric(band[column], errors="coerce")
    band = band.groupby(["ticker", "date"], as_index=False, observed=True).agg(
        band_lower=("band_lower", "min"), band_upper=("band_upper", "max")
    )
    band = _align_date_dtype(band)
    assert pd.api.types.is_datetime64_any_dtype(band["date"]), "band.date not datetime before merge"
    frame = frame.merge(band, on=["ticker", "date"], how="left")
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
    # CP245 — band_lower/upper/upper_return 은 return/width 파생 계산 후 불필요. drop (메모리).
    return frame.drop(columns=["band_lower", "band_upper", "band_upper_return"])


@lru_cache(maxsize=4)
def _load_frame(needs_line: bool = True, needs_band: bool = True) -> pd.DataFrame:
    """base + 전략이 필요로 하는 line/band 만 머지.

    하위호환: 인자 없이 호출하면 line/band 둘 다 머지(기존 _load_frame() 동작과 동일).
    band 를 안 쓰는 전략은 `_load_frame(False, False)` → base 그대로(band 미로드).
    """
    frame = _load_base_frame()
    if not needs_line and not needs_band:
        return frame
    frame = frame.copy()
    if needs_line:
        frame = _merge_line(frame)
    if needs_band:
        frame = _merge_band(frame)
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


def _filtered_windowed_frame(
    rule: Any,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """전략별 frame: 필요한 line/band 머지 + notna 필터 + 최근 365일 윈도우.

    scan/backtest 공통 전처리. _load_frame 이 lru_cache 라 반복 호출해도 디스크
    재로딩은 없다 (필터·윈도우만 매번).
    """
    frame = _load_frame(rule.uses_line, rule.uses_band)
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
    return frame, start_date, end_date


@lru_cache(maxsize=16)
def _scan_results(strategy_id: str) -> dict[str, Any]:
    """scan 용 — 종목별 최신 신호 row + metrics + aggregate.

    cards 는 종목별 최신 1 row 만 쓰므로, by_ticker full signal_frame 을 상주시키지
    않는다 (메모리 75MB → 수MB). full frame 이 필요한 backtest 는
    `_backtest_signal_frame` 으로 종목 1개만 별도 계산한다.
    """
    if strategy_id not in STRATEGIES:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 전략입니다: {strategy_id}")
    rule = STRATEGIES[strategy_id]
    frame, start_date, end_date = _filtered_windowed_frame(rule)

    latest_rows: dict[str, dict[str, Any]] = {}
    metrics_rows = []
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        if len(ticker_frame) < MIN_EVAL_DAYS:
            continue
        signal_frame = _compute_signal_frame(ticker_frame, rule)
        latest_rows[str(ticker)] = signal_frame.iloc[-1].to_dict()
        metrics_rows.append({"ticker": str(ticker), **_ticker_metrics(signal_frame)})

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
        "latest_rows": latest_rows,
        "metrics": metrics_frame,
        "aggregate": aggregate,
    }


def _backtest_signal_frame(strategy_id: str, ticker: str) -> tuple[Any, pd.DataFrame]:
    """backtest 용 — 종목 1개 full signal_frame (전종목 상주 없이)."""
    if strategy_id not in STRATEGIES:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 전략입니다: {strategy_id}")
    rule = STRATEGIES[strategy_id]
    frame, _start, _end = _filtered_windowed_frame(rule)
    normalized = ticker.upper()
    ticker_frame = frame[frame["ticker"] == normalized].copy()
    if len(ticker_frame) < MIN_EVAL_DAYS:
        raise HTTPException(
            status_code=404,
            detail=f"{normalized}에는 {rule.label} 백테스트에 필요한 로컬 데이터가 없습니다.",
        )
    return rule, _compute_signal_frame(ticker_frame, rule)


__all__ = [
    "MIN_EVAL_DAYS",
    "_data_dir",
    "_load_base_frame",
    "_load_frame",
    "_merge_line",
    "_merge_band",
    "_sector_map",
    "_filtered_windowed_frame",
    "_scan_results",
    "_backtest_signal_frame",
]
