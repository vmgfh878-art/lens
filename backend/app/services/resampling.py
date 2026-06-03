"""
Lens 타임프레임 정규화 / 리샘플링.

CP225 분리 산출. 1D/1W/1M 타임프레임 정규화, 가격/컨텍스트 리샘플,
완성된 버킷 컷오프 계산.
의존: feature_definition (상수), validators (OHLC 계약 적용·검증).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from backend.app.services.feature_definition import SUPPORTED_TIMEFRAMES
from backend.app.services.validators import (
    _apply_adjusted_ohlc_contract,
    _validate_adjusted_ohlc_contract,
)


def normalize_timeframe(timeframe: str) -> str:
    normalized = timeframe.strip().upper()
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return normalized


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    sort_columns = ["ticker", "date"] if "ticker" in frame.columns else ["date"]
    return frame.sort_values(sort_columns).reset_index(drop=True)


def latest_complete_period_end(latest_daily_date: object, timeframe: str) -> pd.Timestamp | None:
    """최신 일봉 기준으로 완성된 1W/1M 버킷의 마지막 날짜를 계산한다."""
    timeframe = normalize_timeframe(timeframe)
    latest = pd.to_datetime(latest_daily_date, errors="coerce")
    if pd.isna(latest):
        return None
    latest = latest.normalize()
    if timeframe == "1D":
        return latest
    if timeframe == "1W":
        period_end = latest.to_period("W-FRI").end_time.normalize()
        return period_end if period_end <= latest else period_end - pd.Timedelta(days=7)
    period_end = latest.to_period("M").end_time.normalize()
    return period_end if period_end <= latest else (latest.to_period("M") - 1).end_time.normalize()


def drop_incomplete_resampled_periods(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    latest_daily_date: object | None = None,
) -> pd.DataFrame:
    """1W/1M 리샘플 결과에서 아직 끝나지 않은 현재 주/월 버킷을 제거한다."""
    timeframe = normalize_timeframe(timeframe)
    if timeframe == "1D" or frame.empty or "date" not in frame.columns:
        return frame.copy()
    cutoff = latest_complete_period_end(
        latest_daily_date if latest_daily_date is not None else frame["date"].max(),
        timeframe,
    )
    if cutoff is None:
        return frame.iloc[0:0].copy()
    filtered = frame.copy()
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    return filtered[filtered["date"] <= cutoff].copy()


def _resample_single_ticker(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    frame = _apply_adjusted_ohlc_contract(
        _ensure_datetime(df),
        context=f"resample_price_frame:{timeframe}",
    )
    if timeframe == "1D":
        return frame

    frame = frame.set_index("date")
    rule = "W-FRI" if timeframe == "1W" else "ME"
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for column in ("adjusted_close", "amount", "per", "pbr"):
        if column in frame.columns:
            agg_map[column] = "last" if column in ("adjusted_close", "per", "pbr") else "sum"

    aggregated = frame.resample(rule).agg(agg_map)
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()
    aggregated = drop_incomplete_resampled_periods(
        aggregated, timeframe, latest_daily_date=frame.index.max()
    )
    if "ticker" in df.columns and not aggregated.empty:
        aggregated["ticker"] = df["ticker"].iloc[0]
    if not aggregated.empty:
        _validate_adjusted_ohlc_contract(
            aggregated, context=f"resample_price_frame:{timeframe}:aggregated"
        )
    return aggregated


def resample_price_frame(price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    frame = _ensure_datetime(price_df)
    required_price_cols = ["open", "high", "low", "close"]
    frame = frame.dropna(subset=[col for col in required_price_cols if col in frame.columns])

    if timeframe == "1D" or frame.empty:
        return _apply_adjusted_ohlc_contract(
            frame, context=f"resample_price_frame:{timeframe}"
        ).reset_index(drop=True)

    if "ticker" not in frame.columns:
        return _resample_single_ticker(frame, timeframe)

    chunks: list[pd.DataFrame] = []
    for _, ticker_frame in frame.groupby("ticker", sort=True):
        resampled = _resample_single_ticker(ticker_frame, timeframe)
        if not resampled.empty:
            chunks.append(resampled)
    if not chunks:
        return frame.iloc[0:0].copy()
    return (
        pd.concat(chunks, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    )


def _resample_context_frame(
    df: pd.DataFrame | None, timeframe: str, columns: Iterable[str]
) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", *columns])

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    selected_columns = ["date", *[column for column in columns if column in frame.columns]]
    frame = frame.sort_values("date")[selected_columns]

    if timeframe == "1D":
        return frame

    rule = "W-FRI" if timeframe == "1W" else "ME"
    return frame.set_index("date").resample(rule).last().reset_index()


__all__ = [
    "normalize_timeframe",
    "_ensure_datetime",
    "latest_complete_period_end",
    "drop_incomplete_resampled_periods",
    "_resample_single_ticker",
    "resample_price_frame",
    "_resample_context_frame",
]
