"""
Lens 피처 생성 서비스.

이 파일을 v1 기준 17개 피처 정의의 단일 기준점으로 사용한다.
또한 1D / 1W / 1M 타임프레임별 리샘플링과 피처 생성을 함께 담당한다.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "vol_change",
    "ma_5_ratio",
    "ma_20_ratio",
    "ma_60_ratio",
    "rsi",
    "macd_ratio",
    "bb_position",
    "us10y",
    "yield_spread",
    "vix_close",
    "credit_spread_hy",
    "nh_nl_index",
    "ma200_pct",
]

SUPPORTED_TIMEFRAMES = ("1D", "1W", "1M")
_EPSILON = 1e-9
_CONTEXT_COLUMNS = (
    "us10y",
    "yield_spread",
    "vix_close",
    "credit_spread_hy",
    "nh_nl_index",
    "ma200_pct",
)


def normalize_timeframe(timeframe: str) -> str:
    normalized = timeframe.strip().upper()
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return normalized


def default_horizon_for_timeframe(timeframe: str) -> int:
    normalized = normalize_timeframe(timeframe)
    return {"1D": 10, "1W": 12, "1M": 6}[normalized]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    sort_columns = ["ticker", "date"] if "ticker" in frame.columns else ["date"]
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _resample_single_ticker(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    frame = _ensure_datetime(df)
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
    if "ticker" in df.columns and not aggregated.empty:
        aggregated["ticker"] = df["ticker"].iloc[0]
    return aggregated


def resample_price_frame(price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    frame = _ensure_datetime(price_df)
    required_price_cols = ["open", "high", "low", "close"]
    frame = frame.dropna(subset=[col for col in required_price_cols if col in frame.columns])

    if timeframe == "1D" or frame.empty:
        return frame.reset_index(drop=True)

    if "ticker" not in frame.columns:
        return _resample_single_ticker(frame, timeframe)

    chunks: list[pd.DataFrame] = []
    for _, ticker_frame in frame.groupby("ticker", sort=True):
        resampled = _resample_single_ticker(ticker_frame, timeframe)
        if not resampled.empty:
            chunks.append(resampled)
    if not chunks:
        return frame.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def _resample_context_frame(df: pd.DataFrame | None, timeframe: str, columns: Iterable[str]) -> pd.DataFrame:
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


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(window=period).mean()
    losses = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gains / (losses + _EPSILON)
    return 100 - (100 / (1 + rs))


def _compute_features_for_single_ticker(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().sort_values("date").reset_index(drop=True)
    close_base = frame["adjusted_close"].fillna(frame["close"]) if "adjusted_close" in frame.columns else frame["close"]
    frame["close"] = close_base

    previous_close = frame["close"].shift(1)
    frame["log_return"] = np.log(frame["close"] / previous_close)
    frame["open_ratio"] = (frame["open"] - previous_close) / (previous_close + _EPSILON)
    frame["high_ratio"] = (frame["high"] - previous_close) / (previous_close + _EPSILON)
    frame["low_ratio"] = (frame["low"] - previous_close) / (previous_close + _EPSILON)
    frame["vol_change"] = frame["volume"].pct_change()

    for window in (5, 20, 60):
        moving_average = frame["close"].rolling(window=window).mean()
        frame[f"ma_{window}_ratio"] = (frame["close"] - moving_average) / (moving_average + _EPSILON)

    frame["rsi"] = _compute_rsi(frame["close"], period=14) / 100.0

    exp12 = frame["close"].ewm(span=12, adjust=False).mean()
    exp26 = frame["close"].ewm(span=26, adjust=False).mean()
    frame["macd_ratio"] = (exp12 - exp26) / (frame["close"] + _EPSILON)

    ma20 = frame["close"].rolling(window=20).mean()
    std20 = frame["close"].rolling(window=20).std()
    upper = ma20 + (2 * std20)
    lower = ma20 - (2 * std20)
    frame["bb_position"] = (frame["close"] - lower) / ((upper - lower).replace(0, _EPSILON))

    return frame


def build_features(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    timeframe: str = "1D",
) -> pd.DataFrame:
    """
    지정한 타임프레임 기준으로 Lens의 17개 피처 데이터프레임을 만든다.

    규칙:
    - Market OHLCV 핵심 결측은 먼저 제거
    - Macro / breadth 값은 병합 후 forward fill
    """
    timeframe = normalize_timeframe(timeframe)
    price_frame = resample_price_frame(price_df, timeframe)
    if price_frame.empty:
        return pd.DataFrame(columns=["ticker", "date", "timeframe", *FEATURE_COLUMNS])

    macro_frame = _resample_context_frame(
        macro_df,
        timeframe,
        ("us10y", "yield_spread", "vix_close", "credit_spread_hy"),
    )
    breadth_frame = _resample_context_frame(
        breadth_df,
        timeframe,
        ("nh_nl_index", "ma200_pct"),
    )

    built_frames: list[pd.DataFrame] = []
    grouped_frames = price_frame.groupby("ticker", sort=True) if "ticker" in price_frame.columns else [(None, price_frame)]
    for ticker_name, ticker_frame in grouped_frames:
        feature_frame = _compute_features_for_single_ticker(ticker_frame)
        if "ticker" not in feature_frame.columns:
            feature_frame["ticker"] = ticker_name or "UNKNOWN"
        if not macro_frame.empty:
            feature_frame = feature_frame.merge(macro_frame, on="date", how="left")
        if not breadth_frame.empty:
            feature_frame = feature_frame.merge(breadth_frame, on="date", how="left")

        # 외부 컨텍스트 테이블이 비어 있어도 동일한 컬럼 집합을 유지한다.
        for column in _CONTEXT_COLUMNS:
            if column not in feature_frame.columns:
                feature_frame[column] = np.nan

        fill_columns = [col for col in _CONTEXT_COLUMNS if col in feature_frame.columns]
        if fill_columns:
            feature_frame[fill_columns] = feature_frame[fill_columns].ffill()

        feature_frame["timeframe"] = timeframe
        feature_frame = feature_frame.dropna(subset=FEATURE_COLUMNS)
        built_frames.append(feature_frame[["ticker", "date", "timeframe", *FEATURE_COLUMNS]].copy())

    if not built_frames:
        return pd.DataFrame(columns=["ticker", "date", "timeframe", *FEATURE_COLUMNS])
    return pd.concat(built_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_latest_feature_rows(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    timeframe: str = "1D",
) -> pd.DataFrame:
    features = build_features(price_df=price_df, macro_df=macro_df, breadth_df=breadth_df, timeframe=timeframe)
    if features.empty:
        return features
    return features.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
