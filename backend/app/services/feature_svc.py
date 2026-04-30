"""
Lens 피처 생성 서비스.

이 파일을 v1 기준 기본 피처 정의의 단일 기준점으로 사용한다.
또한 1D / 1W / 1M 타임프레임별 리샘플링과 피처 생성을 함께 담당한다.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_BASE_FEATURE_COLUMNS = [
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
_REGIME_FEATURE_COLUMNS = [
    "regime_calm",
    "regime_neutral",
    "regime_stress",
]
_FUNDAMENTAL_FEATURE_COLUMNS = [
    "revenue",
    "net_income",
    "equity",
    "eps",
    "roe",
    "debt_ratio",
]
_MACRO_FEATURE_COLUMNS = [
    "us10y",
    "yield_spread",
    "vix_close",
    "credit_spread_hy",
]
_BREADTH_FEATURE_COLUMNS = [
    "nh_nl_index",
    "ma200_pct",
]
_FUNDAMENTAL_FLAG_COLUMN = "has_fundamentals"
_MACRO_FLAG_COLUMN = "has_macro"
_BREADTH_FLAG_COLUMN = "has_breadth"
REQUIRED_FEATURE_COLUMNS = [
    *_BASE_FEATURE_COLUMNS,
    *_REGIME_FEATURE_COLUMNS,
    _MACRO_FLAG_COLUMN,
    _BREADTH_FLAG_COLUMN,
    _FUNDAMENTAL_FLAG_COLUMN,
]
FEATURE_COLUMNS = [
    *_BASE_FEATURE_COLUMNS,
    *_REGIME_FEATURE_COLUMNS,
    *_FUNDAMENTAL_FEATURE_COLUMNS,
    _MACRO_FLAG_COLUMN,
    _BREADTH_FLAG_COLUMN,
    _FUNDAMENTAL_FLAG_COLUMN,
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
_REGIME_COLUMNS = ("regime_calm", "regime_neutral", "regime_stress")
_FUNDAMENTAL_SOURCE_COLUMNS = (
    "filing_date",
    "revenue",
    "net_income",
    "equity",
    "eps",
    "total_liabilities",
)
_INDICATOR_ONLY_COLUMNS = ["atr_ratio"]
_OUTPUT_COLUMNS = ["ticker", "date", "timeframe", "regime_label", *FEATURE_COLUMNS, *_INDICATOR_ONLY_COLUMNS]
PRICE_DERIVED_FEATURE_COLUMNS = [
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
]
_ADJUSTED_OHLC_COLUMNS = ("open", "high", "low", "close")
_RATIO_SANITY_COLUMNS = ("open_ratio", "high_ratio", "low_ratio")
_MAX_RATIO_ABS_LIMIT = 5.0
_P99_RATIO_ABS_LIMIT = 1.0


def normalize_timeframe(timeframe: str) -> str:
    normalized = timeframe.strip().upper()
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return normalized


def default_horizon_for_timeframe(timeframe: str) -> int:
    normalized = normalize_timeframe(timeframe)
    return {"1D": 5, "1W": 4, "1M": 3}[normalized]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    sort_columns = ["ticker", "date"] if "ticker" in frame.columns else ["date"]
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _validate_adjusted_ohlc_contract(frame: pd.DataFrame, *, context: str) -> None:
    missing = [column for column in _ADJUSTED_OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: adjusted OHLC 계약에 필요한 컬럼이 없습니다: {missing}")

    ohlc = frame[list(_ADJUSTED_OHLC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(ohlc.to_numpy(dtype=float)).all():
        invalid_counts = (~np.isfinite(ohlc.to_numpy(dtype=float))).sum(axis=0).tolist()
        raise ValueError(f"{context}: adjusted OHLC에 non-finite 값이 있습니다: {dict(zip(_ADJUSTED_OHLC_COLUMNS, invalid_counts, strict=False))}")

    high = ohlc["high"]
    low = ohlc["low"]
    open_ = ohlc["open"]
    close = ohlc["close"]
    if bool((high + _EPSILON < low).any()):
        raise ValueError(f"{context}: adjusted high가 adjusted low보다 작은 행이 있습니다.")
    if bool((high + _EPSILON < pd.concat([open_, close], axis=1).max(axis=1)).any()):
        raise ValueError(f"{context}: adjusted high가 open/close보다 작은 행이 있습니다.")
    if bool((low - _EPSILON > pd.concat([open_, close], axis=1).min(axis=1)).any()):
        raise ValueError(f"{context}: adjusted low가 open/close보다 큰 행이 있습니다.")


def _apply_adjusted_ohlc_contract(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        return frame
    if "adjusted_close" not in frame.columns:
        frame["adjusted_close"] = frame["close"]

    raw_close = pd.to_numeric(frame["close"], errors="coerce")
    adjusted_close = pd.to_numeric(frame["adjusted_close"], errors="coerce").fillna(raw_close)
    adjusted_factor = adjusted_close / raw_close.where(raw_close.abs() > _EPSILON)
    adjusted_factor = adjusted_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    for column in ("open", "high", "low"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") * adjusted_factor
    frame["close"] = adjusted_close
    frame["adjusted_close"] = adjusted_close
    _validate_adjusted_ohlc_contract(frame, context=context)
    return frame


def _validate_ratio_feature_sanity(frame: pd.DataFrame, *, context: str, enforce_distribution: bool = True) -> None:
    if frame.empty:
        return
    ratio_frame = frame[list(_RATIO_SANITY_COLUMNS)].dropna()
    if ratio_frame.empty:
        return
    ratio_values = ratio_frame.to_numpy(dtype=float)
    if not np.isfinite(ratio_values).all():
        raise ValueError(f"{context}: OHLC ratio 피처에 non-finite 값이 있습니다.")
    if not enforce_distribution:
        return

    abs_ratios = ratio_frame.abs()
    p99_abs = abs_ratios.quantile(0.99)
    max_abs = abs_ratios.max()
    failures = {
        column: {
            "p99_abs": float(p99_abs[column]),
            "max_abs": float(max_abs[column]),
        }
        for column in _RATIO_SANITY_COLUMNS
        if float(p99_abs[column]) > _P99_RATIO_ABS_LIMIT or float(max_abs[column]) > _MAX_RATIO_ABS_LIMIT
    }
    if failures:
        raise ValueError(f"{context}: OHLC ratio sanity check 실패: {failures}")


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
    if "ticker" in df.columns and not aggregated.empty:
        aggregated["ticker"] = df["ticker"].iloc[0]
    if not aggregated.empty:
        _validate_adjusted_ohlc_contract(aggregated, context=f"resample_price_frame:{timeframe}:aggregated")
    return aggregated


def resample_price_frame(price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    frame = _ensure_datetime(price_df)
    required_price_cols = ["open", "high", "low", "close"]
    frame = frame.dropna(subset=[col for col in required_price_cols if col in frame.columns])

    if timeframe == "1D" or frame.empty:
        return _apply_adjusted_ohlc_contract(frame, context=f"resample_price_frame:{timeframe}").reset_index(drop=True)

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


def _compute_features_for_single_ticker(df: pd.DataFrame, *, enforce_ratio_distribution: bool = True) -> pd.DataFrame:
    frame = _apply_adjusted_ohlc_contract(
        df.copy().sort_values("date").reset_index(drop=True),
        context="compute_features",
    )
    close_base = frame["adjusted_close"].fillna(frame["close"]) if "adjusted_close" in frame.columns else frame["close"]
    frame["close"] = close_base

    previous_close = frame["close"].shift(1)
    frame["log_return"] = np.log(frame["close"] / previous_close)
    frame["open_ratio"] = (frame["open"] - previous_close) / (previous_close + _EPSILON)
    frame["high_ratio"] = (frame["high"] - previous_close) / (previous_close + _EPSILON)
    frame["low_ratio"] = (frame["low"] - previous_close) / (previous_close + _EPSILON)
    frame["vol_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    for window in (5, 20, 60):
        moving_average = frame["close"].rolling(window=window).mean()
        frame[f"ma_{window}_ratio"] = (frame["close"] - moving_average) / (moving_average + _EPSILON)

    # ATR은 절대값 대신 종가 대비 비율로 정규화해 종목 간 변동성 비교에 쓰기 쉽게 만든다.
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    frame["atr_ratio"] = atr / (frame["close"] + _EPSILON)

    frame["rsi"] = _compute_rsi(frame["close"], period=14) / 100.0

    exp12 = frame["close"].ewm(span=12, adjust=False).mean()
    exp26 = frame["close"].ewm(span=26, adjust=False).mean()
    frame["macd_ratio"] = (exp12 - exp26) / (frame["close"] + _EPSILON)

    ma20 = frame["close"].rolling(window=20).mean()
    std20 = frame["close"].rolling(window=20).std()
    upper = ma20 + (2 * std20)
    lower = ma20 - (2 * std20)
    frame["bb_position"] = (frame["close"] - lower) / ((upper - lower).replace(0, _EPSILON))
    _validate_ratio_feature_sanity(
        frame,
        context="compute_features",
        enforce_distribution=enforce_ratio_distribution,
    )

    return frame


def build_price_features(price_df: pd.DataFrame, timeframe: str = "1D") -> pd.DataFrame:
    """adjusted OHLC 계약으로 가격 파생 피처만 다시 계산한다."""
    timeframe = normalize_timeframe(timeframe)
    price_frame = resample_price_frame(price_df, timeframe)
    if price_frame.empty:
        return pd.DataFrame(columns=["ticker", "date", "timeframe", *PRICE_DERIVED_FEATURE_COLUMNS])

    built_frames: list[pd.DataFrame] = []
    grouped_frames = price_frame.groupby("ticker", sort=True) if "ticker" in price_frame.columns else [(None, price_frame)]
    for ticker_name, ticker_frame in grouped_frames:
        feature_frame = _compute_features_for_single_ticker(
            ticker_frame,
            enforce_ratio_distribution=timeframe != "1M",
        )
        if "ticker" not in feature_frame.columns:
            feature_frame["ticker"] = ticker_name or "UNKNOWN"
        feature_frame["timeframe"] = timeframe
        feature_frame = feature_frame.dropna(subset=PRICE_DERIVED_FEATURE_COLUMNS)
        built_frames.append(feature_frame[["ticker", "date", "timeframe", *PRICE_DERIVED_FEATURE_COLUMNS]].copy())

    if not built_frames:
        return pd.DataFrame(columns=["ticker", "date", "timeframe", *PRICE_DERIVED_FEATURE_COLUMNS])
    return pd.concat(built_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def _apply_regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "vix_close" not in enriched.columns:
        enriched["regime_label"] = np.nan
        for column in _REGIME_COLUMNS:
            enriched[column] = np.nan
        return enriched

    conditions = [
        enriched["vix_close"] < 15,
        enriched["vix_close"] >= 25,
    ]
    enriched["regime_label"] = np.select(
        conditions,
        ["calm", "stress"],
        default="neutral",
    )
    enriched["regime_calm"] = (enriched["regime_label"] == "calm").astype(float)
    enriched["regime_neutral"] = (enriched["regime_label"] == "neutral").astype(float)
    enriched["regime_stress"] = (enriched["regime_label"] == "stress").astype(float)
    return enriched


def _apply_context_flags(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for column in _MACRO_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = np.nan
    for column in _BREADTH_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = np.nan

    enriched[_MACRO_FLAG_COLUMN] = enriched[_MACRO_FEATURE_COLUMNS].notna().any(axis=1)
    enriched[_BREADTH_FLAG_COLUMN] = enriched[_BREADTH_FEATURE_COLUMNS].notna().any(axis=1)
    enriched[_MACRO_FEATURE_COLUMNS] = enriched[_MACRO_FEATURE_COLUMNS].fillna(0.0)
    enriched[_BREADTH_FEATURE_COLUMNS] = enriched[_BREADTH_FEATURE_COLUMNS].fillna(0.0)
    return enriched


def _apply_fundamental_features(
    frame: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None,
) -> pd.DataFrame:
    enriched = frame.copy()

    if fundamentals_df is None or fundamentals_df.empty:
        for column in _FUNDAMENTAL_FEATURE_COLUMNS:
            enriched[column] = 0.0
        enriched[_FUNDAMENTAL_FLAG_COLUMN] = False
        enriched["fundamental_quarter_count"] = 0
        return enriched

    fundamentals = fundamentals_df.copy()
    if "filing_date" not in fundamentals.columns:
        for column in _FUNDAMENTAL_FEATURE_COLUMNS:
            enriched[column] = 0.0
        enriched[_FUNDAMENTAL_FLAG_COLUMN] = False
        enriched["fundamental_quarter_count"] = 0
        return enriched

    fundamentals["filing_date"] = pd.to_datetime(fundamentals["filing_date"], errors="coerce")
    available_columns = [column for column in _FUNDAMENTAL_SOURCE_COLUMNS if column in fundamentals.columns]
    fundamentals = fundamentals.dropna(subset=["filing_date"])[available_columns].sort_values("filing_date")
    if fundamentals.empty:
        for column in _FUNDAMENTAL_FEATURE_COLUMNS:
            enriched[column] = 0.0
        enriched[_FUNDAMENTAL_FLAG_COLUMN] = False
        enriched["fundamental_quarter_count"] = 0
        return enriched

    # filing_date 기준으로만 과거 값을 붙여 누수를 막는다.
    merged = pd.merge_asof(
        enriched.sort_values("date"),
        fundamentals,
        left_on="date",
        right_on="filing_date",
        direction="backward",
        allow_exact_matches=True,
    )

    if "equity" not in merged.columns:
        merged["equity"] = np.nan
    if "net_income" not in merged.columns:
        merged["net_income"] = np.nan
    if "eps" not in merged.columns:
        merged["eps"] = np.nan
    if "revenue" not in merged.columns:
        merged["revenue"] = np.nan
    if "total_liabilities" not in merged.columns:
        merged["total_liabilities"] = np.nan

    valid_equity = merged["equity"].where(merged["equity"] > 0)
    merged["roe"] = merged["net_income"] / valid_equity
    merged["debt_ratio"] = merged["total_liabilities"] / valid_equity

    filing_dates = fundamentals["filing_date"].to_numpy(dtype="datetime64[ns]")
    merged_dates = merged["date"].to_numpy(dtype="datetime64[ns]")
    merged["fundamental_quarter_count"] = np.searchsorted(filing_dates, merged_dates, side="right")

    insufficient_mask = merged["fundamental_quarter_count"] < 8
    merged.loc[insufficient_mask, _FUNDAMENTAL_FEATURE_COLUMNS] = np.nan
    merged[_FUNDAMENTAL_FLAG_COLUMN] = merged[_FUNDAMENTAL_FEATURE_COLUMNS].notna().any(axis=1)
    merged[_FUNDAMENTAL_FEATURE_COLUMNS] = merged[_FUNDAMENTAL_FEATURE_COLUMNS].fillna(0.0)
    return merged


def build_features(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    fundamentals_df: pd.DataFrame | None = None,
    timeframe: str = "1D",
) -> pd.DataFrame:
    """
    지정한 타임프레임 기준으로 Lens의 기본 피처 데이터프레임을 만든다.

    규칙:
    - Market OHLCV 핵심 결측은 먼저 제거
    - Macro / breadth 값은 병합 후 forward fill
    - VIX 기준 시장 국면을 계산해 원핫 3열로 확장
    """
    timeframe = normalize_timeframe(timeframe)
    price_frame = resample_price_frame(price_df, timeframe)
    if price_frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

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
        feature_frame = _compute_features_for_single_ticker(
            ticker_frame,
            enforce_ratio_distribution=timeframe != "1M",
        )
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
        feature_frame = _apply_context_flags(feature_frame)

        ticker_fundamentals = None
        if fundamentals_df is not None and not fundamentals_df.empty and ticker_name is not None:
            ticker_fundamentals = fundamentals_df[fundamentals_df["ticker"] == ticker_name]
        feature_frame = _apply_fundamental_features(feature_frame, ticker_fundamentals)
        feature_frame = _apply_regime_columns(feature_frame)
        feature_frame["timeframe"] = timeframe
        feature_frame = feature_frame.dropna(subset=REQUIRED_FEATURE_COLUMNS)
        built_frames.append(feature_frame[_OUTPUT_COLUMNS].copy())

    if not built_frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(built_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_latest_feature_rows(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    fundamentals_df: pd.DataFrame | None = None,
    timeframe: str = "1D",
) -> pd.DataFrame:
    features = build_features(
        price_df=price_df,
        macro_df=macro_df,
        breadth_df=breadth_df,
        fundamentals_df=fundamentals_df,
        timeframe=timeframe,
    )
    if features.empty:
        return features
    return features.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
