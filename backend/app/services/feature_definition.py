"""
Lens 피처 정의 상수.

CP225 분리 산출. v1 피처 컬럼/타임프레임/sanity 한도 등 정의 상수만 모은다.
의존: 표준 라이브러리만. numpy/pandas import 금지.
"""

from __future__ import annotations

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
_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "timeframe",
    "regime_label",
    *FEATURE_COLUMNS,
    *_INDICATOR_ONLY_COLUMNS,
]
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

__all__ = [
    "FEATURE_COLUMNS",
    "REQUIRED_FEATURE_COLUMNS",
    "PRICE_DERIVED_FEATURE_COLUMNS",
    "SUPPORTED_TIMEFRAMES",
    "_BASE_FEATURE_COLUMNS",
    "_REGIME_FEATURE_COLUMNS",
    "_FUNDAMENTAL_FEATURE_COLUMNS",
    "_MACRO_FEATURE_COLUMNS",
    "_BREADTH_FEATURE_COLUMNS",
    "_FUNDAMENTAL_FLAG_COLUMN",
    "_MACRO_FLAG_COLUMN",
    "_BREADTH_FLAG_COLUMN",
    "_EPSILON",
    "_CONTEXT_COLUMNS",
    "_REGIME_COLUMNS",
    "_FUNDAMENTAL_SOURCE_COLUMNS",
    "_INDICATOR_ONLY_COLUMNS",
    "_OUTPUT_COLUMNS",
    "_ADJUSTED_OHLC_COLUMNS",
    "_RATIO_SANITY_COLUMNS",
    "_MAX_RATIO_ABS_LIMIT",
    "_P99_RATIO_ABS_LIMIT",
]
