"""
Lens 피처 생성 서비스.

이 파일을 v1 기준 기본 피처 정의의 단일 기준점으로 사용한다.
또한 1D / 1W / 1M 타임프레임별 리샘플링과 피처 생성을 함께 담당한다.
"""

from __future__ import annotations

from backend.app.services.feature_calculator import (
    _apply_context_flags,  # noqa: F401 (re-export)
    _apply_fundamental_features,  # noqa: F401 (re-export)
    _apply_regime_columns,  # noqa: F401 (re-export)
    _compute_features_for_single_ticker,  # noqa: F401 (re-export)
    _compute_rsi,  # noqa: F401 (re-export)
    build_features,  # noqa: F401 (re-export)
    build_price_features,  # noqa: F401 (re-export)
)
from backend.app.services.feature_definition import (
    _ADJUSTED_OHLC_COLUMNS,  # noqa: F401 (re-export)
    _BASE_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    _BREADTH_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    _BREADTH_FLAG_COLUMN,  # noqa: F401 (re-export)
    _CONTEXT_COLUMNS,  # noqa: F401 (re-export)
    _EPSILON,  # noqa: F401 (re-export)
    _FUNDAMENTAL_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    _FUNDAMENTAL_FLAG_COLUMN,  # noqa: F401 (re-export)
    _FUNDAMENTAL_SOURCE_COLUMNS,  # noqa: F401 (re-export)
    _INDICATOR_ONLY_COLUMNS,  # noqa: F401 (re-export)
    _MACRO_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    _MACRO_FLAG_COLUMN,  # noqa: F401 (re-export)
    _MAX_RATIO_ABS_LIMIT,  # noqa: F401 (re-export)
    _OUTPUT_COLUMNS,  # noqa: F401 (re-export)
    _P99_RATIO_ABS_LIMIT,  # noqa: F401 (re-export)
    _RATIO_SANITY_COLUMNS,  # noqa: F401 (re-export)
    _REGIME_COLUMNS,  # noqa: F401 (re-export)
    _REGIME_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    FEATURE_COLUMNS,  # noqa: F401 (re-export)
    PRICE_DERIVED_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    REQUIRED_FEATURE_COLUMNS,  # noqa: F401 (re-export)
    SUPPORTED_TIMEFRAMES,  # noqa: F401 (re-export)
)
from backend.app.services.resampling import (
    _ensure_datetime,  # noqa: F401 (re-export)
    _resample_context_frame,  # noqa: F401 (re-export)
    _resample_single_ticker,  # noqa: F401 (re-export)
    drop_incomplete_resampled_periods,  # noqa: F401 (re-export)
    latest_complete_period_end,  # noqa: F401 (re-export)
    normalize_timeframe,  # noqa: F401 (re-export)
    resample_price_frame,  # noqa: F401 (re-export)
)
from backend.app.services.validators import (
    _apply_adjusted_ohlc_contract,  # noqa: F401 (re-export)
    _validate_adjusted_ohlc_contract,  # noqa: F401 (re-export)
    _validate_ratio_feature_sanity,  # noqa: F401 (re-export)
)
