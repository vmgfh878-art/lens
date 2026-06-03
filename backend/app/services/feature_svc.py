"""
Lens 피처 서비스 — 호환 유지용 re-export 파사드.

실제 구현은 CP225 분리 산출 4 모듈에 있다:
  - feature_definition: 컬럼/타임프레임/sanity 한도 등 정의 상수
  - validators        : adjusted OHLC 계약 / ratio sanity 검증
  - resampling        : 타임프레임 정규화 / 가격·컨텍스트 리샘플
  - feature_calculator: RSI/MACD/볼린저/ATR/MA비율, regime/context flag,
                        fundamental merge_asof, build_features 오케스트레이션

기존 caller (백엔드 / ai/ / scripts/ / diagnostics) 의 import 경로
(backend.app.services.feature_svc.X) 와 공개 + 사설 심볼을 보존한다.
새 코드는 4 모듈을 직접 import 하는 것을 권장.
"""

from __future__ import annotations

from backend.app.services.feature_calculator import (
    _apply_context_flags,
    _apply_fundamental_features,
    _apply_regime_columns,
    _compute_features_for_single_ticker,
    _compute_rsi,
    build_features,
    build_price_features,
)
from backend.app.services.feature_definition import (
    _ADJUSTED_OHLC_COLUMNS,
    _BASE_FEATURE_COLUMNS,
    _BREADTH_FEATURE_COLUMNS,
    _BREADTH_FLAG_COLUMN,
    _CONTEXT_COLUMNS,
    _EPSILON,
    _FUNDAMENTAL_FEATURE_COLUMNS,
    _FUNDAMENTAL_FLAG_COLUMN,
    _FUNDAMENTAL_SOURCE_COLUMNS,
    _INDICATOR_ONLY_COLUMNS,
    _MACRO_FEATURE_COLUMNS,
    _MACRO_FLAG_COLUMN,
    _MAX_RATIO_ABS_LIMIT,
    _OUTPUT_COLUMNS,
    _P99_RATIO_ABS_LIMIT,
    _RATIO_SANITY_COLUMNS,
    _REGIME_COLUMNS,
    _REGIME_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    PRICE_DERIVED_FEATURE_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    SUPPORTED_TIMEFRAMES,
)
from backend.app.services.resampling import (
    _ensure_datetime,
    _resample_context_frame,
    _resample_single_ticker,
    drop_incomplete_resampled_periods,
    latest_complete_period_end,
    normalize_timeframe,
    resample_price_frame,
)
from backend.app.services.validators import (
    _apply_adjusted_ohlc_contract,
    _validate_adjusted_ohlc_contract,
    _validate_ratio_feature_sanity,
)

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
    "normalize_timeframe",
    "_ensure_datetime",
    "latest_complete_period_end",
    "drop_incomplete_resampled_periods",
    "_resample_single_ticker",
    "resample_price_frame",
    "_resample_context_frame",
    "_validate_adjusted_ohlc_contract",
    "_apply_adjusted_ohlc_contract",
    "_validate_ratio_feature_sanity",
    "_compute_rsi",
    "_compute_features_for_single_ticker",
    "build_price_features",
    "_apply_regime_columns",
    "_apply_context_flags",
    "_apply_fundamental_features",
    "build_features",
]
