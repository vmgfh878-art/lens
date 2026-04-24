"""
Lens 배치 예측용 모델 메타 서비스.

가격 조회용 타임프레임과 AI 예측 지원 타임프레임을 구분한다.
- 가격/지표 표시: 1D, 1W, 1M
- AI 예측: 1D, 1W
"""

from __future__ import annotations

from app.core.exceptions import TimeframeDisabledError

SUPPORTED_MODELS = ("patchtst", "cnn_lstm", "tide")
SUPPORTED_DISPLAY_TIMEFRAMES = ("1D", "1W", "1M")
SUPPORTED_AI_TIMEFRAMES = ("1D", "1W")
DEFAULT_HORIZONS = {
    "1D": 5,
    "1W": 4,
}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of: {', '.join(SUPPORTED_MODELS)}")
    return normalized


def normalize_display_timeframe(timeframe: str) -> str:
    normalized = timeframe.strip().upper()
    if normalized not in SUPPORTED_DISPLAY_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of: {', '.join(SUPPORTED_DISPLAY_TIMEFRAMES)}"
        )
    return normalized


def normalize_prediction_timeframe(timeframe: str) -> str:
    normalized = normalize_display_timeframe(timeframe)
    if normalized not in SUPPORTED_AI_TIMEFRAMES:
        raise TimeframeDisabledError(
            "월봉 AI 예측은 Phase 1에서 비활성화돼 있습니다.",
            details={
                "timeframe": normalized,
                "disabled_reason": "display_only",
                "supported_prediction_timeframes": list(SUPPORTED_AI_TIMEFRAMES),
            },
        )
    return normalized


def resolve_horizon(timeframe: str, horizon: int | None = None) -> int:
    normalized_timeframe = normalize_prediction_timeframe(timeframe)
    if horizon is None:
        return DEFAULT_HORIZONS[normalized_timeframe]

    resolved = int(horizon)
    if resolved <= 0:
        raise ValueError("horizon must be a positive integer.")
    return resolved
