"""
Lens 배치 예측용 모델 메타 헬퍼.

v1 API는 이미 저장된 배치 예측만 조회하므로,
이 파일은 모델명 / 타임프레임 / 기본 horizon 검증에 집중한다.
"""

from __future__ import annotations

SUPPORTED_MODELS = ("patchtst", "cnn_lstm", "tide")
SUPPORTED_TIMEFRAMES = ("1D", "1W", "1M")
DEFAULT_HORIZONS = {
    "1D": 10,
    "1W": 12,
    "1M": 6,
}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of: {', '.join(SUPPORTED_MODELS)}")
    return normalized


def normalize_timeframe(timeframe: str) -> str:
    normalized = timeframe.strip().upper()
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return normalized


def resolve_horizon(timeframe: str, horizon: int | None = None) -> int:
    normalized_timeframe = normalize_timeframe(timeframe)
    if horizon is None:
        return DEFAULT_HORIZONS[normalized_timeframe]

    resolved = int(horizon)
    if resolved <= 0:
        raise ValueError("horizon must be a positive integer.")
    return resolved
