from __future__ import annotations

import numpy as np


SUPPORTED_TARGET_TYPES = (
    "raw_future_return",
    "market_excess_return",
    "volatility_normalized_return",
    "direction_label",
    "rank_target",
)


def normalize_target_type(target_type: str) -> str:
    normalized = str(target_type or "raw_future_return").strip().lower()
    if normalized not in SUPPORTED_TARGET_TYPES:
        raise ValueError(f"지원하지 않는 target_type입니다: {target_type}")
    return normalized


def build_target_array(
    future_returns: np.ndarray,
    *,
    history_returns: np.ndarray | None = None,
    target_type: str = "raw_future_return",
    market_future_returns: np.ndarray | None = None,
) -> np.ndarray:
    normalized = normalize_target_type(target_type)
    future = np.asarray(future_returns, dtype="float32")

    if normalized == "raw_future_return":
        return future

    if normalized == "volatility_normalized_return":
        if history_returns is None or len(history_returns) == 0:
            scale = 1.0
        else:
            scale = float(np.std(np.asarray(history_returns, dtype="float32"), ddof=0))
        return future / max(scale, 1e-6)

    if normalized == "direction_label":
        return (future > 0.0).astype("float32")

    if normalized == "market_excess_return":
        if market_future_returns is None:
            raise NotImplementedError("market_excess_return은 시장 벤치마크 경로가 아직 연결되지 않았습니다.")
        return future - np.asarray(market_future_returns, dtype="float32")

    if normalized == "rank_target":
        raise NotImplementedError("rank_target은 교차 단면 target 생성 경로가 아직 연결되지 않았습니다.")

    raise ValueError(f"지원하지 않는 target_type입니다: {target_type}")
