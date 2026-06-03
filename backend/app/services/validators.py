"""
Lens 피처 검증.

CP225 분리 산출. adjusted OHLC 계약과 비율 피처 분포 sanity 검증.
의존: feature_definition (상수만).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.services.feature_definition import (
    _ADJUSTED_OHLC_COLUMNS,
    _EPSILON,
    _MAX_RATIO_ABS_LIMIT,
    _P99_RATIO_ABS_LIMIT,
    _RATIO_SANITY_COLUMNS,
)


def _validate_adjusted_ohlc_contract(frame: pd.DataFrame, *, context: str) -> None:
    missing = [column for column in _ADJUSTED_OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: adjusted OHLC 계약에 필요한 컬럼이 없습니다: {missing}")

    ohlc = frame[list(_ADJUSTED_OHLC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(ohlc.to_numpy(dtype=float)).all():
        invalid_counts = (~np.isfinite(ohlc.to_numpy(dtype=float))).sum(axis=0).tolist()
        raise ValueError(
            f"{context}: adjusted OHLC에 non-finite 값이 있습니다: {dict(zip(_ADJUSTED_OHLC_COLUMNS, invalid_counts, strict=False))}"
        )

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


def _validate_ratio_feature_sanity(
    frame: pd.DataFrame, *, context: str, enforce_distribution: bool = True
) -> None:
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
        if float(p99_abs[column]) > _P99_RATIO_ABS_LIMIT
        or float(max_abs[column]) > _MAX_RATIO_ABS_LIMIT
    }
    if failures:
        raise ValueError(f"{context}: OHLC ratio sanity check 실패: {failures}")


__all__ = [
    "_validate_adjusted_ohlc_contract",
    "_apply_adjusted_ohlc_contract",
    "_validate_ratio_feature_sanity",
]
