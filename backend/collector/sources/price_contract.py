from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PROVIDER_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
RATIO_SANITY_COLUMNS = ("open_ratio", "high_ratio", "low_ratio")
MAX_RATIO_ABS_LIMIT = 5.0
P99_RATIO_ABS_LIMIT = 1.0
EPSILON = 1e-9


@dataclass(frozen=True)
class PriceContractValidation:
    passed: bool
    violations: list[str]
    metrics: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def prepare_provider_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        return prepared

    prepared.index = pd.to_datetime(prepared.index, errors="coerce")
    prepared = prepared.sort_index()
    for column in [*REQUIRED_PROVIDER_PRICE_COLUMNS, "Volume", "Amount"]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "Volume" not in prepared.columns:
        prepared["Volume"] = 0
    prepared["Volume"] = prepared["Volume"].fillna(0)
    if "Amount" not in prepared.columns and {"Close", "Volume"}.issubset(prepared.columns):
        prepared["Amount"] = prepared["Close"] * prepared["Volume"]
    return prepared


def validate_adjusted_ohlc_contract(
    ticker: str,
    frame: pd.DataFrame,
    *,
    enforce_ratio_distribution: bool = True,
) -> PriceContractValidation:
    violations: list[str] = []
    metrics: dict[str, Any] = {
        "ticker": ticker,
        "row_count": int(len(frame)),
        "volume_null_policy": "Volume null은 저장 전 0으로 채우고, 음수 volume은 실패 처리한다.",
    }

    if frame.empty:
        return PriceContractValidation(
            passed=False,
            violations=["empty_frame"],
            metrics=metrics,
        )

    working = prepare_provider_price_frame(frame)
    metrics["row_count"] = int(len(working))

    missing_columns = [column for column in REQUIRED_PROVIDER_PRICE_COLUMNS if column not in working.columns]
    if missing_columns:
        violations.append(f"missing_columns:{','.join(missing_columns)}")
        metrics["missing_columns"] = missing_columns
        return PriceContractValidation(False, violations, metrics)

    null_dates = int(pd.isna(working.index).sum())
    metrics["date_null_count"] = null_dates
    if null_dates:
        violations.append("date_null")

    normalized_dates = pd.Series(working.index.normalize())
    duplicate_dates = int(normalized_dates.duplicated().sum())
    metrics["duplicate_date_count"] = duplicate_dates
    if duplicate_dates:
        violations.append("duplicate_date")

    required_null_counts = {
        column: int(working[column].isna().sum())
        for column in REQUIRED_PROVIDER_PRICE_COLUMNS
    }
    metrics["required_null_counts"] = required_null_counts
    if any(count > 0 for count in required_null_counts.values()):
        violations.append("required_price_null")

    required_values = working[REQUIRED_PROVIDER_PRICE_COLUMNS].to_numpy(dtype=float)
    finite_count = int(np.isfinite(required_values).sum())
    total_required = int(required_values.size)
    metrics["required_finite_count"] = finite_count
    metrics["required_value_count"] = total_required
    if finite_count != total_required:
        violations.append("required_price_non_finite")

    volume_null_count = int(frame["Volume"].isna().sum()) if "Volume" in frame.columns else len(frame)
    volume_negative_count = int((working["Volume"] < 0).sum())
    metrics["volume_null_count"] = volume_null_count
    metrics["volume_negative_count"] = volume_negative_count
    if volume_negative_count:
        violations.append("volume_negative")

    close = working["Close"]
    adjusted_close = working["Adj Close"]
    adjusted_factor = adjusted_close / close.where(close.abs() > EPSILON)
    invalid_factor = adjusted_factor.isna() | ~np.isfinite(adjusted_factor) | (adjusted_factor <= 0)
    metrics["adjusted_factor_invalid_count"] = int(invalid_factor.sum())
    metrics["adjusted_factor_min"] = _safe_float(adjusted_factor.min())
    metrics["adjusted_factor_max"] = _safe_float(adjusted_factor.max())
    if bool(invalid_factor.any()):
        violations.append("adjusted_factor_invalid")

    adjusted_open = working["Open"] * adjusted_factor
    adjusted_high = working["High"] * adjusted_factor
    adjusted_low = working["Low"] * adjusted_factor

    adjusted_high_violation = adjusted_high + EPSILON < pd.concat([adjusted_open, adjusted_close], axis=1).max(axis=1)
    adjusted_low_violation = adjusted_low - EPSILON > pd.concat([adjusted_open, adjusted_close], axis=1).min(axis=1)
    adjusted_high_low_violation = adjusted_high + EPSILON < adjusted_low
    metrics["adjusted_high_violation_count"] = int(adjusted_high_violation.sum())
    metrics["adjusted_low_violation_count"] = int(adjusted_low_violation.sum())
    metrics["adjusted_high_low_violation_count"] = int(adjusted_high_low_violation.sum())
    if bool(adjusted_high_violation.any()):
        violations.append("adjusted_high_below_open_or_close")
    if bool(adjusted_low_violation.any()):
        violations.append("adjusted_low_above_open_or_close")
    if bool(adjusted_high_low_violation.any()):
        violations.append("adjusted_high_below_low")

    previous_close = adjusted_close.shift(1)
    ratios = pd.DataFrame(
        {
            "open_ratio": (adjusted_open - previous_close) / (previous_close + EPSILON),
            "high_ratio": (adjusted_high - previous_close) / (previous_close + EPSILON),
            "low_ratio": (adjusted_low - previous_close) / (previous_close + EPSILON),
        },
        index=working.index,
    ).replace([np.inf, -np.inf], np.nan)
    ratio_frame = ratios.dropna()
    metrics["ratio_observation_count"] = int(len(ratio_frame))
    if not ratio_frame.empty:
        ratio_values = ratio_frame.to_numpy(dtype=float)
        ratio_non_finite = int((~np.isfinite(ratio_values)).sum())
        metrics["ratio_non_finite_count"] = ratio_non_finite
        if ratio_non_finite:
            violations.append("ratio_non_finite")
        abs_ratios = ratio_frame.abs()
        p99_abs = abs_ratios.quantile(0.99)
        max_abs = abs_ratios.max()
        metrics["ratio_p99_abs"] = {column: _safe_float(p99_abs[column]) for column in RATIO_SANITY_COLUMNS}
        metrics["ratio_max_abs"] = {column: _safe_float(max_abs[column]) for column in RATIO_SANITY_COLUMNS}
        if enforce_ratio_distribution:
            ratio_failures = [
                column
                for column in RATIO_SANITY_COLUMNS
                if float(p99_abs[column]) > P99_RATIO_ABS_LIMIT or float(max_abs[column]) > MAX_RATIO_ABS_LIMIT
            ]
            if ratio_failures:
                metrics["ratio_failure_columns"] = ratio_failures
                violations.append("ratio_sanity_failed")
    else:
        metrics["ratio_non_finite_count"] = 0
        metrics["ratio_p99_abs"] = {column: None for column in RATIO_SANITY_COLUMNS}
        metrics["ratio_max_abs"] = {column: None for column in RATIO_SANITY_COLUMNS}

    metrics["violation_count"] = len(violations)
    return PriceContractValidation(
        passed=len(violations) == 0,
        violations=violations,
        metrics=metrics,
    )

