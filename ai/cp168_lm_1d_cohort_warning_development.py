from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai import cp159_lm_1d_line_conformal_overlay as cp159  # noqa: E402
from ai import cp160_lm_1d_line_overlay_rejudgement as cp160  # noqa: E402
from ai import cp165_lm_1d_atr_overlay_sweet_spot as cp165  # noqa: E402
from ai import cp167_lm_1d_top_quintile_risk_rescue as cp167  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "cp168_lm_1d_cohort_warning_development_report.md"
METRICS_PATH = DOCS_DIR / "cp168_lm_1d_cohort_warning_development_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp168_lm_1d_cohort_warning_development_summary.csv"
Q5_SWEEP_CSV = DOCS_DIR / "cp168_lm_1d_q5_threshold_sweep.csv"
Q5_COMBO_CSV = DOCS_DIR / "cp168_lm_1d_q5_combination_rule.csv"
TWO_TIER_CSV = DOCS_DIR / "cp168_lm_1d_two_tier_warning_comparison.csv"
CONTINUOUS_CSV = DOCS_DIR / "cp168_lm_1d_continuous_adjustment.csv"
SLIDING_CSV = DOCS_DIR / "cp168_lm_1d_sliding_window.csv"

CP = "CP168-LM"
BASE_RULE_ID = "atr_and_not_self_atr70_self70"
Q5_FEATURES = (
    "intraday_range_5d_mean",
    "intraday_range_20d_mean",
    "close_position_5d_mean",
    "close_position_20d_mean",
    "overnight_gap_abs_5d_mean",
    "overnight_gap_abs_20d_mean",
    "volume_z_20_252",
    "volume_ratio_5_20",
    "drawdown_from_5d_high",
    "drawdown_from_20d_high",
    "downside_vol_ratio_20d",
)
RULE_QS = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
LAMBDA_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else None


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{result:.{digits}f}" if math.isfinite(result) else ""


def _top_q5(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, bottom = cp167.top_bottom_masks(line_score)
    q5 = cp167.top_decile_quintiles(line_score) == 5
    return top, bottom, q5


def _feature_auc_direction(
    *,
    feature: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    severe_threshold: float,
) -> tuple[str, float | None]:
    _top, _bottom, q5 = _top_q5(prediction)
    values = np.asarray(features[feature], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    auc = cp167.auc_score(values[q5], (actual[q5] <= severe_threshold).astype(int))
    return ("high" if (auc or 0.5) >= 0.5 else "low", auc)


def _threshold_for_q5(values: np.ndarray, q5: np.ndarray, q: float, direction: str) -> float:
    q_value = q if direction == "high" else 1.0 - q
    return cp167.threshold_from_values(values[q5], q_value)


def _apply_threshold_rule(
    feature: str, direction: str, threshold: float, features: dict[str, np.ndarray]
) -> np.ndarray:
    values = np.asarray(features[feature], dtype=np.float64)
    return values >= threshold if direction == "high" else values <= threshold


def _row_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        1.0
        if (row.get("spread_retention") or 0.0) >= 0.80
        and (row.get("fee_retention") or 0.0) >= 0.80
        else 0.0,
        float(row.get("false_safe_reduction") or -999.0),
        float(row.get("severe_lift_removed_vs_kept") or -999.0),
        float(row.get("spread_retention") or -999.0),
        -float(row.get("warning_share") or 999.0),
    )


def build_q5_threshold_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    valid_a_prediction: dict[str, Any],
    valid_a_features: dict[str, np.ndarray],
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score_valid = np.asarray(valid_a_prediction["line_score"], dtype=np.float64)
    q5_valid = cp167.top_decile_quintiles(line_score_valid) == 5
    _top, bottom, q5 = _top_q5(prediction)
    rows: list[dict[str, Any]] = []
    for feature in Q5_FEATURES:
        direction, auc = _feature_auc_direction(
            feature=feature,
            prediction=valid_a_prediction,
            features=valid_a_features,
            severe_threshold=severe_threshold,
        )
        valid_values = np.asarray(valid_a_features[feature], dtype=np.float64)
        for q in RULE_QS:
            threshold = _threshold_for_q5(valid_values, q5_valid, q, direction)
            warning = _apply_threshold_rule(feature, direction, threshold, features)
            metrics = cp167.pool_metrics(
                prediction=prediction,
                pool_mask=q5,
                warning_mask=warning,
                severe_threshold=severe_threshold,
                bottom_mask=bottom,
            )
            rows.append(
                {
                    "split": split,
                    "rule_id": f"{feature}_{direction}_q{int(q * 100)}",
                    "feature": feature,
                    "direction": direction,
                    "q": q,
                    "threshold": threshold,
                    "valid_a_auc": auc,
                    **metrics,
                }
            )
    return rows


def select_q5_threshold_rule(
    valid_a_rows: list[dict[str, Any]], valid_b_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    valid_b_by_id = {row["rule_id"]: row for row in valid_b_rows}
    base = [
        row
        for row in valid_a_rows
        if (row.get("warning_share") or 0.0) >= 0.05
        and (row.get("false_safe_reduction") or 0.0) > 0
        and (row.get("severe_lift_removed_vs_kept") or 0.0) > 1.0
    ]
    primary = [
        row
        for row in base
        if (row.get("false_safe_reduction") or 0.0) >= 0.015
        and (row.get("spread_retention") or 0.0) >= 0.80
        and (row.get("fee_retention") or 0.0) >= 0.80
    ]
    for pool, reason in (
        (primary, "valid_a_product_shape_and_valid_b_positive"),
        (base, "valid_a_risk_positive_and_valid_b_positive"),
    ):
        for row in sorted(pool, key=_row_score, reverse=True):
            valid_b = valid_b_by_id.get(row["rule_id"])
            if not valid_b:
                continue
            if (valid_b.get("false_safe_reduction") or 0.0) > 0 and (
                valid_b.get("severe_lift_removed_vs_kept") or 0.0
            ) > 1.0:
                selected = dict(row)
                selected["selection_reason"] = reason
                selected["valid_b_false_safe_reduction"] = valid_b.get("false_safe_reduction")
                selected["valid_b_severe_lift_removed_vs_kept"] = valid_b.get(
                    "severe_lift_removed_vs_kept"
                )
                selected["valid_b_spread_retention"] = valid_b.get("spread_retention")
                selected["valid_b_fee_retention"] = valid_b.get("fee_retention")
                return selected
    if base:
        selected = dict(sorted(base, key=_row_score, reverse=True)[0])
        selected["selection_reason"] = "valid_a_only_valid_b_not_confirmed"
        return selected
    return None


def _fixed_component_masks(
    *,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    valid_a_prediction: dict[str, Any],
    valid_a_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _top_valid, _bottom_valid, q5_valid = _top_q5(valid_a_prediction)

    def high(feature: str, q: float) -> np.ndarray:
        threshold = cp167.threshold_from_values(
            np.asarray(valid_a_features[feature], dtype=np.float64)[q5_valid], q
        )
        return np.asarray(features[feature], dtype=np.float64) >= threshold

    def low(feature: str, q: float) -> np.ndarray:
        threshold = cp167.threshold_from_values(
            np.asarray(valid_a_features[feature], dtype=np.float64)[q5_valid], 1.0 - q
        )
        return np.asarray(features[feature], dtype=np.float64) <= threshold

    return {
        "intraday_range_q80": high("intraday_range_20d_mean", 0.80),
        "intraday_range_q85": high("intraday_range_20d_mean", 0.85),
        "volume_z_q70": high("volume_z_20_252", 0.70),
        "drawdown_5d_q60": high("drawdown_from_5d_high", 0.60),
        "drawdown_5d_q70": high("drawdown_from_5d_high", 0.70),
        "close_position_5d_weak": low("close_position_5d_mean", 0.60),
    }


def build_combo_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    valid_a_prediction: dict[str, Any],
    valid_a_features: dict[str, np.ndarray],
    severe_threshold: float,
    selected_rule_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    _top, bottom, q5 = _top_q5(prediction)
    combos = combo_warning_masks(
        prediction=prediction,
        features=features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    rows: list[dict[str, Any]] = []
    for rule_id, warning in combos.items():
        if selected_rule_ids is not None and split == "test" and rule_id not in selected_rule_ids:
            continue
        metrics = cp167.pool_metrics(
            prediction=prediction,
            pool_mask=q5,
            warning_mask=warning,
            severe_threshold=severe_threshold,
            bottom_mask=bottom,
        )
        rows.append({"split": split, "rule_id": rule_id, **metrics})
    return rows


def combo_warning_masks(
    *,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    valid_a_prediction: dict[str, Any],
    valid_a_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    masks = _fixed_component_masks(
        prediction=prediction,
        features=features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    vote_count = (
        masks["intraday_range_q80"].astype(int)
        + masks["drawdown_5d_q60"].astype(int)
        + masks["volume_z_q70"].astype(int)
        + masks["close_position_5d_weak"].astype(int)
    )
    return {
        "intraday_range_q80_AND_volume_z_q70": masks["intraday_range_q80"] & masks["volume_z_q70"],
        "intraday_range_q80_AND_drawdown_5d_q60": masks["intraday_range_q80"]
        & masks["drawdown_5d_q60"],
        "intraday_range_q80_AND_close_position_5d_weak": masks["intraday_range_q80"]
        & masks["close_position_5d_weak"],
        "intraday_range_q85_OR_drawdown_5d_q70": masks["intraday_range_q85"]
        | masks["drawdown_5d_q70"],
        "vote_2plus_intraday_drawdown_volume_closeweak": vote_count >= 2,
    }


def select_combo_rules(
    valid_a_rows: list[dict[str, Any]], valid_b_rows: list[dict[str, Any]], limit: int = 3
) -> list[str]:
    valid_b_by_id = {row["rule_id"]: row for row in valid_b_rows}
    eligible = [
        row
        for row in valid_a_rows
        if (row.get("warning_share") or 0.0) >= 0.05
        and (row.get("false_safe_reduction") or 0.0) > 0
        and (row.get("severe_lift_removed_vs_kept") or 0.0) > 1.0
    ]
    confirmed = [
        row
        for row in eligible
        if (valid_b_by_id.get(row["rule_id"], {}).get("false_safe_reduction") or 0.0) > 0
        and (valid_b_by_id.get(row["rule_id"], {}).get("severe_lift_removed_vs_kept") or 0.0) > 1.0
    ]
    pool = confirmed or eligible
    return [row["rule_id"] for row in sorted(pool, key=_row_score, reverse=True)[:limit]]


def select_overall_q5_rule(
    *,
    threshold_valid_a: list[dict[str, Any]],
    threshold_valid_b: list[dict[str, Any]],
    combo_valid_a: list[dict[str, Any]],
    combo_valid_b: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    valid_b_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in threshold_valid_b:
        valid_b_by_key[("threshold", str(row["rule_id"]))] = row
    for row in combo_valid_b:
        valid_b_by_key[("combo", str(row["rule_id"]))] = row
    for source, rows in (("threshold", threshold_valid_a), ("combo", combo_valid_a)):
        for row in rows:
            if (row.get("warning_share") or 0.0) < 0.05:
                continue
            if (row.get("false_safe_reduction") or 0.0) <= 0:
                continue
            if (row.get("severe_lift_removed_vs_kept") or 0.0) <= 1.0:
                continue
            valid_b = valid_b_by_key.get((source, str(row["rule_id"])), {})
            valid_b_positive = (valid_b.get("false_safe_reduction") or 0.0) > 0 and (
                valid_b.get("severe_lift_removed_vs_kept") or 0.0
            ) > 1.0
            valid_b_product_shape = (
                valid_b_positive
                and (valid_b.get("spread_retention") or 0.0) >= 0.80
                and (valid_b.get("fee_retention") or 0.0) >= 0.80
            )
            enriched = dict(row)
            enriched["rule_type"] = source
            enriched["valid_b_positive"] = bool(valid_b_positive)
            enriched["valid_b_product_shape"] = bool(valid_b_product_shape)
            enriched["valid_b_false_safe_reduction"] = valid_b.get("false_safe_reduction")
            enriched["valid_b_severe_lift_removed_vs_kept"] = valid_b.get(
                "severe_lift_removed_vs_kept"
            )
            enriched["valid_b_spread_retention"] = valid_b.get("spread_retention")
            enriched["valid_b_fee_retention"] = valid_b.get("fee_retention")
            if (
                (row.get("false_safe_reduction") or 0.0) >= 0.015
                and (row.get("spread_retention") or 0.0) >= 0.80
                and (row.get("fee_retention") or 0.0) >= 0.80
            ):
                enriched["selection_tier"] = "product_shape"
            else:
                enriched["selection_tier"] = "risk_positive"
            candidates.append(enriched)
    confirmed = [row for row in candidates if row.get("valid_b_positive")]
    pool = confirmed or candidates
    if not pool:
        return None
    selected = sorted(
        pool,
        key=lambda row: (
            1.0 if row.get("selection_tier") == "product_shape" else 0.0,
            1.0 if row.get("valid_b_product_shape") else 0.0,
            1.0 if row.get("valid_b_positive") else 0.0,
            *_row_score(row),
        ),
        reverse=True,
    )[0]
    if selected.get("valid_b_product_shape"):
        valid_b_state = "product_shape"
    elif selected.get("valid_b_positive"):
        valid_b_state = "positive"
    else:
        valid_b_state = "not_confirmed"
    selected["selection_reason"] = (
        f"{selected.get('rule_type')}_{selected.get('selection_tier')}_valid_b_{valid_b_state}"
    )
    return selected


def selected_rule_mask(
    row: dict[str, Any] | None,
    features: dict[str, np.ndarray],
    *,
    prediction: dict[str, Any] | None = None,
    valid_a_prediction: dict[str, Any] | None = None,
    valid_a_features: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if not row:
        return np.zeros(len(next(iter(features.values()))), dtype=bool)
    if row.get("rule_type") == "combo":
        if prediction is None or valid_a_prediction is None or valid_a_features is None:
            raise ValueError("combo rule mask에는 prediction과 valid_a 정보가 필요합니다.")
        masks = combo_warning_masks(
            prediction=prediction,
            features=features,
            valid_a_prediction=valid_a_prediction,
            valid_a_features=valid_a_features,
        )
        return masks[str(row["rule_id"])]
    return _apply_threshold_rule(
        str(row["feature"]), str(row["direction"]), float(row["threshold"]), features
    )


def two_tier_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    base_warning: np.ndarray,
    q5_warning: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, bottom, q5 = _top_q5(prediction)
    masks = {
        "base_rule_only": base_warning & top,
        "q5_rule_only": q5_warning & q5,
        "two_tier_rule": (q5 & q5_warning) | (top & ~q5 & base_warning),
    }
    removed_count = int(masks["two_tier_rule"].sum())
    top_indices = np.flatnonzero(top)
    random_mask = np.zeros(len(line_score), dtype=bool)
    if removed_count > 0 and len(top_indices) > 0:
        rng = np.random.default_rng(168)
        random_mask[
            rng.choice(top_indices, size=min(removed_count, len(top_indices)), replace=False)
        ] = True
    trim_mask = np.zeros(len(line_score), dtype=bool)
    if removed_count > 0:
        order = top_indices[np.argsort(line_score[top_indices])]
        trim_mask[order[: min(removed_count, len(order))]] = True
    masks["random_matched_warning"] = random_mask
    masks["line_score_trim_matched_warning"] = trim_mask
    rows: list[dict[str, Any]] = []
    for rule_id, warning in masks.items():
        full = cp167.pool_metrics(
            prediction=prediction,
            pool_mask=top,
            warning_mask=warning,
            severe_threshold=severe_threshold,
            bottom_mask=bottom,
        )
        q5_metrics = cp167.pool_metrics(
            prediction=prediction,
            pool_mask=q5,
            warning_mask=warning,
            severe_threshold=severe_threshold,
            bottom_mask=bottom,
        )
        rows.append(
            {
                "split": split,
                "rule_id": rule_id,
                **full,
                "q5_false_safe_reduction": q5_metrics.get("false_safe_reduction"),
                "q5_severe_lift_removed_vs_kept": q5_metrics.get("severe_lift_removed_vs_kept"),
                "q5_warning_share": q5_metrics.get("warning_share"),
                "q5_removed_severe_rate": q5_metrics.get("removed_severe_rate"),
                "q5_kept_severe_rate": q5_metrics.get("kept_severe_rate"),
                "q5_spread_retention": q5_metrics.get("spread_retention"),
                "q5_fee_retention": q5_metrics.get("fee_retention"),
            }
        )
    return rows


def _prediction_with_line(prediction: dict[str, Any], line_score: np.ndarray) -> dict[str, Any]:
    result = dict(prediction)
    result["line_score"] = np.asarray(line_score, dtype=np.float64)
    return result


def _scaled_penalty(values: np.ndarray, reference_values: np.ndarray, direction: str) -> np.ndarray:
    finite = reference_values[np.isfinite(reference_values)]
    if len(finite) < 10:
        return np.zeros_like(values, dtype=np.float64)
    low = float(np.quantile(finite, 0.05))
    high = float(np.quantile(finite, 0.95))
    if abs(high - low) < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    scaled = (values - low) / (high - low)
    if direction == "low":
        scaled = 1.0 - scaled
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def continuous_adjustment_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    valid_a_prediction: dict[str, Any],
    valid_a_features: dict[str, np.ndarray],
    base_warning: np.ndarray,
    q5_rule: dict[str, Any] | None,
    q5_warning_mask: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, _bottom, q5 = _top_q5(prediction)
    valid_top, _valid_bottom, valid_q5 = _top_q5(valid_a_prediction)
    raw = cp160.line_alpha_metrics("cp168_raw_line", split, prediction, severe_threshold)
    penalty = np.zeros(len(line_score), dtype=np.float64)
    penalty[top & ~q5 & base_warning] = 1.0
    if q5_rule and q5_rule.get("rule_type") != "combo":
        feature = str(q5_rule["feature"])
        direction = str(q5_rule["direction"])
        values = np.asarray(features[feature], dtype=np.float64)
        ref = np.asarray(valid_a_features[feature], dtype=np.float64)[valid_q5]
        penalty[q5] = _scaled_penalty(values[q5], ref, direction)
    elif q5_rule:
        penalty[q5 & q5_warning_mask] = 1.0
    rows: list[dict[str, Any]] = []
    for lambda_value in LAMBDA_VALUES:
        adjusted_score = line_score * (1.0 - lambda_value * penalty)
        adjusted_prediction = _prediction_with_line(prediction, adjusted_score)
        metrics = cp160.line_alpha_metrics(
            f"cp168_adjusted_lambda_{lambda_value}", split, adjusted_prediction, severe_threshold
        )
        rows.append(
            {
                "split": split,
                "lambda": lambda_value,
                "raw_ic_mean": raw.get("ic_mean"),
                "raw_spread": raw.get("long_short_spread"),
                "raw_fee": raw.get("fee_adjusted_return"),
                "raw_top_decile_false_safe_rate": raw.get("line_top_decile_false_safe_rate"),
                "adjusted_ic_mean": metrics.get("ic_mean"),
                "adjusted_spread": metrics.get("long_short_spread"),
                "adjusted_fee": metrics.get("fee_adjusted_return"),
                "adjusted_top_decile_false_safe_rate": metrics.get(
                    "line_top_decile_false_safe_rate"
                ),
                "false_safe_reduction_vs_raw": None
                if raw.get("line_top_decile_false_safe_rate") is None
                or metrics.get("line_top_decile_false_safe_rate") is None
                else float(
                    raw["line_top_decile_false_safe_rate"]
                    - metrics["line_top_decile_false_safe_rate"]
                ),
                "spread_retention_vs_raw_line": None
                if not raw.get("long_short_spread")
                else float((metrics.get("long_short_spread") or 0.0) / raw["long_short_spread"]),
                "fee_retention_vs_raw_line": None
                if not raw.get("fee_adjusted_return")
                else float(
                    (metrics.get("fee_adjusted_return") or 0.0) / raw["fee_adjusted_return"]
                ),
                "penalty_nonzero_share_top_decile": _safe_ratio(
                    int((top & (penalty > 0)).sum()), int(top.sum())
                ),
            }
        )
    return rows


def classify_final(
    *,
    two_tier_test: dict[str, Any],
    base_test: dict[str, Any],
    random_test: dict[str, Any],
    trim_test: dict[str, Any],
    continuous_test_rows: list[dict[str, Any]],
    sliding_rows: list[dict[str, Any]],
) -> str:
    q5_positive = (two_tier_test.get("q5_false_safe_reduction") or 0.0) > 0 and (
        two_tier_test.get("q5_severe_lift_removed_vs_kept") or 0.0
    ) > 1.0
    retention_ok = (two_tier_test.get("spread_retention") or 0.0) >= 0.80 and (
        two_tier_test.get("fee_retention") or 0.0
    ) >= 0.80
    baseline_ok = (two_tier_test.get("false_safe_reduction") or 0.0) > (
        random_test.get("false_safe_reduction") or -999.0
    ) and (two_tier_test.get("false_safe_reduction") or 0.0) > (
        trim_test.get("false_safe_reduction") or -999.0
    )
    windows = [
        row
        for row in sliding_rows
        if row.get("rule_id") == "two_tier_rule"
        and row.get("false_safe_reduction") is not None
        and row.get("spread_retention") is not None
    ]
    bad_windows = [
        row
        for row in windows
        if (row.get("false_safe_reduction") or 0.0) <= 0
        or (row.get("spread_retention") or 0.0) < 0.70
    ]
    catastrophic_windows = [
        row
        for row in windows
        if (row.get("spread_retention") or 0.0) < 0.50 or (row.get("fee_retention") or 0.0) < 0.50
    ]
    window_ok = bool(
        windows and not catastrophic_windows and len(bad_windows) / len(windows) <= 0.25
    )
    continuous_candidates = [
        row
        for row in continuous_test_rows
        if (row.get("false_safe_reduction_vs_raw") or 0.0) > 0
        and (row.get("spread_retention_vs_raw_line") or 0.0) >= 0.80
        and (row.get("fee_retention_vs_raw_line") or 0.0) >= 0.80
    ]
    if q5_positive and retention_ok and baseline_ok and window_ok:
        return "TWO_TIER_WARNING_BETA_CANDIDATE"
    if q5_positive and baseline_ok:
        return "TWO_TIER_RESEARCH_WARNING"
    if continuous_candidates:
        return "CONTINUOUS_ADJUSTMENT_RESEARCH"
    if (base_test.get("false_safe_reduction") or 0.0) > 0 and (
        base_test.get("spread_retention") or 0.0
    ) >= 0.80:
        return "BASE_RULE_ONLY_KEEP"
    if q5_positive or (two_tier_test.get("false_safe_reduction") or 0.0) > 0:
        return "WEAK_SIGNAL"
    return "REJECT"


def _best_by_rule(rows: list[dict[str, Any]], split: str, rule_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("split") == split and row.get("rule_id") == rule_id:
            return row
    return {}


def write_report(payload: dict[str, Any]) -> None:
    selected = payload.get("selected_q5_rule") or {}
    combo_ids = ", ".join(payload.get("selected_combo_rule_ids") or []) or "없음"
    two_tier_test = payload.get("two_tier_test") or {}
    base_test = payload.get("base_test") or {}
    best_cont = payload.get("best_continuous_test") or {}
    lines = [
        "# CP168-LM 1D Cohort-Aware Warning Development",
        "",
        "## 한 줄 결론",
        f"- 최종 판정: **{payload['final_label']}**",
        "- 이번 CP는 warning을 억지로 제품에 붙이기 위한 실험이 아니라, line top 후보군 안에서 risk reason이 cohort별로 달라지는지 검증한 개발 실험이다.",
        "",
        "## 역할 재정의",
        "- line = 수익성/순위 신호",
        "- warning = line 해석 시 주의해야 하는 규칙 기반 risk overlay",
        "- band = 별도 가격 범위 예측",
        "- CP154 계획은 line/risk를 딥러닝 multi-head로 분리하는 방향이었으나, CP154~CP167 결과 risk head는 제품 신호로 충분하지 않았고, ATR/volatility/intraday 기반 규칙형 overlay가 더 설명 가능한 risk 신호로 나타났다.",
        "- 따라서 Phase 1에서는 line은 딥러닝 수익성 신호로 유지하고, warning은 검증된 규칙 기반 risk overlay로 분리해 개발한다.",
        "",
        "## 금지 작업 준수",
        "- 새 딥러닝 학습 없음",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch / EODHD fallback 없음",
        "- composite 실행 없음",
        "- CP153 band artifact 변경 없음",
        "- ticker 전용 rule / 문제 ticker 전용 rule 없음",
        "",
        "## Stage 0 Preflight",
        f"- split_mode: `{payload['split_metadata'].get('split_mode')}`",
        f"- cross_split_date_overlap_count: `{payload['split_metadata'].get('cross_split_date_overlap_count')}`",
        f"- validation q5 sample: `{payload['preflight'].get('validation_q5_count')}`",
        f"- test q5 sample: `{payload['preflight'].get('test_q5_count')}`",
        f"- test q5 base severe rate: `{_fmt(payload['preflight'].get('test_q5_severe_rate'))}`",
        f"- test q5 base rule warning share: `{_fmt(payload['preflight'].get('test_base_rule_q5_warning_share'))}`",
        "",
        "## Stage 1 q5 Rule 정밀화",
        f"- 선택 q5 threshold rule: `{selected.get('rule_id')}`",
        f"- 선택 이유: `{selected.get('selection_reason')}`",
        f"- rule_type: `{selected.get('rule_type')}`",
        f"- feature: `{selected.get('feature', 'combo')}`, direction: `{selected.get('direction', 'combo')}`, threshold: `{_fmt(selected.get('threshold'))}`",
        f"- valid-B false-safe 감소 / severe lift: `{_fmt(selected.get('valid_b_false_safe_reduction'))}` / `{_fmt(selected.get('valid_b_severe_lift_removed_vs_kept'))}`",
        "",
        "## Stage 2 q5 결합 Rule",
        f"- valid-A/valid-B를 통과해 test에 올린 결합 rule: `{combo_ids}`",
        "- 결합 후보가 많아질수록 multiple testing 위험이 있으므로, test 최종 비교는 valid-A와 valid-B에서 유지된 상위 후보만 기록했다.",
        "",
        "## Stage 3 2-tier Warning 비교",
        "| rule | FS 감소 | severe lift | spread retention | fee retention | q5 FS 감소 | q5 severe lift | warning share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["two_tier_rows"]:
        if row["split"] != "test":
            continue
        lines.append(
            f"| {row['rule_id']} | {_fmt(row.get('false_safe_reduction'))} | {_fmt(row.get('severe_lift_removed_vs_kept'))} | "
            f"{_fmt(row.get('spread_retention'))} | {_fmt(row.get('fee_retention'))} | "
            f"{_fmt(row.get('q5_false_safe_reduction'))} | {_fmt(row.get('q5_severe_lift_removed_vs_kept'))} | {_fmt(row.get('warning_share'))} |"
        )
    lines.extend(
        [
            "",
            "## Stage 4 Continuous Adjustment 보조 평가",
            f"- best lambda: `{best_cont.get('lambda')}`",
            f"- adjusted false-safe 감소: `{_fmt(best_cont.get('false_safe_reduction_vs_raw'))}`",
            f"- adjusted spread/fee retention: `{_fmt(best_cont.get('spread_retention_vs_raw_line'))}` / `{_fmt(best_cont.get('fee_retention_vs_raw_line'))}`",
            "- continuous adjustment는 설명 난도가 높으므로 제품 적용 후보가 아니라 binary warning의 비용을 비교하는 진단으로만 봤다.",
            "",
            "## 필수 질문 답변",
            "1. q5에서는 왜 ATR/self rule이 약했는가?",
            "   - q5는 line score가 가장 높은 구간이라 ATR이 위험만 뜻하지 않고 고수익 변동성도 함께 잡았다. 그래서 base ATR/self warning은 q5에서 severe를 안정적으로 가르지 못했다.",
            "2. intraday/microstructure feature가 q5에서 더 나은가?",
            f"   - q5 threshold sweep과 결합 rule 결과는 `{Q5_SWEEP_CSV.name}`, `{Q5_COMBO_CSV.name}`에 기록했다. 최종 test에서는 선택 rule의 q5 FS 감소와 severe lift가 핵심이다.",
            "3. q5 rule은 시간적으로 안정적인가?",
            f"   - sliding window 결과는 `{SLIDING_CSV.name}`에 기록했다. final label은 이 안정성까지 반영했다.",
            "4. 2-tier warning이 base_rule 단독보다 나은가?",
            f"   - base test FS 감소/retention은 `{_fmt(base_test.get('false_safe_reduction'))}` / `{_fmt(base_test.get('spread_retention'))}`이고, two-tier는 `{_fmt(two_tier_test.get('false_safe_reduction'))}` / `{_fmt(two_tier_test.get('spread_retention'))}`이다.",
            "5. continuous adjustment가 더 좋은 대안인가?",
            f"   - best continuous false-safe 감소/retention은 `{_fmt(best_cont.get('false_safe_reduction_vs_raw'))}` / `{_fmt(best_cont.get('spread_retention_vs_raw_line'))}`이다. 좋은 숫자가 나와도 설명성 때문에 연구 보조로만 둔다.",
            "6. 제품에는 단일 warning on/off로 충분한가, 아니면 reason label이 필요한가?",
            "   - Phase 1 프론트 1차 적용은 warning on/off가 더 단순하다. 다만 상세 패널에는 `고변동 주의`, `장중 불안정 주의` 같은 reason label을 노출하는 편이 해석 안전성이 높다.",
            "",
            "## 산출물",
            f"- report: `{REPORT_PATH}`",
            f"- metrics: `{METRICS_PATH}`",
            f"- summary: `{SUMMARY_CSV}`",
            f"- q5 threshold sweep: `{Q5_SWEEP_CSV}`",
            f"- q5 combination rule: `{Q5_COMBO_CSV}`",
            f"- two-tier warning comparison: `{TWO_TIER_CSV}`",
            f"- continuous adjustment: `{CONTINUOUS_CSV}`",
            f"- sliding window: `{SLIDING_CSV}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cp153_before = cp159.cp153_artifact_state()
    cp164_payload = cp165.load_cp164_reference()
    val_prediction, test_prediction, val_features, test_features, split_summary, thresholds = (
        cp165.build_predictions(cp164_payload)
    )
    severe_threshold = float(thresholds[0])
    if (
        split_summary.get("split_mode") != "calendar_aligned"
        or int(split_summary.get("cross_split_date_overlap_count") or 0) != 0
    ):
        raise RuntimeError(f"calendar split preflight 실패: {split_summary}")

    price, _indicators, _pm, _im = cp165.cp164.cp158.load_source_frames()
    price_feature_frame = cp167.compute_price_feature_frame(price)
    val_features = {**val_features, **cp167.map_price_features(val_prediction, price_feature_frame)}
    test_features = {
        **test_features,
        **cp167.map_price_features(test_prediction, price_feature_frame),
    }

    partitions = cp167.validation_partitions(val_prediction)
    valid_a_prediction, valid_a_features = cp167.subset_prediction(
        val_prediction, partitions["valid_a"], val_features
    )
    valid_b_prediction, valid_b_features = cp167.subset_prediction(
        val_prediction, partitions["valid_b"], val_features
    )

    base_val = cp167.base_warning_mask(val_features, val_features)
    base_test = cp167.base_warning_mask(test_features, val_features)
    base_valid_a = cp167.base_warning_mask(valid_a_features, val_features)
    base_valid_b = cp167.base_warning_mask(valid_b_features, val_features)

    q5_valid_a_sweep = build_q5_threshold_rows(
        split="valid_a",
        prediction=valid_a_prediction,
        features=valid_a_features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
        severe_threshold=severe_threshold,
    )
    q5_valid_b_sweep = build_q5_threshold_rows(
        split="valid_b",
        prediction=valid_b_prediction,
        features=valid_b_features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
        severe_threshold=severe_threshold,
    )
    q5_test_sweep = build_q5_threshold_rows(
        split="test",
        prediction=test_prediction,
        features=test_features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
        severe_threshold=severe_threshold,
    )
    q5_sweep_rows = q5_valid_a_sweep + q5_valid_b_sweep + q5_test_sweep

    combo_valid_a = build_combo_rows(
        split="valid_a",
        prediction=valid_a_prediction,
        features=valid_a_features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
        severe_threshold=severe_threshold,
    )
    combo_valid_b = build_combo_rows(
        split="valid_b",
        prediction=valid_b_prediction,
        features=valid_b_features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
        severe_threshold=severe_threshold,
    )
    selected_combo_ids = set(select_combo_rules(combo_valid_a, combo_valid_b, limit=3))
    selected_q5_rule = select_overall_q5_rule(
        threshold_valid_a=q5_valid_a_sweep,
        threshold_valid_b=q5_valid_b_sweep,
        combo_valid_a=combo_valid_a,
        combo_valid_b=combo_valid_b,
    )
    if selected_q5_rule and selected_q5_rule.get("rule_type") == "combo":
        selected_combo_ids.add(str(selected_q5_rule["rule_id"]))
    combo_test = build_combo_rows(
        split="test",
        prediction=test_prediction,
        features=test_features,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
        severe_threshold=severe_threshold,
        selected_rule_ids=selected_combo_ids,
    )
    combo_rows = combo_valid_a + combo_valid_b + combo_test

    q5_warning_val = selected_rule_mask(
        selected_q5_rule,
        val_features,
        prediction=val_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    q5_warning_test = selected_rule_mask(
        selected_q5_rule,
        test_features,
        prediction=test_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )

    two_tier_rows_all: list[dict[str, Any]] = []
    two_tier_rows_all.extend(
        two_tier_rows(
            split="validation",
            prediction=val_prediction,
            base_warning=base_val,
            q5_warning=q5_warning_val,
            severe_threshold=severe_threshold,
        )
    )
    two_tier_rows_all.extend(
        two_tier_rows(
            split="test",
            prediction=test_prediction,
            base_warning=base_test,
            q5_warning=q5_warning_test,
            severe_threshold=severe_threshold,
        )
    )

    continuous_rows: list[dict[str, Any]] = []
    continuous_rows.extend(
        continuous_adjustment_rows(
            split="validation",
            prediction=val_prediction,
            features=val_features,
            valid_a_prediction=valid_a_prediction,
            valid_a_features=valid_a_features,
            base_warning=base_val,
            q5_rule=selected_q5_rule,
            q5_warning_mask=q5_warning_val,
            severe_threshold=severe_threshold,
        )
    )
    continuous_rows.extend(
        continuous_adjustment_rows(
            split="test",
            prediction=test_prediction,
            features=test_features,
            valid_a_prediction=valid_a_prediction,
            valid_a_features=valid_a_features,
            base_warning=base_test,
            q5_rule=selected_q5_rule,
            q5_warning_mask=q5_warning_test,
            severe_threshold=severe_threshold,
        )
    )

    sliding_rows = cp167.sliding_window_rows(
        prediction=test_prediction,
        base_warning=base_test,
        q5_warning=q5_warning_test,
        severe_threshold=severe_threshold,
    )

    test_two_tier = _best_by_rule(two_tier_rows_all, "test", "two_tier_rule")
    test_base = _best_by_rule(two_tier_rows_all, "test", "base_rule_only")
    test_random = _best_by_rule(two_tier_rows_all, "test", "random_matched_warning")
    test_trim = _best_by_rule(two_tier_rows_all, "test", "line_score_trim_matched_warning")
    test_continuous = [row for row in continuous_rows if row["split"] == "test"]
    best_continuous_test = sorted(
        test_continuous,
        key=lambda row: (
            row.get("false_safe_reduction_vs_raw") or -999.0,
            row.get("spread_retention_vs_raw_line") or -999.0,
            row.get("fee_retention_vs_raw_line") or -999.0,
        ),
        reverse=True,
    )[0]
    final_label = classify_final(
        two_tier_test=test_two_tier,
        base_test=test_base,
        random_test=test_random,
        trim_test=test_trim,
        continuous_test_rows=test_continuous,
        sliding_rows=sliding_rows,
    )

    _top_val, _bottom_val, q5_val = _top_q5(val_prediction)
    _top_test, _bottom_test, q5_test = _top_q5(test_prediction)
    actual_test = np.asarray(test_prediction["actual"], dtype=np.float64)
    preflight = {
        "validation_top_decile_count": int(_top_val.sum()),
        "validation_q5_count": int(q5_val.sum()),
        "test_top_decile_count": int(_top_test.sum()),
        "test_q5_count": int(q5_test.sum()),
        "test_q5_severe_rate": _safe_ratio(
            int((q5_test & (actual_test <= severe_threshold)).sum()), int(q5_test.sum())
        ),
        "test_base_rule_q5_warning_share": _safe_ratio(
            int((q5_test & base_test).sum()), int(q5_test.sum())
        ),
    }
    cp153_after = cp159.cp153_artifact_state()
    payload = {
        "cp": CP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_label": final_label,
        "candidate": "patchtst_line_regime_p32_s16",
        "base_rule": BASE_RULE_ID,
        "split_metadata": split_summary,
        "preflight": preflight,
        "severe_threshold": severe_threshold,
        "selected_q5_rule": selected_q5_rule,
        "selected_combo_rule_ids": sorted(selected_combo_ids),
        "two_tier_test": test_two_tier,
        "base_test": test_base,
        "random_test": test_random,
        "trim_test": test_trim,
        "best_continuous_test": best_continuous_test,
        "q5_threshold_sweep_rows": q5_sweep_rows,
        "q5_combination_rows": combo_rows,
        "two_tier_rows": two_tier_rows_all,
        "continuous_rows": continuous_rows,
        "sliding_rows": sliding_rows,
        "cp153_artifact_unchanged": cp153_before == cp153_after,
        "forbidden_work": {
            "new_deep_learning_training": False,
            "product_save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "composite": False,
            "cp153_band_artifact_modified": cp153_before != cp153_after,
        },
    }

    summary_rows = [
        {
            "final_label": final_label,
            "selected_q5_rule_id": (selected_q5_rule or {}).get("rule_id"),
            "selected_q5_feature": (selected_q5_rule or {}).get("feature"),
            "selected_q5_direction": (selected_q5_rule or {}).get("direction"),
            "selected_q5_threshold": (selected_q5_rule or {}).get("threshold"),
            "two_tier_test_false_safe_reduction": test_two_tier.get("false_safe_reduction"),
            "two_tier_test_severe_lift": test_two_tier.get("severe_lift_removed_vs_kept"),
            "two_tier_test_spread_retention": test_two_tier.get("spread_retention"),
            "two_tier_test_fee_retention": test_two_tier.get("fee_retention"),
            "two_tier_test_q5_false_safe_reduction": test_two_tier.get("q5_false_safe_reduction"),
            "two_tier_test_q5_severe_lift": test_two_tier.get("q5_severe_lift_removed_vs_kept"),
            "base_test_false_safe_reduction": test_base.get("false_safe_reduction"),
            "base_test_spread_retention": test_base.get("spread_retention"),
            "best_continuous_lambda": best_continuous_test.get("lambda"),
            "best_continuous_false_safe_reduction": best_continuous_test.get(
                "false_safe_reduction_vs_raw"
            ),
            "best_continuous_spread_retention": best_continuous_test.get(
                "spread_retention_vs_raw_line"
            ),
            "cp153_artifact_unchanged": cp153_before == cp153_after,
        }
    ]

    _write_json(METRICS_PATH, payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(Q5_SWEEP_CSV, q5_sweep_rows)
    _write_csv(Q5_COMBO_CSV, combo_rows)
    _write_csv(TWO_TIER_CSV, two_tier_rows_all)
    _write_csv(CONTINUOUS_CSV, continuous_rows)
    _write_csv(SLIDING_CSV, sliding_rows)
    write_report(payload)

    if cp159.torch.cuda.is_available():
        cp159.torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
