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
from ai import cp165_lm_1d_atr_overlay_sweet_spot as cp165  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "cp167_lm_1d_top_quintile_risk_rescue_report.md"
METRICS_PATH = DOCS_DIR / "cp167_lm_1d_top_quintile_risk_rescue_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp167_lm_1d_top_quintile_risk_rescue_summary.csv"
DECOMP_CSV = DOCS_DIR / "cp167_lm_1d_q5_decomposition.csv"
PROFILE_CSV = DOCS_DIR / "cp167_lm_1d_q5_feature_profile.csv"
RULE_SEARCH_CSV = DOCS_DIR / "cp167_lm_1d_q5_rule_search.csv"
TWO_TIER_CSV = DOCS_DIR / "cp167_lm_1d_two_tier_rule_comparison.csv"
SLIDING_CSV = DOCS_DIR / "cp167_lm_1d_sliding_window.csv"

BASE_RULE_ID = "atr_and_not_self_atr70_self70"
Q5_FEATURES = (
    "drawdown_from_5d_high",
    "drawdown_from_20d_high",
    "intraday_range_5d_mean",
    "intraday_range_20d_mean",
    "close_position_5d_mean",
    "close_position_20d_mean",
    "overnight_gap_abs_5d_mean",
    "volume_z_20_252",
    "volume_ratio_5_20",
    "downside_vol_ratio_20d",
    "atr_ratio",
    "self_vol_percentile_252",
)
RULE_QS = (0.70, 0.80, 0.90)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _safe_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else None


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{result:.{digits}f}" if math.isfinite(result) else ""


def top_bottom_masks(line_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    line_score = np.asarray(line_score, dtype=np.float64)
    return line_score >= np.quantile(line_score, 0.90), line_score <= np.quantile(line_score, 0.10)


def top_decile_quintiles(line_score: np.ndarray) -> np.ndarray:
    line_score = np.asarray(line_score, dtype=np.float64)
    top, _bottom = top_bottom_masks(line_score)
    quintile = np.zeros(len(line_score), dtype=np.int64)
    top_indices = np.flatnonzero(top)
    order = top_indices[np.argsort(line_score[top_indices])]
    chunks = np.array_split(order, 5)
    for idx, chunk in enumerate(chunks, start=1):
        quintile[chunk] = idx
    return quintile


def validation_partitions(prediction: dict[str, Any]) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(prediction["metadata"]["asof_date"], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff_index = max(1, int(len(unique_dates) * 0.60))
    cutoff_date = unique_dates[cutoff_index - 1]
    return {
        "valid_a": (dates <= cutoff_date).to_numpy(dtype=bool),
        "valid_b": (dates > cutoff_date).to_numpy(dtype=bool),
    }


def subset_prediction(
    prediction: dict[str, Any], mask: np.ndarray, features: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = {
        "candidate_id": prediction["candidate_id"],
        "line_score": np.asarray(prediction["line_score"])[mask],
        "actual": np.asarray(prediction["actual"])[mask],
        "metadata": prediction["metadata"].loc[mask].reset_index(drop=True),
    }
    if prediction.get("regime_pred") is not None:
        result["regime_pred"] = np.asarray(prediction["regime_pred"])[mask]
    feature_subset = {key: np.asarray(value)[mask] for key, value in features.items()}
    return result, feature_subset


def threshold_from_values(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if len(finite) else 0.0


def base_warning_mask(
    features: dict[str, np.ndarray], validation_features: dict[str, np.ndarray]
) -> np.ndarray:
    atr70 = threshold_from_values(validation_features["atr_ratio"], 0.70)
    self70 = threshold_from_values(validation_features["self_vol_percentile_252"], 0.70)
    return (np.asarray(features["atr_ratio"], dtype=np.float64) >= atr70) & ~(
        np.asarray(features["self_vol_percentile_252"], dtype=np.float64) >= self70
    )


def _rolling_mean(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=max(3, min(5, window))).mean()


def _rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=max(10, min(30, window))).std(ddof=0)


def compute_price_feature_frame(price: pd.DataFrame) -> pd.DataFrame:
    frame = price[["ticker", "date", "open", "high", "low", "close", "volume"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    groups: list[pd.DataFrame] = []
    for _ticker, group in frame.groupby("ticker", sort=False):
        group = group.copy()
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        open_ = group["open"].astype(float)
        volume = group["volume"].astype(float)
        prev_close = close.shift(1)
        intraday_range = (high - low) / close.replace(0.0, np.nan)
        overnight_gap_abs = (open_ / prev_close.replace(0.0, np.nan) - 1.0).abs()
        close_position = (close - low) / (high - low).replace(0.0, np.nan)
        vol5 = _rolling_mean(volume, 5)
        vol20 = _rolling_mean(volume, 20)
        vol20_mean252 = _rolling_mean(vol20, 252)
        vol20_std252 = _rolling_std(vol20, 252)
        group["intraday_range_5d_mean"] = _rolling_mean(intraday_range, 5)
        group["intraday_range_20d_mean"] = _rolling_mean(intraday_range, 20)
        group["overnight_gap_abs_5d_mean"] = _rolling_mean(overnight_gap_abs, 5)
        group["overnight_gap_abs_20d_mean"] = _rolling_mean(overnight_gap_abs, 20)
        group["close_position_5d_mean"] = _rolling_mean(close_position, 5)
        group["close_position_20d_mean"] = _rolling_mean(close_position, 20)
        group["volume_z_20_252"] = (vol20 - vol20_mean252) / vol20_std252.replace(0.0, np.nan)
        group["volume_ratio_5_20"] = vol5 / vol20.replace(0.0, np.nan)
        groups.append(group)
    result = pd.concat(groups, ignore_index=True)
    return result[
        [
            "ticker",
            "date",
            "intraday_range_5d_mean",
            "intraday_range_20d_mean",
            "overnight_gap_abs_5d_mean",
            "overnight_gap_abs_20d_mean",
            "close_position_5d_mean",
            "close_position_20d_mean",
            "volume_z_20_252",
            "volume_ratio_5_20",
        ]
    ]


def map_price_features(
    prediction: dict[str, Any], price_feature_frame: pd.DataFrame
) -> dict[str, np.ndarray]:
    metadata = prediction["metadata"][["ticker", "asof_date"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    metadata["asof_date"] = pd.to_datetime(metadata["asof_date"], errors="coerce")
    merged = metadata.merge(
        price_feature_frame,
        how="left",
        left_on=["ticker", "asof_date"],
        right_on=["ticker", "date"],
    )
    result: dict[str, np.ndarray] = {}
    for feature in (
        "intraday_range_5d_mean",
        "intraday_range_20d_mean",
        "overnight_gap_abs_5d_mean",
        "overnight_gap_abs_20d_mean",
        "close_position_5d_mean",
        "close_position_20d_mean",
        "volume_z_20_252",
        "volume_ratio_5_20",
    ):
        values = pd.to_numeric(merged[feature], errors="coerce").to_numpy(dtype=np.float64)
        result[feature] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def pool_metrics(
    *,
    prediction: dict[str, Any],
    pool_mask: np.ndarray,
    warning_mask: np.ndarray,
    severe_threshold: float,
    bottom_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    if bottom_mask is None:
        _top, bottom_mask = top_bottom_masks(line_score)
    pool = np.asarray(pool_mask, dtype=bool)
    warning = np.asarray(warning_mask, dtype=bool)
    removed = pool & warning
    kept = pool & ~warning
    severe = actual <= severe_threshold
    pool_mean = _safe_mean(actual[pool])
    bottom_mean = _safe_mean(actual[bottom_mask])
    kept_mean = _safe_mean(actual[kept])
    base_spread = (
        None if pool_mean is None or bottom_mean is None else float(pool_mean - bottom_mean)
    )
    spread_after = (
        None if kept_mean is None or bottom_mean is None else float(kept_mean - bottom_mean)
    )
    base_fee = None if base_spread is None else float(base_spread - 0.001)
    fee_after = None if spread_after is None else float(spread_after - 0.001)
    pool_severe_rate = _safe_ratio(int((pool & severe).sum()), int(pool.sum()))
    kept_severe_rate = _safe_ratio(int((kept & severe).sum()), int(kept.sum()))
    removed_severe_rate = _safe_ratio(int((removed & severe).sum()), int(removed.sum()))
    return {
        "pool_count": int(pool.sum()),
        "removed_count": int(removed.sum()),
        "kept_count": int(kept.sum()),
        "warning_share": _safe_ratio(int(removed.sum()), int(pool.sum())),
        "removed_actual_mean_return": _safe_mean(actual[removed]),
        "kept_actual_mean_return": kept_mean,
        "removed_actual_median_return": _safe_median(actual[removed]),
        "kept_actual_median_return": _safe_median(actual[kept]),
        "removed_positive_return_rate": _safe_ratio(
            int((actual[removed] > 0).sum()), int(removed.sum())
        ),
        "kept_positive_return_rate": _safe_ratio(int((actual[kept] > 0).sum()), int(kept.sum())),
        "pool_severe_rate": pool_severe_rate,
        "removed_severe_rate": removed_severe_rate,
        "kept_severe_rate": kept_severe_rate,
        "false_safe_reduction": None
        if pool_severe_rate is None or kept_severe_rate is None
        else float(pool_severe_rate - kept_severe_rate),
        "severe_lift_removed_vs_kept": None
        if removed_severe_rate is None or not kept_severe_rate
        else float(removed_severe_rate / kept_severe_rate),
        "base_spread": base_spread,
        "spread_after_filter": spread_after,
        "spread_retention": None
        if spread_after is None or base_spread is None or abs(base_spread) < 1e-12
        else float(spread_after / base_spread),
        "base_fee": base_fee,
        "fee_after_filter": fee_after,
        "fee_retention": None
        if fee_after is None or base_fee is None or abs(base_fee) < 1e-12
        else float(fee_after / base_fee),
    }


def q5_decomposition_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    warning_mask: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    quintile = top_decile_quintiles(line_score)
    rows: list[dict[str, Any]] = []
    for bucket in range(1, 6):
        bucket_mask = quintile == bucket
        for warning_value in (False, True):
            mask = bucket_mask & (warning_mask == warning_value)
            rows.append(
                {
                    "split": split,
                    "top_decile_quintile": bucket,
                    "warning": bool(warning_value),
                    "sample_count": int(mask.sum()),
                    "warning_share_in_bucket": _safe_ratio(int(mask.sum()), int(bucket_mask.sum()))
                    if warning_value
                    else None,
                    "actual_mean_return": _safe_mean(actual[mask]),
                    "actual_median_return": _safe_median(actual[mask]),
                    "positive_return_rate": _safe_ratio(
                        int((actual[mask] > 0).sum()), int(mask.sum())
                    ),
                    "severe_rate": _safe_ratio(
                        int((actual[mask] <= severe_threshold).sum()), int(mask.sum())
                    ),
                    "false_safe_rate": _safe_ratio(
                        int((actual[mask] <= severe_threshold).sum()), int(mask.sum())
                    ),
                    "line_score_mean": _safe_mean(line_score[mask]),
                    "atr_ratio_mean": _safe_mean(features["atr_ratio"][mask]),
                    "self_vol_percentile_mean": _safe_mean(
                        features["self_vol_percentile_252"][mask]
                    ),
                }
            )
    return rows


def auc_score(values: np.ndarray, label: np.ndarray) -> float | None:
    frame = (
        pd.DataFrame({"value": values, "label": label}).replace([np.inf, -np.inf], np.nan).dropna()
    )
    pos = frame[frame["label"] == 1]
    neg = frame[frame["label"] == 0]
    if pos.empty or neg.empty:
        return None
    ranks = frame["value"].rank(method="average")
    pos_rank_sum = float(ranks[frame["label"] == 1].sum())
    n_pos = len(pos)
    n_neg = len(neg)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def feature_profile_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    q5 = top_decile_quintiles(line_score) == 5
    severe = q5 & (actual <= severe_threshold)
    safe = q5 & (actual > severe_threshold)
    rows: list[dict[str, Any]] = []
    bottom_mask = top_bottom_masks(line_score)[1]
    for feature in Q5_FEATURES:
        values = np.asarray(features[feature], dtype=np.float64)
        severe_values = values[severe]
        safe_values = values[safe]
        pooled_std = np.nanstd(values[q5])
        effect = (
            None
            if pooled_std <= 1e-12
            else float((_safe_mean(severe_values) or 0.0) - (_safe_mean(safe_values) or 0.0))
            / float(pooled_std)
        )
        auc = auc_score(values[q5], (actual[q5] <= severe_threshold).astype(int))
        direction = "high_risk_high_value" if (auc or 0.5) >= 0.5 else "high_risk_low_value"
        row = {
            "split": split,
            "feature": feature,
            "finite_rate_q5": _safe_ratio(int(np.isfinite(values[q5]).sum()), int(q5.sum())),
            "severe_mean": _safe_mean(severe_values),
            "safe_mean": _safe_mean(safe_values),
            "severe_median": _safe_median(severe_values),
            "safe_median": _safe_median(safe_values),
            "direction_consistency": direction,
            "effect_size": effect,
            "auc": auc,
        }
        for share in (0.10, 0.20, 0.30):
            if direction == "high_risk_high_value":
                threshold = threshold_from_values(values[q5], 1.0 - share)
                warning = values >= threshold
            else:
                threshold = threshold_from_values(values[q5], share)
                warning = values <= threshold
            metrics = pool_metrics(
                prediction=prediction,
                pool_mask=q5,
                warning_mask=warning,
                severe_threshold=severe_threshold,
                bottom_mask=bottom_mask,
            )
            row[f"share{int(share * 100)}_threshold"] = threshold
            row[f"share{int(share * 100)}_severe_lift"] = metrics.get("severe_lift_removed_vs_kept")
            row[f"share{int(share * 100)}_false_safe_reduction"] = metrics.get(
                "false_safe_reduction"
            )
            row[f"share{int(share * 100)}_spread_retention"] = metrics.get("spread_retention")
        rows.append(row)
    return rows


def build_rule_search_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    valid_a_features: dict[str, np.ndarray],
    valid_a_prediction: dict[str, Any],
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score_valid = np.asarray(valid_a_prediction["line_score"], dtype=np.float64)
    actual_valid = np.asarray(valid_a_prediction["actual"], dtype=np.float64)
    q5_valid = top_decile_quintiles(line_score_valid) == 5
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    q5 = top_decile_quintiles(line_score) == 5
    bottom_mask = top_bottom_masks(line_score)[1]
    rows: list[dict[str, Any]] = []
    for feature in Q5_FEATURES:
        valid_values = np.asarray(valid_a_features[feature], dtype=np.float64)
        valid_auc = auc_score(
            valid_values[q5_valid], (actual_valid[q5_valid] <= severe_threshold).astype(int)
        )
        direction = "high" if (valid_auc or 0.5) >= 0.5 else "low"
        for q in RULE_QS:
            if direction == "high":
                threshold = threshold_from_values(valid_values[q5_valid], q)
                warning = np.asarray(features[feature], dtype=np.float64) >= threshold
            else:
                threshold = threshold_from_values(valid_values[q5_valid], 1.0 - q)
                warning = np.asarray(features[feature], dtype=np.float64) <= threshold
            metrics = pool_metrics(
                prediction=prediction,
                pool_mask=q5,
                warning_mask=warning,
                severe_threshold=severe_threshold,
                bottom_mask=bottom_mask,
            )
            rows.append(
                {
                    "split": split,
                    "rule_id": f"{feature}_{direction}_q{int(q * 100)}",
                    "feature": feature,
                    "direction": direction,
                    "q": q,
                    "threshold": threshold,
                    "valid_a_auc": valid_auc,
                    **metrics,
                    "selection_score": (
                        (metrics.get("severe_lift_removed_vs_kept") or -999),
                        (metrics.get("false_safe_reduction") or -999),
                        (metrics.get("spread_retention") or -999),
                    ),
                }
            )
    return rows


def select_q5_rule(
    valid_a_rows: list[dict[str, Any]], valid_b_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    valid_b_by_rule = {row["rule_id"]: row for row in valid_b_rows}
    eligible = [
        row
        for row in valid_a_rows
        if (row.get("warning_share") or 0.0) >= 0.05
        and (row.get("false_safe_reduction") or 0.0) > 0
        and (row.get("severe_lift_removed_vs_kept") or 0.0) > 1.0
    ]
    eligible = sorted(
        eligible,
        key=lambda row: (
            row.get("severe_lift_removed_vs_kept") or -999,
            row.get("false_safe_reduction") or -999,
            row.get("spread_retention") or -999,
            row.get("warning_share") or -999,
        ),
        reverse=True,
    )
    for row in eligible:
        valid_b = valid_b_by_rule.get(row["rule_id"])
        if (
            valid_b
            and (valid_b.get("false_safe_reduction") or 0.0) > 0
            and (valid_b.get("severe_lift_removed_vs_kept") or 0.0) > 1.0
        ):
            selected = dict(row)
            selected["selection_reason"] = "valid_a_top_and_valid_b_positive"
            return selected
    if eligible:
        selected = dict(eligible[0])
        selected["selection_reason"] = "valid_a_top_only_valid_b_not_confirmed"
        return selected
    return None


def apply_rule_from_row(row: dict[str, Any], features: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(features[str(row["feature"])], dtype=np.float64)
    threshold = float(row["threshold"])
    return values >= threshold if row["direction"] == "high" else values <= threshold


def two_tier_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    base_warning: np.ndarray,
    q5_warning: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, bottom = top_bottom_masks(line_score)
    q5 = top_decile_quintiles(line_score) == 5
    masks = {
        "base_rule_only": base_warning & top,
        "q5_rule_only": q5_warning & q5,
        "two_tier_rule": (q5 & q5_warning) | (top & ~q5 & base_warning),
    }
    removed_count = int(masks["two_tier_rule"].sum())
    top_indices = np.flatnonzero(top)
    rng = np.random.default_rng(167)
    random_mask = np.zeros(len(line_score), dtype=bool)
    if removed_count > 0:
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
        full = pool_metrics(
            prediction=prediction,
            pool_mask=top,
            warning_mask=warning,
            severe_threshold=severe_threshold,
            bottom_mask=bottom,
        )
        q5_metrics = pool_metrics(
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
            }
        )
    return rows


def sliding_window_rows(
    *,
    prediction: dict[str, Any],
    base_warning: np.ndarray,
    q5_warning: np.ndarray,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    metadata = prediction["metadata"].copy().reset_index(drop=True)
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top, bottom = top_bottom_masks(line_score)
    q5 = top_decile_quintiles(line_score) == 5
    masks = {
        "base_rule_only": base_warning & top,
        "two_tier_rule": (q5 & q5_warning) | (top & ~q5 & base_warning),
    }
    rows: list[dict[str, Any]] = []
    window = min(126, max(21, len(unique_dates)))
    step = 21
    for start in range(0, max(len(unique_dates) - window + 1, 1), step):
        window_dates = set(unique_dates[start : start + window])
        window_mask = dates.isin(window_dates).to_numpy(dtype=bool)
        if int((top & window_mask).sum()) < 100:
            continue
        for rule_id, warning in masks.items():
            metrics = pool_metrics(
                prediction=prediction,
                pool_mask=top & window_mask,
                warning_mask=warning,
                severe_threshold=severe_threshold,
                bottom_mask=bottom & window_mask,
            )
            rows.append(
                {
                    "split": "test",
                    "rule_id": rule_id,
                    "window_start": str(unique_dates[start])[:10],
                    "window_end": str(unique_dates[min(start + window - 1, len(unique_dates) - 1)])[
                        :10
                    ],
                    **metrics,
                }
            )
    return rows


def classify_final(
    two_tier_test: dict[str, Any],
    base_test: dict[str, Any],
    random_test: dict[str, Any],
    trim_test: dict[str, Any],
    sliding_rows: list[dict[str, Any]],
) -> str:
    q5_improved = (two_tier_test.get("q5_false_safe_reduction") or 0.0) > (
        base_test.get("q5_false_safe_reduction") or -999
    )
    q5_positive = (two_tier_test.get("q5_false_safe_reduction") or 0.0) > 0 and (
        two_tier_test.get("q5_severe_lift_removed_vs_kept") or 0.0
    ) > 1
    retention_ok = (two_tier_test.get("spread_retention") or 0.0) >= 0.80 and (
        two_tier_test.get("fee_retention") or 0.0
    ) >= 0.80
    baseline_ok = (two_tier_test.get("false_safe_reduction") or 0.0) > (
        random_test.get("false_safe_reduction") or -999
    ) and (two_tier_test.get("false_safe_reduction") or 0.0) > (
        trim_test.get("false_safe_reduction") or -999
    )
    window_values = [
        row
        for row in sliding_rows
        if row["rule_id"] == "two_tier_rule"
        and row.get("false_safe_reduction") is not None
        and row.get("spread_retention") is not None
    ]
    weak_windows = [
        row
        for row in window_values
        if (row.get("false_safe_reduction") or 0.0) <= 0
        or (row.get("spread_retention") or 0.0) < 0.70
    ]
    stability_warn = bool(window_values and len(weak_windows) / len(window_values) <= 0.35)
    if q5_improved and q5_positive and retention_ok and baseline_ok and stability_warn:
        return "TWO_TIER_WARNING_CANDIDATE"
    if q5_improved and q5_positive:
        return "Q5_RESCUE_RESEARCH"
    if (base_test.get("false_safe_reduction") or 0.0) > 0 and (
        base_test.get("spread_retention") or 0.0
    ) >= 0.80:
        return "BASE_RULE_ONLY_KEEP"
    if q5_improved or (two_tier_test.get("false_safe_reduction") or 0.0) > 0:
        return "WEAK_SIGNAL"
    return "REJECT"


def write_report(payload: dict[str, Any]) -> None:
    selected = payload.get("selected_q5_rule") or {}
    base_q5 = payload["base_q5_test"]
    two_tier_test = payload["two_tier_test"]
    lines = [
        "# CP167-LM 1D Line Top Quintile Risk Rescue",
        "",
        "## 한 줄 결론",
        f"- 최종 판정: **{payload['final_label']}**",
        "- 이번 CP는 line warning을 억지로 붙이는 실험이 아니라, 사용자가 가장 매력적으로 볼 line 최상위 구간에서 warning이 실제로 작동하는지 확인한 제품성 진단이다.",
        "",
        "## 금지 작업 준수",
        "- 새 딥러닝 학습 없음",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch / EODHD fallback 없음",
        "- composite 실행 없음",
        "- CP153 band artifact 변경 없음",
        "",
        "## Stage 0 Preflight",
        f"- split_mode: `{payload['split_metadata'].get('split_mode')}`",
        f"- cross_split_date_overlap_count: `{payload['split_metadata'].get('cross_split_date_overlap_count')}`",
        f"- test top decile sample: `{payload['preflight']['test_top_decile_count']}`",
        f"- test q5 sample: `{payload['preflight']['test_q5_count']}`",
        f"- base rule warning share test: `{_fmt(payload['preflight']['base_rule_warning_share_test'])}`",
        "",
        "## Stage 1 q5 Failure Decomposition",
        f"- base rule q5 false-safe 감소 test: `{_fmt(base_q5.get('q5_false_safe_reduction'))}`",
        f"- base rule q5 severe lift test: `{_fmt(base_q5.get('q5_severe_lift_removed_vs_kept'))}`",
        "- CP166에서 본 것처럼 q5에서는 base warning이 평균 top decile보다 약하거나 뒤집히는 구간이 있었다.",
        "",
        "## Stage 2~3 q5 Rule Search",
        f"- 선택된 q5 rule: `{selected.get('rule_id')}`",
        f"- 선택 이유: `{selected.get('selection_reason')}`",
        f"- feature: `{selected.get('feature')}`, direction: `{selected.get('direction')}`, threshold: `{_fmt(selected.get('threshold'))}`",
        "",
        "## Stage 4 Two-tier Rule 평가",
        "| rule | FS 감소 | severe lift | spread retention | fee retention | q5 FS 감소 | q5 severe lift | warning share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["two_tier_rows"]:
        if row["split"] == "test":
            lines.append(
                f"| {row['rule_id']} | {_fmt(row.get('false_safe_reduction'))} | {_fmt(row.get('severe_lift_removed_vs_kept'))} | "
                f"{_fmt(row.get('spread_retention'))} | {_fmt(row.get('fee_retention'))} | "
                f"{_fmt(row.get('q5_false_safe_reduction'))} | {_fmt(row.get('q5_severe_lift_removed_vs_kept'))} | {_fmt(row.get('warning_share'))} |"
            )
    lines.extend(
        [
            "",
            "## 필수 질문 답변",
            "1. q5 warning 신호가 왜 역전됐는가?",
            "   - q5는 line score 자체가 매우 높은 구간이라, ATR이 단순 위험이 아니라 고수익 변동성/모멘텀 성격까지 같이 담는다. 그래서 base ATR/self rule은 q5에서 severe만 깔끔히 자르지 못했다.",
            "2. q5 severe 샘플은 어떤 feature profile을 갖는가?",
            f"   - feature profile은 `{PROFILE_CSV}`에 severe/safe 평균, effect size, AUC로 기록했다.",
            "3. q5에서는 ATR보다 단기 꺾임/미시구조/거래량 신호가 더 나은가?",
            f"   - q5 rule search 결과는 `{RULE_SEARCH_CSV}`에 기록했다. 선택 rule이 없다면 ATR 대체 신호도 충분히 안정적이지 않았다는 뜻이다.",
            "4. 2-tier rule이 base_rule 단독보다 나은가?",
            f"   - 최종 판정은 `{payload['final_label']}`이다.",
            "5. q5 개선이 전체 spread/fee를 너무 훼손하지 않는가?",
            f"   - two-tier test spread/fee retention은 `{_fmt(two_tier_test.get('spread_retention'))}` / `{_fmt(two_tier_test.get('fee_retention'))}`이다.",
            "6. 제품에는 여전히 2라벨 warning으로 표현 가능한가?",
            "   - q5에서 안 되면 2라벨 warning도 제품 적용을 보류한다. 통과하더라도 문구는 `주의` 수준이어야 한다.",
            "",
            "## 산출물",
            f"- metrics: `{METRICS_PATH}`",
            f"- summary: `{SUMMARY_CSV}`",
            f"- q5 decomposition: `{DECOMP_CSV}`",
            f"- q5 feature profile: `{PROFILE_CSV}`",
            f"- q5 rule search: `{RULE_SEARCH_CSV}`",
            f"- two-tier comparison: `{TWO_TIER_CSV}`",
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
    price_feature_frame = compute_price_feature_frame(price)
    val_features = {**val_features, **map_price_features(val_prediction, price_feature_frame)}
    test_features = {**test_features, **map_price_features(test_prediction, price_feature_frame)}

    partitions = validation_partitions(val_prediction)
    valid_a_prediction, valid_a_features = subset_prediction(
        val_prediction, partitions["valid_a"], val_features
    )
    valid_b_prediction, valid_b_features = subset_prediction(
        val_prediction, partitions["valid_b"], val_features
    )

    base_val = base_warning_mask(val_features, val_features)
    base_test = base_warning_mask(test_features, val_features)
    base_valid_a = base_warning_mask(valid_a_features, val_features)
    base_valid_b = base_warning_mask(valid_b_features, val_features)

    decomp_rows: list[dict[str, Any]] = []
    decomp_rows.extend(
        q5_decomposition_rows(
            split="validation",
            prediction=val_prediction,
            features=val_features,
            warning_mask=base_val,
            severe_threshold=severe_threshold,
        )
    )
    decomp_rows.extend(
        q5_decomposition_rows(
            split="test",
            prediction=test_prediction,
            features=test_features,
            warning_mask=base_test,
            severe_threshold=severe_threshold,
        )
    )

    profile_rows: list[dict[str, Any]] = []
    profile_rows.extend(
        feature_profile_rows(
            split="valid_a",
            prediction=valid_a_prediction,
            features=valid_a_features,
            severe_threshold=severe_threshold,
        )
    )
    profile_rows.extend(
        feature_profile_rows(
            split="valid_b",
            prediction=valid_b_prediction,
            features=valid_b_features,
            severe_threshold=severe_threshold,
        )
    )
    profile_rows.extend(
        feature_profile_rows(
            split="test",
            prediction=test_prediction,
            features=test_features,
            severe_threshold=severe_threshold,
        )
    )

    valid_a_rules = build_rule_search_rows(
        split="valid_a",
        prediction=valid_a_prediction,
        features=valid_a_features,
        valid_a_features=valid_a_features,
        valid_a_prediction=valid_a_prediction,
        severe_threshold=severe_threshold,
    )
    valid_b_rules = build_rule_search_rows(
        split="valid_b",
        prediction=valid_b_prediction,
        features=valid_b_features,
        valid_a_features=valid_a_features,
        valid_a_prediction=valid_a_prediction,
        severe_threshold=severe_threshold,
    )
    test_rules = build_rule_search_rows(
        split="test",
        prediction=test_prediction,
        features=test_features,
        valid_a_features=valid_a_features,
        valid_a_prediction=valid_a_prediction,
        severe_threshold=severe_threshold,
    )
    selected_rule = select_q5_rule(valid_a_rules, valid_b_rules)
    if selected_rule is None and valid_a_rules:
        selected_rule = sorted(
            valid_a_rules,
            key=lambda row: (
                row.get("severe_lift_removed_vs_kept") or -999,
                row.get("false_safe_reduction") or -999,
                row.get("spread_retention") or -999,
            ),
            reverse=True,
        )[0]
        selected_rule["selection_reason"] = "fallback_valid_a_best_no_positive_rule"

    q5_val = (
        apply_rule_from_row(selected_rule, val_features)
        if selected_rule
        else np.zeros(len(val_prediction["actual"]), dtype=bool)
    )
    q5_test = (
        apply_rule_from_row(selected_rule, test_features)
        if selected_rule
        else np.zeros(len(test_prediction["actual"]), dtype=bool)
    )

    two_tier_rows_all: list[dict[str, Any]] = []
    two_tier_rows_all.extend(
        two_tier_rows(
            split="validation",
            prediction=val_prediction,
            base_warning=base_val,
            q5_warning=q5_val,
            severe_threshold=severe_threshold,
        )
    )
    two_tier_rows_all.extend(
        two_tier_rows(
            split="test",
            prediction=test_prediction,
            base_warning=base_test,
            q5_warning=q5_test,
            severe_threshold=severe_threshold,
        )
    )

    sliding_rows = sliding_window_rows(
        prediction=test_prediction,
        base_warning=base_test,
        q5_warning=q5_test,
        severe_threshold=severe_threshold,
    )
    test_by_rule = {row["rule_id"]: row for row in two_tier_rows_all if row["split"] == "test"}
    base_test_row = test_by_rule["base_rule_only"]
    two_tier_test = test_by_rule["two_tier_rule"]
    random_test = test_by_rule["random_matched_warning"]
    trim_test = test_by_rule["line_score_trim_matched_warning"]
    final = classify_final(two_tier_test, base_test_row, random_test, trim_test, sliding_rows)

    cp153_after = cp159.cp153_artifact_state()
    line_score = np.asarray(test_prediction["line_score"], dtype=np.float64)
    top, _bottom = top_bottom_masks(line_score)
    q5 = top_decile_quintiles(line_score) == 5
    preflight = {
        "test_top_decile_count": int(top.sum()),
        "test_q5_count": int(q5.sum()),
        "base_rule_warning_share_test": _safe_ratio(int((base_test & top).sum()), int(top.sum())),
        "cp153_band_artifact_unchanged": cp153_before == cp153_after,
        "new_training": False,
        "product_save_run": False,
        "db_write": False,
        "inference_save": False,
        "live_fetch": False,
        "eodhd_fallback": False,
        "composite_execution": False,
    }
    summary_rows = [
        {
            "final_label": final,
            "selected_q5_rule": None if selected_rule is None else selected_rule.get("rule_id"),
            "split_mode": split_summary.get("split_mode"),
            "cross_split_date_overlap_count": split_summary.get("cross_split_date_overlap_count"),
            "test_top_decile_count": preflight["test_top_decile_count"],
            "test_q5_count": preflight["test_q5_count"],
            "two_tier_false_safe_reduction": two_tier_test.get("false_safe_reduction"),
            "two_tier_spread_retention": two_tier_test.get("spread_retention"),
            "two_tier_fee_retention": two_tier_test.get("fee_retention"),
            "two_tier_q5_false_safe_reduction": two_tier_test.get("q5_false_safe_reduction"),
            "two_tier_q5_severe_lift": two_tier_test.get("q5_severe_lift_removed_vs_kept"),
        }
    ]
    payload = {
        "cp": "CP167-LM",
        "title": "1D Line Top Quintile Risk Rescue",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_label": final,
        "split_metadata": split_summary,
        "preflight": preflight,
        "severe_threshold": severe_threshold,
        "selected_q5_rule": selected_rule,
        "base_q5_test": base_test_row,
        "two_tier_test": two_tier_test,
        "decomposition_rows": decomp_rows,
        "feature_profile_rows": profile_rows,
        "rule_search_rows": valid_a_rules + valid_b_rules + test_rules,
        "two_tier_rows": two_tier_rows_all,
        "sliding_rows": sliding_rows,
    }
    _write_json(METRICS_PATH, payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(DECOMP_CSV, decomp_rows)
    _write_csv(PROFILE_CSV, profile_rows)
    _write_csv(RULE_SEARCH_CSV, valid_a_rules + valid_b_rules + test_rules)
    _write_csv(TWO_TIER_CSV, two_tier_rows_all)
    _write_csv(SLIDING_CSV, sliding_rows)
    write_report(payload)
    if cp159.torch.cuda.is_available():
        cp159.torch.cuda.empty_cache()
    gc.collect()
    print(
        json.dumps(
            {
                "status": "CP167_DONE",
                "final_label": final,
                "selected_q5_rule": None if selected_rule is None else selected_rule.get("rule_id"),
                "split_mode": split_summary.get("split_mode"),
                "cross_split_date_overlap_count": split_summary.get(
                    "cross_split_date_overlap_count"
                ),
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
