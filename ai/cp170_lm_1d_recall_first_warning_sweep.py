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
from ai import cp167_lm_1d_top_quintile_risk_rescue as cp167  # noqa: E402
from ai import cp168_lm_1d_cohort_warning_development as cp168  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "cp170_lm_1d_recall_first_warning_sweep_report.md"
METRICS_PATH = DOCS_DIR / "cp170_lm_1d_recall_first_warning_sweep_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp170_lm_1d_recall_first_warning_sweep_summary.csv"
SINGLE_CSV = DOCS_DIR / "cp170_lm_1d_single_feature_threshold_sweep.csv"
ENSEMBLE_CSV = DOCS_DIR / "cp170_lm_1d_or_vote_ensemble.csv"
CASCADE_CSV = DOCS_DIR / "cp170_lm_1d_strong_weak_cascade.csv"
BUCKET_CSV = DOCS_DIR / "cp170_lm_1d_line_bucket_recall.csv"
RANDOM_CSV = DOCS_DIR / "cp170_lm_1d_random_matched_baseline.csv"
BOOTSTRAP_CSV = DOCS_DIR / "cp170_lm_1d_bootstrap_ci.csv"
SLIDING_CSV = DOCS_DIR / "cp170_lm_1d_sliding_window.csv"
CONCENTRATION_CSV = DOCS_DIR / "cp170_lm_1d_ticker_date_concentration.csv"

CP = "CP170-LM"
SOURCE_HASH_EXPECTED = "90666b44cbfb8e5c"
SEVERE_THRESHOLD = -0.03
FEE_BPS = 0.001
SINGLE_QS = (0.50, 0.60, 0.70, 0.80, 0.90)
ENSEMBLE_QS = (0.60, 0.70, 0.80)
RANDOM_N = 250
BOOTSTRAP_N = 250

FEATURE_ALIASES = {
    "atr_ratio": "atr_ratio",
    "self_vol_percentile_252": "self_vol_percentile_252",
    "intraday_range": "intraday_range_20d_mean",
    "drawdown_from_5d_high": "drawdown_from_5d_high",
    "drawdown_from_20d_high": "drawdown_from_20d_high",
    "downside_vol_ratio_20d": "downside_vol_ratio_20d",
    "vol_accel_5_20": "vol_accel_5_20",
    "volume_z_score": "volume_z_20_252",
}

FEATURE_SETS = {
    "base_vol_set": ("atr_ratio", "self_vol_percentile_252", "vol_accel_5_20"),
    "microstructure_set": ("intraday_range", "drawdown_from_5d_high", "volume_z_score"),
    "downside_set": ("downside_vol_ratio_20d", "drawdown_from_20d_high", "atr_ratio"),
    "all_set": (
        "atr_ratio",
        "self_vol_percentile_252",
        "vol_accel_5_20",
        "intraday_range",
        "drawdown_from_5d_high",
        "volume_z_score",
        "downside_vol_ratio_20d",
        "drawdown_from_20d_high",
    ),
}


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


def datewise_top_bottom_masks(
    prediction: dict[str, Any], top_q: float = 0.90, bottom_q: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    metadata = prediction["metadata"].reset_index(drop=True)
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce")
    line_score = np.asarray(prediction["line_score"], dtype=np.float64)
    top = np.zeros(len(line_score), dtype=bool)
    bottom = np.zeros(len(line_score), dtype=bool)
    frame = pd.DataFrame({"date": dates, "line": line_score})
    for _date, group in frame.groupby("date", sort=False):
        if len(group) < 10:
            continue
        q_hi = float(group["line"].quantile(top_q))
        q_lo = float(group["line"].quantile(bottom_q))
        idx = group.index.to_numpy()
        top[idx] = line_score[idx] >= q_hi
        bottom[idx] = line_score[idx] <= q_lo
    return top, bottom


def compute_extra_price_features(price: pd.DataFrame) -> pd.DataFrame:
    frame = price[["ticker", "date", "close"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    chunks: list[pd.DataFrame] = []
    for _ticker, group in frame.groupby("ticker", sort=False):
        group = group.copy()
        ret = group["close"].astype(float).pct_change()
        vol5 = ret.rolling(5, min_periods=3).std(ddof=0)
        vol20 = ret.rolling(20, min_periods=10).std(ddof=0)
        group["vol_accel_5_20"] = vol5 / vol20.replace(0.0, np.nan)
        group["log_vol_accel_5_20"] = np.log(group["vol_accel_5_20"].replace(0.0, np.nan))
        chunks.append(group)
    result = pd.concat(chunks, ignore_index=True)
    return result[["ticker", "date", "vol_accel_5_20", "log_vol_accel_5_20"]]


def map_extra_features(
    prediction: dict[str, Any], feature_frame: pd.DataFrame
) -> dict[str, np.ndarray]:
    metadata = prediction["metadata"][["ticker", "asof_date"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    metadata["asof_date"] = pd.to_datetime(metadata["asof_date"], errors="coerce")
    merged = metadata.merge(
        feature_frame, how="left", left_on=["ticker", "asof_date"], right_on=["ticker", "date"]
    )
    result: dict[str, np.ndarray] = {}
    for column in ("vol_accel_5_20", "log_vol_accel_5_20"):
        values = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=np.float64)
        result[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def threshold_from_validation(
    feature_name: str,
    validation_features: dict[str, np.ndarray],
    validation_top: np.ndarray,
    q: float,
) -> float:
    values = np.asarray(validation_features[FEATURE_ALIASES[feature_name]], dtype=np.float64)
    finite = values[validation_top & np.isfinite(values)]
    return float(np.quantile(finite, q)) if len(finite) else 0.0


def feature_warning(
    feature_name: str, features: dict[str, np.ndarray], threshold: float
) -> np.ndarray:
    values = np.asarray(features[FEATURE_ALIASES[feature_name]], dtype=np.float64)
    return values >= threshold


def warning_metrics(
    *,
    rule_id: str,
    split: str,
    prediction: dict[str, Any],
    warning_mask: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    no_warning_missed_rate: float | None = None,
) -> dict[str, Any]:
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    severe = actual <= SEVERE_THRESHOLD
    warning = np.asarray(warning_mask, dtype=bool) & top_mask
    kept = top_mask & ~warning
    removed = top_mask & warning
    base_top_mean = _safe_mean(actual[top_mask])
    bottom_mean = _safe_mean(actual[bottom_mask])
    kept_mean = _safe_mean(actual[kept])
    base_spread = (
        None if base_top_mean is None or bottom_mean is None else float(base_top_mean - bottom_mean)
    )
    spread_after = (
        None if kept_mean is None or bottom_mean is None else float(kept_mean - bottom_mean)
    )
    base_fee = None if base_spread is None else float(base_spread - FEE_BPS)
    fee_after = None if spread_after is None else float(spread_after - FEE_BPS)
    severe_count = int((top_mask & severe).sum())
    warned_severe = int((warning & severe).sum())
    missed_rate = _safe_ratio(int((kept & severe).sum()), int(kept.sum()))
    return {
        "rule_id": rule_id,
        "split": split,
        "top_decile_count": int(top_mask.sum()),
        "warning_count": int(warning.sum()),
        "warning_share": _safe_ratio(int(warning.sum()), int(top_mask.sum())),
        "top_severe_count": severe_count,
        "warning_severe_count": warned_severe,
        "warning_severe_recall": _safe_ratio(warned_severe, severe_count),
        "missed_severe_rate": missed_rate,
        "missed_severe_absolute_reduction_vs_no_warning": None
        if no_warning_missed_rate is None or missed_rate is None
        else float(no_warning_missed_rate - missed_rate),
        "warned_severe_rate": _safe_ratio(warned_severe, int(warning.sum())),
        "unwarned_severe_count": int((kept & severe).sum()),
        "warned_actual_mean_return": _safe_mean(actual[removed]),
        "unwarned_actual_mean_return": _safe_mean(actual[kept]),
        "warned_positive_return_rate": _safe_ratio(
            int((actual[removed] > 0).sum()), int(removed.sum())
        ),
        "unwarned_positive_return_rate": _safe_ratio(
            int((actual[kept] > 0).sum()), int(kept.sum())
        ),
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


def no_warning_metrics(
    split: str, prediction: dict[str, Any], top_mask: np.ndarray, bottom_mask: np.ndarray
) -> dict[str, Any]:
    return warning_metrics(
        rule_id="no_warning_baseline",
        split=split,
        prediction=prediction,
        warning_mask=np.zeros(len(prediction["actual"]), dtype=bool),
        top_mask=top_mask,
        bottom_mask=bottom_mask,
        no_warning_missed_rate=None,
    )


def build_single_feature_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    validation_features: dict[str, np.ndarray],
    validation_top: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    no_warning_missed_rate: float | None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    for feature_name in FEATURE_ALIASES:
        for q in SINGLE_QS:
            threshold = threshold_from_validation(
                feature_name, validation_features, validation_top, q
            )
            mask = feature_warning(feature_name, features, threshold)
            rule_id = f"{feature_name}_q{int(q * 100)}"
            row = warning_metrics(
                rule_id=rule_id,
                split=split,
                prediction=prediction,
                warning_mask=mask,
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                no_warning_missed_rate=no_warning_missed_rate,
            )
            row.update(
                {
                    "feature": feature_name,
                    "q": q,
                    "threshold": threshold,
                    "rule_family": "single_feature",
                }
            )
            row["random_expected_recall"] = row.get("warning_share")
            row["excess_recall_vs_random_expectation"] = (
                None
                if row.get("warning_severe_recall") is None
                else float(row["warning_severe_recall"] - (row.get("warning_share") or 0.0))
            )
            rows.append(row)
            masks[rule_id] = mask
    return rows, masks


def build_ensemble_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    features: dict[str, np.ndarray],
    validation_features: dict[str, np.ndarray],
    validation_top: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    no_warning_missed_rate: float | None,
    selected_rule_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    for set_name, feature_names in FEATURE_SETS.items():
        for q in ENSEMBLE_QS:
            components = []
            thresholds = {}
            for feature_name in feature_names:
                threshold = threshold_from_validation(
                    feature_name, validation_features, validation_top, q
                )
                thresholds[feature_name] = threshold
                components.append(feature_warning(feature_name, features, threshold))
            count = np.sum(np.vstack([comp.astype(int) for comp in components]), axis=0)
            family_masks = {
                "or": count >= 1,
                "vote_2plus": count >= 2,
                "vote_3plus": count >= 3,
            }
            for family, mask in family_masks.items():
                rule_id = f"{set_name}_{family}_q{int(q * 100)}"
                if (
                    selected_rule_ids is not None
                    and split == "test"
                    and rule_id not in selected_rule_ids
                ):
                    continue
                row = warning_metrics(
                    rule_id=rule_id,
                    split=split,
                    prediction=prediction,
                    warning_mask=mask,
                    top_mask=top_mask,
                    bottom_mask=bottom_mask,
                    no_warning_missed_rate=no_warning_missed_rate,
                )
                row.update(
                    {
                        "feature_set": set_name,
                        "rule_family": family,
                        "q": q,
                        "feature_count": len(feature_names),
                        "thresholds_json": json.dumps(
                            thresholds, ensure_ascii=False, sort_keys=True
                        ),
                    }
                )
                rows.append(row)
                masks[rule_id] = mask
    return rows, masks


def select_validation_rules(rows: list[dict[str, Any]], limit: int = 8) -> list[str]:
    eligible = [
        row
        for row in rows
        if (row.get("warning_share") or 0.0) <= 0.55
        and (row.get("warning_share") or 0.0) >= 0.05
        and (row.get("spread_retention") or 0.0) >= 0.65
        and (row.get("fee_retention") or 0.0) >= 0.65
        and (row.get("warning_severe_recall") or 0.0) > (row.get("warning_share") or 0.0)
    ]
    if not eligible:
        eligible = rows
    ordered = sorted(
        eligible,
        key=lambda row: (
            row.get("warning_severe_recall") or -999.0,
            -(row.get("missed_severe_rate") or 999.0),
            row.get("spread_retention") or -999.0,
            row.get("fee_retention") or -999.0,
        ),
        reverse=True,
    )
    return [str(row["rule_id"]) for row in ordered[:limit]]


def random_baseline_rows(
    *,
    rule_id: str,
    prediction: dict[str, Any],
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    warning_share: float,
    candidate_metrics: dict[str, Any],
    no_warning_missed_rate: float | None,
) -> list[dict[str, Any]]:
    top_indices = np.flatnonzero(top_mask)
    rng = np.random.default_rng(170)
    rows = []
    for idx in range(RANDOM_N):
        warning = np.zeros(len(prediction["actual"]), dtype=bool)
        count = int(round(len(top_indices) * warning_share))
        if count > 0:
            warning[rng.choice(top_indices, size=min(count, len(top_indices)), replace=False)] = (
                True
            )
        row = warning_metrics(
            rule_id=f"{rule_id}_random_{idx}",
            split="test",
            prediction=prediction,
            warning_mask=warning,
            top_mask=top_mask,
            bottom_mask=bottom_mask,
            no_warning_missed_rate=no_warning_missed_rate,
        )
        rows.append(row)
    output = []
    for metric in (
        "warning_severe_recall",
        "missed_severe_rate",
        "warned_severe_rate",
        "spread_retention",
        "fee_retention",
    ):
        arr = np.asarray(
            [row.get(metric) for row in rows if row.get(metric) is not None], dtype=np.float64
        )
        candidate_value = candidate_metrics.get(metric)
        output.append(
            {
                "rule_id": rule_id,
                "metric": metric,
                "candidate_value": candidate_value,
                "random_mean": float(arr.mean()) if len(arr) else None,
                "random_std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "candidate_minus_random": None
                if candidate_value is None or not len(arr)
                else float(candidate_value - arr.mean()),
                "random_n": RANDOM_N,
                "warning_share": warning_share,
            }
        )
    return output


def bootstrap_ci_rows(
    *,
    rule_id: str,
    prediction: dict[str, Any],
    warning_mask: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> list[dict[str, Any]]:
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    top_idx = np.flatnonzero(top_mask)
    bottom_mean = _safe_mean(actual[bottom_mask]) or 0.0
    base_top_mean = _safe_mean(actual[top_mask]) or 0.0
    base_spread = base_top_mean - bottom_mean
    base_fee = base_spread - FEE_BPS
    severe = actual <= SEVERE_THRESHOLD
    warning = warning_mask.astype(bool)
    rng = np.random.default_rng(17_000)
    values = {
        key: []
        for key in (
            "warning_severe_recall",
            "missed_severe_rate",
            "spread_retention",
            "fee_retention",
        )
    }
    for _ in range(BOOTSTRAP_N):
        sample_idx = rng.choice(top_idx, size=len(top_idx), replace=True)
        sample_warning = warning[sample_idx]
        sample_severe = severe[sample_idx]
        sample_actual = actual[sample_idx]
        on = sample_warning
        off = ~sample_warning
        severe_count = int(sample_severe.sum())
        recall = _safe_ratio(int((on & sample_severe).sum()), severe_count)
        missed = _safe_ratio(int((off & sample_severe).sum()), int(off.sum()))
        kept_mean = _safe_mean(sample_actual[off])
        spread_after = None if kept_mean is None else kept_mean - bottom_mean
        fee_after = None if spread_after is None else spread_after - FEE_BPS
        if recall is not None:
            values["warning_severe_recall"].append(recall)
        if missed is not None:
            values["missed_severe_rate"].append(missed)
        if spread_after is not None and abs(base_spread) > 1e-12:
            values["spread_retention"].append(spread_after / base_spread)
        if fee_after is not None and abs(base_fee) > 1e-12:
            values["fee_retention"].append(fee_after / base_fee)
    rows = []
    for metric, metric_values in values.items():
        arr = np.asarray(metric_values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        rows.append(
            {
                "rule_id": rule_id,
                "metric": metric,
                "bootstrap_n": BOOTSTRAP_N,
                "mean": float(arr.mean()) if len(arr) else None,
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "ci_lower_95": float(np.quantile(arr, 0.025)) if len(arr) else None,
                "ci_upper_95": float(np.quantile(arr, 0.975)) if len(arr) else None,
            }
        )
    return rows


def cascade_rows(
    *,
    split: str,
    prediction: dict[str, Any],
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    strong_mask: np.ndarray,
    weak_mask: np.ndarray,
    no_warning_missed_rate: float | None,
) -> list[dict[str, Any]]:
    masks = {
        "strong_only": strong_mask,
        "weak_only": weak_mask,
        "strong_or_weak": strong_mask | weak_mask,
        "strong_and_weak": strong_mask & weak_mask,
    }
    rows = []
    for rule_id, mask in masks.items():
        rows.append(
            warning_metrics(
                rule_id=rule_id,
                split=split,
                prediction=prediction,
                warning_mask=mask,
                top_mask=top_mask,
                bottom_mask=bottom_mask,
                no_warning_missed_rate=no_warning_missed_rate,
            )
        )
    actual = np.asarray(prediction["actual"], dtype=np.float64)
    severe = actual <= SEVERE_THRESHOLD
    state = np.zeros(len(actual), dtype=np.int8)
    state[top_mask & weak_mask] = 1
    state[top_mask & strong_mask] = 2
    for level, label in ((0, "no_warning"), (1, "weak_warning"), (2, "strong_warning")):
        mask = top_mask & (state == level)
        rows.append(
            {
                "rule_id": "cascade_level",
                "split": split,
                "level": level,
                "level_label": label,
                "sample_count": int(mask.sum()),
                "sample_share": _safe_ratio(int(mask.sum()), int(top_mask.sum())),
                "actual_mean_return": _safe_mean(actual[mask]),
                "positive_return_rate": _safe_ratio(int((actual[mask] > 0).sum()), int(mask.sum())),
                "severe_rate": _safe_ratio(int((mask & severe).sum()), int(mask.sum())),
            }
        )
    return rows


def line_bucket_rows(
    *,
    rule_id: str,
    split: str,
    prediction: dict[str, Any],
    warning_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for label, q in (("top_10", 0.90), ("top_5", 0.95), ("top_2", 0.98)):
        top, _bottom = datewise_top_bottom_masks(prediction, top_q=q, bottom_q=0.10)
        row = warning_metrics(
            rule_id=rule_id,
            split=split,
            prediction=prediction,
            warning_mask=warning_mask,
            top_mask=top,
            bottom_mask=bottom_mask,
            no_warning_missed_rate=None,
        )
        row["line_bucket"] = label
        rows.append(row)
    return rows


def sliding_window_rows(
    *,
    rule_id: str,
    prediction: dict[str, Any],
    warning_mask: np.ndarray,
    no_warning_missed_rate: float | None,
) -> list[dict[str, Any]]:
    metadata = prediction["metadata"].reset_index(drop=True)
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    rows = []
    window = min(126, max(21, len(unique_dates)))
    step = 21
    for start in range(0, max(len(unique_dates) - window + 1, 1), step):
        window_dates = set(unique_dates[start : start + window])
        mask = dates.isin(window_dates).to_numpy(dtype=bool)
        subset = {
            "candidate_id": prediction["candidate_id"],
            "line_score": np.asarray(prediction["line_score"])[mask],
            "actual": np.asarray(prediction["actual"])[mask],
            "metadata": prediction["metadata"].loc[mask].reset_index(drop=True),
        }
        top, bottom = datewise_top_bottom_masks(subset)
        if int(top.sum()) < 100:
            continue
        row = warning_metrics(
            rule_id=rule_id,
            split="test",
            prediction=subset,
            warning_mask=warning_mask[mask],
            top_mask=top,
            bottom_mask=bottom,
            no_warning_missed_rate=no_warning_missed_rate,
        )
        row["window_start"] = str(unique_dates[start])[:10]
        row["window_end"] = str(unique_dates[min(start + window - 1, len(unique_dates) - 1)])[:10]
        rows.append(row)
    return rows


def concentration_rows(
    *,
    rule_id: str,
    prediction: dict[str, Any],
    warning_mask: np.ndarray,
    top_mask: np.ndarray,
) -> list[dict[str, Any]]:
    metadata = prediction["metadata"].reset_index(drop=True)
    top_warning = top_mask & warning_mask
    rows: list[dict[str, Any]] = []
    for kind, column, limit in (("ticker", "ticker", 15), ("date", "asof_date", 15)):
        series = metadata.loc[top_warning, column].astype(str)
        counts = series.value_counts().head(limit)
        total = int(top_warning.sum())
        for rank, (value, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "rule_id": rule_id,
                    "kind": kind,
                    "rank": rank,
                    "value": value,
                    "count": int(count),
                    "share": _safe_ratio(int(count), total),
                    "total_warning_count": total,
                }
            )
        hhi_values = series.value_counts(normalize=True).to_numpy(dtype=np.float64)
        rows.append(
            {
                "rule_id": rule_id,
                "kind": f"{kind}_summary",
                "rank": 0,
                "value": "hhi",
                "count": int(total),
                "share": float(np.sum(hhi_values * hhi_values)) if len(hhi_values) else None,
                "total_warning_count": total,
            }
        )
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce")
    frame = pd.DataFrame(
        {"ticker": metadata["ticker"].astype(str), "date": dates, "warning": top_warning}
    )
    flips = []
    for _ticker, group in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        vals = group["warning"].to_numpy(dtype=int)
        if len(vals) > 1:
            flips.extend(list(vals[1:] != vals[:-1]))
    rows.append(
        {
            "rule_id": rule_id,
            "kind": "temporal_summary",
            "rank": 0,
            "value": "warning_flip_rate",
            "count": len(flips),
            "share": float(np.mean(flips)) if flips else None,
            "total_warning_count": int(top_warning.sum()),
        }
    )
    return rows


def classify_final(
    best_row: dict[str, Any],
    cp169_row: dict[str, Any],
    random_rows: list[dict[str, Any]],
    bootstrap_rows_: list[dict[str, Any]],
    sliding_rows_: list[dict[str, Any]],
) -> str:
    random_by_metric = {
        row["metric"]: row for row in random_rows if row["rule_id"] == best_row["rule_id"]
    }
    boot = {(row["rule_id"], row["metric"]): row for row in bootstrap_rows_}
    recall = best_row.get("warning_severe_recall") or 0.0
    missed_reduction = best_row.get("missed_severe_absolute_reduction_vs_no_warning") or 0.0
    warning_share = best_row.get("warning_share") or 0.0
    spread = best_row.get("spread_retention") or 0.0
    fee = best_row.get("fee_retention") or 0.0
    recall_excess = (
        random_by_metric.get("warning_severe_recall", {}).get("candidate_minus_random") or -999.0
    )
    missed_excess = random_by_metric.get("missed_severe_rate", {}).get("candidate_minus_random")
    missed_better_than_random = missed_excess is not None and missed_excess < 0
    bad_windows = [
        row
        for row in sliding_rows_
        if row.get("rule_id") == best_row["rule_id"]
        and (
            (row.get("spread_retention") or 0.0) < 0.60 or (row.get("fee_retention") or 0.0) < 0.60
        )
    ]
    strong = (
        recall >= 0.50
        and missed_reduction >= 0.03
        and spread >= 0.80
        and fee >= 0.80
        and warning_share <= 0.40
        and recall_excess > 0
        and missed_better_than_random
        and not bad_windows
        and (boot.get((best_row["rule_id"], "warning_severe_recall"), {}).get("ci_lower_95") or 0.0)
        >= 0.45
    )
    if strong:
        return "RECALL_FIRST_STRONG_BETA_CANDIDATE"
    primary = (
        recall >= 0.50
        and (best_row.get("missed_severe_rate") or 1.0)
        < (cp169_row.get("missed_severe_rate") or 1.0)
        and warning_share <= 0.45
        and spread >= 0.75
        and fee >= 0.75
        and recall_excess > 0
        and missed_better_than_random
    )
    if primary:
        return "RECALL_FIRST_WARNING_CANDIDATE"
    if recall >= (cp169_row.get("warning_severe_recall") or 0.0) and (
        best_row.get("missed_severe_rate") or 1.0
    ) < (cp169_row.get("missed_severe_rate") or 1.0):
        return "RECALL_RESEARCH_WARNING"
    return "WARNING_RECALL_NOT_READY"


def write_report(payload: dict[str, Any]) -> None:
    best = payload["best_rule_test"]
    cp169 = payload["cp169_two_tier_test"]
    lines = [
        "# CP170-LM 1D Recall-First Warning Sweep",
        "",
        "## 한 줄 결론",
        f"- 최종 판정: **{payload['final_label']}**",
        f"- best recall-first rule: `{best.get('rule_id')}`",
        "- 이번 CP는 line을 탈락시키는 실험이 아니라, line top decile 안의 실제 큰 하락을 warning이 얼마나 회수하는지 본 실험이다.",
        "",
        "## 금지 작업 준수",
        "- 새 딥러닝 학습 없음",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch / EODHD fallback 없음",
        "- band/composite 실행 없음",
        "- CP153 band artifact 변경 없음",
        "",
        "## Stage 0 Preflight",
        f"- split_mode: `{payload['split_metadata'].get('split_mode')}`",
        f"- cross_split_date_overlap_count: `{payload['split_metadata'].get('cross_split_date_overlap_count')}`",
        f"- source_data_hash: `{payload['split_metadata'].get('source_data_hash')}`",
        f"- test datewise top decile sample: `{payload['preflight'].get('test_top_decile_count')}`",
        f"- test severe base rate: `{_fmt(payload['preflight'].get('test_no_warning_missed_rate'))}`",
        "",
        "## CP169 기준선 재계산",
        f"- CP169 two-tier recall: `{_fmt(cp169.get('warning_severe_recall'))}`",
        f"- CP169 two-tier missed severe: `{_fmt(cp169.get('missed_severe_rate'))}`",
        f"- CP169 two-tier spread/fee retention: `{_fmt(cp169.get('spread_retention'))}` / `{_fmt(cp169.get('fee_retention'))}`",
        "",
        "## Recall-first Best",
        f"- recall: `{_fmt(best.get('warning_severe_recall'))}`",
        f"- missed severe: `{_fmt(best.get('missed_severe_rate'))}`",
        f"- warning share: `{_fmt(best.get('warning_share'))}`",
        f"- warned severe rate: `{_fmt(best.get('warned_severe_rate'))}`",
        f"- spread/fee retention: `{_fmt(best.get('spread_retention'))}` / `{_fmt(best.get('fee_retention'))}`",
        "",
        "## 해석",
        "- false alarm은 이번 CP의 primary penalty가 아니다. 이번 목표는 실제 큰 하락을 더 많이 잡는 것이다.",
        "- 다만 warning share가 높거나 spread/fee retention이 무너지면 제품에서는 alarm fatigue 또는 alpha 손상으로 해석해야 한다.",
        "- 통과하지 못하면 post-hoc rule의 한계로 보고 CP171 binary downside classifier로 격상한다.",
        "",
        "## 산출물",
        f"- metrics: `{METRICS_PATH}`",
        f"- summary: `{SUMMARY_CSV}`",
        f"- single feature sweep: `{SINGLE_CSV}`",
        f"- OR/vote ensemble: `{ENSEMBLE_CSV}`",
        f"- strong/weak cascade: `{CASCADE_CSV}`",
        f"- line bucket: `{BUCKET_CSV}`",
        f"- random matched baseline: `{RANDOM_CSV}`",
        f"- bootstrap CI: `{BOOTSTRAP_CSV}`",
        f"- sliding window: `{SLIDING_CSV}`",
        f"- concentration: `{CONCENTRATION_CSV}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cp153_before = cp159.cp153_artifact_state()
    cp164_payload = cp165.load_cp164_reference()
    val_prediction, test_prediction, val_features, test_features, split_summary, _thresholds = (
        cp165.build_predictions(cp164_payload)
    )
    if (
        split_summary.get("split_mode") != "calendar_aligned"
        or int(split_summary.get("cross_split_date_overlap_count") or 0) != 0
    ):
        raise RuntimeError(f"calendar split preflight 실패: {split_summary}")
    if str(split_summary.get("source_data_hash")) != SOURCE_HASH_EXPECTED:
        raise RuntimeError(
            f"source_data_hash 불일치: {split_summary.get('source_data_hash')} != {SOURCE_HASH_EXPECTED}"
        )

    price, _indicators, _pm, _im = cp165.cp164.cp158.load_source_frames()
    price_features = cp167.compute_price_feature_frame(price)
    extra_features = compute_extra_price_features(price)
    price_features = price_features.merge(extra_features, how="left", on=["ticker", "date"])
    val_features = {
        **val_features,
        **cp167.map_price_features(val_prediction, price_features),
        **map_extra_features(val_prediction, price_features),
    }
    test_features = {
        **test_features,
        **cp167.map_price_features(test_prediction, price_features),
        **map_extra_features(test_prediction, price_features),
    }

    partitions = cp167.validation_partitions(val_prediction)
    valid_a_prediction, valid_a_features = cp167.subset_prediction(
        val_prediction, partitions["valid_a"], val_features
    )

    val_top, val_bottom = datewise_top_bottom_masks(val_prediction)
    test_top, test_bottom = datewise_top_bottom_masks(test_prediction)
    valid_a_top, _valid_a_bottom = datewise_top_bottom_masks(valid_a_prediction)

    no_warning_val = no_warning_metrics("validation", val_prediction, val_top, val_bottom)
    no_warning_test = no_warning_metrics("test", test_prediction, test_top, test_bottom)
    no_warning_missed_test = no_warning_test.get("missed_severe_rate")
    no_warning_missed_val = no_warning_val.get("missed_severe_rate")

    single_val, single_val_masks = build_single_feature_rows(
        split="validation",
        prediction=val_prediction,
        features=val_features,
        validation_features=val_features,
        validation_top=val_top,
        top_mask=val_top,
        bottom_mask=val_bottom,
        no_warning_missed_rate=no_warning_missed_val,
    )
    single_test, single_test_masks = build_single_feature_rows(
        split="test",
        prediction=test_prediction,
        features=test_features,
        validation_features=val_features,
        validation_top=val_top,
        top_mask=test_top,
        bottom_mask=test_bottom,
        no_warning_missed_rate=no_warning_missed_test,
    )
    selected_single_ids = set(select_validation_rules(single_val, limit=10))

    ensemble_val, ensemble_val_masks = build_ensemble_rows(
        split="validation",
        prediction=val_prediction,
        features=val_features,
        validation_features=val_features,
        validation_top=val_top,
        top_mask=val_top,
        bottom_mask=val_bottom,
        no_warning_missed_rate=no_warning_missed_val,
    )
    selected_ensemble_ids = set(select_validation_rules(ensemble_val, limit=10))
    ensemble_test, ensemble_test_masks = build_ensemble_rows(
        split="test",
        prediction=test_prediction,
        features=test_features,
        validation_features=val_features,
        validation_top=val_top,
        top_mask=test_top,
        bottom_mask=test_bottom,
        no_warning_missed_rate=no_warning_missed_test,
    )

    cp168_payload = json.loads(
        (DOCS_DIR / "cp168_lm_1d_cohort_warning_development_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    q5_rule = cp168_payload.get("selected_q5_rule") or {}
    q5_val = cp168.selected_rule_mask(
        q5_rule,
        val_features,
        prediction=val_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    q5_test = cp168.selected_rule_mask(
        q5_rule,
        test_features,
        prediction=test_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    q5_valid_a = cp168.selected_rule_mask(
        q5_rule,
        valid_a_features,
        prediction=valid_a_prediction,
        valid_a_prediction=valid_a_prediction,
        valid_a_features=valid_a_features,
    )
    base_val = cp167.base_warning_mask(val_features, val_features)
    base_test = cp167.base_warning_mask(test_features, val_features)
    base_valid_a = cp167.base_warning_mask(valid_a_features, val_features)
    _top_val_global, _bottom_val_global, q5_val_global = cp168._top_q5(val_prediction)
    _top_test_global, _bottom_test_global, q5_test_global = cp168._top_q5(test_prediction)
    cp169_two_tier_val_mask = (q5_val_global & q5_val) | (~q5_val_global & base_val)
    cp169_two_tier_test_mask = (q5_test_global & q5_test) | (~q5_test_global & base_test)
    cp169_two_tier_val = warning_metrics(
        rule_id="cp169_two_tier_rule",
        split="validation",
        prediction=val_prediction,
        warning_mask=cp169_two_tier_val_mask,
        top_mask=val_top,
        bottom_mask=val_bottom,
        no_warning_missed_rate=no_warning_missed_val,
    )
    cp169_two_tier_test = warning_metrics(
        rule_id="cp169_two_tier_rule",
        split="test",
        prediction=test_prediction,
        warning_mask=cp169_two_tier_test_mask,
        top_mask=test_top,
        bottom_mask=test_bottom,
        no_warning_missed_rate=no_warning_missed_test,
    )

    validation_candidates = single_val + ensemble_val + [cp169_two_tier_val]
    test_candidates = single_test + ensemble_test + [cp169_two_tier_test]
    candidate_masks_test = {
        **single_test_masks,
        **ensemble_test_masks,
        "cp169_two_tier_rule": cp169_two_tier_test_mask,
    }
    candidate_masks_val = {
        **single_val_masks,
        **ensemble_val_masks,
        "cp169_two_tier_rule": cp169_two_tier_val_mask,
    }

    eligible_val = [
        row
        for row in validation_candidates
        if row["rule_id"] in selected_single_ids
        or row["rule_id"] in selected_ensemble_ids
        or row["rule_id"] == "cp169_two_tier_rule"
    ]
    best_val = sorted(
        eligible_val,
        key=lambda row: (
            row.get("warning_severe_recall") or -999.0,
            -(row.get("missed_severe_rate") or 999.0),
            row.get("spread_retention") or -999.0,
            row.get("fee_retention") or -999.0,
        ),
        reverse=True,
    )[0]
    best_rule_id = str(best_val["rule_id"])
    test_by_rule = {str(row["rule_id"]): row for row in test_candidates}
    best_test = dict(test_by_rule[best_rule_id])

    # Strong은 precision, weak는 recall을 validation에서 고른다.
    strong_val = sorted(
        [
            row
            for row in validation_candidates
            if (row.get("warning_share") or 0.0) <= 0.30
            and (row.get("spread_retention") or 0.0) >= 0.75
        ],
        key=lambda row: (
            row.get("warned_severe_rate") or -999.0,
            row.get("warning_severe_recall") or -999.0,
        ),
        reverse=True,
    )[0]
    weak_val = sorted(
        [
            row
            for row in validation_candidates
            if (row.get("warning_share") or 0.0) <= 0.45
            and (row.get("spread_retention") or 0.0) >= 0.70
        ],
        key=lambda row: (
            row.get("warning_severe_recall") or -999.0,
            row.get("spread_retention") or -999.0,
        ),
        reverse=True,
    )[0]
    strong_id = str(strong_val["rule_id"])
    weak_id = str(weak_val["rule_id"])
    strong_test_mask = candidate_masks_test[strong_id]
    weak_test_mask = candidate_masks_test[weak_id]
    strong_val_mask = candidate_masks_val[strong_id]
    weak_val_mask = candidate_masks_val[weak_id]
    cascade_output_rows = []
    cascade_output_rows.extend(
        cascade_rows(
            split="validation",
            prediction=val_prediction,
            top_mask=val_top,
            bottom_mask=val_bottom,
            strong_mask=strong_val_mask,
            weak_mask=weak_val_mask,
            no_warning_missed_rate=no_warning_missed_val,
        )
    )
    cascade_output_rows.extend(
        cascade_rows(
            split="test",
            prediction=test_prediction,
            top_mask=test_top,
            bottom_mask=test_bottom,
            strong_mask=strong_test_mask,
            weak_mask=weak_test_mask,
            no_warning_missed_rate=no_warning_missed_test,
        )
    )
    strong_or_weak_test = next(
        row
        for row in cascade_output_rows
        if row.get("split") == "test" and row.get("rule_id") == "strong_or_weak"
    )
    strong_or_weak_val = next(
        row
        for row in cascade_output_rows
        if row.get("split") == "validation" and row.get("rule_id") == "strong_or_weak"
    )
    strong_or_weak_mask = strong_test_mask | weak_test_mask
    if (strong_or_weak_val.get("warning_severe_recall") or 0.0) > (
        best_val.get("warning_severe_recall") or 0.0
    ) and (strong_or_weak_val.get("spread_retention") or 0.0) >= 0.70:
        best_test = dict(strong_or_weak_test)
        best_test["rule_id"] = "strong_or_weak"
        best_rule_id = "strong_or_weak"
        candidate_masks_test[best_rule_id] = strong_or_weak_mask

    random_rows = random_baseline_rows(
        rule_id=best_rule_id,
        prediction=test_prediction,
        top_mask=test_top,
        bottom_mask=test_bottom,
        warning_share=best_test.get("warning_share") or 0.0,
        candidate_metrics=best_test,
        no_warning_missed_rate=no_warning_missed_test,
    )
    bootstrap_rows_ = bootstrap_ci_rows(
        rule_id=best_rule_id,
        prediction=test_prediction,
        warning_mask=candidate_masks_test[best_rule_id],
        top_mask=test_top,
        bottom_mask=test_bottom,
    )
    bucket_rows = []
    for rule_id in ("cp169_two_tier_rule", best_rule_id):
        bucket_rows.extend(
            line_bucket_rows(
                rule_id=rule_id,
                split="test",
                prediction=test_prediction,
                warning_mask=candidate_masks_test[rule_id],
                bottom_mask=test_bottom,
            )
        )
    sliding_rows_ = []
    for rule_id in ("cp169_two_tier_rule", best_rule_id):
        sliding_rows_.extend(
            sliding_window_rows(
                rule_id=rule_id,
                prediction=test_prediction,
                warning_mask=candidate_masks_test[rule_id],
                no_warning_missed_rate=no_warning_missed_test,
            )
        )
    concentration_output_rows = concentration_rows(
        rule_id=best_rule_id,
        prediction=test_prediction,
        warning_mask=candidate_masks_test[best_rule_id],
        top_mask=test_top,
    )
    final_label = classify_final(
        best_test, cp169_two_tier_test, random_rows, bootstrap_rows_, sliding_rows_
    )
    cp153_after = cp159.cp153_artifact_state()

    payload = {
        "cp": CP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_label": final_label,
        "best_rule_test": best_test,
        "strong_rule_from_validation": strong_id,
        "weak_rule_from_validation": weak_id,
        "cp169_two_tier_test": cp169_two_tier_test,
        "no_warning_test": no_warning_test,
        "selected_single_rule_ids": sorted(selected_single_ids),
        "selected_ensemble_rule_ids": sorted(selected_ensemble_ids),
        "split_metadata": split_summary,
        "preflight": {
            "source_data_hash_expected": SOURCE_HASH_EXPECTED,
            "test_top_decile_count": int(test_top.sum()),
            "test_no_warning_missed_rate": no_warning_missed_test,
            "test_no_warning_recall": 0.0,
        },
        "single_feature_rows": single_val + single_test,
        "ensemble_rows": ensemble_val + ensemble_test,
        "cascade_rows": cascade_output_rows,
        "bucket_rows": bucket_rows,
        "random_rows": random_rows,
        "bootstrap_rows": bootstrap_rows_,
        "sliding_rows": sliding_rows_,
        "concentration_rows": concentration_output_rows,
        "cp153_artifact_unchanged": cp153_before == cp153_after,
        "forbidden_work": {
            "new_deep_learning_training": False,
            "product_save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "band_or_composite": False,
            "cp153_band_artifact_modified": cp153_before != cp153_after,
        },
    }
    summary_rows = [
        {
            "final_label": final_label,
            "best_rule_id": best_rule_id,
            "best_warning_severe_recall": best_test.get("warning_severe_recall"),
            "best_missed_severe_rate": best_test.get("missed_severe_rate"),
            "best_warning_share": best_test.get("warning_share"),
            "best_warned_severe_rate": best_test.get("warned_severe_rate"),
            "best_spread_retention": best_test.get("spread_retention"),
            "best_fee_retention": best_test.get("fee_retention"),
            "cp169_warning_severe_recall": cp169_two_tier_test.get("warning_severe_recall"),
            "cp169_missed_severe_rate": cp169_two_tier_test.get("missed_severe_rate"),
            "strong_rule": strong_id,
            "weak_rule": weak_id,
            "cp153_artifact_unchanged": cp153_before == cp153_after,
        }
    ]
    _write_json(METRICS_PATH, payload)
    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(SINGLE_CSV, single_val + single_test)
    _write_csv(ENSEMBLE_CSV, ensemble_val + ensemble_test)
    _write_csv(CASCADE_CSV, cascade_output_rows)
    _write_csv(BUCKET_CSV, bucket_rows)
    _write_csv(RANDOM_CSV, random_rows)
    _write_csv(BOOTSTRAP_CSV, bootstrap_rows_)
    _write_csv(SLIDING_CSV, sliding_rows_)
    _write_csv(CONCENTRATION_CSV, concentration_output_rows)
    write_report(payload)

    if cp159.torch.cuda.is_available():
        cp159.torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
