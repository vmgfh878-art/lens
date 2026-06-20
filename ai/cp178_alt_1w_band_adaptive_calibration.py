from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai import cp178_bm_1w_band_500_stage3_5 as base
from ai import cp178_cal_1w_band_lower_calibration as cal

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

REPORT_PATH = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_report.md"
METRICS_PATH = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_summary.csv"
FOLD_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_fold_results.csv"
PARAMS_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_params.csv"
STRESS_LABEL_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_stress_label.csv"
ROLLING_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_rolling_miss.csv"
BOOTSTRAP_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_bootstrap_ci.csv"
STRESS_CALM_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_stress_calm.csv"
TRANSFER_CSV = DOCS_DIR / "cp178_alt_1w_band_adaptive_calibration_transfer.csv"

CP178_CAL_SUMMARY = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_summary.csv"
CP153_1D_METRICS = DOCS_DIR / "cp153_bm_1d_band_500_stage5t_true_walk_forward_metrics.json"

TARGET_CANDIDATE_ID = "tide_s60_q10_q90_param"
Q_LOW = 0.10
Q_HIGH = 0.90
N_BOOTSTRAP = 200
SUCCESS_P90_WIDTH_WORST = 0.27
LABEL_AVAILABILITY_LAG_DATES = 4


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    return base.safe_float(value)


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.6f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_rows(path, rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    base.write_json(path, payload)


def quantile(values: pd.Series | np.ndarray, q: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def finite_mean(values: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def load_stress_feature_frame() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W_500.parquet"
    wanted = [
        "ticker",
        "date",
        "log_return",
        "vix_close",
        "credit_spread_hy",
        "has_macro",
        "has_breadth",
        "source",
        "provider",
    ]
    columns = pd.read_parquet(path).columns.tolist()
    df = pd.read_parquet(path, columns=[col for col in wanted if col in columns])
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["log_return"] = pd.to_numeric(df.get("log_return"), errors="coerce")
    grouped = df.groupby("ticker", sort=False)
    df["weekly_realized_vol_20w"] = grouped["log_return"].transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    df["recent_5w_drawdown"] = grouped["log_return"].transform(
        lambda s: s.rolling(5, min_periods=2).sum()
    )
    df["max_down_week_20w"] = grouped["log_return"].transform(
        lambda s: s.rolling(20, min_periods=5).min()
    )
    df["negative_week_count_20w"] = grouped["log_return"].transform(
        lambda s: (s < 0).rolling(20, min_periods=5).sum()
    )
    if "vix_close" in df.columns:
        df["vix_close"] = pd.to_numeric(df["vix_close"], errors="coerce")
        df["vix_5w_delta"] = grouped["vix_close"].transform(lambda s: s - s.shift(5))
    if "credit_spread_hy" in df.columns:
        df["credit_spread_hy"] = pd.to_numeric(df["credit_spread_hy"], errors="coerce")
        df["credit_spread_5w_delta"] = grouped["credit_spread_hy"].transform(
            lambda s: s - s.shift(5)
        )
    return df


def attach_stress_features(frame: pd.DataFrame, stress_features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ticker",
        "date",
        "weekly_realized_vol_20w",
        "recent_5w_drawdown",
        "max_down_week_20w",
        "negative_week_count_20w",
        "vix_5w_delta",
        "credit_spread_5w_delta",
        "has_macro",
        "has_breadth",
    ]
    available = [col for col in cols if col in stress_features.columns]
    merged = frame.copy()
    merged["asof_date"] = pd.to_datetime(merged["asof_date"], errors="coerce")
    features = stress_features[available].rename(columns={"date": "asof_date"})
    return merged.merge(features, how="left", on=["ticker", "asof_date"])


def finite_coverage(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(np.isfinite(values).mean()) if len(values) else 0.0


def fit_stress_rule(val: pd.DataFrame) -> dict[str, Any]:
    candidates = [
        ("weekly_realized_vol_20w", "high", 0.80, "price"),
        ("recent_5w_drawdown", "low", 0.20, "price"),
        ("max_down_week_20w", "low", 0.20, "price"),
        ("vix_5w_delta", "high", 0.80, "macro"),
        ("credit_spread_5w_delta", "high", 0.80, "macro"),
    ]
    thresholds: dict[str, Any] = {}
    used: list[dict[str, Any]] = []
    macro_used = False
    for name, direction, q, group in candidates:
        if name not in val.columns:
            continue
        coverage = finite_coverage(val[name])
        if coverage < 0.70:
            thresholds[name] = {"coverage": coverage, "used": False, "reason": "low_coverage"}
            continue
        value = quantile(pd.to_numeric(val[name], errors="coerce"), q)
        if value is None:
            thresholds[name] = {"coverage": coverage, "used": False, "reason": "no_finite_value"}
            continue
        thresholds[name] = {
            "coverage": coverage,
            "used": True,
            "direction": direction,
            "quantile": q,
            "threshold": value,
            "group": group,
        }
        used.append({"name": name, "direction": direction, "threshold": value, "group": group})
        macro_used = macro_used or group == "macro"
    price_used_count = sum(1 for item in used if item["group"] == "price")
    if macro_used and len(used) >= 4:
        vote_threshold = 2
        rule_type = "vol_drawdown_macro_vote"
    elif price_used_count >= 2:
        used = [item for item in used if item["group"] == "price"]
        vote_threshold = 2 if len(used) >= 3 else 1
        rule_type = "price_only_vote"
    else:
        used = [item for item in used if item["name"] == "weekly_realized_vol_20w"]
        vote_threshold = 1
        rule_type = "price_only_fallback_vol"
    return {
        "rule_type": rule_type,
        "used_features": used,
        "thresholds": thresholds,
        "vote_threshold": vote_threshold,
        "fit_split": "validation",
        "test_used_for_rule": False,
    }


def apply_stress_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    votes = np.zeros(len(frame), dtype=np.int32)
    for item in rule.get("used_features") or []:
        name = item["name"]
        values = pd.to_numeric(frame.get(name), errors="coerce").to_numpy(dtype=np.float64)
        if item["direction"] == "high":
            hit = values >= float(item["threshold"])
        else:
            hit = values <= float(item["threshold"])
        hit = np.where(np.isfinite(values), hit, False)
        votes += hit.astype(np.int32)
    return pd.Series(votes >= int(rule.get("vote_threshold") or 1), index=frame.index)


def label_payloads(
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stress_features = load_stress_feature_frame()
    label_rows: list[dict[str, Any]] = []
    out_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        val = attach_stress_features(payload["val"], stress_features)
        test = attach_stress_features(payload["test"], stress_features)
        rule = fit_stress_rule(val)
        val["alt_stress_label"] = apply_stress_rule(val, rule).to_numpy(dtype=bool)
        test["alt_stress_label"] = apply_stress_rule(test, rule).to_numpy(dtype=bool)
        row = payload["row"]
        label_rows.append(
            {
                "fold_id": row.get("fold_id"),
                "seed": row.get("seed"),
                "rule_type": rule["rule_type"],
                "vote_threshold": rule["vote_threshold"],
                "used_features": [item["name"] for item in rule.get("used_features") or []],
                "val_stress_rate": float(val["alt_stress_label"].mean()),
                "test_stress_rate": float(test["alt_stress_label"].mean()),
                "macro_coverage_vix_5w_delta": (rule.get("thresholds") or {})
                .get("vix_5w_delta", {})
                .get("coverage"),
                "macro_coverage_credit_spread_5w_delta": (rule.get("thresholds") or {})
                .get("credit_spread_5w_delta", {})
                .get("coverage"),
                "test_used_for_rule": False,
            }
        )
        updated = dict(payload)
        updated["val"] = val
        updated["test"] = test
        updated["stress_rule"] = rule
        out_payloads.append(updated)
    return out_payloads, label_rows


def objective(metrics: dict[str, Any]) -> float:
    coverage = safe_float(metrics.get("coverage_abs_error")) or 1.0
    lower = safe_float(metrics.get("lower_breach_rate")) or 1.0
    upper = safe_float(metrics.get("upper_breach_rate")) or 1.0
    interval = safe_float(metrics.get("asymmetric_interval_score")) or 1.0
    p90 = safe_float(metrics.get("p90_band_width")) or 1.0
    downside_ic = safe_float(metrics.get("downside_width_ic")) or 0.0
    return (
        5.0 * coverage
        + 5.0 * abs(lower - Q_LOW)
        + 1.5 * abs(upper - (1.0 - Q_HIGH))
        + 0.20 * interval
        + 0.75 * max(0.0, p90 - SUCCESS_P90_WIDTH_WORST)
        + 0.50 * max(0.0, 0.04 - downside_ic)
    )


def apply_constant_shift(
    frame: pd.DataFrame, shift: float, method: str, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[float]]:
    calibrated = cal.apply_lower_extension(frame, float(shift))
    calibrated["extension_applied"] = float(shift)
    calibrated["method"] = method
    return calibrated, date_rolling_rows(calibrated, method, params), date_lower_history(calibrated)


def date_lower_history(frame: pd.DataFrame) -> list[float]:
    values: list[float] = []
    for _, group in frame.sort_values("asof_date").groupby("asof_date", sort=True):
        values.append(float(group["calibrated_lower_breach"].mean()))
    return values


def date_rolling_rows(
    frame: pd.DataFrame, method: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history: list[float] = []
    for asof_date, group in frame.sort_values("asof_date").groupby("asof_date", sort=True):
        observed = (
            history[:-LABEL_AVAILABILITY_LAG_DATES] if LABEL_AVAILABILITY_LAG_DATES else history
        )
        rolling = (
            float(np.mean(observed[-int(params.get("rolling_window", 5)) :])) if observed else None
        )
        rows.append(
            {
                "method": method,
                "fold_id": group["fold_id"].iloc[0],
                "seed": int(group["seed"].iloc[0]),
                "split": group["split"].iloc[0],
                "asof_date": pd.Timestamp(asof_date).date().isoformat(),
                "date_lower_breach_rate": float(group["calibrated_lower_breach"].mean()),
                "lagged_rolling_lower_miss_rate": rolling,
                "extension_mean": float(group["extension_applied"].mean())
                if "extension_applied" in group.columns
                else None,
                "stress_rate": float(group["alt_stress_label"].mean())
                if "alt_stress_label" in group.columns
                else None,
                "label_availability_lag_dates": LABEL_AVAILABILITY_LAG_DATES,
            }
        )
        history.append(float(group["calibrated_lower_breach"].mean()))
    return rows


def apply_adaptive_shift(
    frame: pd.DataFrame,
    *,
    method: str,
    params: dict[str, Any],
    initial_history: list[float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[float]]:
    rolling_window = int(params.get("rolling_window", 5))
    trigger = float(params.get("miss_rate_trigger", 0.14))
    boost = float(params.get("adaptive_boost", 0.0))
    calm_shift = float(params.get("calm_shift", params.get("global_shift", 0.0)))
    stress_shift = float(params.get("stress_shift", params.get("global_shift", 0.0)))
    history = list(initial_history or [])
    parts: list[pd.DataFrame] = []
    rolling_rows: list[dict[str, Any]] = []
    for asof_date, group in frame.sort_values("asof_date").groupby("asof_date", sort=True):
        observed_history = (
            history[:-LABEL_AVAILABILITY_LAG_DATES] if LABEL_AVAILABILITY_LAG_DATES else history
        )
        rolling = float(np.mean(observed_history[-rolling_window:])) if observed_history else 0.0
        adaptive_extra = boost if rolling > trigger else 0.0
        stress = group.get("alt_stress_label", pd.Series(False, index=group.index)).to_numpy(
            dtype=bool
        )
        base_extension = np.where(stress, stress_shift, calm_shift)
        extension = base_extension + adaptive_extra
        calibrated = cal.apply_lower_extension(group, extension)
        calibrated["extension_applied"] = extension
        calibrated["method"] = method
        calibrated["adaptive_extra"] = adaptive_extra
        parts.append(calibrated)
        lower_miss = float(calibrated["calibrated_lower_breach"].mean())
        rolling_rows.append(
            {
                "method": method,
                "fold_id": group["fold_id"].iloc[0],
                "seed": int(group["seed"].iloc[0]),
                "split": group["split"].iloc[0],
                "asof_date": pd.Timestamp(asof_date).date().isoformat(),
                "date_lower_breach_rate": lower_miss,
                "lagged_rolling_lower_miss_rate": rolling,
                "extension_mean": float(np.mean(extension)),
                "extension_stress_mean": float(np.mean(extension[stress]))
                if bool(stress.any())
                else None,
                "extension_calm_mean": float(np.mean(extension[~stress]))
                if bool((~stress).any())
                else None,
                "stress_rate": float(stress.mean()),
                "adaptive_extra": adaptive_extra,
                "miss_rate_trigger": trigger,
                "rolling_window": rolling_window,
                "label_availability_lag_dates": LABEL_AVAILABILITY_LAG_DATES,
            }
        )
        history.append(lower_miss)
    if not parts:
        empty = frame.copy()
        empty["extension_applied"] = np.nan
        return empty, rolling_rows, history
    return pd.concat(parts).sort_index(), rolling_rows, history


def fit_adaptive_conformal(val: pd.DataFrame, *, stress_aware: bool) -> dict[str, Any]:
    if stress_aware:
        stress_mask = val["alt_stress_label"].to_numpy(dtype=bool)
        calm_shift = (
            cal.conformal_shift(val.loc[~stress_mask])
            if bool((~stress_mask).any())
            else cal.conformal_shift(val)
        )
        stress_shift = (
            cal.conformal_shift(val.loc[stress_mask]) if bool(stress_mask.any()) else calm_shift
        )
    else:
        calm_shift = cal.conformal_shift(val)
        stress_shift = calm_shift
    best_params: dict[str, Any] = {
        "calm_shift": calm_shift,
        "stress_shift": stress_shift,
        "adaptive_boost": 0.0,
        "miss_rate_trigger": 0.14,
        "rolling_window": 5,
        "stress_aware": stress_aware,
    }
    best_score = float("inf")
    best_metrics: dict[str, Any] = {}
    for rolling_window in (4, 6, 8):
        for trigger in (0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20):
            for boost in (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
                params = {
                    "calm_shift": calm_shift,
                    "stress_shift": stress_shift,
                    "adaptive_boost": boost,
                    "miss_rate_trigger": trigger,
                    "rolling_window": rolling_window,
                    "stress_aware": stress_aware,
                }
                calibrated, _, _ = apply_adaptive_shift(
                    val, method="fit", params=params, initial_history=[]
                )
                metrics = cal.interval_metrics(calibrated)
                score = objective(metrics)
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
    return {
        "method": "stress_aware_adaptive_conformal" if stress_aware else "adaptive_conformal_lower",
        "params": best_params,
        "fit_metrics": best_metrics,
        "fit_score": best_score,
        "test_used_for_fit": False,
    }


def fit_walk_forward_shift(
    payload: dict[str, Any], all_payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    row = payload["row"]
    fold_id = str(row.get("fold_id"))
    seed = int(row.get("seed"))
    past_frames: list[pd.DataFrame] = []
    fold_order = {"fold_1": 1, "fold_2": 2, "fold_3": 3}
    current_order = fold_order[fold_id]
    for item in all_payloads:
        item_row = item["row"]
        if int(item_row.get("seed")) != seed:
            continue
        if fold_order[str(item_row.get("fold_id"))] < current_order:
            past_frames.append(item["val"])
    source = "past_fold_validation"
    if not past_frames:
        past_frames = [payload["val"]]
        source = "cold_start_current_validation"
    fit_frame = pd.concat(past_frames, ignore_index=True)
    shift, metrics = cal.fit_shift_grid(fit_frame)
    return {
        "method": "walk_forward_lower_calibration",
        "params": {
            "global_shift": shift,
            "calibration_source": source,
            "past_validation_frame_count": len(past_frames),
        },
        "fit_metrics": metrics,
        "test_used_for_fit": False,
    }


def fit_payload_method(
    payload: dict[str, Any], method: str, all_payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    val = payload["val"]
    if method == "raw":
        return {"method": "raw", "params": {"global_shift": 0.0}, "test_used_for_fit": False}
    if method == "existing_stress_aware_lower_shift":
        fitted = cal.fit_method("stress_aware_lower_shift", val)
        fitted["method"] = method
        fitted["params"]["source"] = "cp178_cal_vix_or_realized_move_stress_rule"
        return fitted
    if method == "walk_forward_lower_calibration":
        return fit_walk_forward_shift(payload, all_payloads)
    if method == "adaptive_conformal_lower":
        return fit_adaptive_conformal(val, stress_aware=False)
    if method == "stress_aware_adaptive_conformal":
        fitted = fit_adaptive_conformal(val, stress_aware=True)
        fitted["params"]["stress_rule"] = payload.get("stress_rule")
        return fitted
    raise ValueError(f"지원하지 않는 method: {method}")


def apply_fitted_method(
    frame: pd.DataFrame,
    fitted: dict[str, Any],
    *,
    initial_history: list[float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[float]]:
    method = str(fitted["method"])
    params = dict(fitted.get("params") or {})
    if method == "raw":
        return apply_constant_shift(frame, 0.0, method, params)
    if method == "existing_stress_aware_lower_shift":
        adapted = {"method": "stress_aware_lower_shift", "params": params}
        calibrated = cal.apply_method(frame, adapted)
        mask = np.asarray(calibrated["calibrated_lower"] < frame["lower"], dtype=bool)
        calibrated["extension_applied"] = np.where(
            mask, frame["lower"] - calibrated["calibrated_lower"], 0.0
        )
        calibrated["method"] = method
        return (
            calibrated,
            date_rolling_rows(calibrated, method, params),
            date_lower_history(calibrated),
        )
    if method == "walk_forward_lower_calibration":
        return apply_constant_shift(frame, float(params.get("global_shift", 0.0)), method, params)
    if method in {"adaptive_conformal_lower", "stress_aware_adaptive_conformal"}:
        return apply_adaptive_shift(
            frame, method=method, params=params, initial_history=initial_history
        )
    raise ValueError(f"지원하지 않는 method: {method}")


def row_metrics(
    payload: dict[str, Any], fitted: dict[str, Any], all_payloads: list[dict[str, Any]]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    del all_payloads
    method = str(fitted["method"])
    val_cal, val_rolling, val_history = apply_fitted_method(
        payload["val"], fitted, initial_history=[]
    )
    test_cal, test_rolling, _ = apply_fitted_method(
        payload["test"], fitted, initial_history=val_history
    )
    val_metrics = cal.interval_metrics(val_cal)
    test_metrics = cal.interval_metrics(test_cal)
    row = payload["row"]
    out: dict[str, Any] = {
        "method": method,
        "fold_id": row.get("fold_id"),
        "seed": int(row.get("seed")),
        "candidate_id": TARGET_CANDIDATE_ID,
        "calibration_params": fitted.get("params") or {},
        "test_used_for_fit": False,
        "new_training": False,
        "save_run": False,
        "db_write": False,
        "inference_saved": False,
    }
    for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            out[f"{prefix}_{key}"] = value
    for key in [
        "lower_breach_rate",
        "upper_breach_rate",
        "coverage_abs_error",
        "asymmetric_interval_score",
        "p90_band_width",
        "band_width_ic",
        "downside_width_ic",
    ]:
        val = safe_float(val_metrics.get(key))
        test = safe_float(test_metrics.get(key))
        out[f"{key}_transfer_gap"] = test - val if val is not None and test is not None else None
    return out, val_cal, test_cal, val_rolling + test_rolling


def stress_calm_rows(
    method: str, fold_id: str, seed: int, test_cal: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket_value, group in test_cal.groupby("alt_stress_label", sort=True):
        metrics = cal.interval_metrics(group)
        rows.append(
            {
                "method": method,
                "fold_id": fold_id,
                "seed": seed,
                "bucket": "stress" if bool(bucket_value) else "calm",
                **metrics,
            }
        )
    return rows


def aggregate_summary(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_key = {(row["fold_id"], row["seed"]): row for row in fold_rows if row["method"] == "raw"}
    rows: list[dict[str, Any]] = []
    for method, group_rows in base.group_by(fold_rows, "method").items():
        row: dict[str, Any] = {"method": method, "fold_seed_count": len(group_rows)}
        for key, higher in [
            ("test_lower_breach_rate", False),
            ("test_upper_breach_rate", False),
            ("test_coverage_abs_error", False),
            ("test_asymmetric_interval_score", False),
            ("test_p90_band_width", False),
            ("test_band_width_ic", True),
            ("test_downside_width_ic", True),
        ]:
            values = [item.get(key) for item in group_rows]
            row[f"{key}_mean"] = base.mean(values)
            row[f"{key}_worst"] = base.worst(values, higher_is_better=higher)
        raw_pairs = []
        for item in group_rows:
            raw = raw_by_key.get((item["fold_id"], item["seed"]))
            if raw:
                raw_pairs.append((raw, item))
        if raw_pairs:
            raw_lower = np.mean([float(pair[0]["test_lower_breach_rate"]) for pair in raw_pairs])
            method_lower = np.mean([float(pair[1]["test_lower_breach_rate"]) for pair in raw_pairs])
            raw_cov = np.mean([float(pair[0]["test_coverage_abs_error"]) for pair in raw_pairs])
            method_cov = np.mean([float(pair[1]["test_coverage_abs_error"]) for pair in raw_pairs])
            row["raw_lower_breach_improvement_rate"] = (
                (raw_lower - method_lower) / raw_lower if raw_lower else None
            )
            row["raw_coverage_abs_error_improvement_rate"] = (
                (raw_cov - method_cov) / raw_cov if raw_cov else None
            )
        cal_row = load_cp178_cal_best()
        if cal_row:
            cal_lower = safe_float(cal_row.get("test_lower_breach_rate_mean"))
            cal_cov = safe_float(cal_row.get("test_coverage_abs_error_mean"))
            lower_mean = safe_float(row.get("test_lower_breach_rate_mean"))
            cov_mean = safe_float(row.get("test_coverage_abs_error_mean"))
            row["cp178_cal_lower_breach_improvement_rate"] = (
                (cal_lower - lower_mean) / cal_lower
                if cal_lower and lower_mean is not None
                else None
            )
            row["cp178_cal_coverage_abs_error_improvement_rate"] = (
                (cal_cov - cov_mean) / cal_cov if cal_cov and cov_mean is not None else None
            )
        row["decision_hint"] = decision_hint(row)
        row["selection_score"] = summary_score(row)
        rows.append(row)
    return sorted(rows, key=lambda item: safe_float(item.get("selection_score")) or 999.0)


def decision_hint(row: dict[str, Any]) -> str:
    lower_mean = safe_float(row.get("test_lower_breach_rate_mean"))
    lower_worst = safe_float(row.get("test_lower_breach_rate_worst"))
    cov_mean = safe_float(row.get("test_coverage_abs_error_mean"))
    cov_worst = safe_float(row.get("test_coverage_abs_error_worst"))
    p90_worst = safe_float(row.get("test_p90_band_width_worst"))
    downside = safe_float(row.get("test_downside_width_ic_mean"))
    interval = safe_float(row.get("test_asymmetric_interval_score_mean"))
    if None not in (lower_mean, lower_worst, cov_mean, cov_worst, p90_worst, downside, interval):
        if (
            0.09 <= lower_mean <= 0.11
            and lower_worst <= 0.13
            and cov_mean <= 0.03
            and cov_worst <= 0.055
            and p90_worst <= SUCCESS_P90_WIDTH_WORST
            and downside >= 0.04
            and interval <= 0.34
        ):
            return "PASS"
        if lower_mean <= 0.12 and cov_mean <= 0.05 and p90_worst <= 0.28 and downside >= 0.035:
            return "WARN"
    return "FAIL"


def summary_score(row: dict[str, Any]) -> float:
    lower_mean = safe_float(row.get("test_lower_breach_rate_mean")) or 1.0
    lower_worst = safe_float(row.get("test_lower_breach_rate_worst")) or 1.0
    cov_mean = safe_float(row.get("test_coverage_abs_error_mean")) or 1.0
    cov_worst = safe_float(row.get("test_coverage_abs_error_worst")) or 1.0
    p90_worst = safe_float(row.get("test_p90_band_width_worst")) or 1.0
    downside = safe_float(row.get("test_downside_width_ic_mean")) or 0.0
    interval = safe_float(row.get("test_asymmetric_interval_score_mean")) or 1.0
    return (
        4.0 * cov_mean
        + 1.5 * cov_worst
        + 5.0 * abs(lower_mean - Q_LOW)
        + 2.0 * max(0.0, lower_worst - 0.13)
        + 0.30 * interval
        + 1.0 * max(0.0, p90_worst - SUCCESS_P90_WIDTH_WORST)
        + 0.50 * max(0.0, 0.04 - downside)
    )


def load_cp178_cal_best() -> dict[str, Any] | None:
    if not CP178_CAL_SUMMARY.exists():
        return None
    df = pd.read_csv(CP178_CAL_SUMMARY)
    if df.empty:
        return None
    warn = df[df["method"].astype(str) == "stress_aware_lower_shift"]
    row = warn.iloc[0] if not warn.empty else df.iloc[0]
    return row.to_dict()


def load_cp153_reference() -> dict[str, Any] | None:
    if not CP153_1D_METRICS.exists():
        return None
    metrics = read_json(CP153_1D_METRICS)
    rows = metrics.get("aggregate_rows") or []
    if not rows:
        return None
    primary = [row for row in rows if row.get("candidate_id") == "tide_s60_q15_param"]
    row = primary[0] if primary else rows[0]
    return {
        "candidate_id": row.get("candidate_id"),
        "decision": row.get("decision"),
        "test_coverage_abs_error_mean": row.get("test_coverage_abs_error_mean"),
        "test_coverage_abs_error_worst": row.get("test_coverage_abs_error_worst"),
        "test_lower_breach_rate_mean": row.get("test_lower_breach_rate_mean"),
        "test_lower_breach_rate_worst": row.get("test_lower_breach_rate_worst"),
        "test_p90_band_width_worst": row.get("test_p90_band_width_worst"),
        "test_band_width_ic_mean": row.get("test_band_width_ic_mean"),
        "test_downside_width_ic_mean": row.get("test_downside_width_ic_mean"),
    }


def write_report(
    summary_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], metrics: dict[str, Any]
) -> None:
    best = summary_rows[0] if summary_rows else {}
    cal_best = load_cp178_cal_best() or {}
    cp153 = load_cp153_reference() or {}
    lines = [
        "# CP178-ALT 1W Band Adaptive Calibration",
        "",
        "이 CP는 새 후보를 찾는 실험이 아니라, 1W band rescue 후보가 제품 후보로 못 올라간 원인을 보정 계층에서 닫을 수 있는지 확인한 진단이다. 다음 학습 또는 calibration 방향은 이 결과 이후 결정한다.",
        "",
        "## 실행 계약",
        "",
        "- 대상 후보: tide_s60_q10_q90_param",
        "- 새 학습: 없음",
        "- save-run / DB write / inference 저장 / live fetch / EODHD fallback / composite: 없음",
        "- calibration과 stress rule은 validation에서만 fit",
        "- test 결과로 threshold/calibration 재선정 없음",
        "- adaptive rolling miss rate는 h4 라벨 지연을 피하기 위해 4개 asof-date lag를 둔 진단값으로 계산",
        "",
        "## Stress Label",
        "",
    ]
    if label_rows:
        rule_types = sorted({str(row.get("rule_type")) for row in label_rows})
        used_features = sorted(
            {feature for row in label_rows for feature in (row.get("used_features") or [])}
        )
        lines.extend(
            [
                f"- rule_type: {', '.join(rule_types)}",
                f"- used_features: {', '.join(used_features)}",
                f"- validation stress rate 평균: {fmt(base.mean([row.get('val_stress_rate') for row in label_rows]))}",
                f"- test stress rate 평균: {fmt(base.mean([row.get('test_stress_rate') for row in label_rows]))}",
                "- macro coverage가 충분한 fold/seed에서는 vol/drawdown/macro vote를 사용했고, 부족하면 price-only fallback을 쓰도록 고정했다.",
                "",
            ]
        )
    lines.extend(
        [
            "## 후보 비교",
            "",
            "| method | decision | lower mean/worst | coverage error mean/worst | p90 width worst | interval mean | downside width IC mean | CP178-CAL lower 개선율 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method")),
                    str(row.get("decision_hint")),
                    f"{fmt(row.get('test_lower_breach_rate_mean'))}/{fmt(row.get('test_lower_breach_rate_worst'))}",
                    f"{fmt(row.get('test_coverage_abs_error_mean'))}/{fmt(row.get('test_coverage_abs_error_worst'))}",
                    fmt(row.get("test_p90_band_width_worst")),
                    fmt(row.get("test_asymmetric_interval_score_mean")),
                    fmt(row.get("test_downside_width_ic_mean")),
                    fmt(row.get("cp178_cal_lower_breach_improvement_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 최선 후보",
            "",
            f"- best_method: {best.get('method')}",
            f"- decision: {best.get('decision_hint')}",
            f"- lower_breach mean/worst: {fmt(best.get('test_lower_breach_rate_mean'))} / {fmt(best.get('test_lower_breach_rate_worst'))}",
            f"- coverage_abs_error mean/worst: {fmt(best.get('test_coverage_abs_error_mean'))} / {fmt(best.get('test_coverage_abs_error_worst'))}",
            f"- p90_band_width worst: {fmt(best.get('test_p90_band_width_worst'))}",
            f"- downside_width_ic mean: {fmt(best.get('test_downside_width_ic_mean'))}",
            "",
            "## CP178-CAL 대비",
            "",
            f"- CP178-CAL best: {cal_best.get('method', '')}",
            f"- CP178-CAL lower mean/worst: {fmt(cal_best.get('test_lower_breach_rate_mean'))} / {fmt(cal_best.get('test_lower_breach_rate_worst'))}",
            f"- CP178-CAL coverage_abs_error mean/worst: {fmt(cal_best.get('test_coverage_abs_error_mean'))} / {fmt(cal_best.get('test_coverage_abs_error_worst'))}",
            "- ALT는 CP178-CAL의 best lower-only shift와 같은 기준에서 비교했다.",
            "",
            "## 1D CP153 참고",
            "",
        ]
    )
    if cp153:
        lines.extend(
            [
                f"- 1D reference candidate: {cp153.get('candidate_id')} / {cp153.get('decision')}",
                f"- 1D lower_breach mean/worst: {fmt(cp153.get('test_lower_breach_rate_mean'))} / {fmt(cp153.get('test_lower_breach_rate_worst'))}",
                f"- 1D coverage_abs_error mean/worst: {fmt(cp153.get('test_coverage_abs_error_mean'))} / {fmt(cp153.get('test_coverage_abs_error_worst'))}",
                "- 1D에서도 worst fold lower breach가 낮지 않은 편이므로, 기준 완화 가능성은 사용자 판단 보조로만 기록한다. 이 보고서는 기준 완화를 결정하지 않는다.",
                "",
            ]
        )
    else:
        lines.extend(["- 1D CP153 metrics 파일을 찾지 못해 비교를 생략했다.", ""])
    lines.extend(
        [
            "## 판정",
            "",
            f"- final_decision: {metrics.get('final_decision')}",
            f"- reason: {metrics.get('decision_reason')}",
            "",
            "ALT가 실패 또는 WARN이면 다음 후보는 CP178-LOSS의 stress-weighted lower pinball 및 stress feature pack이다. 이번 CP에서는 새 학습을 실행하지 않았다.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_decision(summary_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not summary_rows:
        return "FAIL", "평가 가능한 ALT 후보가 없다."
    best = summary_rows[0]
    if best.get("decision_hint") == "PASS":
        return (
            "PASS",
            "worst fold lower breach와 coverage, width, downside IC 기준을 모두 충족했다.",
        )
    lower_worst = safe_float(best.get("test_lower_breach_rate_worst"))
    cal_improvement = safe_float(best.get("cp178_cal_lower_breach_improvement_rate"))
    if best.get("decision_hint") == "WARN" or (
        cal_improvement is not None and cal_improvement > 0.0
    ):
        return (
            "WARN",
            f"ALT는 효과가 있으나 worst fold lower breach={fmt(lower_worst)}로 성공 기준을 완전히 닫지 못했다.",
        )
    return "FAIL", "ALT 방식으로는 CP178-CAL 대비 의미 있는 worst fold 개선을 만들지 못했다."


def run() -> dict[str, Any]:
    rows = cal.load_target_rows()
    raw_payloads = cal.collect_raw_frames(rows)
    payloads, label_rows = label_payloads(raw_payloads)
    methods = [
        "raw",
        "existing_stress_aware_lower_shift",
        "walk_forward_lower_calibration",
        "adaptive_conformal_lower",
        "stress_aware_adaptive_conformal",
    ]
    fold_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    stress_calm: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for payload in payloads:
        for method in methods:
            fitted = fit_payload_method(payload, method, payloads)
            result, val_cal, test_cal, rolling = row_metrics(payload, fitted, payloads)
            fold_rows.append(result)
            params_rows.append(
                {
                    "method": method,
                    "fold_id": result["fold_id"],
                    "seed": result["seed"],
                    "params": fitted.get("params") or {},
                    "fit_metrics": fitted.get("fit_metrics") or {},
                    "fit_score": fitted.get("fit_score"),
                    "test_used_for_fit": False,
                }
            )
            rolling_rows.extend(rolling)
            bootstrap_rows.append(
                cal.bootstrap_ci(method, str(result["fold_id"]), int(result["seed"]), test_cal)
            )
            stress_calm.extend(
                stress_calm_rows(method, str(result["fold_id"]), int(result["seed"]), test_cal)
            )
            for key in [
                "lower_breach_rate",
                "upper_breach_rate",
                "coverage_abs_error",
                "asymmetric_interval_score",
                "p90_band_width",
                "band_width_ic",
                "downside_width_ic",
            ]:
                transfer_rows.append(
                    {
                        "method": method,
                        "fold_id": result["fold_id"],
                        "seed": result["seed"],
                        "metric": key,
                        "validation": result.get(f"val_{key}"),
                        "test": result.get(f"test_{key}"),
                        "test_minus_validation": result.get(f"{key}_transfer_gap"),
                        "test_used_for_fit": False,
                    }
                )
            del val_cal
    summary_rows = aggregate_summary(fold_rows)
    decision, reason = final_decision(summary_rows)
    metrics = {
        "cp": "CP178-ALT",
        "created_at_utc": now_utc(),
        "candidate_id": TARGET_CANDIDATE_ID,
        "source_data_hash": "90666b44cbfb8e5c",
        "timeframe": "1W",
        "horizon": 4,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "new_training": False,
        "save_run": False,
        "db_write": False,
        "inference_saved": False,
        "live_fetch": False,
        "eodhd_fallback": False,
        "composite": False,
        "test_used_for_calibration_selection": False,
        "label_availability_lag_dates": LABEL_AVAILABILITY_LAG_DATES,
        "final_decision": decision,
        "decision_reason": reason,
        "success_criteria": {
            "worst_fold_lower_breach_lte": 0.13,
            "lower_breach_mean_range": [0.09, 0.11],
            "coverage_abs_error_mean_lte": 0.03,
            "coverage_abs_error_worst_lte": 0.055,
            "p90_width_worst_lte": SUCCESS_P90_WIDTH_WORST,
            "downside_width_ic_gte": 0.04,
        },
        "summary_rows": summary_rows,
        "fold_rows": fold_rows,
        "stress_label_rows": label_rows,
        "cp178_cal_reference": load_cp178_cal_best(),
        "cp153_1d_reference": load_cp153_reference(),
    }
    write_rows(SUMMARY_CSV, summary_rows)
    write_rows(FOLD_CSV, fold_rows)
    write_rows(PARAMS_CSV, params_rows)
    write_rows(STRESS_LABEL_CSV, label_rows)
    write_rows(ROLLING_CSV, rolling_rows)
    write_rows(BOOTSTRAP_CSV, bootstrap_rows)
    write_rows(STRESS_CALM_CSV, stress_calm)
    write_rows(TRANSFER_CSV, transfer_rows)
    write_json(METRICS_PATH, metrics)
    write_report(summary_rows, label_rows, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CP178-ALT 1W band adaptive calibration diagnostic"
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run 필요")
    metrics = run()
    print(
        json.dumps(
            {"final_decision": metrics["final_decision"], "report": str(REPORT_PATH)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
