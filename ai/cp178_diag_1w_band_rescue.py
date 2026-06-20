from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai import cp178_bm_1w_band_500_stage3_5 as base

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
RESCUE_METRICS = DOCS_DIR / "cp178_bm_1w_band_500_rescue_expansion_metrics.json"
REPORT_PATH = DOCS_DIR / "cp178_diag_1w_band_rescue_report.md"
METRICS_PATH = DOCS_DIR / "cp178_diag_1w_band_rescue_metrics.json"
FOLD_STRESS_CSV = DOCS_DIR / "cp178_diag_1w_band_rescue_fold_stress_profile.csv"
WIDTH_CSV = DOCS_DIR / "cp178_diag_1w_band_rescue_width_adequacy.csv"
BOOTSTRAP_CSV = DOCS_DIR / "cp178_diag_1w_band_rescue_bootstrap_asymmetry.csv"
REGIME_CSV = DOCS_DIR / "cp178_diag_1w_band_rescue_regime_breach.csv"
TRANSFER_CSV = DOCS_DIR / "cp178_diag_1w_band_rescue_calibration_transfer.csv"
TICKER_CSV = DOCS_DIR / "cp178_diag_1w_band_rescue_ticker_concentration.csv"

TARGET_CANDIDATE_ID = "tide_s60_q10_q90_param"
Q_LOW = 0.10
Q_HIGH = 0.90
N_BOOTSTRAP = 200


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    base.write_json(path, payload)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_rows(path, rows)


def safe_float(value: Any) -> float | None:
    return base.safe_float(value)


def fmt(value: Any) -> str:
    value_f = safe_float(value)
    return "" if value_f is None else f"{value_f:.6f}"


def quantile(values: pd.Series | np.ndarray, q: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def mean(values: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def rate(mask: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    frame = pd.DataFrame(
        {"x": np.asarray(x, dtype=np.float64), "y": np.asarray(y, dtype=np.float64)}
    )
    frame = frame[np.isfinite(frame["x"]) & np.isfinite(frame["y"])]
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    value = frame["x"].corr(frame["y"], method="spearman")
    return float(value) if pd.notna(value) else None


def load_target_rows() -> list[dict[str, Any]]:
    metrics = read_json(RESCUE_METRICS)
    rows = [
        row
        for row in metrics.get("r3", {}).get("rows", [])
        if row.get("candidate_id") == TARGET_CANDIDATE_ID
    ]
    if not rows:
        raise RuntimeError(f"{TARGET_CANDIDATE_ID} true WF row를 찾지 못했다.")
    return rows


def build_candidate() -> base.Candidate:
    return base.Candidate(
        candidate_id=TARGET_CANDIDATE_ID,
        model="tide",
        family="tide",
        seq_len=60,
        q_low=Q_LOW,
        q_high=Q_HIGH,
        band_mode="param",
    )


def fold_by_id(fold_id: str) -> base.FoldSpec:
    for fold in base.FOLDS:
        if fold.fold_id == fold_id:
            return fold
    raise KeyError(f"fold 없음: {fold_id}")


def load_context() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W_500.parquet"
    cols = [
        "ticker",
        "date",
        "vix_close",
        "yield_spread",
        "credit_spread_hy",
        "has_macro",
        "has_breadth",
    ]
    df = pd.read_parquet(
        path, columns=[col for col in cols if col in pd.read_parquet(path).columns]
    )
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["vix_close", "yield_spread", "credit_spread_hy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_delta"] = df.sort_values(["ticker", "date"]).groupby("ticker")[col].diff()
    return df


def flatten_prediction(
    pred: dict[str, Any],
    calibration: dict[str, Any],
    row: dict[str, Any],
    split: str,
    context: pd.DataFrame,
) -> pd.DataFrame:
    lower, upper = base.apply_calibration(pred, calibration)
    actual = np.asarray(pred["actual"], dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    width = upper - lower
    n_samples, horizon = actual.shape
    metadata = pred["metadata"].reset_index(drop=True).copy()
    flat = pd.DataFrame(
        {
            "fold_id": row.get("fold_id"),
            "seed": int(row.get("seed")),
            "split": split,
            "ticker": np.repeat(metadata["ticker"].astype(str).to_numpy(), horizon),
            "asof_date": np.repeat(pd.to_datetime(metadata["asof_date"]).to_numpy(), horizon),
            "horizon_step": np.tile(np.arange(1, horizon + 1), n_samples),
            "actual_return": actual.reshape(-1),
            "lower": lower.reshape(-1),
            "upper": upper.reshape(-1),
            "predicted_width": width.reshape(-1),
        }
    )
    flat["realized_move"] = flat["actual_return"].abs()
    flat["width_ratio"] = flat["predicted_width"] / (flat["realized_move"] + 1e-8)
    flat["lower_breach"] = flat["actual_return"] < flat["lower"]
    flat["upper_breach"] = flat["actual_return"] > flat["upper"]
    flat["covered"] = ~(flat["lower_breach"] | flat["upper_breach"])
    flat["lower_width"] = ((flat["lower"] + flat["upper"]) / 2.0) - flat["lower"]
    flat["downside_magnitude"] = np.maximum(-flat["actual_return"], 0.0)
    ctx = context.rename(columns={"date": "asof_date"})
    flat = flat.merge(ctx, how="left", on=["ticker", "asof_date"])
    return flat


def collect_frames(
    rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    candidate = build_candidate()
    context = load_context()
    frames: list[pd.DataFrame] = []
    transfer_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        fold = fold_by_id(str(row["fold_id"]))
        bundles = base.build_fold_bundles(candidate, fold)
        checkpoint_path = row.get("checkpoint_path")
        calibration = {
            "method": row.get("calibration_method"),
            "params": row.get("calibration_params") or {},
        }
        val_pred = base.collect_predictions_for_bundle(checkpoint_path, bundles["val_raw"])
        test_pred = base.collect_predictions_for_bundle(checkpoint_path, bundles["test_raw"])
        val_flat = flatten_prediction(val_pred, calibration, row, "val", context)
        test_flat = flatten_prediction(test_pred, calibration, row, "test", context)
        frames.append(test_flat)
        transfer_rows.append(calibration_transfer_row(row, val_flat, test_flat))
        detail_rows.append(
            {
                "row": row,
                "fold_event_profile": row.get("fold_event_profile"),
                "split_diag": row.get("split_diag"),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(transfer_rows), detail_rows


def metric_from_flat(
    df: pd.DataFrame, q_low: float = Q_LOW, q_high: float = Q_HIGH
) -> dict[str, Any]:
    total = len(df)
    if total == 0:
        return {}
    lower = float(df["lower_breach"].mean())
    upper = float(df["upper_breach"].mean())
    coverage = float(df["covered"].mean())
    nominal = q_high - q_low
    return {
        "row_count": total,
        "nominal_coverage": nominal,
        "empirical_coverage": coverage,
        "coverage_abs_error": abs(coverage - nominal),
        "lower_breach_rate": lower,
        "upper_breach_rate": upper,
        "lower_breach_abs_error": abs(lower - q_low),
        "upper_breach_abs_error": abs(upper - (1.0 - q_high)),
        "median_band_width": quantile(df["predicted_width"], 0.5),
        "p90_band_width": quantile(df["predicted_width"], 0.9),
        "band_width_ic": spearman(df["predicted_width"], df["realized_move"]),
        "downside_width_ic": spearman(df["lower_width"], df["downside_magnitude"]),
    }


def calibration_transfer_row(
    row: dict[str, Any], val_flat: pd.DataFrame, test_flat: pd.DataFrame
) -> dict[str, Any]:
    val_metrics = metric_from_flat(val_flat)
    test_metrics = metric_from_flat(test_flat)
    out = {
        "fold_id": row.get("fold_id"),
        "seed": row.get("seed"),
        "calibration_method": row.get("calibration_method"),
        "calibration_params": row.get("calibration_params") or {},
    }
    for key in [
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "band_width_ic",
        "downside_width_ic",
        "p90_band_width",
    ]:
        out[f"val_{key}"] = val_metrics.get(key)
        out[f"test_{key}"] = test_metrics.get(key)
        val = safe_float(val_metrics.get(key))
        test = safe_float(test_metrics.get(key))
        out[f"{key}_test_minus_val"] = test - val if val is not None and test is not None else None
    return out


def fold_stress_profile(
    test_flat: pd.DataFrame, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    thresholds = {
        str(row["fold_id"]): (row.get("fold_event_profile") or {}).get("train_quantile_low")
        for row in rows
        if int(row.get("seed")) == 42
    }
    profiles: list[dict[str, Any]] = []
    for fold_id, group in test_flat[test_flat["seed"] == 42].groupby("fold_id", sort=True):
        threshold = safe_float(thresholds.get(str(fold_id)))
        severe_mask = (
            group["actual_return"] <= threshold
            if threshold is not None
            else group["actual_return"] <= group["actual_return"].quantile(Q_LOW)
        )
        unique_dates = group.drop_duplicates(["asof_date"])
        row = {
            "fold_id": fold_id,
            "seed_basis": 42,
            "target_points": len(group),
            "date_min": str(pd.to_datetime(group["asof_date"]).min().date()),
            "date_max": str(pd.to_datetime(group["asof_date"]).max().date()),
            "realized_weekly_vol_mean": mean(group["realized_move"]),
            "realized_weekly_vol_p90": quantile(group["realized_move"], 0.9),
            "actual_return_abs_p90": quantile(group["actual_return"].abs(), 0.9),
            "severe_down_threshold": threshold,
            "severe_down_week_count": int(severe_mask.sum()),
            "severe_down_rate": float(severe_mask.mean()),
            "lower_breach_count": int(group["lower_breach"].sum()),
            "upper_breach_count": int(group["upper_breach"].sum()),
            "lower_breach_rate": float(group["lower_breach"].mean()),
            "upper_breach_rate": float(group["upper_breach"].mean()),
            "vix_close_mean": mean(unique_dates["vix_close"])
            if "vix_close" in unique_dates
            else None,
            "vix_close_p90": quantile(unique_dates["vix_close"], 0.9)
            if "vix_close" in unique_dates
            else None,
            "vix_delta_mean": mean(unique_dates["vix_close_delta"])
            if "vix_close_delta" in unique_dates
            else None,
            "vix_delta_abs_p90": quantile(unique_dates["vix_close_delta"].abs(), 0.9)
            if "vix_close_delta" in unique_dates
            else None,
            "credit_spread_hy_mean": mean(unique_dates["credit_spread_hy"])
            if "credit_spread_hy" in unique_dates
            else None,
            "credit_spread_hy_p90": quantile(unique_dates["credit_spread_hy"], 0.9)
            if "credit_spread_hy" in unique_dates
            else None,
            "credit_spread_hy_delta_mean": mean(unique_dates["credit_spread_hy_delta"])
            if "credit_spread_hy_delta" in unique_dates
            else None,
            "credit_spread_hy_delta_abs_p90": quantile(
                unique_dates["credit_spread_hy_delta"].abs(), 0.9
            )
            if "credit_spread_hy_delta" in unique_dates
            else None,
        }
        profiles.append(row)
    vol_median = np.nanmedian([row["realized_weekly_vol_p90"] for row in profiles])
    severe_median = np.nanmedian([row["severe_down_rate"] for row in profiles])
    vix_median = np.nanmedian(
        [row["vix_close_mean"] for row in profiles if row["vix_close_mean"] is not None]
    )
    for row in profiles:
        stress_flags = [
            safe_float(row.get("realized_weekly_vol_p90")) is not None
            and float(row["realized_weekly_vol_p90"]) >= vol_median,
            safe_float(row.get("severe_down_rate")) is not None
            and float(row["severe_down_rate"]) >= severe_median,
            safe_float(row.get("vix_close_mean")) is not None
            and float(row["vix_close_mean"]) >= vix_median,
        ]
        row["stress_flag_count"] = int(sum(stress_flags))
        row["stress_classification"] = "stress" if row["stress_flag_count"] >= 2 else "calm"
    return profiles


def width_adequacy_rows(test_flat: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = list(test_flat.groupby(["fold_id", "seed"], sort=True)) + [(("ALL", "ALL"), test_flat)]
    for (fold_id, seed), group in groups:
        lower_miss = group[group["lower_breach"]]
        upper_miss = group[group["upper_breach"]]
        rows.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "row_count": len(group),
                "width_ratio_mean": mean(group["width_ratio"]),
                "width_ratio_p10": quantile(group["width_ratio"], 0.1),
                "width_ratio_p50": quantile(group["width_ratio"], 0.5),
                "width_ratio_p90": quantile(group["width_ratio"], 0.9),
                "ratio_lt_1_rate": rate(group["width_ratio"] < 1.0),
                "lower_miss_count": len(lower_miss),
                "lower_miss_width_ratio_p10": quantile(lower_miss["width_ratio"], 0.1),
                "lower_miss_width_ratio_p50": quantile(lower_miss["width_ratio"], 0.5),
                "lower_miss_width_ratio_p90": quantile(lower_miss["width_ratio"], 0.9),
                "lower_miss_ratio_lt_1_rate": rate(lower_miss["width_ratio"] < 1.0)
                if len(lower_miss)
                else None,
                "upper_miss_count": len(upper_miss),
                "upper_miss_width_ratio_p50": quantile(upper_miss["width_ratio"], 0.5),
                "specific_narrowness": "local_lower_miss"
                if len(lower_miss)
                and rate(lower_miss["width_ratio"] < 1.0)
                and rate(lower_miss["width_ratio"] < 1.0) > 0.65
                else "broad_or_mixed",
            }
        )
    return rows


def bootstrap_asymmetry_rows(test_flat: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = list(test_flat.groupby(["fold_id", "seed"], sort=True)) + [(("ALL", "ALL"), test_flat)]
    rng = np.random.default_rng(178)
    for (fold_id, seed), group in groups:
        diff = (
            group["lower_breach"].astype(float).to_numpy()
            - group["upper_breach"].astype(float).to_numpy()
        )
        if diff.size == 0:
            continue
        boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for idx in range(N_BOOTSTRAP):
            sample = rng.choice(diff, size=diff.size, replace=True)
            boot[idx] = sample.mean()
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        observed = float(diff.mean())
        includes_zero = bool(ci_low <= 0.0 <= ci_high)
        if includes_zero:
            verdict = "symmetric_width_noise_possible"
        elif ci_low > 0:
            verdict = "lower_specific_breach"
        else:
            verdict = "upper_specific_breach"
        rows.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "row_count": diff.size,
                "lower_breach_rate": float(group["lower_breach"].mean()),
                "upper_breach_rate": float(group["upper_breach"].mean()),
                "lower_minus_upper": observed,
                "bootstrap_n": N_BOOTSTRAP,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "ci_includes_zero": includes_zero,
                "verdict": verdict,
            }
        )
    return rows


def regime_rows(test_flat: pd.DataFrame) -> list[dict[str, Any]]:
    frame = test_flat.copy()
    abs_cut = frame["realized_move"].quantile(0.5)
    stress_cut = frame["vix_close"].quantile(0.75) if "vix_close" in frame else np.nan
    frame["vol_regime"] = np.where(frame["realized_move"] >= abs_cut, "high_vol", "low_vol")
    frame["direction_regime"] = np.where(frame["actual_return"] < 0, "falling", "rising")
    if math.isfinite(stress_cut):
        frame["stress_regime"] = np.where(frame["vix_close"] >= stress_cut, "stress", "calm")
    else:
        frame["stress_regime"] = np.where(
            frame["realized_move"] >= frame["realized_move"].quantile(0.75), "stress", "calm"
        )
    rows: list[dict[str, Any]] = []
    for regime_col in ["vol_regime", "direction_regime", "stress_regime"]:
        for (fold_id, seed, bucket), group in frame.groupby(
            ["fold_id", "seed", regime_col], sort=True
        ):
            metrics = metric_from_flat(group)
            metrics.update(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "regime_type": regime_col,
                    "regime_bucket": bucket,
                }
            )
            rows.append(metrics)
    for regime_col in ["vol_regime", "direction_regime", "stress_regime"]:
        for bucket, group in frame.groupby(regime_col, sort=True):
            metrics = metric_from_flat(group)
            metrics.update(
                {
                    "fold_id": "ALL",
                    "seed": "ALL",
                    "regime_type": regime_col,
                    "regime_bucket": bucket,
                }
            )
            rows.append(metrics)
    return rows


def transfer_rows_with_groups(transfer: pd.DataFrame) -> list[dict[str, Any]]:
    rows = transfer.to_dict("records")
    for method, group in transfer.groupby("calibration_method", sort=True):
        row = {
            "fold_id": "ALL",
            "seed": "ALL",
            "calibration_method": method,
            "calibration_params": "GROUP_MEAN",
            "group_count": len(group),
        }
        for key in [
            "coverage_abs_error",
            "lower_breach_rate",
            "upper_breach_rate",
            "band_width_ic",
            "downside_width_ic",
            "p90_band_width",
        ]:
            row[f"{key}_test_minus_val_mean"] = mean(group[f"{key}_test_minus_val"])
            row[f"{key}_test_minus_val_p90_abs"] = quantile(
                group[f"{key}_test_minus_val"].abs(), 0.9
            )
        rows.append(row)
    return rows


def ticker_concentration_rows(test_flat: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    high_vol_cut = test_flat["realized_move"].quantile(0.75)
    event_defs = {
        "any_breach": test_flat["lower_breach"] | test_flat["upper_breach"],
        "lower_breach": test_flat["lower_breach"],
        "high_vol": test_flat["realized_move"] >= high_vol_cut,
    }
    for event_type, mask in event_defs.items():
        event_df = test_flat[mask]
        total = len(event_df)
        counts = event_df.groupby("ticker").size().sort_values(ascending=False)
        top10_share = float(counts.head(10).sum() / total) if total else None
        rows.append(
            {
                "event_type": event_type,
                "ticker": "__TOTAL__",
                "count": total,
                "share": 1.0 if total else 0.0,
                "top10_share": top10_share,
            }
        )
        for ticker, count in counts.head(20).items():
            rows.append(
                {
                    "event_type": event_type,
                    "ticker": ticker,
                    "count": int(count),
                    "share": float(count / total) if total else None,
                    "top10_share": top10_share,
                }
            )
    return rows


def decide_branches(
    fold_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    regime: list[dict[str, Any]],
    transfer: list[dict[str, Any]],
    width_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    stress_folds = [
        row["fold_id"] for row in fold_rows if row.get("stress_classification") == "stress"
    ]
    overall_boot = next((row for row in bootstrap_rows if row["fold_id"] == "ALL"), {})
    overall_width = next((row for row in width_rows if row["fold_id"] == "ALL"), {})
    transfer_gaps = [
        abs(safe_float(row.get("coverage_abs_error_test_minus_val")) or 0.0)
        for row in transfer
        if row.get("fold_id") != "ALL"
    ]
    max_transfer_gap = max(transfer_gaps) if transfer_gaps else None
    downside_rows = [
        row
        for row in regime
        if row.get("regime_type") == "direction_regime"
        and row.get("regime_bucket") == "falling"
        and row.get("fold_id") == "ALL"
    ]
    falling_downside_ic = (
        safe_float(downside_rows[0].get("downside_width_ic")) if downside_rows else None
    )
    branches = {
        "A_stress_fold_confirmed": {
            "active": set(stress_folds) >= {"fold_1", "fold_3"},
            "stress_folds": stress_folds,
        },
        "B_downside_signal_weak": {
            "active": falling_downside_ic is not None and falling_downside_ic < 0.03,
            "falling_downside_width_ic": falling_downside_ic,
        },
        "C_asymmetry_noise_or_symmetric_width": {
            "active": bool(overall_boot.get("ci_includes_zero")),
            "bootstrap_verdict": overall_boot.get("verdict"),
        },
        "D_validation_test_transfer_large": {
            "active": max_transfer_gap is not None and max_transfer_gap > 0.05,
            "max_coverage_abs_error_transfer_gap": max_transfer_gap,
        },
        "width_narrowness": {
            "ratio_lt_1_rate": overall_width.get("ratio_lt_1_rate"),
            "lower_miss_ratio_lt_1_rate": overall_width.get("lower_miss_ratio_lt_1_rate"),
            "active": safe_float(overall_width.get("lower_miss_ratio_lt_1_rate")) is not None
            and float(overall_width["lower_miss_ratio_lt_1_rate"]) > 0.65,
        },
    }
    return branches


def write_report(metrics: dict[str, Any]) -> None:
    branches = metrics["branch_decisions"]
    fold_rows = metrics["fold_stress_profile"]
    width_overall = next((row for row in metrics["width_adequacy"] if row["fold_id"] == "ALL"), {})
    boot_overall = next(
        (row for row in metrics["bootstrap_asymmetry"] if row["fold_id"] == "ALL"), {}
    )
    transfer_group = [row for row in metrics["calibration_transfer"] if row.get("fold_id") == "ALL"]
    lines = [
        "# CP178-DIAG 1W Band Rescue 원인 진단 보고서",
        "",
        "이 CP는 새 후보를 찾는 실험이 아니라, 1W band rescue 후보가 제품 후보로 못 올라간 원인을 분해하는 진단이다. 다음 학습 또는 calibration 방향은 이 결과 이후 결정한다.",
        "",
        "## 범위",
        "",
        "- 대상: `tide_s60_q10_q90_param` true WF 결과",
        "- 새 학습 없음, save-run 없음, DB write 없음, inference 저장 없음",
        "- test 결과로 calibration 재선정 없음",
        "- fold별 prediction은 checkpoint에서 read-only로 재구성했고 per-row prediction artifact는 저장하지 않았다.",
        "",
        "## Fold Stress Profile",
        "",
        "| fold | class | vol_mean | abs_p90 | severe_down_rate | lower | upper | vix_mean | vix_delta_abs_p90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row.get('fold_id')} | {row.get('stress_classification')} | {fmt(row.get('realized_weekly_vol_mean'))} | {fmt(row.get('actual_return_abs_p90'))} | {fmt(row.get('severe_down_rate'))} | {fmt(row.get('lower_breach_rate'))} | {fmt(row.get('upper_breach_rate'))} | {fmt(row.get('vix_close_mean'))} | {fmt(row.get('vix_delta_abs_p90'))} |"
        )
    lines.extend(
        [
            "",
            "## Width Adequacy",
            "",
            f"- 전체 width_ratio p50: `{fmt(width_overall.get('width_ratio_p50'))}`",
            f"- 전체 ratio < 1 비율: `{fmt(width_overall.get('ratio_lt_1_rate'))}`",
            f"- lower miss 구간 ratio < 1 비율: `{fmt(width_overall.get('lower_miss_ratio_lt_1_rate'))}`",
            "",
            "## Lower/Upper Asymmetry Bootstrap",
            "",
            f"- lower-upper 차이: `{fmt(boot_overall.get('lower_minus_upper'))}`",
            f"- 95% CI: `{fmt(boot_overall.get('ci95_low'))}` ~ `{fmt(boot_overall.get('ci95_high'))}`",
            f"- verdict: `{boot_overall.get('verdict')}`",
            "",
            "## Calibration Transfer",
            "",
        ]
    )
    for row in transfer_group:
        lines.append(
            f"- `{row.get('calibration_method')}`: coverage gap mean `{fmt(row.get('coverage_abs_error_test_minus_val_mean'))}`, p90_abs `{fmt(row.get('coverage_abs_error_test_minus_val_p90_abs'))}`"
        )
    lines.extend(
        [
            "",
            "## 판정 분기",
            "",
            f"- A stress/fold 문제: `{branches['A_stress_fold_confirmed']['active']}` / stress folds `{branches['A_stress_fold_confirmed']['stress_folds']}`",
            f"- B downside signal 부족: `{branches['B_downside_signal_weak']['active']}` / falling downside_width_ic `{fmt(branches['B_downside_signal_weak']['falling_downside_width_ic'])}`",
            f"- C symmetric width/noise: `{branches['C_asymmetry_noise_or_symmetric_width']['active']}` / `{branches['C_asymmetry_noise_or_symmetric_width']['bootstrap_verdict']}`",
            f"- D validation-test transfer 문제: `{branches['D_validation_test_transfer_large']['active']}` / max gap `{fmt(branches['D_validation_test_transfer_large']['max_coverage_abs_error_transfer_gap'])}`",
            f"- width narrowness: `{branches['width_narrowness']['active']}` / lower miss ratio<1 `{fmt(branches['width_narrowness']['lower_miss_ratio_lt_1_rate'])}`",
            "",
            "## 다음 방향",
            "",
        ]
    )
    if branches["A_stress_fold_confirmed"]["active"]:
        lines.append("- fold 1/3이 stress로 확인되면 다음은 regime-aware calibration이다.")
    if branches["B_downside_signal_weak"]["active"]:
        lines.append(
            "- downside_width_ic가 핵심 약점이면 downside_vol_ratio_20w, max_weekly_drawdown_20w, negative_week_count_20w 같은 1W downside-specific feature가 필요하다."
        )
    if branches["C_asymmetry_noise_or_symmetric_width"]["active"]:
        lines.append(
            "- lower/upper asymmetry가 noise면 symmetric width expansion 또는 conformal calibration을 우선한다."
        )
    if branches["D_validation_test_transfer_large"]["active"]:
        lines.append("- validation-test transfer가 크면 conformal prediction wrapper가 우선이다.")
    if not any(
        branch["active"]
        for key, branch in branches.items()
        if isinstance(branch, dict) and key.startswith(("A_", "B_", "C_", "D_"))
    ):
        lines.append("- 단일 원인보다 stress와 fold별 calibration 안정성의 복합 문제로 본다.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    rows = load_target_rows()
    test_flat, transfer_df, detail_rows = collect_frames(rows)
    fold_rows = fold_stress_profile(test_flat, rows)
    width_rows = width_adequacy_rows(test_flat)
    bootstrap_rows = bootstrap_asymmetry_rows(test_flat)
    regime = regime_rows(test_flat)
    transfer = transfer_rows_with_groups(transfer_df)
    concentration = ticker_concentration_rows(test_flat)
    branch_decisions = decide_branches(fold_rows, bootstrap_rows, regime, transfer, width_rows)
    metrics = {
        "cp": "CP178-DIAG",
        "created_at_utc": now_utc(),
        "target_candidate_id": TARGET_CANDIDATE_ID,
        "source_metrics": str(RESCUE_METRICS),
        "policy": {
            "new_training": False,
            "save_run": False,
            "db_write": False,
            "inference_saved": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "composite": False,
            "test_used_for_calibration_reselection": False,
            "per_row_prediction_artifact_saved": False,
        },
        "fold_stress_profile": fold_rows,
        "width_adequacy": width_rows,
        "bootstrap_asymmetry": bootstrap_rows,
        "regime_breach": regime,
        "calibration_transfer": transfer,
        "ticker_concentration": concentration,
        "branch_decisions": branch_decisions,
        "details": detail_rows,
    }
    write_rows(FOLD_STRESS_CSV, fold_rows)
    write_rows(WIDTH_CSV, width_rows)
    write_rows(BOOTSTRAP_CSV, bootstrap_rows)
    write_rows(REGIME_CSV, regime)
    write_rows(TRANSFER_CSV, transfer)
    write_rows(TICKER_CSV, concentration)
    write_json(METRICS_PATH, metrics)
    write_report(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP178-DIAG 1W band rescue diagnostics")
    return parser.parse_args()


def main() -> None:
    parse_args()
    metrics = run()
    summary = {
        "cp": metrics["cp"],
        "target_candidate_id": TARGET_CANDIDATE_ID,
        "branch_decisions": metrics["branch_decisions"],
        "outputs": {
            "report": str(REPORT_PATH),
            "metrics": str(METRICS_PATH),
            "fold_stress_profile_csv": str(FOLD_STRESS_CSV),
            "width_adequacy_csv": str(WIDTH_CSV),
            "bootstrap_asymmetry_csv": str(BOOTSTRAP_CSV),
            "regime_breach_csv": str(REGIME_CSV),
            "calibration_transfer_csv": str(TRANSFER_CSV),
            "ticker_concentration_csv": str(TICKER_CSV),
        },
    }
    print(json.dumps(base.clean_json(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
