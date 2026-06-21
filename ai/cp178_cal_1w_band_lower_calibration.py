from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai import cp178_bm_1w_band_500_stage3_5 as base
from ai import cp178_diag_1w_band_rescue as diag

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

REPORT_PATH = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_report.md"
METRICS_PATH = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_metrics.json"
SUMMARY_CSV = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_summary.csv"
FOLD_CSV = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_fold_results.csv"
PARAMS_CSV = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_params.csv"
BOOTSTRAP_CSV = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_bootstrap_ci.csv"
REGIME_CSV = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_stress_calm.csv"
TRANSFER_CSV = DOCS_DIR / "cp178_cal_1w_band_lower_calibration_transfer.csv"

TARGET_CANDIDATE_ID = "tide_s60_q10_q90_param"
Q_LOW = 0.10
Q_HIGH = 0.90
N_BOOTSTRAP = 200


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    base.write_json(path, payload)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_rows(path, rows)


def safe_float(value: Any) -> float | None:
    return base.safe_float(value)


def fmt(value: Any) -> str:
    value_f = safe_float(value)
    return "" if value_f is None else f"{value_f:.6f}"


def load_target_rows() -> list[dict[str, Any]]:
    rows = diag.load_target_rows()
    return sorted(rows, key=lambda row: (str(row["fold_id"]), int(row["seed"])))


def build_candidate() -> base.Candidate:
    return diag.build_candidate()


def raw_flatten(
    pred: dict[str, Any], row: dict[str, Any], split: str, context: pd.DataFrame
) -> pd.DataFrame:
    return diag.flatten_prediction(pred, {"method": "raw", "params": {}}, row, split, context)


def collect_raw_frames(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = [
        str(row.get("checkpoint_path"))
        for row in rows
        if not (row.get("checkpoint_path") and Path(str(row["checkpoint_path"])).exists())
    ]
    if missing:
        print(
            f"[cp178 rescue] 비운영 후보 {TARGET_CANDIDATE_ID} 의 체크포인트 {len(missing)}개가 "
            "패키지에 없어 walk-forward 하단 보정 분석을 건너뜁니다. "
            "운영 1W 밴드 재현은 재현 경로 A 를 사용하세요."
        )
        raise SystemExit(0)
    candidate = build_candidate()
    context = diag.load_context()
    payloads: list[dict[str, Any]] = []
    for row in rows:
        fold = diag.fold_by_id(str(row["fold_id"]))
        bundles = base.build_fold_bundles(candidate, fold)
        checkpoint_path = row.get("checkpoint_path")
        val_pred = base.collect_predictions_for_bundle(checkpoint_path, bundles["val_raw"])
        test_pred = base.collect_predictions_for_bundle(checkpoint_path, bundles["test_raw"])
        val_flat = raw_flatten(val_pred, row, "val", context)
        test_flat = raw_flatten(test_pred, row, "test", context)
        payloads.append({"row": row, "fold": fold, "val": val_flat, "test": test_flat})
    return payloads


def apply_lower_extension(frame: pd.DataFrame, extension: np.ndarray | float) -> pd.DataFrame:
    out = frame.copy()
    out["calibrated_lower"] = out["lower"] - extension
    out["calibrated_upper"] = out["upper"]
    out["calibrated_width"] = out["calibrated_upper"] - out["calibrated_lower"]
    out["calibrated_lower_breach"] = out["actual_return"] < out["calibrated_lower"]
    out["calibrated_upper_breach"] = out["actual_return"] > out["calibrated_upper"]
    out["calibrated_covered"] = ~(out["calibrated_lower_breach"] | out["calibrated_upper_breach"])
    center = (out["calibrated_lower"] + out["calibrated_upper"]) / 2.0
    out["calibrated_lower_width"] = center - out["calibrated_lower"]
    return out


def stress_threshold_from_val(val: pd.DataFrame) -> float | None:
    if "vix_close" not in val.columns:
        return None
    values = pd.to_numeric(val["vix_close"], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    return float(values.quantile(0.75))


def stress_mask(frame: pd.DataFrame, threshold: float | None) -> pd.Series:
    if threshold is None or "vix_close" not in frame.columns:
        fallback = frame["realized_move"].quantile(0.75)
        return frame["realized_move"] >= fallback
    return pd.to_numeric(frame["vix_close"], errors="coerce").fillna(-np.inf) >= threshold


def interval_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    if total == 0:
        return {}
    lower = frame["calibrated_lower"].astype(float)
    upper = frame["calibrated_upper"].astype(float)
    actual = frame["actual_return"].astype(float)
    width = np.maximum((upper - lower).to_numpy(), 0.0)
    lower_excess = np.maximum((lower - actual).to_numpy(), 0.0)
    upper_excess = np.maximum((actual - upper).to_numpy(), 0.0)
    nominal = Q_HIGH - Q_LOW
    alpha = max(1.0 - nominal, 1e-6)
    lower_penalty = 2.0 * (2.0 / alpha) * lower_excess
    upper_penalty = 1.0 * (2.0 / alpha) * upper_excess
    lower_breach = frame["calibrated_lower_breach"].to_numpy(dtype=bool)
    upper_breach = frame["calibrated_upper_breach"].to_numpy(dtype=bool)
    covered = frame["calibrated_covered"].to_numpy(dtype=bool)
    return {
        "row_count": total,
        "lower_breach_rate": float(lower_breach.mean()),
        "upper_breach_rate": float(upper_breach.mean()),
        "empirical_coverage": float(covered.mean()),
        "coverage_abs_error": abs(float(covered.mean()) - nominal),
        "lower_breach_abs_error": abs(float(lower_breach.mean()) - Q_LOW),
        "upper_breach_abs_error": abs(float(upper_breach.mean()) - (1.0 - Q_HIGH)),
        "asymmetric_interval_score": float((width + lower_penalty + upper_penalty).mean()),
        "interval_lower_penalty": float(lower_penalty.mean()),
        "interval_upper_penalty": float(upper_penalty.mean()),
        "median_band_width": diag.quantile(frame["calibrated_width"], 0.5),
        "p90_band_width": diag.quantile(frame["calibrated_width"], 0.9),
        "band_width_ic": diag.spearman(frame["calibrated_width"], frame["realized_move"]),
        "downside_width_ic": diag.spearman(
            frame["calibrated_lower_width"], frame["downside_magnitude"]
        ),
    }


def objective(metrics: dict[str, Any]) -> float:
    return (
        4.0 * float(metrics.get("coverage_abs_error") or 1.0)
        + 3.0 * float(metrics.get("lower_breach_abs_error") or 1.0)
        + 1.0 * float(metrics.get("upper_breach_abs_error") or 1.0)
        + 0.25 * float(metrics.get("asymmetric_interval_score") or 1.0)
        + 0.3 * max(0.0, float(metrics.get("p90_band_width") or 0.0) - 0.25)
    )


def fit_shift_grid(val: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    best_shift = 0.0
    best_metrics: dict[str, Any] = {}
    best_score = float("inf")
    for shift in np.linspace(0.0, 0.05, 51):
        calibrated = apply_lower_extension(val, float(shift))
        metrics = interval_metrics(calibrated)
        score = objective(metrics)
        if score < best_score:
            best_score = score
            best_shift = float(shift)
            best_metrics = metrics
    return best_shift, {**best_metrics, "fit_score": best_score}


def conformal_shift(val: pd.DataFrame, alpha: float = Q_LOW) -> float:
    residual = (val["lower"] - val["actual_return"]).astype(float).to_numpy()
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return 0.0
    rank = int(np.ceil((residual.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), residual.size - 1)
    return float(max(0.0, np.partition(residual, rank)[rank]))


def fit_method(method: str, val: pd.DataFrame) -> dict[str, Any]:
    threshold = stress_threshold_from_val(val)
    if method == "raw":
        return {
            "method": method,
            "params": {"global_shift": 0.0, "stress_threshold_vix": threshold},
        }
    if method == "global_lower_shift":
        shift, metrics = fit_shift_grid(val)
        return {
            "method": method,
            "params": {"global_shift": shift, "stress_threshold_vix": threshold},
            "fit_metrics": metrics,
        }
    if method == "stress_aware_lower_shift":
        mask = stress_mask(val, threshold)
        calm_shift, calm_metrics = fit_shift_grid(val.loc[~mask])
        stress_shift, stress_metrics = fit_shift_grid(val.loc[mask])
        return {
            "method": method,
            "params": {
                "calm_shift": calm_shift,
                "stress_shift": stress_shift,
                "stress_threshold_vix": threshold,
            },
            "fit_metrics": {"calm": calm_metrics, "stress": stress_metrics},
        }
    if method == "global_lower_conformal":
        shift = conformal_shift(val)
        return {
            "method": method,
            "params": {"global_shift": shift, "stress_threshold_vix": threshold},
        }
    if method == "stress_aware_adaptive_conformal":
        mask = stress_mask(val, threshold)
        calm_shift = conformal_shift(val.loc[~mask])
        stress_shift = conformal_shift(val.loc[mask])
        return {
            "method": method,
            "params": {
                "calm_shift": calm_shift,
                "stress_shift": stress_shift,
                "stress_threshold_vix": threshold,
            },
        }
    raise ValueError(f"지원하지 않는 method: {method}")


def apply_method(frame: pd.DataFrame, fitted: dict[str, Any]) -> pd.DataFrame:
    method = fitted["method"]
    params = fitted.get("params") or {}
    if method in {"raw", "global_lower_shift", "global_lower_conformal"}:
        return apply_lower_extension(frame, float(params.get("global_shift", 0.0)))
    if method in {"stress_aware_lower_shift", "stress_aware_adaptive_conformal"}:
        threshold = safe_float(params.get("stress_threshold_vix"))
        mask = stress_mask(frame, threshold).to_numpy(dtype=bool)
        extension = np.where(
            mask, float(params.get("stress_shift", 0.0)), float(params.get("calm_shift", 0.0))
        )
        out = apply_lower_extension(frame, extension)
        out["stress_bucket"] = np.where(mask, "stress", "calm")
        return out
    raise ValueError(f"지원하지 않는 method: {method}")


def row_metrics(
    payload: dict[str, Any], method: str, fitted: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    val_cal = apply_method(payload["val"], fitted)
    test_cal = apply_method(payload["test"], fitted)
    val_metrics = interval_metrics(val_cal)
    test_metrics = interval_metrics(test_cal)
    row = payload["row"]
    result = {
        "method": method,
        "fold_id": row.get("fold_id"),
        "seed": row.get("seed"),
        "calibration_params": fitted.get("params") or {},
        "test_used_for_fit": False,
    }
    for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            result[f"{prefix}_{key}"] = value
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
        result[f"{key}_transfer_gap"] = test - val if val is not None and test is not None else None
    return result, val_cal, test_cal


def stress_calm_rows(
    method: str, fold_id: str, seed: int, test_cal: pd.DataFrame
) -> list[dict[str, Any]]:
    if "stress_bucket" not in test_cal.columns:
        threshold = stress_threshold_from_val(test_cal)
        mask = stress_mask(test_cal, threshold)
        test_cal = test_cal.copy()
        test_cal["stress_bucket"] = np.where(mask, "stress", "calm")
    rows: list[dict[str, Any]] = []
    for bucket, group in test_cal.groupby("stress_bucket", sort=True):
        metrics = interval_metrics(group)
        rows.append(
            {"method": method, "fold_id": fold_id, "seed": seed, "bucket": bucket, **metrics}
        )
    return rows


def bootstrap_ci(method: str, fold_id: str, seed: Any, frame: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(178)
    lower = frame["calibrated_lower_breach"].to_numpy(dtype=float)
    covered = frame["calibrated_covered"].to_numpy(dtype=float)
    n = len(frame)
    lower_boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
    coverage_boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
    coverage_error_boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for idx in range(N_BOOTSTRAP):
        sample_idx = rng.integers(0, n, size=n)
        lower_rate = lower[sample_idx].mean()
        coverage = covered[sample_idx].mean()
        lower_boot[idx] = lower_rate
        coverage_boot[idx] = coverage
        coverage_error_boot[idx] = abs(coverage - (Q_HIGH - Q_LOW))
    return {
        "method": method,
        "fold_id": fold_id,
        "seed": seed,
        "row_count": n,
        "bootstrap_n": N_BOOTSTRAP,
        "lower_breach_mean": float(lower.mean()),
        "lower_breach_ci95_low": float(np.quantile(lower_boot, 0.025)),
        "lower_breach_ci95_high": float(np.quantile(lower_boot, 0.975)),
        "coverage_mean": float(covered.mean()),
        "coverage_ci95_low": float(np.quantile(coverage_boot, 0.025)),
        "coverage_ci95_high": float(np.quantile(coverage_boot, 0.975)),
        "coverage_abs_error_mean": abs(float(covered.mean()) - (Q_HIGH - Q_LOW)),
        "coverage_abs_error_ci95_low": float(np.quantile(coverage_error_boot, 0.025)),
        "coverage_abs_error_ci95_high": float(np.quantile(coverage_error_boot, 0.975)),
    }


def aggregate_summary(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_by_key = {(row["fold_id"], row["seed"]): row for row in fold_rows if row["method"] == "raw"}
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
        row["decision_hint"] = decision_hint(row)
        rows.append(row)
    return sorted(
        rows, key=lambda row: safe_float(row.get("test_coverage_abs_error_mean")) or 999.0
    )


def decision_hint(row: dict[str, Any]) -> str:
    lower_mean = safe_float(row.get("test_lower_breach_rate_mean"))
    lower_worst = safe_float(row.get("test_lower_breach_rate_worst"))
    cov_mean = safe_float(row.get("test_coverage_abs_error_mean"))
    cov_worst = safe_float(row.get("test_coverage_abs_error_worst"))
    p90_worst = safe_float(row.get("test_p90_band_width_worst"))
    interval = safe_float(row.get("test_asymmetric_interval_score_mean"))
    if (
        lower_mean is not None
        and lower_worst is not None
        and cov_mean is not None
        and cov_worst is not None
    ):
        if (
            lower_mean <= 0.11
            and lower_worst <= 0.13
            and cov_mean <= 0.03
            and cov_worst <= 0.055
            and (p90_worst or 9) <= 0.26
            and (interval or 9) <= 0.34
        ):
            return "PASS"
        if lower_mean <= 0.12 and cov_mean <= 0.05 and (p90_worst or 9) <= 0.28:
            return "WARN"
    return "FAIL"


def final_decision(summary_rows: list[dict[str, Any]]) -> str:
    hints = {row.get("decision_hint") for row in summary_rows}
    if "PASS" in hints:
        return "PASS"
    if "WARN" in hints:
        return "WARN"
    return "FAIL"


def write_report(metrics: dict[str, Any]) -> None:
    summary = metrics["summary_rows"]
    best = summary[0] if summary else {}
    lines = [
        "# CP178-CAL 1W Band Lower-Only Calibration Report",
        "",
        "- 대상: `tide_s60_q10_q90_param`",
        "- 새 학습 없음, save-run 없음, DB write 없음, inference 저장 없음",
        "- upper band는 고정했고 lower band만 validation에서 fit한 값으로 보정했다.",
        "- test 결과로 보정값을 변경하지 않았다.",
        "",
        f"- 최종 판정: `{metrics.get('final_decision')}`",
        f"- 최상위 후보: `{best.get('method')}`",
        "",
        "| method | hint | lower_mean/worst | upper_mean | cov_mean/worst | interval | p90_worst | width_ic | downside_ic | raw lower 개선율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row.get('method')} | {row.get('decision_hint')} | {fmt(row.get('test_lower_breach_rate_mean'))}/{fmt(row.get('test_lower_breach_rate_worst'))} | {fmt(row.get('test_upper_breach_rate_mean'))} | {fmt(row.get('test_coverage_abs_error_mean'))}/{fmt(row.get('test_coverage_abs_error_worst'))} | {fmt(row.get('test_asymmetric_interval_score_mean'))} | {fmt(row.get('test_p90_band_width_worst'))} | {fmt(row.get('test_band_width_ic_mean'))} | {fmt(row.get('test_downside_width_ic_mean'))} | {fmt(row.get('raw_lower_breach_improvement_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## 결론",
            "",
        ]
    )
    decision = metrics.get("final_decision")
    if decision == "PASS":
        lines.append(
            "- lower-only stress/conformal calibration이 성공 기준을 통과했다. 1W band 제품 후보 직전 단계로 이동 가능하다."
        )
    elif decision == "WARN":
        lines.append(
            "- lower 보정은 효과가 있으나 성공 기준 전체를 만족하지 못했다. 추가 feature 또는 loss가 필요하다."
        )
    else:
        lines.append(
            "- lower-only calibration만으로는 성공 기준을 만족하지 못했다. 1W band는 research reserve를 유지한다."
        )
    lines.append(
        "- 이 CP는 새 후보 학습이 아니라 기존 1W band rescue 후보의 하방 보정 가능성 검증이다."
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    payloads = collect_raw_frames(load_target_rows())
    methods = [
        "raw",
        "global_lower_shift",
        "stress_aware_lower_shift",
        "global_lower_conformal",
        "stress_aware_adaptive_conformal",
    ]
    fold_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    test_frames_by_method: dict[str, list[pd.DataFrame]] = {method: [] for method in methods}
    for payload in payloads:
        for method in methods:
            fitted = fit_method(method, payload["val"])
            result, val_cal, test_cal = row_metrics(payload, method, fitted)
            fold_rows.append(result)
            params_rows.append(
                {
                    "method": method,
                    "fold_id": result["fold_id"],
                    "seed": result["seed"],
                    **(fitted.get("params") or {}),
                }
            )
            transfer_rows.append(
                {
                    key: value
                    for key, value in result.items()
                    if key in {"method", "fold_id", "seed"} or key.endswith("_transfer_gap")
                }
            )
            regime_rows.extend(
                stress_calm_rows(method, str(result["fold_id"]), int(result["seed"]), test_cal)
            )
            bootstrap_rows.append(
                bootstrap_ci(method, str(result["fold_id"]), int(result["seed"]), test_cal)
            )
            test_frames_by_method[method].append(test_cal)
    for method, frames in test_frames_by_method.items():
        combined = pd.concat(frames, ignore_index=True)
        bootstrap_rows.append(bootstrap_ci(method, "ALL", "ALL", combined))
    summary_rows = aggregate_summary(fold_rows)
    metrics = {
        "cp": "CP178-CAL",
        "created_at_utc": now_utc(),
        "target_candidate_id": TARGET_CANDIDATE_ID,
        "policy": {
            "new_training": False,
            "save_run": False,
            "db_write": False,
            "inference_saved": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "composite": False,
            "upper_band_modified": False,
            "fit_split": "validation_only",
            "test_used_for_calibration_reselection": False,
        },
        "summary_rows": summary_rows,
        "fold_rows": fold_rows,
        "params_rows": params_rows,
        "regime_rows": regime_rows,
        "transfer_rows": transfer_rows,
        "bootstrap_rows": bootstrap_rows,
        "final_decision": final_decision(summary_rows),
    }
    write_rows(SUMMARY_CSV, summary_rows)
    write_rows(FOLD_CSV, fold_rows)
    write_rows(PARAMS_CSV, params_rows)
    write_rows(BOOTSTRAP_CSV, bootstrap_rows)
    write_rows(REGIME_CSV, regime_rows)
    write_rows(TRANSFER_CSV, transfer_rows)
    write_json(METRICS_PATH, metrics)
    write_report(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CP178-CAL lower-only stress/conformal calibration"
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    metrics = run()
    summary = {
        "cp": metrics["cp"],
        "final_decision": metrics["final_decision"],
        "best_method": metrics["summary_rows"][0]["method"] if metrics["summary_rows"] else None,
        "outputs": {
            "report": str(REPORT_PATH),
            "metrics": str(METRICS_PATH),
            "summary_csv": str(SUMMARY_CSV),
            "fold_csv": str(FOLD_CSV),
            "params_csv": str(PARAMS_CSV),
            "bootstrap_csv": str(BOOTSTRAP_CSV),
            "regime_csv": str(REGIME_CSV),
            "transfer_csv": str(TRANSFER_CSV),
        },
    }
    print(json.dumps(base.clean_json(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
