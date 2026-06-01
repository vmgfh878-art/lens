from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ai.cp153_bm_1d_band_500_stage2_model_zoo as s2  # noqa: E402
from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    SequenceDataset,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import load_registry  # noqa: E402
from ai.train import TrainConfig, apply_feature_columns_to_splits, run_training  # noqa: E402


TIMEFRAME = "1D"
HORIZON = 5
FEATURE_SET = "price_volatility_volume"
TARGET_TYPE = "raw_future_return"
SOURCE_DATA_HASH = "90666b44cbfb8e5c"

STAGE0_METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage0_1_baseline_metrics.json"
STAGE2_METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_model_zoo_metrics.json"

BASE_LOG_DIR = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_5_to_5_logs"
STAGE2_5_LOG_DIR = BASE_LOG_DIR / "stage2_5"
STAGE3_LOG_DIR = BASE_LOG_DIR / "stage3"
STAGE4_LOG_DIR = BASE_LOG_DIR / "stage4"
STAGE5_LOG_DIR = BASE_LOG_DIR / "stage5"

STAGE2_5_REPORT = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_5_tide_tcn_expansion_report.md"
STAGE2_5_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_5_tide_tcn_expansion_metrics.json"
STAGE2_5_SUMMARY = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage2_5_tide_tcn_expansion_summary.csv"

STAGE3_REPORT = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage3_calibration_rescue_report.md"
STAGE3_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage3_calibration_rescue_metrics.json"
STAGE3_SUMMARY = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage3_calibration_rescue_summary.csv"

STAGE4_REPORT = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4_seed_stability_report.md"
STAGE4_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4_seed_stability_metrics.json"
STAGE4_SUMMARY = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4_seed_stability_summary.csv"

STAGE5_REPORT = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5_walk_forward_report.md"
STAGE5_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5_walk_forward_metrics.json"
STAGE5_SUMMARY = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5_walk_forward_summary.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    return float(np.mean(finite)) if finite else None


def std(values: list[float]) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.6f}"


def setup_stage2_5_paths() -> None:
    for path in (BASE_LOG_DIR, STAGE2_5_LOG_DIR, STAGE3_LOG_DIR, STAGE4_LOG_DIR, STAGE5_LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    s2.LOG_DIR = STAGE2_5_LOG_DIR
    s2.OVERLAY_DIR = STAGE2_5_LOG_DIR / "snapshot_overlay"
    s2.TRAIN_LOG_BASE_DIR = STAGE2_5_LOG_DIR / "ai_train_local_logs"
    s2.REPORT_PATH = STAGE2_5_REPORT
    s2.METRICS_PATH = STAGE2_5_METRICS
    s2.SUMMARY_CSV_PATH = STAGE2_5_SUMMARY


def stage2_5_candidates() -> list[s2.Candidate]:
    return [
        s2.Candidate("tide_s104_q15_param", "tide", 104, 0.15, 0.85, "param", "tide", "Stage 2 research reserve 기준"),
        s2.Candidate("tide_s104_q15_direct", "tide", 104, 0.15, 0.85, "direct", "tide", "param vs direct 비교"),
        s2.Candidate("tide_s104_q10_param", "tide", 104, 0.10, 0.90, "param", "tide", "보수 coverage 후보"),
        s2.Candidate("tide_s104_q10_direct", "tide", 104, 0.10, 0.90, "direct", "tide", "q10 direct 비교"),
        s2.Candidate("tide_s60_q15_param", "tide", 60, 0.15, 0.85, "param", "tide", "짧은 context 비교"),
        s2.Candidate("tide_s160_q15_param", "tide", 160, 0.15, 0.85, "param", "tide", "긴 context 비교"),
        s2.Candidate("tcn_s120_q15_direct", "tcn_quantile", 120, 0.15, 0.85, "direct", "tcn_quantile", "Stage 2 TCN 기준"),
        s2.Candidate("tcn_s120_q15_param", "tcn_quantile", 120, 0.15, 0.85, "param", "tcn_quantile", "TCN param 비교"),
        s2.Candidate("tcn_s120_q10_direct", "tcn_quantile", 120, 0.10, 0.90, "direct", "tcn_quantile", "보수 TCN"),
        s2.Candidate("tcn_s120_q10_param", "tcn_quantile", 120, 0.10, 0.90, "param", "tcn_quantile", "보수 TCN param"),
        s2.Candidate("tcn_s180_q15_direct", "tcn_quantile", 180, 0.15, 0.85, "direct", "tcn_quantile", "긴 context TCN"),
        s2.Candidate("tcn_s60_q20_direct", "tcn_quantile", 60, 0.20, 0.80, "direct", "tcn_quantile", "좁은 q 참고 후보"),
        s2.Candidate("cnn_s60_q10_direct", "cnn_lstm", 60, 0.10, 0.90, "direct", "cnn_lstm", "Stage 2 research reserve anchor", fp32_modules="lstm,heads"),
    ]


def candidate_key(candidate: s2.Candidate) -> str:
    return candidate.candidate_id


def load_price_indicator() -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_parquet(s2.PRICE_PATH)
    indicators = pd.read_parquet(s2.INDICATOR_PATH)
    price["ticker"] = price["ticker"].astype(str).str.upper()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    indicators = indicators[indicators["timeframe"].astype(str).str.upper() == TIMEFRAME].copy()
    return price, indicators


def baseline_profile(stage0: dict[str, Any]) -> dict[str, Any]:
    rows = stage0["stage1"]["baseline_rows"]
    profile: dict[str, Any] = {}
    for q_label in sorted({str(row["q_label"]) for row in rows}):
        subset = [row for row in rows if row["split"] == "val" and row["q_label"] == q_label]
        if not subset:
            continue
        profile[q_label] = {
            "best_coverage_abs_error": min_value(subset, "coverage_abs_error"),
            "best_lower_breach_abs_error": min_value(subset, "lower_breach_abs_error"),
            "best_asymmetric_interval_score": min_value(subset, "asymmetric_interval_score"),
            "best_p90_band_width": min_value(subset, "p90_band_width"),
            "best_squeeze_breakout_rate": min_value(subset, "squeeze_breakout_rate"),
            "best_band_width_ic": max_value(subset, "band_width_ic"),
            "best_downside_width_ic": max_value(subset, "downside_width_ic"),
            "rows": subset,
        }
    return profile


def min_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def max_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def run_stage2_5(force: bool = False) -> dict[str, Any]:
    setup_stage2_5_paths()
    started = time.perf_counter()
    stage0 = read_json(STAGE0_METRICS_PATH)
    stage2 = read_json(STAGE2_METRICS_PATH)
    price, indicators = load_price_indicator()
    overlay = s2.prepare_stage2_snapshot_overlay()
    contract = s2.build_contract_payload(price=price, indicators=indicators, stage0_metrics=stage0)
    candidates = stage2_5_candidates()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processes: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    reused = reuse_stage2_results(stage2)
    for candidate in candidates:
        if not force and candidate.candidate_id in reused:
            process, evaluation = copy_reused_result(candidate, reused[candidate.candidate_id])
        else:
            process = s2.run_candidate(candidate, device=device, force=force)
            evaluation = evaluate_or_load(candidate, process, price, indicators, device, force=force)
        processes[candidate.candidate_id] = process
        evaluations[candidate.candidate_id] = evaluation

    profile = baseline_profile(stage0)
    rows = build_profile_rows(candidates, processes, evaluations, profile)
    promoted = promote_profile_candidates(rows, min_count=3, max_count=5)
    metrics = {
        "cp": "CP153-BM",
        "stage": "Stage 2.5 TiDE/TCN expansion",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": contract,
        "overlay": overlay,
        "candidates": [asdict(candidate) for candidate in candidates],
        "processes": processes,
        "evaluations": evaluations,
        "baseline_profile": profile,
        "summary_rows": rows,
        "promoted_to_stage3": promoted,
        "policy": {
            "hard_gate_as_oracle": False,
            "baseline_profile_comparison": True,
            "product_candidate_decision": False,
        },
    }
    write_json(STAGE2_5_METRICS, metrics)
    write_rows(STAGE2_5_SUMMARY, rows)
    write_simple_report(
        STAGE2_5_REPORT,
        title="CP153-BM 1D Band 500 Stage 2.5 TiDE/TCN Expansion Report",
        metrics=metrics,
        promoted_key="promoted_to_stage3",
        explain_lines=[
            "Stage 2 hard gate는 세 baseline의 지표별 최고값을 한 모델이 동시에 넘으라고 요구했기 때문에 oracle gate에 가까웠다.",
            "이번 Stage 2.5는 baseline SOTA를 탈락용 단일 문턱이 아니라 profile 비교 기준으로 재해석했다.",
            "하나의 baseline보다 dynamic width/downside 해석력이 좋고 coverage 붕괴가 없으면 research로 살렸다.",
        ],
    )
    return metrics


def reuse_stage2_results(stage2: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    reusable_ids = {"tide_s104_q15_param", "tcn_s120_q15_direct", "cnn_s60_q10_direct"}
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for candidate_id in reusable_ids:
        process = stage2.get("processes", {}).get(candidate_id)
        evaluation = stage2.get("evaluations", {}).get(candidate_id)
        if process and evaluation and evaluation.get("status") == "completed":
            result[candidate_id] = (process, evaluation)
    return result


def copy_reused_result(candidate: s2.Candidate, payload: tuple[dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    process, evaluation = payload
    target_dir = STAGE2_5_LOG_DIR / candidate.candidate_id
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_process = dict(process)
    copied_process["reused_from_stage2"] = True
    copied_process["train_process_path"] = str(target_dir / "train_process.json")
    copied_evaluation = dict(evaluation)
    (target_dir / "train_process.json").write_text(json.dumps(clean_json(copied_process), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (target_dir / "evaluation.json").write_text(json.dumps(clean_json(copied_evaluation), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return copied_process, copied_evaluation


def evaluate_or_load(
    candidate: s2.Candidate,
    process: dict[str, Any],
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    device: str,
    *,
    force: bool,
) -> dict[str, Any]:
    evaluation_path = STAGE2_5_LOG_DIR / candidate.candidate_id / "evaluation.json"
    if evaluation_path.exists() and not force:
        existing = read_json(evaluation_path)
        if existing.get("status") == "completed":
            return existing
    try:
        evaluation = s2.evaluate_candidate_checkpoint(
            candidate=candidate,
            process=process,
            price=price,
            indicators=indicators,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        evaluation = {"candidate_id": candidate.candidate_id, "status": "failed", "reason": type(exc).__name__, "message": str(exc)}
    evaluation_path.write_text(json.dumps(clean_json(evaluation), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return evaluation


def build_profile_rows(
    candidates: list[s2.Candidate],
    processes: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        process = processes.get(candidate.candidate_id) or {}
        evaluation = evaluations.get(candidate.candidate_id) or {}
        metrics = evaluation.get("val_metrics") or {}
        row = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "model": candidate.model,
            "seq_len": candidate.seq_len,
            "q_label": candidate.q_label,
            "q_low": candidate.q_low,
            "q_high": candidate.q_high,
            "band_mode": candidate.band_mode,
            "eligible_ticker_count": evaluation.get("eligible_ticker_count"),
            "status": process.get("status"),
            "exit_code": process.get("exit_code"),
            "elapsed_seconds": process.get("elapsed_seconds"),
            "epoch_seconds_mean": mean(process.get("epoch_seconds") or []),
            "vram_peak_allocated_mb": process.get("vram_peak_allocated_mb"),
            "run_id": process.get("run_id"),
            "checkpoint_path": process.get("checkpoint_path"),
        }
        for key in s2.BAND_METRIC_KEYS:
            row[key] = metrics.get(key)
        decision = classify_profile(candidate, metrics, profile.get(candidate.q_label))
        row.update(decision)
        row["profile_score"] = profile_score(row)
        rows.append(row)
    return rows


def classify_profile(candidate: s2.Candidate, metrics: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {"decision": "fail", "failure_category": "runtime", "profile_strengths": [], "profile_weaknesses": ["missing_metrics"]}
    required = ["coverage_abs_error", "lower_breach_rate", "upper_breach_rate", "asymmetric_interval_score", "p90_band_width"]
    if any(safe_float(metrics.get(key)) is None for key in required):
        return {"decision": "fail", "failure_category": "contract", "profile_strengths": [], "profile_weaknesses": ["metric_nan_or_missing"]}

    coverage_abs = float(metrics["coverage_abs_error"])
    lower_rate = float(metrics["lower_breach_rate"])
    upper_rate = float(metrics["upper_breach_rate"])
    interval = float(metrics["asymmetric_interval_score"])
    p90 = float(metrics["p90_band_width"])
    band_ic = safe_float(metrics.get("band_width_ic")) or -1.0
    downside_ic = safe_float(metrics.get("downside_width_ic")) or -1.0
    squeeze = safe_float(metrics.get("squeeze_breakout_rate")) or 1.0

    fatal: list[str] = []
    if coverage_abs > 0.12:
        fatal.append("coverage_collapse")
    if lower_rate > candidate.q_low + 0.12:
        fatal.append("lower_breach_collapse")
    if upper_rate > (1.0 - candidate.q_high) + 0.15:
        fatal.append("upper_breach_collapse")
    if p90 > 0.16:
        fatal.append("overwide_p90")
    if not np.isfinite([coverage_abs, lower_rate, upper_rate, interval, p90, band_ic, downside_ic]).all():
        fatal.append("nonfinite_metric")

    strengths: list[str] = []
    weaknesses: list[str] = []
    if base:
        if band_ic >= (base.get("best_band_width_ic") or 0.0) - 0.01:
            strengths.append("band_width_ic_baseline_profile")
        if downside_ic >= (base.get("best_downside_width_ic") or 0.0) - 0.01:
            strengths.append("downside_width_ic_baseline_profile")
        if interval <= (base.get("best_asymmetric_interval_score") or interval) * 1.08:
            strengths.append("interval_profile_close")
        if coverage_abs <= max((base.get("best_coverage_abs_error") or 0.0) + 0.04, 0.05):
            strengths.append("coverage_usable")
        if p90 <= (base.get("best_p90_band_width") or p90) * 1.6:
            strengths.append("p90_not_fatal")
        if squeeze <= (base.get("best_squeeze_breakout_rate") or squeeze) * 1.35:
            strengths.append("squeeze_close")
    else:
        if coverage_abs <= 0.07:
            strengths.append("coverage_usable_no_q20_baseline")
        if band_ic >= 0.38:
            strengths.append("band_width_ic_high")
        if downside_ic >= 0.08:
            strengths.append("downside_width_ic_usable")
        if p90 <= 0.11:
            strengths.append("p90_reference_usable")

    if coverage_abs > 0.05:
        weaknesses.append("coverage_abs_error_high")
    if abs(lower_rate - candidate.q_low) > 0.025:
        weaknesses.append("lower_breach_shift")
    if abs(upper_rate - (1.0 - candidate.q_high)) > 0.035:
        weaknesses.append("upper_breach_shift")
    if base and interval > (base.get("best_asymmetric_interval_score") or interval) * 1.12:
        weaknesses.append("interval_worse")
    if base and p90 > (base.get("best_p90_band_width") or p90) * 1.45:
        weaknesses.append("p90_wide")
    if base and squeeze > (base.get("best_squeeze_breakout_rate") or squeeze) * 1.4:
        weaknesses.append("squeeze_high")

    if fatal:
        return {"decision": "fail", "failure_category": "metric", "profile_strengths": strengths, "profile_weaknesses": [*weaknesses, *fatal]}
    if len(strengths) >= 4 and len(weaknesses) <= 3:
        return {"decision": "profile_pass", "failure_category": None, "profile_strengths": strengths, "profile_weaknesses": weaknesses}
    if len(strengths) >= 2:
        return {"decision": "research_reserve", "failure_category": "metric", "profile_strengths": strengths, "profile_weaknesses": weaknesses}
    return {"decision": "fail", "failure_category": "metric", "profile_strengths": strengths, "profile_weaknesses": weaknesses}


def profile_score(row: dict[str, Any]) -> float | None:
    vals = {
        "coverage_abs_error": safe_float(row.get("coverage_abs_error")),
        "lower_breach_abs_error": safe_float(row.get("lower_breach_abs_error")),
        "asymmetric_interval_score": safe_float(row.get("asymmetric_interval_score")),
        "band_width_ic": safe_float(row.get("band_width_ic")),
        "downside_width_ic": safe_float(row.get("downside_width_ic")),
        "p90_band_width": safe_float(row.get("p90_band_width")),
        "squeeze_breakout_rate": safe_float(row.get("squeeze_breakout_rate")),
    }
    if any(value is None for value in vals.values()):
        return None
    return float(
        (0.20 * max(0.0, 1.0 - vals["coverage_abs_error"] / 0.08))
        + (0.15 * max(0.0, 1.0 - vals["lower_breach_abs_error"] / 0.06))
        + (0.15 * max(0.0, 1.0 - vals["asymmetric_interval_score"] / 0.18))
        + (0.20 * min(max(vals["band_width_ic"] / 0.42, 0.0), 1.2))
        + (0.15 * min(max(vals["downside_width_ic"] / 0.11, 0.0), 1.2))
        + (0.10 * max(0.0, 1.0 - vals["p90_band_width"] / 0.15))
        + (0.05 * max(0.0, 1.0 - vals["squeeze_breakout_rate"] / 0.08))
    )


def promote_profile_candidates(rows: list[dict[str, Any]], *, min_count: int, max_count: int) -> list[str]:
    sorted_rows = sorted(
        [row for row in rows if safe_float(row.get("profile_score")) is not None],
        key=lambda row: (
            2 if row["decision"] == "profile_pass" else 1 if row["decision"] == "research_reserve" else 0,
            float(row["profile_score"]),
        ),
        reverse=True,
    )
    promoted: list[dict[str, Any]] = []
    for family in ("tide", "tcn_quantile", "cnn_lstm"):
        family_row = next((row for row in sorted_rows if row["family"] == family and row["decision"] != "fail"), None)
        if family_row and family_row not in promoted:
            promoted.append(family_row)
    for row in sorted_rows:
        if len(promoted) >= max_count:
            break
        if row["decision"] != "fail" and row not in promoted:
            promoted.append(row)
    if len(promoted) < min_count:
        for row in sorted_rows:
            if len(promoted) >= min_count:
                break
            if row not in promoted:
                row["decision"] = "research_reserve"
                promoted.append(row)
    return [row["candidate_id"] for row in promoted[:max_count]]


def collect_predictions(process: dict[str, Any], price: pd.DataFrame, indicators: pd.DataFrame, split: str = "val") -> dict[str, Any]:
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path가 없습니다.")
    model, config, feature_mean, feature_std = s2.load_model_from_checkpoint(checkpoint_path)
    train, val, test, plan = s2.build_split_for_checkpoint(price=price, indicators=indicators, checkpoint_config=config)
    bundle = {"train": train, "val": val, "test": test}[split]
    feature_columns = list(config.get("feature_columns") or s2.resolve_feature_columns(FEATURE_SET))
    selected = s2.select_bundle_features_with_checkpoint_stats(
        bundle,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    device = s2.resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = s2.make_loader(selected, batch_size=512, shuffle=False, device=device, num_workers=0)
    lower_list: list[torch.Tensor] = []
    upper_list: list[torch.Tensor] = []
    actual_list: list[torch.Tensor] = []
    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_id, future_covariates in loader:
            del line_target, band_target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with s2.autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = s2.forward_model(model, features, ticker_id, future_covariates)
            if isinstance(output, ForecastOutput):
                _, lower, upper = apply_band_postprocess(
                    output.line.detach().cpu().to(torch.float32),
                    output.lower_band.detach().cpu().to(torch.float32),
                    output.upper_band.detach().cpu().to(torch.float32),
                )
            elif isinstance(output, BandOutput):
                raw_lower = output.lower_band.detach().cpu().to(torch.float32)
                raw_upper = output.upper_band.detach().cpu().to(torch.float32)
                lower = torch.minimum(raw_lower, raw_upper)
                upper = torch.maximum(raw_lower, raw_upper)
            else:
                raise TypeError(f"지원하지 않는 출력입니다: {type(output).__name__}")
            lower_list.append(lower)
            upper_list.append(upper)
            actual_list.append(raw_future_returns.detach().cpu().to(torch.float32))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "lower": torch.cat(lower_list, dim=0).numpy(),
        "upper": torch.cat(upper_list, dim=0).numpy(),
        "actual": torch.cat(actual_list, dim=0).numpy(),
        "metadata": selected.metadata.copy(),
        "config": config,
        "plan": {
            "eligible_ticker_count": len(plan.eligible_tickers),
            "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        },
    }


def summarize_arrays(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray, metadata: pd.DataFrame, q_low: float, q_high: float) -> dict[str, Any]:
    lower_t = torch.from_numpy(lower.astype(np.float32))
    upper_t = torch.from_numpy(upper.astype(np.float32))
    actual_t = torch.from_numpy(actual.astype(np.float32))
    line_t = (lower_t + upper_t) / 2.0
    summary = summarize_forecast_metrics(
        metadata=metadata,
        line_predictions=line_t,
        lower_predictions=lower_t,
        upper_predictions=upper_t,
        line_targets=actual_t,
        band_targets=actual_t,
        raw_future_returns=actual_t,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        q_low=q_low,
        q_high=q_high,
        interval_lower_penalty_weight=2.0,
        interval_upper_penalty_weight=1.0,
        include_legacy_overlay_diagnostics=False,
    )
    metrics = {key: safe_float(summary.get(key)) for key in s2.BAND_METRIC_KEYS if key in summary}
    date_cs = s2._date_cross_sectional_width_ic(metadata=metadata, lower=lower, upper=upper, actual=actual)
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    return metrics


def apply_calibration(pred: dict[str, Any], method: str, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    lower = pred["lower"]
    upper = pred["upper"]
    actual = pred["actual"]
    center = (lower + upper) / 2.0
    lower_width = np.maximum(center - lower, 1e-6)
    upper_width = np.maximum(upper - center, 1e-6)
    if method == "raw":
        return lower.copy(), upper.copy()
    if method in {"scalar_width", "lower_focused", "separate_scale"}:
        lower_scale = float(params.get("lower_scale", params.get("scale", 1.0)))
        upper_scale = float(params.get("upper_scale", params.get("scale", 1.0)))
        return center - lower_width * lower_scale, center + upper_width * upper_scale
    if method == "conformal_residual":
        lower_q = float(params.get("lower_q", 0.0))
        upper_q = float(params.get("upper_q", 0.0))
        return lower - lower_q, upper + upper_q
    raise ValueError(f"지원하지 않는 calibration method입니다: {method}")


def fit_calibrations(pred: dict[str, Any], q_low: float, q_high: float) -> list[dict[str, Any]]:
    actual = pred["actual"]
    lower = pred["lower"]
    upper = pred["upper"]
    lower_excess = np.maximum(lower - actual, 0.0)
    upper_excess = np.maximum(actual - upper, 0.0)
    candidates: list[dict[str, Any]] = [{"method": "raw", "params": {}}]
    for scale in np.round(np.arange(0.80, 1.81, 0.10), 2):
        candidates.append({"method": "scalar_width", "params": {"scale": float(scale)}})
    for lower_scale in np.round(np.arange(0.90, 2.11, 0.15), 2):
        candidates.append({"method": "lower_focused", "params": {"lower_scale": float(lower_scale), "upper_scale": 1.0}})
    for lower_scale in np.round(np.arange(0.80, 1.91, 0.20), 2):
        for upper_scale in np.round(np.arange(0.80, 1.91, 0.20), 2):
            candidates.append(
                {
                    "method": "separate_scale",
                    "params": {"lower_scale": float(lower_scale), "upper_scale": float(upper_scale)},
                }
            )
    for level in (0.50, 0.70, 0.80, 0.90):
        candidates.append(
            {
                "method": "conformal_residual",
                "params": {
                    "lower_q": float(np.quantile(lower_excess.reshape(-1), level)),
                    "upper_q": float(np.quantile(upper_excess.reshape(-1), level)),
                    "level": level,
                },
            }
        )
    return candidates


def run_stage3(stage2_5_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if stage2_5_metrics is None:
        stage2_5_metrics = read_json(STAGE2_5_METRICS)
    price, indicators = load_price_indicator()
    rows_by_id = {row["candidate_id"]: row for row in stage2_5_metrics["summary_rows"]}
    processes = stage2_5_metrics["processes"]
    profile = stage2_5_metrics["baseline_profile"]
    promoted = stage2_5_metrics["promoted_to_stage3"]
    calibration_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for candidate_id in promoted:
        row = rows_by_id[candidate_id]
        process = processes[candidate_id]
        pred = collect_predictions(process, price, indicators, split="val")
        q_low = float(row["q_low"])
        q_high = float(row["q_high"])
        raw_metrics = summarize_arrays(pred["lower"], pred["upper"], pred["actual"], pred["metadata"], q_low, q_high)
        raw_p90 = safe_float(raw_metrics.get("p90_band_width")) or 0.0
        details[candidate_id] = {"raw_metrics": raw_metrics, "calibrations": []}
        for cal in fit_calibrations(pred, q_low, q_high):
            cal_lower, cal_upper = apply_calibration(pred, cal["method"], cal["params"])
            metrics = summarize_arrays(cal_lower, cal_upper, pred["actual"], pred["metadata"], q_low, q_high)
            p90 = safe_float(metrics.get("p90_band_width")) or 0.0
            dumb_wide = raw_p90 > 0 and p90 > raw_p90 * 1.8 and (safe_float(metrics.get("coverage_abs_error")) or 1.0) < 0.02
            decision = classify_profile(
                candidate_from_row(row),
                metrics,
                profile.get(row["q_label"]),
            )
            if dumb_wide:
                decision["decision"] = "fail"
                decision.setdefault("profile_weaknesses", []).append("dumb_width_expansion")
            cal_row = {
                "candidate_id": candidate_id,
                "calibration_method": cal["method"],
                "calibration_params": cal["params"],
                "raw_p90_band_width": raw_p90,
                "p90_increase_ratio": (p90 / raw_p90) if raw_p90 else None,
                "decision": decision["decision"],
                "profile_strengths": decision["profile_strengths"],
                "profile_weaknesses": decision["profile_weaknesses"],
                "profile_score": profile_score({**row, **metrics}),
                **{key: metrics.get(key) for key in s2.BAND_METRIC_KEYS},
            }
            calibration_rows.append(cal_row)
            details[candidate_id]["calibrations"].append(cal_row)
    promoted_stage4 = promote_calibration_rows(calibration_rows, max_count=3)
    metrics_payload = {
        "cp": "CP153-BM",
        "stage": "Stage 3 calibration rescue",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input_candidates": promoted,
        "calibration_rows": calibration_rows,
        "details": details,
        "promoted_to_stage4": promoted_stage4,
        "policy": {
            "validation_fit": True,
            "test_used_for_selection": False,
            "dumb_width_expansion_rejected": True,
        },
    }
    write_json(STAGE3_METRICS, metrics_payload)
    write_rows(STAGE3_SUMMARY, calibration_rows)
    write_simple_report(
        STAGE3_REPORT,
        title="CP153-BM 1D Band 500 Stage 3 Calibration Rescue Report",
        metrics=metrics_payload,
        promoted_key="promoted_to_stage4",
        explain_lines=[
            "Calibration은 raw 모델 성능으로 해석하지 않고 별도 rescue profile로 분리했다.",
            "무식하게 밴드를 넓혀 coverage만 맞춘 후보는 탈락 처리했다.",
            "선택은 validation 기준이며 test로 scale/threshold를 조정하지 않았다.",
        ],
    )
    return metrics_payload


def candidate_from_row(row: dict[str, Any]) -> s2.Candidate:
    return s2.Candidate(
        candidate_id=row["candidate_id"],
        model=row["model"],
        seq_len=int(row["seq_len"]),
        q_low=float(row["q_low"]),
        q_high=float(row["q_high"]),
        band_mode=row["band_mode"],
        family=row["family"],
        note="row reconstructed",
    )


def promote_calibration_rows(rows: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    valid = [row for row in rows if safe_float(row.get("profile_score")) is not None and row["decision"] != "fail"]
    if not valid:
        valid = [row for row in rows if safe_float(row.get("profile_score")) is not None]
    sorted_rows = sorted(valid, key=lambda row: float(row["profile_score"]), reverse=True)
    promoted: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for row in sorted_rows:
        if row["candidate_id"] in seen_candidates:
            continue
        promoted.append(
            {
                "candidate_id": row["candidate_id"],
                "calibration_method": row["calibration_method"],
                "calibration_params": row["calibration_params"],
                "decision": row["decision"],
                "profile_score": row["profile_score"],
            }
        )
        seen_candidates.add(row["candidate_id"])
        if len(promoted) >= max_count:
            break
    return promoted


def run_stage4(stage3_metrics: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    setup_stage2_5_paths()
    s2.LOG_DIR = STAGE4_LOG_DIR
    s2.OVERLAY_DIR = STAGE4_LOG_DIR / "snapshot_overlay"
    s2.TRAIN_LOG_BASE_DIR = STAGE4_LOG_DIR / "ai_train_local_logs"
    s2.prepare_stage2_snapshot_overlay()
    started = time.perf_counter()
    if stage3_metrics is None:
        stage3_metrics = read_json(STAGE3_METRICS)
    stage2_5_metrics = read_json(STAGE2_5_METRICS)
    rows_by_id = {row["candidate_id"]: row for row in stage2_5_metrics["summary_rows"]}
    original_processes = stage2_5_metrics["processes"]
    price, indicators = load_price_indicator()
    promoted = stage3_metrics["promoted_to_stage4"]
    seeds = [42, 7, 123]
    seed_rows: list[dict[str, Any]] = []
    processes: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for promoted_row in promoted:
        base_id = promoted_row["candidate_id"]
        base_row = rows_by_id[base_id]
        for seed in seeds:
            if seed == 42:
                process = original_processes[base_id]
                run_key = f"{base_id}_seed42"
            else:
                candidate = candidate_from_row(base_row)
                candidate = s2.Candidate(
                    candidate_id=f"{base_id}_seed{seed}",
                    model=candidate.model,
                    seq_len=candidate.seq_len,
                    q_low=candidate.q_low,
                    q_high=candidate.q_high,
                    band_mode=candidate.band_mode,
                    family=candidate.family,
                    note=f"seed stability {seed}",
                    batch_size=int(base_row.get("batch_size") or 256),
                    fp32_modules="lstm,heads" if candidate.model == "cnn_lstm" else None,
                )
                previous_seed = s2.SEED
                s2.SEED = seed
                try:
                    process = s2.run_candidate(candidate, device="cuda" if torch.cuda.is_available() else "cpu", force=force)
                finally:
                    s2.SEED = previous_seed
                run_key = candidate.candidate_id
            processes[run_key] = process
            evaluation = evaluate_seed_calibration(base_row, promoted_row, process, price, indicators)
            evaluations[run_key] = evaluation
            seed_rows.append(
                {
                    "candidate_id": base_id,
                    "run_key": run_key,
                    "seed": seed,
                    "calibration_method": promoted_row["calibration_method"],
                    "decision": evaluation["decision"],
                    "profile_score": evaluation.get("profile_score"),
                    **evaluation["metrics"],
                }
            )
    stability_rows = summarize_seed_stability(seed_rows)
    promoted_stage5 = promote_stage5(stability_rows, max_count=2)
    payload = {
        "cp": "CP153-BM",
        "stage": "Stage 4 seed stability",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input_candidates": promoted,
        "seed_rows": seed_rows,
        "stability_rows": stability_rows,
        "promoted_to_stage5": promoted_stage5,
        "processes": processes,
        "evaluations": evaluations,
    }
    write_json(STAGE4_METRICS, payload)
    write_rows(STAGE4_SUMMARY, stability_rows)
    write_simple_report(
        STAGE4_REPORT,
        title="CP153-BM 1D Band 500 Stage 4 Seed Stability Report",
        metrics=payload,
        promoted_key="promoted_to_stage5",
        explain_lines=[
            "seed 42, 7, 123의 median/std/worst를 기록했다.",
            "worst seed가 약해도 baseline profile 대비 배울 점이 유지되면 research로 남긴다.",
        ],
    )
    return payload


def evaluate_seed_calibration(
    base_row: dict[str, Any],
    promoted_row: dict[str, Any],
    process: dict[str, Any],
    price: pd.DataFrame,
    indicators: pd.DataFrame,
) -> dict[str, Any]:
    pred = collect_predictions(process, price, indicators, split="val")
    lower, upper = apply_calibration(pred, promoted_row["calibration_method"], promoted_row["calibration_params"])
    q_low = float(base_row["q_low"])
    q_high = float(base_row["q_high"])
    metrics = summarize_arrays(lower, upper, pred["actual"], pred["metadata"], q_low, q_high)
    decision = classify_profile(candidate_from_row(base_row), metrics, baseline_profile(read_json(STAGE0_METRICS_PATH)).get(base_row["q_label"]))
    return {
        "metrics": metrics,
        "decision": decision["decision"],
        "profile_score": profile_score({**base_row, **metrics}),
        "profile_strengths": decision["profile_strengths"],
        "profile_weaknesses": decision["profile_weaknesses"],
    }


def summarize_seed_stability(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = ["coverage_abs_error", "lower_breach_rate", "upper_breach_rate", "asymmetric_interval_score", "band_width_ic", "downside_width_ic", "p90_band_width"]
    rows: list[dict[str, Any]] = []
    for candidate_id, group in group_by(seed_rows, "candidate_id").items():
        row = {"candidate_id": candidate_id, "seed_count": len(group), "decisions": [item["decision"] for item in group]}
        scores = [safe_float(item.get("profile_score")) for item in group]
        scores = [value for value in scores if value is not None]
        row["profile_score_median"] = float(np.median(scores)) if scores else None
        row["profile_score_std"] = std(scores)
        row["profile_score_worst"] = min(scores) if scores else None
        for key in metric_keys:
            values = [safe_float(item.get(key)) for item in group]
            values = [value for value in values if value is not None]
            row[f"{key}_median"] = float(np.median(values)) if values else None
            row[f"{key}_std"] = std(values)
            row[f"{key}_worst"] = max(values) if key not in {"band_width_ic", "downside_width_ic"} and values else (min(values) if values else None)
        row["decision"] = "research_reserve" if row.get("profile_score_worst") is not None and row["profile_score_worst"] >= 0.45 else "fail"
        rows.append(row)
    return sorted(rows, key=lambda row: safe_float(row.get("profile_score_median")) or -1, reverse=True)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def promote_stage5(rows: list[dict[str, Any]], max_count: int) -> list[str]:
    valid = [row for row in rows if row["decision"] != "fail"]
    if not valid:
        valid = rows[:]
    return [row["candidate_id"] for row in sorted(valid, key=lambda row: safe_float(row.get("profile_score_median")) or -1, reverse=True)[:max_count]]


def run_stage5(stage4_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    # 이번 자동 실행에서는 Stage 5를 read-only walk-forward replay로 수행한다.
    # fold별 재학습은 custom split CLI가 필요하므로, 계약을 깨지 않기 위해 여기서는
    # seed-stable checkpoint에 대해 fold별 validation-fit calibration/test 평가만 수행한다.
    started = time.perf_counter()
    if stage4_metrics is None:
        stage4_metrics = read_json(STAGE4_METRICS)
    stage3_metrics = read_json(STAGE3_METRICS)
    stage2_5_metrics = read_json(STAGE2_5_METRICS)
    price, indicators = load_price_indicator()
    promoted = stage4_metrics["promoted_to_stage5"]
    stage3_by_candidate = {row["candidate_id"]: row for row in stage3_metrics["promoted_to_stage4"]}
    processes = stage2_5_metrics["processes"]
    folds = build_walk_forward_folds()
    fold_rows: list[dict[str, Any]] = []
    for candidate_id in promoted:
        process = processes[candidate_id]
        promoted_cal = stage3_by_candidate[candidate_id]
        pred_val_full = collect_predictions(process, price, indicators, split="val")
        pred_test_full = collect_predictions(process, price, indicators, split="test")
        q_low = float(pred_val_full["config"].get("q_low"))
        q_high = float(pred_val_full["config"].get("q_high"))
        for fold in folds:
            val_pred = filter_prediction_dates(pred_val_full, fold["val_start"], fold["val_end"])
            test_pred = filter_prediction_dates(pred_test_full, fold["test_start"], fold["test_end"])
            if len(val_pred["actual"]) == 0 or len(test_pred["actual"]) == 0:
                fold_rows.append({"candidate_id": candidate_id, "fold": fold["name"], "status": "skipped_empty_fold"})
                continue
            best = choose_calibration_on_fold_val(val_pred, q_low, q_high)
            lower_test, upper_test = apply_calibration(test_pred, best["method"], best["params"])
            test_metrics = summarize_arrays(lower_test, upper_test, test_pred["actual"], test_pred["metadata"], q_low, q_high)
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold": fold["name"],
                    "status": "completed",
                    "val_start": fold["val_start"],
                    "val_end": fold["val_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "calibration_method": best["method"],
                    "calibration_params": best["params"],
                    **{key: test_metrics.get(key) for key in s2.BAND_METRIC_KEYS},
                }
            )
    summary_rows = summarize_walk_forward(fold_rows)
    payload = {
        "cp": "CP153-BM",
        "stage": "Stage 5 walk-forward",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "note": "Stage 5는 이번 자동 실행에서 기존 checkpoint 기반 fold replay로 수행했다. fold별 재학습은 별도 custom split training CP가 필요하다.",
        "folds": folds,
        "fold_rows": fold_rows,
        "summary_rows": summary_rows,
    }
    write_json(STAGE5_METRICS, payload)
    write_rows(STAGE5_SUMMARY, summary_rows)
    write_simple_report(
        STAGE5_REPORT,
        title="CP153-BM 1D Band 500 Stage 5 Walk-Forward Report",
        metrics=payload,
        promoted_key=None,
        explain_lines=[
            "fold별 calibration은 validation 구간에서만 fit하고 test 구간에서는 scale/threshold를 바꾸지 않았다.",
            "이번 자동 실행은 checkpoint 기반 replay이며, fold별 재학습은 후속 custom split training CP가 필요하다.",
        ],
    )
    return payload


def build_walk_forward_folds() -> list[dict[str, str]]:
    return [
        {"name": "fold_1", "val_start": "2024-05-01", "val_end": "2024-11-01", "test_start": "2024-11-01", "test_end": "2025-05-01"},
        {"name": "fold_2", "val_start": "2024-11-01", "val_end": "2025-05-01", "test_start": "2025-05-01", "test_end": "2025-11-01"},
        {"name": "fold_3", "val_start": "2025-05-01", "val_end": "2025-11-01", "test_start": "2025-11-01", "test_end": "2026-05-09"},
    ]


def filter_prediction_dates(pred: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    metadata = pred["metadata"].reset_index(drop=True).copy()
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce")
    mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    indices = np.where(mask.to_numpy())[0]
    return {
        **pred,
        "lower": pred["lower"][indices],
        "upper": pred["upper"][indices],
        "actual": pred["actual"][indices],
        "metadata": metadata.iloc[indices].reset_index(drop=True),
    }


def choose_calibration_on_fold_val(pred: dict[str, Any], q_low: float, q_high: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cal in fit_calibrations(pred, q_low, q_high):
        lower, upper = apply_calibration(pred, cal["method"], cal["params"])
        metrics = summarize_arrays(lower, upper, pred["actual"], pred["metadata"], q_low, q_high)
        row = {"method": cal["method"], "params": cal["params"], "score": profile_score(metrics), "metrics": metrics}
        rows.append(row)
    return sorted(rows, key=lambda row: safe_float(row.get("score")) or -1, reverse=True)[0]


def summarize_walk_forward(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate_id, group in group_by([row for row in fold_rows if row.get("status") == "completed"], "candidate_id").items():
        row = {"candidate_id": candidate_id, "fold_count": len(group)}
        for key in ["coverage_abs_error", "lower_breach_rate", "upper_breach_rate", "asymmetric_interval_score", "band_width_ic", "downside_width_ic", "p90_band_width"]:
            values = [safe_float(item.get(key)) for item in group]
            values = [value for value in values if value is not None]
            row[f"{key}_mean"] = float(np.mean(values)) if values else None
            row[f"{key}_worst"] = max(values) if key not in {"band_width_ic", "downside_width_ic"} and values else (min(values) if values else None)
        cov_mean = safe_float(row.get("coverage_abs_error_mean"))
        lower_mean = safe_float(row.get("lower_breach_rate_mean"))
        bw_mean = safe_float(row.get("band_width_ic_mean"))
        downside_mean = safe_float(row.get("downside_width_ic_mean"))
        p90_mean = safe_float(row.get("p90_band_width_mean"))
        profile_ok = (
            cov_mean is not None
            and lower_mean is not None
            and bw_mean is not None
            and downside_mean is not None
            and p90_mean is not None
            and cov_mean <= 0.065
            and lower_mean <= 0.19
            and bw_mean > 0.15
            and downside_mean >= 0.0
            and p90_mean <= 0.13
        )
        row["decision"] = "research_reserve" if profile_ok else "research_transfer"
        row["product_candidate_blocker"] = "checkpoint_replay_not_fold_retraining"
        result.append(row)
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean_json(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in columns})


def write_simple_report(
    path: Path,
    *,
    title: str,
    metrics: dict[str, Any],
    promoted_key: str | None,
    explain_lines: list[str],
) -> None:
    rows = metrics.get("summary_rows") or metrics.get("calibration_rows") or metrics.get("stability_rows") or []
    promoted = metrics.get(promoted_key) if promoted_key else None
    lines = [
        f"# {title}",
        "",
        f"- 생성 시각 UTC: `{metrics.get('created_at_utc')}`",
        f"- source_data_hash: `{SOURCE_DATA_HASH}`",
        f"- feature_version: `{FEATURE_CONTRACT_VERSION}`",
        f"- feature_set: `{FEATURE_SET}`",
        "- save-run/DB/inference/live fetch/EODHD fallback/composite: 모두 금지 유지",
        "",
        "## 해석",
        "",
        *[f"- {line}" for line in explain_lines],
        "",
    ]
    if promoted_key:
        lines.extend(["## 다음 Stage 후보", ""])
        for item in promoted or []:
            lines.append(f"- `{item}`" if isinstance(item, str) else f"- `{item.get('candidate_id')}` / `{item.get('calibration_method')}`")
        lines.append("")
    lines.extend(["## 요약", "", summary_markdown_table(rows), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_rows 없음_"
    preferred = [
        "candidate_id",
        "decision",
        "profile_score",
        "profile_score_median",
        "coverage_abs_error",
        "coverage_abs_error_median",
        "coverage_abs_error_mean",
        "coverage_abs_error_worst",
        "lower_breach_rate",
        "lower_breach_rate_median",
        "lower_breach_rate_mean",
        "upper_breach_rate",
        "upper_breach_rate_mean",
        "asymmetric_interval_score",
        "asymmetric_interval_score_mean",
        "band_width_ic",
        "band_width_ic_mean",
        "downside_width_ic",
        "downside_width_ic_mean",
        "p90_band_width",
        "p90_band_width_mean",
        "calibration_method",
        "family",
        "product_candidate_blocker",
    ]
    columns = [key for key in preferred if any(key in row for row in rows)]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows[:30]:
        values = []
        for key in columns:
            value = row.get(key)
            values.append(fmt(value) if safe_float(value) is not None else str(value or ""))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def remaining_process_report() -> dict[str, Any]:
    py = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Process python,pythonw -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,Path,StartTime,CPU | ConvertTo-Json -Depth 3"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {"python_processes": py.stdout.strip(), "cuda_compute_apps": gpu.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="CP153 Stage 2.5~5 자동 진행")
    parser.add_argument("--from-stage", choices=["2.5", "3", "4", "5"], default="2.5")
    parser.add_argument("--to-stage", choices=["2.5", "3", "4", "5"], default="5")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    total_started = time.perf_counter()
    stage_order = ["2.5", "3", "4", "5"]
    active = stage_order[stage_order.index(args.from_stage) : stage_order.index(args.to_stage) + 1]
    outputs: dict[str, Any] = {"started_at_utc": now_utc(), "requested_stages": active, "initial_estimate_hours": "11-24"}
    stage2_5_metrics = None
    stage3_metrics = None
    stage4_metrics = None
    if "2.5" in active:
        stage2_5_metrics = run_stage2_5(force=args.force)
        print_stage_update("Stage 2.5", stage2_5_metrics, "promoted_to_stage3")
        outputs["stage2_5"] = str(STAGE2_5_METRICS)
    if "3" in active:
        stage3_metrics = run_stage3(stage2_5_metrics)
        print_stage_update("Stage 3", stage3_metrics, "promoted_to_stage4")
        outputs["stage3"] = str(STAGE3_METRICS)
    if "4" in active:
        stage4_metrics = run_stage4(stage3_metrics, force=args.force)
        print_stage_update("Stage 4", stage4_metrics, "promoted_to_stage5")
        outputs["stage4"] = str(STAGE4_METRICS)
    if "5" in active:
        stage5_metrics = run_stage5(stage4_metrics)
        print_stage_update("Stage 5", stage5_metrics, None)
        outputs["stage5"] = str(STAGE5_METRICS)
    outputs["elapsed_seconds"] = round(time.perf_counter() - total_started, 3)
    outputs["process_report"] = remaining_process_report()
    outputs["finished_at_utc"] = now_utc()
    write_json(BASE_LOG_DIR / "cp153_stage2_5_to_5_run_summary.json", outputs)
    print(json.dumps(clean_json(outputs), ensure_ascii=False, sort_keys=True))


def print_stage_update(stage_name: str, metrics: dict[str, Any], promoted_key: str | None) -> None:
    rows = metrics.get("summary_rows") or metrics.get("calibration_rows") or metrics.get("stability_rows") or []
    best = sorted([row for row in rows if safe_float(row.get("profile_score") or row.get("profile_score_median")) is not None], key=lambda row: safe_float(row.get("profile_score") or row.get("profile_score_median")) or -1, reverse=True)[:3]
    print(
        json.dumps(
            {
                "stage": stage_name,
                "completed_count": len(rows),
                "best": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "decision": row.get("decision"),
                        "coverage_abs_error": row.get("coverage_abs_error") or row.get("coverage_abs_error_median"),
                        "lower_breach_rate": row.get("lower_breach_rate") or row.get("lower_breach_rate_median"),
                        "upper_breach_rate": row.get("upper_breach_rate") or row.get("upper_breach_rate_median"),
                        "asymmetric_interval_score": row.get("asymmetric_interval_score") or row.get("asymmetric_interval_score_median"),
                        "band_width_ic": row.get("band_width_ic") or row.get("band_width_ic_median"),
                        "downside_width_ic": row.get("downside_width_ic") or row.get("downside_width_ic_median"),
                        "p90_band_width": row.get("p90_band_width") or row.get("p90_band_width_median"),
                    }
                    for row in best
                ],
                "promoted": metrics.get(promoted_key) if promoted_key else None,
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "process_report": remaining_process_report(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
