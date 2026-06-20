from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 이번 CP는 local yfinance 500 parquet만 사용한다.
os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ai.cp178_bm_1w_band_500_stage0_2 as s02  # noqa: E402
from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
)
from ai.ticker_registry import load_registry  # noqa: E402
from ai.train import TrainConfig, run_training  # noqa: E402


CP_NAME = "CP178-BM"
TIMEFRAME = "1W"
HORIZON = 4
FEATURE_SET = "price_volatility_volume"
TARGET_TYPE = "raw_future_return"
SOURCE_DATA_HASH = "90666b44cbfb8e5c"
Q_LABEL = "q10_q90"
Q_LOW = 0.10
Q_HIGH = 0.90
STAGE_TRAIN_EPOCHS = 3
SEEDS = [42, 7, 123]

BASE_LOG_DIR = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage3_5_logs"
STAGE3_LOG_DIR = BASE_LOG_DIR / "stage3"
STAGE4_LOG_DIR = BASE_LOG_DIR / "stage4"
STAGE5_LOG_DIR = BASE_LOG_DIR / "stage5_true_walk_forward"

STAGE0_2_METRICS = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage0_2_metrics.json"

STAGE3_REPORT = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage3_calibration_rescue_report.md"
STAGE3_METRICS = (
    PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage3_calibration_rescue_metrics.json"
)
STAGE3_SWEEP_CSV = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage3_calibration_sweep.csv"
STAGE3_SELECTED_JSON = (
    PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage3_selected_calibrations.json"
)
STAGE3_SELECTED_CSV = (
    PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage3_selected_calibrations.csv"
)

STAGE4_REPORT = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage4_seed_stability_report.md"
STAGE4_METRICS = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage4_seed_stability_metrics.json"
STAGE4_SUMMARY_CSV = (
    PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage4_seed_stability_summary.csv"
)

STAGE5_REPORT = PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage5_true_walk_forward_report.md"
STAGE5_METRICS = (
    PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage5_true_walk_forward_metrics.json"
)
STAGE5_SUMMARY_CSV = (
    PROJECT_ROOT / "docs" / "cp178_bm_1w_band_500_stage5_true_walk_forward_summary.csv"
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    family: str
    seq_len: int
    q_low: float
    q_high: float
    band_mode: str
    batch_size: int = 256
    fp32_modules: str = "none"
    patch_len: int = 16
    patch_stride: int = 8


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


CANDIDATES = [
    Candidate(
        "cnn_s104_q10q90_direct",
        "cnn_lstm",
        "cnn_lstm",
        104,
        Q_LOW,
        Q_HIGH,
        "direct",
        fp32_modules="lstm,heads",
    ),
    Candidate("tide_s104_q10q90_param", "tide", "tide", 104, Q_LOW, Q_HIGH, "param"),
    Candidate("tcn_s104_q10q90_param", "tcn_quantile", "tcn_quantile", 104, Q_LOW, Q_HIGH, "param"),
]

FOLDS = [
    FoldSpec(
        "fold_1", "2016-02-19", "2024-05-01", "2024-05-01", "2024-11-01", "2024-11-01", "2025-05-01"
    ),
    FoldSpec(
        "fold_2", "2016-02-19", "2024-11-01", "2024-11-01", "2025-05-01", "2025-05-01", "2025-11-01"
    ),
    FoldSpec(
        "fold_3", "2016-02-19", "2025-05-01", "2025-05-01", "2025-11-01", "2025-11-01", "2026-05-09"
    ),
]

_FOLD_BUNDLE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(clean_json(row.get(key)), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    return float(np.mean(finite)) if finite else None


def std(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def worst(values: list[Any], *, higher_is_better: bool = False) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    if not finite:
        return None
    return min(finite) if higher_is_better else max(finite)


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.6f}"


def ensure_dirs() -> None:
    for path in (BASE_LOG_DIR, STAGE3_LOG_DIR, STAGE4_LOG_DIR, STAGE5_LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def candidate_by_id(candidate_id: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def load_stage0_2() -> dict[str, Any]:
    metrics = read_json(STAGE0_2_METRICS)
    if metrics.get("final_status") != "PASS_STAGE2_CANDIDATES_FOUND":
        raise RuntimeError(
            f"Stage 0~2 상태가 Stage 3 진입 가능 상태가 아니다: {metrics.get('final_status')}"
        )
    return metrics


def q10_baseline_profile(stage0_2: dict[str, Any]) -> dict[str, Any]:
    profile = stage0_2["stage1"]["profiles"]["by_q"][Q_LABEL]
    val_rows = profile["validation_rows"]
    return {
        "sota": profile["baseline_sota_validation"],
        "p90_overwide_threshold": profile["p90_overwide_threshold"],
        "validation_rows": val_rows,
    }


def stage2_process(stage0_2: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    process = stage0_2["stage2"]["processes"].get(candidate_id)
    if not process:
        raise KeyError(f"Stage 2 process가 없다: {candidate_id}")
    return process


def collect_predictions_from_checkpoint(
    candidate: Candidate, checkpoint_path: str | Path, split: str = "val"
) -> dict[str, Any]:
    price, indicators, _, _ = s02.load_source_frames()
    model, config, feature_mean, feature_std = s02.load_model_from_checkpoint(checkpoint_path)
    train, val, test, plan = s02.build_split_for_checkpoint(
        price=price,
        indicators=indicators,
        checkpoint_config=config,
        source_data_hash=SOURCE_DATA_HASH,
    )
    bundle = {"train": train, "val": val, "test": test}[split]
    feature_columns = list(
        config.get("feature_columns") or s02.resolve_feature_columns(FEATURE_SET)
    )
    selected = s02.select_bundle_features_with_checkpoint_stats(
        bundle,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    device = s02.resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = s02.make_loader(selected, batch_size=512, shuffle=False, device=device, num_workers=0)
    lower_list: list[torch.Tensor] = []
    upper_list: list[torch.Tensor] = []
    actual_list: list[torch.Tensor] = []
    with torch.no_grad():
        for (
            features,
            line_target,
            band_target,
            raw_future_returns,
            ticker_id,
            future_covariates,
        ) in loader:
            del line_target, band_target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with s02.autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = s02.forward_model(model, features, ticker_id, future_covariates)
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
                raise TypeError(f"band 후보에서 허용하지 않는 출력 타입: {type(output).__name__}")
            lower_list.append(lower)
            upper_list.append(upper)
            actual_list.append(raw_future_returns.detach().cpu().to(torch.float32))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "candidate_id": candidate.candidate_id,
        "split": split,
        "lower": torch.cat(lower_list, dim=0).numpy(),
        "upper": torch.cat(upper_list, dim=0).numpy(),
        "actual": torch.cat(actual_list, dim=0).numpy(),
        "metadata": selected.metadata.copy(),
        "config": config,
        "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "eligible_ticker_count": len(plan.eligible_tickers),
    }


def summarize_arrays(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    metadata: pd.DataFrame,
    q_low: float,
    q_high: float,
) -> dict[str, Any]:
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
    metrics = {key: safe_float(summary.get(key)) for key in s02.BAND_METRIC_KEYS if key in summary}
    date_cs = s02._date_cross_sectional_width_ic(
        metadata=metadata, lower=lower, upper=upper, actual=actual
    )
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    metrics.update(regime_lower_breach_metrics(lower, upper, actual))
    return metrics


def regime_lower_breach_metrics(
    lower: np.ndarray, upper: np.ndarray, actual: np.ndarray
) -> dict[str, Any]:
    del upper
    row_abs = np.mean(np.abs(actual), axis=1)
    row_mean = np.mean(actual, axis=1)
    vol_cut = float(np.nanmedian(row_abs)) if row_abs.size else math.nan
    masks = {
        "high_vol": row_abs >= vol_cut,
        "low_vol": row_abs < vol_cut,
        "falling": row_mean < 0.0,
        "rising": row_mean >= 0.0,
    }
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            out[f"{name}_lower_breach_rate"] = None
            out[f"{name}_sample_rows"] = 0
            continue
        out[f"{name}_lower_breach_rate"] = float((actual[mask] < lower[mask]).mean())
        out[f"{name}_sample_rows"] = int(mask.sum())
    return out


def calibration_candidates(raw_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [{"method": "raw", "params": {}, "complexity": 0.0}]
    for scale in (1.05, 1.10, 1.15):
        candidates.append(
            {
                "method": "symmetric_expand",
                "params": {"lower_scale": scale, "upper_scale": scale},
                "complexity": 0.02,
            }
        )
    for lower_scale in (1.05, 1.10, 1.15):
        candidates.append(
            {
                "method": "lower_focused",
                "params": {"lower_scale": lower_scale, "upper_scale": 1.0},
                "complexity": 0.02,
            }
        )
    for upper_scale in (0.95, 1.00):
        candidates.append(
            {
                "method": "upper_trimmed",
                "params": {"lower_scale": 1.05, "upper_scale": upper_scale},
                "complexity": 0.03,
            }
        )
    for width_scale in (0.90, 1.00, 1.10):
        candidates.append(
            {
                "method": "center_preserving_width_scale",
                "params": {"width_scale": width_scale},
                "complexity": 0.01,
            }
        )
    raw_lower_rate = safe_float(raw_metrics.get("lower_breach_rate"))
    if raw_lower_rate is not None and raw_lower_rate > Q_LOW + 0.02:
        for lower_scale in (1.05, 1.10, 1.15):
            candidates.append(
                {
                    "method": "lower_breach_guard",
                    "params": {"lower_scale": lower_scale, "upper_scale": 1.0},
                    "complexity": 0.04,
                    "guard_triggered": True,
                }
            )
    else:
        candidates.append(
            {
                "method": "lower_breach_guard",
                "params": {"lower_scale": 1.0, "upper_scale": 1.0},
                "complexity": 0.04,
                "guard_triggered": False,
            }
        )
    return candidates


def apply_calibration(
    pred: dict[str, Any], calibration: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    raw_lower = np.asarray(pred["lower"], dtype=np.float32)
    raw_upper = np.asarray(pred["upper"], dtype=np.float32)
    lower = np.minimum(raw_lower, raw_upper)
    upper = np.maximum(raw_lower, raw_upper)
    method = calibration["method"]
    params = calibration.get("params") or {}
    if method == "raw":
        return lower.copy(), upper.copy()
    center = (lower + upper) / 2.0
    lower_width = np.maximum(center - lower, 0.0)
    upper_width = np.maximum(upper - center, 0.0)
    if method in {"symmetric_expand", "lower_focused", "upper_trimmed", "lower_breach_guard"}:
        lower_scale = float(params.get("lower_scale", 1.0))
        upper_scale = float(params.get("upper_scale", 1.0))
        return center - lower_width * lower_scale, center + upper_width * upper_scale
    if method == "center_preserving_width_scale":
        width_scale = float(params.get("width_scale", 1.0))
        return center - lower_width * width_scale, center + upper_width * width_scale
    raise ValueError(f"지원하지 않는 calibration method: {method}")


def calibration_score(metrics: dict[str, Any], profile: dict[str, Any], complexity: float) -> float:
    coverage = safe_float(metrics.get("coverage_abs_error")) or 1.0
    lower_abs = safe_float(metrics.get("lower_breach_abs_error")) or 1.0
    upper_abs = safe_float(metrics.get("upper_breach_abs_error")) or 1.0
    interval = safe_float(metrics.get("asymmetric_interval_score")) or 1.0
    band_ic = safe_float(metrics.get("band_width_ic")) or 0.0
    downside_ic = safe_float(metrics.get("downside_width_ic")) or 0.0
    p90 = safe_float(metrics.get("p90_band_width")) or 1.0
    p90_gate = safe_float(profile.get("p90_overwide_threshold"))
    overwide_penalty = 0.0
    if p90_gate is not None and p90 > p90_gate:
        overwide_penalty = (p90 / max(p90_gate, 1e-6)) - 1.0
    return (
        -2.0 * coverage
        - 1.0 * lower_abs
        - 0.5 * upper_abs
        - 0.8 * interval
        + 0.25 * band_ic
        + 0.25 * downside_ic
        - 0.05 * p90
        - 0.3 * overwide_penalty
        - complexity
    )


def classify_calibration_row(row: dict[str, Any], profile: dict[str, Any]) -> str:
    coverage = safe_float(row.get("coverage_abs_error"))
    lower_abs = safe_float(row.get("lower_breach_abs_error"))
    p90 = safe_float(row.get("p90_band_width"))
    p90_gate = safe_float(profile.get("p90_overwide_threshold"))
    band_ic = safe_float(row.get("band_width_ic"))
    downside_ic = safe_float(row.get("downside_width_ic"))
    if coverage is None or lower_abs is None or p90 is None:
        return "fail_metric_missing"
    if coverage > 0.20:
        return "fail_coverage_collapse"
    if p90_gate is not None and p90 > p90_gate:
        return "fail_overwide"
    if band_ic is not None and band_ic <= 0:
        return "fail_negative_width_ic"
    if downside_ic is not None and downside_ic < -0.02:
        return "fail_negative_downside_ic"
    return "eligible"


def run_stage3(force: bool = False) -> dict[str, Any]:
    if STAGE3_METRICS.exists() and not force:
        return json.loads(STAGE3_METRICS.read_text(encoding="utf-8"))
    started = time.perf_counter()
    stage0_2 = load_stage0_2()
    profile = q10_baseline_profile(stage0_2)
    sweep_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        process = stage2_process(stage0_2, candidate.candidate_id)
        checkpoint_path = process.get("checkpoint_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            sweep_rows.append(
                {"candidate_id": candidate.candidate_id, "status": "missing_checkpoint"}
            )
            continue
        val_pred = collect_predictions_from_checkpoint(candidate, checkpoint_path, split="val")
        raw_lower, raw_upper = apply_calibration(val_pred, {"method": "raw", "params": {}})
        raw_metrics = summarize_arrays(
            raw_lower,
            raw_upper,
            val_pred["actual"],
            val_pred["metadata"],
            candidate.q_low,
            candidate.q_high,
        )
        for cal in calibration_candidates(raw_metrics):
            lower, upper = apply_calibration(val_pred, cal)
            metrics = summarize_arrays(
                lower,
                upper,
                val_pred["actual"],
                val_pred["metadata"],
                candidate.q_low,
                candidate.q_high,
            )
            score = calibration_score(metrics, profile, float(cal.get("complexity", 0.0)))
            decision = classify_calibration_row(metrics, profile)
            row = {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "model": candidate.model,
                "seq_len": candidate.seq_len,
                "q_label": Q_LABEL,
                "calibration_method": cal["method"],
                "calibration_params": cal.get("params") or {},
                "guard_triggered": cal.get("guard_triggered"),
                "complexity": cal.get("complexity", 0.0),
                "selection_score": score,
                "validation_decision": decision,
                "split": "validation",
                **metrics,
            }
            sweep_rows.append(row)
    for candidate_id, group in group_by(sweep_rows, "candidate_id").items():
        eligible = [row for row in group if row.get("validation_decision") == "eligible"]
        pool = eligible or [
            row for row in group if safe_float(row.get("selection_score")) is not None
        ]
        if not pool:
            continue
        best = max(pool, key=lambda row: float(row.get("selection_score") or -999.0))
        promoted = dict(best)
        promoted["stage4_promotion"] = (
            "promoted" if best.get("validation_decision") == "eligible" else "research_reserve"
        )
        selected.append(promoted)
        candidate = candidate_by_id(candidate_id)
        checkpoint_path = stage2_process(stage0_2, candidate_id).get("checkpoint_path")
        if checkpoint_path:
            test_pred = collect_predictions_from_checkpoint(
                candidate, checkpoint_path, split="test"
            )
            lower, upper = apply_calibration(
                test_pred,
                {
                    "method": best["calibration_method"],
                    "params": best.get("calibration_params") or {},
                },
            )
            metrics = summarize_arrays(
                lower,
                upper,
                test_pred["actual"],
                test_pred["metadata"],
                candidate.q_low,
                candidate.q_high,
            )
            test_rows.append(
                {
                    "candidate_id": candidate_id,
                    "calibration_method": best["calibration_method"],
                    "calibration_params": best.get("calibration_params") or {},
                    "split": "test",
                    "diagnostic_only": True,
                    **metrics,
                }
            )
    selected = sorted(
        selected, key=lambda row: float(row.get("selection_score") or -999.0), reverse=True
    )[:3]
    all_bad = not selected or all(
        str(row.get("validation_decision", "")).startswith("fail") for row in selected
    )
    metrics = {
        "cp": CP_NAME,
        "stage": "Stage 3 calibration/rescue",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": contract_payload(),
        "baseline_profile": profile,
        "calibration_rows": sweep_rows,
        "selected_calibrations": selected,
        "test_diagnostic_rows": test_rows,
        "stage4_promoted_candidate_ids": [row["candidate_id"] for row in selected]
        if not all_bad
        else [],
        "stop_rule": {
            "all_candidates_overwide_or_collapse": all_bad,
            "stage4_allowed": not all_bad,
        },
        "policy": {
            "calibration_fit_split": "validation",
            "test_used_for_selection": False,
            "q_changed": False,
        },
    }
    write_json(STAGE3_METRICS, metrics)
    write_json(STAGE3_SELECTED_JSON, {"selected_calibrations": selected})
    write_rows(STAGE3_SWEEP_CSV, sweep_rows)
    write_rows(STAGE3_SELECTED_CSV, selected)
    write_stage3_report(metrics)
    return metrics


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key)), []).append(row)
    return grouped


def contract_payload() -> dict[str, Any]:
    return {
        "provider": "yfinance",
        "source": "yfinance",
        "source_data_hash": SOURCE_DATA_HASH,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": FEATURE_SET,
        "feature_columns": s02.resolve_feature_columns(FEATURE_SET),
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "target": TARGET_TYPE,
        "split": "calendar_aligned",
        "stage3_or_4_test_usage": "diagnostic_only",
    }


def build_calendar_payload(candidate: Candidate) -> dict[str, Any]:
    price, indicators, _, _ = s02.load_source_frames()
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=candidate.seq_len,
        horizon=HORIZON,
        market_data_provider="yfinance",
        source_data_hash=SOURCE_DATA_HASH,
        split_mode="calendar_aligned",
    )
    registry = load_registry(TIMEFRAME, Path(plan.ticker_registry_path))
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(plan.eligible_tickers)].copy(),
        price_df=price[price["ticker"].isin(plan.eligible_tickers)].copy(),
        timeframe=TIMEFRAME,
        seq_len=candidate.seq_len,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train, val, test, calendar_plan = s02.split_sequence_dataset_calendar_aligned(
        dataset,
        purge_gap_trading_days=plan.h_max,
        min_fold_samples=plan.min_fold_samples,
    )
    s02.apply_calendar_split_metadata(plan, calendar_plan)
    train_norm, val_norm, test_norm, mean_t, std_t = normalize_sequence_splits(train, val, test)
    return {
        "plan": plan,
        "dataset": dataset,
        "train_raw": train,
        "val_raw": val,
        "test_raw": test,
        "train_norm": train_norm,
        "val_norm": val_norm,
        "test_norm": test_norm,
        "mean": mean_t,
        "std": std_t,
    }


def make_train_config(candidate: Candidate, seed: int) -> TrainConfig:
    return TrainConfig(
        model=candidate.model,
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        seq_len=candidate.seq_len,
        epochs=STAGE_TRAIN_EPOCHS,
        batch_size=candidate.batch_size,
        lr=1e-4,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=candidate.q_low,
        q_high=candidate.q_high,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.2,
        band_mode=candidate.band_mode,
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=candidate.model == "tide",
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules=candidate.fp32_modules,
        use_wandb=False,
        wandb_project="lens-ai",
        model_ver="v2-multihead",
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
        checkpoint_selection="band_gate",
        amp_dtype="bf16",
        detect_anomaly=False,
        explicit_cuda_cleanup=True,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=candidate.patch_len,
        patch_stride=candidate.patch_stride,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=FEATURE_SET,
        market_data_provider="yfinance",
        split_mode="calendar_aligned",
        lower_band_loss_weight=1.0,
        upper_band_loss_weight=1.0,
        model_role="band",
        lambda_risk=0.5,
        lambda_regime=0.5,
        lambda_ordinal=0.1,
        risk_decision_threshold=0.5,
    )


def train_with_bundles(
    *,
    candidate: Candidate,
    seed: int,
    bundles: dict[str, Any],
    run_dir: Path,
    force: bool,
    fold: FoldSpec | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    process_path = run_dir / "train_process.json"
    if not force and process_path.exists():
        existing = read_json(process_path)
        checkpoint_path = existing.get("checkpoint_path")
        if existing.get("status") == "PASS" and checkpoint_path and Path(checkpoint_path).exists():
            return existing
    config = make_train_config(candidate, seed)
    started = time.perf_counter()
    stdout_path = run_dir / "train_stdout.log"
    print(
        json.dumps(
            {
                "event": "train_start",
                "candidate": candidate.candidate_id,
                "seed": seed,
                "fold": fold.fold_id if fold else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    try:
        with stdout_path.open("w", encoding="utf-8", newline="") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                result = run_training(
                    config,
                    save_run=False,
                    precomputed_bundles=(
                        bundles["train_norm"],
                        bundles["val_norm"],
                        bundles["test_norm"],
                        bundles["mean"],
                        bundles["std"],
                        bundles["plan"],
                    ),
                    enable_compile=False,
                    wandb_required=False,
                    local_log=True,
                    local_log_dir=run_dir / "ai_train_local_logs",
                )
        status = "PASS"
        failure_reason = None
        failure_category = None
    except Exception as exc:  # noqa: BLE001
        result = {"error": repr(exc)}
        status = "FAIL"
        failure_reason = repr(exc)
        failure_category = (
            "contract"
            if "split" in repr(exc).lower() or "finite" in repr(exc).lower()
            else "runtime"
        )
    elapsed = time.perf_counter() - started
    process = {
        "candidate": asdict(candidate),
        "seed": seed,
        "fold": asdict(fold) if fold else None,
        "status": status,
        "failure_reason": failure_reason,
        "failure_category": failure_category,
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": 0 if status == "PASS" else 1,
        "run_id": result.get("run_id"),
        "checkpoint_path": result.get("checkpoint_path"),
        "local_log_dir": result.get("local_log_dir"),
        "best_metrics": result.get("best_metrics"),
        "test_metrics_from_train_readonly": result.get("test_metrics"),
        "dataset_plan": result.get("dataset_plan"),
        "feature_set": result.get("feature_set"),
        "feature_columns": result.get("feature_columns"),
        "n_features": result.get("n_features"),
        "stdout_path": str(stdout_path),
        "train_process_path": str(process_path),
        "save_run": False,
        "db_write": False,
        "inference_saved": False,
        "wandb": False,
        "wandb_status": result.get("wandb_status"),
    }
    write_json(process_path, process)
    print(
        json.dumps(
            {
                "event": "train_end",
                "candidate": candidate.candidate_id,
                "seed": seed,
                "fold": fold.fold_id if fold else None,
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return process


def collect_predictions_for_bundle(checkpoint_path: str | Path, bundle: Any) -> dict[str, Any]:
    model, config, feature_mean, feature_std = s02.load_model_from_checkpoint(checkpoint_path)
    feature_columns = list(
        config.get("feature_columns") or s02.resolve_feature_columns(FEATURE_SET)
    )
    selected = s02.select_bundle_features_with_checkpoint_stats(
        bundle,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    device = s02.resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = s02.make_loader(selected, batch_size=512, shuffle=False, device=device, num_workers=0)
    lower_list: list[torch.Tensor] = []
    upper_list: list[torch.Tensor] = []
    actual_list: list[torch.Tensor] = []
    with torch.no_grad():
        for (
            features,
            line_target,
            band_target,
            raw_future_returns,
            ticker_id,
            future_covariates,
        ) in loader:
            del line_target, band_target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with s02.autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = s02.forward_model(model, features, ticker_id, future_covariates)
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
                raise TypeError(f"지원하지 않는 출력 타입: {type(output).__name__}")
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
    }


def fit_best_calibration(pred: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    raw_lower, raw_upper = apply_calibration(pred, {"method": "raw", "params": {}})
    raw_metrics = summarize_arrays(
        raw_lower, raw_upper, pred["actual"], pred["metadata"], Q_LOW, Q_HIGH
    )
    rows = []
    for cal in calibration_candidates(raw_metrics):
        lower, upper = apply_calibration(pred, cal)
        metrics = summarize_arrays(lower, upper, pred["actual"], pred["metadata"], Q_LOW, Q_HIGH)
        row = {
            "method": cal["method"],
            "params": cal.get("params") or {},
            "complexity": cal.get("complexity", 0.0),
            "guard_triggered": cal.get("guard_triggered"),
            "metrics": metrics,
            "decision": classify_calibration_row(metrics, profile),
            "selection_score": calibration_score(
                metrics, profile, float(cal.get("complexity", 0.0))
            ),
        }
        rows.append(row)
    eligible = [row for row in rows if row["decision"] == "eligible"]
    pool = eligible or rows
    best = max(pool, key=lambda row: float(row.get("selection_score") or -999.0))
    return {
        "method": best["method"],
        "params": best.get("params") or {},
        "selection_metrics": best["metrics"],
        "selection_score": best["selection_score"],
        "decision": best["decision"],
        "sweep_rows": rows,
    }


def evaluate_process_with_calibration(
    candidate: Candidate, process: dict[str, Any], bundles: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {"status": "FAIL", "failure_reason": "checkpoint_missing"}
    val_pred = collect_predictions_for_bundle(checkpoint_path, bundles["val_raw"])
    test_pred = collect_predictions_for_bundle(checkpoint_path, bundles["test_raw"])
    calibration = fit_best_calibration(val_pred, profile)
    lower_val, upper_val = apply_calibration(val_pred, calibration)
    lower_test, upper_test = apply_calibration(test_pred, calibration)
    val_metrics = summarize_arrays(
        lower_val,
        upper_val,
        val_pred["actual"],
        val_pred["metadata"],
        candidate.q_low,
        candidate.q_high,
    )
    test_metrics = summarize_arrays(
        lower_test,
        upper_test,
        test_pred["actual"],
        test_pred["metadata"],
        candidate.q_low,
        candidate.q_high,
    )
    return {
        "status": "PASS",
        "calibration": calibration,
        "val_metrics": val_metrics,
        "test_metrics_diagnostic": test_metrics,
        "test_used_for_selection": False,
    }


def metric_gap(val_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "asymmetric_interval_score",
        "band_width_ic",
        "downside_width_ic",
        "p90_band_width",
    ]
    return {
        f"{key}_test_minus_val": (
            safe_float(test_metrics.get(key)) - safe_float(val_metrics.get(key))
            if safe_float(test_metrics.get(key)) is not None
            and safe_float(val_metrics.get(key)) is not None
            else None
        )
        for key in keys
    }


def flatten_eval_row(
    candidate: Candidate,
    seed: int,
    process: dict[str, Any],
    evaluation: dict[str, Any],
    stage: str,
    fold: FoldSpec | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "model": candidate.model,
        "seq_len": candidate.seq_len,
        "q_label": Q_LABEL,
        "band_mode": candidate.band_mode,
        "seed": seed,
        "fold_id": fold.fold_id if fold else None,
        "status": process.get("status"),
        "elapsed_seconds": process.get("elapsed_seconds"),
        "run_id": process.get("run_id"),
        "checkpoint_path": process.get("checkpoint_path"),
        "failure_category": process.get("failure_category"),
        "failure_reason": process.get("failure_reason"),
    }
    if evaluation.get("status") == "PASS":
        calibration = evaluation["calibration"]
        row["calibration_method"] = calibration["method"]
        row["calibration_params"] = calibration.get("params") or {}
        row["calibration_selection_score"] = calibration.get("selection_score")
        for prefix, metrics in (
            ("val", evaluation["val_metrics"]),
            ("test", evaluation["test_metrics_diagnostic"]),
        ):
            for key in [
                "coverage_abs_error",
                "lower_breach_rate",
                "upper_breach_rate",
                "lower_breach_abs_error",
                "upper_breach_abs_error",
                "asymmetric_interval_score",
                "median_band_width",
                "p90_band_width",
                "band_width_ic",
                "downside_width_ic",
                "width_bucket_realized_vol_ratio",
                "squeeze_breakout_rate",
                "high_vol_lower_breach_rate",
                "low_vol_lower_breach_rate",
                "falling_lower_breach_rate",
                "rising_lower_breach_rate",
            ]:
                row[f"{prefix}_{key}"] = metrics.get(key)
        row.update(metric_gap(evaluation["val_metrics"], evaluation["test_metrics_diagnostic"]))
    return row


def run_stage4(stage3: dict[str, Any], force: bool = False) -> dict[str, Any]:
    if STAGE4_METRICS.exists() and not force:
        metrics = json.loads(STAGE4_METRICS.read_text(encoding="utf-8"))
        promoted = [
            row["candidate_id"]
            for row in metrics.get("summary_rows", [])
            if row.get("stage5_promotion") in {"promoted", "research_reserve"}
        ][:3]
        metrics["stage5_promoted_candidate_ids"] = promoted
        metrics.setdefault("policy", {})["stage5_candidate_limit_note"] = (
            "세 후보 모두 validation 기준 promoted라 true WF seed42에서 먼저 확인한다."
        )
        metrics.setdefault("stop_rule", {})["stage5_allowed"] = bool(promoted)
        write_json(STAGE4_METRICS, metrics)
        write_stage4_report(metrics)
        return metrics
    started = time.perf_counter()
    profile = stage3["baseline_profile"]
    promoted_ids = stage3.get("stage4_promoted_candidate_ids") or []
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for candidate_id in promoted_ids:
        candidate = candidate_by_id(candidate_id)
        bundles = build_calendar_payload(candidate)
        for seed in SEEDS:
            run_dir = STAGE4_LOG_DIR / f"{candidate.candidate_id}_seed{seed}"
            process = train_with_bundles(
                candidate=candidate, seed=seed, bundles=bundles, run_dir=run_dir, force=force
            )
            if process.get("status") == "PASS":
                evaluation = evaluate_process_with_calibration(candidate, process, bundles, profile)
                write_json(run_dir / "evaluation.json", evaluation)
                write_json(run_dir / "calibration_params.json", evaluation.get("calibration") or {})
            else:
                evaluation = {"status": "FAIL", "failure_reason": process.get("failure_reason")}
            details[f"{candidate.candidate_id}_seed{seed}"] = {
                "process": process,
                "evaluation": evaluation,
            }
            rows.append(flatten_eval_row(candidate, seed, process, evaluation, "stage4"))
    summary = summarize_stage4(rows)
    promoted_stage5 = [
        row["candidate_id"]
        for row in summary
        if row.get("stage5_promotion") in {"promoted", "research_reserve"}
    ][:3]
    metrics = {
        "cp": CP_NAME,
        "stage": "Stage 4 seed stability",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": contract_payload(),
        "stage3_promoted": promoted_ids,
        "seed_rows": rows,
        "summary_rows": summary,
        "details": details,
        "stage5_promoted_candidate_ids": promoted_stage5,
        "stop_rule": {
            "stage5_allowed": bool(promoted_stage5),
            "reason": None if promoted_stage5 else "all_candidates_unstable_or_collapsed",
        },
        "policy": {
            "selection_basis": "validation_median_and_worst_seed",
            "test_used_for_selection": False,
            "calibration_fit_split": "seed_validation",
        },
    }
    write_json(STAGE4_METRICS, metrics)
    write_rows(STAGE4_SUMMARY_CSV, rows + summary)
    write_stage4_report(metrics)
    return metrics


def summarize_stage4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    metric_keys = [
        ("val_coverage_abs_error", False),
        ("val_lower_breach_rate", False),
        ("val_asymmetric_interval_score", False),
        ("val_band_width_ic", True),
        ("val_downside_width_ic", True),
        ("val_p90_band_width", False),
    ]
    for candidate_id, group in group_by(rows, "candidate_id").items():
        item: dict[str, Any] = {
            "stage": "stage4_summary",
            "candidate_id": candidate_id,
            "seed_count": len(group),
        }
        methods = [
            str(row.get("calibration_method")) for row in group if row.get("calibration_method")
        ]
        item["calibration_methods"] = ",".join(sorted(set(methods)))
        for key, higher in metric_keys:
            values = [row.get(key) for row in group]
            item[f"{key}_median"] = (
                float(np.nanmedian([float(v) for v in values if safe_float(v) is not None]))
                if any(safe_float(v) is not None for v in values)
                else None
            )
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = std(values)
            item[f"{key}_worst"] = worst(values, higher_is_better=higher)
        item["stage5_promotion"] = classify_stage4_summary(item)
        summary.append(item)
    return sorted(
        summary, key=lambda row: safe_float(row.get("val_asymmetric_interval_score_mean")) or 999.0
    )


def classify_stage4_summary(row: dict[str, Any]) -> str:
    cov_median = safe_float(row.get("val_coverage_abs_error_median"))
    cov_worst = safe_float(row.get("val_coverage_abs_error_worst"))
    p90_worst = safe_float(row.get("val_p90_band_width_worst"))
    band_worst = safe_float(row.get("val_band_width_ic_worst"))
    downside_worst = safe_float(row.get("val_downside_width_ic_worst"))
    profile = q10_baseline_profile(load_stage0_2())
    p90_gate = safe_float(profile.get("p90_overwide_threshold"))
    if cov_worst is not None and cov_worst > 0.20:
        return "rejected"
    if p90_gate is not None and p90_worst is not None and p90_worst > p90_gate * 1.25:
        return "rejected"
    if band_worst is not None and band_worst <= 0:
        return "rejected"
    if (
        cov_median is not None
        and cov_median <= 0.08
        and (downside_worst is None or downside_worst >= -0.02)
    ):
        return "promoted"
    if cov_median is not None and cov_median <= 0.12:
        return "research_reserve"
    return "rejected"


def build_fold_bundles(candidate: Candidate, fold: FoldSpec) -> dict[str, Any]:
    cache_key = (candidate.candidate_id, fold.fold_id)
    if cache_key in _FOLD_BUNDLE_CACHE:
        return _FOLD_BUNDLE_CACHE[cache_key]
    price, indicators, _, _ = s02.load_source_frames()
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=candidate.seq_len,
        horizon=HORIZON,
        market_data_provider="yfinance",
        source_data_hash=SOURCE_DATA_HASH,
        split_mode="calendar_aligned",
    )
    registry = load_registry(TIMEFRAME, Path(plan.ticker_registry_path))
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(plan.eligible_tickers)].copy(),
        price_df=price[price["ticker"].isin(plan.eligible_tickers)].copy(),
        timeframe=TIMEFRAME,
        seq_len=candidate.seq_len,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train_idx, val_idx, test_idx, split_diag = date_split_indices(dataset, fold, plan.h_max)
    train_raw = dataset.subset(train_idx)
    val_raw = dataset.subset(val_idx)
    test_raw = dataset.subset(test_idx)
    train_norm, val_norm, test_norm, mean_t, std_t = normalize_sequence_splits(
        train_raw, val_raw, test_raw
    )
    payload = {
        "plan": plan,
        "dataset": dataset,
        "train_raw": train_raw,
        "val_raw": val_raw,
        "test_raw": test_raw,
        "train_norm": train_norm,
        "val_norm": val_norm,
        "test_norm": test_norm,
        "mean": mean_t,
        "std": std_t,
        "split_diag": split_diag,
    }
    _FOLD_BUNDLE_CACHE[cache_key] = payload
    return payload


def date_split_indices(
    dataset: Any, fold: FoldSpec, h_max: int
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    metadata = dataset.metadata.copy()
    metadata["order_idx"] = range(len(metadata))
    metadata["asof_ts"] = pd.to_datetime(metadata["asof_date"], errors="coerce")
    train_start = pd.Timestamp(fold.train_start)
    train_end = pd.Timestamp(fold.train_end)
    val_start = pd.Timestamp(fold.val_start)
    val_end = pd.Timestamp(fold.val_end)
    test_start = pd.Timestamp(fold.test_start)
    test_end = pd.Timestamp(fold.test_end)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    excluded_tickers: dict[str, Any] = {}
    for ticker, group in metadata.groupby("ticker", sort=True):
        group = group.sort_values("sample_index").copy()
        train_mask = (group["asof_ts"] >= train_start) & (group["asof_ts"] < train_end)
        val_mask = (group["asof_ts"] >= val_start) & (group["asof_ts"] < val_end)
        test_mask = (group["asof_ts"] >= test_start) & (group["asof_ts"] < test_end)
        val_samples = group.loc[val_mask, "sample_index"]
        test_samples = group.loc[test_mask, "sample_index"]
        if not val_samples.empty:
            train_mask &= group["sample_index"] < int(val_samples.min()) - h_max
        if not test_samples.empty:
            val_mask &= group["sample_index"] < int(test_samples.min()) - h_max
        ticker_train = group.loc[train_mask, "order_idx"].astype(int).tolist()
        ticker_val = group.loc[val_mask, "order_idx"].astype(int).tolist()
        ticker_test = group.loc[test_mask, "order_idx"].astype(int).tolist()
        if ticker_train and ticker_val and ticker_test:
            train_indices.extend(ticker_train)
            val_indices.extend(ticker_val)
            test_indices.extend(ticker_test)
        else:
            excluded_tickers[str(ticker)] = {
                "train": len(ticker_train),
                "val": len(ticker_val),
                "test": len(ticker_test),
            }
    overlap = {
        "train_val": len(set(train_indices) & set(val_indices)),
        "train_test": len(set(train_indices) & set(test_indices)),
        "val_test": len(set(val_indices) & set(test_indices)),
    }
    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            json.dumps(
                {
                    "error": "empty_custom_fold_split",
                    "fold": asdict(fold),
                    "train": len(train_indices),
                    "val": len(val_indices),
                    "test": len(test_indices),
                },
                ensure_ascii=False,
            )
        )
    if any(overlap.values()):
        raise ValueError(
            json.dumps(
                {"error": "custom_split_overlap", "fold": asdict(fold), "overlap": overlap},
                ensure_ascii=False,
            )
        )
    split_diag = {
        "fold": asdict(fold),
        "h_max_gap": int(h_max),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "overlap": overlap,
        "excluded_tickers_for_fold_count": len(excluded_tickers),
        "excluded_tickers_for_fold_sample_counts": excluded_tickers,
    }
    return train_indices, val_indices, test_indices, split_diag


def fold_event_profile(bundles: dict[str, Any]) -> dict[str, Any]:
    train_targets = s02.collect_targets(bundles["train_raw"])
    low = float(np.nanquantile(train_targets.reshape(-1), Q_LOW))
    high = float(np.nanquantile(train_targets.reshape(-1), Q_HIGH))
    profile = {
        "train_quantile_low": low,
        "train_quantile_high": high,
        "event_count_unit": "horizon_point",
    }
    for split in ("train", "val", "test"):
        targets = s02.collect_targets(bundles[f"{split}_raw"])
        profile[f"{split}_event_counts"] = s02._target_event_counts(targets, low, high)
    return profile


def event_month_membership(fold: FoldSpec) -> dict[str, bool]:
    months = {
        "2024_12": ("2024-12-01", "2025-01-01"),
        "2025_04": ("2025-04-01", "2025-05-01"),
        "2026_02": ("2026-02-01", "2026-03-01"),
    }
    out: dict[str, bool] = {}
    for label, (start, end) in months.items():
        out[label] = pd.Timestamp(start) < pd.Timestamp(fold.test_end) and pd.Timestamp(
            end
        ) > pd.Timestamp(fold.test_start)
    return out


def run_stage5(stage4: dict[str, Any], force: bool = False) -> dict[str, Any]:
    if STAGE5_METRICS.exists() and not force:
        metrics = json.loads(STAGE5_METRICS.read_text(encoding="utf-8"))
        summary = summarize_stage5(metrics.get("fold_rows", []))
        metrics["summary_rows"] = summary
        metrics["final_candidate_label"] = final_true_wf_label(summary)
        write_json(STAGE5_METRICS, metrics)
        write_rows(STAGE5_SUMMARY_CSV, metrics.get("fold_rows", []) + summary)
        write_stage5_report(metrics)
        return metrics
    started = time.perf_counter()
    profile = q10_baseline_profile(load_stage0_2())
    candidate_ids = stage4.get("stage5_promoted_candidate_ids") or []
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    seed42_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = candidate_by_id(candidate_id)
        for fold in FOLDS:
            row, detail = run_stage5_one(candidate, fold, 42, profile, force)
            rows.append(row)
            seed42_rows.append(row)
            details[f"{candidate_id}_{fold.fold_id}_seed42"] = detail
    seed42_summary = summarize_stage5(rows)
    should_expand = any(
        row.get("decision")
        in {"product_candidate_true_wf", "research_reserve_true_wf", "warn_unstable_true_wf"}
        for row in seed42_summary
    )
    if should_expand:
        for seed in (7, 123):
            for candidate_id in candidate_ids:
                candidate = candidate_by_id(candidate_id)
                for fold in FOLDS:
                    row, detail = run_stage5_one(candidate, fold, seed, profile, force)
                    rows.append(row)
                    details[f"{candidate_id}_{fold.fold_id}_seed{seed}"] = detail
    summary = summarize_stage5(rows)
    final_label = final_true_wf_label(summary)
    metrics = {
        "cp": CP_NAME,
        "stage": "Stage 5 true walk-forward",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": contract_payload(),
        "folds": [asdict(fold) for fold in FOLDS],
        "seed42_summary_before_expansion": seed42_summary,
        "expanded_to_seeds_7_123": should_expand,
        "fold_rows": rows,
        "summary_rows": summary,
        "details": details,
        "final_candidate_label": final_label,
        "policy": {
            "true_fold_retraining": True,
            "checkpoint_selection_split": "fold_validation",
            "calibration_fit_split": "fold_validation",
            "test_used_for_calibration": False,
            "test_used_for_threshold_or_candidate_change": False,
            "existing_checkpoint_replay_used_as_true_wf": False,
        },
    }
    write_json(STAGE5_METRICS, metrics)
    write_rows(STAGE5_SUMMARY_CSV, rows + summary)
    write_stage5_report(metrics)
    return metrics


def run_stage5_one(
    candidate: Candidate, fold: FoldSpec, seed: int, profile: dict[str, Any], force: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundles = build_fold_bundles(candidate, fold)
    run_dir = STAGE5_LOG_DIR / f"{candidate.candidate_id}_{fold.fold_id}_seed{seed}"
    process = train_with_bundles(
        candidate=candidate, seed=seed, bundles=bundles, run_dir=run_dir, force=force, fold=fold
    )
    if process.get("status") == "PASS":
        evaluation = evaluate_process_with_calibration(candidate, process, bundles, profile)
        evaluation["fold_event_profile"] = fold_event_profile(bundles)
        evaluation["event_month_membership"] = event_month_membership(fold)
        write_json(run_dir / "fold_metrics.json", evaluation)
        write_json(run_dir / "calibration_params.json", evaluation.get("calibration") or {})
    else:
        evaluation = {"status": "FAIL", "failure_reason": process.get("failure_reason")}
    row = flatten_eval_row(candidate, seed, process, evaluation, "stage5_true_wf", fold=fold)
    if evaluation.get("status") == "PASS":
        row["test_used_for_selection"] = False
        row["fold_event_profile"] = evaluation.get("fold_event_profile")
        row["event_month_membership"] = evaluation.get("event_month_membership")
        row["split_diag"] = bundles.get("split_diag")
    return row, {"process": process, "evaluation": evaluation}


def summarize_stage5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    metric_keys = [
        ("test_coverage_abs_error", False),
        ("test_lower_breach_rate", False),
        ("test_upper_breach_rate", False),
        ("test_asymmetric_interval_score", False),
        ("test_band_width_ic", True),
        ("test_downside_width_ic", True),
        ("test_p90_band_width", False),
    ]
    for candidate_id, group in group_by(rows, "candidate_id").items():
        item: dict[str, Any] = {
            "stage": "stage5_summary",
            "candidate_id": candidate_id,
            "fold_seed_count": len(group),
        }
        item["fold_list"] = ",".join(
            sorted({str(row.get("fold_id")) for row in group if row.get("fold_id")})
        )
        item["seed_list"] = ",".join(
            sorted({str(row.get("seed")) for row in group if row.get("seed") is not None})
        )
        for key, higher in metric_keys:
            values = [row.get(key) for row in group]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = std(values)
            item[f"{key}_worst"] = worst(values, higher_is_better=higher)
        item["decision"] = classify_true_wf_candidate(item)
        summary.append(item)
    return sorted(
        summary, key=lambda row: safe_float(row.get("test_asymmetric_interval_score_mean")) or 999.0
    )


def classify_true_wf_candidate(row: dict[str, Any]) -> str:
    cov_mean = safe_float(row.get("test_coverage_abs_error_mean"))
    cov_worst = safe_float(row.get("test_coverage_abs_error_worst"))
    lower_mean = safe_float(row.get("test_lower_breach_rate_mean"))
    p90_worst = safe_float(row.get("test_p90_band_width_worst"))
    band_mean = safe_float(row.get("test_band_width_ic_mean"))
    downside_mean = safe_float(row.get("test_downside_width_ic_mean"))
    profile = q10_baseline_profile(load_stage0_2())
    p90_gate = safe_float(profile.get("p90_overwide_threshold"))
    if cov_mean is None:
        return "fail_no_true_wf_candidate"
    overwide = p90_gate is not None and p90_worst is not None and p90_worst > p90_gate * 1.25
    if (
        cov_mean <= 0.05
        and (cov_worst is None or cov_worst <= 0.10)
        and (lower_mean is None or lower_mean <= 0.18)
        and not overwide
        and (band_mean or 0.0) > 0.15
        and (downside_mean or 0.0) >= 0.0
    ):
        return "product_candidate_true_wf"
    if cov_mean <= 0.10 and not overwide and (band_mean or 0.0) > 0.05:
        return "research_reserve_true_wf"
    if cov_mean <= 0.15 and not overwide:
        return "warn_unstable_true_wf"
    return "fail_no_true_wf_candidate"


def final_true_wf_label(summary: list[dict[str, Any]]) -> str:
    decisions = {row.get("decision") for row in summary}
    for label in ("product_candidate_true_wf", "research_reserve_true_wf", "warn_unstable_true_wf"):
        if label in decisions:
            return label
    return "fail_no_true_wf_candidate"


def current_python_process_snapshot() -> dict[str, Any]:
    try:
        result = subprocess_run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "$p=Get-Process -Name python,pythonw -ErrorAction SilentlyContinue; if($p){$p|Select-Object Id,ProcessName,Path,StartTime|ConvertTo-Json -Depth 3}else{'NO_PYTHON_PROCESS'}",
            ]
        )
        return result
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": type(exc).__name__, "message": str(exc)}


def current_cuda_python_snapshot() -> dict[str, Any]:
    try:
        result = subprocess_run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "$r=nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | Select-String -Pattern 'python'; if($r){$r}else{'NO_VISIBLE_CUDA_PYTHON_PROCESS'}",
            ]
        )
        return result
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": type(exc).__name__, "message": str(exc)}


def subprocess_run(cmd: list[str]) -> dict[str, Any]:
    import subprocess

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_all(
    force: bool = False, stop_after_stage3: bool = False, stop_after_stage4: bool = False
) -> dict[str, Any]:
    ensure_dirs()
    started = time.perf_counter()
    process_before = {
        "python": current_python_process_snapshot(),
        "cuda_python": current_cuda_python_snapshot(),
    }
    stage3 = run_stage3(force=force)
    stage4 = None
    stage5 = None
    if not stop_after_stage3 and stage3["stop_rule"]["stage4_allowed"]:
        stage4 = run_stage4(stage3, force=force)
    if not stop_after_stage4 and stage4 and stage4["stop_rule"]["stage5_allowed"]:
        stage5 = run_stage5(stage4, force=force)
    final_label = "fail_no_true_wf_candidate"
    if stage5:
        final_label = stage5["final_candidate_label"]
    elif stage4 is None or not (stage4.get("stop_rule") or {}).get("stage5_allowed"):
        final_label = "fail_no_true_wf_candidate"
    metrics = {
        "cp": CP_NAME,
        "stage": "Stage 3~5 integrated",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": contract_payload(),
        "stage3_metrics_path": str(STAGE3_METRICS),
        "stage4_metrics_path": str(STAGE4_METRICS) if stage4 else None,
        "stage5_metrics_path": str(STAGE5_METRICS) if stage5 else None,
        "final_candidate_label": final_label,
        "stage3_executed": True,
        "stage4_executed": stage4 is not None,
        "stage5_executed": stage5 is not None,
        "process_before": process_before,
        "process_after": {
            "python": current_python_process_snapshot(),
            "cuda_python": current_cuda_python_snapshot(),
        },
        "policy": {
            "stage5_required_for_product_candidate": True,
            "stage3_4_are_candidate_compression": True,
            "one_week_independent_from_one_day": True,
            "one_day_artifacts_modified": False,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "composite": False,
        },
    }
    write_json(BASE_LOG_DIR / "cp178_stage3_5_integrated_metrics.json", metrics)
    return metrics


def write_stage3_report(metrics: dict[str, Any]) -> None:
    selected = metrics.get("selected_calibrations", [])
    lines = [
        "# CP178-BM 1W Band 500 Stage 3 Calibration Rescue Report",
        "",
        "- calibration은 validation에서만 fit했다.",
        "- test diagnostic table은 후보 선정에 사용하지 않았다.",
        "- q는 q10/q90으로 유지했다.",
        "",
        "| 후보 | method | cov_abs | lower_abs | interval | p90 | width_ic | downside_ic | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row.get('candidate_id')} | {row.get('calibration_method')} | {fmt(row.get('coverage_abs_error'))} | {fmt(row.get('lower_breach_abs_error'))} | {fmt(row.get('asymmetric_interval_score'))} | {fmt(row.get('p90_band_width'))} | {fmt(row.get('band_width_ic'))} | {fmt(row.get('downside_width_ic'))} | {row.get('validation_decision')} |"
        )
    lines.extend(
        [
            "",
            f"- Stage 4 진입 허용: `{metrics['stop_rule']['stage4_allowed']}`",
            f"- Stage 4 후보: `{metrics.get('stage4_promoted_candidate_ids')}`",
        ]
    )
    STAGE3_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage4_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# CP178-BM 1W Band 500 Stage 4 Seed Stability Report",
        "",
        "- seed 42/7/123 재학습 기준이다.",
        "- calibration은 seed별 validation에서 fit했고 test는 diagnostic only다.",
        "",
        "| 후보 | promotion | cov_median | cov_worst | p90_worst | width_ic_worst | downside_ic_worst |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.get("summary_rows", []):
        lines.append(
            f"| {row.get('candidate_id')} | {row.get('stage5_promotion')} | {fmt(row.get('val_coverage_abs_error_median'))} | {fmt(row.get('val_coverage_abs_error_worst'))} | {fmt(row.get('val_p90_band_width_worst'))} | {fmt(row.get('val_band_width_ic_worst'))} | {fmt(row.get('val_downside_width_ic_worst'))} |"
        )
    lines.extend(
        [
            "",
            f"- Stage 5 진입 허용: `{metrics['stop_rule']['stage5_allowed']}`",
            f"- Stage 5 후보: `{metrics.get('stage5_promoted_candidate_ids')}`",
        ]
    )
    STAGE4_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage5_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# CP178-BM 1W Band 500 Stage 5 True Walk-Forward Report",
        "",
        "- 이 보고서는 fold별 새 checkpoint 학습 결과만 제품 후보 판단 근거로 둔다.",
        "- 기존 Stage 2 test metric은 diagnostic_only다.",
        "- calibration은 각 fold validation에서만 fit했고 test 결과로 변경하지 않았다.",
        "- Stage 5 true WF를 통과하기 전까지는 제품 후보 확정이 아니다.",
        "- Stage 3/4는 후보 압축과 안정성 진단이다.",
        "- Stage 5가 실패해도 1W band가 실패했다는 뜻이지 1D band 결과를 훼손하지 않는다.",
        "",
        f"- 최종 label: `{metrics.get('final_candidate_label')}`",
        "",
        "| 후보 | decision | test_cov_mean | test_cov_worst | lower_mean | upper_mean | p90_worst | width_ic_mean | downside_ic_mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.get("summary_rows", []):
        lines.append(
            f"| {row.get('candidate_id')} | {row.get('decision')} | {fmt(row.get('test_coverage_abs_error_mean'))} | {fmt(row.get('test_coverage_abs_error_worst'))} | {fmt(row.get('test_lower_breach_rate_mean'))} | {fmt(row.get('test_upper_breach_rate_mean'))} | {fmt(row.get('test_p90_band_width_worst'))} | {fmt(row.get('test_band_width_ic_mean'))} | {fmt(row.get('test_downside_width_ic_mean'))} |"
        )
    lines.append("")
    lines.append("## Replay/Diagnostic vs True WF")
    lines.append("")
    lines.append(
        "- Stage 2 checkpoint 기반 test metric은 diagnostic_only였고 true WF 제품 판단에는 사용하지 않았다."
    )
    lines.append("- Stage 5는 fold마다 해당 train 기간으로 새 checkpoint를 학습했다.")
    STAGE5_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP178 1W band 500 Stage 3~5 runner")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-after-stage3", action="store_true")
    parser.add_argument("--stop-after-stage4", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_all(
        force=args.force,
        stop_after_stage3=args.stop_after_stage3,
        stop_after_stage4=args.stop_after_stage4,
    )
    print(json.dumps(clean_json(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
