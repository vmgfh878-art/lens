from __future__ import annotations

import argparse
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
import ai.cp153_bm_1d_band_500_stage2_5_to_5 as s25  # noqa: E402
from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
)
from ai.ticker_registry import load_registry  # noqa: E402
from ai.train import TrainConfig, run_training  # noqa: E402


TIMEFRAME = "1D"
HORIZON = 5
FEATURE_SET = "price_volatility_volume"
TARGET_TYPE = "raw_future_return"
SOURCE_DATA_HASH = "90666b44cbfb8e5c"
SEEDS = [42, 7, 123]

STAGE3_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage3_calibration_rescue_metrics.json"
STAGE4_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4_seed_stability_metrics.json"
STAGE5_REPLAY_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5_walk_forward_metrics.json"

BASE_LOG_DIR = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4r_5t_logs"
STAGE4R_LOG_DIR = BASE_LOG_DIR / "stage4r"
STAGE5T_LOG_DIR = BASE_LOG_DIR / "stage5t"

STAGE4R_REPORT = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4r_seed_val_test_reassessment_report.md"
STAGE4R_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4r_seed_val_test_reassessment_metrics.json"
STAGE4R_SUMMARY = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage4r_seed_val_test_reassessment_summary.csv"

STAGE5T_REPORT = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5t_true_walk_forward_report.md"
STAGE5T_METRICS = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5t_true_walk_forward_metrics.json"
STAGE5T_SUMMARY = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage5t_true_walk_forward_summary.csv"


@dataclass(frozen=True)
class TargetCandidate:
    candidate_id: str
    model: str
    family: str
    seq_len: int
    q_low: float
    q_high: float
    band_mode: str
    calibration_policy: str
    batch_size: int = 256
    epochs: int = 3
    lower_band_loss_weight: float = 1.0
    fp32_modules: str = "none"

    @property
    def q_label(self) -> str:
        return f"q{int(self.q_low * 100):02d}_q{int(self.q_high * 100):02d}"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


TARGETS = [
    TargetCandidate(
        candidate_id="tide_s60_q15_param",
        model="tide",
        family="tide",
        seq_len=60,
        q_low=0.15,
        q_high=0.85,
        band_mode="param",
        calibration_policy="lower_focused",
    ),
    TargetCandidate(
        candidate_id="tcn_s120_q15_param",
        model="tcn_quantile",
        family="tcn_quantile",
        seq_len=120,
        q_low=0.15,
        q_high=0.85,
        band_mode="param",
        calibration_policy="raw",
    ),
]

FOLDS = [
    FoldSpec("fold_1", "2019-05-01", "2024-05-01", "2024-05-01", "2024-11-01", "2024-11-01", "2025-05-01"),
    FoldSpec("fold_2", "2019-11-01", "2024-11-01", "2024-11-01", "2025-05-01", "2025-05-01", "2025-11-01"),
    FoldSpec("fold_3", "2020-05-01", "2025-05-01", "2025-05-01", "2025-11-01", "2025-11-01", "2026-05-09"),
]

_RAW_BUNDLE_CACHE: dict[tuple[int, bool, str], dict[str, Any]] = {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_json(row.get(key)) for key in fieldnames})


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    return float(np.mean(finite)) if finite else None


def std(values: list[float]) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def metric_worst(values: list[float], higher_is_better: bool = False) -> float | None:
    finite = [float(value) for value in values if safe_float(value) is not None]
    if not finite:
        return None
    return min(finite) if higher_is_better else max(finite)


def target_by_id(candidate_id: str) -> TargetCandidate:
    for candidate in TARGETS:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def ensure_dirs() -> None:
    for path in (BASE_LOG_DIR, STAGE4R_LOG_DIR, STAGE5T_LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


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
    metrics = {key: safe_float(summary.get(key)) for key in s2.BAND_METRIC_KEYS if key in summary}
    date_cs = s2._date_cross_sectional_width_ic(metadata=metadata, lower=lower, upper=upper, actual=actual)
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    return metrics


def apply_calibration(pred: dict[str, Any], calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return s25.apply_calibration(pred, calibration["method"], calibration.get("params") or {})


def calibration_rank(metrics: dict[str, Any]) -> float:
    coverage = safe_float(metrics.get("coverage_abs_error")) or 1.0
    lower_abs = safe_float(metrics.get("lower_breach_abs_error")) or 1.0
    interval = safe_float(metrics.get("asymmetric_interval_score")) or 1.0
    band_ic = safe_float(metrics.get("band_width_ic")) or 0.0
    downside_ic = safe_float(metrics.get("downside_width_ic")) or 0.0
    p90 = safe_float(metrics.get("p90_band_width")) or 1.0
    return -2.0 * coverage - lower_abs - interval + 0.25 * band_ic + 0.25 * downside_ic - 0.05 * p90


def fit_calibration_for_policy(pred: dict[str, Any], candidate: TargetCandidate) -> dict[str, Any]:
    if candidate.calibration_policy == "raw":
        lower, upper = apply_calibration(pred, {"method": "raw", "params": {}})
        metrics = summarize_arrays(lower, upper, pred["actual"], pred["metadata"], candidate.q_low, candidate.q_high)
        return {"method": "raw", "params": {}, "selection_metrics": metrics, "selection_score": calibration_rank(metrics)}
    candidates = [
        item
        for item in s25.fit_calibrations(pred, candidate.q_low, candidate.q_high)
        if item.get("method") == candidate.calibration_policy
    ]
    if not candidates:
        raise ValueError(f"calibration policy를 찾을 수 없습니다: {candidate.calibration_policy}")
    best: dict[str, Any] | None = None
    best_score: float | None = None
    for item in candidates:
        lower, upper = apply_calibration(pred, item)
        metrics = summarize_arrays(lower, upper, pred["actual"], pred["metadata"], candidate.q_low, candidate.q_high)
        score = calibration_rank(metrics)
        row = {"method": item["method"], "params": item.get("params") or {}, "selection_metrics": metrics, "selection_score": score}
        if best is None or best_score is None or score > best_score:
            best = row
            best_score = score
    assert best is not None
    return best


def evaluate_prediction(pred: dict[str, Any], calibration: dict[str, Any], candidate: TargetCandidate) -> dict[str, Any]:
    lower, upper = apply_calibration(pred, calibration)
    metrics = summarize_arrays(lower, upper, pred["actual"], pred["metadata"], candidate.q_low, candidate.q_high)
    regimes = regime_metrics(lower, upper, pred["actual"], pred["metadata"])
    return {"metrics": metrics, "regimes": regimes}


def lower_breach_rate(lower: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.size == 0 or not bool(mask.any()):
        return None
    return float((actual[mask] < lower[mask]).mean())


def regime_metrics(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    del metadata
    row_abs = np.mean(np.abs(actual), axis=1)
    row_mean = np.mean(actual, axis=1)
    row_width = np.mean(np.maximum(upper - lower, 0.0), axis=1)
    vol_cut = float(np.median(row_abs)) if row_abs.size else 0.0
    width_cut = float(np.median(row_width)) if row_width.size else 0.0
    high_vol = row_abs >= vol_cut
    low_vol = row_abs < vol_cut
    falling = row_mean < 0.0
    rising = row_mean >= 0.0
    narrow = row_width < width_cut
    wide = row_width >= width_cut
    narrow_vol = float(np.mean(row_abs[narrow])) if bool(narrow.any()) else None
    wide_vol = float(np.mean(row_abs[wide])) if bool(wide.any()) else None
    return {
        "high_vol_lower_breach_rate": lower_breach_rate(lower, actual, high_vol),
        "low_vol_lower_breach_rate": lower_breach_rate(lower, actual, low_vol),
        "falling_lower_breach_rate": lower_breach_rate(lower, actual, falling),
        "rising_lower_breach_rate": lower_breach_rate(lower, actual, rising),
        "narrow_band_realized_vol": narrow_vol,
        "wide_band_realized_vol": wide_vol,
        "narrow_wide_realized_vol_ratio": (wide_vol / narrow_vol) if narrow_vol and wide_vol is not None else None,
        "narrow_band_count": int(narrow.sum()),
        "wide_band_count": int(wide.sum()),
        "vol_cut": vol_cut,
        "width_cut": width_cut,
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
    gaps: dict[str, Any] = {}
    for key in keys:
        val = safe_float(val_metrics.get(key))
        test = safe_float(test_metrics.get(key))
        gaps[f"{key}_gap_test_minus_val"] = (test - val) if val is not None and test is not None else None
    return gaps


def stage4_process_for(candidate_id: str, seed: int, stage4: dict[str, Any]) -> dict[str, Any]:
    run_key = f"{candidate_id}_seed{seed}"
    process = (stage4.get("processes") or {}).get(run_key)
    if not process:
        raise KeyError(f"Stage 4 process가 없습니다: {run_key}")
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint가 없습니다: {run_key} {checkpoint_path}")
    return process


def run_stage4r() -> dict[str, Any]:
    ensure_dirs()
    started = time.perf_counter()
    price, indicators = s25.load_price_indicator()
    stage4 = read_json(STAGE4_METRICS)
    seed_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for candidate in TARGETS:
        for seed in SEEDS:
            run_key = f"{candidate.candidate_id}_seed{seed}"
            process = stage4_process_for(candidate.candidate_id, seed, stage4)
            val_pred = s25.collect_predictions(process, price, indicators, split="val")
            test_pred = s25.collect_predictions(process, price, indicators, split="test")
            calibration = fit_calibration_for_policy(val_pred, candidate)
            val_eval = evaluate_prediction(val_pred, calibration, candidate)
            test_eval = evaluate_prediction(test_pred, calibration, candidate)
            gaps = metric_gap(val_eval["metrics"], test_eval["metrics"])
            risk_flags = risk_flags_for_test(test_eval["metrics"], gaps)
            calibration_path = STAGE4R_LOG_DIR / run_key / "calibration_params.json"
            write_json(calibration_path, calibration)
            detail[run_key] = {
                "candidate": asdict(candidate),
                "seed": seed,
                "process": process,
                "calibration": calibration,
                "calibration_path": calibration_path,
                "val": val_eval,
                "test": test_eval,
                "gaps": gaps,
                "risk_flags": risk_flags,
                "policy": {
                    "calibration_fit_split": "validation",
                    "test_used_for_calibration": False,
                    "test_used_for_candidate_selection": False,
                },
            }
            seed_rows.append(flatten_stage4r_row(candidate, seed, process, calibration, val_eval, test_eval, gaps, risk_flags))
            print(json.dumps({"stage": "4R", "run_key": run_key, "risk_flags": risk_flags}, ensure_ascii=False, sort_keys=True))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    stability_rows = summarize_stage4r(seed_rows)
    payload = {
        "cp": "CP153-BM",
        "stage": "Stage 4R seed validation/test reassessment",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_data_hash": SOURCE_DATA_HASH,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": FEATURE_SET,
        "target_type": TARGET_TYPE,
        "targets": [asdict(candidate) for candidate in TARGETS],
        "seed_rows": seed_rows,
        "stability_rows": stability_rows,
        "details": detail,
        "promoted_to_stage5t": [candidate.candidate_id for candidate in TARGETS],
        "policy": {
            "existing_stage5_replay_product_evidence": False,
            "calibration_fit_split": "validation",
            "test_fixed_after_validation_calibration": True,
        },
    }
    write_json(STAGE4R_METRICS, payload)
    write_rows(STAGE4R_SUMMARY, seed_rows)
    write_stage4r_report(payload)
    return payload


def risk_flags_for_test(test_metrics: dict[str, Any], gaps: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    cov = safe_float(test_metrics.get("coverage_abs_error"))
    lower = safe_float(test_metrics.get("lower_breach_rate"))
    band_ic = safe_float(test_metrics.get("band_width_ic"))
    down_ic = safe_float(test_metrics.get("downside_width_ic"))
    cov_gap = safe_float(gaps.get("coverage_abs_error_gap_test_minus_val"))
    if cov is not None and cov > 0.10:
        flags.append("test_coverage_collapse")
    if lower is not None and lower > 0.25:
        flags.append("test_lower_breach_collapse")
    if band_ic is not None and band_ic < 0.0:
        flags.append("test_band_width_ic_negative")
    if down_ic is not None and down_ic < -0.05:
        flags.append("test_downside_width_ic_negative")
    if cov_gap is not None and cov_gap > 0.06:
        flags.append("large_validation_test_coverage_gap")
    return flags


def flatten_stage4r_row(
    candidate: TargetCandidate,
    seed: int,
    process: dict[str, Any],
    calibration: dict[str, Any],
    val_eval: dict[str, Any],
    test_eval: dict[str, Any],
    gaps: dict[str, Any],
    risk_flags: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "model": candidate.model,
        "family": candidate.family,
        "seq_len": candidate.seq_len,
        "q_low": candidate.q_low,
        "q_high": candidate.q_high,
        "band_mode": candidate.band_mode,
        "seed": seed,
        "run_id": process.get("run_id"),
        "checkpoint_path": process.get("checkpoint_path"),
        "calibration_method": calibration["method"],
        "calibration_params": json.dumps(clean_json(calibration.get("params") or {}), ensure_ascii=False, sort_keys=True),
        "risk_flags": ",".join(risk_flags),
    }
    for prefix, evaluation in (("val", val_eval), ("test", test_eval)):
        for key, value in evaluation["metrics"].items():
            row[f"{prefix}_{key}"] = value
        for key, value in evaluation["regimes"].items():
            row[f"{prefix}_{key}"] = value
    row.update(gaps)
    return row


def summarize_stage4r(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_pairs = [
        ("val_coverage_abs_error", False),
        ("test_coverage_abs_error", False),
        ("val_lower_breach_rate", False),
        ("test_lower_breach_rate", False),
        ("val_asymmetric_interval_score", False),
        ("test_asymmetric_interval_score", False),
        ("val_band_width_ic", True),
        ("test_band_width_ic", True),
        ("val_downside_width_ic", True),
        ("test_downside_width_ic", True),
        ("val_p90_band_width", False),
        ("test_p90_band_width", False),
    ]
    out: list[dict[str, Any]] = []
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        group = [row for row in rows if row["candidate_id"] == candidate_id]
        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "seed_count": len(group),
            "risk_flags": ",".join(sorted({flag for row in group for flag in str(row.get("risk_flags") or "").split(",") if flag})),
        }
        for key, higher in metric_pairs:
            values = [safe_float(row.get(key)) for row in group]
            values = [value for value in values if value is not None]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = std(values)
            item[f"{key}_worst"] = metric_worst(values, higher_is_better=higher)
        out.append(item)
    return out


def build_raw_fold_bundles(candidate: TargetCandidate, fold: FoldSpec) -> dict[str, Any]:
    cache_key = (candidate.seq_len, candidate.model == "tide", fold.fold_id)
    if cache_key in _RAW_BUNDLE_CACHE:
        return _RAW_BUNDLE_CACHE[cache_key]
    price, indicators = s25.load_price_indicator()
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=candidate.seq_len,
        horizon=HORIZON,
        market_data_provider="yfinance",
        source_data_hash=SOURCE_DATA_HASH,
    )
    registry = load_registry(TIMEFRAME, Path(plan.ticker_registry_path))
    eligible = set(plan.eligible_tickers)
    dataset = build_lazy_sequence_dataset(
        feature_df=indicators[indicators["ticker"].isin(eligible)].copy(),
        price_df=price[price["ticker"].isin(eligible)].copy(),
        timeframe=TIMEFRAME,
        seq_len=candidate.seq_len,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=candidate.model == "tide",
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train_idx, val_idx, test_idx, split_diag = date_split_indices(dataset, fold, plan.h_max)
    train_raw = dataset.subset(train_idx)
    val_raw = dataset.subset(val_idx)
    test_raw = dataset.subset(test_idx)
    train_norm, val_norm, test_norm, mean_t, std_t = normalize_sequence_splits(train_raw, val_raw, test_raw)
    payload = {
        "plan": plan,
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
    _RAW_BUNDLE_CACHE[cache_key] = payload
    return payload


def date_split_indices(dataset: Any, fold: FoldSpec, h_max: int) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
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
    ticker_diag: dict[str, Any] = {}
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
            ticker_diag[str(ticker)] = {
                "train": len(ticker_train),
                "val": len(ticker_val),
                "test": len(ticker_test),
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
                    "empty_ticker_count": len(ticker_diag),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    split_sets = {
        "train": set(train_indices),
        "val": set(val_indices),
        "test": set(test_indices),
    }
    overlap = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(json.dumps({"error": "custom_split_overlap", "fold": asdict(fold), "overlap": overlap}, ensure_ascii=False, sort_keys=True))
    return train_indices, val_indices, test_indices, {
        "fold": asdict(fold),
        "h_max_gap": int(h_max),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "overlap": overlap,
        "excluded_tickers_for_fold_count": len(ticker_diag),
        "excluded_tickers_for_fold_sample_counts": ticker_diag,
    }


def make_train_config(candidate: TargetCandidate, seed: int) -> TrainConfig:
    return TrainConfig(
        model=candidate.model,
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        seq_len=candidate.seq_len,
        epochs=candidate.epochs,
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
        patch_len=16,
        patch_stride=8,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=FEATURE_SET,
        market_data_provider="yfinance",
        lower_band_loss_weight=candidate.lower_band_loss_weight,
        upper_band_loss_weight=1.0,
        model_role="band",
        lambda_risk=0.5,
        risk_decision_threshold=0.5,
    )


def run_fold_training(candidate: TargetCandidate, fold: FoldSpec, seed: int, force: bool) -> dict[str, Any]:
    run_dir = STAGE5T_LOG_DIR / f"{candidate.candidate_id}_{fold.fold_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    process_path = run_dir / "train_process.json"
    if not force and process_path.exists():
        existing = read_json(process_path)
        checkpoint_path = existing.get("checkpoint_path")
        if existing.get("status") == "PASS" and checkpoint_path and Path(checkpoint_path).exists():
            return existing
    bundles = build_raw_fold_bundles(candidate, fold)
    config = make_train_config(candidate, seed)
    started = time.perf_counter()
    start_utc = now_utc()
    print(json.dumps({"stage": "5T", "event": "train_start", "candidate": candidate.candidate_id, "fold": fold.fold_id, "seed": seed}, ensure_ascii=False, sort_keys=True))
    try:
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
    except Exception as exc:
        result = {"error": repr(exc)}
        status = "FAIL"
        failure_reason = repr(exc)
        failure_category = "contract" if "split" in repr(exc).lower() or "finite" in repr(exc).lower() else "runtime"
    elapsed = time.perf_counter() - started
    process = {
        "candidate": asdict(candidate),
        "fold": asdict(fold),
        "seed": seed,
        "status": status,
        "failure_reason": failure_reason,
        "failure_category": failure_category,
        "start_time_utc": start_utc,
        "end_time_utc": now_utc(),
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": 0 if status == "PASS" else 1,
        "run_id": result.get("run_id"),
        "checkpoint_path": result.get("checkpoint_path"),
        "local_log_dir": result.get("local_log_dir"),
        "best_metrics": result.get("best_metrics"),
        "test_metrics_from_train_readonly": result.get("test_metrics"),
        "dataset_plan": result.get("dataset_plan"),
        "split_diag": bundles["split_diag"],
        "feature_set": FEATURE_SET,
        "feature_columns": result.get("feature_columns"),
        "n_features": result.get("n_features"),
        "save_run": False,
        "wandb": False,
        "wandb_status": result.get("wandb_status"),
        "train_process_path": str(process_path),
    }
    write_json(process_path, process)
    print(json.dumps({"stage": "5T", "event": "train_end", "candidate": candidate.candidate_id, "fold": fold.fold_id, "seed": seed, "status": status, "elapsed_seconds": round(elapsed, 3)}, ensure_ascii=False, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return process


def collect_predictions_for_bundle(checkpoint_path: str | Path, bundle: Any) -> dict[str, Any]:
    model, config, feature_mean, feature_std = s2.load_model_from_checkpoint(checkpoint_path)
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
    }


def evaluate_fold_checkpoint(candidate: TargetCandidate, fold: FoldSpec, seed: int, process: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {"status": "FAIL", "failure_reason": "checkpoint_missing"}
    bundles = build_raw_fold_bundles(candidate, fold)
    val_pred = collect_predictions_for_bundle(checkpoint_path, bundles["val_raw"])
    test_pred = collect_predictions_for_bundle(checkpoint_path, bundles["test_raw"])
    calibration = fit_calibration_for_policy(val_pred, candidate)
    val_eval = evaluate_prediction(val_pred, calibration, candidate)
    test_eval = evaluate_prediction(test_pred, calibration, candidate)
    gaps = metric_gap(val_eval["metrics"], test_eval["metrics"])
    calibration_path = STAGE5T_LOG_DIR / f"{candidate.candidate_id}_{fold.fold_id}_seed{seed}" / "calibration_params.json"
    fold_metrics_path = STAGE5T_LOG_DIR / f"{candidate.candidate_id}_{fold.fold_id}_seed{seed}" / "fold_metrics.json"
    payload = {
        "status": "PASS",
        "candidate": asdict(candidate),
        "fold": asdict(fold),
        "seed": seed,
        "calibration": calibration,
        "calibration_path": calibration_path,
        "val": val_eval,
        "test": test_eval,
        "gaps": gaps,
        "risk_flags": risk_flags_for_test(test_eval["metrics"], gaps),
        "split_diag": bundles["split_diag"],
        "policy": {
            "fold_checkpoint_retrained": True,
            "calibration_fit_split": "validation",
            "test_used_for_calibration": False,
            "test_used_for_candidate_selection": False,
        },
    }
    write_json(calibration_path, calibration)
    write_json(fold_metrics_path, payload)
    return payload


def flatten_stage5t_row(candidate: TargetCandidate, fold: FoldSpec, seed: int, process: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "model": candidate.model,
        "family": candidate.family,
        "seq_len": candidate.seq_len,
        "q_low": candidate.q_low,
        "q_high": candidate.q_high,
        "band_mode": candidate.band_mode,
        "calibration_policy": candidate.calibration_policy,
        "fold_id": fold.fold_id,
        "seed": seed,
        "status": process.get("status"),
        "exit_code": process.get("exit_code"),
        "elapsed_seconds": process.get("elapsed_seconds"),
        "run_id": process.get("run_id"),
        "checkpoint_path": process.get("checkpoint_path"),
        "train_process_path": process.get("train_process_path"),
        "failure_reason": process.get("failure_reason"),
        "failure_category": process.get("failure_category"),
    }
    split_diag = evaluation.get("split_diag") or process.get("split_diag") or {}
    row["train_samples"] = split_diag.get("train_samples")
    row["val_samples"] = split_diag.get("val_samples")
    row["test_samples"] = split_diag.get("test_samples")
    row["split_overlap_train_val"] = (split_diag.get("overlap") or {}).get("train_val")
    row["split_overlap_train_test"] = (split_diag.get("overlap") or {}).get("train_test")
    row["split_overlap_val_test"] = (split_diag.get("overlap") or {}).get("val_test")
    if evaluation.get("status") == "PASS":
        calibration = evaluation["calibration"]
        row["calibration_method"] = calibration["method"]
        row["calibration_params"] = json.dumps(clean_json(calibration.get("params") or {}), ensure_ascii=False, sort_keys=True)
        row["risk_flags"] = ",".join(evaluation.get("risk_flags") or [])
        for prefix, item in (("val", evaluation["val"]), ("test", evaluation["test"])):
            for key, value in item["metrics"].items():
                row[f"{prefix}_{key}"] = value
            for key, value in item["regimes"].items():
                row[f"{prefix}_{key}"] = value
        row.update(evaluation.get("gaps") or {})
    else:
        row["risk_flags"] = evaluation.get("failure_reason")
    return row


def aggregate_stage5t(rows: list[dict[str, Any]], seeds: list[int] | None = None) -> list[dict[str, Any]]:
    metric_pairs = [
        ("val_coverage_abs_error", False),
        ("test_coverage_abs_error", False),
        ("val_lower_breach_rate", False),
        ("test_lower_breach_rate", False),
        ("val_upper_breach_rate", False),
        ("test_upper_breach_rate", False),
        ("val_asymmetric_interval_score", False),
        ("test_asymmetric_interval_score", False),
        ("val_band_width_ic", True),
        ("test_band_width_ic", True),
        ("val_downside_width_ic", True),
        ("test_downside_width_ic", True),
        ("val_p90_band_width", False),
        ("test_p90_band_width", False),
        ("test_falling_lower_breach_rate", False),
        ("test_high_vol_lower_breach_rate", False),
    ]
    filtered = [row for row in rows if seeds is None or int(row["seed"]) in set(seeds)]
    out: list[dict[str, Any]] = []
    for candidate_id in sorted({row["candidate_id"] for row in filtered}):
        group = [row for row in filtered if row["candidate_id"] == candidate_id and row.get("status") == "PASS"]
        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "run_count": len(group),
            "seed_list": ",".join(str(seed) for seed in sorted({int(row["seed"]) for row in group})),
            "fold_list": ",".join(sorted({str(row["fold_id"]) for row in group})),
            "risk_flags": ",".join(sorted({flag for row in group for flag in str(row.get("risk_flags") or "").split(",") if flag})),
            "failed_run_count": len([row for row in filtered if row["candidate_id"] == candidate_id and row.get("status") != "PASS"]),
        }
        for key, higher in metric_pairs:
            values = [safe_float(row.get(key)) for row in group]
            values = [value for value in values if value is not None]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = std(values)
            item[f"{key}_worst"] = metric_worst(values, higher_is_better=higher)
        item["decision"] = classify_true_wf_candidate(item)
        out.append(item)
    return sorted(out, key=lambda row: safe_float(row.get("test_asymmetric_interval_score_mean")) or 999.0)


def classify_true_wf_candidate(row: dict[str, Any]) -> str:
    cov = safe_float(row.get("val_coverage_abs_error_mean"))
    lower = safe_float(row.get("val_lower_breach_rate_mean"))
    band_ic = safe_float(row.get("val_band_width_ic_mean"))
    down_ic = safe_float(row.get("val_downside_width_ic_mean"))
    p90 = safe_float(row.get("val_p90_band_width_worst"))
    failed = int(row.get("failed_run_count") or 0)
    if failed:
        return "research_reserve_runtime_risk"
    if cov is not None and lower is not None and band_ic is not None and down_ic is not None:
        if cov <= 0.05 and lower <= 0.18 and band_ic > 0.15 and down_ic >= 0.0 and (p90 is None or p90 <= 0.13):
            return "product_candidate_raw_true_wf"
        if cov <= 0.09 and lower <= 0.24 and band_ic > 0.10 and down_ic >= -0.05:
            return "research_reserve_true_wf"
    return "rejected_true_wf"


def has_research_or_product_possibility(seed42_aggregate: list[dict[str, Any]]) -> bool:
    return any(row.get("decision") in {"product_candidate_raw_true_wf", "research_reserve_true_wf"} for row in seed42_aggregate)


def run_stage5t(force: bool = False, seed_expansion: str = "auto") -> dict[str, Any]:
    ensure_dirs()
    started = time.perf_counter()
    stage4r = read_json(STAGE4R_METRICS) if STAGE4R_METRICS.exists() else None
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    completed_seeds: list[int] = []
    for seed in [42]:
        completed_seeds.append(seed)
        run_stage5t_seed(seed, rows, details, force=force)
    seed42_aggregate = aggregate_stage5t(rows, seeds=[42])
    expand = seed_expansion == "all" or (seed_expansion == "auto" and has_research_or_product_possibility(seed42_aggregate))
    stopped_reason = None
    if expand:
        for seed in [7, 123]:
            completed_seeds.append(seed)
            run_stage5t_seed(seed, rows, details, force=force)
    else:
        stopped_reason = "seed42에서 두 후보 모두 명확한 true WF 연구/제품 가능성을 보이지 않아 seed 7/123 확장 전 중단"
    aggregate_rows = aggregate_stage5t(rows)
    payload = {
        "cp": "CP153-BM",
        "stage": "Stage 5T true walk-forward fold retraining",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_data_hash": SOURCE_DATA_HASH,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": FEATURE_SET,
        "target_type": TARGET_TYPE,
        "targets": [asdict(candidate) for candidate in TARGETS],
        "folds": [asdict(fold) for fold in FOLDS],
        "completed_seeds": completed_seeds,
        "seed_expansion": {
            "mode": seed_expansion,
            "expanded_after_seed42": expand,
            "stopped_reason": stopped_reason,
            "seed42_aggregate": seed42_aggregate,
        },
        "stage4r_risk_flags": stage4r.get("stability_rows") if stage4r else None,
        "fold_rows": rows,
        "aggregate_rows": aggregate_rows,
        "details": details,
        "policy": {
            "true_fold_retraining": True,
            "checkpoint_replay_product_evidence": False,
            "checkpoint_selection_split": "fold_validation",
            "calibration_fit_split": "fold_validation",
            "test_used_for_calibration": False,
            "test_used_for_candidate_selection": False,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": False,
        },
        "replay_comparison": replay_comparison(),
    }
    write_json(STAGE5T_METRICS, payload)
    write_rows(STAGE5T_SUMMARY, rows)
    write_stage5t_report(payload)
    return payload


def run_stage5t_seed(seed: int, rows: list[dict[str, Any]], details: dict[str, Any], *, force: bool) -> None:
    for candidate in TARGETS:
        for fold in FOLDS:
            key = f"{candidate.candidate_id}_{fold.fold_id}_seed{seed}"
            process = run_fold_training(candidate, fold, seed, force)
            if process.get("status") == "PASS":
                evaluation = evaluate_fold_checkpoint(candidate, fold, seed, process)
            else:
                evaluation = {"status": "FAIL", "failure_reason": process.get("failure_reason"), "split_diag": process.get("split_diag")}
            details[key] = {"process": process, "evaluation": evaluation}
            rows.append(flatten_stage5t_row(candidate, fold, seed, process, evaluation))
            write_rows(STAGE5T_SUMMARY, rows)
            partial_payload = {
                "created_at_utc": now_utc(),
                "partial": True,
                "fold_rows": rows,
                "aggregate_rows": aggregate_stage5t(rows),
            }
            write_json(STAGE5T_METRICS, partial_payload)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def replay_comparison() -> dict[str, Any]:
    if not STAGE5_REPLAY_METRICS.exists():
        return {"available": False}
    replay = read_json(STAGE5_REPLAY_METRICS)
    return {
        "available": True,
        "source_path": STAGE5_REPLAY_METRICS,
        "status": "diagnostic_only_not_product_evidence",
        "stage": replay.get("stage"),
        "summary_rows": replay.get("fold_summary_rows") or replay.get("summary_rows") or replay.get("fold_rows"),
    }


def write_stage4r_report(payload: dict[str, Any]) -> None:
    lines = [
        "# CP153-BM 1D Band 500 Stage 4R Seed Validation/Test Reassessment",
        "",
        "## 판정 범위",
        "",
        "- 기존 Stage 5 checkpoint replay는 제품 후보 확정 근거로 사용하지 않는다.",
        "- Stage 4R은 seed별 validation/test 안정성 재평가이며, 두 후보 모두 Stage 5T로 보낸다.",
        "- calibration은 validation에서만 fit했고 test에는 고정 적용했다.",
        "",
        "## 요약",
        "",
        "| candidate | seeds | val_cov_mean | test_cov_mean | val_lower_mean | test_lower_mean | val_band_ic_mean | test_band_ic_mean | risk_flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["stability_rows"]:
        lines.append(
            "| {candidate_id} | {seed_count} | {val_cov:.6f} | {test_cov:.6f} | {val_lower:.6f} | {test_lower:.6f} | {val_ic:.6f} | {test_ic:.6f} | {risk} |".format(
                candidate_id=row["candidate_id"],
                seed_count=row["seed_count"],
                val_cov=safe_float(row.get("val_coverage_abs_error_mean")) or float("nan"),
                test_cov=safe_float(row.get("test_coverage_abs_error_mean")) or float("nan"),
                val_lower=safe_float(row.get("val_lower_breach_rate_mean")) or float("nan"),
                test_lower=safe_float(row.get("test_lower_breach_rate_mean")) or float("nan"),
                val_ic=safe_float(row.get("val_band_width_ic_mean")) or float("nan"),
                test_ic=safe_float(row.get("test_band_width_ic_mean")) or float("nan"),
                risk=row.get("risk_flags") or "",
            )
        )
    lines.extend(
        [
            "",
            "## 산출물",
            "",
            f"- metrics: `{STAGE4R_METRICS}`",
            f"- summary: `{STAGE4R_SUMMARY}`",
            f"- calibration params: `{STAGE4R_LOG_DIR}`",
        ]
    )
    STAGE4R_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage5t_report(payload: dict[str, Any]) -> None:
    lines = [
        "# CP153-BM 1D Band 500 Stage 5T True Walk-Forward Report",
        "",
        "## 핵심 원칙",
        "",
        "- 이 보고서는 fold별 새 checkpoint 학습 결과만 제품 후보 판단 근거로 둔다.",
        "- 기존 Stage 5 replay는 참고 진단으로만 남기며 true walk-forward로 표현하지 않는다.",
        "- 각 fold calibration은 해당 fold validation에서만 fit했고 test에는 고정 적용했다.",
        "- save-run, DB write, inference 저장, W&B, live fetch, EODHD fallback, composite는 사용하지 않았다.",
        "",
        "## True WF 요약",
        "",
        "| candidate | runs | decision | val_cov_mean | test_cov_mean | val_lower_mean | test_lower_mean | val_band_ic_mean | test_band_ic_mean | val_downside_ic_mean | test_downside_ic_mean | risk_flags |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            "| {candidate_id} | {runs} | {decision} | {val_cov:.6f} | {test_cov:.6f} | {val_lower:.6f} | {test_lower:.6f} | {val_ic:.6f} | {test_ic:.6f} | {val_down:.6f} | {test_down:.6f} | {risk} |".format(
                candidate_id=row["candidate_id"],
                runs=row["run_count"],
                decision=row["decision"],
                val_cov=safe_float(row.get("val_coverage_abs_error_mean")) or float("nan"),
                test_cov=safe_float(row.get("test_coverage_abs_error_mean")) or float("nan"),
                val_lower=safe_float(row.get("val_lower_breach_rate_mean")) or float("nan"),
                test_lower=safe_float(row.get("test_lower_breach_rate_mean")) or float("nan"),
                val_ic=safe_float(row.get("val_band_width_ic_mean")) or float("nan"),
                test_ic=safe_float(row.get("test_band_width_ic_mean")) or float("nan"),
                val_down=safe_float(row.get("val_downside_width_ic_mean")) or float("nan"),
                test_down=safe_float(row.get("test_downside_width_ic_mean")) or float("nan"),
                risk=row.get("risk_flags") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Replay 비교",
            "",
            "- 기존 Stage 5 replay는 checkpoint replay였고 제품 후보 확정 근거가 아니다.",
            "- Stage 5T는 fold마다 train 기간으로 새 checkpoint를 학습했다.",
            "",
            "## 산출물",
            "",
            f"- metrics: `{STAGE5T_METRICS}`",
            f"- summary: `{STAGE5T_SUMMARY}`",
            f"- fold logs: `{STAGE5T_LOG_DIR}`",
        ]
    )
    STAGE5T_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP153-BM Stage 4R/5T runner")
    parser.add_argument("--stage", choices=["stage4r", "stage5t", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-expansion", choices=["auto", "all", "none"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    print(
        json.dumps(
            {
                "event": "start",
                "stage": args.stage,
                "python": sys.executable,
                "torch": torch.__version__,
                "cuda": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "source_data_hash": SOURCE_DATA_HASH,
                "feature_set": FEATURE_SET,
                "feature_columns": s2.resolve_feature_columns(FEATURE_SET),
                "estimated_hours_seed42_only": "2~4",
                "estimated_hours_all_seeds": "6~10",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if args.stage in {"stage4r", "all"}:
        run_stage4r()
    if args.stage in {"stage5t", "all"}:
        run_stage5t(force=args.force, seed_expansion=args.seed_expansion)


if __name__ == "__main__":
    main()
