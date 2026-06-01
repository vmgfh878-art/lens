from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
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
import ai.cp153_bm_1d_band_500_stage4r_5t as s45  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    build_calendar_feature_frame,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
    split_sequence_dataset_by_plan,
)
from ai.storage import STORAGE_CONTRACT_PRODUCT_LATEST_ONLY, with_prediction_storage_contract  # noqa: E402
from ai.ticker_registry import load_registry  # noqa: E402
from ai.train import TrainConfig, run_training  # noqa: E402


TIMEFRAME = "1D"
HORIZON = 5
FEATURE_SET = "price_volatility_volume"
TARGET_TYPE = "raw_future_return"
SOURCE_DATA_HASH = "90666b44cbfb8e5c"
MODEL_NAME = "tide_s60_q15_param_lower_focused"
BASE_LOG_DIR = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_save_run_logs"
REPORT_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_save_run_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_save_run_metrics.json"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_save_run_summary.csv"
RUN_META_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_run_meta.json"
CALIBRATION_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_calibration_params.json"
LATEST_ARTIFACT_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_latest_predictions.json"
LATEST_JSONL_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_primary_product_candidate_latest_predictions.jsonl"


STAGE5T_REFERENCE = {
    "test_coverage_abs_error_mean": 0.025440,
    "test_lower_breach_rate_mean": 0.142540,
    "test_upper_breach_rate_mean": 0.156973,
    "test_band_width_ic_mean": 0.373995,
    "test_downside_width_ic_mean": 0.086193,
    "risk_flags": "",
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str = "tide_s60_q15_param"
    model: str = "tide"
    family: str = "tide"
    seq_len: int = 60
    q_low: float = 0.15
    q_high: float = 0.85
    band_mode: str = "param"
    calibration_policy: str = "lower_focused"
    epochs: int = 3
    batch_size: int = 256
    seed: int = 42


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(clean_json(row), ensure_ascii=False, sort_keys=True) + "\n")


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


def build_precomputed_default_splits(spec: CandidateSpec) -> dict[str, Any]:
    price, indicators = s25.load_price_indicator()
    plan = build_dataset_plan(
        indicators,
        timeframe=TIMEFRAME,
        seq_len=spec.seq_len,
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
        seq_len=spec.seq_len,
        horizon=HORIZON,
        ticker_registry=registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train_raw, val_raw, test_raw = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    train_norm, val_norm, test_norm, mean_t, std_t = normalize_sequence_splits(train_raw, val_raw, test_raw)
    return {
        "price": price,
        "indicators": indicators,
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
    }


def make_config(spec: CandidateSpec) -> TrainConfig:
    return TrainConfig(
        model=spec.model,
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        seq_len=spec.seq_len,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        lr=1e-4,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=spec.q_low,
        q_high=spec.q_high,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.2,
        band_mode=spec.band_mode,
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=spec.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-ai",
        model_ver="cp153-bm-1d-band-v1-primary-local",
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
        lower_band_loss_weight=1.0,
        upper_band_loss_weight=1.0,
        model_role="band",
        lambda_risk=0.5,
        risk_decision_threshold=0.5,
    )


def train_local_candidate(spec: CandidateSpec, bundles: dict[str, Any], force: bool) -> dict[str, Any]:
    process_path = BASE_LOG_DIR / "train_process.json"
    if not force and process_path.exists():
        existing = json.loads(process_path.read_text(encoding="utf-8"))
        checkpoint_path = existing.get("checkpoint_path")
        if existing.get("status") == "PASS" and checkpoint_path and Path(checkpoint_path).exists():
            return existing
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = make_config(spec)
    start = time.perf_counter()
    start_utc = now_utc()
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
            local_log_dir=BASE_LOG_DIR / "ai_train_local_logs",
        )
        process = {
            "status": "PASS",
            "exit_code": 0,
            "start_time_utc": start_utc,
            "end_time_utc": now_utc(),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "candidate": asdict(spec),
            "run_id": result.get("run_id"),
            "checkpoint_path": result.get("checkpoint_path"),
            "local_log_dir": result.get("local_log_dir"),
            "best_metrics": result.get("best_metrics"),
            "test_metrics_from_train_readonly": result.get("test_metrics"),
            "dataset_plan": result.get("dataset_plan"),
            "feature_columns": result.get("feature_columns"),
            "n_features": result.get("n_features"),
            "wandb_status": result.get("wandb_status"),
            "save_run": False,
            "db_write": False,
        }
    except Exception as exc:
        process = {
            "status": "FAIL",
            "exit_code": 1,
            "start_time_utc": start_utc,
            "end_time_utc": now_utc(),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "candidate": asdict(spec),
            "failure_reason": repr(exc),
            "save_run": False,
            "db_write": False,
        }
    write_json(process_path, process)
    return process


def fit_and_evaluate_calibration(spec: CandidateSpec, process: dict[str, Any], bundles: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = process.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path가 없습니다.")
    val_pred = s45.collect_predictions_for_bundle(checkpoint_path, bundles["val_raw"])
    test_pred = s45.collect_predictions_for_bundle(checkpoint_path, bundles["test_raw"])
    target = s45.TargetCandidate(
        candidate_id=spec.candidate_id,
        model=spec.model,
        family=spec.family,
        seq_len=spec.seq_len,
        q_low=spec.q_low,
        q_high=spec.q_high,
        band_mode=spec.band_mode,
        calibration_policy=spec.calibration_policy,
    )
    calibration = s45.fit_calibration_for_policy(val_pred, target)
    val_eval = s45.evaluate_prediction(val_pred, calibration, target)
    test_eval = s45.evaluate_prediction(test_pred, calibration, target)
    write_json(CALIBRATION_PATH, calibration)
    return {
        "calibration": calibration,
        "val": val_eval,
        "test": test_eval,
        "stage5t_reference": STAGE5T_REFERENCE,
        "comparison": compare_to_stage5t(test_eval["metrics"]),
    }


def compare_to_stage5t(test_metrics: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "coverage_abs_error": "test_coverage_abs_error_mean",
        "lower_breach_rate": "test_lower_breach_rate_mean",
        "upper_breach_rate": "test_upper_breach_rate_mean",
        "band_width_ic": "test_band_width_ic_mean",
        "downside_width_ic": "test_downside_width_ic_mean",
    }
    comparison: dict[str, Any] = {}
    for metric_key, ref_key in keys.items():
        value = s45.safe_float(test_metrics.get(metric_key))
        ref = float(STAGE5T_REFERENCE[ref_key])
        comparison[f"{metric_key}_save_run"] = value
        comparison[f"{metric_key}_stage5t_ref"] = ref
        comparison[f"{metric_key}_delta"] = (value - ref) if value is not None else None
    return comparison


def product_gate(test_metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    coverage = s45.safe_float(test_metrics.get("coverage_abs_error"))
    lower = s45.safe_float(test_metrics.get("lower_breach_rate"))
    band_ic = s45.safe_float(test_metrics.get("band_width_ic"))
    downside_ic = s45.safe_float(test_metrics.get("downside_width_ic"))
    if coverage is None or coverage > STAGE5T_REFERENCE["test_coverage_abs_error_mean"] + 0.05:
        reasons.append("coverage_abs_error_collapse")
    if lower is None or lower > 0.20:
        reasons.append("lower_breach_rate_collapse")
    if band_ic is None or band_ic < 0.15:
        reasons.append("band_width_ic_collapse")
    if downside_ic is None or downside_ic < 0.0:
        reasons.append("downside_width_ic_negative")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "thresholds": {
            "coverage_abs_error_max": STAGE5T_REFERENCE["test_coverage_abs_error_mean"] + 0.05,
            "lower_breach_rate_max": 0.20,
            "band_width_ic_min": 0.15,
            "downside_width_ic_min": 0.0,
        },
    }


def latest_forecast_dates(asof_date: str, horizon: int) -> list[str]:
    start = pd.Timestamp(asof_date) + pd.offsets.BDay(1)
    return [str(ts.date()) for ts in pd.bdate_range(start=start, periods=horizon)]


def collect_latest_product_predictions(
    spec: CandidateSpec,
    process: dict[str, Any],
    bundles: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = process["checkpoint_path"]
    model, config, feature_mean, feature_std = s2.load_model_from_checkpoint(checkpoint_path)
    feature_columns = list(config.get("feature_columns") or s2.resolve_feature_columns(FEATURE_SET))
    indices = [MODEL_FEATURE_COLUMNS.index(column) for column in feature_columns]
    dataset = bundles["dataset"]
    latest_rows: list[dict[str, Any]] = []
    for ticker, arrays in sorted(dataset.ticker_arrays.items()):
        features_full = np.asarray(arrays["features"], dtype=np.float32)
        closes = np.asarray(arrays["closes"], dtype=np.float32)
        dates = pd.to_datetime(arrays["dates"])
        if len(features_full) < spec.seq_len:
            continue
        end_idx = len(features_full) - 1
        if float(closes[end_idx]) == 0.0:
            continue
        latest_rows.append(
            {
                "ticker": ticker,
                "end_idx": end_idx,
                "asof_date": str(pd.Timestamp(dates[end_idx]).date()),
                "anchor_close": float(closes[end_idx]),
                "ticker_id": int(arrays["ticker_id"]),
                "features": features_full[end_idx - spec.seq_len + 1 : end_idx + 1, :][:, indices],
            }
        )
    if not latest_rows:
        raise ValueError("latest prediction을 만들 수 있는 ticker가 없습니다.")
    asof_counts = pd.Series([row["asof_date"] for row in latest_rows]).value_counts().sort_index()
    selected_asof = str(asof_counts.index[-1])
    selected = [row for row in latest_rows if row["asof_date"] == selected_asof]
    future_dates = latest_forecast_dates(selected_asof, spec.horizon if hasattr(spec, "horizon") else HORIZON)
    future_calendar = build_calendar_feature_frame(pd.to_datetime(future_dates)).to_numpy(dtype=np.float32)
    features = torch.from_numpy(np.stack([row["features"] for row in selected]).astype(np.float32))
    features = (features - feature_mean.to(torch.float32).view(1, 1, -1)) / feature_std.to(torch.float32).view(1, 1, -1)
    ticker_ids = torch.tensor([row["ticker_id"] for row in selected], dtype=torch.long)
    future_covariates = torch.from_numpy(np.stack([future_calendar for _ in selected]).astype(np.float32))
    device = s2.resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(selected), 512):
            end = min(start + 512, len(selected))
            batch_features = features[start:end].to(device)
            batch_ticker_ids = ticker_ids[start:end].to(device)
            batch_covariates = future_covariates[start:end].to(device)
            with s2.autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = s2.forward_model(model, batch_features, batch_ticker_ids, batch_covariates)
            if isinstance(output, ForecastOutput):
                _, lower_t, upper_t = apply_band_postprocess(
                    output.line.detach().cpu().to(torch.float32),
                    output.lower_band.detach().cpu().to(torch.float32),
                    output.upper_band.detach().cpu().to(torch.float32),
                )
            elif isinstance(output, BandOutput):
                raw_lower = output.lower_band.detach().cpu().to(torch.float32)
                raw_upper = output.upper_band.detach().cpu().to(torch.float32)
                lower_t = torch.minimum(raw_lower, raw_upper)
                upper_t = torch.maximum(raw_lower, raw_upper)
            else:
                raise TypeError(f"지원하지 않는 출력입니다: {type(output).__name__}")
            pred = {
                "lower": lower_t.numpy(),
                "upper": upper_t.numpy(),
                "actual": np.zeros_like(lower_t.numpy()),
            }
            lower_np, upper_np = s25.apply_calibration(pred, calibration["method"], calibration.get("params") or {})
            for offset, (lower_returns, upper_returns) in enumerate(zip(lower_np, upper_np, strict=False)):
                row = selected[start + offset]
                anchor = float(row["anchor_close"])
                lower_prices = (anchor * (1.0 + lower_returns)).astype(float).tolist()
                upper_prices = (anchor * (1.0 + upper_returns)).astype(float).tolist()
                records.append(
                    {
                        "ticker": row["ticker"],
                        "model_name": MODEL_NAME,
                        "timeframe": TIMEFRAME,
                        "horizon": HORIZON,
                        "asof_date": selected_asof,
                        "decision_time": now_utc(),
                        "run_id": process["run_id"],
                        "model_ver": "cp153-bm-1d-band-v1-primary-local",
                        "signal": "HOLD",
                        "forecast_dates": future_dates,
                        "line_series": [],
                        "conservative_series": [],
                        "lower_band_series": lower_prices,
                        "upper_band_series": upper_prices,
                        "band_quantile_low": spec.q_low,
                        "band_quantile_high": spec.q_high,
                        "meta": {
                            "layer": "band",
                            "model_role": "band",
                            "output_role": "band",
                            "storage_contract": STORAGE_CONTRACT_PRODUCT_LATEST_ONLY,
                            "composite": False,
                            "calibration_method": calibration["method"],
                            "calibration_params": calibration.get("params") or {},
                            "source_data_hash": SOURCE_DATA_HASH,
                            "provider": "yfinance",
                            "source": "yfinance",
                            "feature_set": FEATURE_SET,
                        },
                    }
                )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "asof_date": selected_asof,
        "eligible_latest_count": len(latest_rows),
        "prediction_count": len(records),
        "asof_counts": {str(key): int(value) for key, value in asof_counts.items()},
        "records": records,
    }


def validate_local_product_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if not records:
        errors.append("empty_records")
    asof_dates = sorted({str(record.get("asof_date")) for record in records})
    if len(asof_dates) != 1:
        errors.append(f"multiple_asof_dates={asof_dates}")
    duplicate_keys: set[tuple[Any, ...]] = set()
    forbidden_top_level = {"line_score", "conservative_prediction", "composite_payload"}
    for record in records:
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        if meta.get("layer") != "band":
            errors.append(f"invalid_layer:{record.get('ticker')}")
        if meta.get("model_role") != "band" or meta.get("output_role") != "band":
            errors.append(f"invalid_role:{record.get('ticker')}")
        if bool(meta.get("composite")):
            errors.append(f"composite_meta:{record.get('ticker')}")
        for key in forbidden_top_level:
            if key in record:
                errors.append(f"forbidden_key:{key}:{record.get('ticker')}")
        if record.get("line_series"):
            errors.append(f"line_series_not_empty:{record.get('ticker')}")
        if record.get("conservative_series"):
            errors.append(f"conservative_series_not_empty:{record.get('ticker')}")
        if not record.get("lower_band_series") or not record.get("upper_band_series"):
            errors.append(f"band_series_empty:{record.get('ticker')}")
        if len(record.get("lower_band_series") or []) != HORIZON or len(record.get("upper_band_series") or []) != HORIZON:
            errors.append(f"band_series_horizon_mismatch:{record.get('ticker')}")
        key = (
            record.get("run_id"),
            record.get("ticker"),
            record.get("model_name"),
            record.get("timeframe"),
            record.get("horizon"),
            record.get("asof_date"),
        )
        if key in duplicate_keys:
            errors.append(f"duplicate_key:{key}")
        duplicate_keys.add(key)
    annotated = with_prediction_storage_contract(records, STORAGE_CONTRACT_PRODUCT_LATEST_ONLY)
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors[:50],
        "record_count": len(records),
        "single_asof_date": asof_dates[0] if len(asof_dates) == 1 else None,
        "storage_contract": STORAGE_CONTRACT_PRODUCT_LATEST_ONLY,
        "db_write_performed": False,
        "annotated_sample_has_storage_contract": bool(annotated and annotated[0].get("meta", {}).get("storage_contract") == STORAGE_CONTRACT_PRODUCT_LATEST_ONLY),
        "line_payload_absent": not any(record.get("line_series") for record in records),
        "composite_payload_absent": not any(bool((record.get("meta") or {}).get("composite")) for record in records),
    }


def build_run_meta(
    spec: CandidateSpec,
    process: dict[str, Any],
    bundles: dict[str, Any],
    calibration_eval: dict[str, Any],
    product_gate_result: dict[str, Any],
    latest_info: dict[str, Any] | None,
    contract_result: dict[str, Any] | None,
) -> dict[str, Any]:
    plan = bundles["plan"]
    return {
        "run_id": process.get("run_id"),
        "checkpoint_path": process.get("checkpoint_path"),
        "local_save_run": True,
        "db_attach": False,
        "db_attach_not_before": "2026-05-16",
        "requires_user_approval_for_db_attach": True,
        "role": "band_model",
        "model_role": "band",
        "output_role": "band",
        "model_name": MODEL_NAME,
        "candidate": asdict(spec),
        "provider": "yfinance",
        "source": "yfinance",
        "source_data_hash": SOURCE_DATA_HASH,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": FEATURE_SET,
        "feature_columns": process.get("feature_columns"),
        "n_features": process.get("n_features"),
        "target": TARGET_TYPE,
        "line_target_type": TARGET_TYPE,
        "band_target_type": TARGET_TYPE,
        "q_low": spec.q_low,
        "q_high": spec.q_high,
        "lambda_band": 2.0,
        "band_mode": spec.band_mode,
        "checkpoint_selection": "band_gate",
        "calibration": {
            "method": calibration_eval["calibration"]["method"],
            "params": calibration_eval["calibration"].get("params") or {},
            "fit_split": "validation",
            "test_or_latest_used_for_fit": False,
        },
        "dataset": {
            "timeframe": TIMEFRAME,
            "horizon": HORIZON,
            "eligible_ticker_count": len(plan.eligible_tickers),
            "excluded_ticker_count": len(plan.excluded_reasons),
            "train_samples": len(bundles["train_raw"]),
            "val_samples": len(bundles["val_raw"]),
            "test_samples": len(bundles["test_raw"]),
        },
        "product_gate": product_gate_result,
        "latest_artifact": {
            "path": str(LATEST_ARTIFACT_PATH) if latest_info else None,
            "jsonl_path": str(LATEST_JSONL_PATH) if latest_info else None,
            "asof_date": latest_info.get("asof_date") if latest_info else None,
            "prediction_count": latest_info.get("prediction_count") if latest_info else 0,
        },
        "storage_contract_validation": contract_result,
    }


def write_report(payload: dict[str, Any]) -> None:
    metrics = payload["calibration_eval"]["test"]["metrics"]
    comparison = payload["calibration_eval"]["comparison"]
    gate = payload["product_gate"]
    contract = payload.get("storage_contract_validation") or {}
    latest = payload.get("latest_artifact") or {}
    lines = [
        "# CP153-BM 1D Band Primary Product Candidate Local Save-Run Report",
        "",
        "## 결론",
        "",
        f"- 판정: `{payload['decision']}`",
        "- DB write는 수행하지 않았다.",
        "- DB attach는 Supabase quota refill 예정일인 2026-05-16 이후 사용자 승인 전까지 금지한다.",
        "- 이번 저장 대상은 primary `tide_s60_q15_param + lower_focused`뿐이며 TCN은 저장하지 않았다.",
        "",
        "## 계약",
        "",
        f"- provider/source: `yfinance / yfinance`",
        f"- source_data_hash: `{SOURCE_DATA_HASH}`",
        f"- timeframe/horizon: `{TIMEFRAME} / h{HORIZON}`",
        f"- feature_version: `{FEATURE_CONTRACT_VERSION}`",
        f"- feature_set: `{FEATURE_SET}`",
        f"- feature_columns: `{', '.join(payload['run_meta'].get('feature_columns') or [])}`",
        f"- target: `{TARGET_TYPE}`",
        "- model_role/output_role: `band / band`",
        f"- q_low/q_high: `{payload['spec']['q_low']} / {payload['spec']['q_high']}`",
        f"- calibration: `{payload['calibration_eval']['calibration']['method']}` validation-only fit",
        "",
        "## Save-Run Test Metrics",
        "",
        "| metric | save-run test | Stage 5T Tide ref | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("coverage_abs_error", "lower_breach_rate", "upper_breach_rate", "band_width_ic", "downside_width_ic"):
        lines.append(
            f"| {key} | {s45.safe_float(metrics.get(key)):.6f} | {comparison[f'{key}_stage5t_ref']:.6f} | {s45.safe_float(comparison.get(f'{key}_delta')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Product Gate",
            "",
            f"- gate pass: `{gate['pass']}`",
            f"- reasons: `{', '.join(gate['reasons']) if gate['reasons'] else 'none'}`",
            "",
            "## Latest-Only Artifact",
            "",
            f"- artifact path: `{LATEST_ARTIFACT_PATH}`",
            f"- jsonl path: `{LATEST_JSONL_PATH}`",
            f"- asof_date: `{latest.get('asof_date')}`",
            f"- prediction_count: `{latest.get('prediction_count')}`",
            f"- contract status: `{contract.get('status')}`",
            f"- line payload absent: `{contract.get('line_payload_absent')}`",
            f"- composite payload absent: `{contract.get('composite_payload_absent')}`",
            "",
            "## Product Payload Samples",
            "",
        ]
    )
    for sample in payload.get("product_payload_samples") or []:
        lines.append(
            f"- `{sample['ticker']}` lower={sample['lower_band_series']} upper={sample['upper_band_series']}"
        )
    lines.extend(
        [
            "",
            "## 산출물",
            "",
            f"- metrics: `{METRICS_PATH}`",
            f"- run meta: `{RUN_META_PATH}`",
            f"- calibration params: `{CALIBRATION_PATH}`",
            f"- latest artifact: `{LATEST_ARTIFACT_PATH}`",
            f"- logs: `{BASE_LOG_DIR}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    spec = CandidateSpec()
    bundles = build_precomputed_default_splits(spec)
    process = train_local_candidate(spec, bundles, force)
    if process.get("status") != "PASS":
        payload = {
            "cp": "CP153-BM",
            "stage": "1D band primary local save-run",
            "created_at_utc": now_utc(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "decision": "FAIL_LOCAL_TRAINING",
            "spec": asdict(spec),
            "process": process,
        }
        write_json(METRICS_PATH, payload)
        write_report_failure(payload)
        return payload
    calibration_eval = fit_and_evaluate_calibration(spec, process, bundles)
    gate = product_gate(calibration_eval["test"]["metrics"])
    latest_info: dict[str, Any] | None = None
    contract_result: dict[str, Any] | None = None
    product_payload_samples: list[dict[str, Any]] = []
    if gate["pass"]:
        latest_info = collect_latest_product_predictions(spec, process, bundles, calibration_eval["calibration"])
        records = latest_info["records"]
        contract_result = validate_local_product_payload(records)
        if contract_result["status"] == "PASS":
            artifact_payload = {
                "artifact_kind": "local_product_latest_only_band_predictions",
                "created_at_utc": now_utc(),
                "run_id": process.get("run_id"),
                "model_name": MODEL_NAME,
                "provider": "yfinance",
                "source": "yfinance",
                "source_data_hash": SOURCE_DATA_HASH,
                "feature_version": FEATURE_CONTRACT_VERSION,
                "feature_set": FEATURE_SET,
                "storage_contract": STORAGE_CONTRACT_PRODUCT_LATEST_ONLY,
                "db_write": False,
                "asof_date": latest_info["asof_date"],
                "prediction_count": latest_info["prediction_count"],
                "records": records,
            }
            write_json(LATEST_ARTIFACT_PATH, artifact_payload)
            write_jsonl(LATEST_JSONL_PATH, records)
            product_payload_samples = records[:5]
        else:
            gate = {**gate, "pass": False, "reasons": [*gate["reasons"], "storage_contract_failed"]}
    run_meta = build_run_meta(spec, process, bundles, calibration_eval, gate, latest_info, contract_result)
    write_json(RUN_META_PATH, run_meta)
    decision = "1D band primary product candidate ready for DB attach" if gate["pass"] and contract_result and contract_result["status"] == "PASS" else "LOCAL_SAVE_RUN_FAILED_OR_ABORTED"
    summary_rows = [
        {
            "candidate_id": spec.candidate_id,
            "run_id": process.get("run_id"),
            "decision": decision,
            "test_coverage_abs_error": calibration_eval["test"]["metrics"].get("coverage_abs_error"),
            "test_lower_breach_rate": calibration_eval["test"]["metrics"].get("lower_breach_rate"),
            "test_upper_breach_rate": calibration_eval["test"]["metrics"].get("upper_breach_rate"),
            "test_band_width_ic": calibration_eval["test"]["metrics"].get("band_width_ic"),
            "test_downside_width_ic": calibration_eval["test"]["metrics"].get("downside_width_ic"),
            "latest_prediction_count": latest_info.get("prediction_count") if latest_info else 0,
            "storage_contract_status": contract_result.get("status") if contract_result else None,
        }
    ]
    payload = {
        "cp": "CP153-BM",
        "stage": "1D band primary local save-run",
        "created_at_utc": now_utc(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decision": decision,
        "spec": asdict(spec),
        "process": process,
        "run_meta": run_meta,
        "calibration_eval": calibration_eval,
        "product_gate": gate,
        "latest_artifact": {
            "path": str(LATEST_ARTIFACT_PATH) if latest_info else None,
            "jsonl_path": str(LATEST_JSONL_PATH) if latest_info else None,
            "asof_date": latest_info.get("asof_date") if latest_info else None,
            "prediction_count": latest_info.get("prediction_count") if latest_info else 0,
            "asof_counts": latest_info.get("asof_counts") if latest_info else {},
        },
        "storage_contract_validation": contract_result,
        "product_payload_samples": product_payload_samples,
        "summary_rows": summary_rows,
        "forbidden_actions": {
            "supabase_db_write": False,
            "remote_db_dependency": False,
            "live_fetch": False,
            "eodhd_fallback": False,
            "composite": False,
            "line_payload_save": False,
            "wandb": False,
            "product_db_attach": False,
        },
    }
    write_json(METRICS_PATH, payload)
    write_rows(SUMMARY_PATH, summary_rows)
    write_report(payload)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return payload


def write_report_failure(payload: dict[str, Any]) -> None:
    lines = [
        "# CP153-BM 1D Band Primary Product Candidate Local Save-Run Report",
        "",
        f"- 판정: `{payload['decision']}`",
        "- DB write는 수행하지 않았다.",
        f"- failure: `{payload.get('process', {}).get('failure_reason')}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP153 BM 1D primary band local-only save-run")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "event": "start",
                "python": sys.executable,
                "torch": torch.__version__,
                "cuda": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "source_data_hash": SOURCE_DATA_HASH,
                "feature_set": FEATURE_SET,
                "feature_columns": s2.resolve_feature_columns(FEATURE_SET),
                "db_write": False,
                "wandb": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    result = run(force=args.force)
    print(json.dumps({"event": "end", "decision": result.get("decision"), "elapsed_seconds": result.get("elapsed_seconds")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
