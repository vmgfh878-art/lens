from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import warnings

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
try:  # noqa: E402
    from scipy.stats import ConstantInputWarning  # type: ignore
except Exception:  # pragma: no cover
    ConstantInputWarning = Warning  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai import cp146_lm_eodhd500_line_full_training as cp146  # noqa: E402
from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    SequenceDataset,
    SequenceDatasetBundle,
    build_dataset_plan,
    build_registry,
    normalize_sequence_splits,
    split_sequence_dataset_by_plan,
)
from ai.train import (  # noqa: E402
    FUTURE_COVARIATE_DIM,
    TrainConfig,
    estimate_train_risk_thresholds,
    line_gate_eligible,
    run_training,
    summarize_dataset_plan,
)


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_metrics.json"
BASELINE_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_baselines.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_logs"
ALIAS_DIR = LOG_DIR / "snapshot_alias"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"
CONTEXT_CHECKSUM = "1aa6452d82369cc6"

LINE_KEYS = [
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "direction_accuracy",
    "mae",
    "smape",
]

COMPOSITE_WEIGHTS = {
    "ic_mean": 0.20,
    "long_short_spread": 0.15,
    "fee_adjusted_return": 0.15,
    "false_safe_tail_rate": 0.25,
    "severe_downside_recall": 0.20,
    "upside_sacrifice": 0.05,
}

STAGE2_CANDIDATES = [
    {
        "candidate": "cp148_s2_patchtst_pvv_p32_s16",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "purpose": "CP146 pvv 기준의 Stage 2 재검증",
    },
    {
        "candidate": "cp148_s2_patchtst_no_fund_p32_s16",
        "model": "patchtst",
        "feature_set": "no_fundamentals",
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "purpose": "fundamentals 제거가 1D h5 line에 주는 영향 확인",
    },
    {
        "candidate": "cp148_s2_patchtst_pvv_p16_s8",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 16,
        "patch_stride": 8,
        "purpose": "dense patch가 하방 포착을 돕는지 확인",
    },
    {
        "candidate": "cp148_s2_patchtst_pvv_seq180",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": 180,
        "patch_len": 16,
        "patch_stride": 8,
        "purpose": "긴 context를 줄였을 때 false-safe가 개선되는지 확인",
    },
    {
        "candidate": "cp148_s2_tide_pvv",
        "model": "tide",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 16,
        "patch_stride": 8,
        "purpose": "TiDE가 PatchTST 대비 line 대안인지 확인",
    },
    {
        "candidate": "cp148_s2_cnn_lstm_pvv",
        "model": "cnn_lstm",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 16,
        "patch_stride": 8,
        "purpose": "CNN-LSTM의 국소 패턴/시차 기억 효과 확인",
    },
]


def _bind_cp146_paths() -> None:
    cp146.LOG_DIR = LOG_DIR
    cp146.ALIAS_DIR = ALIAS_DIR
    cp146.PREFLIGHT_PATH = PREFLIGHT_PATH


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt(value: Any, digits: int = 6) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _set_env() -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["MARKET_DATA_PROVIDER"] = "eodhd"
    os.environ["LENS_DATA_BACKEND"] = "local"
    os.environ["LENS_USE_LOCAL_SNAPSHOTS"] = "1"
    os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
    os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(ALIAS_DIR)


def _log(stage: str, payload: dict[str, Any] | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **(payload or {}),
    }
    with (LOG_DIR / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(record), ensure_ascii=False), flush=True)


def _make_config(
    *,
    candidate: dict[str, Any],
    epochs: int,
    seed: int,
    limit_tickers: int | None,
    device: str,
    batch_size: int,
) -> TrainConfig:
    return TrainConfig(
        model=str(candidate["model"]),
        timeframe="1D",
        horizon=5,
        seq_len=int(candidate["seq_len"]),
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-4,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.1,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=FUTURE_COVARIATE_DIM,
        use_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=limit_tickers,
        seed=seed,
        device=device,
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-cp148-local",
        model_ver="v2-multihead",
        early_stop_patience=10,
        early_stop_min_delta=1e-4,
        checkpoint_selection="line_gate",
        amp_dtype="bf16",
        detect_anomaly=False,
        explicit_cuda_cleanup=True,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=int(candidate["patch_len"]),
        patch_stride=int(candidate["patch_stride"]),
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=str(candidate["feature_set"]),
        feature_columns=None,
        n_features=len(MODEL_FEATURE_COLUMNS),
        market_data_provider="eodhd",
        lower_band_loss_weight=1.0,
        upper_band_loss_weight=1.0,
    )


def _source_hash_for_timeframe(timeframe: str) -> str:
    if PREFLIGHT_PATH.exists():
        payload = _read_json(PREFLIGHT_PATH)
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        check = checks.get(timeframe) if isinstance(checks, dict) else None
        if isinstance(check, dict) and check.get("source_data_hash"):
            return str(check["source_data_hash"])
        quick = payload.get("quick_check") if isinstance(payload.get("quick_check"), dict) else {}
        if quick.get("source_data_hash"):
            return str(quick["source_data_hash"])
    return cp146._preflight_source_hash(timeframe)


def _candidate_by_name(name: str) -> dict[str, Any]:
    for candidate in STAGE2_CANDIDATES:
        if candidate["candidate"] == name:
            return candidate
    if name == "timing_smoke_50ticker_1epoch":
        return {
            "candidate": name,
            "model": "patchtst",
            "feature_set": "price_volatility_volume",
            "seq_len": 252,
            "patch_len": 32,
            "patch_stride": 16,
            "purpose": "50티커 1epoch timing smoke",
        }
    raise KeyError(name)


def _prepare_dataset_splits_with_progress(config: TrainConfig, candidate_dir: Path) -> tuple:
    cp146._log_progress(candidate_dir, "feature_index_start", {"timeframe": config.timeframe})
    index_frame = cp146._read_cp146_index_frame(config.timeframe, config.market_data_provider, candidate_dir)
    index_frame = index_frame.copy()
    if config.tickers:
        selected = {ticker.upper() for ticker in config.tickers}
        index_frame = index_frame[index_frame["ticker"].isin(selected)].copy()
    elif config.limit_tickers is not None:
        selected = sorted(index_frame["ticker"].astype(str).str.upper().unique().tolist())[: int(config.limit_tickers)]
        index_frame = index_frame[index_frame["ticker"].isin(selected)].copy()
    cp146._log_progress(
        candidate_dir,
        "feature_index_done",
        {
            "rows": int(len(index_frame)),
            "ticker_count": int(index_frame["ticker"].nunique()) if "ticker" in index_frame.columns else None,
            "limit_tickers": config.limit_tickers,
        },
    )

    data_hash = _source_hash_for_timeframe(config.timeframe)
    cp146._log_progress(candidate_dir, "fingerprint_done", {"source_data_hash": data_hash})
    plan = build_dataset_plan(
        index_frame,
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
        market_data_provider=config.market_data_provider,
        source_data_hash=data_hash,
    )
    cp146._log_progress(
        candidate_dir,
        "dataset_plan_done",
        {
            "eligible_ticker_count": len(plan.eligible_tickers),
            "excluded_ticker_count": len(plan.excluded_reasons),
            "estimated_usable_sample_count": plan.estimated_usable_sample_count,
            "ticker_registry_path": plan.ticker_registry_path,
        },
    )
    if not plan.eligible_tickers:
        raise ValueError("eligible_tickers가 비어 있습니다.")

    feature_df, price_df = cp146._read_cp146_training_frames(
        config.timeframe,
        config.market_data_provider,
        plan.eligible_tickers,
        candidate_dir,
    )
    cp146._log_progress(
        candidate_dir,
        "training_frames_done",
        {
            "feature_rows": int(len(feature_df)),
            "price_rows": int(len(price_df)),
            "feature_ticker_count": int(feature_df["ticker"].nunique()) if "ticker" in feature_df.columns else None,
            "price_ticker_count": int(price_df["ticker"].nunique()) if "ticker" in price_df.columns else None,
        },
    )
    dataset = cp146._build_cp146_lazy_sequence_dataset_with_progress(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=config.timeframe,
        seq_len=config.seq_len,
        horizon=config.horizon,
        ticker_registry=build_registry(plan.eligible_tickers, plan.timeframe),
        include_future_covariate=config.model == "tide" and config.use_future_covariate,
        line_target_type=config.line_target_type,
        band_target_type=config.band_target_type,
        candidate_dir=candidate_dir,
    )
    cp146._log_progress(
        candidate_dir,
        "dataset_build_done",
        {
            "sample_count": len(dataset),
            "metadata_rows": int(len(dataset.metadata)),
            "ticker_count": len(dataset.ticker_arrays),
        },
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    cp146._log_progress(
        candidate_dir,
        "split_done",
        {"train_samples": len(train_bundle), "val_samples": len(val_bundle), "test_samples": len(test_bundle)},
    )
    train_bundle, val_bundle, test_bundle, mean, std = normalize_sequence_splits(train_bundle, val_bundle, test_bundle)
    cp146._log_progress(candidate_dir, "normalize_done", {"feature_dim": int(mean.shape[0])})
    return train_bundle, val_bundle, test_bundle, mean, std, plan


def _bundle_targets(bundle: SequenceDataset | SequenceDatasetBundle) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = bundle.metadata.reset_index(drop=True).copy()
    if isinstance(bundle, SequenceDatasetBundle):
        return metadata, bundle.raw_future_returns.detach().cpu().numpy().astype("float32")

    horizon = int(bundle.horizon)
    targets = np.empty((len(bundle.sample_refs), horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        anchor = float(closes[end_idx])
        future = closes[end_idx + 1 : end_idx + 1 + horizon]
        targets[row_idx, :] = (future / max(anchor, 1e-6)) - 1.0
    return metadata, targets


def _finite_tensor_count(tensor: torch.Tensor | np.ndarray) -> dict[str, int]:
    values = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    finite = np.isfinite(values)
    return {"elements": int(values.size), "nonfinite": int((~finite).sum())}


def _split_overlap_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    def keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
        return {
            (str(row.ticker), str(row.asof_date))
            for row in frame[["ticker", "asof_date"]].itertuples(index=False)
        }

    train_keys = keys(train)
    val_keys = keys(val)
    test_keys = keys(test)
    return {
        "train_val_overlap": len(train_keys & val_keys),
        "train_test_overlap": len(train_keys & test_keys),
        "val_test_overlap": len(val_keys & test_keys),
    }


def _history_signal_predictions(
    bundle: SequenceDataset | SequenceDatasetBundle,
    *,
    mode: str,
    lookback: int = 5,
) -> np.ndarray:
    if isinstance(bundle, SequenceDatasetBundle):
        return np.zeros((len(bundle), int(bundle.raw_future_returns.shape[1])), dtype="float32")

    horizon = int(bundle.horizon)
    steps = np.arange(1, horizon + 1, dtype="float32")
    predictions = np.zeros((len(bundle.sample_refs), horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        if mode in {"momentum", "reversal"}:
            if end_idx - lookback < 0:
                scalar = 0.0
            else:
                scalar = float(closes[end_idx] / max(float(closes[end_idx - lookback]), 1e-6) - 1.0)
            if mode == "reversal":
                scalar = -scalar
            predictions[row_idx, :] = (scalar / float(max(lookback, 1))) * steps
        elif mode == "rolling_mean":
            returns = np.diff(closes[: end_idx + 1]) / np.clip(closes[:end_idx], 1e-6, None)
            recent = returns[-lookback:] if returns.size else np.asarray([0.0], dtype="float32")
            scalar = float(np.nanmean(recent)) if recent.size else 0.0
            predictions[row_idx, :] = scalar * steps
    return predictions


def _historical_mean_predictions(
    train_meta: pd.DataFrame,
    train_targets: np.ndarray,
    eval_meta: pd.DataFrame,
) -> np.ndarray:
    global_mean = np.nanmean(train_targets, axis=0).astype("float32")
    ticker_means: dict[str, np.ndarray] = {}
    working = train_meta.reset_index(drop=True).copy()
    for ticker, group in working.groupby("ticker", sort=False):
        ticker_means[str(ticker)] = np.nanmean(train_targets[group.index.to_numpy()], axis=0).astype("float32")
    predictions = np.empty((len(eval_meta), train_targets.shape[1]), dtype="float32")
    for idx, ticker in enumerate(eval_meta["ticker"].astype(str).tolist()):
        predictions[idx, :] = ticker_means.get(ticker, global_mean)
    return predictions


def _evaluate_line_predictions(
    *,
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    targets: np.ndarray,
    severe_threshold: float | None,
) -> dict[str, Any]:
    line = torch.from_numpy(predictions.astype("float32"))
    actual = torch.from_numpy(targets.astype("float32"))
    metrics = summarize_forecast_metrics(
        metadata=metadata,
        line_predictions=line,
        lower_predictions=line,
        upper_predictions=line,
        line_targets=actual,
        band_targets=actual,
        raw_future_returns=actual,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        severe_downside_threshold=severe_threshold,
        fee_bps=10.0,
    )
    metrics["line_gate_pass"] = line_gate_eligible(metrics)
    return _extract_line_metrics(metrics)


def _extract_line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    source = metrics.get("line_metrics") if isinstance(metrics.get("line_metrics"), dict) else metrics
    result = {key: source.get(key) for key in LINE_KEYS if source.get(key) is not None}
    result["line_gate_pass"] = bool(metrics.get("line_gate_pass") or source.get("line_gate_pass"))
    return result


def run_preflight(*, device: str, batch_size: int) -> dict[str, Any]:
    _bind_cp146_paths()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cp146.ensure_snapshot_alias()
    _set_env()
    check = cp146._timeframe_quick_check("1D", horizon=5, seq_len=252)
    _write_json(
        PREFLIGHT_PATH,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "provider": "eodhd",
            "source": "eodhd",
            "data_backend": "local_parquet",
            "feature_contract": FEATURE_CONTRACT_VERSION,
            "checks": {"1D": check},
            "quick_check": check,
        },
    )

    timing_candidate = _candidate_by_name("timing_smoke_50ticker_1epoch")
    config = _make_config(
        candidate=timing_candidate,
        epochs=1,
        seed=42,
        limit_tickers=50,
        device=device,
        batch_size=batch_size,
    )
    candidate_dir = LOG_DIR / "timing_smoke_50ticker_1epoch"
    started = time.perf_counter()
    precomputed = _prepare_dataset_splits_with_progress(config, candidate_dir)
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    train_meta, train_targets = _bundle_targets(train_bundle)
    val_meta, val_targets = _bundle_targets(val_bundle)
    test_meta, test_targets = _bundle_targets(test_bundle)
    target_finite = {
        "train": _finite_tensor_count(train_targets),
        "validation": _finite_tensor_count(val_targets),
        "test": _finite_tensor_count(test_targets),
    }
    split_overlap = _split_overlap_summary(train_meta, val_meta, test_meta)
    result = run_training(
        config,
        save_run=False,
        precomputed_bundles=precomputed,
        local_log=True,
        local_log_dir=candidate_dir / "train_local_logs",
        wandb_name="cp148_timing_smoke_50ticker_1epoch",
        wandb_group="cp148_lm_1d",
        wandb_config_override={**asdict(config), "cp": "CP148-LM-1D", "candidate": timing_candidate["candidate"]},
    )
    elapsed = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    estimate_low = elapsed * 10.0 * 1.2
    estimate_high = elapsed * 10.0 * 1.5
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "eodhd",
        "source": "eodhd",
        "data_backend": "local_parquet",
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "model_feature_columns": len(MODEL_FEATURE_COLUMNS),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "context_checksum": CONTEXT_CHECKSUM,
        "checks": {"1D": check},
        "quick_check": check,
        "timing_smoke": {
            "candidate": timing_candidate["candidate"],
            "limit_tickers": 50,
            "epochs": 1,
            "elapsed_seconds": elapsed,
            "eligible_ticker_count": len(plan.eligible_tickers),
            "train_samples": len(train_bundle),
            "val_samples": len(val_bundle),
            "test_samples": len(test_bundle),
            "estimated_500ticker_1epoch_seconds_low": estimate_low,
            "estimated_500ticker_1epoch_seconds_high": estimate_high,
            "peak_vram_bytes": peak_vram,
            "run_id": result.get("run_id"),
            "line_gate_pass": (result.get("best_metrics") or {}).get("line_gate_pass"),
        },
        "target_finite": target_finite,
        "split_overlap": split_overlap,
        "full_eligible_ticker_count_expected": 473,
        "full_features_fundamentals_coverage_warn": True,
    }
    _write_json(PREFLIGHT_PATH, payload)
    _log("preflight_done", {"path": str(PREFLIGHT_PATH), "elapsed_seconds": round(elapsed, 3)})
    return payload


def _baseline_predictions(
    *,
    baseline: str,
    train_meta: pd.DataFrame,
    train_targets: np.ndarray,
    eval_meta: pd.DataFrame,
    eval_targets: np.ndarray,
    eval_bundle: SequenceDataset | SequenceDatasetBundle,
    seed: int,
) -> np.ndarray:
    if baseline == "zero_return":
        return np.zeros_like(eval_targets, dtype="float32")
    if baseline == "historical_mean":
        return _historical_mean_predictions(train_meta, train_targets, eval_meta)
    if baseline == "momentum_5d":
        return _history_signal_predictions(eval_bundle, mode="momentum", lookback=5)
    if baseline == "momentum_20d":
        return _history_signal_predictions(eval_bundle, mode="momentum", lookback=20)
    if baseline == "short_reversal":
        return _history_signal_predictions(eval_bundle, mode="reversal", lookback=5)
    if baseline == "rolling_mean":
        return _history_signal_predictions(eval_bundle, mode="rolling_mean", lookback=20)
    if baseline == "random_shuffled":
        rng = np.random.default_rng(seed)
        shuffled = eval_targets.copy()
        order = rng.permutation(len(shuffled))
        return shuffled[order]
    raise KeyError(baseline)


def _existing_product_line_reference() -> dict[str, Any]:
    path = PROJECT_ROOT / "docs" / "cp75_lm_1d_h5_full_line_product_candidate_metrics.json"
    if not path.exists():
        return {
            "baseline": "existing_1d_product_line_patchtst-1D-efad3c29d803",
            "status": "missing_reference",
            "note": "CP75 metrics 파일을 찾지 못했습니다.",
        }
    data = _read_json(path)
    metrics = data.get("val_line_metrics") if isinstance(data.get("val_line_metrics"), dict) else {}
    return {
        "baseline": "existing_1d_product_line_patchtst-1D-efad3c29d803",
        "status": "reference_only",
        "source_data_hash": (data.get("preflight") or {}).get("source_data_hash"),
        "feature_set": (data.get("final_product_candidate") or {}).get("feature_set"),
        "note": "기존 제품 line은 baseline/reference로만 사용하며 EODHD 500 Stage 2 후보와 혼동하지 않습니다.",
        "validation": {key: metrics.get(key) for key in LINE_KEYS if metrics.get(key) is not None},
    }


def run_baselines(*, device: str, batch_size: int) -> dict[str, Any]:
    _bind_cp146_paths()
    cp146.ensure_snapshot_alias()
    _set_env()
    candidate = {
        "candidate": "baseline_data_full_473",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
    }
    config = _make_config(candidate=candidate, epochs=1, seed=42, limit_tickers=None, device=device, batch_size=batch_size)
    candidate_dir = LOG_DIR / "baseline_data_full_473"
    precomputed = _prepare_dataset_splits_with_progress(config, candidate_dir)
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    train_meta, train_targets = _bundle_targets(train_bundle)
    val_meta, val_targets = _bundle_targets(val_bundle)
    test_meta, test_targets = _bundle_targets(test_bundle)
    thresholds = estimate_train_risk_thresholds(train_bundle)
    severe_threshold = thresholds.get("severe_downside_threshold")

    baselines = [
        "zero_return",
        "historical_mean",
        "momentum_5d",
        "momentum_20d",
        "short_reversal",
        "rolling_mean",
        "random_shuffled",
    ]
    rows = []
    for baseline in baselines:
        val_pred = _baseline_predictions(
            baseline=baseline,
            train_meta=train_meta,
            train_targets=train_targets,
            eval_meta=val_meta,
            eval_targets=val_targets,
            eval_bundle=val_bundle,
            seed=148,
        )
        test_pred = _baseline_predictions(
            baseline=baseline,
            train_meta=train_meta,
            train_targets=train_targets,
            eval_meta=test_meta,
            eval_targets=test_targets,
            eval_bundle=test_bundle,
            seed=149,
        )
        rows.append(
            {
                "baseline": baseline,
                "status": "evaluated",
                "validation": _evaluate_line_predictions(
                    metadata=val_meta,
                    predictions=val_pred,
                    targets=val_targets,
                    severe_threshold=severe_threshold,
                ),
                "test": _evaluate_line_predictions(
                    metadata=test_meta,
                    predictions=test_pred,
                    targets=test_targets,
                    severe_threshold=severe_threshold,
                ),
            }
        )
        _log("baseline_evaluated", {"baseline": baseline})

    rows.append(_existing_product_line_reference())
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_plan": summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle),
        "severe_downside_threshold": severe_threshold,
        "target_finite": {
            "train": _finite_tensor_count(train_targets),
            "validation": _finite_tensor_count(val_targets),
            "test": _finite_tensor_count(test_targets),
        },
        "split_overlap": _split_overlap_summary(train_meta, val_meta, test_meta),
        "baselines": rows,
    }
    _write_json(BASELINE_PATH, payload)
    _log("baseline_done", {"path": str(BASELINE_PATH)})
    return payload


def _candidate_process_path(candidate_name: str) -> Path:
    return LOG_DIR / candidate_name / "train_process.json"


def run_candidate(
    candidate_name: str,
    *,
    epochs: int,
    seed: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    _bind_cp146_paths()
    cp146.ensure_snapshot_alias()
    _set_env()
    candidate = _candidate_by_name(candidate_name)
    candidate_dir = LOG_DIR / candidate_name
    local_log_dir = candidate_dir / "train_local_logs"
    config = _make_config(
        candidate=candidate,
        epochs=epochs,
        seed=seed,
        limit_tickers=None,
        device=device,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    error = None
    result: dict[str, Any] | None = None
    try:
        _log("candidate_start", {"candidate": candidate_name, "epochs": epochs, "seed": seed, "device": device})
        precomputed = _prepare_dataset_splits_with_progress(config, candidate_dir)
        train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
        _log(
            "candidate_data_ready",
            {
                "candidate": candidate_name,
                "eligible_ticker_count": len(plan.eligible_tickers),
                "train_samples": len(train_bundle),
                "val_samples": len(val_bundle),
                "test_samples": len(test_bundle),
            },
        )
        result = run_training(
            config,
            save_run=False,
            precomputed_bundles=precomputed,
            local_log=True,
            local_log_dir=local_log_dir,
            wandb_name=f"cp148_{candidate_name}_seed{seed}",
            wandb_group="cp148_lm_1d",
            wandb_config_override={**asdict(config), "cp": "CP148-LM-1D", "candidate": candidate_name},
        )
        exit_code = 0
    except Exception as exc:
        exit_code = 1
        error = str(exc)
        _log("candidate_failed", {"candidate": candidate_name, "error": error})
    elapsed = time.perf_counter() - started
    process = {
        "candidate": candidate_name,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "epochs": epochs,
        "seed": seed,
        "save_run": False,
        "db_write": False,
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "result_run_id": (result or {}).get("run_id") if isinstance(result, dict) else None,
        "checkpoint_path": (result or {}).get("checkpoint_path") if isinstance(result, dict) else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "local_log_dir": str(local_log_dir),
        "error": error,
    }
    _write_json(_candidate_process_path(candidate_name), process)
    if exit_code != 0:
        raise RuntimeError(error or f"{candidate_name} failed")
    _log("candidate_done", {"candidate": candidate_name, "elapsed_seconds": round(elapsed, 3)})
    return process


def _find_latest_summary(candidate_name: str) -> dict[str, Any]:
    base = LOG_DIR / candidate_name / "train_local_logs"
    if not base.exists():
        return {}
    run_dirs = [path for path in base.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not run_dirs:
        return {}
    latest = sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    payload = _read_json(latest / "summary.json")
    payload["_run_dir"] = str(latest)
    config_path = latest / "config.json"
    if config_path.exists():
        payload["_config"] = _read_json(config_path).get("config", {})
    return payload


def _minimum_gate(metrics: dict[str, Any], baseline_summary: dict[str, Any]) -> bool:
    best = baseline_summary.get("best_validation") or {}
    ic = _safe_float(metrics.get("ic_mean")) or -999.0
    spread = _safe_float(metrics.get("long_short_spread")) or -999.0
    fee = _safe_float(metrics.get("fee_adjusted_return")) or -999.0
    false_safe = _safe_float(metrics.get("false_safe_tail_rate")) or 999.0
    severe = _safe_float(metrics.get("severe_downside_recall")) or -999.0
    best_false_safe = _safe_float(best.get("false_safe_tail_rate")) or 999.0
    best_severe = _safe_float(best.get("severe_downside_recall")) or -999.0
    return bool(ic > 0.0 and spread > 0.0 and fee > 0.0 and (false_safe < best_false_safe or severe > best_severe))


def _product_target(metrics: dict[str, Any]) -> bool:
    return bool(
        (_safe_float(metrics.get("ic_mean")) or -999.0) > 0.0
        and (_safe_float(metrics.get("long_short_spread")) or -999.0) > 0.0
        and (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) > 0.0
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.20
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.75
    )


def _failure_types(metrics: dict[str, Any], *, status: str) -> list[str]:
    if status != "PASS":
        return ["cache_contract_fail"]
    failures: list[str] = []
    if (_safe_float(metrics.get("ic_mean")) or -999.0) <= 0.0:
        failures.append("ranking_fail")
    if (_safe_float(metrics.get("long_short_spread")) or -999.0) <= 0.0:
        failures.append("spread_fail")
    if (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) <= 0.0:
        failures.append("fee_negative_fail")
    if (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) > 0.20:
        failures.append("false_safe_fail")
    if (_safe_float(metrics.get("severe_downside_recall")) or -999.0) < 0.75:
        failures.append("severe_recall_fail")
    if (_safe_float(metrics.get("upside_sacrifice")) or 0.0) > 0.08:
        failures.append("over_conservative_fail")
    return failures or ["none"]


def _baseline_best_summary(baseline_payload: dict[str, Any]) -> dict[str, Any]:
    evaluated = [
        row
        for row in baseline_payload.get("baselines", [])
        if row.get("status") == "evaluated" and isinstance(row.get("validation"), dict)
    ]
    if not evaluated:
        return {}
    best_false_safe = min(evaluated, key=lambda row: _safe_float((row.get("validation") or {}).get("false_safe_tail_rate")) or 999.0)
    best_severe = max(evaluated, key=lambda row: _safe_float((row.get("validation") or {}).get("severe_downside_recall")) or -999.0)
    best_ic = max(evaluated, key=lambda row: _safe_float((row.get("validation") or {}).get("ic_mean")) or -999.0)
    best_spread = max(evaluated, key=lambda row: _safe_float((row.get("validation") or {}).get("long_short_spread")) or -999.0)
    return {
        "best_false_safe_baseline": best_false_safe.get("baseline"),
        "best_severe_baseline": best_severe.get("baseline"),
        "best_ic_baseline": best_ic.get("baseline"),
        "best_spread_baseline": best_spread.get("baseline"),
        "best_validation": {
            "false_safe_tail_rate": (best_false_safe.get("validation") or {}).get("false_safe_tail_rate"),
            "severe_downside_recall": (best_severe.get("validation") or {}).get("severe_downside_recall"),
            "ic_mean": (best_ic.get("validation") or {}).get("ic_mean"),
            "long_short_spread": (best_spread.get("validation") or {}).get("long_short_spread"),
        },
    }


def _rank_scores(records: list[dict[str, Any]], key: str, *, lower_is_better: bool = False) -> dict[str, float]:
    values = []
    for record in records:
        value = _safe_float((record.get("validation") or {}).get(key))
        if value is not None:
            values.append((record["candidate"], value))
    if not values:
        return {record["candidate"]: 0.0 for record in records}
    values = sorted(values, key=lambda item: item[1], reverse=not lower_is_better)
    if len(values) == 1:
        return {values[0][0]: 1.0}
    scores = {}
    for rank, (candidate, _value) in enumerate(values):
        scores[candidate] = 1.0 - (rank / float(len(values) - 1))
    return {record["candidate"]: scores.get(record["candidate"], 0.0) for record in records}


def _apply_composite(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_maps = {
        "ic_mean": _rank_scores(records, "ic_mean"),
        "long_short_spread": _rank_scores(records, "long_short_spread"),
        "fee_adjusted_return": _rank_scores(records, "fee_adjusted_return"),
        "false_safe_tail_rate": _rank_scores(records, "false_safe_tail_rate", lower_is_better=True),
        "severe_downside_recall": _rank_scores(records, "severe_downside_recall"),
        "upside_sacrifice": _rank_scores(records, "upside_sacrifice", lower_is_better=True),
    }
    for record in records:
        score = 0.0
        parts = {}
        for key, weight in COMPOSITE_WEIGHTS.items():
            part = rank_maps[key].get(record["candidate"], 0.0)
            parts[key] = part
            score += weight * part
        record["composite_score"] = score
        record["composite_rank_parts"] = parts
    return sorted(records, key=lambda item: item.get("composite_score", 0.0), reverse=True)


def collect_stage2_results(baseline_payload: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_summary = _baseline_best_summary(baseline_payload)
    records = []
    for spec in STAGE2_CANDIDATES:
        candidate_name = spec["candidate"]
        process_path = _candidate_process_path(candidate_name)
        process = _read_json(process_path) if process_path.exists() else {}
        summary = _find_latest_summary(candidate_name)
        best = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
        test = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
        validation = _extract_line_metrics(best)
        test_line = _extract_line_metrics(test)
        status = "PASS" if process.get("exit_code") == 0 and validation else "MISSING"
        minimum = _minimum_gate(validation, baseline_summary) if validation else False
        product_target = _product_target(validation) if validation else False
        records.append(
            {
                "candidate": candidate_name,
                "status": status,
                "spec": spec,
                "process": process,
                "run_id": summary.get("run_id"),
                "run_dir": summary.get("_run_dir"),
                "checkpoint_path": summary.get("checkpoint_path"),
                "validation": validation,
                "test": test_line,
                "line_gate_pass": bool(best.get("line_gate_pass")),
                "minimum_candidate_gate": minimum,
                "product_target_pass_reference_only": product_target,
                "failure_types": _failure_types(validation, status=status),
            }
        )
    return _apply_composite(records)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _key in columns) + " |"
    separator = "| " + " | ".join("---" for _title, _key in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _title, key in columns) + " |")
    return "\n".join([header, separator, *body])


def _baseline_rows_for_report(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("baselines", []):
        metrics = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        rows.append(
            {
                "baseline": row.get("baseline"),
                "status": row.get("status"),
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fee": _fmt(metrics.get("fee_adjusted_return")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "severe": _fmt(metrics.get("severe_downside_recall")),
                "direction": _fmt(metrics.get("direction_accuracy")),
                "note": row.get("note", ""),
            }
        )
    return rows


def _candidate_rows_for_report(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record.get("validation") or {}
        spec = record.get("spec") or {}
        rows.append(
            {
                "candidate": record.get("candidate"),
                "model": spec.get("model"),
                "feature_set": spec.get("feature_set"),
                "seq": spec.get("seq_len"),
                "gate": record.get("line_gate_pass"),
                "min_gate": record.get("minimum_candidate_gate"),
                "score": _fmt(record.get("composite_score")),
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fee": _fmt(metrics.get("fee_adjusted_return")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "severe": _fmt(metrics.get("severe_downside_recall")),
                "failures": ", ".join(record.get("failure_types") or []),
            }
        )
    return rows


def _failure_taxonomy_table(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    actions = {
        "ranking_fail": "model family 변경, seq_len 축소, pvv/no_fundamentals 재비교",
        "spread_fail": "ranking이 실제 long-short로 연결되는지 baseline과 재비교",
        "false_safe_fail": "noisy feature 축소, dropout/weight_decay 강화, beta 변경은 별도 CP로 분리",
        "severe_recall_fail": "downside sample weighting 또는 false-safe-aware selector를 다음 CP로 제안",
        "over_conservative_fail": "risk-only 기록, beta/offset 조정은 별도 CP로 분리",
        "fee_negative_fail": "turnover/rebalance 민감도 재평가, 거래비용 취약 후보로 분류",
        "val_test_gap": "walk-forward 후보 제외 또는 기간 안정성 재검증",
        "seed_unstable": "seed median 기준 재정렬",
        "feature_noise_fail": "pvv/no_fundamentals 우선, full_features WARN 유지",
        "cache_contract_fail": "실험 폐기, Stage 0 재실행",
        "none": "Stage 3 또는 seed 재평가 후보",
    }
    rows = []
    for record in records:
        for failure in record.get("failure_types") or ["none"]:
            rows.append({"candidate": record["candidate"], "failure": failure, "next_action": actions.get(failure, "")})
    return rows


def _process_check() -> dict[str, Any]:
    current_pid = os.getpid()
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Where-Object { "
            "($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') "
            f"-and $_.ProcessId -ne {current_pid} "
            "-and ($_.CommandLine -notmatch 'cp148_lm_1d_stage0_2\\s+report') "
            "} | "
            "Select-Object @{Name='Id';Expression={$_.ProcessId}},"
            "@{Name='ProcessName';Expression={$_.Name}},"
            "CreationDate,ExecutablePath,CommandLine | "
            "ConvertTo-Json -Depth 3"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {"status": "check_failed", "error": str(exc)}
    text = completed.stdout.strip()
    if not text:
        return {"status": "none_visible", "processes": []}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "raw", "stdout": text}
    processes = parsed if isinstance(parsed, list) else [parsed]
    return {"status": "visible", "processes": processes}


def build_report() -> dict[str, Any]:
    preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.exists() else {}
    baseline_payload = _read_json(BASELINE_PATH) if BASELINE_PATH.exists() else {}
    baseline_summary = _baseline_best_summary(baseline_payload)
    candidates = collect_stage2_results(baseline_payload)
    process_check = _process_check()
    status = "PASS" if preflight and baseline_payload and any(row.get("status") == "PASS" for row in candidates) else "IN_PROGRESS"
    payload = {
        "cp": "CP148-LM-1D",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "scope_compliance": {
            "line_model_only": True,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "band_or_composite_experiment": False,
            "live_fetch": False,
            "product_promotion": False,
            "tcn_stage2_line_candidate": False,
        },
        "preflight": preflight,
        "baseline_summary": baseline_summary,
        "baselines": baseline_payload.get("baselines", []),
        "stage2_candidates": candidates,
        "process_check": process_check,
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(candidates)
    _write_report_markdown(payload)
    _log("report_done", {"report": str(REPORT_PATH), "metrics": str(METRICS_PATH)})
    return payload


def _write_summary_csv(candidates: list[dict[str, Any]]) -> None:
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate",
        "status",
        "model",
        "feature_set",
        "seq_len",
        "line_gate_pass",
        "minimum_candidate_gate",
        "composite_score",
        *[f"val_{key}" for key in LINE_KEYS],
        "failure_types",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in candidates:
            spec = record.get("spec") or {}
            metrics = record.get("validation") or {}
            row = {
                "candidate": record.get("candidate"),
                "status": record.get("status"),
                "model": spec.get("model"),
                "feature_set": spec.get("feature_set"),
                "seq_len": spec.get("seq_len"),
                "line_gate_pass": record.get("line_gate_pass"),
                "minimum_candidate_gate": record.get("minimum_candidate_gate"),
                "composite_score": record.get("composite_score"),
                "failure_types": ",".join(record.get("failure_types") or []),
            }
            row.update({f"val_{key}": metrics.get(key) for key in LINE_KEYS})
            writer.writerow(row)


def _write_report_markdown(payload: dict[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    timing = preflight.get("timing_smoke") or {}
    quick = preflight.get("quick_check") or {}
    baseline_rows = _baseline_rows_for_report(payload)
    candidate_rows = _candidate_rows_for_report(payload.get("stage2_candidates") or [])
    failure_rows = _failure_taxonomy_table(payload.get("stage2_candidates") or [])
    top_candidates = [
        row
        for row in payload.get("stage2_candidates", [])
        if row.get("status") == "PASS" and row.get("minimum_candidate_gate")
    ][:3]
    if not top_candidates:
        top_candidates = [row for row in payload.get("stage2_candidates", []) if row.get("status") == "PASS"][:3]

    lines = [
        "# CP148-LM-1D Stage 0~2 보고서",
        "",
        f"- 생성 시각: {payload.get('generated_at')}",
        "- 범위: EODHD 500 local parquet 기준 1D h5 line_model 연구 실험",
        "- 금지 준수: save-run 없음, DB write 없음, inference 저장 없음, band/composite 실험 없음, live fetch 없음, 제품 저장/연결 판단 없음",
        "- TCNQuantile: CP148-LM-1D Stage 2 line 후보에서 제외, BM 후보로 이관",
        "",
        "## 1. Stage 0 Preflight",
        "",
        _markdown_table(
            [
                {
                    "provider": "eodhd",
                    "feature_contract": preflight.get("feature_contract"),
                    "features": preflight.get("model_feature_columns"),
                    "atr": preflight.get("atr_ratio_in_model_features"),
                    "source_hash": quick.get("source_data_hash"),
                    "eligible": timing.get("eligible_ticker_count"),
                    "full_expected": preflight.get("full_eligible_ticker_count_expected"),
                    "split_overlap": json.dumps(preflight.get("split_overlap", {}), ensure_ascii=False),
                    "target_nonfinite": json.dumps(preflight.get("target_finite", {}), ensure_ascii=False),
                }
            ],
            [
                ("provider", "provider"),
                ("feature_contract", "feature_contract"),
                ("MODEL_FEATURE_COLUMNS", "features"),
                ("atr_ratio 포함", "atr"),
                ("source_data_hash", "source_hash"),
                ("50티커 eligible", "eligible"),
                ("전체 expected", "full_expected"),
                ("split overlap", "split_overlap"),
                ("target finite", "target_nonfinite"),
            ],
        ),
        "",
        "50티커 1epoch timing smoke 기준:",
        "",
        _markdown_table(
            [
                {
                    "elapsed": _fmt(timing.get("elapsed_seconds"), 2),
                    "low": _fmt(timing.get("estimated_500ticker_1epoch_seconds_low"), 2),
                    "high": _fmt(timing.get("estimated_500ticker_1epoch_seconds_high"), 2),
                    "vram": _fmt((timing.get("peak_vram_bytes") or 0) / (1024 * 1024), 2) if timing.get("peak_vram_bytes") else "",
                    "run_id": timing.get("run_id"),
                }
            ],
            [
                ("50티커 1epoch 초", "elapsed"),
                ("500티커 1epoch 추정 low", "low"),
                ("500티커 1epoch 추정 high", "high"),
                ("peak VRAM MB", "vram"),
                ("smoke run_id", "run_id"),
            ],
        ),
        "",
        "full_features는 fundamentals coverage가 낮아 기본 후보가 아니라 WARN 비교 후보로만 해석한다.",
        "",
        "## 2. Stage 1 Baseline",
        "",
        _markdown_table(
            baseline_rows,
            [
                ("baseline", "baseline"),
                ("status", "status"),
                ("IC", "ic"),
                ("spread", "spread"),
                ("fee_adj", "fee"),
                ("false_safe", "false_safe"),
                ("severe_recall", "severe"),
                ("direction", "direction"),
                ("note", "note"),
            ],
        ),
        "",
        "Stage 1 이후 gate 정리:",
        "",
        "- product target은 false_safe_tail_rate <= 0.20, severe_downside_recall >= 0.75를 유지한다.",
        "- Stage 2 minimum candidate gate는 IC > 0, spread > 0, fee_adjusted_return > 0, 그리고 baseline 대비 false-safe 개선 또는 severe recall 개선으로 표시한다.",
        "- 기존 1D product line `patchtst-1D-efad3c29d803`은 reference-only이며, EODHD 500 Stage 2 제품 후보와 혼동하지 않는다.",
        "",
        "## 3. Stage 2 Coarse 후보 결과",
        "",
        _markdown_table(
            candidate_rows,
            [
                ("candidate", "candidate"),
                ("model", "model"),
                ("feature_set", "feature_set"),
                ("seq", "seq"),
                ("line_gate", "gate"),
                ("min_gate", "min_gate"),
                ("score", "score"),
                ("IC", "ic"),
                ("spread", "spread"),
                ("fee_adj", "fee"),
                ("false_safe", "false_safe"),
                ("severe", "severe"),
                ("failure", "failures"),
            ],
        ),
        "",
        "## 4. Composite Score",
        "",
        "Composite는 validation 기준 rank score이며 가중치는 IC 0.20, spread 0.15, fee_adjusted_return 0.15, false_safe_tail 낮음 0.25, severe_recall 0.20, upside_sacrifice 낮음 0.05다.",
        "",
        _markdown_table(
            [
                {
                    "candidate": row.get("candidate"),
                    "score": _fmt(row.get("composite_score")),
                    "parts": json.dumps(row.get("composite_rank_parts", {}), ensure_ascii=False),
                }
                for row in payload.get("stage2_candidates", [])
            ],
            [("candidate", "candidate"), ("composite", "score"), ("rank parts", "parts")],
        ),
        "",
        "## 5. Failure Taxonomy",
        "",
        _markdown_table(
            failure_rows,
            [("candidate", "candidate"), ("failure", "failure"), ("next action", "next_action")],
        ),
        "",
        "## 6. 다음 단계 추천",
        "",
    ]
    if top_candidates:
        lines.append("Seed 재평가 또는 Stage 3 narrow sweep 후보:")
        lines.append("")
        for row in top_candidates:
            lines.append(f"- {row.get('candidate')}: composite {_fmt(row.get('composite_score'))}, minimum_gate={row.get('minimum_candidate_gate')}")
    else:
        lines.append("- Stage 2에서 baseline 대비 개선 후보가 없으면 Stage 3 진입을 보류하고 failure taxonomy를 확정한다.")
    lines.extend(
        [
            "",
            "이번 보고서는 제품 저장 결론을 내리지 않는다. 가능한 결론 표현은 Stage 3 narrow sweep 후보, seed 재평가 후보, baseline 대비 개선 후보, 탈락 후보로 제한한다.",
            "",
            "## 7. 잔여 프로세스 확인",
            "",
            "```json",
            json.dumps(_json_safe(payload.get("process_check")), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 8. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- baseline metrics: `{BASELINE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary csv: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    run_preflight(device=args.device, batch_size=args.batch_size)
    run_baselines(device=args.device, batch_size=args.batch_size)
    for candidate in STAGE2_CANDIDATES:
        process_path = _candidate_process_path(candidate["candidate"])
        if args.skip_completed and process_path.exists():
            process = _read_json(process_path)
            if process.get("exit_code") == 0:
                _log("candidate_skip_completed", {"candidate": candidate["candidate"]})
                continue
        run_candidate(
            candidate["candidate"],
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
        )
    return build_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148-LM-1D Stage 0~2 line 실험 실행기")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda")
    common.add_argument("--batch-size", type=int, default=256)
    common.add_argument("--epochs", type=int, default=3)
    common.add_argument("--seed", type=int, default=42)

    parser_preflight = subparsers.add_parser("preflight", parents=[common])
    parser_preflight.set_defaults(func=lambda args: run_preflight(device=args.device, batch_size=args.batch_size))

    parser_baseline = subparsers.add_parser("baseline", parents=[common])
    parser_baseline.set_defaults(func=lambda args: run_baselines(device=args.device, batch_size=args.batch_size))

    parser_candidate = subparsers.add_parser("run-candidate", parents=[common])
    parser_candidate.add_argument("--candidate", required=True)
    parser_candidate.set_defaults(
        func=lambda args: run_candidate(
            args.candidate,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
        )
    )

    parser_report = subparsers.add_parser("report")
    parser_report.set_defaults(func=lambda args: build_report())

    parser_all = subparsers.add_parser("all", parents=[common])
    parser_all.add_argument("--skip-completed", action="store_true")
    parser_all.set_defaults(func=run_all)
    return parser.parse_args()


def main() -> None:
    _bind_cp146_paths()
    _set_env()
    args = parse_args()
    payload = args.func(args)
    if isinstance(payload, dict):
        print(json.dumps(_json_safe({"status": payload.get("status", "OK"), "report": str(REPORT_PATH)}), ensure_ascii=False))


if __name__ == "__main__":
    main()
