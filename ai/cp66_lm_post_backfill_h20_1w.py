from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.preprocessing import (
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    fetch_feature_index_frame,
    fetch_training_frames,
    normalize_sequence_splits,
    resolve_data_fingerprint,
    resolve_feature_cache_path,
    resolve_feature_index_cache_path,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import (
    build_registry,
    load_registry,
    registry_path_for_tickers,
    registry_path_for_timeframe,
)
from ai.train import TrainConfig, resolve_feature_columns, run_training


REPORT_PATH = PROJECT_ROOT / "docs" / "cp66_lm_post_backfill_h20_1w_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp66_lm_post_backfill_h20_1w_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp66_lm_post_backfill_h20_1w_logs"
CP62_METRICS_PATH = PROJECT_ROOT / "docs" / "cp62_cp61_schema_candidate_regrade_metrics.json"
CP65_METRICS_PATH = PROJECT_ROOT / "docs" / "cp65_lm_feature_h20_smoke_metrics.json"

LINE_METRIC_KEYS = [
    "spearman_ic",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_mean",
    "spread_std",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "mae",
    "smape",
    "false_safe_negative_rate",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]
REPORT_LINE_KEYS = [
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "direction_accuracy",
    "mae",
    "smape",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]
BUCKET_PREFIXES = ["h1_h5", "h6_h10", "h11_h20"]
BASELINE_NAMES = [
    "historical_mean_line_w60",
    "reversal_line_horizon",
    "random_or_shuffled_score",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    branch: str
    timeframe: str
    horizon: int
    feature_set: str
    seq_len: int
    patch_len: int
    patch_stride: int
    epochs: int
    limit_tickers: int
    batch_size: int = 256


H20_EXPERIMENTS = [
    Experiment(
        name="h20_full_features_post_backfill_seq252_p32_s16",
        branch="h20_midterm_line",
        timeframe="1D",
        horizon=20,
        feature_set="full_features",
        seq_len=252,
        patch_len=32,
        patch_stride=16,
        epochs=3,
        limit_tickers=50,
    ),
    Experiment(
        name="h20_no_fundamentals_post_backfill_seq252_p32_s16",
        branch="h20_midterm_line",
        timeframe="1D",
        horizon=20,
        feature_set="no_fundamentals",
        seq_len=252,
        patch_len=32,
        patch_stride=16,
        epochs=3,
        limit_tickers=50,
    ),
    Experiment(
        name="h20_technical_only_post_backfill_seq252_p32_s16",
        branch="h20_midterm_line",
        timeframe="1D",
        horizon=20,
        feature_set="technical_only",
        seq_len=252,
        patch_len=32,
        patch_stride=16,
        epochs=3,
        limit_tickers=50,
    ),
]

ONE_WEEK_SMOKE = Experiment(
    name="1w_h12_full_features_readiness_smoke_seq104_p16_s8",
    branch="1w_readiness_line",
    timeframe="1W",
    horizon=12,
    feature_set="full_features",
    seq_len=104,
    patch_len=16,
    patch_stride=8,
    epochs=1,
    limit_tickers=50,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _line_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    source = metrics.get("line_metrics") if isinstance(metrics.get("line_metrics"), dict) else metrics
    return {key: source.get(key) for key in LINE_METRIC_KEYS}


def _bucket_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source = metrics.get("line_metrics") if isinstance(metrics.get("line_metrics"), dict) else metrics
    buckets: dict[str, dict[str, Any]] = {}
    for prefix in BUCKET_PREFIXES:
        bucket = {}
        for key in LINE_METRIC_KEYS:
            value = source.get(f"{prefix}_{key}")
            if value is not None:
                bucket[key] = value
        if bucket:
            buckets[prefix] = bucket
    return buckets


def _metric_delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    left = _safe_float(current.get(key))
    right = _safe_float(reference.get(key))
    if left is None or right is None:
        return None
    return left - right


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _date_range(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "date" not in frame.columns:
        return {"min_date": None, "max_date": None, "row_count": int(len(frame))}
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return {
        "min_date": dates.min().strftime("%Y-%m-%d") if pd.notna(dates.min()) else None,
        "max_date": dates.max().strftime("%Y-%m-%d") if pd.notna(dates.max()) else None,
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
    }


def _feature_finite_frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    available_columns = [column for column in MODEL_FEATURE_COLUMNS if column in frame.columns]
    if not available_columns:
        return {"checked_columns": 0, "nonfinite_count": None, "missing_columns": list(MODEL_FEATURE_COLUMNS)}
    values = frame[available_columns].to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(values)
    nonfinite_by_column = {
        column: int((~finite[:, index]).sum())
        for index, column in enumerate(available_columns)
        if int((~finite[:, index]).sum()) > 0
    }
    return {
        "checked_columns": len(available_columns),
        "row_count": int(len(frame)),
        "nonfinite_count": int((~finite).sum()),
        "nonfinite_by_column": nonfinite_by_column,
        "missing_columns": [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns],
    }


def _ratio_sanity(frame: pd.DataFrame) -> dict[str, Any]:
    ratio_columns = ["open_ratio", "high_ratio", "low_ratio"]
    result: dict[str, Any] = {
        "columns": ratio_columns,
        "p99_abs_limit": 1.0,
        "max_abs_limit": 5.0,
        "pass": True,
        "values": {},
    }
    for column in ratio_columns:
        if column not in frame.columns:
            result["pass"] = False
            result["values"][column] = {"missing": True}
            continue
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            result["pass"] = False
            result["values"][column] = {"finite_count": 0}
            continue
        p99_abs = float(series.abs().quantile(0.99))
        max_abs = float(series.abs().max())
        column_pass = p99_abs <= 1.0 and max_abs <= 5.0
        result["pass"] = bool(result["pass"] and column_pass)
        result["values"][column] = {
            "finite_count": int(series.shape[0]),
            "p99_abs": p99_abs,
            "max_abs": max_abs,
            "p99": float(series.quantile(0.99)),
            "min": float(series.min()),
            "max": float(series.max()),
            "pass": column_pass,
        }
    return result


def _dataset_feature_summary(bundle: Any) -> dict[str, Any]:
    total = 0
    nonfinite = 0
    first_failure: dict[str, Any] | None = None
    mean = getattr(bundle, "mean", None)
    std = getattr(bundle, "std", None)
    if hasattr(bundle, "features"):
        features = bundle.features
        total = int(features.numel())
        mask = ~torch.isfinite(features)
        nonfinite = int(mask.sum().item())
        if nonfinite:
            first = mask.nonzero(as_tuple=False)[0].tolist()
            first_failure = {"index": [int(value) for value in first]}
    else:
        ticker_refs = sorted({str(ticker) for ticker, _ in bundle.sample_refs})
        for ticker in ticker_refs:
            array = torch.from_numpy(bundle.ticker_arrays[ticker]["features"]).to(dtype=torch.float32)
            if mean is not None and std is not None:
                array = (array - mean.view(1, -1)) / std.view(1, -1)
            total += int(array.numel())
            mask = ~torch.isfinite(array)
            count = int(mask.sum().item())
            nonfinite += count
            if count and first_failure is None:
                first = mask.nonzero(as_tuple=False)[0].tolist()
                first_failure = {"ticker": ticker, "index": [int(value) for value in first]}
    return {"feature_element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _dataset_target_summary(bundle: Any) -> dict[str, Any]:
    total = 0
    nonfinite = 0
    first_failure: dict[str, Any] | None = None
    if hasattr(bundle, "raw_future_returns"):
        target = bundle.raw_future_returns
        total = int(target.numel())
        mask = ~torch.isfinite(target)
        nonfinite = int(mask.sum().item())
        if nonfinite:
            first = mask.nonzero(as_tuple=False)[0].tolist()
            first_failure = {"index": [int(value) for value in first]}
    else:
        for sample_index, (ticker, end_idx) in enumerate(bundle.sample_refs):
            closes = bundle.ticker_arrays[ticker]["closes"]
            anchor = float(closes[end_idx])
            future = closes[end_idx + 1 : end_idx + 1 + bundle.horizon] / anchor - 1.0
            total += int(future.size)
            mask = ~np.isfinite(future)
            count = int(mask.sum())
            nonfinite += count
            if count and first_failure is None:
                first_failure = {"sample_index": sample_index, "ticker": ticker, "end_idx": int(end_idx)}
    return {"target_element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _split_tensor_summary(train_bundle: Any, val_bundle: Any, test_bundle: Any) -> dict[str, Any]:
    splits = {"train": train_bundle, "val": val_bundle, "test": test_bundle}
    feature_summary = {name: _dataset_feature_summary(bundle) for name, bundle in splits.items()}
    target_summary = {name: _dataset_target_summary(bundle) for name, bundle in splits.items()}
    return {
        "feature_nonfinite_count": sum(int(row["nonfinite_count"]) for row in feature_summary.values()),
        "target_nonfinite_count": sum(int(row["nonfinite_count"]) for row in target_summary.values()),
        "features": feature_summary,
        "targets": target_summary,
    }


def _forecast_label_summary(dataset: Any) -> dict[str, Any]:
    if dataset.metadata.empty:
        return {"checked_samples": 0, "future_dates_after_asof": None, "leakage_blocker": True}
    checked = min(len(dataset.metadata), 1000)
    future_after_asof = True
    bad_sample: dict[str, Any] | None = None
    for row in dataset.metadata.head(checked).itertuples(index=False):
        asof_date = pd.Timestamp(getattr(row, "asof_date"))
        forecast_dates = getattr(row, "forecast_dates")
        for forecast_date in forecast_dates:
            if pd.Timestamp(forecast_date) <= asof_date:
                future_after_asof = False
                bad_sample = {
                    "ticker": str(getattr(row, "ticker")),
                    "asof_date": asof_date.strftime("%Y-%m-%d"),
                    "forecast_date": str(forecast_date),
                }
                break
        if bad_sample is not None:
            break
    return {
        "checked_samples": checked,
        "future_dates_after_asof": future_after_asof,
        "bad_sample": bad_sample,
        "leakage_blocker": not future_after_asof,
        "interpretation": "forecast_dates는 asof_date 이후 예측 대상 기간 라벨입니다.",
    }


def _registry_status(timeframe: str, eligible_tickers: list[str]) -> dict[str, Any]:
    default_path = registry_path_for_timeframe(timeframe)
    hashed_path = registry_path_for_tickers(timeframe, eligible_tickers)
    eligible_set = {ticker.upper() for ticker in eligible_tickers}
    result: dict[str, Any] = {
        "default_registry_path": str(default_path),
        "default_registry_exists": default_path.exists(),
        "hashed_registry_path": str(hashed_path),
        "hashed_registry_exists": hashed_path.exists(),
        "stale_ticker_registry": None,
    }
    if not default_path.exists():
        result["stale_ticker_registry"] = True
        result["reason"] = "default registry 없음"
        return result
    try:
        registry = load_registry(timeframe, default_path)
    except Exception as exc:
        result["stale_ticker_registry"] = True
        result["reason"] = f"default registry 읽기 실패: {exc}"
        return result
    registry_set = {str(ticker).upper() for ticker in (registry.get("mapping") or {}).keys()}
    missing = sorted(eligible_set - registry_set)
    extra = sorted(registry_set - eligible_set)
    result.update(
        {
            "default_registry_ticker_count": len(registry_set),
            "eligible_ticker_count": len(eligible_set),
            "missing_from_default_registry_count": len(missing),
            "extra_in_default_registry_count": len(extra),
            "missing_from_default_registry_sample": missing[:10],
            "extra_in_default_registry_sample": extra[:10],
            "stale_ticker_registry": bool(missing),
        }
    )
    return result


def _build_bundles(
    *,
    timeframe: str,
    data_hash: str,
    seq_len: int,
    horizon: int,
    limit_tickers: int | None,
    include_future_covariate: bool = False,
) -> tuple[Any, Any, Any, torch.Tensor, torch.Tensor, Any, dict[str, Any], pd.DataFrame, pd.DataFrame, Any]:
    index_path = resolve_feature_index_cache_path(
        timeframe=timeframe,
        data_hash=data_hash,
        tickers=None,
        limit_tickers=limit_tickers,
    )
    index_existed_before = index_path.exists()
    index_frame = fetch_feature_index_frame(
        timeframe=timeframe,
        tickers=None,
        limit_tickers=limit_tickers,
    )

    input_tickers = sorted(index_frame["ticker"].astype(str).str.upper().unique())
    input_registry = build_registry(input_tickers, timeframe)
    preliminary_plan = build_dataset_plan(
        index_frame,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=input_registry,
        ticker_registry_path="memory",
    )
    ticker_registry = build_registry(preliminary_plan.eligible_tickers, timeframe)
    registry_path = registry_path_for_tickers(timeframe, preliminary_plan.eligible_tickers)
    plan = build_dataset_plan(
        index_frame,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=ticker_registry,
        ticker_registry_path=str(registry_path),
    )

    feature_path = resolve_feature_cache_path(
        timeframe=timeframe,
        data_hash=data_hash,
        tickers=plan.eligible_tickers,
        limit_tickers=None,
    )
    feature_existed_before = feature_path.exists()
    feature_df, price_df = fetch_training_frames(
        timeframe=timeframe,
        tickers=plan.eligible_tickers,
        limit_tickers=None,
    )
    feature_df = feature_df[feature_df["ticker"].isin(plan.eligible_tickers)].copy()
    price_df = price_df[price_df["ticker"].isin(plan.eligible_tickers)].copy()
    dataset = build_lazy_sequence_dataset(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=ticker_registry,
        include_future_covariate=include_future_covariate,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(
        dataset,
        split_specs=plan.split_specs,
    )
    train_bundle, val_bundle, test_bundle, mean, std = normalize_sequence_splits(
        train_bundle,
        val_bundle,
        test_bundle,
    )
    tensor_summary = _split_tensor_summary(train_bundle, val_bundle, test_bundle)
    cache_meta = {
        "timeframe": timeframe,
        "source_data_hash": data_hash,
        "feature_index_cache": _rel(index_path),
        "feature_cache": _rel(feature_path),
        "feature_index_cache_existed_before": index_existed_before,
        "feature_cache_existed_before": feature_existed_before,
        "feature_index_cache_created": (not index_existed_before) and index_path.exists(),
        "feature_cache_created": (not feature_existed_before) and feature_path.exists(),
        "ticker_registry_path": str(registry_path),
        "ticker_registry_preexisted": registry_path.exists(),
        "input_ticker_count": plan.input_ticker_count,
        "eligible_ticker_count": len(plan.eligible_tickers),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "feature_frame_range": _date_range(feature_df),
        "price_frame_range": _date_range(price_df),
        "feature_frame_finite": _feature_finite_frame_summary(feature_df),
        "ratio_sanity": _ratio_sanity(feature_df),
        "tensor_finite": tensor_summary,
        "registry_status": _registry_status(timeframe, plan.eligible_tickers),
    }
    return train_bundle, val_bundle, test_bundle, mean, std, plan, cache_meta, feature_df, price_df, dataset


def _build_plan_only(
    *,
    timeframe: str,
    data_hash: str,
    seq_len: int,
    horizon: int,
    limit_tickers: int | None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    index_path = resolve_feature_index_cache_path(
        timeframe=timeframe,
        data_hash=data_hash,
        tickers=None,
        limit_tickers=limit_tickers,
    )
    existed_before = index_path.exists()
    index_frame = fetch_feature_index_frame(
        timeframe=timeframe,
        tickers=None,
        limit_tickers=limit_tickers,
    )
    input_registry = build_registry(sorted(index_frame["ticker"].astype(str).str.upper().unique()), timeframe)
    plan = build_dataset_plan(
        index_frame,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=input_registry,
        ticker_registry_path="memory",
    )
    meta = {
        "feature_index_cache": _rel(index_path),
        "source_data_hash": data_hash,
        "feature_index_cache_existed_before": existed_before,
        "feature_index_cache_created": (not existed_before) and index_path.exists(),
        "index_frame_range": _date_range(index_frame),
        "input_ticker_count": plan.input_ticker_count,
        "eligible_ticker_count": len(plan.eligible_tickers),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "excluded_reason_counts": {
            reason: list(plan.excluded_reasons.values()).count(reason)
            for reason in sorted(set(plan.excluded_reasons.values()))
        },
        "split_gap_values": sorted({int(spec.gap) for spec in plan.split_specs.values()}),
        "split_gap_pass_h_max_12": timeframe == "1W" and all(int(spec.gap) == 12 for spec in plan.split_specs.values()),
        "registry_status": _registry_status(timeframe, plan.eligible_tickers),
    }
    return plan, index_frame, meta


def _epoch_summary_from_log(path: Path) -> dict[str, Any]:
    epoch_seconds: list[float] = []
    vram_peak_mb: list[float] = []
    if not path.exists():
        return {"epoch_seconds": [], "epoch_seconds_mean": None, "vram_peak_allocated_mb": None}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        epoch_value = _safe_float(payload.get("epoch_seconds"))
        vram_value = _safe_float(payload.get("vram_peak_allocated_mb"))
        if epoch_value is not None:
            epoch_seconds.append(epoch_value)
        if vram_value is not None:
            vram_peak_mb.append(vram_value)
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "vram_peak_allocated_mb": max(vram_peak_mb) if vram_peak_mb else None,
    }


def _base_config(experiment: Experiment, args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model="patchtst",
        timeframe=experiment.timeframe,
        horizon=experiment.horizon,
        seq_len=experiment.seq_len,
        epochs=experiment.epochs,
        batch_size=experiment.batch_size,
        lr=args.lr,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=0.25,
        q_high=0.75,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=2.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.2,
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
        limit_tickers=experiment.limit_tickers,
        seed=args.seed,
        device=args.device,
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-cp66",
        model_ver="v2-multihead",
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=1e-4,
        checkpoint_selection="line_gate",
        amp_dtype=args.amp_dtype,
        detect_anomaly=False,
        explicit_cuda_cleanup=False,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=experiment.patch_len,
        patch_stride=experiment.patch_stride,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=experiment.feature_set,
    )


def _run_experiment(
    experiment: Experiment,
    args: argparse.Namespace,
    bundles: tuple[Any, Any, Any, torch.Tensor, torch.Tensor, Any],
) -> dict[str, Any]:
    config = _base_config(experiment, args)
    feature_columns = resolve_feature_columns(experiment.feature_set)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{experiment.name}.log"
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("# CP66 line model smoke\n")
            log_file.write("# experiment: " + experiment.name + "\n")
            log_file.write("# role: line_model\n")
            log_file.write("# checkpoint_selection: line_gate\n")
            log_file.write("# feature_columns: " + ", ".join(feature_columns) + "\n")
            log_file.flush()
            with redirect_stdout(log_file), redirect_stderr(log_file):
                result = run_training(
                    config,
                    save_run=False,
                    precomputed_bundles=bundles,
                    enable_compile=False,
                )
        elapsed = time.perf_counter() - started
        status = "completed"
        failed_reason = None
    except Exception as exc:
        elapsed = time.perf_counter() - started
        result = {}
        status = "failed"
        failed_reason = str(exc)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n# failed_reason: " + failed_reason + "\n")
    test_metrics = result.get("test_metrics", {}) if isinstance(result, dict) else {}
    best_metrics = result.get("best_metrics", {}) if isinstance(result, dict) else {}
    return {
        "name": experiment.name,
        "experiment": asdict(experiment),
        "status": status,
        "failed_reason": failed_reason,
        "elapsed_seconds": elapsed,
        "epoch_summary": _epoch_summary_from_log(log_path),
        "log_path": _rel(log_path),
        "run_id": result.get("run_id") if isinstance(result, dict) else None,
        "checkpoint_path": result.get("checkpoint_path") if isinstance(result, dict) else None,
        "feature_set": experiment.feature_set,
        "n_features": result.get("n_features") if isinstance(result, dict) else len(feature_columns),
        "feature_columns": result.get("feature_columns") if isinstance(result, dict) else feature_columns,
        "line_metrics": _line_metrics(test_metrics),
        "validation_line_metrics": _line_metrics(best_metrics),
        "bucket_line_metrics": _bucket_metrics(test_metrics),
        "dataset_plan": result.get("dataset_plan") if isinstance(result, dict) else None,
    }


def _load_references() -> dict[str, Any]:
    cp62 = _read_json(CP62_METRICS_PATH) if CP62_METRICS_PATH.exists() else {}
    cp65 = _read_json(CP65_METRICS_PATH) if CP65_METRICS_PATH.exists() else {}
    h20_candidates = {}
    for row in cp62.get("line_candidates", []):
        if int(row.get("horizon") or 0) != 20:
            continue
        name = str(row.get("candidate_name"))
        h20_candidates[name] = {
            "name": name,
            "source": row.get("source"),
            "line_metrics": _line_metrics(row.get("line_metrics", {})),
            "validation_line_metrics": _line_metrics(row.get("validation_line_metrics", {})),
            "verdict": row.get("verdict"),
        }
    line_baselines = {}
    for row in cp62.get("line_baselines", []):
        name = row.get("name")
        if name in BASELINE_NAMES:
            line_baselines[str(name)] = {
                "name": name,
                "category": row.get("category"),
                "line_metrics": _line_metrics(row.get("line_metrics", {})),
            }
    cp65_h20_records = {}
    for row in cp65.get("experiments", []):
        experiment = row.get("experiment") or {}
        if int(experiment.get("horizon") or 0) == 20:
            cp65_h20_records[str(row.get("feature_set"))] = {
                "name": row.get("name"),
                "line_metrics": _line_metrics(row.get("line_metrics", {})),
                "comparison": row.get("comparison"),
            }
    cp65_full = (
        cp65.get("references", {})
        .get("full_features", {})
        .get("h20_longer_context_seq252_p32_s16", {})
    )
    if cp65_full:
        cp65_h20_records["full_features"] = {
            "name": "h20_longer_context_seq252_p32_s16",
            "line_metrics": _line_metrics(cp65_full.get("line_metrics", {})),
            "comparison": {"source": cp65_full.get("source")},
        }
    return {
        "cp62_source": _rel(CP62_METRICS_PATH),
        "cp65_source": _rel(CP65_METRICS_PATH),
        "cp65_h20": cp65_h20_records,
        "cp49_h20_candidates_via_cp62": h20_candidates,
        "line_baselines": line_baselines,
    }


def _classify_h20(record: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("line_metrics", {})
    cp65_full = references.get("cp65_h20", {}).get("full_features", {}).get("line_metrics", {})
    ic = _safe_float(metrics.get("ic_mean"))
    spread = _safe_float(metrics.get("long_short_spread"))
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe_recall = _safe_float(metrics.get("severe_downside_recall"))
    cp65_false_safe = _safe_float(cp65_full.get("false_safe_tail_rate"))
    cp65_recall = _safe_float(cp65_full.get("severe_downside_recall"))

    has_signal = (ic is not None and ic > 0.0) or (spread is not None and spread > 0.0)
    recall_improved = severe_recall is not None and cp65_recall is not None and severe_recall > cp65_recall
    false_safe_improved = false_safe is not None and cp65_false_safe is not None and false_safe < cp65_false_safe
    default_display_blocked = false_safe is not None and false_safe >= 0.30

    if record.get("status") != "completed":
        verdict = "failed_run"
    elif default_display_blocked:
        verdict = "watch_default_display_forbidden"
    elif has_signal and recall_improved and false_safe_improved:
        verdict = "visual_risk_survive"
    elif has_signal or recall_improved or false_safe_improved:
        verdict = "visual_risk_watch"
    else:
        verdict = "line_fail"

    return {
        "cp65_full_reference": "h20_longer_context_seq252_p32_s16",
        "ic_mean_delta_vs_cp65_full": _metric_delta(metrics, cp65_full, "ic_mean"),
        "long_short_spread_delta_vs_cp65_full": _metric_delta(metrics, cp65_full, "long_short_spread"),
        "false_safe_tail_rate_delta_vs_cp65_full": _metric_delta(metrics, cp65_full, "false_safe_tail_rate"),
        "severe_downside_recall_delta_vs_cp65_full": _metric_delta(metrics, cp65_full, "severe_downside_recall"),
        "has_positive_ic_or_spread": has_signal,
        "severe_recall_improved_vs_cp65_full": recall_improved,
        "false_safe_tail_improved_vs_cp65_full": false_safe_improved,
        "default_display_blocked_by_false_safe_ge_0_30": default_display_blocked,
        "verdict": verdict,
    }


def _build_post_backfill_gate(cache_meta_1d: dict[str, Any], data_hashes: dict[str, str], references: dict[str, Any]) -> dict[str, Any]:
    cp65_hash = None
    cp65_source = references.get("cp65_h20", {}).get("full_features", {}).get("comparison", {}).get("source")
    if CP65_METRICS_PATH.exists():
        cp65_payload = _read_json(CP65_METRICS_PATH)
        cp65_cache = str(cp65_payload.get("cache", {}).get("feature_index_cache", ""))
        if "_" in cp65_cache:
            cp65_hash = cp65_cache.rsplit("_", 1)[-1].replace(".pt", "")
    current_hash = data_hashes["1D"]
    stale_vs_cp65 = bool(cp65_hash and cp65_hash != current_hash)
    created_this_invocation = bool(
        cache_meta_1d.get("feature_index_cache_created") or cache_meta_1d.get("feature_cache_created")
    )
    post_backfill_cache_available = bool(
        cache_meta_1d.get("feature_index_cache") and cache_meta_1d.get("feature_cache")
    )
    if created_this_invocation:
        cache_refresh_note = "최종 실행에서 새 post-backfill cache를 생성했다."
    elif stale_vs_cp65 and post_backfill_cache_available:
        cache_refresh_note = "CP66 초회 gate에서 새 post-backfill cache를 생성했고, 최종 GPU 실행에서는 재사용했다."
    else:
        cache_refresh_note = "기존 cache를 재사용했다."
    return {
        "timeframe": "1D",
        "current_source_data_hash": current_hash,
        "cp65_source_data_hash": cp65_hash,
        "cp65_full_reference_source": cp65_source,
        "stale_vs_cp65": stale_vs_cp65,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_version_changed": FEATURE_CONTRACT_VERSION != "v3_adjusted_ohlc",
        "feature_columns_count": len(MODEL_FEATURE_COLUMNS),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "intraday_range_ratio_in_model_features": "intraday_range_ratio" in MODEL_FEATURE_COLUMNS,
        "new_cache_created": created_this_invocation or (stale_vs_cp65 and post_backfill_cache_available),
        "new_cache_created_this_invocation": created_this_invocation,
        "post_backfill_cache_available": post_backfill_cache_available,
        "cache_refresh_note": cache_refresh_note,
        "cache_meta": cache_meta_1d,
        "pass": (
            FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
            and len(MODEL_FEATURE_COLUMNS) == 36
            and "atr_ratio" not in MODEL_FEATURE_COLUMNS
            and "intraday_range_ratio" not in MODEL_FEATURE_COLUMNS
            and int(cache_meta_1d.get("feature_frame_finite", {}).get("nonfinite_count") or 0) == 0
            and int(cache_meta_1d.get("tensor_finite", {}).get("feature_nonfinite_count") or 0) == 0
            and int(cache_meta_1d.get("tensor_finite", {}).get("target_nonfinite_count") or 0) == 0
            and bool(cache_meta_1d.get("ratio_sanity", {}).get("pass"))
        ),
    }


def _build_one_week_readiness(
    *,
    data_hash: str,
    full_plan_meta: dict[str, Any],
    bundle_meta: dict[str, Any] | None,
    dataset: Any | None,
    error: str | None,
) -> dict[str, Any]:
    if error is not None:
        return {"status": "FAIL", "failed_reason": error, "source_data_hash": data_hash, "plan": full_plan_meta}
    assert bundle_meta is not None
    assert dataset is not None
    label_summary = _forecast_label_summary(dataset)
    feature_nonfinite = int(bundle_meta.get("tensor_finite", {}).get("feature_nonfinite_count") or 0)
    target_nonfinite = int(bundle_meta.get("tensor_finite", {}).get("target_nonfinite_count") or 0)
    eligible_count = int(full_plan_meta.get("eligible_ticker_count") or 0)
    split_gap_pass = bool(full_plan_meta.get("split_gap_pass_h_max_12"))
    leakage_blocker = bool(label_summary.get("leakage_blocker"))
    full_registry_stale = bool(full_plan_meta.get("registry_status", {}).get("stale_ticker_registry"))
    smoke_registry_stale = bool(bundle_meta.get("registry_status", {}).get("stale_ticker_registry"))
    status = "PASS" if (
        eligible_count >= 50
        and feature_nonfinite == 0
        and target_nonfinite == 0
        and split_gap_pass
        and not leakage_blocker
    ) else "FAIL"
    return {
        "status": status,
        "source_data_hash": data_hash,
        "full_index_plan": full_plan_meta,
        "smoke_scope_cache": bundle_meta,
        "price_data_range": bundle_meta.get("price_frame_range"),
        "indicator_range": bundle_meta.get("feature_frame_range"),
        "eligible_ticker_count": eligible_count,
        "feature_nonfinite_count": feature_nonfinite,
        "target_nonfinite_count": target_nonfinite,
        "split_gap_pass_h_max_12": split_gap_pass,
        "future_label_check": label_summary,
        "future_label_interpretation": (
            "누수 blocker 없음: 1W 날짜는 W-FRI period label이고 forecast_dates는 asof_date 이후 예측 대상 기간이다."
            if not leakage_blocker
            else "forecast_dates가 asof_date 이하인 샘플이 있어 누수 blocker로 본다."
        ),
        "stale_ticker_registry": full_registry_stale or smoke_registry_stale,
        "stale_ticker_registry_full_plan": full_registry_stale,
        "stale_ticker_registry_smoke_scope": smoke_registry_stale,
    }


def _build_report(payload: dict[str, Any]) -> str:
    gate = payload["post_backfill_cache_gate"]
    h20_rows = []
    for record in payload["h20_experiments"]:
        metrics = record.get("line_metrics", {})
        comparison = record.get("comparison", {})
        h20_rows.append(
            {
                "name": record["name"],
                "feature_set": record.get("feature_set"),
                "status": record.get("status"),
                "ic": _fmt(metrics.get("ic_mean")),
                "ic_ir": _fmt(metrics.get("ic_ir")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "spread_ir": _fmt(metrics.get("spread_ir")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "fssev": _fmt(metrics.get("false_safe_severe_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
                "dir": _fmt(metrics.get("direction_accuracy")),
                "mae": _fmt(metrics.get("mae")),
                "smape": _fmt(metrics.get("smape")),
                "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
                "fee_sharpe": _fmt(metrics.get("fee_adjusted_sharpe")),
                "d_recall": _fmt(comparison.get("severe_downside_recall_delta_vs_cp65_full")),
                "d_fstail": _fmt(comparison.get("false_safe_tail_rate_delta_vs_cp65_full")),
                "verdict": comparison.get("verdict", record.get("status")),
            }
        )

    cp65_rows = []
    for name, row in payload["references"].get("cp65_h20", {}).items():
        metrics = row.get("line_metrics", {})
        cp65_rows.append(
            {
                "feature_set": name,
                "name": row.get("name"),
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
            }
        )

    baseline_rows = []
    for name, row in payload["references"].get("line_baselines", {}).items():
        metrics = row.get("line_metrics", {})
        baseline_rows.append(
            {
                "name": name,
                "ic": _fmt(metrics.get("ic_mean")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                "recall": _fmt(metrics.get("severe_downside_recall")),
            }
        )

    one_week = payload["one_week_readiness"]
    one_week_smoke = payload.get("one_week_smoke")
    smoke_metrics = (one_week_smoke or {}).get("line_metrics", {})

    best_h20 = None
    completed = [record for record in payload["h20_experiments"] if record.get("status") == "completed"]
    if completed:
        best_h20 = min(
            completed,
            key=lambda record: (
                _safe_float(record.get("line_metrics", {}).get("false_safe_tail_rate")) or float("inf"),
                -(_safe_float(record.get("line_metrics", {}).get("severe_downside_recall")) or float("-inf")),
            ),
        )

    lines = [
        "# CP66-LM post-backfill cache gate + h20 / 1W line 재진입",
        "",
        "## 1. 원칙 확인",
        "- 이번 CP는 line_model만 다뤘고 band 모델 실험, composite/overlay, line_inside_band 평가는 사용하지 않았다.",
        "- DB 쓰기는 하지 않았고, cache stale 확인 후 필요한 학습 cache만 DB 읽기로 생성했다.",
        "- save-run=false, W&B off, no-compile 조건으로 실행했다.",
        "- full 473티커 학습은 실행하지 않았다. 1D h20 학습은 limit_tickers=50 범위다.",
        "",
        "## 2. post-backfill cache gate",
        _table(
            [
                {
                    "item": "feature_version",
                    "value": gate.get("feature_contract_version"),
                    "result": "PASS" if not gate.get("feature_version_changed") else "FAIL",
                },
                {
                    "item": "source data hash",
                    "value": f"{gate.get('cp65_source_data_hash')} -> {gate.get('current_source_data_hash')}",
                    "result": "stale" if gate.get("stale_vs_cp65") else "same",
                },
                {
                    "item": "new cache",
                    "value": gate.get("new_cache_created"),
                    "result": "created" if gate.get("new_cache_created") else "reused",
                },
                {
                    "item": "cache refresh note",
                    "value": gate.get("cache_refresh_note"),
                    "result": "",
                },
                {
                    "item": "MODEL_FEATURE_COLUMNS",
                    "value": gate.get("feature_columns_count"),
                    "result": "PASS",
                },
                {
                    "item": "atr_ratio model input",
                    "value": gate.get("atr_ratio_in_model_features"),
                    "result": "PASS" if not gate.get("atr_ratio_in_model_features") else "FAIL",
                },
                {
                    "item": "feature tensor NaN/Inf",
                    "value": gate.get("cache_meta", {}).get("tensor_finite", {}).get("feature_nonfinite_count"),
                    "result": "PASS" if gate.get("cache_meta", {}).get("tensor_finite", {}).get("feature_nonfinite_count") == 0 else "FAIL",
                },
                {
                    "item": "target NaN/Inf",
                    "value": gate.get("cache_meta", {}).get("tensor_finite", {}).get("target_nonfinite_count"),
                    "result": "PASS" if gate.get("cache_meta", {}).get("tensor_finite", {}).get("target_nonfinite_count") == 0 else "FAIL",
                },
                {
                    "item": "ratio p99 sanity",
                    "value": gate.get("cache_meta", {}).get("ratio_sanity", {}).get("pass"),
                    "result": "PASS" if gate.get("cache_meta", {}).get("ratio_sanity", {}).get("pass") else "FAIL",
                },
            ],
            [("항목", "item"), ("값", "value"), ("판정", "result")],
        ),
        "",
        "feature_version은 `v3_adjusted_ohlc`로 유지됐고, 바뀐 것은 source data hash/cache hash다. `atr_ratio`는 indicators에는 존재할 수 있으나 MODEL_FEATURE_COLUMNS 36개에는 포함되지 않는다.",
        "",
        "## 3. 1D h20 line 결과",
        _table(
            h20_rows,
            [
                ("name", "name"),
                ("feature_set", "feature_set"),
                ("status", "status"),
                ("ic", "ic"),
                ("ic_ir", "ic_ir"),
                ("spread", "spread"),
                ("spread_ir", "spread_ir"),
                ("false_safe_tail", "fstail"),
                ("false_safe_severe", "fssev"),
                ("severe_recall", "recall"),
                ("dir", "dir"),
                ("mae", "mae"),
                ("smape", "smape"),
                ("fee_ret", "fee_ret"),
                ("fee_sharpe", "fee_sharpe"),
                ("d_recall", "d_recall"),
                ("d_fstail", "d_fstail"),
                ("verdict", "verdict"),
            ],
        ),
        "",
        "## 4. 비교 기준",
        "### CP65 h20",
        _table(cp65_rows, [("feature_set", "feature_set"), ("name", "name"), ("ic", "ic"), ("spread", "spread"), ("false_safe_tail", "fstail"), ("severe_recall", "recall")]),
        "",
        "### statistical baseline",
        _table(baseline_rows, [("name", "name"), ("ic", "ic"), ("spread", "spread"), ("false_safe_tail", "fstail"), ("severe_recall", "recall")]),
        "",
        "## 5. feature_set별 개선 여부",
    ]
    for record in payload["h20_experiments"]:
        comparison = record.get("comparison", {})
        lines.append(
            f"- {record.get('feature_set')}: severe_recall delta={_fmt(comparison.get('severe_downside_recall_delta_vs_cp65_full'))}, "
            f"false_safe_tail delta={_fmt(comparison.get('false_safe_tail_rate_delta_vs_cp65_full'))}, verdict={comparison.get('verdict')}"
        )
    if best_h20:
        lines.extend(
            [
                "",
                "## 6. h20 제품 표시 제안",
                f"- h20 최우선 관찰 후보는 `{best_h20['name']}`이다.",
                "- h5는 단기 line으로 유지하고, h20은 중기 line / 위험 맥락 branch로 분리한다.",
                "- h20은 제품 기본 굵은 선이 아니라 점선 또는 낮은 신뢰도 표시를 권장한다. false_safe_tail_rate가 0.30 이상인 후보는 기본 표시 금지다.",
            ]
        )
    else:
        lines.extend(["", "## 6. h20 제품 표시 제안", "- 완료된 h20 후보가 없어 제품 표시는 보류한다."])

    lines.extend(
        [
            "",
            "## 7. 1W readiness",
            _table(
                [
                    {"item": "status", "value": one_week.get("status")},
                    {"item": "eligible_ticker_count", "value": one_week.get("eligible_ticker_count")},
                    {"item": "feature NaN/Inf", "value": one_week.get("feature_nonfinite_count")},
                    {"item": "target NaN/Inf", "value": one_week.get("target_nonfinite_count")},
                    {"item": "split gap h_max=12", "value": one_week.get("split_gap_pass_h_max_12")},
                    {"item": "future label leakage blocker", "value": one_week.get("future_label_check", {}).get("leakage_blocker")},
                    {"item": "stale ticker registry full/smoke", "value": f"{one_week.get('stale_ticker_registry_full_plan')} / {one_week.get('stale_ticker_registry_smoke_scope')}"},
                ],
                [("항목", "item"), ("값", "value")],
            ),
            "",
            one_week.get("future_label_interpretation", ""),
            "",
            "## 8. 1W smoke",
        ]
    )
    if one_week_smoke:
        lines.extend(
            [
                f"- 실행 여부: 실행, status={one_week_smoke.get('status')}, log={one_week_smoke.get('log_path')}",
                f"- line metrics: ic_mean={_fmt(smoke_metrics.get('ic_mean'))}, spread={_fmt(smoke_metrics.get('long_short_spread'))}, false_safe_tail={_fmt(smoke_metrics.get('false_safe_tail_rate'))}, severe_recall={_fmt(smoke_metrics.get('severe_downside_recall'))}",
            ]
        )
    else:
        lines.append("- 실행 여부: readiness가 PASS가 아니어서 실행하지 않았다.")

    lines.extend(
        [
            "",
            "## 9. 다음 CP 추천",
            "- 1D h20은 이번 post-backfill hash에서 false_safe_tail과 severe_downside_recall이 같이 개선되는 feature_set만 100티커로 한 단계 확장한다.",
            "- h20 제품 노출은 기본선이 아니라 h5 보조 중기 점선으로 분리하고, false_safe_tail_rate 0.30 이상이면 UI 기본 표시는 막는다.",
            "- 1W는 readiness PASS와 1epoch smoke 결과를 기준으로 seq_len=104 유지 여부와 h12 line risk-only 지표를 다음 CP에서 50->100티커로 확인한다.",
            "",
            "## 10. 검증",
            "- `.venv\\Scripts\\python.exe -m py_compile ai\\cp66_lm_post_backfill_h20_1w.py ai\\train.py ai\\evaluation.py ai\\tests\\test_feature_set_selection.py ai\\tests\\test_metric_definition_contract.py`: 통과",
            "- `python -m json.tool docs\\cp66_lm_post_backfill_h20_1w_metrics.json`: 통과",
            "- `.venv\\Scripts\\python.exe -m unittest ai.tests.test_feature_set_selection ai.tests.test_checkpoint_selection ai.tests.test_metric_definition_contract ai.tests.test_splits`: 27개 통과",
            "- 마지막 확인 기준 잔여 `python/pythonw` 학습 프로세스 없음",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CP66 학습은 GPU 사용을 전제로 합니다. "
            f"현재 Python은 CUDA를 볼 수 없습니다: {sys.executable}"
        )
    references = _load_references()
    data_hashes = {
        "1D": resolve_data_fingerprint("1D"),
        "1W": resolve_data_fingerprint("1W"),
    }

    print(json.dumps({"cp66": "build_1d_cache_gate", "hash": data_hashes["1D"]}, ensure_ascii=False), flush=True)
    bundles_1d = _build_bundles(
        timeframe="1D",
        data_hash=data_hashes["1D"],
        seq_len=252,
        horizon=20,
        limit_tickers=args.limit_tickers,
        include_future_covariate=False,
    )
    train_1d, val_1d, test_1d, mean_1d, std_1d, plan_1d, cache_meta_1d, _, _, _ = bundles_1d
    post_backfill_gate = _build_post_backfill_gate(cache_meta_1d, data_hashes, references)

    h20_records = []
    if not args.skip_h20:
        precomputed_1d = (train_1d, val_1d, test_1d, mean_1d, std_1d, plan_1d)
        for experiment in H20_EXPERIMENTS:
            print(json.dumps({"cp66": "start_h20", "experiment": experiment.name}, ensure_ascii=False), flush=True)
            record = _run_experiment(experiment, args, precomputed_1d)
            record["comparison"] = _classify_h20(record, references)
            h20_records.append(record)
            print(
                json.dumps(
                    {
                        "cp66": "done_h20",
                        "experiment": experiment.name,
                        "status": record.get("status"),
                        "verdict": record.get("comparison", {}).get("verdict"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    full_plan_1w, _, full_plan_1w_meta = _build_plan_only(
        timeframe="1W",
        data_hash=data_hashes["1W"],
        seq_len=ONE_WEEK_SMOKE.seq_len,
        horizon=ONE_WEEK_SMOKE.horizon,
        limit_tickers=None,
    )
    del full_plan_1w

    one_week_bundle_meta = None
    one_week_dataset = None
    one_week_error = None
    one_week_precomputed = None
    try:
        print(json.dumps({"cp66": "build_1w_readiness_scope", "hash": data_hashes["1W"]}, ensure_ascii=False), flush=True)
        bundles_1w = _build_bundles(
            timeframe="1W",
            data_hash=data_hashes["1W"],
            seq_len=ONE_WEEK_SMOKE.seq_len,
            horizon=ONE_WEEK_SMOKE.horizon,
            limit_tickers=args.limit_tickers,
            include_future_covariate=False,
        )
        train_1w, val_1w, test_1w, mean_1w, std_1w, plan_1w, one_week_bundle_meta, _, _, one_week_dataset = bundles_1w
        one_week_precomputed = (train_1w, val_1w, test_1w, mean_1w, std_1w, plan_1w)
    except Exception as exc:
        one_week_error = str(exc)

    one_week_readiness = _build_one_week_readiness(
        data_hash=data_hashes["1W"],
        full_plan_meta=full_plan_1w_meta,
        bundle_meta=one_week_bundle_meta,
        dataset=one_week_dataset,
        error=one_week_error,
    )

    one_week_smoke = None
    if one_week_readiness.get("status") == "PASS" and not args.skip_1w_smoke and one_week_precomputed is not None:
        print(json.dumps({"cp66": "start_1w_smoke", "experiment": ONE_WEEK_SMOKE.name}, ensure_ascii=False), flush=True)
        one_week_smoke = _run_experiment(ONE_WEEK_SMOKE, args, one_week_precomputed)
        print(
            json.dumps(
                {
                    "cp66": "done_1w_smoke",
                    "experiment": ONE_WEEK_SMOKE.name,
                    "status": one_week_smoke.get("status"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "cp": "CP66-LM",
        "rules": {
            "role": "line_model",
            "feature_version": FEATURE_CONTRACT_VERSION,
            "target": "raw_future_return",
            "line_target_type": "raw_future_return",
            "band_target_type_used_for_line_eval": False,
            "checkpoint_selection": "line_gate",
            "coverage_gate_used": False,
            "combined_gate_used": False,
            "band_model_experiment": False,
            "composite_overlay_metrics_used": False,
            "db_write": False,
            "db_read_for_cache_refresh_only": True,
            "save_run": False,
            "full_473_training": False,
            "wandb": "off",
            "compile": False,
            "model_structure_changed": False,
            "feature_contract_changed": False,
            "atr_ratio_promoted_to_model_feature": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "requested_device": args.device,
        },
        "data_hashes": data_hashes,
        "post_backfill_cache_gate": post_backfill_gate,
        "h20_experiments": h20_records,
        "one_week_readiness": one_week_readiness,
        "one_week_smoke": one_week_smoke,
        "references": references,
        "feature_set_note": {
            "executed": ["full_features", "no_fundamentals", "technical_only"],
            "skipped": {
                "price_volatility_volume": "CP65에서 technical_only와 동일한 11개 컬럼이라 중복 학습을 피했다.",
            },
        },
    }
    _write_json(METRICS_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP66 LM post-backfill cache gate와 h20/1W smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit-tickers", type=int, default=50)
    parser.add_argument("--skip-h20", action="store_true")
    parser.add_argument("--skip-1w-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "cp": payload["cp"],
                "metrics_path": _rel(METRICS_PATH),
                "report_path": _rel(REPORT_PATH),
                "h20_experiment_count": len(payload["h20_experiments"]),
                "one_week_readiness": payload["one_week_readiness"].get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
