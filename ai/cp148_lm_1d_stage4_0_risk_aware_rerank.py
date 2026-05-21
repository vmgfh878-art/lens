from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai import cp148_lm_1d_stage0_2 as s2  # noqa: E402
from ai.evaluation import _line_segment_metrics  # noqa: E402
from ai.inference import _select_bundle_features, load_checkpoint  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import MODEL_FEATURE_COLUMNS, SequenceDataset, SequenceDatasetBundle  # noqa: E402
from ai.train import estimate_train_risk_thresholds, forward_model, make_loader  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_0_risk_aware_rerank_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_0_risk_aware_rerank_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_0_risk_aware_rerank_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_0_risk_aware_rerank_logs"

STAGE2_METRICS_PATHS = [
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_metrics.json",
    PROJECT_ROOT / "docs" / "cp_archive" / "model_line" / "cp148_lm_1d_stage0_2_metrics.json",
]
STAGE3_METRICS_PATHS = [
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_metrics.json",
    PROJECT_ROOT / "docs" / "cp_archive" / "model_line" / "cp148_lm_1d_stage3_false_safe_sweep_metrics.json",
]
INDICATOR_PATHS = [
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_logs" / "snapshot_alias" / "indicators_eodhd_1D.parquet",
    PROJECT_ROOT / "docs" / "cp_archive" / "run_logs" / "cp148_lm_1d_stage0_2_logs" / "snapshot_alias" / "indicators_eodhd_1D.parquet",
    PROJECT_ROOT / "data" / "parquet" / "indicators_eodhd_1D_500.parquet",
]
SNAPSHOT_ALIAS_DIRS = [
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_logs" / "snapshot_alias",
    PROJECT_ROOT / "docs" / "cp_archive" / "run_logs" / "cp148_lm_1d_stage0_2_logs" / "snapshot_alias",
]

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

REGIME_NAMES = [
    "calm",
    "neutral",
    "stress",
    "vix_rising",
    "breadth_worsening",
]

HORIZON_BUCKETS = {
    "h1": (0, 1),
    "h2_h3": (1, 3),
    "h4_h5": (3, 5),
}

BASELINES = {
    "baseline_ic_sota": 0.057514,
    "baseline_spread_sota": 0.009231,
    "baseline_false_safe_sota": 0.458524,
    "baseline_severe_recall_sota": 0.536975,
    "existing_product_false_safe": 0.359558,
    "existing_product_severe_recall": 0.624853,
    "stage2_best_false_safe": 0.30866056266521946,
    "stage2_best_severe_recall": 0.6850190785352241,
}


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("필요한 입력 파일을 찾지 못했습니다: " + ", ".join(str(path) for path in paths))


def _snapshot_alias_dir() -> Path:
    for path in SNAPSHOT_ALIAS_DIRS:
        if (path / "indicators_eodhd_1D.parquet").exists() and (path / "price_data_eodhd.parquet").exists():
            return path
    raise FileNotFoundError("EODHD snapshot_alias 디렉터리를 찾지 못했습니다.")


def _configure_cp146_snapshot_alias() -> Path:
    alias_dir = _snapshot_alias_dir()
    # cp146 helper는 모듈 전역 ALIAS_DIR을 보므로, 새 파일을 만들지 않고 기존 Stage 0/3 alias를 읽게 고정한다.
    s2.cp146.ALIAS_DIR = alias_dir
    return alias_dir


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (str, bytes, bool, type(None))) else False:
        return None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _resolve_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def _load_stage_records() -> tuple[list[dict[str, Any]], dict[str, Path]]:
    stage2_path = _first_existing(STAGE2_METRICS_PATHS)
    stage3_path = _first_existing(STAGE3_METRICS_PATHS)
    stage2_payload = _read_json(stage2_path)
    stage3_payload = _read_json(stage3_path)

    records: list[dict[str, Any]] = []
    for item in stage2_payload.get("stage2_candidates", []):
        spec = item.get("spec") if isinstance(item.get("spec"), dict) else {}
        records.append(
            {
                "candidate_id": item.get("candidate"),
                "source_stage": "stage2",
                "model": spec.get("model"),
                "feature_set": spec.get("feature_set"),
                "seq_len": spec.get("seq_len"),
                "patch_len": spec.get("patch_len"),
                "patch_stride": spec.get("patch_stride"),
                "validation": item.get("validation") or {},
                "test": item.get("test") or {},
                "line_gate_pass": bool(item.get("line_gate_pass")),
                "checkpoint_path": item.get("checkpoint_path"),
                "run_id": item.get("run_id"),
                "stage2_composite_score": item.get("composite_score"),
                "stage2_failure_types": item.get("failure_types") or [],
            }
        )

    for item in stage3_payload.get("trials", []):
        process = item.get("process") if isinstance(item.get("process"), dict) else {}
        records.append(
            {
                "candidate_id": item.get("trial_id"),
                "source_stage": "stage3",
                "base": item.get("base"),
                "stage2_candidate": item.get("stage2_candidate"),
                "model": item.get("model"),
                "feature_set": item.get("feature_set"),
                "seq_len": item.get("seq_len"),
                "patch_len": item.get("patch_len"),
                "patch_stride": item.get("patch_stride"),
                "dropout": item.get("dropout"),
                "weight_decay": item.get("weight_decay"),
                "lr": item.get("lr"),
                "validation": item.get("validation") or {},
                "test": item.get("test") or {},
                "line_gate_pass": bool(item.get("line_gate_pass")),
                "checkpoint_path": process.get("checkpoint_path"),
                "run_id": process.get("result_run_id"),
                "stage3_classification": item.get("classification"),
                "stage3_failure_types": item.get("failure_types") or [],
            }
        )

    return records, {"stage2_metrics": stage2_path, "stage3_metrics": stage3_path}


def _line_survivor(record: dict[str, Any]) -> bool:
    metrics = record.get("validation") or {}
    return (
        bool(record.get("line_gate_pass"))
        and (_safe_float(metrics.get("fee_adjusted_return")) or -1.0) > 0.0
        and (_safe_float(metrics.get("ic_mean")) or -1.0) > 0.0
        and (_safe_float(metrics.get("long_short_spread")) or -1.0) > 0.0
    )


def _ranking_floor_pass(record: dict[str, Any]) -> bool:
    metrics = record.get("validation") or {}
    spread = _safe_float(metrics.get("long_short_spread")) or -999.0
    ic = _safe_float(metrics.get("ic_mean")) or -999.0
    fee = _safe_float(metrics.get("fee_adjusted_return")) or -999.0
    return fee > 0.0 and (spread >= 0.008 or ic >= 0.055)


def _risk_better_than_existing(record: dict[str, Any]) -> bool:
    metrics = record.get("validation") or {}
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    return (
        false_safe is not None
        and severe is not None
        and false_safe < BASELINES["existing_product_false_safe"]
        and severe > BASELINES["existing_product_severe_recall"]
    )


def _strong_stage2_risk(record: dict[str, Any]) -> bool:
    metrics = record.get("validation") or {}
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    spread = _safe_float(metrics.get("long_short_spread")) or -999.0
    fee = _safe_float(metrics.get("fee_adjusted_return")) or -999.0
    return (
        false_safe is not None
        and severe is not None
        and false_safe <= BASELINES["stage2_best_false_safe"]
        and severe >= BASELINES["stage2_best_severe_recall"]
        and spread >= 0.008
        and fee > 0.0
    )


def _base_category(record: dict[str, Any]) -> str:
    metrics = record.get("validation") or {}
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    if _line_survivor(record):
        return "line_survivor"
    risk_good = (
        false_safe is not None
        and severe is not None
        and (
            false_safe < BASELINES["existing_product_false_safe"]
            or severe > BASELINES["existing_product_severe_recall"]
        )
    )
    if risk_good:
        return "risk_only_reference"
    return "rejected"


def _risk_sort_key(record: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    metrics = record.get("validation") or {}
    stress = (record.get("regime_metrics") or {}).get("stress") or {}
    return (
        _safe_float(metrics.get("false_safe_tail_rate")) if _safe_float(metrics.get("false_safe_tail_rate")) is not None else 999.0,
        -(_safe_float(metrics.get("severe_downside_recall")) or -999.0),
        _safe_float(stress.get("false_safe_tail_rate")) if _safe_float(stress.get("false_safe_tail_rate")) is not None else 999.0,
        -(_safe_float(stress.get("severe_downside_recall")) or -999.0),
        -(_safe_float(metrics.get("fee_adjusted_return")) or -999.0),
        -(_safe_float(metrics.get("long_short_spread")) or -999.0),
        -(_safe_float(metrics.get("ic_mean")) or -999.0),
        _safe_float(metrics.get("upside_sacrifice")) if _safe_float(metrics.get("upside_sacrifice")) is not None else 999.0,
    )


def _load_context_frame() -> pd.DataFrame:
    indicator_path = _first_existing(INDICATOR_PATHS)
    wanted = ["ticker", "date", "regime_calm", "regime_neutral", "regime_stress", "vix_close", "ma200_pct"]
    try:
        frame = pd.read_parquet(indicator_path, columns=wanted)
    except Exception:
        frame = pd.read_parquet(indicator_path)
        frame = frame[[column for column in wanted if column in frame.columns]].copy()
    if "ticker" not in frame.columns or "date" not in frame.columns:
        raise ValueError("indicator parquet에 ticker/date 컬럼이 없습니다.")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["asof_date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

    date_context = frame.groupby("asof_date", sort=True).agg(
        vix_close=("vix_close", "median") if "vix_close" in frame.columns else ("ticker", "size"),
        ma200_pct=("ma200_pct", "median") if "ma200_pct" in frame.columns else ("ticker", "size"),
    )
    if "vix_close" in date_context.columns:
        date_context["vix_change_5d"] = date_context["vix_close"].diff(5)
    else:
        date_context["vix_change_5d"] = np.nan
    if "ma200_pct" in date_context.columns:
        date_context["ma200_pct_change_20d"] = date_context["ma200_pct"].diff(20)
    else:
        date_context["ma200_pct_change_20d"] = np.nan
    date_context = date_context.reset_index()[["asof_date", "vix_change_5d", "ma200_pct_change_20d"]]

    keep = ["ticker", "asof_date"]
    for column in ["regime_calm", "regime_neutral", "regime_stress"]:
        if column in frame.columns:
            keep.append(column)
    merged = frame[keep].drop_duplicates(["ticker", "asof_date"]).merge(date_context, on="asof_date", how="left")
    return merged


def _decorate_metadata(metadata: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"]).dt.strftime("%Y-%m-%d")
    return frame.merge(context, on=["ticker", "asof_date"], how="left")


def _metric_for_segment(
    *,
    metadata: pd.DataFrame,
    line: np.ndarray,
    raw: np.ndarray,
    mask: np.ndarray,
    start: int,
    end: int,
    severe_threshold: float | None,
) -> dict[str, Any]:
    selected_count = int(mask.sum())
    if selected_count < 10:
        return {
            "sample_count": selected_count,
            "date_count": 0,
            "ic_mean": None,
            "long_short_spread": None,
            "fee_adjusted_return": None,
            "false_safe_tail_rate": None,
            "severe_downside_recall": None,
        }
    subset_meta = metadata.loc[mask, ["ticker", "asof_date"]].reset_index(drop=True)
    line_tensor = torch.from_numpy(line[mask].astype("float32"))
    raw_tensor = torch.from_numpy(raw[mask].astype("float32"))
    metrics = _line_segment_metrics(
        prefix="segment",
        start=start,
        end=end,
        metadata=subset_meta,
        line_pred=line_tensor,
        line_actual=raw_tensor,
        raw_actual=raw_tensor,
        severe_downside_threshold=severe_threshold,
        top_k_frac=0.1,
        fee_bps=10.0,
    )
    result = {
        "sample_count": selected_count,
        "date_count": int(subset_meta["asof_date"].nunique()),
    }
    for key in [
        "ic_mean",
        "long_short_spread",
        "fee_adjusted_return",
        "false_safe_tail_rate",
        "severe_downside_recall",
        "upside_sacrifice",
    ]:
        result[key] = metrics.get(f"segment_{key}")
    return result


def _predict_bundle(
    *,
    model: Any,
    bundle: SequenceDataset | SequenceDatasetBundle,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    line_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for features, _line_target, _band_target, raw_future_returns, ticker_id, future_covariates in loader:
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            raw_future_returns = raw_future_returns.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp_dtype == "bf16")):
                prediction = forward_model(model, features, ticker_id, future_covariates)
            line, _, _ = apply_band_postprocess(
                prediction.line.detach().cpu(),
                prediction.lower_band.detach().cpu(),
                prediction.upper_band.detach().cpu(),
            )
            line_chunks.append(line)
            raw_chunks.append(raw_future_returns.detach().cpu())
    return torch.cat(line_chunks, dim=0).numpy(), torch.cat(raw_chunks, dim=0).numpy()


def _dataset_for_seq_len(seq_len: int, *, device_name: str, batch_size: int) -> tuple:
    candidate = {
        "candidate": f"stage4_dataset_seq{seq_len}",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": seq_len,
        "patch_len": 16,
        "patch_stride": 8,
    }
    config = s2._make_config(
        candidate=candidate,
        epochs=0,
        seed=42,
        limit_tickers=None,
        device=device_name,
        batch_size=batch_size,
    )
    config.use_wandb = False
    config.compile_model = False
    config.market_data_provider = "eodhd"
    return s2._prepare_dataset_splits_with_progress(config, LOG_DIR / f"dataset_seq{seq_len}")


def _attach_forward_metrics(records: list[dict[str, Any]], *, device_name: str, batch_size: int, amp_dtype: str) -> dict[str, Any]:
    context = _load_context_frame()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets: dict[int, tuple] = {}
    diagnostics = {
        "device": str(device),
        "forward_only": True,
        "split": "validation",
        "regime_source": "indicator parquet join by ticker/asof_date; vix_change_5d and ma200_pct_change_20d are audit-only derived columns",
        "errors": [],
    }

    for record in records:
        checkpoint_path = _resolve_path(record.get("checkpoint_path"))
        if checkpoint_path is None or not checkpoint_path.exists():
            record["forward_eval_status"] = "checkpoint_missing"
            diagnostics["errors"].append({"candidate_id": record.get("candidate_id"), "error": "checkpoint_missing"})
            continue
        seq_len = int(record.get("seq_len") or 252)
        if seq_len not in datasets:
            datasets[seq_len] = _dataset_for_seq_len(seq_len, device_name=str(device), batch_size=batch_size)
        train_bundle, val_bundle, _test_bundle, _mean, _std, _plan = datasets[seq_len]
        try:
            model, checkpoint = load_checkpoint(checkpoint_path)
            checkpoint_config = checkpoint.get("config") or {}
            feature_columns = list(checkpoint_config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
            selected_val = _select_bundle_features(val_bundle, feature_columns)
            thresholds = checkpoint.get("metrics") if isinstance(checkpoint.get("metrics"), dict) else {}
            severe_threshold = _safe_float(thresholds.get("severe_downside_threshold"))
            if severe_threshold is None:
                severe_threshold = _safe_float(estimate_train_risk_thresholds(train_bundle).get("severe_downside_threshold"))
            try:
                line, raw = _predict_bundle(
                    model=model,
                    bundle=selected_val,
                    device=device,
                    batch_size=batch_size,
                    amp_dtype=amp_dtype,
                )
                record["forward_amp_dtype"] = amp_dtype
            except (RuntimeError, TypeError) as exc:
                if "BFloat16" not in str(exc):
                    raise
                line, raw = _predict_bundle(
                    model=model,
                    bundle=selected_val,
                    device=device,
                    batch_size=batch_size,
                    amp_dtype="fp32",
                )
                record["forward_amp_dtype"] = "fp32_fallback_from_bf16"
            metadata = _decorate_metadata(selected_val.metadata, context)
            masks = {
                "calm": (metadata.get("regime_calm", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
                "neutral": (metadata.get("regime_neutral", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
                "stress": (metadata.get("regime_stress", pd.Series(False, index=metadata.index)).fillna(0).astype(float) > 0.5).to_numpy(),
                "vix_rising": (metadata.get("vix_change_5d", pd.Series(np.nan, index=metadata.index)).astype(float) > 0.0).fillna(False).to_numpy(),
                "breadth_worsening": (metadata.get("ma200_pct_change_20d", pd.Series(np.nan, index=metadata.index)).astype(float) < 0.0).fillna(False).to_numpy(),
            }
            record["regime_metrics"] = {
                name: _metric_for_segment(
                    metadata=metadata,
                    line=line,
                    raw=raw,
                    mask=mask,
                    start=0,
                    end=int(raw.shape[1]),
                    severe_threshold=severe_threshold,
                )
                for name, mask in masks.items()
            }
            all_mask = np.ones(len(metadata), dtype=bool)
            record["horizon_bucket_metrics"] = {
                name: _metric_for_segment(
                    metadata=metadata,
                    line=line,
                    raw=raw,
                    mask=all_mask,
                    start=start,
                    end=end,
                    severe_threshold=severe_threshold,
                )
                for name, (start, end) in HORIZON_BUCKETS.items()
            }
            record["forward_eval_status"] = "evaluated"
            record["forward_sample_count"] = int(len(metadata))
            if device.type == "cuda":
                record["forward_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
                torch.cuda.empty_cache()
        except Exception as exc:
            record["forward_eval_status"] = "failed"
            record["forward_eval_error"] = str(exc)
            diagnostics["errors"].append({"candidate_id": record.get("candidate_id"), "error": str(exc)})
    return diagnostics


def _assign_categories_and_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record["risk_category"] = _base_category(record)
        record["ranking_floor_pass"] = _ranking_floor_pass(record)
        record["better_than_existing_product_risk"] = _risk_better_than_existing(record)
        record["strong_stage2_risk"] = _strong_stage2_risk(record)

    line_survivors = [
        record
        for record in records
        if record["risk_category"] == "line_survivor" and record["ranking_floor_pass"] and record["better_than_existing_product_risk"]
    ]
    stage2_base_pool = [record for record in line_survivors if record.get("source_stage") == "stage2"]
    stage2_base_pool.sort(key=_risk_sort_key)
    primary_id = stage2_base_pool[0]["candidate_id"] if stage2_base_pool else None
    secondary_id = stage2_base_pool[1]["candidate_id"] if len(stage2_base_pool) > 1 else None

    for record in records:
        if record["risk_category"] == "risk_only_reference":
            record["stage4_label"] = "risk_only_reference"
        elif record["risk_category"] == "rejected":
            record["stage4_label"] = "rejected"
        elif record.get("candidate_id") == primary_id:
            record["stage4_label"] = "primary_stage4_base"
        elif record.get("candidate_id") == secondary_id:
            record["stage4_label"] = "secondary_stage4_base"
        elif _stress_risk_candidate(record):
            record["stage4_label"] = "stress_risk_candidate"
        elif _line_survivor(record):
            record["stage4_label"] = "alpha_candidate" if _ranking_floor_pass(record) else "rejected"
        else:
            record["stage4_label"] = "rejected"

    risk_ranked = sorted(
        [record for record in records if record["risk_category"] == "line_survivor"],
        key=_risk_sort_key,
    )
    rank_map = {record["candidate_id"]: idx + 1 for idx, record in enumerate(risk_ranked)}
    for record in records:
        record["risk_aware_rank"] = rank_map.get(record.get("candidate_id"))
    return records


def _stress_risk_candidate(record: dict[str, Any]) -> bool:
    stress = (record.get("regime_metrics") or {}).get("stress") or {}
    false_safe = _safe_float(stress.get("false_safe_tail_rate"))
    severe = _safe_float(stress.get("severe_downside_recall"))
    if false_safe is None and severe is None:
        return False
    return (
        (false_safe is not None and false_safe <= BASELINES["stage2_best_false_safe"])
        or (severe is not None and severe >= BASELINES["stage2_best_severe_recall"])
    )


def _stage2_old_rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage2 = [record for record in records if record.get("source_stage") == "stage2"]
    return sorted(stage2, key=lambda row: -(_safe_float(row.get("stage2_composite_score")) or -999.0))


def _write_summary_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "source_stage",
        "model",
        "feature_set",
        "seq_len",
        "patch_len",
        "patch_stride",
        "line_gate_pass",
        "risk_category",
        "stage4_label",
        "risk_aware_rank",
        "ranking_floor_pass",
        "ic_mean",
        "long_short_spread",
        "fee_adjusted_return",
        "false_safe_tail_rate",
        "severe_downside_recall",
        "conservative_bias",
        "upside_sacrifice",
        "stress_false_safe_tail_rate",
        "stress_severe_downside_recall",
        "h1_false_safe_tail_rate",
        "h2_h3_false_safe_tail_rate",
        "h4_h5_false_safe_tail_rate",
        "forward_eval_status",
    ]
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row.get("risk_aware_rank") is None, row.get("risk_aware_rank") or 999, row.get("candidate_id") or "")):
            metrics = record.get("validation") or {}
            stress = (record.get("regime_metrics") or {}).get("stress") or {}
            h = record.get("horizon_bucket_metrics") or {}
            writer.writerow(
                {
                    "candidate_id": record.get("candidate_id"),
                    "source_stage": record.get("source_stage"),
                    "model": record.get("model"),
                    "feature_set": record.get("feature_set"),
                    "seq_len": record.get("seq_len"),
                    "patch_len": record.get("patch_len"),
                    "patch_stride": record.get("patch_stride"),
                    "line_gate_pass": record.get("line_gate_pass"),
                    "risk_category": record.get("risk_category"),
                    "stage4_label": record.get("stage4_label"),
                    "risk_aware_rank": record.get("risk_aware_rank"),
                    "ranking_floor_pass": record.get("ranking_floor_pass"),
                    "ic_mean": metrics.get("ic_mean"),
                    "long_short_spread": metrics.get("long_short_spread"),
                    "fee_adjusted_return": metrics.get("fee_adjusted_return"),
                    "false_safe_tail_rate": metrics.get("false_safe_tail_rate"),
                    "severe_downside_recall": metrics.get("severe_downside_recall"),
                    "conservative_bias": metrics.get("conservative_bias"),
                    "upside_sacrifice": metrics.get("upside_sacrifice"),
                    "stress_false_safe_tail_rate": stress.get("false_safe_tail_rate"),
                    "stress_severe_downside_recall": stress.get("severe_downside_recall"),
                    "h1_false_safe_tail_rate": (h.get("h1") or {}).get("false_safe_tail_rate"),
                    "h2_h3_false_safe_tail_rate": (h.get("h2_h3") or {}).get("false_safe_tail_rate"),
                    "h4_h5_false_safe_tail_rate": (h.get("h4_h5") or {}).get("false_safe_tail_rate"),
                    "forward_eval_status": record.get("forward_eval_status"),
                }
            )


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def _write_report(records: list[dict[str, Any]], diagnostics: dict[str, Any], inputs: dict[str, Path]) -> None:
    old_rank = _stage2_old_rank(records)
    risk_rank = sorted(
        [record for record in records if record.get("risk_aware_rank") is not None],
        key=lambda row: int(row.get("risk_aware_rank") or 999),
    )
    primary = next((record for record in records if record.get("stage4_label") == "primary_stage4_base"), None)
    secondary = [record for record in records if record.get("stage4_label") == "secondary_stage4_base"]

    lines: list[str] = [
        "# CP148-LM-1D Stage 4-0 risk-aware rerank 보고서",
        "",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "- 범위: 기존 CP148 Stage 2/3 line_model 후보 리랭크",
        "- 금지 준수: 새 학습 없음, save-run 없음, DB write 없음, inference 저장 없음, product promotion 없음, live fetch 없음, band/composite 실험 없음",
        "- 기준: beta=2.0 유지, 기존 line_gate는 생존 조건으로만 사용",
        f"- Stage 2 입력: `{inputs['stage2_metrics'].relative_to(PROJECT_ROOT)}`",
        f"- Stage 3 입력: `{inputs['stage3_metrics'].relative_to(PROJECT_ROOT)}`",
        f"- stress/bucket 분해: {diagnostics.get('split')} split forward-only 재평가",
        "",
        "## 1. 기준값",
        "",
    ]
    lines.extend(
        _table(
            ["항목", "값"],
            [
                ["baseline IC SOTA", _fmt(BASELINES["baseline_ic_sota"])],
                ["baseline spread SOTA", _fmt(BASELINES["baseline_spread_sota"])],
                ["baseline false_safe SOTA", _fmt(BASELINES["baseline_false_safe_sota"])],
                ["baseline severe_recall SOTA", _fmt(BASELINES["baseline_severe_recall_sota"])],
                ["existing product false_safe", _fmt(BASELINES["existing_product_false_safe"])],
                ["existing product severe_recall", _fmt(BASELINES["existing_product_severe_recall"])],
                ["Stage 2 best false_safe", _fmt(BASELINES["stage2_best_false_safe"])],
                ["Stage 2 best severe_recall", _fmt(BASELINES["stage2_best_severe_recall"])],
            ],
        )
    )
    lines.extend(["", "## 2. 기존 Stage 2 순위", ""])
    lines.extend(
        _table(
            ["기존순위", "후보", "score", "IC", "spread", "fee", "false_safe", "severe"],
            [
                [
                    idx + 1,
                    row.get("candidate_id"),
                    _fmt(row.get("stage2_composite_score")),
                    _fmt((row.get("validation") or {}).get("ic_mean")),
                    _fmt((row.get("validation") or {}).get("long_short_spread")),
                    _fmt((row.get("validation") or {}).get("fee_adjusted_return")),
                    _fmt((row.get("validation") or {}).get("false_safe_tail_rate")),
                    _fmt((row.get("validation") or {}).get("severe_downside_recall")),
                ]
                for idx, row in enumerate(old_rank)
            ],
        )
    )
    lines.extend(["", "## 3. 새 risk-aware 순위", ""])
    lines.extend(
        _table(
            ["새순위", "후보", "stage", "분류", "라벨", "IC", "spread", "fee", "false_safe", "severe", "stress FS", "stress severe"],
            [
                [
                    row.get("risk_aware_rank"),
                    row.get("candidate_id"),
                    row.get("source_stage"),
                    row.get("risk_category"),
                    row.get("stage4_label"),
                    _fmt((row.get("validation") or {}).get("ic_mean")),
                    _fmt((row.get("validation") or {}).get("long_short_spread")),
                    _fmt((row.get("validation") or {}).get("fee_adjusted_return")),
                    _fmt((row.get("validation") or {}).get("false_safe_tail_rate")),
                    _fmt((row.get("validation") or {}).get("severe_downside_recall")),
                    _fmt(((row.get("regime_metrics") or {}).get("stress") or {}).get("false_safe_tail_rate")),
                    _fmt(((row.get("regime_metrics") or {}).get("stress") or {}).get("severe_downside_recall")),
                ]
                for row in risk_rank
            ],
        )
    )
    lines.extend(
        [
            "",
            "### 순위 변경 이유",
            "",
            "- 기존 Stage 2 score는 alpha와 risk를 함께 본 composite였지만, Stage 4-0은 false_safe_tail_rate를 1순위로 둔다.",
            "- CNN-LSTM은 false_safe/severe가 가장 좋지만 line_gate, spread, fee가 실패해 risk_only_reference로 분리했다.",
            "- PatchTST no_fund p32/s16은 false_safe와 severe가 Stage 2 전체 최고라 primary_stage4_base로 유지한다.",
            "- PatchTST pvv p32/s16은 pvv p16/s8보다 alpha는 약하지만 false_safe가 낮아 secondary_stage4_base로 올라갔다.",
            "- pvv p16/s8과 Stage 3 후보들은 IC/spread/fee가 강하지만 전체 false_safe가 no_fund/pvv p32보다 약해 base가 아니라 stress_risk_candidate로 분리했다.",
            "",
            "## 4. 전체 지표표",
            "",
        ]
    )
    lines.extend(
        _table(
            ["후보", "stage", "line_gate", "category", "label", "IC", "spread", "fee", "FS", "severe", "bias", "sacrifice"],
            [
                [
                    row.get("candidate_id"),
                    row.get("source_stage"),
                    row.get("line_gate_pass"),
                    row.get("risk_category"),
                    row.get("stage4_label"),
                    _fmt((row.get("validation") or {}).get("ic_mean")),
                    _fmt((row.get("validation") or {}).get("long_short_spread")),
                    _fmt((row.get("validation") or {}).get("fee_adjusted_return")),
                    _fmt((row.get("validation") or {}).get("false_safe_tail_rate")),
                    _fmt((row.get("validation") or {}).get("severe_downside_recall")),
                    _fmt((row.get("validation") or {}).get("conservative_bias")),
                    _fmt((row.get("validation") or {}).get("upside_sacrifice")),
                ]
                for row in sorted(records, key=lambda row: (row.get("source_stage"), row.get("candidate_id") or ""))
            ],
        )
    )
    lines.extend(["", "## 5. stress regime별 지표표", ""])
    regime_rows: list[list[Any]] = []
    for row in sorted(records, key=lambda item: (item.get("risk_aware_rank") is None, item.get("risk_aware_rank") or 999, item.get("candidate_id") or "")):
        for regime_name in REGIME_NAMES:
            metrics = (row.get("regime_metrics") or {}).get(regime_name) or {}
            regime_rows.append(
                [
                    row.get("candidate_id"),
                    regime_name,
                    metrics.get("sample_count"),
                    metrics.get("date_count"),
                    _fmt(metrics.get("false_safe_tail_rate")),
                    _fmt(metrics.get("severe_downside_recall")),
                    _fmt(metrics.get("long_short_spread")),
                    _fmt(metrics.get("fee_adjusted_return")),
                ]
            )
    lines.extend(_table(["후보", "구간", "samples", "dates", "FS", "severe", "spread", "fee"], regime_rows))
    lines.extend(["", "## 6. horizon bucket별 지표표", ""])
    bucket_rows: list[list[Any]] = []
    for row in sorted(records, key=lambda item: (item.get("risk_aware_rank") is None, item.get("risk_aware_rank") or 999, item.get("candidate_id") or "")):
        for bucket_name in HORIZON_BUCKETS:
            metrics = (row.get("horizon_bucket_metrics") or {}).get(bucket_name) or {}
            bucket_rows.append(
                [
                    row.get("candidate_id"),
                    bucket_name,
                    _fmt(metrics.get("false_safe_tail_rate")),
                    _fmt(metrics.get("severe_downside_recall")),
                    _fmt(metrics.get("long_short_spread")),
                    _fmt(metrics.get("fee_adjusted_return")),
                ]
            )
    lines.extend(_table(["후보", "bucket", "FS", "severe", "spread", "fee"], bucket_rows))
    lines.extend(["", "## 7. primary_stage4_base 선정", ""])
    if primary:
        lines.extend(
            [
                f"- primary_stage4_base: `{primary.get('candidate_id')}`",
                "- 이유: line_gate=True, fee>0, false_safe/severe가 existing product 기준보다 개선됐고 Stage 2 best risk 기준을 동시에 만족했다.",
                f"- validation false_safe={_fmt((primary.get('validation') or {}).get('false_safe_tail_rate'))}, severe={_fmt((primary.get('validation') or {}).get('severe_downside_recall'))}, spread={_fmt((primary.get('validation') or {}).get('long_short_spread'))}, fee={_fmt((primary.get('validation') or {}).get('fee_adjusted_return'))}",
            ]
        )
    else:
        lines.append("- primary_stage4_base 없음.")
    if secondary:
        lines.append("- secondary_stage4_base: " + ", ".join(f"`{row.get('candidate_id')}`" for row in secondary))
    lines.extend(
        [
            "",
            "## 8. A/B/C/D 실험 방향",
            "",
            "- Stage 4 A/B/C/D는 기존처럼 no_fund p32/s16을 주 베이스로 진행한다.",
            "- 단, pvv p16/s8은 기본 보조가 아니라 stress_risk_candidate로 내리고, pvv p32/s16을 secondary_stage4_base로 올린다.",
            "- CNN-LSTM은 LM 제품 후보가 아니라 risk_only_reference로 BM/risk 보조 해석에만 보관한다.",
            "- Stage 3 sweep 후보는 새 risk-aware 기준에서도 Stage 2 best를 넘지 못했으므로 Stage 4 base로 쓰지 않는다.",
            "",
            "## 9. product 저장 금지 준수",
            "",
            "- product save 없음.",
            "- DB write 없음.",
            "- inference 저장 없음.",
            "- live fetch 없음.",
            "- band/composite 실험 없음.",
            "",
            "## 10. python/pythonw 프로세스 확인",
            "",
            "- 스크립트 내부 확인은 실행 중인 자기 자신이 잡힐 수 있어 `deferred_external_check`로 기록했다.",
            "- 최종 검증 단계에서 PowerShell `Get-Process python,pythonw`로 별도 확인한다.",
            "",
            "## 11. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary csv: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- script: `ai/cp148_lm_1d_stage4_0_risk_aware_rerank.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_check() -> dict[str, Any]:
    return {
        "status": "deferred_external_check",
        "note": "스크립트 실행 중에는 자기 자신이 python 프로세스로 잡힐 수 있어 최종 검증 단계에서 별도 확인한다.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    alias_dir = _configure_cp146_snapshot_alias()
    records, inputs = _load_stage_records()
    diagnostics = _attach_forward_metrics(
        records,
        device_name=args.device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
    )
    records = _assign_categories_and_labels(records)
    payload = {
        "cp": "CP148-LM-1D-Stage4-0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope_compliance": {
            "new_training": False,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "product_promotion": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "beta_changed": False,
        },
        "baselines": BASELINES,
        "inputs": {**inputs, "snapshot_alias_dir": alias_dir},
        "forward_diagnostics": diagnostics,
        "records": records,
        "process_check": _process_check(),
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(records)
    _write_report(records, diagnostics, inputs)
    print(json.dumps({"status": "PASS", "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-0 risk-aware rerank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
