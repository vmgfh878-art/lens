from __future__ import annotations

import os

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.inference import load_checkpoint  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    prepare_dataset_splits,
    resolve_data_fingerprint,
    resolve_feature_cache_path,
    resolve_feature_index_cache_path,
)
from ai.train import apply_feature_columns_to_splits, resolve_feature_columns, summarize_dataset_plan  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp112_lm_1w_line_smoke_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp112_lm_1w_line_smoke_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp112_lm_1w_line_smoke_logs"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"
BM_METRICS_PATH = PROJECT_ROOT / "docs" / "cp112_bm_1w_band_smoke_metrics.json"

LINE_KEYS = [
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

BUCKET_PREFIXES = ["h1_h5", "h6_h10", "h11_h20"]


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
    return json.loads(path.read_text(encoding="utf-8"))


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


def _snapshot_paths() -> dict[str, Any]:
    base = Path(os.environ.get("LENS_LOCAL_SNAPSHOT_DIR") or PROJECT_ROOT / "data" / "parquet")
    price = base / "price_data_yfinance_1W.parquet"
    indicators = base / "indicators_yfinance_1W.parquet"
    return {
        "base_dir": str(base),
        "price_data_yfinance_1W": str(price),
        "price_exists": price.exists(),
        "price_size_bytes": price.stat().st_size if price.exists() else None,
        "indicators_yfinance_1W": str(indicators),
        "indicators_exists": indicators.exists(),
        "indicators_size_bytes": indicators.stat().st_size if indicators.exists() else None,
    }


def _metadata_range(bundle: Any) -> dict[str, Any]:
    metadata = getattr(bundle, "metadata", None)
    if metadata is None or len(metadata) == 0 or "asof_date" not in metadata.columns:
        return {"row_count": int(len(bundle)), "min_asof_date": None, "max_asof_date": None}
    dates = pd.to_datetime(metadata["asof_date"], errors="coerce")
    return {
        "row_count": int(len(metadata)),
        "min_asof_date": dates.min().strftime("%Y-%m-%d") if pd.notna(dates.min()) else None,
        "max_asof_date": dates.max().strftime("%Y-%m-%d") if pd.notna(dates.max()) else None,
    }


def _feature_finite_summary(bundle: Any) -> dict[str, Any]:
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
            first_failure = {"index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()]}
    else:
        for ticker in sorted({str(ticker) for ticker, _ in bundle.sample_refs}):
            array = torch.from_numpy(bundle.ticker_arrays[ticker]["features"]).to(dtype=torch.float32)
            if mean is not None and std is not None:
                array = (array - mean.view(1, -1)) / std.view(1, -1)
            total += int(array.numel())
            mask = ~torch.isfinite(array)
            count = int(mask.sum().item())
            nonfinite += count
            if count and first_failure is None:
                first_failure = {"ticker": ticker, "index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()]}
    return {"element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _target_finite_summary(bundle: Any, horizon: int) -> dict[str, Any]:
    total = 0
    nonfinite = 0
    first_failure: dict[str, Any] | None = None
    if hasattr(bundle, "raw_future_returns"):
        target = bundle.raw_future_returns
        total = int(target.numel())
        mask = ~torch.isfinite(target)
        nonfinite = int(mask.sum().item())
        if nonfinite:
            first_failure = {"index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()]}
    else:
        for sample_index, (ticker, end_idx) in enumerate(bundle.sample_refs):
            closes = bundle.ticker_arrays[str(ticker)]["closes"]
            anchor = float(closes[int(end_idx)])
            future = closes[int(end_idx) + 1 : int(end_idx) + 1 + horizon]
            target = torch.tensor((future / anchor) - 1.0, dtype=torch.float32)
            total += int(target.numel())
            mask = ~torch.isfinite(target)
            count = int(mask.sum().item())
            nonfinite += count
            if count and first_failure is None:
                first_failure = {
                    "sample_index": int(sample_index),
                    "ticker": str(ticker),
                    "end_idx": int(end_idx),
                    "index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()],
                }
    return {"element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _line_metric_source(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    nested = metrics.get("line_metrics")
    if isinstance(nested, dict):
        return nested
    return metrics


def extract_line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = _line_metric_source(metrics)
    return {key: source.get(key) for key in LINE_KEYS}


def extract_bucket_metrics(metrics: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    source = _line_metric_source(metrics)
    result: dict[str, dict[str, Any]] = {}
    for prefix in BUCKET_PREFIXES:
        bucket = {}
        for key in LINE_KEYS:
            value = source.get(f"{prefix}_{key}")
            if value is not None:
                bucket[key] = value
        if bucket:
            result[prefix] = bucket
    return result


def classify_smoke_result(
    *,
    preflight_pass: bool,
    exit_code: int,
    line_metrics_present: bool,
    h4_shape_ok: bool,
) -> tuple[str, str | None]:
    if not preflight_pass:
        return "FAIL", "split_data_failure"
    if int(exit_code) != 0:
        return "FAIL", "training_runtime_failure"
    if not line_metrics_present or not h4_shape_ok:
        return "FAIL", "metric_shape_failure"
    return "PASS", None


def _shape_check(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"pass": False, "checkpoint_exists": False, "error": "checkpoint_missing"}
    try:
        model, checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint.get("config") or {}
        n_features = int(config.get("n_features") or len(config.get("feature_columns") or MODEL_FEATURE_COLUMNS))
        seq_len = int(config.get("seq_len") or 104)
        horizon = int(config.get("horizon") or 4)
        features = torch.randn(2, seq_len, n_features, dtype=torch.float32)
        ticker_id = torch.zeros(2, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            try:
                output = model(features, ticker_id=ticker_id)
            except TypeError:
                output = model(features)
        shapes = {
            "line": list(output.line.shape),
            "lower_band": list(output.lower_band.shape),
            "upper_band": list(output.upper_band.shape),
        }
        expected = [2, horizon]
        return {
            "pass": shapes["line"] == expected and shapes["lower_band"] == expected and shapes["upper_band"] == expected,
            "checkpoint_exists": True,
            "checkpoint_path": str(checkpoint_path),
            "expected_shape": expected,
            "shapes": shapes,
        }
    except Exception as exc:
        return {
            "pass": False,
            "checkpoint_exists": True,
            "checkpoint_path": str(checkpoint_path),
            "error": str(exc),
        }


def _find_run_dir(run_id: str, local_log_dir: Path | None) -> Path | None:
    candidates = []
    if local_log_dir is not None:
        candidates.append(local_log_dir)
        candidates.append(local_log_dir / run_id)
    candidates.append(LOG_DIR / "train_local_logs" / run_id)
    candidates.append(PROJECT_ROOT / "logs" / "runs" / run_id)
    for candidate in candidates:
        if candidate.exists() and (candidate / "summary.json").exists():
            return candidate
    return None


def _load_training_summary(run_id: str, local_log_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
    run_dir = _find_run_dir(run_id, local_log_dir)
    if run_dir is None:
        return {}, None
    return _read_json(run_dir / "summary.json"), run_dir


def build_preflight_payload() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data_hash = resolve_data_fingerprint("1W", market_data_provider="yfinance")
    feature_columns = resolve_feature_columns("full_features")
    train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
        timeframe="1W",
        seq_len=104,
        horizon=4,
        include_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        market_data_provider="yfinance",
    )
    train_bundle, val_bundle, test_bundle, mean, std = apply_feature_columns_to_splits(
        train_bundle,
        val_bundle,
        test_bundle,
        mean,
        std,
        feature_columns,
    )
    plan_summary = summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle)
    split_ranges = {
        "train": _metadata_range(train_bundle),
        "val": _metadata_range(val_bundle),
        "test": _metadata_range(test_bundle),
    }
    finite = {
        "train": {
            "features": _feature_finite_summary(train_bundle),
            "targets": _target_finite_summary(train_bundle, 4),
        },
        "val": {
            "features": _feature_finite_summary(val_bundle),
            "targets": _target_finite_summary(val_bundle, 4),
        },
        "test": {
            "features": _feature_finite_summary(test_bundle),
            "targets": _target_finite_summary(test_bundle, 4),
        },
    }
    feature_nonfinite_count = sum(int(split["features"]["nonfinite_count"]) for split in finite.values())
    target_nonfinite_count = sum(int(split["targets"]["nonfinite_count"]) for split in finite.values())
    snapshot_paths = _snapshot_paths()
    feature_cache_path = resolve_feature_cache_path(
        timeframe="1W",
        data_hash=data_hash,
        tickers=plan.eligible_tickers,
        market_data_provider="yfinance",
    )
    feature_index_cache_path = resolve_feature_index_cache_path(
        timeframe="1W",
        data_hash=data_hash,
        market_data_provider="yfinance",
    )
    bm_metrics = _read_json(BM_METRICS_PATH) if BM_METRICS_PATH.exists() else {}
    preflight = {
        "provider": getattr(plan, "provider", None),
        "source": getattr(plan, "source", None),
        "source_data_hash": data_hash,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": "full_features",
        "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
        "resolved_feature_columns_count": len(feature_columns),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "atr_ratio_in_feature_set": "atr_ratio" in feature_columns,
        "snapshot_paths": snapshot_paths,
        "feature_cache_path": str(feature_cache_path),
        "feature_cache_exists": feature_cache_path.exists(),
        "feature_index_cache_path": str(feature_index_cache_path),
        "feature_index_cache_exists": feature_index_cache_path.exists(),
        "dataset_plan": plan_summary,
        "split_ranges": split_ranges,
        "finite": finite,
        "feature_nonfinite_count": feature_nonfinite_count,
        "target_nonfinite_count": target_nonfinite_count,
        "cp112_bm_source_data_hash": ((bm_metrics.get("data") or {}).get("source_data_hash")),
        "data_hash_matches_cp112_bm": data_hash == ((bm_metrics.get("data") or {}).get("source_data_hash")),
    }
    gate_pass = bool(
        preflight["provider"] == "yfinance"
        and preflight["source"] == "yfinance"
        and FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
        and preflight["model_feature_columns_count"] == 36
        and preflight["resolved_feature_columns_count"] == 36
        and not preflight["atr_ratio_in_model_features"]
        and not preflight["atr_ratio_in_feature_set"]
        and snapshot_paths["price_exists"]
        and snapshot_paths["indicators_exists"]
        and int(plan_summary["train_samples"]) > 0
        and int(plan_summary["val_samples"]) > 0
        and int(plan_summary["test_samples"]) > 0
        and feature_nonfinite_count == 0
        and target_nonfinite_count == 0
    )
    payload = {
        "cp": "CP112-LM",
        "purpose": "1W PatchTST line smoke validation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PREFLIGHT_PASS" if gate_pass else "PREFLIGHT_FAIL",
        "preflight_gate_pass": gate_pass,
        "environment": {
            "MARKET_DATA_PROVIDER": os.environ.get("MARKET_DATA_PROVIDER"),
            "LENS_USE_LOCAL_SNAPSHOTS": os.environ.get("LENS_USE_LOCAL_SNAPSHOTS"),
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": os.environ.get("LENS_REQUIRE_LOCAL_SNAPSHOTS"),
            "LENS_LOCAL_SNAPSHOT_DIR": os.environ.get("LENS_LOCAL_SNAPSHOT_DIR"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
        },
        "scope_guard": {
            "line_model_only": True,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "off",
            "band_model_experiment": False,
            "composite_overlay_experiment": False,
        },
        "preflight": preflight,
    }
    _write_json(PREFLIGHT_PATH, payload)
    _write_json(METRICS_PATH, payload)
    return payload


def _training_config_status(config: dict[str, Any], best_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model"),
        "timeframe": config.get("timeframe"),
        "horizon": config.get("horizon"),
        "seq_len": config.get("seq_len"),
        "patch_len": config.get("patch_len"),
        "patch_stride": config.get("patch_stride"),
        "feature_set": config.get("feature_set"),
        "line_target_type": config.get("line_target_type"),
        "band_target_type": config.get("band_target_type"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "ci_aggregate": config.get("ci_aggregate"),
        "alpha": config.get("alpha"),
        "beta": config.get("beta"),
        "delta": config.get("delta"),
        "role": best_metrics.get("role") or config.get("role"),
        "n_features": config.get("n_features"),
        "amp_dtype": config.get("amp_dtype"),
        "compile_model": config.get("compile_model"),
        "use_wandb": config.get("use_wandb"),
        "market_data_provider": config.get("market_data_provider"),
    }


def _gate_status(best_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_selection": best_metrics.get("checkpoint_selection"),
        "gate_type": best_metrics.get("gate_type"),
        "gate_failed": best_metrics.get("gate_failed"),
        "line_gate_pass": best_metrics.get("line_gate_pass"),
        "band_gate_pass": best_metrics.get("band_gate_pass"),
        "combined_gate_pass": best_metrics.get("combined_gate_pass"),
        "role": best_metrics.get("role"),
        "best_epoch": best_metrics.get("best_epoch"),
        "selected_epoch": best_metrics.get("selected_epoch"),
        "selection_reason": best_metrics.get("selection_reason"),
    }


def build_postflight_payload(args: argparse.Namespace) -> dict[str, Any]:
    existing = _read_json(METRICS_PATH) if METRICS_PATH.exists() else build_preflight_payload()
    preflight_pass = bool(existing.get("preflight_gate_pass"))
    local_log_dir = Path(args.local_log_dir).resolve() if args.local_log_dir else None
    summary, run_dir = _load_training_summary(args.run_id, local_log_dir)
    best_metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
    test_metrics = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    line_metrics = extract_line_metrics(test_metrics)
    bucket_metrics = extract_bucket_metrics(test_metrics)
    line_metrics_present = any(value is not None for value in line_metrics.values())
    checkpoint_path = Path(str(summary.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    if not config and checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = dict(checkpoint.get("config") or {})
    shape_check = _shape_check(checkpoint_path) if str(checkpoint_path) else {"pass": False, "error": "checkpoint_path_missing"}
    status, failure_class = classify_smoke_result(
        preflight_pass=preflight_pass,
        exit_code=int(args.exit_code),
        line_metrics_present=line_metrics_present,
        h4_shape_ok=bool(shape_check.get("pass")),
    )
    payload = {
        **existing,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "failure_class": failure_class,
        "execution": {
            "run_id": args.run_id,
            "exit_code": int(args.exit_code),
            "started_at": args.started_at,
            "ended_at": args.ended_at,
            "elapsed_seconds": _safe_float(args.elapsed_seconds),
            "stdout_log": str(Path(args.stdout_log).resolve()) if args.stdout_log else None,
            "local_log_dir": str(run_dir) if run_dir else (str(local_log_dir) if local_log_dir else None),
            "python_executable": sys.executable,
            "device": config.get("device"),
            "amp_dtype": config.get("amp_dtype"),
            "save_run": False,
            "wandb": "off",
        },
        "training_config": _training_config_status(config, best_metrics),
        "gate_status": _gate_status(best_metrics),
        "checkpoint": {
            "path": str(checkpoint_path) if str(checkpoint_path) else None,
            "exists": bool(checkpoint_path and checkpoint_path.exists()),
        },
        "h4_forecast_shape": shape_check,
        "line_metrics": line_metrics,
        "bucket_line_metrics": bucket_metrics,
        "line_metrics_present": line_metrics_present,
        "smoke_interpretation": {
            "performance_judged": False,
            "note": "이번 CP는 1W LM 학습 파이프라인 smoke validation이며 성능 채택 판단이 아니다.",
            "failure_class_rule": {
                "split_data_failure": "preflight 또는 split/data/NaN/Inf 실패",
                "training_runtime_failure": "학습 프로세스 exit code 비정상",
                "metric_shape_failure": "line_metrics 또는 h4 forecast shape 실패",
            },
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_report(payload)
    return payload


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _metric_table(metrics: dict[str, Any]) -> str:
    rows = ["| metric | value |", "|---|---:|"]
    for key in LINE_KEYS:
        rows.append(f"| `{key}` | {_fmt(metrics.get(key), 6)} |")
    return "\n".join(rows)


def _bucket_table(bucket_metrics: dict[str, dict[str, Any]]) -> str:
    rows = []
    for bucket, metrics in bucket_metrics.items():
        rows.append(
            {
                "bucket": bucket,
                "ic_mean": _fmt(metrics.get("ic_mean"), 6),
                "long_short_spread": _fmt(metrics.get("long_short_spread"), 6),
                "false_safe_tail_rate": _fmt(metrics.get("false_safe_tail_rate"), 6),
                "severe_downside_recall": _fmt(metrics.get("severe_downside_recall"), 6),
            }
        )
    if not rows:
        return "bucket별 line_metrics는 생성되지 않았다. horizon=4라 `h1_h5`만 생성되는 것이 정상 범위다."
    return _table(
        rows,
        [
            ("bucket", "bucket"),
            ("ic_mean", "ic_mean"),
            ("long_short_spread", "long_short_spread"),
            ("false_safe_tail_rate", "false_safe_tail_rate"),
            ("severe_downside_recall", "severe_downside_recall"),
        ],
    )


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preflight = payload.get("preflight") or {}
    dataset_plan = preflight.get("dataset_plan") or {}
    split_ranges = preflight.get("split_ranges") or {}
    gate = payload.get("gate_status") or {}
    execution = payload.get("execution") or {}
    shape = payload.get("h4_forecast_shape") or {}
    report = [
        "# CP112-LM 1W 보수적 예측선 스모크",
        "",
        "## 요약",
        "",
        f"- 상태: `{payload.get('status')}` / 실패 분류: `{payload.get('failure_class')}`",
        f"- run_id: `{execution.get('run_id')}`",
        "- 범위: PatchTST 1W h4 `line_model` smoke validation. 성능 채택 판단이 아니라 학습 파이프라인 작동 확인이다.",
        "- `save-run`, DB write, inference 저장, W&B, band/composite/overlay 실험은 실행하지 않았다.",
        "",
        "## 로컬 스냅샷 확인",
        "",
        f"- provider/source: `{preflight.get('provider')}` / `{preflight.get('source')}`",
        f"- source_data_hash: `{preflight.get('source_data_hash')}`",
        f"- BM smoke hash와 동일: `{preflight.get('data_hash_matches_cp112_bm')}`",
        f"- feature_version: `{preflight.get('feature_version')}`",
        f"- snapshot price 존재: `{((preflight.get('snapshot_paths') or {}).get('price_exists'))}`",
        f"- snapshot indicators 존재: `{((preflight.get('snapshot_paths') or {}).get('indicators_exists'))}`",
        "",
        "## 데이터 게이트",
        "",
        f"- feature_set: `{preflight.get('feature_set')}`",
        f"- MODEL_FEATURE_COLUMNS: `{preflight.get('model_feature_columns_count')}`",
        f"- resolved feature columns: `{preflight.get('resolved_feature_columns_count')}`",
        f"- atr_ratio 모델 입력 포함: `{preflight.get('atr_ratio_in_model_features')}`",
        f"- feature NaN/Inf: `{preflight.get('feature_nonfinite_count')}`",
        f"- target NaN/Inf: `{preflight.get('target_nonfinite_count')}`",
        f"- eligible ticker count: `{dataset_plan.get('eligible_ticker_count')}`",
        "",
        "## split row 수",
        "",
        _table(
            [
                {
                    "split": split,
                    "rows": ranges.get("row_count"),
                    "min": ranges.get("min_asof_date"),
                    "max": ranges.get("max_asof_date"),
                }
                for split, ranges in split_ranges.items()
            ],
            [("split", "split"), ("rows", "rows"), ("min", "min"), ("max", "max")],
        ),
        "",
        "## 학습 실행",
        "",
        f"- exit code: `{execution.get('exit_code')}`",
        f"- elapsed seconds: `{execution.get('elapsed_seconds')}`",
        f"- device/amp: `{execution.get('device')}` / `{execution.get('amp_dtype')}`",
        f"- checkpoint_selection: `{gate.get('checkpoint_selection')}`",
        f"- line_gate pass: `{gate.get('line_gate_pass')}`",
        f"- gate_failed: `{gate.get('gate_failed')}`",
        f"- role: `{gate.get('role')}`",
        f"- h4 forecast shape pass: `{shape.get('pass')}` / shapes: `{shape.get('shapes')}`",
        "",
        "## line_metrics",
        "",
        _metric_table(payload.get("line_metrics") or {}),
        "",
        "## bucket line_metrics",
        "",
        _bucket_table(payload.get("bucket_line_metrics") or {}),
        "",
        "## 실패 분리",
        "",
        f"- split/data/shape 실패 여부: `{payload.get('failure_class')}`",
        "- line_metrics가 약하더라도 이번 CP에서는 성능 실패로 보지 않는다. 성능 평가는 후속 LM CP에서 별도로 해야 한다.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def run_preflight(_: argparse.Namespace) -> None:
    payload = build_preflight_payload()
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "preflight_path": str(PREFLIGHT_PATH), "preflight_gate_pass": payload["preflight_gate_pass"]}), ensure_ascii=False, indent=2))


def run_postflight(args: argparse.Namespace) -> None:
    payload = build_postflight_payload(args)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH), "status": payload.get("status"), "failure_class": payload.get("failure_class"), "gate_status": payload.get("gate_status"), "h4_forecast_shape": payload.get("h4_forecast_shape")}), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP112-LM 1W line smoke 사전/사후 검증")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=run_preflight)
    postflight = subparsers.add_parser("postflight")
    postflight.add_argument("--run-id", required=True)
    postflight.add_argument("--exit-code", type=int, required=True)
    postflight.add_argument("--started-at", default=None)
    postflight.add_argument("--ended-at", default=None)
    postflight.add_argument("--elapsed-seconds", default=None)
    postflight.add_argument("--stdout-log", default=None)
    postflight.add_argument("--local-log-dir", default=str(LOG_DIR / "train_local_logs"))
    postflight.set_defaults(func=run_postflight)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
