from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.inference import load_checkpoint  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from ai.train import apply_feature_columns_to_splits, resolve_feature_columns, summarize_dataset_plan  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp136_lm_1w_line_horizon_feature_expansion_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp136_lm_1w_line_horizon_feature_expansion_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp136_lm_1w_line_candidate_registry.json"
CSV_PATH = PROJECT_ROOT / "docs" / "cp136_lm_1w_line_experiment_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp136_lm_1w_line_horizon_feature_expansion_logs"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"
CP113_METRICS_PATH = PROJECT_ROOT / "docs" / "cp113_lm_1w_line_rescue_metrics.json"

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")

CANDIDATE_SPECS = [
    {
        "name": "h4_pvv_patch16_stride8_repro",
        "display_name": "1W h4 PVV PatchTST p16/s8 repro",
        "model": "patchtst",
        "horizon": 4,
        "feature_set": "price_volatility_volume",
        "patch_len": 16,
        "patch_stride": 8,
        "epochs": 3,
    },
    {
        "name": "h4_no_fundamentals_patch16_stride8",
        "display_name": "1W h4 no fundamentals PatchTST p16/s8",
        "model": "patchtst",
        "horizon": 4,
        "feature_set": "no_fundamentals",
        "patch_len": 16,
        "patch_stride": 8,
        "epochs": 3,
    },
    {
        "name": "h4_pvv_dense_stride4",
        "display_name": "1W h4 PVV PatchTST dense p16/s4",
        "model": "patchtst",
        "horizon": 4,
        "feature_set": "price_volatility_volume",
        "patch_len": 16,
        "patch_stride": 4,
        "epochs": 3,
    },
    {
        "name": "h4_tide_pvv",
        "display_name": "1W h4 PVV TiDE",
        "model": "tide",
        "horizon": 4,
        "feature_set": "price_volatility_volume",
        "patch_len": 16,
        "patch_stride": 8,
        "epochs": 3,
    },
    {
        "name": "h6_pvv_patch16_stride8",
        "display_name": "1W h6 PVV PatchTST p16/s8",
        "model": "patchtst",
        "horizon": 6,
        "feature_set": "price_volatility_volume",
        "patch_len": 16,
        "patch_stride": 8,
        "epochs": 2,
    },
    {
        "name": "h8_pvv_patch16_stride8",
        "display_name": "1W h8 PVV PatchTST p16/s8",
        "model": "patchtst",
        "horizon": 8,
        "feature_set": "price_volatility_volume",
        "patch_len": 16,
        "patch_stride": 8,
        "epochs": 2,
    },
    {
        "name": "h8_no_fundamentals_patch16_stride8",
        "display_name": "1W h8 no fundamentals PatchTST p16/s8",
        "model": "patchtst",
        "horizon": 8,
        "feature_set": "no_fundamentals",
        "patch_len": 16,
        "patch_stride": 8,
        "epochs": 2,
    },
]

OPTIONAL_CONTEXT_SPEC = {
    "name": "h4_context_light_patch16_stride8",
    "display_name": "1W h4 light context PatchTST p16/s8",
    "model": "patchtst",
    "horizon": 4,
    "feature_set": "context_light",
    "patch_len": 16,
    "patch_stride": 8,
    "epochs": 3,
}

LINE_KEYS = [
    "ic_mean",
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
]

HIGHER_IS_BETTER = {
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
    "severe_downside_recall",
}

LOWER_IS_BETTER = {
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "upside_sacrifice",
}


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _sha256_file(path: Path, *, short: int = 16) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:short]


def _snapshot_meta() -> dict[str, Any]:
    base = Path(os.environ.get("LENS_LOCAL_SNAPSHOT_DIR") or PROJECT_ROOT / "data" / "parquet")
    price = base / "price_data_yfinance_1W.parquet"
    indicators = base / "indicators_yfinance_1W.parquet"
    return {
        "base_dir": str(base),
        "price_path": str(price),
        "price_exists": price.exists(),
        "price_size_bytes": price.stat().st_size if price.exists() else None,
        "price_mtime": datetime.fromtimestamp(price.stat().st_mtime).isoformat(timespec="seconds") if price.exists() else None,
        "price_checksum": _sha256_file(price),
        "indicator_path": str(indicators),
        "indicator_exists": indicators.exists(),
        "indicator_size_bytes": indicators.stat().st_size if indicators.exists() else None,
        "indicator_mtime": datetime.fromtimestamp(indicators.stat().st_mtime).isoformat(timespec="seconds") if indicators.exists() else None,
        "context_checksum": _sha256_file(indicators),
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
            array = torch.from_numpy(bundle.ticker_arrays[str(ticker)]["features"]).to(dtype=torch.float32)
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
                first_failure = {"sample_index": int(sample_index), "ticker": str(ticker), "end_idx": int(end_idx)}
    return {"element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def _line_source(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    nested = metrics.get("line_metrics")
    if isinstance(nested, dict):
        return nested
    return metrics


def extract_line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = _line_source(metrics)
    return {key: source.get(key) for key in LINE_KEYS}


def extract_bucket_metrics(metrics: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    source = _line_source(metrics)
    buckets: dict[str, dict[str, Any]] = {}
    for prefix in ("h1_h5", "h6_h10", "h11_h20"):
        bucket = {key: source.get(f"{prefix}_{key}") for key in LINE_KEYS if source.get(f"{prefix}_{key}") is not None}
        if bucket:
            buckets[prefix] = bucket
    return buckets


def compare_metric(key: str, current: Any, baseline: Any) -> dict[str, Any]:
    left = _safe_float(current)
    right = _safe_float(baseline)
    if left is None or right is None:
        return {"delta": None, "direction": "unknown"}
    delta = left - right
    if abs(delta) < 1e-12:
        direction = "flat"
    elif key in HIGHER_IS_BETTER:
        direction = "improved" if delta > 0 else "worsened"
    elif key in LOWER_IS_BETTER:
        direction = "improved" if delta < 0 else "worsened"
    elif key == "conservative_bias":
        direction = "more_conservative" if delta < 0 else "less_conservative"
    else:
        direction = "changed"
    return {"delta": delta, "direction": direction}


def _feature_set_status(name: str) -> dict[str, Any]:
    try:
        columns = resolve_feature_columns(name)
        return {"exists": True, "feature_set": name, "column_count": len(columns), "columns": columns}
    except Exception as exc:
        return {"exists": False, "feature_set": name, "error": str(exc)}


def _cp113_best_reference() -> dict[str, Any]:
    if not CP113_METRICS_PATH.exists():
        return {"available": False}
    cp113 = _read_json(CP113_METRICS_PATH)
    best_name = (cp113.get("best_candidate") or {}).get("candidate")
    best_record = None
    for candidate in cp113.get("candidates") or []:
        if candidate.get("candidate") == best_name:
            best_record = candidate
            break
    if best_record is None:
        return {"available": False, "source": str(CP113_METRICS_PATH)}
    summary_path = Path(str(best_record.get("run_dir") or "")) / "summary.json"
    validation_line = {}
    test_line = best_record.get("line_metrics") or {}
    if summary_path.exists():
        summary = _read_json(summary_path)
        validation_line = extract_line_metrics(summary.get("best_metrics"))
        test_line = extract_line_metrics(summary.get("test_metrics"))
    return {
        "available": True,
        "source": str(CP113_METRICS_PATH),
        "candidate": best_record.get("candidate"),
        "run_id": best_record.get("run_id"),
        "source_data_hash": ((best_record.get("dataset_plan") or {}).get("source_data_hash")),
        "validation_line_metrics": validation_line,
        "test_line_metrics": test_line,
    }


def build_preflight_payload() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = resolve_data_fingerprint("1W", market_data_provider="yfinance")
    snapshot = _snapshot_meta()
    cp113_reference = _cp113_best_reference()
    feature_sets = {
        name: _feature_set_status(name)
        for name in sorted({"price_volatility_volume", "no_fundamentals", "context_light"})
    }
    horizons: dict[str, Any] = {}
    feature_nonfinite_total = 0
    target_nonfinite_total = 0
    for horizon in (4, 6, 8):
        train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
            timeframe="1W",
            seq_len=104,
            horizon=horizon,
            include_future_covariate=False,
            line_target_type="raw_future_return",
            band_target_type="raw_future_return",
            market_data_provider="yfinance",
        )
        # 최소 feature set인 PVV 기준으로 실제 후보 입력 shape도 확인한다.
        pvv_columns = resolve_feature_columns("price_volatility_volume")
        train_selected, val_selected, test_selected, _, _ = apply_feature_columns_to_splits(
            train_bundle,
            val_bundle,
            test_bundle,
            mean,
            std,
            pvv_columns,
        )
        split_finite = {
            "train": {
                "features": _feature_finite_summary(train_selected),
                "targets": _target_finite_summary(train_selected, horizon),
            },
            "val": {
                "features": _feature_finite_summary(val_selected),
                "targets": _target_finite_summary(val_selected, horizon),
            },
            "test": {
                "features": _feature_finite_summary(test_selected),
                "targets": _target_finite_summary(test_selected, horizon),
            },
        }
        feature_nonfinite = sum(int(split["features"]["nonfinite_count"]) for split in split_finite.values())
        target_nonfinite = sum(int(split["targets"]["nonfinite_count"]) for split in split_finite.values())
        feature_nonfinite_total += feature_nonfinite
        target_nonfinite_total += target_nonfinite
        plan_summary = summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle)
        horizons[f"h{horizon}"] = {
            "dataset_plan": plan_summary,
            "finite": split_finite,
            "feature_nonfinite_count": feature_nonfinite,
            "target_nonfinite_count": target_nonfinite,
            "test_exposure_count": int(plan_summary.get("test_samples") or 0) * horizon,
        }
    pass_gate = bool(
        snapshot.get("price_exists")
        and snapshot.get("indicator_exists")
        and source_hash
        and FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
        and len(MODEL_FEATURE_COLUMNS) == 36
        and feature_sets["price_volatility_volume"]["exists"]
        and feature_sets["no_fundamentals"]["exists"]
        and feature_nonfinite_total == 0
        and target_nonfinite_total == 0
    )
    payload = {
        "cp": "CP136-LM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PREFLIGHT_PASS" if pass_gate else "PREFLIGHT_FAIL",
        "preflight_gate_pass": pass_gate,
        "environment": {
            "python_executable": sys.executable,
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
            "composite": False,
            "frontend_modified": False,
            "live_yfinance_fetch": False,
            "eodhd_call": False,
        },
        "data": {
            "provider": "yfinance",
            "source": "yfinance",
            "source_data_hash": source_hash,
            "previous_cp113_source_data_hash": cp113_reference.get("source_data_hash"),
            "source_data_hash_changed_vs_cp113": source_hash != cp113_reference.get("source_data_hash"),
            "feature_version": FEATURE_CONTRACT_VERSION,
            "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
            "snapshot": snapshot,
            "feature_sets": feature_sets,
            "horizons": horizons,
            "feature_nonfinite_count": feature_nonfinite_total,
            "target_nonfinite_count": target_nonfinite_total,
        },
        "cp113_best_reference": cp113_reference,
    }
    _write_json(PREFLIGHT_PATH, payload)
    return payload


def _find_run_dir(candidate: str) -> Path | None:
    local_root = LOG_DIR / candidate / "train_local_logs"
    if not local_root.exists():
        return None
    run_dirs = [path for path in local_root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _shape_check(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"pass": False, "checkpoint_exists": False, "error": "checkpoint_missing"}
    try:
        model, checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint.get("config") or {}
        n_features = int(config.get("n_features") or len(config.get("feature_columns") or []))
        seq_len = int(config.get("seq_len") or 104)
        horizon = int(config.get("horizon") or 4)
        features = torch.randn(2, seq_len, n_features, dtype=torch.float32)
        ticker_id = torch.zeros(2, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            output = model(features, ticker_id=ticker_id)
        expected = [2, horizon]
        shapes = {
            "line": list(output.line.shape),
            "lower_band": list(output.lower_band.shape),
            "upper_band": list(output.upper_band.shape),
        }
        return {
            "pass": all(shape == expected for shape in shapes.values()),
            "checkpoint_exists": True,
            "expected_shape": expected,
            "shapes": shapes,
        }
    except Exception as exc:
        return {"pass": False, "checkpoint_exists": True, "error": str(exc)}


def _candidate_spec(name: str) -> dict[str, Any]:
    for spec in CANDIDATE_SPECS:
        if spec["name"] == name:
            return spec
    if name == OPTIONAL_CONTEXT_SPEC["name"]:
        return OPTIONAL_CONTEXT_SPEC
    raise KeyError(name)


def _load_candidate(spec: dict[str, Any], baseline_validation: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    name = spec["name"]
    process_path = LOG_DIR / name / "train_process.json"
    process = _read_json(process_path) if process_path.exists() else {}
    run_dir = _find_run_dir(name)
    if run_dir is None:
        return {
            "candidate": name,
            "status": "missing_run",
            "spec": spec,
            "process": process,
            "validation_line_metrics": {},
            "test_line_metrics": {},
            "comparison_vs_cp113_best_validation": {},
            "decision_basis": "validation",
            "classification": "rejected",
        }
    summary = _read_json(run_dir / "summary.json")
    config_payload = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    config = config_payload.get("config") if isinstance(config_payload.get("config"), dict) else {}
    best_metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
    test_metrics = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    validation_line = extract_line_metrics(best_metrics)
    test_line = extract_line_metrics(test_metrics)
    validation_buckets = extract_bucket_metrics(best_metrics)
    test_buckets = extract_bucket_metrics(test_metrics)
    checkpoint_path = Path(str(summary.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    shape = _shape_check(checkpoint_path) if str(checkpoint_path) else {"pass": False, "error": "checkpoint_path_missing"}
    horizon = int(config.get("horizon") or summary.get("horizon") or spec["horizon"])
    plan = summary.get("dataset_plan") if isinstance(summary.get("dataset_plan"), dict) else {}
    test_samples = int(plan.get("test_samples") or (preflight.get("data", {}).get("horizons", {}).get(f"h{horizon}", {}).get("dataset_plan", {}).get("test_samples") or 0))
    comparison = {
        key: {
            "current_validation": validation_line.get(key),
            "cp113_best_validation": baseline_validation.get(key),
            **compare_metric(key, validation_line.get(key), baseline_validation.get(key)),
        }
        for key in LINE_KEYS
    }
    status = "PASS" if int(process.get("exit_code") or 0) == 0 and shape.get("pass") and any(value is not None for value in validation_line.values()) else "FAIL"
    return {
        "candidate": name,
        "status": status,
        "run_id": summary.get("run_id") or process.get("run_id"),
        "spec": spec,
        "process": process,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path) if str(checkpoint_path) else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "shape_check": shape,
        "config": {
            "model": config.get("model") or summary.get("model"),
            "timeframe": config.get("timeframe") or summary.get("timeframe"),
            "horizon": horizon,
            "seq_len": config.get("seq_len"),
            "patch_len": config.get("patch_len"),
            "patch_stride": config.get("patch_stride"),
            "feature_set": config.get("feature_set") or summary.get("feature_set"),
            "line_target_type": config.get("line_target_type"),
            "checkpoint_selection": config.get("checkpoint_selection") or best_metrics.get("checkpoint_selection"),
            "ci_aggregate": config.get("ci_aggregate"),
            "alpha": config.get("alpha"),
            "beta": config.get("beta"),
            "delta": config.get("delta"),
            "n_features": config.get("n_features"),
            "use_future_covariate": config.get("use_future_covariate"),
        },
        "dataset_plan": plan,
        "test_exposure_count": test_samples * horizon,
        "gate_status": {
            "checkpoint_selection": best_metrics.get("checkpoint_selection"),
            "gate_type": best_metrics.get("gate_type"),
            "gate_failed": best_metrics.get("gate_failed"),
            "line_gate_pass": best_metrics.get("line_gate_pass"),
            "role": best_metrics.get("role"),
            "best_epoch": best_metrics.get("best_epoch"),
        },
        "decision_basis": "validation",
        "validation_line_metrics": validation_line,
        "test_line_metrics": test_line,
        "validation_bucket_metrics": validation_buckets,
        "test_bucket_metrics": test_buckets,
        "comparison_vs_cp113_best_validation": comparison,
    }


def _product_ok(candidate: dict[str, Any]) -> bool:
    config = candidate.get("config") or {}
    horizon = int(config.get("horizon") or 0)
    metrics = candidate.get("validation_line_metrics") or {}
    gate = bool((candidate.get("gate_status") or {}).get("line_gate_pass"))
    bias = _safe_float(metrics.get("conservative_bias"))
    return bool(
        candidate.get("status") == "PASS"
        and gate
        and horizon in {4, 6}
        and (_safe_float(metrics.get("ic_mean")) or -999.0) > 0
        and (_safe_float(metrics.get("long_short_spread")) or -999.0) > 0
        and (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) > 0
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.20
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.75
        and (bias is None or abs(bias) <= 0.25)
    )


def _risk_ok(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("validation_line_metrics") or {}
    gate = bool((candidate.get("gate_status") or {}).get("line_gate_pass"))
    return bool(
        candidate.get("status") == "PASS"
        and gate
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.80
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.20
    )


def _h8_watch(candidate: dict[str, Any]) -> bool:
    config = candidate.get("config") or {}
    if int(config.get("horizon") or 0) != 8:
        return False
    metrics = candidate.get("validation_line_metrics") or {}
    gate = bool((candidate.get("gate_status") or {}).get("line_gate_pass"))
    return bool(
        candidate.get("status") == "PASS"
        and gate
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.25
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.75
    )


def _select_default(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    product = [candidate for candidate in candidates if _product_ok(candidate)]
    if not product:
        return None
    return sorted(
        product,
        key=lambda candidate: (
            1 if (candidate.get("config") or {}).get("feature_set") == "price_volatility_volume" else 0,
            1 if int((candidate.get("config") or {}).get("horizon") or 0) == 4 else 0,
            1 if (_safe_float((candidate.get("validation_line_metrics") or {}).get("false_safe_tail_rate")) or 999.0) <= 0.10 else 0,
            1 if (_safe_float((candidate.get("validation_line_metrics") or {}).get("severe_downside_recall")) or -999.0) >= 0.85 else 0,
            _safe_float((candidate.get("validation_line_metrics") or {}).get("ic_mean")) or -999.0,
            _safe_float((candidate.get("validation_line_metrics") or {}).get("fee_adjusted_return")) or -999.0,
            _safe_float((candidate.get("validation_line_metrics") or {}).get("long_short_spread")) or -999.0,
            -(_safe_float((candidate.get("validation_line_metrics") or {}).get("false_safe_tail_rate")) or 999.0),
            _safe_float((candidate.get("validation_line_metrics") or {}).get("severe_downside_recall")) or -999.0,
        ),
        reverse=True,
    )[0]


def _strength_summary(candidate: dict[str, Any]) -> str:
    metrics = candidate.get("validation_line_metrics") or {}
    return (
        f"validation IC {_fmt(metrics.get('ic_mean'))}, spread {_fmt(metrics.get('long_short_spread'))}, "
        f"false_safe_tail {_fmt(metrics.get('false_safe_tail_rate'))}, severe_recall {_fmt(metrics.get('severe_downside_recall'))}"
    )


def _weakness_summary(candidate: dict[str, Any]) -> str:
    metrics = candidate.get("validation_line_metrics") or {}
    weak = []
    if (_safe_float(metrics.get("ic_mean")) or -999.0) <= 0:
        weak.append("IC가 양수가 아님")
    if (_safe_float(metrics.get("long_short_spread")) or -999.0) <= 0:
        weak.append("spread가 양수가 아님")
    if (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) > 0.20:
        weak.append("false_safe_tail 기준 초과")
    if (_safe_float(metrics.get("severe_downside_recall")) or -999.0) < 0.75:
        weak.append("severe recall 기준 미달")
    if not weak:
        return "제품 후보 기준의 주요 약점은 smoke 범위에서 확인되지 않음"
    return ", ".join(weak)


def _why_not_default(candidate: dict[str, Any], default_name: str | None) -> str:
    if candidate.get("candidate") == default_name:
        return "recommended_default"
    if candidate.get("classification") == "design_needed":
        return "feature_set 정의가 없어 실행하지 않음"
    if _product_ok(candidate):
        return "제품 기준은 통과했지만 기본 후보보다 feature/horizon 해석성 또는 안정성이 낮음"
    if _h8_watch(candidate):
        return "h8은 제품 기본 후보가 아니라 중기 참고선 후보"
    if _risk_ok(candidate):
        return "수익 방향선보다 위험/보수선 성격"
    return _weakness_summary(candidate)


def _classify_candidates(candidates: list[dict[str, Any]], preflight: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    default = _select_default(candidates)
    default_name = default.get("candidate") if default else None
    registry = []
    for candidate in candidates:
        status = candidate.get("status")
        if candidate.get("classification") == "design_needed":
            classification = "design_needed"
        elif status != "PASS":
            classification = "rejected"
        elif candidate.get("candidate") == default_name:
            classification = "recommended_default"
        elif _product_ok(candidate):
            classification = "selectable_verified"
        elif _h8_watch(candidate) or _risk_ok(candidate):
            classification = "visual_or_risk_watch"
        elif int((candidate.get("config") or {}).get("horizon") or 0) == 8:
            classification = "experiment_record"
        else:
            classification = "rejected"
        candidate["classification"] = classification
        registry.append(_registry_entry(candidate, classification, default_name, preflight))
    return candidates, registry


def _registry_entry(candidate: dict[str, Any], classification: str, default_name: str | None, preflight: dict[str, Any]) -> dict[str, Any]:
    spec = candidate.get("spec") or {}
    config = candidate.get("config") or {}
    validation = candidate.get("validation_line_metrics") or {}
    test = candidate.get("test_line_metrics") or {}
    return {
        "candidate_id": candidate.get("candidate"),
        "display_name": spec.get("display_name") or candidate.get("candidate"),
        "version_label": f"cp136_{candidate.get('candidate')}",
        "classification": classification,
        "role": "line_model",
        "timeframe": "1W",
        "horizon": config.get("horizon") or spec.get("horizon"),
        "model_family": config.get("model") or spec.get("model"),
        "feature_set": config.get("feature_set") or spec.get("feature_set"),
        "line_loss": {
            "class": "AsymmetricHuberLoss",
            "alpha": config.get("alpha"),
            "beta": config.get("beta"),
            "delta": config.get("delta"),
            "target": "raw_future_return",
        },
        "validation_stability": {
            "line_gate_pass": (candidate.get("gate_status") or {}).get("line_gate_pass"),
            "basis": "validation_line_metrics",
            "status": classification,
        },
        "test_exposure_count": candidate.get("test_exposure_count"),
        "strength_summary": _strength_summary(candidate) if classification != "design_needed" else "미실행 후보",
        "weakness_summary": _weakness_summary(candidate) if classification != "design_needed" else "feature_set 정의 필요",
        "why_not_default": _why_not_default(candidate, default_name),
        "key_metrics": {
            "validation": validation,
            "test": test,
        },
        "source_data_hash": (preflight.get("data") or {}).get("source_data_hash"),
        "context_checksum": ((preflight.get("data") or {}).get("snapshot") or {}).get("context_checksum"),
    }


def _design_needed_candidate(preflight: dict[str, Any]) -> dict[str, Any] | None:
    feature_status = ((preflight.get("data") or {}).get("feature_sets") or {}).get("context_light") or {}
    if feature_status.get("exists"):
        return None
    spec = dict(OPTIONAL_CONTEXT_SPEC)
    return {
        "candidate": spec["name"],
        "status": "not_run",
        "spec": spec,
        "classification": "design_needed",
        "decision_basis": "feature_set_definition",
        "design_note": "context_light feature_set이 현재 cp63 feature_set plan에 없어 구현하지 않고 design_needed로 기록했다.",
        "validation_line_metrics": {},
        "test_line_metrics": {},
        "test_exposure_count": None,
    }


def build_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.exists() else build_preflight_payload()
    baseline = ((preflight.get("cp113_best_reference") or {}).get("validation_line_metrics") or {})
    candidates = [_load_candidate(spec, baseline, preflight) for spec in CANDIDATE_SPECS]
    optional = _design_needed_candidate(preflight)
    if optional is not None:
        candidates.append(optional)
    candidates, registry = _classify_candidates(candidates, preflight)
    default = next((candidate for candidate in candidates if candidate.get("classification") == "recommended_default"), None)
    payload = {
        "cp": "CP136-LM",
        "purpose": "1W line horizon / feature / model expansion",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if all(candidate.get("status") in {"PASS", "not_run"} for candidate in candidates) else "PARTIAL_OR_FAIL",
        "preflight": preflight,
        "candidate_registry_path": str(REGISTRY_PATH),
        "experiment_summary_csv_path": str(CSV_PATH),
        "candidates": candidates,
        "summary_decision": {
            "recommended_default": default.get("candidate") if default else None,
            "recommended_default_run_id": default.get("run_id") if default else None,
            "selection_basis": "validation_line_metrics",
            "test_used_for_selection": False,
            "product_candidate_available": default is not None,
            "h8_visual_or_risk_watch": [
                candidate.get("candidate")
                for candidate in candidates
                if int((candidate.get("config") or candidate.get("spec") or {}).get("horizon") or 0) == 8
                and candidate.get("classification") == "visual_or_risk_watch"
            ],
            "next_cp_recommendation": _next_cp_recommendation(default),
        },
    }
    return payload, registry


def _next_cp_recommendation(default: dict[str, Any] | None) -> str:
    if default is None:
        return "1W line은 저장 후보로 올리지 말고 pvv 기반 loss/selector 개선 CP를 먼저 연다."
    return f"{default.get('candidate')}를 같은 CP133 이후 snapshot/hash에서 save-run 전용 재현 CP로 승격한다."


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        metrics = candidate.get("validation_line_metrics") or {}
        comparison = candidate.get("comparison_vs_cp113_best_validation") or {}
        config = candidate.get("config") or candidate.get("spec") or {}
        rows.append(
            {
                "candidate": candidate.get("candidate"),
                "class": candidate.get("classification"),
                "run_id": candidate.get("run_id"),
                "model": config.get("model"),
                "h": config.get("horizon"),
                "feature_set": config.get("feature_set"),
                "gate": (candidate.get("gate_status") or {}).get("line_gate_pass"),
                "ic": _fmt(metrics.get("ic_mean")),
                "d_ic": _fmt((comparison.get("ic_mean") or {}).get("delta")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "severe": _fmt(metrics.get("severe_downside_recall")),
                "fee": _fmt(metrics.get("fee_adjusted_return")),
            }
        )
    return rows


def _comparison_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = candidate.get("comparison_vs_cp113_best_validation") or {}
    rows = []
    for key in LINE_KEYS:
        item = comparison.get(key) or {}
        rows.append(
            {
                "metric": key,
                "cp113_best_val": _fmt(item.get("cp113_best_validation")),
                "current_val": _fmt(item.get("current_validation")),
                "delta": _fmt(item.get("delta")),
                "direction": item.get("direction"),
            }
        )
    return rows


def _bucket_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        for bucket, metrics in (candidate.get("validation_bucket_metrics") or {}).items():
            rows.append(
                {
                    "candidate": candidate.get("candidate"),
                    "bucket": bucket,
                    "ic": _fmt(metrics.get("ic_mean")),
                    "spread": _fmt(metrics.get("long_short_spread")),
                    "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                    "severe": _fmt(metrics.get("severe_downside_recall")),
                }
            )
    return rows


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preflight = payload.get("preflight") or {}
    data = preflight.get("data") or {}
    snapshot = data.get("snapshot") or {}
    candidates = payload.get("candidates") or []
    decision = payload.get("summary_decision") or {}
    report = [
        "# CP136-LM 1W Line Horizon / Feature Expansion 재실험",
        "",
        "## 요약",
        "",
        f"- 상태: `{payload.get('status')}`",
        f"- recommended_default: `{decision.get('recommended_default')}` / run_id `{decision.get('recommended_default_run_id')}`",
        "- 후보 선택은 validation line_metrics 기준이며 test는 직접 선택에 쓰지 않았다.",
        "- save-run, DB write, inference 저장, W&B, composite, 프론트 수정, live yfinance fetch, EODHD 호출은 하지 않았다.",
        "",
        "## 데이터 기준",
        "",
        f"- source_data_hash: `{data.get('source_data_hash')}`",
        f"- CP113 대비 source hash 변경: `{data.get('source_data_hash_changed_vs_cp113')}`",
        f"- context checksum: `{snapshot.get('context_checksum')}`",
        f"- indicator snapshot mtime: `{snapshot.get('indicator_mtime')}`",
        f"- feature/target NaN/Inf: `{data.get('feature_nonfinite_count')}` / `{data.get('target_nonfinite_count')}`",
        f"- context_light feature_set: `{((data.get('feature_sets') or {}).get('context_light') or {}).get('exists')}`",
        "",
        "## 후보 결과",
        "",
        _table(
            _candidate_rows(candidates),
            [
                ("candidate", "candidate"),
                ("class", "class"),
                ("run_id", "run_id"),
                ("model", "model"),
                ("h", "h"),
                ("feature_set", "feature_set"),
                ("gate", "gate"),
                ("val_ic", "ic"),
                ("d_ic", "d_ic"),
                ("val_spread", "spread"),
                ("val_false_safe", "false_safe"),
                ("val_severe", "severe"),
                ("val_fee", "fee"),
            ],
        ),
        "",
        "## CP113 best 대비 validation 개선/악화",
        "",
    ]
    for candidate in candidates:
        if candidate.get("classification") == "design_needed":
            report.extend([f"### {candidate.get('candidate')}", "", candidate.get("design_note", ""), ""])
            continue
        report.extend(
            [
                f"### {candidate.get('candidate')}",
                "",
                _table(
                    _comparison_rows(candidate),
                    [("metric", "metric"), ("CP113 best val", "cp113_best_val"), ("current val", "current_val"), ("delta", "delta"), ("판정", "direction")],
                ),
                "",
            ]
        )
    bucket_rows = _bucket_rows(candidates)
    report.extend(
        [
            "## horizon bucket metrics",
            "",
            _table(
                bucket_rows,
                [
                    ("candidate", "candidate"),
                    ("bucket", "bucket"),
                    ("ic_mean", "ic"),
                    ("long_short_spread", "spread"),
                    ("false_safe_tail_rate", "false_safe"),
                    ("severe_downside_recall", "severe"),
                ],
            )
            if bucket_rows
            else "bucket metrics 없음",
            "",
            "## 제품 판단",
            "",
            f"- 기본 후보: `{decision.get('recommended_default')}`",
            f"- h8 visual/risk watch: `{decision.get('h8_visual_or_risk_watch')}`",
            "- full_features는 이번 CP에서 기본 후보로 가정하지 않았고 실행하지 않았다. 최소 feature group인 pvv를 우선했다.",
            f"- 다음 CP 추천: {decision.get('next_cp_recommendation')}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def _write_registry(registry: list[dict[str, Any]]) -> None:
    _write_json(REGISTRY_PATH, {"cp": "CP136-LM", "generated_at": datetime.now().isoformat(timespec="seconds"), "candidates": registry})


def _write_csv(candidates: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "classification",
        "run_id",
        "model",
        "horizon",
        "feature_set",
        "patch_len",
        "patch_stride",
        "line_gate_pass",
        "test_exposure_count",
        "val_ic_mean",
        "val_long_short_spread",
        "val_fee_adjusted_return",
        "val_false_safe_tail_rate",
        "val_severe_downside_recall",
        "val_conservative_bias",
        "test_ic_mean",
        "test_long_short_spread",
        "test_false_safe_tail_rate",
        "test_severe_downside_recall",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            config = candidate.get("config") or candidate.get("spec") or {}
            val = candidate.get("validation_line_metrics") or {}
            test = candidate.get("test_line_metrics") or {}
            writer.writerow(
                {
                    "candidate": candidate.get("candidate"),
                    "classification": candidate.get("classification"),
                    "run_id": candidate.get("run_id"),
                    "model": config.get("model"),
                    "horizon": config.get("horizon"),
                    "feature_set": config.get("feature_set"),
                    "patch_len": config.get("patch_len"),
                    "patch_stride": config.get("patch_stride"),
                    "line_gate_pass": (candidate.get("gate_status") or {}).get("line_gate_pass"),
                    "test_exposure_count": candidate.get("test_exposure_count"),
                    "val_ic_mean": val.get("ic_mean"),
                    "val_long_short_spread": val.get("long_short_spread"),
                    "val_fee_adjusted_return": val.get("fee_adjusted_return"),
                    "val_false_safe_tail_rate": val.get("false_safe_tail_rate"),
                    "val_severe_downside_recall": val.get("severe_downside_recall"),
                    "val_conservative_bias": val.get("conservative_bias"),
                    "test_ic_mean": test.get("ic_mean"),
                    "test_long_short_spread": test.get("long_short_spread"),
                    "test_false_safe_tail_rate": test.get("false_safe_tail_rate"),
                    "test_severe_downside_recall": test.get("severe_downside_recall"),
                }
            )


def run_preflight(_: argparse.Namespace) -> None:
    payload = build_preflight_payload()
    _write_json(PREFLIGHT_PATH, payload)
    print(json.dumps(_json_safe({"preflight_path": str(PREFLIGHT_PATH), "preflight_gate_pass": payload["preflight_gate_pass"], "source_data_hash": payload["data"]["source_data_hash"], "context_checksum": payload["data"]["snapshot"]["context_checksum"]}), ensure_ascii=False, indent=2))


def run_report(_: argparse.Namespace) -> None:
    payload, registry = build_payload()
    _write_json(METRICS_PATH, payload)
    _write_registry(registry)
    _write_csv(payload.get("candidates") or [])
    _write_report(payload)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "registry_path": str(REGISTRY_PATH), "csv_path": str(CSV_PATH), "report_path": str(REPORT_PATH), "status": payload.get("status"), "summary_decision": payload.get("summary_decision")}), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP136-LM 1W line horizon/feature expansion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=run_preflight)
    report = subparsers.add_parser("report")
    report.set_defaults(func=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
