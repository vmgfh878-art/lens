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


REPORT_PATH = PROJECT_ROOT / "docs" / "cp137_lm_1w_line_conservatism_loss_study_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp137_lm_1w_line_conservatism_loss_study_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp137_lm_1w_line_loss_candidate_registry.json"
CSV_PATH = PROJECT_ROOT / "docs" / "cp137_lm_1w_line_loss_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp137_lm_1w_line_conservatism_loss_study_logs"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"
CP136_METRICS_PATH = PROJECT_ROOT / "docs" / "cp136_lm_1w_line_horizon_feature_expansion_metrics.json"

EXPECTED_SOURCE_HASH = "13a7f83d"
EXPECTED_CONTEXT_CHECKSUM = "ecb532122fca5eee"

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")

CANDIDATE_SPECS = [
    {"name": "beta_1p5", "display_name": "1W h4 PVV PatchTST beta 1.5", "beta": 1.5, "optional": False},
    {"name": "beta_2p0", "display_name": "1W h4 PVV PatchTST beta 2.0 repro", "beta": 2.0, "optional": False},
    {"name": "beta_2p5", "display_name": "1W h4 PVV PatchTST beta 2.5", "beta": 2.5, "optional": False},
    {"name": "beta_3p0", "display_name": "1W h4 PVV PatchTST beta 3.0", "beta": 3.0, "optional": True},
]

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
    "direction_accuracy",
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
    "direction_accuracy",
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


def _normalize_bucket_labels(buckets: dict[str, dict[str, Any]], horizon: int) -> dict[str, dict[str, Any]]:
    if horizon <= 4 and "h1_h5" in buckets:
        normalized = dict(buckets)
        normalized["h1_h4"] = normalized.pop("h1_h5")
        return normalized
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


def _cp136_reference() -> dict[str, Any]:
    cp136 = _read_json(CP136_METRICS_PATH)
    default_name = (cp136.get("summary_decision") or {}).get("recommended_default")
    record = None
    for candidate in cp136.get("candidates") or []:
        if candidate.get("candidate") == default_name:
            record = candidate
            break
    if record is None:
        raise RuntimeError("CP136 recommended_default 후보를 찾지 못했습니다.")
    return {
        "candidate": default_name,
        "run_id": record.get("run_id"),
        "source_data_hash": ((cp136.get("preflight") or {}).get("data") or {}).get("source_data_hash"),
        "context_checksum": ((((cp136.get("preflight") or {}).get("data") or {}).get("snapshot") or {}).get("context_checksum")),
        "validation_line_metrics": record.get("validation_line_metrics") or {},
        "test_line_metrics": record.get("test_line_metrics") or {},
    }


def build_preflight_payload() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot_meta()
    source_hash = resolve_data_fingerprint("1W", market_data_provider="yfinance")
    cp136_reference = _cp136_reference()
    feature_columns = resolve_feature_columns("price_volatility_volume")
    train_bundle, val_bundle, test_bundle, mean, std, plan = prepare_dataset_splits(
        timeframe="1W",
        seq_len=104,
        horizon=4,
        include_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        market_data_provider="yfinance",
    )
    train_selected, val_selected, test_selected, _, _ = apply_feature_columns_to_splits(
        train_bundle,
        val_bundle,
        test_bundle,
        mean,
        std,
        feature_columns,
    )
    finite = {
        "train": {
            "features": _feature_finite_summary(train_selected),
            "targets": _target_finite_summary(train_selected, 4),
        },
        "val": {
            "features": _feature_finite_summary(val_selected),
            "targets": _target_finite_summary(val_selected, 4),
        },
        "test": {
            "features": _feature_finite_summary(test_selected),
            "targets": _target_finite_summary(test_selected, 4),
        },
    }
    feature_nonfinite = sum(int(split["features"]["nonfinite_count"]) for split in finite.values())
    target_nonfinite = sum(int(split["targets"]["nonfinite_count"]) for split in finite.values())
    plan_summary = summarize_dataset_plan(plan, train_bundle, val_bundle, test_bundle)
    pass_gate = bool(
        source_hash == EXPECTED_SOURCE_HASH
        and snapshot.get("context_checksum") == EXPECTED_CONTEXT_CHECKSUM
        and source_hash == cp136_reference.get("source_data_hash")
        and snapshot.get("context_checksum") == cp136_reference.get("context_checksum")
        and FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
        and len(MODEL_FEATURE_COLUMNS) == 36
        and "atr_ratio" not in MODEL_FEATURE_COLUMNS
        and len(feature_columns) == 11
        and feature_nonfinite == 0
        and target_nonfinite == 0
    )
    payload = {
        "cp": "CP137-LM",
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
            "expected_source_data_hash": EXPECTED_SOURCE_HASH,
            "source_hash_matches_cp136": source_hash == cp136_reference.get("source_data_hash"),
            "feature_version": FEATURE_CONTRACT_VERSION,
            "snapshot": snapshot,
            "expected_context_checksum": EXPECTED_CONTEXT_CHECKSUM,
            "context_checksum_matches_cp136": snapshot.get("context_checksum") == cp136_reference.get("context_checksum"),
            "feature_set": "price_volatility_volume",
            "feature_columns": feature_columns,
            "feature_column_count": len(feature_columns),
            "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
            "finite": finite,
            "feature_nonfinite_count": feature_nonfinite,
            "target_nonfinite_count": target_nonfinite,
            "dataset_plan": plan_summary,
            "test_exposure_count": int(plan_summary.get("test_samples") or 0) * 4,
        },
        "cp136_reference": cp136_reference,
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


def _load_candidate(spec: dict[str, Any], cp136_validation: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    name = spec["name"]
    process_path = LOG_DIR / name / "train_process.json"
    process = _read_json(process_path) if process_path.exists() else {}
    run_dir = _find_run_dir(name)
    if run_dir is None:
        status = "omitted" if spec.get("optional") else "missing_run"
        return {
            "candidate": name,
            "status": status,
            "spec": spec,
            "process": process,
            "validation_line_metrics": {},
            "test_line_metrics": {},
            "comparison_vs_cp136_validation": {},
            "classification": "experiment_record" if spec.get("optional") else "rejected",
            "omission_reason": process.get("omission_reason"),
        }
    summary = _read_json(run_dir / "summary.json")
    config_payload = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    config = config_payload.get("config") if isinstance(config_payload.get("config"), dict) else {}
    best_metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
    test_metrics = summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}
    validation_line = extract_line_metrics(best_metrics)
    test_line = extract_line_metrics(test_metrics)
    horizon = int(config.get("horizon") or summary.get("horizon") or 4)
    validation_buckets = _normalize_bucket_labels(extract_bucket_metrics(best_metrics), horizon)
    test_buckets = _normalize_bucket_labels(extract_bucket_metrics(test_metrics), horizon)
    checkpoint_path = Path(str(summary.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    shape = _shape_check(checkpoint_path) if str(checkpoint_path) else {"pass": False, "error": "checkpoint_path_missing"}
    plan = summary.get("dataset_plan") if isinstance(summary.get("dataset_plan"), dict) else {}
    comparison = {
        key: {
            "current_validation": validation_line.get(key),
            "cp136_validation": cp136_validation.get(key),
            **compare_metric(key, validation_line.get(key), cp136_validation.get(key)),
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
        },
        "dataset_plan": plan,
        "test_exposure_count": int(plan.get("test_samples") or ((preflight.get("data") or {}).get("dataset_plan") or {}).get("test_samples") or 0) * 4,
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
        "comparison_vs_cp136_validation": comparison,
    }


def _product_ok(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("validation_line_metrics") or {}
    gate = bool((candidate.get("gate_status") or {}).get("line_gate_pass"))
    bias = _safe_float(metrics.get("conservative_bias"))
    return bool(
        candidate.get("status") == "PASS"
        and gate
        and (_safe_float(metrics.get("ic_mean")) or -999.0) > 0
        and (_safe_float(metrics.get("long_short_spread")) or -999.0) > 0
        and (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) > 0
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.20
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.75
        and (bias is None or bias >= -0.25)
        and (_safe_float(metrics.get("upside_sacrifice")) or 999.0) <= 0.30
    )


def _strong_product(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("validation_line_metrics") or {}
    return bool(
        _product_ok(candidate)
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.10
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.85
    )


def _risk_variant(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("validation_line_metrics") or {}
    gate = bool((candidate.get("gate_status") or {}).get("line_gate_pass"))
    rank_collapsed = (_safe_float(metrics.get("ic_mean")) or -999.0) <= 0 and (_safe_float(metrics.get("long_short_spread")) or -999.0) <= 0
    return bool(
        candidate.get("status") == "PASS"
        and gate
        and not rank_collapsed
        and (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) <= 0.10
        and (_safe_float(metrics.get("severe_downside_recall")) or -999.0) >= 0.85
    )


def _select_default(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    product = [candidate for candidate in candidates if _product_ok(candidate)]
    if not product:
        return None
    return sorted(
        product,
        key=lambda candidate: (
            1 if _strong_product(candidate) else 0,
            _safe_float((candidate.get("validation_line_metrics") or {}).get("ic_mean")) or -999.0,
            _safe_float((candidate.get("validation_line_metrics") or {}).get("long_short_spread")) or -999.0,
            _safe_float((candidate.get("validation_line_metrics") or {}).get("fee_adjusted_return")) or -999.0,
            -(_safe_float((candidate.get("validation_line_metrics") or {}).get("false_safe_tail_rate")) or 999.0),
            _safe_float((candidate.get("validation_line_metrics") or {}).get("severe_downside_recall")) or -999.0,
        ),
        reverse=True,
    )[0]


def _classify(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default = _select_default(candidates)
    default_name = default.get("candidate") if default else None
    for candidate in candidates:
        if candidate.get("status") in {"omitted", "missing_run"}:
            candidate["classification"] = "experiment_record" if candidate.get("status") == "omitted" else "rejected"
        elif candidate.get("status") != "PASS":
            candidate["classification"] = "rejected"
        elif candidate.get("candidate") == default_name:
            candidate["classification"] = "recommended_default"
        elif _product_ok(candidate):
            candidate["classification"] = "selectable_verified"
        elif _risk_variant(candidate):
            candidate["classification"] = "risk_conservative_variant"
        elif bool((candidate.get("gate_status") or {}).get("line_gate_pass")):
            candidate["classification"] = "experiment_record"
        else:
            candidate["classification"] = "rejected"
    return candidates


def _strength_summary(candidate: dict[str, Any]) -> str:
    metrics = candidate.get("validation_line_metrics") or {}
    return (
        f"validation IC {_fmt(metrics.get('ic_mean'))}, spread {_fmt(metrics.get('long_short_spread'))}, "
        f"false_safe {_fmt(metrics.get('false_safe_tail_rate'))}, severe_recall {_fmt(metrics.get('severe_downside_recall'))}"
    )


def _weakness_summary(candidate: dict[str, Any]) -> str:
    if candidate.get("status") == "omitted":
        return candidate.get("omission_reason") or "optional candidate omitted"
    metrics = candidate.get("validation_line_metrics") or {}
    weak = []
    if (_safe_float(metrics.get("ic_mean")) or -999.0) <= 0:
        weak.append("IC 미달")
    if (_safe_float(metrics.get("long_short_spread")) or -999.0) <= 0:
        weak.append("spread 미달")
    if (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) <= 0:
        weak.append("fee return 미달")
    if (_safe_float(metrics.get("false_safe_tail_rate")) or 999.0) > 0.20:
        weak.append("false_safe 최대 기준 초과")
    if (_safe_float(metrics.get("severe_downside_recall")) or -999.0) < 0.75:
        weak.append("severe recall 미달")
    if (_safe_float(metrics.get("upside_sacrifice")) or 999.0) > 0.30:
        weak.append("upside sacrifice 과다")
    return ", ".join(weak) if weak else "주요 약점 없음"


def _why_not_default(candidate: dict[str, Any], default_name: str | None) -> str:
    if candidate.get("candidate") == default_name:
        return "recommended_default"
    if candidate.get("status") == "omitted":
        return candidate.get("omission_reason") or "optional omitted"
    if _product_ok(candidate):
        return "제품 기준은 통과했으나 default보다 validation 수익 추종성 또는 tradeoff가 낮음"
    if _risk_variant(candidate):
        return "risk 보수성은 좋지만 제품 기본 수익 line 기준은 부족함"
    return _weakness_summary(candidate)


def _registry_entry(candidate: dict[str, Any], default_name: str | None, preflight: dict[str, Any]) -> dict[str, Any]:
    spec = candidate.get("spec") or {}
    config = candidate.get("config") or {}
    validation = candidate.get("validation_line_metrics") or {}
    test = candidate.get("test_line_metrics") or {}
    return {
        "candidate_id": candidate.get("candidate"),
        "display_name": spec.get("display_name") or candidate.get("candidate"),
        "version_label": f"cp137_{candidate.get('candidate')}",
        "classification": candidate.get("classification"),
        "role": "line_model",
        "timeframe": "1W",
        "horizon": config.get("horizon") or 4,
        "model_family": config.get("model") or "patchtst",
        "feature_set": config.get("feature_set") or "price_volatility_volume",
        "line_loss": {
            "class": "AsymmetricHuberLoss",
            "alpha": config.get("alpha") or 1.0,
            "beta": config.get("beta") or spec.get("beta"),
            "delta": config.get("delta") or 1.0,
            "target": "raw_future_return",
        },
        "validation_stability": {
            "line_gate_pass": (candidate.get("gate_status") or {}).get("line_gate_pass"),
            "basis": "validation_line_metrics",
            "status": candidate.get("classification"),
        },
        "test_exposure_count": candidate.get("test_exposure_count"),
        "strength_summary": _strength_summary(candidate) if candidate.get("status") != "omitted" else "미실행 optional 후보",
        "weakness_summary": _weakness_summary(candidate),
        "why_not_default": _why_not_default(candidate, default_name),
        "key_metrics": {
            "validation": validation,
            "test": test,
        },
        "source_data_hash": ((preflight.get("data") or {}).get("source_data_hash")),
        "context_checksum": (((preflight.get("data") or {}).get("snapshot") or {}).get("context_checksum")),
    }


def build_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.exists() else build_preflight_payload()
    cp136_validation = ((preflight.get("cp136_reference") or {}).get("validation_line_metrics") or {})
    candidates = [_load_candidate(spec, cp136_validation, preflight) for spec in CANDIDATE_SPECS]
    candidates = _classify(candidates)
    default = next((candidate for candidate in candidates if candidate.get("classification") == "recommended_default"), None)
    default_name = default.get("candidate") if default else None
    registry = [_registry_entry(candidate, default_name, preflight) for candidate in candidates]
    payload = {
        "cp": "CP137-LM",
        "purpose": "1W line conservatism / beta loss study",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if all(candidate.get("status") in {"PASS", "omitted"} for candidate in candidates) else "PARTIAL_OR_FAIL",
        "preflight": preflight,
        "candidate_registry_path": str(REGISTRY_PATH),
        "loss_summary_csv_path": str(CSV_PATH),
        "candidates": candidates,
        "summary_decision": {
            "recommended_default": default_name,
            "recommended_default_run_id": default.get("run_id") if default else None,
            "selection_basis": "validation_line_metrics",
            "test_used_for_selection": False,
            "beta_tradeoff_summary": _tradeoff_summary(candidates, default_name),
            "next_cp_recommendation": _next_cp_recommendation(default),
        },
    }
    return payload, registry


def _tradeoff_summary(candidates: list[dict[str, Any]], default_name: str | None) -> str:
    if default_name is None:
        return "제품 기준을 통과한 beta 후보가 없다."
    default = next(candidate for candidate in candidates if candidate.get("candidate") == default_name)
    metrics = default.get("validation_line_metrics") or {}
    beta = (default.get("config") or {}).get("beta") or (default.get("spec") or {}).get("beta")
    return (
        f"beta {beta}가 validation 기준 수익 추종성과 false-safe tradeoff가 가장 좋았다. "
        f"IC {_fmt(metrics.get('ic_mean'))}, spread {_fmt(metrics.get('long_short_spread'))}, "
        f"false_safe {_fmt(metrics.get('false_safe_tail_rate'))}, severe_recall {_fmt(metrics.get('severe_downside_recall'))}."
    )


def _next_cp_recommendation(default: dict[str, Any] | None) -> str:
    if default is None:
        return "1W line 저장 후보로 올리지 말고 loss/selector를 재설계한다."
    return f"{default.get('candidate')} 설정을 CP136 h4 pvv 후보의 저장 전용 재현 CP로 검토한다."


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        metrics = candidate.get("validation_line_metrics") or {}
        comparison = candidate.get("comparison_vs_cp136_validation") or {}
        config = candidate.get("config") or candidate.get("spec") or {}
        rows.append(
            {
                "candidate": candidate.get("candidate"),
                "class": candidate.get("classification"),
                "run_id": candidate.get("run_id"),
                "beta": config.get("beta"),
                "gate": (candidate.get("gate_status") or {}).get("line_gate_pass"),
                "ic": _fmt(metrics.get("ic_mean")),
                "d_ic": _fmt((comparison.get("ic_mean") or {}).get("delta")),
                "spread": _fmt(metrics.get("long_short_spread")),
                "fee": _fmt(metrics.get("fee_adjusted_return")),
                "false_safe": _fmt(metrics.get("false_safe_tail_rate")),
                "d_false_safe": _fmt((comparison.get("false_safe_tail_rate") or {}).get("delta")),
                "severe": _fmt(metrics.get("severe_downside_recall")),
                "bias": _fmt(metrics.get("conservative_bias")),
                "upside": _fmt(metrics.get("upside_sacrifice")),
            }
        )
    return rows


def _comparison_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in LINE_KEYS:
        item = (candidate.get("comparison_vs_cp136_validation") or {}).get(key) or {}
        rows.append(
            {
                "metric": key,
                "cp136_val": _fmt(item.get("cp136_validation")),
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
    decision = payload.get("summary_decision") or {}
    candidates = payload.get("candidates") or []
    report = [
        "# CP137-LM 1W 라인 보수 손실 실험",
        "",
        "## 요약",
        "",
        f"- 상태: `{payload.get('status')}`",
        f"- recommended_default: `{decision.get('recommended_default')}` / run_id `{decision.get('recommended_default_run_id')}`",
        "- 후보 선택은 validation 중심이며 test는 직접 선택에 쓰지 않았다.",
        "- save-run, DB write, inference 저장, W&B, composite, 프론트 수정, live yfinance fetch, EODHD 호출은 하지 않았다.",
        "",
        "## 데이터 기준",
        "",
        f"- source_data_hash: `{data.get('source_data_hash')}` / expected `{data.get('expected_source_data_hash')}`",
        f"- context_checksum: `{snapshot.get('context_checksum')}` / expected `{data.get('expected_context_checksum')}`",
        f"- source/context CP136 일치: `{data.get('source_hash_matches_cp136')}` / `{data.get('context_checksum_matches_cp136')}`",
        f"- feature/target NaN/Inf: `{data.get('feature_nonfinite_count')}` / `{data.get('target_nonfinite_count')}`",
        f"- test_exposure_count: `{data.get('test_exposure_count')}`",
        "",
        "## beta 후보 결과",
        "",
        _table(
            _candidate_rows(candidates),
            [
                ("candidate", "candidate"),
                ("class", "class"),
                ("run_id", "run_id"),
                ("beta", "beta"),
                ("gate", "gate"),
                ("val_ic", "ic"),
                ("d_ic", "d_ic"),
                ("val_spread", "spread"),
                ("val_fee", "fee"),
                ("val_false_safe", "false_safe"),
                ("d_false_safe", "d_false_safe"),
                ("val_severe", "severe"),
                ("bias", "bias"),
                ("upside", "upside"),
            ],
        ),
        "",
        "## CP136 기준 대비 validation 개선/악화",
        "",
    ]
    for candidate in candidates:
        if candidate.get("status") == "omitted":
            report.extend([f"### {candidate.get('candidate')}", "", candidate.get("omission_reason") or "optional omitted", ""])
            continue
        report.extend(
            [
                f"### {candidate.get('candidate')}",
                "",
                _table(
                    _comparison_rows(candidate),
                    [("metric", "metric"), ("CP136 val", "cp136_val"), ("current val", "current_val"), ("delta", "delta"), ("판정", "direction")],
                ),
                "",
            ]
        )
    bucket_rows = _bucket_rows(candidates)
    report.extend(
        [
            "## h1_h4 버킷 지표",
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
            "## 판단",
            "",
            f"- beta tradeoff: {decision.get('beta_tradeoff_summary')}",
            f"- 다음 CP 추천: {decision.get('next_cp_recommendation')}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8-sig")


def _write_registry(registry: list[dict[str, Any]]) -> None:
    _write_json(REGISTRY_PATH, {"cp": "CP137-LM", "generated_at": datetime.now().isoformat(timespec="seconds"), "candidates": registry})


def _write_csv(candidates: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "classification",
        "run_id",
        "beta",
        "line_gate_pass",
        "test_exposure_count",
        "val_ic_mean",
        "val_long_short_spread",
        "val_fee_adjusted_return",
        "val_false_safe_tail_rate",
        "val_false_safe_severe_rate",
        "val_severe_downside_recall",
        "val_conservative_bias",
        "val_upside_sacrifice",
        "val_direction_accuracy",
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
                    "beta": config.get("beta"),
                    "line_gate_pass": (candidate.get("gate_status") or {}).get("line_gate_pass"),
                    "test_exposure_count": candidate.get("test_exposure_count"),
                    "val_ic_mean": val.get("ic_mean"),
                    "val_long_short_spread": val.get("long_short_spread"),
                    "val_fee_adjusted_return": val.get("fee_adjusted_return"),
                    "val_false_safe_tail_rate": val.get("false_safe_tail_rate"),
                    "val_false_safe_severe_rate": val.get("false_safe_severe_rate"),
                    "val_severe_downside_recall": val.get("severe_downside_recall"),
                    "val_conservative_bias": val.get("conservative_bias"),
                    "val_upside_sacrifice": val.get("upside_sacrifice"),
                    "val_direction_accuracy": val.get("direction_accuracy"),
                    "test_ic_mean": test.get("ic_mean"),
                    "test_long_short_spread": test.get("long_short_spread"),
                    "test_false_safe_tail_rate": test.get("false_safe_tail_rate"),
                    "test_severe_downside_recall": test.get("severe_downside_recall"),
                }
            )


def run_preflight(_: argparse.Namespace) -> None:
    payload = build_preflight_payload()
    print(json.dumps(_json_safe({"preflight_path": str(PREFLIGHT_PATH), "preflight_gate_pass": payload["preflight_gate_pass"], "source_data_hash": payload["data"]["source_data_hash"], "context_checksum": payload["data"]["snapshot"]["context_checksum"]}), ensure_ascii=False, indent=2))


def run_report(_: argparse.Namespace) -> None:
    payload, registry = build_payload()
    _write_json(METRICS_PATH, payload)
    _write_registry(registry)
    _write_csv(payload.get("candidates") or [])
    _write_report(payload)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "registry_path": str(REGISTRY_PATH), "csv_path": str(CSV_PATH), "report_path": str(REPORT_PATH), "status": payload.get("status"), "summary_decision": payload.get("summary_decision")}), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP137-LM 1W line beta loss study")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=run_preflight)
    report = subparsers.add_parser("report")
    report.set_defaults(func=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
