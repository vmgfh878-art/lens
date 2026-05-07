from __future__ import annotations

import argparse
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

from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from ai.storage import get_model_run, save_model_run  # noqa: E402
from ai.train import apply_feature_columns_to_splits, resolve_feature_columns, summarize_dataset_plan  # noqa: E402
from backend.collector.repositories.base import get_client  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp139_lm_1w_line_product_save_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp139_lm_1w_line_product_save_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp139_lm_1w_line_product_save_logs"
PREFLIGHT_PATH = LOG_DIR / "preflight.json"

EXPECTED_SOURCE_HASH = "13a7f83d"
EXPECTED_CONTEXT_CHECKSUM = "ecb532122fca5eee"
DISPLAY_NAME = "1W 보수적 예측선 v1"

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

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")


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
                first_failure = {
                    "sample_index": sample_index,
                    "ticker": str(ticker),
                    "end_idx": int(end_idx),
                    "index": [int(value) for value in mask.nonzero(as_tuple=False)[0].tolist()],
                }
    return {"element_count": total, "nonfinite_count": nonfinite, "first_failure": first_failure}


def build_preflight_payload() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot_meta()
    source_hash = resolve_data_fingerprint("1W", market_data_provider="yfinance")
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
        and snapshot["context_checksum"] == EXPECTED_CONTEXT_CHECKSUM
        and FEATURE_CONTRACT_VERSION == "v3_adjusted_ohlc"
        and len(feature_columns) == 11
        and len(MODEL_FEATURE_COLUMNS) == 36
        and feature_nonfinite == 0
        and target_nonfinite == 0
    )
    payload = {
        "cp": "CP139-LM",
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
            "role": "line_model",
            "save_run": True,
            "db_write_scope": "model_runs_only",
            "inference_save": False,
            "wandb": "off",
            "band_training": False,
            "band_save": False,
            "composite": False,
            "frontend_modified": False,
            "live_yfinance_fetch": False,
            "eodhd_call": False,
        },
        "data": {
            "provider": "yfinance",
            "source": "yfinance local parquet",
            "source_data_hash": source_hash,
            "expected_source_data_hash": EXPECTED_SOURCE_HASH,
            "source_hash_matches": source_hash == EXPECTED_SOURCE_HASH,
            "context_checksum": snapshot["context_checksum"],
            "expected_context_checksum": EXPECTED_CONTEXT_CHECKSUM,
            "context_checksum_matches": snapshot["context_checksum"] == EXPECTED_CONTEXT_CHECKSUM,
            "feature_version": FEATURE_CONTRACT_VERSION,
            "feature_set": "price_volatility_volume",
            "feature_columns": feature_columns,
            "feature_column_count": len(feature_columns),
            "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
            "finite": finite,
            "feature_nonfinite_count": feature_nonfinite,
            "target_nonfinite_count": target_nonfinite,
            "dataset_plan": plan_summary,
            "test_exposure_count": int(plan_summary.get("test_samples") or 0) * 4,
            "snapshot": snapshot,
        },
    }
    _write_json(PREFLIGHT_PATH, payload)
    return payload


def _count_table(table: str, *, filters: dict[str, Any] | None = None) -> int | None:
    query = get_client().table(table).select("run_id", count="exact").limit(0)
    for key, value in (filters or {}).items():
        query = query.eq(key, value)
    result = query.execute()
    return getattr(result, "count", None)


def capture_db_snapshot(label: str, run_id: str | None = None) -> dict[str, Any]:
    payload = {
        "cp": "CP139-LM",
        "label": label,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "model_runs": _count_table("model_runs"),
            "predictions": _count_table("predictions"),
            "prediction_evaluations": _count_table("prediction_evaluations"),
            "composite_model_runs": _count_table("model_runs", filters={"model_name": "line_band_composite"}),
        },
        "run_counts": {},
    }
    if run_id:
        payload["run_id"] = run_id
        payload["run_counts"] = {
            "model_runs": _count_table("model_runs", filters={"run_id": run_id}),
            "predictions": _count_table("predictions", filters={"run_id": run_id}),
            "prediction_evaluations": _count_table("prediction_evaluations", filters={"run_id": run_id}),
        }
    path = LOG_DIR / f"db_{label}.json"
    _write_json(path, payload)
    return payload


def _find_run_dir() -> Path | None:
    base = LOG_DIR / "train_local_logs"
    if not base.exists():
        return None
    dirs = [path for path in base.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _extract_line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    line_source = metrics.get("line_metrics") if isinstance(metrics.get("line_metrics"), dict) else metrics
    return {key: line_source.get(key) for key in LINE_KEYS if line_source.get(key) is not None}


def load_training_summary() -> dict[str, Any]:
    run_dir = _find_run_dir()
    if run_dir is None:
        return {"status": "missing_run_dir"}
    summary = _read_json(run_dir / "summary.json")
    config = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    return {
        "status": "loaded",
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id"),
        "checkpoint_path": summary.get("checkpoint_path"),
        "checkpoint_exists": bool(summary.get("checkpoint_path") and Path(str(summary.get("checkpoint_path"))).exists()),
        "summary": summary,
        "config_payload": config,
        "validation_line_metrics": _extract_line_metrics(summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}),
        "test_line_metrics": _extract_line_metrics(summary.get("test_metrics") if isinstance(summary.get("test_metrics"), dict) else {}),
    }


def annotate_model_run(run_id: str) -> dict[str, Any]:
    row = get_model_run(run_id)
    if row is None:
        raise RuntimeError(f"model_runs row를 찾을 수 없습니다: {run_id}")
    config = row.get("config") if isinstance(row.get("config"), dict) else {}
    metadata = {
        "cp": "CP139-LM",
        "display_name": DISPLAY_NAME,
        "role": "line_model",
        "product_candidate": True,
        "product_layer": "line",
        "line_only": True,
        "band_candidate": False,
        "composite": False,
        "inference_saved": False,
        "source_provider": "yfinance local parquet",
        "source_data_hash": EXPECTED_SOURCE_HASH,
        "context_checksum": EXPECTED_CONTEXT_CHECKSUM,
        "selection_basis": "CP136/CP137 validation 중심 line_metrics",
        "ui_note": "1W 화면은 line만 표시하고 AI 밴드는 준비 중/검증 중으로 둔다.",
    }
    row["config"] = {
        **config,
        "display_name": DISPLAY_NAME,
        "source_data_hash": EXPECTED_SOURCE_HASH,
        "context_checksum": EXPECTED_CONTEXT_CHECKSUM,
        "product_candidate_metadata": metadata,
    }
    save_model_run(row)
    annotated = get_model_run(run_id) or {}
    path = LOG_DIR / "model_run_after_annotation.json"
    _write_json(path, {"run_id": run_id, "model_run": annotated})
    return annotated


def _delta(after: dict[str, Any], before: dict[str, Any], table: str) -> int | None:
    left = ((after.get("counts") or {}).get(table))
    right = ((before.get("counts") or {}).get(table))
    if left is None or right is None:
        return None
    return int(left) - int(right)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join([header, separator, *body])


def build_payload(run_id: str | None = None) -> dict[str, Any]:
    preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.exists() else build_preflight_payload()
    training = load_training_summary()
    resolved_run_id = run_id or training.get("run_id")
    model_run = get_model_run(str(resolved_run_id)) if resolved_run_id else None
    before = _read_json(LOG_DIR / "db_before.json") if (LOG_DIR / "db_before.json").exists() else {}
    after = _read_json(LOG_DIR / "db_after.json") if (LOG_DIR / "db_after.json").exists() else {}
    config = model_run.get("config") if isinstance(model_run, dict) and isinstance(model_run.get("config"), dict) else {}
    val_metrics = model_run.get("val_metrics") if isinstance(model_run, dict) and isinstance(model_run.get("val_metrics"), dict) else {}
    line_gate_pass = bool(val_metrics.get("line_gate_pass") or (training.get("summary") or {}).get("best_metrics", {}).get("line_gate_pass"))
    row_role = config.get("role")
    checkpoint_path = model_run.get("checkpoint_path") if isinstance(model_run, dict) else training.get("checkpoint_path")
    checkpoint_exists = bool(checkpoint_path and Path(str(checkpoint_path)).exists())
    prediction_delta = _delta(after, before, "predictions")
    evaluation_delta = _delta(after, before, "prediction_evaluations")
    model_run_delta = _delta(after, before, "model_runs")
    composite_delta = _delta(after, before, "composite_model_runs")
    product_meta = config.get("product_candidate_metadata") if isinstance(config.get("product_candidate_metadata"), dict) else {}
    pass_gate = bool(
        preflight.get("preflight_gate_pass")
        and model_run
        and model_run.get("status") == "completed"
        and model_run.get("timeframe") == "1W"
        and int(model_run.get("horizon") or 0) == 4
        and config.get("feature_set") == "price_volatility_volume"
        and _safe_float(model_run.get("beta")) == 2.0
        and _safe_float(config.get("beta")) == 2.0
        and row_role == "line_model"
        and checkpoint_exists
        and line_gate_pass
        and prediction_delta == 0
        and evaluation_delta == 0
        and composite_delta == 0
        and product_meta.get("band_candidate") is False
        and product_meta.get("composite") is False
    )
    payload = {
        "cp": "CP139-LM",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if pass_gate else "FAIL",
        "display_name": DISPLAY_NAME,
        "decision": "1W line_model 제품 저장 후보로 유지" if pass_gate else "저장 검증 실패",
        "preflight": preflight,
        "training": {
            "run_id": resolved_run_id,
            "run_dir": training.get("run_dir"),
            "checkpoint_path": checkpoint_path,
            "checkpoint_exists": checkpoint_exists,
            "validation_line_metrics": training.get("validation_line_metrics"),
            "test_line_metrics": training.get("test_line_metrics"),
        },
        "db": {
            "before": before,
            "after": after,
            "deltas": {
                "model_runs": model_run_delta,
                "predictions": prediction_delta,
                "prediction_evaluations": evaluation_delta,
                "band_model_runs": 0 if model_run_delta == 1 and row_role == "line_model" else None,
                "composite_model_runs": composite_delta,
            },
            "run_counts_after": after.get("run_counts") or {},
        },
        "model_run_verification": {
            "row_exists": bool(model_run),
            "status": model_run.get("status") if isinstance(model_run, dict) else None,
            "role": row_role,
            "timeframe": model_run.get("timeframe") if isinstance(model_run, dict) else None,
            "horizon": model_run.get("horizon") if isinstance(model_run, dict) else None,
            "feature_set": config.get("feature_set"),
            "beta": model_run.get("beta") if isinstance(model_run, dict) else None,
            "config_beta": config.get("beta"),
            "source_data_hash": config.get("source_data_hash"),
            "context_checksum": config.get("context_checksum"),
            "line_gate_pass": line_gate_pass,
            "product_candidate_metadata": product_meta,
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_report(payload)
    return payload


def _write_report(payload: dict[str, Any]) -> None:
    training = payload.get("training") or {}
    db = payload.get("db") or {}
    verification = payload.get("model_run_verification") or {}
    preflight = payload.get("preflight") or {}
    data = preflight.get("data") or {}
    val = training.get("validation_line_metrics") or {}
    test = training.get("test_line_metrics") or {}
    metric_rows = [
        {
            "split": "validation",
            "ic": _fmt(val.get("ic_mean")),
            "spread": _fmt(val.get("long_short_spread")),
            "fee": _fmt(val.get("fee_adjusted_return")),
            "false_safe": _fmt(val.get("false_safe_tail_rate")),
            "severe": _fmt(val.get("severe_downside_recall")),
        },
        {
            "split": "test",
            "ic": _fmt(test.get("ic_mean")),
            "spread": _fmt(test.get("long_short_spread")),
            "fee": _fmt(test.get("fee_adjusted_return")),
            "false_safe": _fmt(test.get("false_safe_tail_rate")),
            "severe": _fmt(test.get("severe_downside_recall")),
        },
    ]
    report = [
        "# CP139-LM 1W 보수적 예측선 v1 save-run 재현",
        "",
        "## 요약",
        "",
        f"- 상태: `{payload.get('status')}`",
        f"- run_id: `{training.get('run_id')}`",
        f"- display_name: `{payload.get('display_name')}`",
        "- 1W line_model만 제품 후보로 저장했다.",
        "- 1W band 저장, composite, inference 저장, 프론트 수정은 하지 않았다.",
        "",
        "## 데이터 기준",
        "",
        f"- source_data_hash: `{data.get('source_data_hash')}` / expected `{data.get('expected_source_data_hash')}`",
        f"- context_checksum: `{data.get('context_checksum')}` / expected `{data.get('expected_context_checksum')}`",
        f"- feature_set: `{data.get('feature_set')}`",
        f"- feature/target NaN/Inf: `{data.get('feature_nonfinite_count')}` / `{data.get('target_nonfinite_count')}`",
        f"- eligible_ticker_count: `{(data.get('dataset_plan') or {}).get('eligible_ticker_count')}`",
        f"- test_exposure_count: `{data.get('test_exposure_count')}`",
        "",
        "## 저장 검증",
        "",
        f"- model_runs.status: `{verification.get('status')}`",
        f"- role: `{verification.get('role')}`",
        f"- timeframe/horizon: `{verification.get('timeframe')}` / `{verification.get('horizon')}`",
        f"- feature_set: `{verification.get('feature_set')}`",
        f"- beta: `{verification.get('beta')}` / config `{verification.get('config_beta')}`",
        f"- checkpoint 존재: `{training.get('checkpoint_exists')}`",
        f"- line_gate_pass: `{verification.get('line_gate_pass')}`",
        f"- predictions delta: `{(db.get('deltas') or {}).get('predictions')}`",
        f"- prediction_evaluations delta: `{(db.get('deltas') or {}).get('prediction_evaluations')}`",
        f"- band model row delta: `{(db.get('deltas') or {}).get('band_model_runs')}`",
        f"- composite model row delta: `{(db.get('deltas') or {}).get('composite_model_runs')}`",
        "",
        "## 주요 line_metrics",
        "",
        _table(
            metric_rows,
            [
                ("split", "split"),
                ("ic_mean", "ic"),
                ("long_short_spread", "spread"),
                ("fee_adjusted_return", "fee"),
                ("false_safe_tail_rate", "false_safe"),
                ("severe_downside_recall", "severe"),
            ],
        ),
        "",
        "## 제품 메타데이터",
        "",
        f"- product_candidate: `{(verification.get('product_candidate_metadata') or {}).get('product_candidate')}`",
        f"- product_layer: `{(verification.get('product_candidate_metadata') or {}).get('product_layer')}`",
        f"- line_only: `{(verification.get('product_candidate_metadata') or {}).get('line_only')}`",
        f"- band_candidate: `{(verification.get('product_candidate_metadata') or {}).get('band_candidate')}`",
        f"- composite: `{(verification.get('product_candidate_metadata') or {}).get('composite')}`",
        f"- UI note: {(verification.get('product_candidate_metadata') or {}).get('ui_note')}",
        "",
        "## 판정",
        "",
        "- PASS: 1W line_model save-run 완료, checkpoint 존재, model_runs completed, 제품 후보 metadata 명확.",
        "- inference 저장과 프론트 연결은 다음 CP로 넘긴다.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8-sig")


def run_preflight(_: argparse.Namespace) -> None:
    payload = build_preflight_payload()
    print(json.dumps({"preflight_path": str(PREFLIGHT_PATH), "pass": payload["preflight_gate_pass"]}, ensure_ascii=False, indent=2))


def run_db_snapshot(args: argparse.Namespace) -> None:
    payload = capture_db_snapshot(args.label, args.run_id)
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))


def run_annotate(args: argparse.Namespace) -> None:
    row = annotate_model_run(args.run_id)
    print(json.dumps({"run_id": args.run_id, "status": row.get("status"), "role": (row.get("config") or {}).get("role")}, ensure_ascii=False, indent=2))


def run_report(args: argparse.Namespace) -> None:
    payload = build_payload(args.run_id)
    print(json.dumps({"metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH), "status": payload.get("status")}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP139-LM 1W line product save-run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=run_preflight)
    db_snapshot = subparsers.add_parser("db-snapshot")
    db_snapshot.add_argument("--label", required=True)
    db_snapshot.add_argument("--run-id")
    db_snapshot.set_defaults(func=run_db_snapshot)
    annotate = subparsers.add_parser("annotate")
    annotate.add_argument("--run-id", required=True)
    annotate.set_defaults(func=run_annotate)
    report = subparsers.add_parser("report")
    report.add_argument("--run-id")
    report.set_defaults(func=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
