from __future__ import annotations

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


REPORT_PATH = PROJECT_ROOT / "docs" / "cp75_lm_1d_h5_full_line_product_candidate_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp75_lm_1d_h5_full_line_product_candidate_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp75_lm_1d_h5_full_line_product_candidate_logs"
CP72_METRICS_PATH = PROJECT_ROOT / "docs" / "cp72_bm_1d_full_band_product_candidate_metrics.json"
CP49_METRICS_PATH = PROJECT_ROOT / "docs" / "cp49_patchtst_horizon_rescue_metrics.json"
CP53_METRICS_PATH = PROJECT_ROOT / "docs" / "cp53_existing_candidate_regrade_metrics.json"
CP54_METRICS_PATH = PROJECT_ROOT / "docs" / "cp54_baseline_metric_comparison_metrics.json"

REQUIRED_LINE_KEYS = [
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "mae",
    "smape",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value.__class__.__module__.startswith("pandas") and value.__class__.__name__ == "Timestamp":
        return value.strftime("%Y-%m-%d")
    if value.__class__.__module__.startswith("numpy") and hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _metric_source(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    nested = metrics.get("line_metrics")
    if isinstance(nested, dict):
        return nested
    return metrics


def extract_required_line_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = _metric_source(metrics)
    return {key: source.get(key) for key in REQUIRED_LINE_KEYS}


def classify_line_candidate(line_metrics: dict[str, Any], storage_ok: bool) -> str:
    if not storage_ok:
        return "storage_incomplete"
    ic_mean = _safe_float(line_metrics.get("ic_mean"))
    spread = _safe_float(line_metrics.get("long_short_spread"))
    tail = _safe_float(line_metrics.get("false_safe_tail_rate"))
    recall = _safe_float(line_metrics.get("severe_downside_recall"))
    if ic_mean is not None and spread is not None and tail is not None and recall is not None:
        if ic_mean > 0 and spread > 0 and tail < 0.30 and recall >= 0.70:
            return "product_candidate_ready"
        if ic_mean > 0 and spread > 0:
            return "completed_line_watch"
    if ic_mean is not None and spread is not None and ic_mean < 0 and spread < 0:
        return "completed_line_fail"
    return "completed_needs_review"


def _contains_target_candidate(record: dict[str, Any]) -> bool:
    text = json.dumps(record, ensure_ascii=False).lower()
    if "h5_longer_context_seq252_p32_s16" in text:
        return True
    if str(record.get("model") or record.get("model_name") or "").lower() != "patchtst":
        return False
    return (
        int(record.get("horizon") or -1) == 5
        and int(record.get("seq_len") or -1) == 252
        and int(record.get("patch_len") or -1) == 32
        and int(record.get("patch_stride") or -1) == 16
    )


def _walk_dict_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        records.append(value)
        for item in value.values():
            records.extend(_walk_dict_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_walk_dict_records(item))
    return records


def _compact_reference_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    metrics = record.get("line_metrics")
    if not isinstance(metrics, dict):
        metrics = record.get("test_metrics") if isinstance(record.get("test_metrics"), dict) else record
    line_metrics = extract_required_line_metrics(metrics)
    name = record.get("name") or record.get("candidate") or record.get("run_id") or record.get("label")
    return {
        "source": source,
        "name": name,
        "run_id": record.get("run_id"),
        "horizon": record.get("horizon"),
        "seq_len": record.get("seq_len"),
        "patch_len": record.get("patch_len"),
        "patch_stride": record.get("patch_stride"),
        "line_metrics": {key: line_metrics.get(key) for key in ["ic_mean", "long_short_spread", "false_safe_tail_rate", "severe_downside_recall"]},
    }


def _load_line_references() -> dict[str, Any]:
    references: dict[str, Any] = {"cp49_cp53_h5_longer_context": [], "statistical_baseline_summary": []}
    for path in [CP49_METRICS_PATH, CP53_METRICS_PATH]:
        payload = _read_json(path)
        matches = [_compact_reference_record(record, path.name) for record in _walk_dict_records(payload) if _contains_target_candidate(record)]
        references["cp49_cp53_h5_longer_context"].extend(matches[:8])

    baseline_payload = _read_json(CP54_METRICS_PATH)
    baseline_names = ("historical_mean_line", "reversal_line", "random", "shuffled")
    baseline_records = []
    for record in _walk_dict_records(baseline_payload):
        text = json.dumps(record, ensure_ascii=False).lower()
        if any(name in text for name in baseline_names):
            metrics = extract_required_line_metrics(record.get("line_metrics") if isinstance(record.get("line_metrics"), dict) else record)
            if any(metrics.get(key) is not None for key in ["ic_mean", "long_short_spread", "false_safe_tail_rate"]):
                baseline_records.append(
                    {
                        "source": CP54_METRICS_PATH.name,
                        "name": record.get("name") or record.get("baseline") or record.get("candidate") or record.get("label"),
                        "line_metrics": {
                            "ic_mean": metrics.get("ic_mean"),
                            "long_short_spread": metrics.get("long_short_spread"),
                            "false_safe_tail_rate": metrics.get("false_safe_tail_rate"),
                            "severe_downside_recall": metrics.get("severe_downside_recall"),
                        },
                    }
                )
    references["statistical_baseline_summary"] = baseline_records[:12]
    return references


def _load_local_log_summary(run_id: str, log_dir: Path) -> dict[str, Any]:
    run_dir = log_dir / run_id
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "config_exists": (run_dir / "config.json").exists(),
        "summary_exists": (run_dir / "summary.json").exists(),
        "metrics_exists": (run_dir / "metrics.jsonl").exists(),
    }
    if (run_dir / "config.json").exists():
        result["config"] = _read_json(run_dir / "config.json")
    if (run_dir / "summary.json").exists():
        result["summary"] = _read_json(run_dir / "summary.json")
    if (run_dir / "metrics.jsonl").exists():
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        epochs = [json.loads(line) for line in lines if line.strip()]
        result["epoch_count"] = len(epochs)
        if epochs:
            result["last_epoch"] = epochs[-1]
            result["vram_peak_allocated_mb_max"] = max(
                [_safe_float(row.get("vram_peak_allocated_mb")) or 0.0 for row in epochs]
            )
    return result


def build_preflight_payload() -> dict[str, Any]:
    import torch  # noqa: F401
    from ai.cp66_lm_post_backfill_h20_1w import _build_bundles
    from ai.preprocessing import FEATURE_CONTRACT_VERSION, MODEL_FEATURE_COLUMNS, resolve_data_fingerprint
    from ai.train import resolve_feature_columns

    data_hash = resolve_data_fingerprint("1D")
    feature_columns = resolve_feature_columns("full_features")
    train_bundle, val_bundle, test_bundle, _, _, plan, cache_meta, _, _, _ = _build_bundles(
        timeframe="1D",
        data_hash=data_hash,
        seq_len=252,
        horizon=5,
        limit_tickers=None,
        include_future_covariate=False,
    )
    cp72 = _read_json(CP72_METRICS_PATH)
    cp72_hash = ((cp72.get("preflight") or {}).get("source_data_hash"))
    tensor_finite = cache_meta.get("tensor_finite") or {}
    ratio_sanity = cache_meta.get("ratio_sanity") or {}
    preflight = {
        "timeframe": "1D",
        "horizon": 5,
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "source_data_hash": data_hash,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_set": "full_features",
        "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
        "resolved_feature_columns_count": len(feature_columns),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "intraday_range_ratio_in_model_features": "intraday_range_ratio" in MODEL_FEATURE_COLUMNS,
        "feature_nonfinite_count": int(tensor_finite.get("feature_nonfinite_count") or 0),
        "target_nonfinite_count": int(tensor_finite.get("target_nonfinite_count") or 0),
        "ratio_sanity_pass": bool(ratio_sanity.get("pass")),
        "ratio_sanity": ratio_sanity,
        "eligible_ticker_count": len(plan.eligible_tickers),
        "input_ticker_count": int(plan.input_ticker_count),
        "excluded_ticker_count": len(plan.excluded_reasons),
        "train_samples": len(train_bundle),
        "val_samples": len(val_bundle),
        "test_samples": len(test_bundle),
        "cp72_bm_source_data_hash": cp72_hash,
        "data_hash_matches_cp72_bm": bool(cp72_hash == data_hash),
        "cache_meta": cache_meta,
    }
    gate_pass = (
        preflight["feature_version"] == "v3_adjusted_ohlc"
        and preflight["feature_set"] == "full_features"
        and preflight["model_feature_columns_count"] == 36
        and preflight["resolved_feature_columns_count"] == 36
        and not preflight["atr_ratio_in_model_features"]
        and preflight["feature_nonfinite_count"] == 0
        and preflight["target_nonfinite_count"] == 0
        and preflight["ratio_sanity_pass"]
        and preflight["eligible_ticker_count"] > 0
    )
    return {
        "cp": "CP75-LM",
        "title": "1D h5 line full product candidate training",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preflight": preflight,
        "preflight_gate_pass": gate_pass,
        "line_contract": {
            "role": "line_model",
            "target": "raw_future_return",
            "line_target_type": "raw_future_return",
            "checkpoint_selection": "line_gate",
            "ci_aggregate": "target",
            "line_metrics_only": True,
            "band_or_composite_used": False,
        },
    }


def _config_field_status(config: dict[str, Any], model_run: dict[str, Any]) -> dict[str, Any]:
    wandb_status = config.get("wandb_status") if isinstance(config.get("wandb_status"), dict) else {}
    return {
        "feature_set": config.get("feature_set"),
        "horizon": model_run.get("horizon"),
        "seq_len": config.get("seq_len"),
        "patch_len": config.get("patch_len"),
        "patch_stride": config.get("patch_stride"),
        "alpha": config.get("alpha"),
        "beta": config.get("beta"),
        "delta": config.get("delta"),
        "lambda_line": config.get("lambda_line"),
        "lambda_band": config.get("lambda_band"),
        "lambda_cross": config.get("lambda_cross"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "line_target_type": config.get("line_target_type"),
        "band_target_type": config.get("band_target_type"),
        "role": config.get("role"),
        "feature_version": config.get("feature_version"),
        "config_hash": config.get("config_hash"),
        "wandb_status": wandb_status.get("status") or config.get("wandb_status"),
    }


def _sample_prediction_check(run_id: str, limit: int = 3) -> dict[str, Any]:
    from backend.collector.repositories.base import fetch_all_rows, get_client

    samples = fetch_all_rows(
        "prediction_evaluations",
        columns="run_id,ticker,timeframe,asof_date",
        filters=[("eq", "run_id", run_id)],
        limit=limit,
    )
    checked: list[dict[str, Any]] = []
    for row in samples:
        result = (
            get_client()
            .table("predictions")
            .select("run_id,ticker,model_name,timeframe,horizon,asof_date,line_series")
            .eq("run_id", run_id)
            .eq("ticker", row["ticker"])
            .eq("timeframe", row["timeframe"])
            .eq("horizon", 5)
            .eq("asof_date", row["asof_date"])
            .limit(1)
            .execute()
        )
        data = result.data or []
        checked.append(
            {
                "ticker": row["ticker"],
                "asof_date": row["asof_date"],
                "prediction_exists": bool(data),
                "line_series_present": bool(data and data[0].get("line_series")),
            }
        )
    return {
        "checked_count": len(checked),
        "all_found": bool(checked) and all(bool(row["prediction_exists"]) for row in checked),
        "all_have_line_series": bool(checked) and all(bool(row["line_series_present"]) for row in checked),
        "samples": checked,
    }


def build_postflight_payload(
    run_id: str,
    log_dir: Path,
    *,
    inference_prediction_count: int | None = None,
    inference_evaluation_count: int | None = None,
) -> dict[str, Any]:
    from ai.storage import fetch_run_evaluations, fetch_run_predictions, get_model_run

    existing = _read_json(METRICS_PATH)
    model_run = get_model_run(run_id)
    predictions_count = 0
    evaluations_count = 0
    predictions_count_source = "full_table_query"
    evaluations_count_source = "full_table_query"
    predictions_error = None
    evaluations_error = None
    prediction_sample_check: dict[str, Any] | None = None
    if model_run:
        try:
            predictions_count = int(len(fetch_run_predictions(run_id, "1D")))
        except Exception as exc:  # pragma: no cover - 운영 DB 상태 확인용
            predictions_error = str(exc)
            if inference_prediction_count is not None:
                predictions_count = int(inference_prediction_count)
                predictions_count_source = "inference_stdout_count_after_sample_check"
            else:
                predictions_count_source = "full_table_query_failed"
        try:
            evaluations_count = int(len(fetch_run_evaluations(run_id, "1D")))
        except Exception as exc:  # pragma: no cover - 운영 DB 상태 확인용
            evaluations_error = str(exc)
            if inference_evaluation_count is not None:
                evaluations_count = int(inference_evaluation_count)
                evaluations_count_source = "inference_stdout_count"
            else:
                evaluations_count_source = "full_table_query_failed"
        if predictions_error:
            try:
                prediction_sample_check = _sample_prediction_check(run_id)
            except Exception as exc:  # pragma: no cover - 운영 DB 상태 확인용
                prediction_sample_check = {"error": str(exc), "all_found": False, "all_have_line_series": False}

    config = model_run.get("config") if isinstance(model_run, dict) and isinstance(model_run.get("config"), dict) else {}
    checkpoint_path = Path(str(model_run.get("checkpoint_path"))) if model_run and model_run.get("checkpoint_path") else None
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    line_metrics = extract_required_line_metrics(model_run.get("test_metrics") if model_run else {})
    val_line_metrics = extract_required_line_metrics(model_run.get("val_metrics") if model_run else {})
    storage_ok = bool(
        model_run
        and model_run.get("status") == "completed"
        and config.get("role") == "line_model"
        and checkpoint_path
        and checkpoint_path.exists()
        and predictions_count > 0
        and evaluations_count > 0
    )
    storage_verification = {
        "model_runs_exists": bool(model_run),
        "status": model_run.get("status") if model_run else None,
        "run_id": run_id,
        "role_identification": f"config.role={config.get('role')}" if config else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "predictions_count": predictions_count,
        "prediction_evaluations_count": evaluations_count,
        "predictions_count_source": predictions_count_source,
        "prediction_evaluations_count_source": evaluations_count_source,
        "predictions_error": predictions_error,
        "prediction_evaluations_error": evaluations_error,
        "prediction_sample_check": prediction_sample_check,
        "config_saved": _config_field_status(config, model_run or {}),
    }
    payload = {
        **existing,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_product_candidate": {
            "run_id": run_id,
            "status": model_run.get("status") if model_run else None,
            "model_name": model_run.get("model_name") if model_run else None,
            "role": config.get("role"),
            "timeframe": model_run.get("timeframe") if model_run else None,
            "horizon": model_run.get("horizon") if model_run else None,
            "seq_len": config.get("seq_len"),
            "patch_len": config.get("patch_len"),
            "patch_stride": config.get("patch_stride"),
            "feature_set": config.get("feature_set"),
            "feature_version": config.get("feature_version"),
            "target": config.get("line_target_type"),
            "line_target_type": config.get("line_target_type"),
            "checkpoint_selection": config.get("checkpoint_selection"),
            "alpha": config.get("alpha"),
            "beta": config.get("beta"),
            "wandb_status": config.get("wandb_status"),
        },
        "execution": {
            "python_executable": sys.executable,
            "device": config.get("device"),
            "amp_dtype": config.get("amp_dtype"),
            "compile_model": config.get("compile_model"),
            "batch_size": config.get("batch_size"),
            "epochs": config.get("epochs"),
            "num_workers": config.get("num_workers"),
            "save_run": True,
            "explicit_cuda_cleanup": config.get("explicit_cuda_cleanup"),
            "estimated_before_run": {
                "wall_minutes": "70-140",
                "vram_gb": "3-6",
                "basis": "CP72 no-limit band run wall time, CP67 PatchTST CUDA VRAM, and CP75 no-limit eligible sample count",
            },
            "local_log": _load_local_log_summary(run_id, log_dir),
        },
        "line_metrics": line_metrics,
        "val_line_metrics": val_line_metrics,
        "storage_verification": storage_verification,
        "comparison_reference": {
            "cp72_bm_source_data_hash": ((existing.get("preflight") or {}).get("cp72_bm_source_data_hash")),
            "cp75_source_data_hash": ((existing.get("preflight") or {}).get("source_data_hash")),
            "data_hash_matches_cp72_bm": ((existing.get("preflight") or {}).get("data_hash_matches_cp72_bm")),
            "line_references": _load_line_references(),
        },
        "decision": {
            "candidate_classification": classify_line_candidate(line_metrics, storage_ok),
            "line_metrics_only": True,
            "band_model_experiment": False,
            "composite_overlay_experiment": False,
            "product_artifact": "line_series",
            "note": "line_model의 lower_band/upper_band 출력은 제품 band 후보로 쓰지 않는다.",
        },
    }
    return payload


def _line_metrics_table(metrics: dict[str, Any]) -> str:
    rows = ["| metric | value |", "|---|---:|"]
    for key in REQUIRED_LINE_KEYS:
        rows.append(f"| `{key}` | {_fmt(metrics.get(key), 6)} |")
    return "\n".join(rows)


def _write_report(payload: dict[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    storage = payload.get("storage_verification") or {}
    candidate = payload.get("final_product_candidate") or {}
    decision = payload.get("decision") or {}
    execution = payload.get("execution") or {}
    local_log = execution.get("local_log") or {}
    line_metrics = payload.get("line_metrics") or {}
    references = ((payload.get("comparison_reference") or {}).get("line_references") or {})
    cp49_refs = references.get("cp49_cp53_h5_longer_context") or []
    baseline_refs = references.get("statistical_baseline_summary") or []

    report = [
        "# CP75-LM 1D h5 line full product candidate training",
        "",
        "## 요약",
        "",
        f"- 후보 run_id: `{candidate.get('run_id')}`",
        f"- 저장 상태: `{storage.get('status')}` / 판단: `{decision.get('candidate_classification')}`",
        "- 범위: PatchTST 1D h5 line_model 전용. band/composite/overlay 실험은 실행하지 않았다.",
        "- 제품 산출물은 `line_series`이며, line_model의 lower/upper 출력은 제품 band 후보로 쓰지 않는다.",
        "",
        "## 사전 게이트",
        "",
        f"- source_data_hash: `{preflight.get('source_data_hash')}`",
        f"- CP72 BM source_data_hash 동일 여부: `{preflight.get('data_hash_matches_cp72_bm')}`",
        f"- feature_version: `{preflight.get('feature_version')}`",
        f"- feature_set: `{preflight.get('feature_set')}`",
        f"- MODEL_FEATURE_COLUMNS: `{preflight.get('model_feature_columns_count')}`",
        f"- full_features resolved columns: `{preflight.get('resolved_feature_columns_count')}`",
        f"- atr_ratio 모델 입력 포함: `{preflight.get('atr_ratio_in_model_features')}`",
        f"- feature NaN/Inf: `{preflight.get('feature_nonfinite_count')}`",
        f"- target NaN/Inf: `{preflight.get('target_nonfinite_count')}`",
        f"- open/high/low_ratio sanity: `{preflight.get('ratio_sanity_pass')}`",
        f"- eligible ticker count: `{preflight.get('eligible_ticker_count')}`",
        f"- 학습 샘플 train/val/test: `{preflight.get('train_samples')}` / `{preflight.get('val_samples')}` / `{preflight.get('test_samples')}`",
        "",
        "## 실행",
        "",
        f"- python: `{execution.get('python_executable')}`",
        f"- device: `{execution.get('device')}` / amp_dtype: `{execution.get('amp_dtype')}` / compile_model: `{execution.get('compile_model')}`",
        f"- epochs: `{execution.get('epochs')}` / batch_size: `{execution.get('batch_size')}` / num_workers: `{execution.get('num_workers')}`",
        f"- 실행 전 예상: `{((execution.get('estimated_before_run') or {}).get('wall_minutes'))}`분, `{((execution.get('estimated_before_run') or {}).get('vram_gb'))}`GB VRAM",
        f"- local log dir: `{local_log.get('run_dir')}`",
        f"- epoch_count: `{local_log.get('epoch_count')}` / peak VRAM MB: `{_fmt(local_log.get('vram_peak_allocated_mb_max'), 2)}`",
        f"- W&B status: `{((candidate.get('wandb_status') or {}).get('status'))}`",
        "",
        "## 저장 확인",
        "",
        f"- model_runs row 존재: `{storage.get('model_runs_exists')}`",
        f"- role/config: `{storage.get('role_identification')}`",
        f"- checkpoint 존재: `{storage.get('checkpoint_exists')}`",
        f"- predictions 저장 수: `{storage.get('predictions_count')}` / 확인 방식: `{storage.get('predictions_count_source')}`",
        f"- prediction_evaluations 저장 수: `{storage.get('prediction_evaluations_count')}` / 확인 방식: `{storage.get('prediction_evaluations_count_source')}`",
        f"- predictions 샘플 line_series 확인: `{((storage.get('prediction_sample_check') or {}).get('all_have_line_series'))}`",
        "",
        "## h5 line 결과",
        "",
        _line_metrics_table(line_metrics),
        "",
        "## 비교 기준",
        "",
        f"- CP49/CP53 h5 longer_context 참조 후보 수: `{len(cp49_refs)}`",
        f"- 통계 baseline 참조 수: `{len(baseline_refs)}`",
        "- 참조 후보는 기존 문서의 line_metrics만 요약했다. band/composite 지표는 ranking에 쓰지 않았다.",
        "",
        "## 제품 판단",
        "",
        f"- h5 line 제품 기본 후보 판단: `{decision.get('candidate_classification')}`",
        "- CP72 BM과 결합 저장하지 않았고, 이번 CP의 저장 run은 line_model 단독 후보로 남겼다.",
        "- 다음 LM 추천: 저장 run의 실서비스 노출 전, 동일 해시에서 추론 latency와 최신 asof_date 샘플 line_series sanity만 별도 smoke로 확인한다.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def run_preflight(_: argparse.Namespace) -> None:
    payload = build_preflight_payload()
    _write_json(METRICS_PATH, payload)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "preflight_gate_pass": payload["preflight_gate_pass"], "preflight": payload["preflight"]}), ensure_ascii=False, indent=2))


def run_postflight(args: argparse.Namespace) -> None:
    payload = build_postflight_payload(
        args.run_id,
        Path(args.log_dir),
        inference_prediction_count=args.inference_prediction_count,
        inference_evaluation_count=args.inference_evaluation_count,
    )
    _write_json(METRICS_PATH, payload)
    _write_report(payload)
    print(json.dumps(_json_safe({"metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH), "decision": payload.get("decision"), "storage_verification": payload.get("storage_verification")}), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP75-LM h5 line 제품 후보 사전/사후 검증")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(func=run_preflight)
    postflight = subparsers.add_parser("postflight")
    postflight.add_argument("--run-id", required=True)
    postflight.add_argument("--log-dir", default=str(LOG_DIR))
    postflight.add_argument("--inference-prediction-count", type=int, default=None)
    postflight.add_argument("--inference-evaluation-count", type=int, default=None)
    postflight.set_defaults(func=run_postflight)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
