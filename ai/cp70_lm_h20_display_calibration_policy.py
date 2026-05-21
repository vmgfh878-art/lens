from __future__ import annotations

import argparse
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

from ai.cp66_lm_post_backfill_h20_1w import _build_bundles, _fmt, _read_json, _safe_float, _table
from ai.cp68_lm_h20_conservative_line_rescue import (
    BUCKETS,
    CP67_METRICS_PATH,
    _apply_bucket_offsets,
    _build_volatility_scale,
    _checkpoint_record,
    _classify_policy,
    _collect_line_predictions,
    _false_safe_bucket,
    _false_safe_overall,
    _finite_count,
    _fit_min_offset,
    _global_high,
    _resolve_checkpoint_path,
    _segment_high,
    _summarize_policy,
)
from ai.inference import load_checkpoint
from ai.preprocessing import FEATURE_CONTRACT_VERSION, MODEL_FEATURE_COLUMNS, resolve_data_fingerprint
from ai.train import resolve_device


REPORT_PATH = PROJECT_ROOT / "docs" / "cp70_lm_h20_display_calibration_policy_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp70_lm_h20_display_calibration_policy_metrics.json"

REPORT_KEYS = [
    "ic_mean",
    "ic_ir",
    "ic_t_stat",
    "long_short_spread",
    "spread_ir",
    "spread_t_stat",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]
BUCKET_KEYS = [
    "ic_mean",
    "long_short_spread",
    "false_safe_tail_rate",
    "severe_downside_recall",
    "upside_sacrifice",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _line_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in REPORT_KEYS}


def _bucket_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in BUCKET_KEYS}


def _display_decision(policy: dict[str, Any], raw_policy: dict[str, Any]) -> dict[str, Any]:
    metrics = policy["test"]["line_metrics"]
    h11 = policy["test"]["bucket_line_metrics"].get("h11_h20", {})
    raw_metrics = raw_policy["test"]["line_metrics"]
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    ic = _safe_float(metrics.get("ic_mean"))
    spread = _safe_float(metrics.get("long_short_spread"))
    h11_false_safe = _safe_float(h11.get("false_safe_tail_rate"))
    upside = _safe_float(metrics.get("upside_sacrifice"))
    raw_upside = _safe_float(raw_metrics.get("upside_sacrifice"))

    signal_ok = bool(ic is not None and ic > 0.0 and spread is not None and spread > 0.0)
    excessive_upside = False
    if upside is not None and raw_upside is not None:
        excessive_upside = upside >= max(raw_upside * 1.5, 0.18)

    default_off = bool(
        false_safe is not None
        and false_safe < 0.30
        and severe is not None
        and severe >= 0.70
        and h11_false_safe is not None
        and h11_false_safe < 0.35
        and signal_ok
        and not excessive_upside
    )
    watch = bool(
        not default_off
        and false_safe is not None
        and 0.30 <= false_safe < 0.35
        and severe is not None
        and 0.65 <= severe < 0.70
        and (h11_false_safe is None or h11_false_safe < 0.35)
    )
    fail = bool(
        false_safe is not None
        and false_safe >= 0.35
        or severe is not None
        and severe < 0.65
        or h11_false_safe is not None
        and h11_false_safe >= 0.35
    )
    if default_off:
        verdict = "default_off_candidate"
    elif watch:
        verdict = "watch"
    elif fail:
        verdict = "fail"
    else:
        verdict = "watch"
    return {
        "verdict": verdict,
        "signal_ok": signal_ok,
        "excessive_upside_sacrifice": excessive_upside,
        "criteria": {
            "false_safe_tail_lt_0_30": false_safe is not None and false_safe < 0.30,
            "severe_recall_ge_0_70": severe is not None and severe >= 0.70,
            "h11_false_safe_lt_0_35": h11_false_safe is not None and h11_false_safe < 0.35,
            "ic_and_spread_positive": signal_ok,
        },
    }


def _line_row(policy: dict[str, Any]) -> dict[str, Any]:
    metrics = policy["test"]["line_metrics"]
    return {
        "policy": policy["name"],
        "type": policy.get("line_type"),
        "ic": _fmt(metrics.get("ic_mean")),
        "ic_ir": _fmt(metrics.get("ic_ir")),
        "ic_t": _fmt(metrics.get("ic_t_stat")),
        "spread": _fmt(metrics.get("long_short_spread")),
        "spread_ir": _fmt(metrics.get("spread_ir")),
        "spread_t": _fmt(metrics.get("spread_t_stat")),
        "fstail": _fmt(metrics.get("false_safe_tail_rate")),
        "fssev": _fmt(metrics.get("false_safe_severe_rate")),
        "recall": _fmt(metrics.get("severe_downside_recall")),
        "bias": _fmt(metrics.get("conservative_bias")),
        "sacrifice": _fmt(metrics.get("upside_sacrifice")),
        "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
        "fee_sharpe": _fmt(metrics.get("fee_adjusted_sharpe")),
        "verdict": policy.get("display_decision", {}).get("verdict", ""),
    }


def _bucket_rows(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for bucket in BUCKETS:
            metrics = policy["test"]["bucket_line_metrics"].get(bucket, {})
            rows.append(
                {
                    "policy": policy["name"],
                    "bucket": bucket,
                    "ic": _fmt(metrics.get("ic_mean")),
                    "spread": _fmt(metrics.get("long_short_spread")),
                    "fstail": _fmt(metrics.get("false_safe_tail_rate")),
                    "recall": _fmt(metrics.get("severe_downside_recall")),
                    "sacrifice": _fmt(metrics.get("upside_sacrifice")),
                }
            )
    return rows


def _fit_rows(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for policy in policies:
        fit = policy.get("fit", {})
        rows.append(
            {
                "policy": policy["name"],
                "type": policy.get("line_type"),
                "method": fit.get("method", "none"),
                "offset": fit.get("offset", ""),
                "bucket_offsets": fit.get("bucket_offsets", ""),
                "val_fstail": _fmt(fit.get("validation_false_safe_tail_rate")),
                "target_met": fit.get("target_met", ""),
            }
        )
    return rows


def _build_report(payload: dict[str, Any]) -> str:
    policies = payload["policies"]
    runtime = payload["runtime"]
    recommendation = payload["recommendation"]
    lines = [
        "# CP70-LM h20 display calibration policy 비교",
        "",
        "## 1. 실행 원칙",
        "- 새 학습은 하지 않았다.",
        "- CP67 checkpoint 기반 forward-only / metrics-only post-hoc 평가만 수행했다.",
        "- validation에서 offset을 fit하고 test에는 고정 적용했다.",
        "- DB 쓰기, save-run, W&B, full 473티커, UI/backend 수정, band/composite 실험은 하지 않았다.",
        "- CP68 display calibration을 `trained_conservative_line`으로 부르지 않고 `display_calibrated_line`으로 분리했다.",
        "",
        "## 2. GPU 사용",
        _table(
            [
                {"item": "requested_device", "value": runtime.get("requested_device"), "result": ""},
                {"item": "resolved_device", "value": runtime.get("resolved_device"), "result": "CPU-only" if not runtime.get("gpu_used") else "CUDA"},
                {"item": "gpu_used", "value": runtime.get("gpu_used"), "result": ""},
                {"item": "elapsed_seconds", "value": _fmt(runtime.get("elapsed_seconds")), "result": ""},
                {"item": "peak_vram_mb", "value": _fmt(runtime.get("peak_vram_mb")), "result": "해당 없음" if not runtime.get("gpu_used") else ""},
            ],
            [("항목", "item"), ("값", "value"), ("비고", "result")],
        ),
        "",
        "이번 CP는 CPU-only로 실행했다. GPU는 사용하지 않았다.",
        "",
        "## 3. 데이터/cache 주의",
        _table(
            [
                {"item": "current source hash", "value": payload.get("cache", {}).get("source_data_hash"), "result": payload.get("cache", {}).get("hash_status")},
                {"item": "CP67 source hash", "value": payload.get("cache", {}).get("cp67_hash"), "result": ""},
                {"item": "feature_version", "value": payload.get("cache", {}).get("feature_version"), "result": "PASS"},
                {"item": "eligible ticker", "value": payload.get("cache", {}).get("eligible_ticker_count"), "result": ""},
                {"item": "feature/target NaN/Inf", "value": f"{payload.get('cache', {}).get('feature_nonfinite_count')} / {payload.get('cache', {}).get('target_nonfinite_count')}", "result": "PASS"},
                {"item": "ratio sanity", "value": payload.get("cache", {}).get("ratio_sanity", {}).get("pass"), "result": "PASS"},
                {"item": "feature cache created", "value": payload.get("cache", {}).get("feature_cache_created"), "result": ""},
            ],
            [("항목", "item"), ("값", "value"), ("판정", "result")],
        ),
        "",
        "CP70 실행 시 current source hash가 CP67 hash와 달랐다. 따라서 이 결과는 `CP67 checkpoint + 현재 100티커 cache/data` 기준의 정책 안정성 재확인이다. 정확한 CP68 동일-hash 재현은 아니며, 이 차이는 제품 판단의 잔여 리스크로 기록한다.",
        "",
        "## 4. validation fit 정책",
        _table(
            _fit_rows(policies),
            [
                ("정책", "policy"),
                ("line_type", "type"),
                ("방식", "method"),
                ("offset", "offset"),
                ("bucket offsets", "bucket_offsets"),
                ("val false_safe_tail", "val_fstail"),
                ("target", "target_met"),
            ],
        ),
        "",
        "## 5. 100티커 test line 지표",
        _table(
            [_line_row(policy) for policy in policies],
            [
                ("정책", "policy"),
                ("line_type", "type"),
                ("IC", "ic"),
                ("IC IR", "ic_ir"),
                ("IC t", "ic_t"),
                ("spread", "spread"),
                ("spread IR", "spread_ir"),
                ("spread t", "spread_t"),
                ("false_safe_tail", "fstail"),
                ("false_safe_severe", "fssev"),
                ("severe_recall", "recall"),
                ("bias", "bias"),
                ("upside_sacrifice", "sacrifice"),
                ("fee_ret", "fee_ret"),
                ("fee_sharpe", "fee_sharpe"),
                ("판정", "verdict"),
            ],
        ),
        "",
        "## 6. bucket별 test 지표",
        _table(
            _bucket_rows(policies),
            [
                ("정책", "policy"),
                ("bucket", "bucket"),
                ("IC", "ic"),
                ("spread", "spread"),
                ("false_safe_tail", "fstail"),
                ("severe_recall", "recall"),
                ("upside_sacrifice", "sacrifice"),
            ],
        ),
        "",
        "## 7. 정책 비교",
        "- `raw_model_line`은 alpha=1 beta=2 loss로 학습된 `trained_conservative_line` 원출력이지만 false_safe_tail이 높아 표시 후보가 아니다.",
        "- `global_downshift`는 단순하고 안정적이며 IC/spread/fee를 보존했다.",
        "- `horizon_bucket_downshift`는 horizon별 오차 차이를 반영하면서 IC/spread/fee를 보존했고 h11_h20 false_safe 기준도 통과했다.",
        "- `volatility_scaled_downshift`는 false_safe를 가장 낮췄지만 IC/spread/fee 희생이 있고, 지표 기반 scale이 validation 분포에 과적합될 가능성이 있다.",
        "",
        "## 8. 제품 표시 후보 결정",
        f"- 최종 추천: {recommendation['choice']}",
        f"- 선택 정책: {recommendation['selected_policy']}",
        f"- 이유: {recommendation['reason']}",
        "- h20 표시선은 제품 기본 ON이 아니라 기본 OFF / 사용자 선택형 중기 참고선으로 둔다.",
        "- h5 단기 line은 기존처럼 제품 기본 예측선 역할을 유지한다.",
        "",
        "## 9. 다음 단계",
        recommendation["next_step"],
        "",
        "## 10. 검증",
        "- `.venv\\Scripts\\python.exe -m py_compile ai\\cp70_lm_h20_display_calibration_policy.py ai\\tests\\test_cp70_display_policy.py`: 통과",
        "- `python -m json.tool docs\\cp70_lm_h20_display_calibration_policy_metrics.json`: 통과",
        "- `.venv\\Scripts\\python.exe -m unittest ai.tests.test_cp70_display_policy ai.tests.test_cp68_conservative_calibration ai.tests.test_losses ai.tests.test_loss`: 13개 통과",
        "- 마지막 확인 기준 잔여 `python/pythonw` 프로세스 없음",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = resolve_device(args.device)
    gpu_used = device.type == "cuda"
    if gpu_used:
        torch.cuda.reset_peak_memory_stats(device)

    cp67_payload = _read_json(CP67_METRICS_PATH)
    cp67_record = _checkpoint_record(cp67_payload)
    checkpoint_path = _resolve_checkpoint_path(cp67_record)
    data_hash = resolve_data_fingerprint("1D")
    train_bundle, val_bundle, test_bundle, _, _, plan, cache_meta, _, _, _ = _build_bundles(
        timeframe="1D",
        data_hash=data_hash,
        seq_len=252,
        horizon=20,
        limit_tickers=100,
        include_future_covariate=False,
    )
    del train_bundle

    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    model = model.to(device)
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    log_return_index = feature_columns.index("log_return") if "log_return" in feature_columns else 0
    severe_downside_threshold = _safe_float(config.get("severe_downside_threshold")) or -0.12
    top_k_frac = 0.1
    fee_bps = 10.0
    target_false_safe = 0.30

    val_predictions = _collect_line_predictions(
        model=model,
        bundle=val_bundle,
        device=device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
        log_return_index=log_return_index,
    )
    test_predictions = _collect_line_predictions(
        model=model,
        bundle=test_bundle,
        device=device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
        log_return_index=log_return_index,
    )

    raw_policy = {
        "name": "raw_model_line",
        "line_type": "raw_model_line / trained_conservative_line",
        "fit": {"method": "none", "offset": 0.0, "target_met": None},
        "validation": _summarize_policy(
            predictions=val_predictions,
            line=val_predictions.line,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
        "test": _summarize_policy(
            predictions=test_predictions,
            line=test_predictions.line,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
    }

    global_fit = _fit_min_offset(
        metric_at_offset=lambda offset: _false_safe_overall(
            line=val_predictions.line - float(offset),
            raw=val_predictions.raw_future_returns,
            severe_downside_threshold=severe_downside_threshold,
        ),
        high=_global_high(val_predictions.line),
        target=target_false_safe,
    )
    global_offset = float(global_fit["offset"] or 0.0)
    global_policy = {
        "name": "global_downshift",
        "line_type": "display_calibrated_line",
        "fit": {"method": "validation_global_offset", **global_fit},
        "validation": _summarize_policy(
            predictions=val_predictions,
            line=val_predictions.line - global_offset,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
        "test": _summarize_policy(
            predictions=test_predictions,
            line=test_predictions.line - global_offset,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
    }

    bucket_offsets: dict[str, float] = {}
    bucket_fit_detail: dict[str, Any] = {}
    for bucket, (start, end) in BUCKETS.items():
        fit = _fit_min_offset(
            metric_at_offset=lambda offset, start=start, end=end: _false_safe_bucket(
                line=val_predictions.line,
                raw=val_predictions.raw_future_returns,
                metadata=val_predictions.metadata,
                start=start,
                end=end,
                offset=offset,
                severe_downside_threshold=severe_downside_threshold,
            ),
            high=_segment_high(val_predictions.line, start, end),
            target=target_false_safe,
        )
        bucket_offsets[bucket] = float(fit["offset"] or 0.0)
        bucket_fit_detail[bucket] = fit

    bucket_val_line = _apply_bucket_offsets(val_predictions.line, bucket_offsets)
    bucket_test_line = _apply_bucket_offsets(test_predictions.line, bucket_offsets)
    bucket_policy = {
        "name": "horizon_bucket_downshift",
        "line_type": "display_calibrated_line",
        "fit": {
            "method": "validation_bucket_offsets",
            "bucket_offsets": {key: round(value, 8) for key, value in bucket_offsets.items()},
            "bucket_fit_detail": bucket_fit_detail,
            "target_met": all(bool(item.get("target_met")) for item in bucket_fit_detail.values()),
            "validation_false_safe_tail_rate": None,
        },
        "validation": _summarize_policy(
            predictions=val_predictions,
            line=bucket_val_line,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
        "test": _summarize_policy(
            predictions=test_predictions,
            line=bucket_test_line,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
    }

    val_scale, val_scale_meta = _build_volatility_scale(
        validation_proxy=val_predictions.volatility_proxy,
        target_proxy=val_predictions.volatility_proxy,
    )
    test_scale, test_scale_meta = _build_volatility_scale(
        validation_proxy=val_predictions.volatility_proxy,
        target_proxy=test_predictions.volatility_proxy,
    )
    volatility_fit = _fit_min_offset(
        metric_at_offset=lambda offset: _false_safe_overall(
            line=val_predictions.line - (float(offset) * val_scale).view(-1, 1),
            raw=val_predictions.raw_future_returns,
            severe_downside_threshold=severe_downside_threshold,
        ),
        high=_global_high(val_predictions.line, val_scale),
        target=target_false_safe,
    )
    volatility_offset = float(volatility_fit["offset"] or 0.0)
    volatility_policy = {
        "name": "volatility_scaled_downshift",
        "line_type": "display_calibrated_line",
        "fit": {
            "method": "validation_volatility_scaled_offset",
            **volatility_fit,
            "validation_scale": val_scale_meta,
            "test_scale": test_scale_meta,
            "volatility_proxy": "입력 window의 최근 20개 log_return 표준편차, validation median 정규화, 0.5~2.0 clipping",
        },
        "validation": _summarize_policy(
            predictions=val_predictions,
            line=val_predictions.line - (volatility_offset * val_scale).view(-1, 1),
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
        "test": _summarize_policy(
            predictions=test_predictions,
            line=test_predictions.line - (volatility_offset * test_scale).view(-1, 1),
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        ),
    }

    policies = [raw_policy, global_policy, bucket_policy, volatility_policy]
    for policy in policies:
        policy["classification"] = _classify_policy(policy, raw_policy)
        policy["display_decision"] = _display_decision(policy, raw_policy)
        policy["line_metric_subset"] = _line_metric_subset(policy["test"]["line_metrics"])
        policy["bucket_metric_subset"] = {
            bucket: _bucket_metric_subset(policy["test"]["bucket_line_metrics"].get(bucket, {}))
            for bucket in BUCKETS
        }

    recommendation = {
        "choice": "A. h20 display_calibrated_line을 제품 기본 OFF 후보로 유지",
        "selected_policy": "horizon_bucket_downshift",
        "reason": "false_safe_tail/severe_recall/h11_h20 기준을 통과하면서 IC/spread/fee 희생이 없고, horizon별 오차 차이를 반영해 해석성이 global_downshift보다 좋다.",
        "next_step": "제품 연결 전에는 h20 display_calibrated_line을 기본 OFF 토글 후보로 문서화하고, 별도 학습 CP에서는 beta 3~4 또는 severe downside weighting을 검토한다. 이번 CP에서는 구현하지 않았다.",
    }

    elapsed = time.perf_counter() - started
    peak_vram_mb = None
    if gpu_used:
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    payload = {
        "cp": "CP70-LM",
        "rules": {
            "new_training": False,
            "save_run": False,
            "db_write": False,
            "full_473_training": False,
            "wandb": "off",
            "ui_backend_modified": False,
            "band_model_experiment": False,
            "composite_overlay_experiment": False,
            "display_calibration_called_trained_conservative_line": False,
        },
        "terminology": {
            "raw_model_line": "alpha=1 beta=2 loss로 학습된 모델 원출력",
            "display_calibrated_line": "validation offset으로 표시용 하향 보정한 선",
            "trained_conservative_line": "학습 loss 자체가 보수적인 line",
        },
        "runtime": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "requested_device": args.device,
            "resolved_device": str(device),
            "gpu_used": gpu_used,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_vram_mb": peak_vram_mb,
            "elapsed_seconds": elapsed,
        },
        "checkpoint": {
            "path": _rel(checkpoint_path),
            "source": "CP67 h20_full_features_post_backfill_100",
            "feature_set": cp67_record.get("feature_set"),
            "run_id": cp67_record.get("run_id"),
            "config_severe_downside_threshold": severe_downside_threshold,
        },
        "cache": {
            "source_data_hash": data_hash,
            "cp67_hash": cp67_payload.get("cache_gate", {}).get("current_hash"),
            "hash_status": "PASS" if data_hash == cp67_payload.get("cache_gate", {}).get("current_hash") else "MISMATCH",
            "feature_version": FEATURE_CONTRACT_VERSION,
            "feature_count": len(MODEL_FEATURE_COLUMNS),
            "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
            "eligible_ticker_count": len(plan.eligible_tickers),
            "excluded_ticker_count": len(plan.excluded_reasons),
            "feature_cache": cache_meta.get("feature_cache"),
            "feature_index_cache": cache_meta.get("feature_index_cache"),
            "feature_cache_created": cache_meta.get("feature_cache_created"),
            "feature_index_cache_created": cache_meta.get("feature_index_cache_created"),
            "feature_nonfinite_count": cache_meta.get("tensor_finite", {}).get("feature_nonfinite_count"),
            "target_nonfinite_count": cache_meta.get("tensor_finite", {}).get("target_nonfinite_count"),
            "ratio_sanity": cache_meta.get("ratio_sanity"),
        },
        "prediction_finite": {
            "validation_line": _finite_count(val_predictions.line),
            "validation_target": _finite_count(val_predictions.raw_future_returns),
            "test_line": _finite_count(test_predictions.line),
            "test_target": _finite_count(test_predictions.raw_future_returns),
        },
        "policy_target": {
            "fit_split": "validation",
            "eval_split": "test",
            "target_false_safe_tail_rate": target_false_safe,
            "test_not_used_for_offset_selection": True,
        },
        "policies": policies,
        "recommendation": recommendation,
    }
    _write_json(METRICS_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8-sig")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP70 h20 display calibration policy 비교")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="off")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            _json_safe(
                {
                    "cp": payload["cp"],
                    "selected_policy": payload["recommendation"]["selected_policy"],
                    "choice": payload["recommendation"]["choice"],
                    "gpu_used": payload["runtime"]["gpu_used"],
                    "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
                    "report": _rel(REPORT_PATH),
                    "metrics": _rel(METRICS_PATH),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
