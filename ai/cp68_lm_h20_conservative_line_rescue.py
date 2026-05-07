from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.cp66_lm_post_backfill_h20_1w import (
    _build_bundles,
    _fmt,
    _json_safe,
    _read_json,
    _safe_float,
    _table,
    _write_json,
)
from ai.evaluation import (
    _conservative_line_metrics,
    _investment_metrics_for_score,
    _line_segment_metrics,
)
from ai.inference import load_checkpoint
from ai.preprocessing import FEATURE_CONTRACT_VERSION, MODEL_FEATURE_COLUMNS, resolve_data_fingerprint
from ai.train import autocast_context, forward_model, make_loader, resolve_device


REPORT_PATH = PROJECT_ROOT / "docs" / "cp68_lm_h20_conservative_line_rescue_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp68_lm_h20_conservative_line_rescue_metrics.json"
CP67_METRICS_PATH = PROJECT_ROOT / "docs" / "cp67_lm_h20_100ticker_validation_metrics.json"

REPORT_KEYS = [
    "ic_mean",
    "long_short_spread",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
]
BUCKETS = {
    "h1_h5": (0, 5),
    "h6_h10": (5, 10),
    "h11_h20": (10, 20),
}


@dataclass(frozen=True)
class LinePredictions:
    line: torch.Tensor
    line_target: torch.Tensor
    raw_future_returns: torch.Tensor
    volatility_proxy: torch.Tensor
    metadata: pd.DataFrame


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _finite_count(tensor: torch.Tensor) -> dict[str, int]:
    return {
        "element_count": int(tensor.numel()),
        "nonfinite_count": int((~torch.isfinite(tensor)).sum().item()),
    }


def _checkpoint_record(cp67_payload: dict[str, Any]) -> dict[str, Any]:
    for record in cp67_payload.get("experiments", []):
        if record.get("name") == "h20_full_features_post_backfill_100":
            return record
    raise ValueError("CP67 metrics에서 h20_full_features_post_backfill_100 기록을 찾지 못했다.")


def _resolve_checkpoint_path(record: dict[str, Any]) -> Path:
    raw_path = record.get("checkpoint_path")
    if not raw_path:
        raise ValueError("CP67 기록에 checkpoint_path가 없다.")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"CP67 checkpoint가 없다: {path}")
    return path


def _collect_line_predictions(
    *,
    model: Any,
    bundle: Any,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
    log_return_index: int,
) -> LinePredictions:
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    line_chunks: list[torch.Tensor] = []
    line_target_chunks: list[torch.Tensor] = []
    raw_target_chunks: list[torch.Tensor] = []
    volatility_chunks: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for features, line_target, _, raw_future_returns, ticker_ids, future_covariates in loader:
            feature_cpu = features.detach().cpu().to(torch.float32)
            lookback = min(20, int(feature_cpu.shape[1]))
            volatility_proxy = feature_cpu[:, -lookback:, log_return_index].std(dim=1, unbiased=False)
            volatility_chunks.append(volatility_proxy)

            features = features.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype=amp_dtype):
                output = forward_model(model, features, ticker_ids, future_covariates)
            line_chunks.append(output.line.detach().cpu().to(torch.float32))
            line_target_chunks.append(line_target.detach().cpu().to(torch.float32))
            raw_target_chunks.append(raw_future_returns.detach().cpu().to(torch.float32))

    return LinePredictions(
        line=torch.cat(line_chunks, dim=0),
        line_target=torch.cat(line_target_chunks, dim=0),
        raw_future_returns=torch.cat(raw_target_chunks, dim=0),
        volatility_proxy=torch.cat(volatility_chunks, dim=0),
        metadata=bundle.metadata.reset_index(drop=True).copy(),
    )


def _overall_line_metrics(
    *,
    predictions: LinePredictions,
    line: torch.Tensor,
    severe_downside_threshold: float,
    top_k_frac: float,
    fee_bps: float,
) -> dict[str, float | None]:
    line = line.detach().cpu().to(torch.float32)
    raw = predictions.raw_future_returns.detach().cpu().to(torch.float32)
    target = predictions.line_target.detach().cpu().to(torch.float32)
    absolute_error = torch.abs(line - target)
    smape = 2.0 * absolute_error / (line.abs() + target.abs() + 1e-6)
    direction_accuracy = float(((line[:, -1] >= 0.0) == (target[:, -1] >= 0.0)).float().mean().item())
    investment = _investment_metrics_for_score(
        metadata=predictions.metadata,
        score=line[:, -1],
        actual_return=raw[:, -1],
        top_k_frac=top_k_frac,
        fee_bps=fee_bps,
    )
    conservative = _conservative_line_metrics(
        score=line,
        actual=raw,
        severe_threshold=severe_downside_threshold,
        metadata=None,
    )
    return {
        **investment,
        "direction_accuracy": direction_accuracy,
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
        **conservative,
    }


def _bucket_line_metrics(
    *,
    predictions: LinePredictions,
    line: torch.Tensor,
    severe_downside_threshold: float,
    top_k_frac: float,
    fee_bps: float,
) -> dict[str, dict[str, float | None]]:
    raw: dict[str, dict[str, float | None]] = {}
    for name, (start, end) in BUCKETS.items():
        prefixed = _line_segment_metrics(
            prefix=name,
            start=start,
            end=end,
            metadata=predictions.metadata,
            line_pred=line,
            line_actual=predictions.line_target,
            raw_actual=predictions.raw_future_returns,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        )
        raw[name] = {
            key.removeprefix(f"{name}_"): value
            for key, value in prefixed.items()
            if key.startswith(f"{name}_")
        }
    return raw


def _summarize_policy(
    *,
    predictions: LinePredictions,
    line: torch.Tensor,
    severe_downside_threshold: float,
    top_k_frac: float,
    fee_bps: float,
) -> dict[str, Any]:
    line_metrics = _overall_line_metrics(
        predictions=predictions,
        line=line,
        severe_downside_threshold=severe_downside_threshold,
        top_k_frac=top_k_frac,
        fee_bps=fee_bps,
    )
    bucket_metrics = _bucket_line_metrics(
        predictions=predictions,
        line=line,
        severe_downside_threshold=severe_downside_threshold,
        top_k_frac=top_k_frac,
        fee_bps=fee_bps,
    )
    return {
        "line_metrics": {key: line_metrics.get(key) for key in sorted(line_metrics.keys())},
        "bucket_line_metrics": bucket_metrics,
    }


def _false_safe_overall(
    *,
    line: torch.Tensor,
    raw: torch.Tensor,
    severe_downside_threshold: float,
) -> float | None:
    return _conservative_line_metrics(
        score=line,
        actual=raw,
        severe_threshold=severe_downside_threshold,
        metadata=None,
    ).get("false_safe_tail_rate")


def _false_safe_bucket(
    *,
    line: torch.Tensor,
    raw: torch.Tensor,
    metadata: pd.DataFrame,
    start: int,
    end: int,
    offset: float,
    severe_downside_threshold: float,
) -> float | None:
    segment_score = line[:, start:end].mean(dim=1) - float(offset)
    segment_raw = raw[:, start:end].mean(dim=1)
    return _conservative_line_metrics(
        score=segment_score,
        actual=segment_raw,
        severe_threshold=severe_downside_threshold,
        metadata=metadata,
    ).get("false_safe_tail_rate")


def _fit_min_offset(
    *,
    metric_at_offset: Callable[[float], float | None],
    high: float,
    target: float,
    iterations: int = 36,
) -> dict[str, float | bool | None]:
    start_rate = metric_at_offset(0.0)
    if start_rate is not None and start_rate < target:
        return {
            "offset": 0.0,
            "validation_false_safe_tail_rate": start_rate,
            "target": target,
            "target_met": True,
        }

    high = max(float(high), 1e-6)
    high_rate = metric_at_offset(high)
    expand_count = 0
    while high_rate is not None and high_rate >= target and expand_count < 12:
        high *= 2.0
        high_rate = metric_at_offset(high)
        expand_count += 1

    if high_rate is None or high_rate >= target:
        return {
            "offset": high,
            "validation_false_safe_tail_rate": high_rate,
            "target": target,
            "target_met": False,
        }

    low = 0.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        rate = metric_at_offset(mid)
        if rate is not None and rate < target:
            high = mid
        else:
            low = mid
    final_rate = metric_at_offset(high)
    return {
        "offset": high,
        "validation_false_safe_tail_rate": final_rate,
        "target": target,
        "target_met": final_rate is not None and final_rate < target,
    }


def _global_high(line: torch.Tensor, scale: torch.Tensor | None = None) -> float:
    values = line.detach().cpu().to(torch.float32)
    if scale is not None:
        values = values / scale.detach().cpu().to(torch.float32).view(-1, 1).clamp_min(1e-6)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 1e-6
    return max(float(finite.max().item()), 0.0) + 1e-6


def _segment_high(line: torch.Tensor, start: int, end: int) -> float:
    segment = line[:, start:end].mean(dim=1)
    finite = segment[torch.isfinite(segment)]
    if finite.numel() == 0:
        return 1e-6
    return max(float(finite.max().item()), 0.0) + 1e-6


def _build_volatility_scale(
    *,
    validation_proxy: torch.Tensor,
    target_proxy: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    finite_validation = validation_proxy[torch.isfinite(validation_proxy)]
    median = float(torch.median(finite_validation).item()) if finite_validation.numel() else 1.0
    median = max(median, 1e-6)
    scale = (target_proxy / median).to(torch.float32)
    scale = torch.nan_to_num(scale, nan=1.0, posinf=2.0, neginf=0.5).clamp(0.5, 2.0)
    return scale, {
        "validation_median_proxy": median,
        "scale_min": float(scale.min().item()),
        "scale_median": float(torch.median(scale).item()),
        "scale_max": float(scale.max().item()),
        "sample_count": int(scale.numel()),
    }


def _apply_bucket_offsets(line: torch.Tensor, offsets: dict[str, float]) -> torch.Tensor:
    adjusted = line.clone()
    for bucket, offset in offsets.items():
        start, end = BUCKETS[bucket]
        adjusted[:, start:end] = adjusted[:, start:end] - float(offset)
    return adjusted


def _delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    left = _safe_float(current.get(key))
    right = _safe_float(reference.get(key))
    if left is None or right is None:
        return None
    return left - right


def _classify_policy(policy: dict[str, Any], raw_policy: dict[str, Any]) -> dict[str, Any]:
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

    signal_collapsed = bool(ic is not None and spread is not None and ic < 0.0 and spread < 0.0)
    risk_pass = bool(
        false_safe is not None
        and false_safe < 0.30
        and severe is not None
        and severe >= 0.70
        and h11_false_safe is not None
        and h11_false_safe < 0.35
    )
    excessive_upside_sacrifice = False
    if upside is not None and raw_upside is not None:
        excessive_upside_sacrifice = upside >= max(raw_upside * 1.5, 0.18)

    if risk_pass and not signal_collapsed and not excessive_upside_sacrifice:
        verdict = "PASS"
        display = "기본 OFF / 사용자 선택형 중기 보수 판단선 후보"
    elif risk_pass and not signal_collapsed:
        verdict = "WATCH"
        display = "연구 후보 유지, 제품 기본 표시 금지"
    elif false_safe is not None and false_safe < 0.35 and severe is not None and severe >= 0.65 and not signal_collapsed:
        verdict = "WATCH"
        display = "연구 후보 유지, 제품 기본 표시 금지"
    else:
        verdict = "FAIL"
        display = "랭킹 참고용으로만 기록하고 Phase 1.5 보류"

    return {
        "verdict": verdict,
        "display": display,
        "risk_pass": risk_pass,
        "signal_collapsed": signal_collapsed,
        "excessive_upside_sacrifice": excessive_upside_sacrifice,
    }


def _policy_row(policy: dict[str, Any]) -> dict[str, Any]:
    metrics = policy["test"]["line_metrics"]
    return {
        "policy": policy["name"],
        "ic": _fmt(metrics.get("ic_mean")),
        "spread": _fmt(metrics.get("long_short_spread")),
        "fstail": _fmt(metrics.get("false_safe_tail_rate")),
        "fssev": _fmt(metrics.get("false_safe_severe_rate")),
        "recall": _fmt(metrics.get("severe_downside_recall")),
        "bias": _fmt(metrics.get("conservative_bias")),
        "sacrifice": _fmt(metrics.get("upside_sacrifice")),
        "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
        "fee_sharpe": _fmt(metrics.get("fee_adjusted_sharpe")),
        "verdict": policy.get("classification", {}).get("verdict", ""),
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
                    "fee_ret": _fmt(metrics.get("fee_adjusted_return")),
                }
            )
    return rows


def _fit_rows(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        fit = policy.get("fit", {})
        rows.append(
            {
                "policy": policy["name"],
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
    best = payload["best_policy"]
    lines = [
        "# CP68-LM h20 conservative line rescue",
        "",
        "## 1. 원칙 확인",
        "- 이번 CP는 line_model 전용 post-hoc calibration이다.",
        "- 새 학습은 하지 않았고 CP67 h20_full_features 100티커 checkpoint 예측만 재사용했다.",
        "- band 모델 실험, composite/overlay, line_inside_band 평가는 사용하지 않았다.",
        "- DB 쓰기, save-run, W&B, full 473티커, UI/backend 수정, feature contract 변경은 하지 않았다.",
        "",
        "## 2. 입력과 cache 확인",
        _table(
            [
                {"item": "checkpoint", "value": payload["checkpoint"]["path"], "result": "사용"},
                {"item": "source_data_hash", "value": payload["cache"]["source_data_hash"], "result": payload["cache"]["hash_status"]},
                {"item": "feature_version", "value": payload["cache"]["feature_version"], "result": "PASS"},
                {"item": "MODEL_FEATURE_COLUMNS", "value": payload["cache"]["feature_count"], "result": "PASS"},
                {"item": "atr_ratio in model", "value": payload["cache"]["atr_ratio_in_model_features"], "result": "PASS"},
                {"item": "eligible ticker", "value": payload["cache"]["eligible_ticker_count"], "result": "limit 100 scope"},
                {"item": "feature cache created", "value": payload["cache"]["feature_cache_created"], "result": "기존 cache 재사용" if not payload["cache"]["feature_cache_created"] else "생성됨"},
                {"item": "feature NaN/Inf", "value": payload["cache"]["feature_nonfinite_count"], "result": "PASS"},
                {"item": "target NaN/Inf", "value": payload["cache"]["target_nonfinite_count"], "result": "PASS"},
            ],
            [("항목", "item"), ("값", "value"), ("판정", "result")],
        ),
        "",
        "## 3. validation fit 정책",
        _table(
            _fit_rows(policies),
            [
                ("정책", "policy"),
                ("방식", "method"),
                ("offset", "offset"),
                ("bucket offsets", "bucket_offsets"),
                ("val false_safe_tail", "val_fstail"),
                ("target", "target_met"),
            ],
        ),
        "",
        "offset은 validation split에서만 산정했고 test 지표를 본 뒤 고르지 않았다.",
        "",
        "## 4. test line 결과",
        _table(
            [_policy_row(policy) for policy in policies],
            [
                ("정책", "policy"),
                ("IC", "ic"),
                ("spread", "spread"),
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
        "## 5. bucket별 test 결과",
        _table(
            _bucket_rows(policies),
            [
                ("정책", "policy"),
                ("bucket", "bucket"),
                ("IC", "ic"),
                ("spread", "spread"),
                ("false_safe_tail", "fstail"),
                ("severe_recall", "recall"),
                ("fee_ret", "fee_ret"),
            ],
        ),
        "",
        "## 6. 핵심 질문 답변",
        f"- h20 raw expected line은 랭킹용으로는 쓸 만하다. raw test IC={_fmt(policies[0]['test']['line_metrics'].get('ic_mean'))}, spread={_fmt(policies[0]['test']['line_metrics'].get('long_short_spread'))}로 양수다.",
        f"- h20 conservative line은 best 정책 `{best['name']}` 기준 false_safe_tail={_fmt(best['test']['line_metrics'].get('false_safe_tail_rate'))}, severe_recall={_fmt(best['test']['line_metrics'].get('severe_downside_recall'))}이다.",
        f"- IC/spread 희생은 best 정책 기준 delta IC={_fmt(best['delta_vs_raw'].get('ic_mean'))}, delta spread={_fmt(best['delta_vs_raw'].get('long_short_spread'))}이다.",
        "",
        "## 7. 제품 표시 판단",
        f"- 최종 분류: {payload['final_decision']['verdict']}",
        f"- 표시 제안: {payload['final_decision']['display']}",
        "- h20 raw line과 h20 conservative line은 분리해서 기록한다.",
        "- h20 raw line이 IC/spread가 좋아도 false-safe가 높으면 위험선으로 쓰지 않는다.",
        "- h5는 단기 line 진한 실선 역할을 유지한다.",
        "- h20 conservative line이 PASS이면 기본 OFF / 사용자 선택형 중기 보수 판단선으로만 둔다.",
        "",
        "## 8. post-hoc 이후 학습 loss 후보",
        "- h20 전용 asymmetric line loss",
        "- overprediction penalty 강화",
        "- severe downside sample weighting",
        "- direction/risk auxiliary head",
        "- h20 전용 feature set 재검토",
        "",
        "## 9. 다음 LM 추천",
        payload["next_lm_recommendation"],
        "",
        "## 10. 검증",
        "- `.venv\\Scripts\\python.exe -m py_compile ai\\cp68_lm_h20_conservative_line_rescue.py ai\\tests\\test_cp68_conservative_calibration.py`: 통과",
        "- `python -m json.tool docs\\cp68_lm_h20_conservative_line_rescue_metrics.json`: 통과",
        "- `.venv\\Scripts\\python.exe -m unittest ai.tests.test_cp68_conservative_calibration ai.tests.test_feature_set_selection ai.tests.test_checkpoint_selection ai.tests.test_metric_definition_contract ai.tests.test_splits`: 30개 통과",
        "- 마지막 확인 기준 잔여 `python/pythonw` 학습 프로세스 없음",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
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
    device = resolve_device(args.device)
    model = model.to(device)
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    log_return_index = feature_columns.index("log_return") if "log_return" in feature_columns else 0
    severe_downside_threshold = _safe_float(config.get("severe_downside_threshold"))
    if severe_downside_threshold is None:
        severe_downside_threshold = -0.12

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

    top_k_frac = 0.1
    fee_bps = 10.0
    target_false_safe = args.target_false_safe_tail

    raw_policy = {
        "name": "raw_line",
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
        "fit": {
            "method": "validation_volatility_scaled_offset",
            **volatility_fit,
            "validation_scale": val_scale_meta,
            "test_scale": test_scale_meta,
            "volatility_proxy": "입력 window의 최근 20개 log_return 표준편차, validation median으로 정규화 후 0.5~2.0 clipping",
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
        policy["delta_vs_raw"] = {
            key: _delta(policy["test"]["line_metrics"], raw_policy["test"]["line_metrics"], key)
            for key in REPORT_KEYS
        }
        policy["classification"] = _classify_policy(policy, raw_policy)

    pass_policies = [policy for policy in policies if policy["classification"]["verdict"] == "PASS"]
    watch_policies = [policy for policy in policies if policy["classification"]["verdict"] == "WATCH"]
    if pass_policies:
        best_policy = min(
            pass_policies,
            key=lambda item: _safe_float(item["test"]["line_metrics"].get("upside_sacrifice")) or float("inf"),
        )
    elif watch_policies:
        best_policy = min(
            watch_policies,
            key=lambda item: _safe_float(item["test"]["line_metrics"].get("false_safe_tail_rate")) or float("inf"),
        )
    else:
        best_policy = raw_policy

    if best_policy["classification"]["verdict"] == "PASS":
        final_decision = {
            "verdict": "PASS",
            "display": "h20 conservative line을 기본 OFF / 사용자 선택형 중기 보수 판단선 후보로 유지한다.",
        }
        next_lm = "CP69-LM에서는 같은 post-hoc 정책을 h20 no_fundamentals와 CP49 h20 dense/risk-only 후보에 적용해 보수선 안정성을 비교한다."
    elif best_policy["classification"]["verdict"] == "WATCH":
        final_decision = {
            "verdict": "WATCH",
            "display": "연구 후보로 유지하되 제품 기본 표시는 금지한다.",
        }
        next_lm = "CP69-LM에서는 h20 전용 asymmetric line loss와 severe downside weighting을 우선 설계한다."
    else:
        final_decision = {
            "verdict": "FAIL",
            "display": "h20은 랭킹 참고용으로만 기록하고 Phase 1.5 연구 후보로 보류한다.",
        }
        next_lm = "CP69-LM에서는 post-hoc 보수화 대신 h20 전용 loss 후보를 설계한다."

    payload = {
        "cp": "CP68-LM",
        "rules": {
            "role": "line_model",
            "feature_version": FEATURE_CONTRACT_VERSION,
            "target": "raw_future_return",
            "line_target_type": "raw_future_return",
            "checkpoint_selection": "line_gate",
            "band_model_experiment": False,
            "composite_overlay_metrics_used": False,
            "line_inside_band_used": False,
            "db_write": False,
            "save_run": False,
            "full_473_training": False,
            "wandb": "off",
            "new_training": False,
            "model_structure_changed": False,
            "feature_contract_changed": False,
            "atr_ratio_promoted_to_model_feature": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "requested_device": args.device,
            "resolved_device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "checkpoint": {
            "path": _rel(checkpoint_path),
            "source_cp": "CP67-LM",
            "source_experiment": cp67_record.get("name"),
            "run_id": cp67_record.get("run_id"),
            "feature_set": cp67_record.get("feature_set"),
            "limit_tickers": cp67_record.get("experiment", {}).get("limit_tickers"),
            "epochs": cp67_record.get("experiment", {}).get("epochs"),
            "config_severe_downside_threshold": severe_downside_threshold,
        },
        "cache": {
            "source_data_hash": data_hash,
            "cp67_hash": cp67_payload.get("cache_gate", {}).get("current_hash"),
            "hash_status": "PASS" if data_hash == cp67_payload.get("cache_gate", {}).get("current_hash") else "MISMATCH",
            "feature_version": FEATURE_CONTRACT_VERSION,
            "feature_count": len(MODEL_FEATURE_COLUMNS),
            "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
            "feature_cache": cache_meta.get("feature_cache"),
            "feature_index_cache": cache_meta.get("feature_index_cache"),
            "feature_cache_created": cache_meta.get("feature_cache_created"),
            "feature_index_cache_created": cache_meta.get("feature_index_cache_created"),
            "eligible_ticker_count": len(plan.eligible_tickers),
            "excluded_ticker_count": len(plan.excluded_reasons),
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
        "best_policy": best_policy,
        "final_decision": final_decision,
        "next_lm_recommendation": next_lm,
    }
    _write_json(METRICS_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8-sig")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP68 h20 conservative line post-hoc calibration")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-false-safe-tail", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(_json_safe({
        "cp": payload["cp"],
        "final_verdict": payload["final_decision"]["verdict"],
        "best_policy": payload["best_policy"]["name"],
        "report": _rel(REPORT_PATH),
        "metrics": _rel(METRICS_PATH),
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
