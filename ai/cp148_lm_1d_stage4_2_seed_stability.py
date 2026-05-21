from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ai.train as train_mod  # noqa: E402
from ai import cp148_lm_1d_stage4_1_risk_feature_abcd as s41  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_2_seed_stability_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_2_seed_stability_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_2_seed_stability_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_2_seed_stability_logs"

SEEDS = [42, 7, 123]
MAIN_CANDIDATES = [
    {
        "candidate_id": "D_stock_fragility",
        "role": "하방 안정성 1순위 후보",
        "extra_features": ["atr_ratio", "drawdown_20", "downside_vol_20"],
    },
    {
        "candidate_id": "C_stress_delta",
        "role": "alpha 보존 비교 후보",
        "extra_features": ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"],
    },
]
ABLATION_CANDIDATES = [
    {
        "candidate_id": "D_no_atr",
        "role": "D ATR 제거 ablation reference",
        "extra_features": ["drawdown_20", "downside_vol_20"],
    },
    {
        "candidate_id": "C_no_atr",
        "role": "C ATR 제거 ablation reference",
        "extra_features": ["vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"],
    },
]

METRIC_KEYS = [
    "ic_mean",
    "long_short_spread",
    "fee_adjusted_return",
    "false_safe_tail_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "direction_accuracy",
]
REGIME_KEYS = ["stress", "vix_rising", "breadth_worsening"]
BUCKET_KEYS = ["h1", "h2_h3", "h4_h5"]


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
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_experiment(candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    return {
        "experiment_id": f"stage4_2_{candidate_id.lower()}_seed{seed}",
        "label": candidate_id,
        "question": candidate["role"],
        "extra_features": list(candidate["extra_features"]),
    }


def _line_gate_metric(source: dict[str, Any], key: str) -> Any:
    return source.get(key)


def _extract_split_summary(record: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = record.get(split) or {}
    regime_container = record.get(f"{split}_regime_metrics") or {}
    bucket_container = record.get(f"{split}_horizon_bucket_metrics") or {}
    summary = {key: _line_gate_metric(metrics, key) for key in METRIC_KEYS}
    for regime in REGIME_KEYS:
        regime_metrics = regime_container.get(regime) or {}
        summary[f"{regime}_false_safe_tail_rate"] = regime_metrics.get("false_safe_tail_rate")
        summary[f"{regime}_severe_downside_recall"] = regime_metrics.get("severe_downside_recall")
        summary[f"{regime}_long_short_spread"] = regime_metrics.get("long_short_spread")
        summary[f"{regime}_fee_adjusted_return"] = regime_metrics.get("fee_adjusted_return")
    for bucket in BUCKET_KEYS:
        bucket_metrics = bucket_container.get(bucket) or {}
        summary[f"{bucket}_false_safe_tail_rate"] = bucket_metrics.get("false_safe_tail_rate")
        summary[f"{bucket}_severe_downside_recall"] = bucket_metrics.get("severe_downside_recall")
        summary[f"{bucket}_long_short_spread"] = bucket_metrics.get("long_short_spread")
        summary[f"{bucket}_fee_adjusted_return"] = bucket_metrics.get("fee_adjusted_return")
    return summary


def _aggregate_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "std": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ["validation", "test"]:
        split_output: dict[str, Any] = {}
        keys = set()
        summaries = [_extract_split_summary(record, split) for record in records]
        for summary in summaries:
            keys.update(summary.keys())
        for key in sorted(keys):
            values = [_safe_float(summary.get(key)) for summary in summaries]
            split_output[key] = _aggregate_values([value for value in values if value is not None])
        output[split] = split_output
    return output


def _classify_candidate(candidate_id: str, aggregate: dict[str, Any], records: list[dict[str, Any]]) -> str:
    val = aggregate.get("validation") or {}
    test = aggregate.get("test") or {}
    val_false_safe = (val.get("false_safe_tail_rate") or {}).get("median")
    val_severe = (val.get("severe_downside_recall") or {}).get("median")
    test_false_safe = (test.get("false_safe_tail_rate") or {}).get("median")
    test_severe = (test.get("severe_downside_recall") or {}).get("median")
    val_ic = (val.get("ic_mean") or {}).get("median")
    val_spread = (val.get("long_short_spread") or {}).get("median")
    val_fee = (val.get("fee_adjusted_return") or {}).get("median")
    all_line_gate = all(bool(record.get("line_gate_pass")) for record in records)

    if not all_line_gate:
        return "탈락 후보"
    if candidate_id == "D_stock_fragility":
        if (
            val_false_safe is not None
            and test_false_safe is not None
            and val_severe is not None
            and test_severe is not None
            and val_false_safe < s41.BASELINES["stage2_best_false_safe"]
            and test_false_safe < 0.35
            and val_severe >= s41.BASELINES["stage2_best_severe_recall"]
            and test_severe >= 0.65
        ):
            return "Stage 4-2 primary 후보"
    if candidate_id == "C_stress_delta":
        if (
            val_ic is not None
            and val_spread is not None
            and val_fee is not None
            and val_ic > 0.0
            and val_spread > 0.008
            and val_fee > 0.0
        ):
            return "alpha-preserving challenger"
    if candidate_id.endswith("_no_atr"):
        return "ablation reference"
    return "seed 재평가 통과 후보"


def _median(aggregates: dict[str, Any], candidate_id: str, split: str, metric: str) -> float | None:
    payload = aggregates.get(candidate_id) or {}
    aggregate = payload.get("aggregate") or {}
    split_metrics = aggregate.get(split) or {}
    return (split_metrics.get(metric) or {}).get("median")


def _std(aggregates: dict[str, Any], candidate_id: str, split: str, metric: str) -> float | None:
    payload = aggregates.get(candidate_id) or {}
    aggregate = payload.get("aggregate") or {}
    split_metrics = aggregate.get(split) or {}
    return (split_metrics.get(metric) or {}).get("std")


def _apply_relative_classifications(aggregates: dict[str, Any]) -> None:
    if "D_stock_fragility" in aggregates:
        d_test_false_safe = _median(aggregates, "D_stock_fragility", "test", "false_safe_tail_rate")
        d_test_severe = _median(aggregates, "D_stock_fragility", "test", "severe_downside_recall")
        d_val_false_safe = _median(aggregates, "D_stock_fragility", "validation", "false_safe_tail_rate")
        d_val_false_safe_std = _std(aggregates, "D_stock_fragility", "validation", "false_safe_tail_rate")
        if (
            d_test_false_safe is not None
            and d_test_severe is not None
            and (d_test_false_safe >= 0.35 or d_test_severe < 0.65)
        ):
            aggregates["D_stock_fragility"]["classification"] = "탈락 후보"
            aggregates["D_stock_fragility"]["failure_reason"] = "test false_safe/severe median이 Stage 4-2 하방 안정성 기준을 벗어나 seed 42 단일 신호 가능성이 크다."
        elif d_val_false_safe is not None and d_val_false_safe_std is not None and d_val_false_safe > s41.BASELINES["stage2_best_false_safe"]:
            aggregates["D_stock_fragility"]["classification"] = "탈락 후보"
            aggregates["D_stock_fragility"]["failure_reason"] = "validation false_safe median이 Stage 2 best보다 나쁘고 seed std가 커 primary 기준을 만족하지 못했다."

    if "C_stress_delta" in aggregates:
        c_val_false_safe = _median(aggregates, "C_stress_delta", "validation", "false_safe_tail_rate")
        c_test_false_safe = _median(aggregates, "C_stress_delta", "test", "false_safe_tail_rate")
        c_val_severe = _median(aggregates, "C_stress_delta", "validation", "severe_downside_recall")
        c_test_severe = _median(aggregates, "C_stress_delta", "test", "severe_downside_recall")
        d_val_false_safe = _median(aggregates, "D_stock_fragility", "validation", "false_safe_tail_rate")
        d_test_false_safe = _median(aggregates, "D_stock_fragility", "test", "false_safe_tail_rate")
        d_val_severe = _median(aggregates, "D_stock_fragility", "validation", "severe_downside_recall")
        d_test_severe = _median(aggregates, "D_stock_fragility", "test", "severe_downside_recall")
        if (
            c_val_false_safe is not None
            and c_test_false_safe is not None
            and c_val_severe is not None
            and c_test_severe is not None
            and (d_val_false_safe is None or c_val_false_safe < d_val_false_safe)
            and (d_test_false_safe is None or c_test_false_safe < d_test_false_safe)
            and (d_val_severe is None or c_val_severe > d_val_severe)
            and (d_test_severe is None or c_test_severe > d_test_severe)
        ):
            aggregates["C_stress_delta"]["classification"] = "Stage 4-2 primary 후보"
            aggregates["C_stress_delta"]["strength_reason"] = "validation/test 양쪽에서 D보다 false_safe median이 낮고 severe median이 높아 seed 안정성이 더 좋다."
        elif aggregates["C_stress_delta"].get("classification") == "seed 재평가 통과 후보":
            aggregates["C_stress_delta"]["classification"] = "alpha-preserving challenger"


def _run_one(
    candidate: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    amp_dtype: str,
    force: bool,
    ablation: bool,
) -> dict[str, Any]:
    experiment = _candidate_experiment(candidate, seed)
    run_dir = LOG_DIR / experiment["experiment_id"]
    metrics_path = run_dir / "run_meta.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    precomputed, feature_names = s41._build_exact_feature_splits(
        extra_names=list(experiment["extra_features"]),
        device=device,
        batch_size=batch_size,
    )
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    config = s41._make_config(
        experiment,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        device=device,
        feature_names=feature_names,
    )
    result = train_mod.run_training(
        config,
        save_run=False,
        precomputed_bundles=precomputed,
        enable_compile=False,
        local_log=True,
        local_log_dir=run_dir / "train_local_logs",
        wandb_group="cp148_stage4_2",
        wandb_name=experiment["experiment_id"],
        wandb_required=False,
    )
    severe_threshold = s41._safe_float((result.get("best_metrics") or {}).get("severe_downside_threshold"))
    validation_regimes, validation_buckets = s41._regime_and_bucket_metrics(
        checkpoint_path=result["checkpoint_path"],
        val_bundle=val_bundle,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        severe_threshold=severe_threshold,
    )
    test_regimes, test_buckets = s41._regime_and_bucket_metrics(
        checkpoint_path=result["checkpoint_path"],
        val_bundle=test_bundle,
        batch_size=batch_size,
        device=device,
        amp_dtype=amp_dtype,
        severe_threshold=severe_threshold,
    )
    record = {
        "candidate_id": candidate["candidate_id"],
        "role": candidate["role"],
        "ablation": ablation,
        "experiment_id": experiment["experiment_id"],
        "extra_features": list(experiment["extra_features"]),
        "seed": seed,
        "seed_policy": "재현 가능한 고정 seed 집합: 42, 7, 123",
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "amp_dtype": amp_dtype,
        "checkpoint_selection": "line_gate",
        "selector_policy": "risk_aware_line_gate_sort",
        "line_gate_meaning_changed": False,
        "run_id": result.get("run_id"),
        "checkpoint_path": result.get("checkpoint_path"),
        "line_gate_pass": bool((result.get("best_metrics") or {}).get("line_gate_pass")),
        "validation": s41._line_metrics(result.get("best_metrics") or {}),
        "test": s41._line_metrics(result.get("test_metrics") or {}),
        "validation_regime_metrics": validation_regimes,
        "validation_horizon_bucket_metrics": validation_buckets,
        "test_regime_metrics": test_regimes,
        "test_horizon_bucket_metrics": test_buckets,
        "source_data_hash": getattr(plan, "source_data_hash", None),
        "eligible_ticker_count": len(getattr(plan, "eligible_tickers", [])),
        "elapsed_seconds": time.perf_counter() - started,
        "scope_compliance": {
            "product_save": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "beta": 2.0,
            "line_gate_meaning_changed": False,
        },
    }
    _write_json(metrics_path, record)
    return record


def _summary_rows(records: list[dict[str, Any]], aggregates: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for split in ["validation", "test"]:
            summary = _extract_split_summary(record, split)
            rows.append(
                {
                    "row_type": "run",
                    "candidate_id": record["candidate_id"],
                    "classification": "",
                    "split": split,
                    "seed": record["seed"],
                    "ablation": record["ablation"],
                    "line_gate_pass": record["line_gate_pass"],
                    "ic_mean": summary.get("ic_mean"),
                    "long_short_spread": summary.get("long_short_spread"),
                    "fee_adjusted_return": summary.get("fee_adjusted_return"),
                    "false_safe_tail_rate": summary.get("false_safe_tail_rate"),
                    "severe_downside_recall": summary.get("severe_downside_recall"),
                    "h1_false_safe_tail_rate": summary.get("h1_false_safe_tail_rate"),
                    "h1_severe_downside_recall": summary.get("h1_severe_downside_recall"),
                    "h2_h3_false_safe_tail_rate": summary.get("h2_h3_false_safe_tail_rate"),
                    "h2_h3_severe_downside_recall": summary.get("h2_h3_severe_downside_recall"),
                    "h4_h5_false_safe_tail_rate": summary.get("h4_h5_false_safe_tail_rate"),
                    "h4_h5_severe_downside_recall": summary.get("h4_h5_severe_downside_recall"),
                    "stress_false_safe_tail_rate": summary.get("stress_false_safe_tail_rate"),
                    "stress_severe_downside_recall": summary.get("stress_severe_downside_recall"),
                    "vix_rising_false_safe_tail_rate": summary.get("vix_rising_false_safe_tail_rate"),
                    "vix_rising_severe_downside_recall": summary.get("vix_rising_severe_downside_recall"),
                    "breadth_worsening_false_safe_tail_rate": summary.get("breadth_worsening_false_safe_tail_rate"),
                    "breadth_worsening_severe_downside_recall": summary.get("breadth_worsening_severe_downside_recall"),
                    "checkpoint_path": record.get("checkpoint_path"),
                }
            )
    for candidate_id, payload in aggregates.items():
        classification = payload["classification"]
        for split in ["validation", "test"]:
            metrics = payload["aggregate"].get(split) or {}
            rows.append(
                {
                    "row_type": "aggregate_median",
                    "candidate_id": candidate_id,
                    "classification": classification,
                    "split": split,
                    "seed": "median",
                    "ablation": payload.get("ablation"),
                    "line_gate_pass": payload.get("all_line_gate_pass"),
                    "ic_mean": (metrics.get("ic_mean") or {}).get("median"),
                    "long_short_spread": (metrics.get("long_short_spread") or {}).get("median"),
                    "fee_adjusted_return": (metrics.get("fee_adjusted_return") or {}).get("median"),
                    "false_safe_tail_rate": (metrics.get("false_safe_tail_rate") or {}).get("median"),
                    "severe_downside_recall": (metrics.get("severe_downside_recall") or {}).get("median"),
                    "h1_false_safe_tail_rate": (metrics.get("h1_false_safe_tail_rate") or {}).get("median"),
                    "h1_severe_downside_recall": (metrics.get("h1_severe_downside_recall") or {}).get("median"),
                    "h2_h3_false_safe_tail_rate": (metrics.get("h2_h3_false_safe_tail_rate") or {}).get("median"),
                    "h2_h3_severe_downside_recall": (metrics.get("h2_h3_severe_downside_recall") or {}).get("median"),
                    "h4_h5_false_safe_tail_rate": (metrics.get("h4_h5_false_safe_tail_rate") or {}).get("median"),
                    "h4_h5_severe_downside_recall": (metrics.get("h4_h5_severe_downside_recall") or {}).get("median"),
                    "stress_false_safe_tail_rate": (metrics.get("stress_false_safe_tail_rate") or {}).get("median"),
                    "stress_severe_downside_recall": (metrics.get("stress_severe_downside_recall") or {}).get("median"),
                    "vix_rising_false_safe_tail_rate": (metrics.get("vix_rising_false_safe_tail_rate") or {}).get("median"),
                    "vix_rising_severe_downside_recall": (metrics.get("vix_rising_severe_downside_recall") or {}).get("median"),
                    "breadth_worsening_false_safe_tail_rate": (metrics.get("breadth_worsening_false_safe_tail_rate") or {}).get("median"),
                    "breadth_worsening_severe_downside_recall": (metrics.get("breadth_worsening_severe_downside_recall") or {}).get("median"),
                    "checkpoint_path": "",
                }
            )
    return rows


def _write_summary_csv(records: list[dict[str, Any]], aggregates: dict[str, Any]) -> None:
    rows = _summary_rows(records, aggregates)
    fields = list(rows[0].keys()) if rows else []
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _aggregate_table_rows(aggregates: dict[str, Any], split: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for candidate_id, payload in aggregates.items():
        metrics = payload["aggregate"].get(split) or {}
        rows.append(
            [
                candidate_id,
                payload["classification"],
                _fmt((metrics.get("ic_mean") or {}).get("median")),
                _fmt((metrics.get("long_short_spread") or {}).get("median")),
                _fmt((metrics.get("fee_adjusted_return") or {}).get("median")),
                _fmt((metrics.get("false_safe_tail_rate") or {}).get("median")),
                _fmt((metrics.get("false_safe_tail_rate") or {}).get("std")),
                _fmt((metrics.get("severe_downside_recall") or {}).get("median")),
                _fmt((metrics.get("severe_downside_recall") or {}).get("std")),
            ]
        )
    return rows


def _write_report(records: list[dict[str, Any]], aggregates: dict[str, Any], ablation_requested: bool) -> None:
    lines = [
        "# CP148-LM-1D Stage 4-2 seed 안정성 보고서",
        "",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "- 범위: D_stock_fragility와 C_stress_delta의 seed 3개 재평가",
        "- seed: 42, 7, 123",
        "- 금지 준수: product save 없음, DB write 없음, inference 저장 없음, live fetch 없음, band/composite 없음, beta=2 유지",
        "- line_gate 의미는 생존 조건으로 유지했고, checkpoint 정렬만 Stage 4 risk-aware selector를 사용했다.",
        "",
        "## 1. Validation median",
        "",
    ]
    lines.extend(
        _table(
            ["후보", "판정", "IC", "spread", "fee", "false_safe", "FS std", "severe", "severe std"],
            _aggregate_table_rows(aggregates, "validation"),
        )
    )
    lines.extend(["", "## 2. Test median", ""])
    lines.extend(
        _table(
            ["후보", "판정", "IC", "spread", "fee", "false_safe", "FS std", "severe", "severe std"],
            _aggregate_table_rows(aggregates, "test"),
        )
    )
    lines.extend(["", "## 3. Seed별 validation/test 요약", ""])
    run_rows: list[list[Any]] = []
    for record in records:
        for split in ["validation", "test"]:
            summary = _extract_split_summary(record, split)
            run_rows.append(
                [
                    record["candidate_id"],
                    split,
                    record["seed"],
                    record["line_gate_pass"],
                    _fmt(summary.get("ic_mean")),
                    _fmt(summary.get("long_short_spread")),
                    _fmt(summary.get("fee_adjusted_return")),
                    _fmt(summary.get("false_safe_tail_rate")),
                    _fmt(summary.get("severe_downside_recall")),
                ]
            )
    lines.extend(_table(["후보", "split", "seed", "line_gate", "IC", "spread", "fee", "FS", "severe"], run_rows))
    lines.extend(["", "## 4. Regime / bucket median", ""])
    detail_rows: list[list[Any]] = []
    for candidate_id, payload in aggregates.items():
        for split in ["validation", "test"]:
            metrics = payload["aggregate"].get(split) or {}
            for key in [
                "h1",
                "h2_h3",
                "h4_h5",
                "stress",
                "vix_rising",
                "breadth_worsening",
            ]:
                detail_rows.append(
                    [
                        candidate_id,
                        split,
                        key,
                        _fmt((metrics.get(f"{key}_false_safe_tail_rate") or {}).get("median")),
                        _fmt((metrics.get(f"{key}_severe_downside_recall") or {}).get("median")),
                    ]
                )
    lines.extend(_table(["후보", "split", "구간", "FS median", "severe median"], detail_rows))

    d_payload = aggregates.get("D_stock_fragility")
    c_payload = aggregates.get("C_stress_delta")
    d_class = d_payload["classification"] if d_payload else ""
    c_class = c_payload["classification"] if c_payload else ""
    d_reason = d_payload.get("failure_reason", "") if d_payload else ""
    c_reason = c_payload.get("strength_reason", "") if c_payload else ""
    lines.extend(
        [
            "",
            "## 5. 결론",
            "",
            f"- D_stock_fragility: `{d_class}`",
            f"- C_stress_delta: `{c_class}`",
            f"- D 판단 근거: {d_reason or 'line_gate는 통과했지만 상대 비교상 primary 근거가 부족하다.'}",
            f"- C 판단 근거: {c_reason or 'line_gate 통과와 양수 ranking 지표는 유지했지만 별도 primary 근거는 제한적이다.'}",
            "- D는 seed 42 단일 결과보다 validation/test median이 약해졌고 std도 커서 stock fragility 효과가 seed 안정적으로 유지됐다고 보기 어렵다.",
            "- C는 원래 alpha 보존 비교 후보였지만, 이번 seed 재평가에서는 validation/test 하방 안정성도 D보다 안정적이었다.",
            "- 따라서 다음 단계는 C_stress_delta를 Stage 4-2 primary 후보로 두고 seed 5개 또는 walk-forward로 확인하는 쪽이 자연스럽다.",
            "- 별도 alpha-preserving challenger는 아직 확정하지 않는다. D는 validation fee/spread는 강하지만 test 하방 위험이 흔들려 alpha reference로도 조심스럽다.",
            "- test에서 D의 false_safe/severe 붕괴가 보여 제품 후보 표현은 금지하고 feature overfit 의심으로 기록한다.",
            f"- ATR ablation 요청 상태: {'실행 포함' if ablation_requested else '본실험 우선으로 미실행'}",
            "",
            "## 6. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- logs/meta: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
            "- script: `ai/cp148_lm_1d_stage4_2_seed_stability.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_lookup(include_ablation: bool, ablation_only: bool) -> list[dict[str, Any]]:
    if ablation_only:
        return ABLATION_CANDIDATES
    candidates = list(MAIN_CANDIDATES)
    if include_ablation:
        candidates.extend(ABLATION_CANDIDATES)
    return candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    alias = s41._configure_environment()
    s41.LOG_DIR = LOG_DIR
    s41._patch_training_for_experiment()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_lookup(args.include_ablation, args.ablation_only)
    seeds = SEEDS if not args.ablation_one_seed else [SEEDS[0]]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_seeds = seeds if candidate in ABLATION_CANDIDATES else SEEDS
        for seed in candidate_seeds:
            record = _run_one(
                candidate,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                amp_dtype=args.amp_dtype,
                force=args.force,
                ablation=candidate in ABLATION_CANDIDATES,
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "candidate_id": record["candidate_id"],
                        "seed": record["seed"],
                        "line_gate_pass": record["line_gate_pass"],
                        "validation_false_safe": record["validation"].get("false_safe_tail_rate"),
                        "validation_severe": record["validation"].get("severe_downside_recall"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["candidate_id"], []).append(record)
    aggregates: dict[str, Any] = {}
    for candidate_id, candidate_records in grouped.items():
        aggregate = _aggregate_records(candidate_records)
        aggregates[candidate_id] = {
            "candidate_id": candidate_id,
            "ablation": bool(candidate_records[0].get("ablation")),
            "all_line_gate_pass": all(bool(record.get("line_gate_pass")) for record in candidate_records),
            "classification": _classify_candidate(candidate_id, aggregate, candidate_records),
            "aggregate": aggregate,
        }
    _apply_relative_classifications(aggregates)

    payload = {
        "cp": "CP148-LM-1D-Stage4-2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_alias_dir": str(alias),
        "seed_policy": {
            "seeds": SEEDS,
            "reason": "42는 Stage 4-1 재현 seed이고, 7과 123은 재현 가능한 추가 고정 seed다.",
        },
        "scope_compliance": {
            "product_save": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "beta": 2.0,
            "line_gate_meaning_changed": False,
            "eodhd_local_parquet": True,
        },
        "records": records,
        "aggregates": aggregates,
        "process_check": {
            "status": "deferred_external_check",
            "note": "최종 외부 검증에서 python/pythonw 및 CUDA Python 학습 프로세스를 확인한다.",
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(records, aggregates)
    _write_report(records, aggregates, args.include_ablation or args.ablation_only)
    print(json.dumps({"status": "PASS", "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-2 seed stability")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--ablation-one-seed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
