from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
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


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4_seed_stability_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4_seed_stability_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4_seed_stability_summary.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "cp148_lm_1d_stage4_4_seed_stability"

SEEDS = [42, 7, 123]
C_FEATURES = ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"]

CANDIDATES = [
    {
        "candidate_id": "trial006_c_balanced",
        "display_name": "trial006 C-balanced primary",
        "role": "Stage 4-3 test 하방 안정성이 가장 좋았던 primary 후보",
        "patch_len": 32,
        "patch_stride": 16,
        "lr": 0.0007362816234925851,
        "weight_decay": 0.00008143270337695065,
        "dropout": 0.10,
        "lambda_direction": 0.20,
        "stage4_3_reference": {
            "validation": {
                "ic_mean": 0.0544,
                "long_short_spread": 0.00952,
                "fee_adjusted_return": 63.08,
                "false_safe_tail_rate": 0.2373,
                "severe_downside_recall": 0.7499,
            },
            "test": {
                "ic_mean": 0.0422,
                "long_short_spread": 0.00530,
                "fee_adjusted_return": 3.77,
                "false_safe_tail_rate": 0.2817,
                "severe_downside_recall": 0.7180,
            },
        },
    },
    {
        "candidate_id": "trial024_c_risk",
        "display_name": "trial024 C-risk challenger",
        "role": "Stage 4-3 validation risk 개선폭이 가장 컸던 challenger",
        "patch_len": 16,
        "patch_stride": 8,
        "lr": 0.0013385598971335333,
        "weight_decay": 0.000190190440463508,
        "dropout": 0.18,
        "lambda_direction": 0.20,
        "stage4_3_reference": {
            "validation": {
                "ic_mean": 0.0481,
                "long_short_spread": 0.00854,
                "fee_adjusted_return": 41.57,
                "false_safe_tail_rate": 0.2249,
                "severe_downside_recall": 0.7635,
            },
            "test": {
                "ic_mean": 0.0425,
                "long_short_spread": 0.00581,
                "fee_adjusted_return": 5.14,
                "false_safe_tail_rate": 0.2950,
                "severe_downside_recall": 0.7065,
            },
        },
    },
]

REFERENCES = {
    "stage4_2_c_stress_delta_seed_median": {
        "validation": {
            "ic_mean": 0.04885,
            "long_short_spread": 0.00770,
            "fee_adjusted_return": 19.68,
            "false_safe_tail_rate": 0.28177,
            "severe_downside_recall": 0.70529,
        },
        "test": {
            "ic_mean": 0.04251,
            "long_short_spread": 0.00603,
            "fee_adjusted_return": 4.91,
            "false_safe_tail_rate": 0.31171,
            "severe_downside_recall": 0.68708,
        },
    },
    "stage2_stage4_0_no_fund_p32_s16_primary_base": {
        "validation": {
            "ic_mean": 0.05697,
            "long_short_spread": 0.00839,
            "fee_adjusted_return": 22.43,
            "false_safe_tail_rate": 0.30866056266521946,
            "severe_downside_recall": 0.6850190785352241,
        }
    },
}

METRIC_INFO = {
    "ic_mean": {
        "label": "IC",
        "direction": "높을수록 좋음",
        "meaning": "날짜별 종목 순위 상관이다.",
    },
    "long_short_spread": {
        "label": "spread",
        "direction": "높을수록 좋음",
        "meaning": "예측 상위 10% 실제 수익률에서 예측 하위 10% 실제 수익률을 뺀 값이다.",
    },
    "fee_adjusted_return": {
        "label": "fee_adjusted_return",
        "direction": "높을수록 좋지만 보조 지표",
        "meaning": "10bp 비용 가정 후 ranking signal이 비용을 견디는지 보는 보조 지표다.",
    },
    "false_safe_tail_rate": {
        "label": "false_safe_tail_rate",
        "direction": "낮을수록 좋음",
        "meaning": "실제 하위 꼬리인데 line이 0 이상으로 안전하다고 오판한 비율이다.",
    },
    "severe_downside_recall": {
        "label": "severe_downside_recall",
        "direction": "높을수록 좋음",
        "meaning": "심한 하락을 음수 위험으로 잡아낸 비율이다.",
    },
    "conservative_bias": {
        "label": "conservative_bias",
        "direction": "음수면 보수적이나 너무 음수면 수익 희생 확인",
        "meaning": "예측이 실제보다 평균적으로 낮은지 보는 값이다.",
    },
    "upside_sacrifice": {
        "label": "upside_sacrifice",
        "direction": "낮을수록 좋음",
        "meaning": "실제 상승 종목을 과하게 낮춰 잡는지 보는 값이다.",
    },
    "direction_accuracy": {
        "label": "direction_accuracy",
        "direction": "높을수록 좋음",
        "meaning": "부호 방향이 맞은 비율이다.",
    },
}

METRIC_KEYS = list(METRIC_INFO.keys())
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


def _append_progress(run_dir: Path, message: str, **payload: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), "message": message, **payload}
    with (run_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_experiment(candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    return {
        "experiment_id": f"stage4_4_{candidate_id}_seed{seed}",
        "label": candidate_id,
        "question": candidate["role"],
        "extra_features": list(C_FEATURES),
    }


def _make_config(
    experiment: dict[str, Any],
    candidate: dict[str, Any],
    *,
    epochs: int,
    seed: int,
    batch_size: int,
    device: str,
    feature_names: list[str],
) -> train_mod.TrainConfig:
    config = s41._make_config(
        experiment,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        device=device,
        feature_names=feature_names,
    )
    config.patch_len = int(candidate["patch_len"])
    config.patch_stride = int(candidate["patch_stride"])
    config.lr = float(candidate["lr"])
    config.weight_decay = float(candidate["weight_decay"])
    config.dropout = float(candidate["dropout"])
    config.lambda_direction = float(candidate["lambda_direction"])
    config.use_direction_head = False
    config.use_wandb = False
    config.feature_set = "C_stress_delta"
    config.checkpoint_selection = "line_gate"
    return config


def _extract_split_summary(record: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = record.get(split) or {}
    regime_container = record.get(f"{split}_regime_metrics") or {}
    bucket_container = record.get(f"{split}_horizon_bucket_metrics") or {}
    summary = {key: metrics.get(key) for key in METRIC_KEYS}
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
        return {"count": 0, "median": None, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ["validation", "test"]:
        summaries = [_extract_split_summary(record, split) for record in records]
        keys: set[str] = set()
        for summary in summaries:
            keys.update(summary.keys())
        split_output: dict[str, Any] = {}
        for key in sorted(keys):
            values = [_safe_float(summary.get(key)) for summary in summaries]
            split_output[key] = _aggregate_values([value for value in values if value is not None])
        output[split] = split_output
    return output


def _agg(aggregate: dict[str, Any], split: str, metric: str, stat: str = "median") -> float | None:
    return ((aggregate.get(split) or {}).get(metric) or {}).get(stat)


def _classify_candidate(aggregate: dict[str, Any], records: list[dict[str, Any]]) -> tuple[str, str]:
    all_line_gate = all(bool(record.get("line_gate_pass")) for record in records)
    if not all_line_gate:
        return "탈락 후보", "seed 중 line_gate 실패가 있어 기본 생존 조건을 만족하지 못했다."

    val_ic = _agg(aggregate, "validation", "ic_mean")
    test_ic = _agg(aggregate, "test", "ic_mean")
    val_spread = _agg(aggregate, "validation", "long_short_spread")
    test_spread = _agg(aggregate, "test", "long_short_spread")
    val_fee = _agg(aggregate, "validation", "fee_adjusted_return")
    test_fee = _agg(aggregate, "test", "fee_adjusted_return")
    val_fs = _agg(aggregate, "validation", "false_safe_tail_rate")
    test_fs = _agg(aggregate, "test", "false_safe_tail_rate")
    val_severe = _agg(aggregate, "validation", "severe_downside_recall")
    test_severe = _agg(aggregate, "test", "severe_downside_recall")
    test_fs_std = _agg(aggregate, "test", "false_safe_tail_rate", "std")

    reference = REFERENCES["stage4_2_c_stress_delta_seed_median"]
    stable = (
        val_ic is not None
        and test_ic is not None
        and val_spread is not None
        and test_spread is not None
        and val_fee is not None
        and test_fee is not None
        and val_fs is not None
        and test_fs is not None
        and val_severe is not None
        and test_severe is not None
        and val_ic > 0
        and test_ic > 0
        and val_spread > 0
        and test_spread > 0
        and test_spread >= 0.005
        and val_fee > 0
        and test_fee > 0
        and val_fs < reference["validation"]["false_safe_tail_rate"]
        and test_fs < reference["test"]["false_safe_tail_rate"]
        and val_severe > reference["validation"]["severe_downside_recall"]
        and test_severe > reference["test"]["severe_downside_recall"]
        and (test_fs_std is None or test_fs_std <= 0.035)
    )
    if stable:
        return "Stage 4-5 walk-forward 후보", "validation/test 양쪽에서 기준 C median보다 false_safe가 낮고 severe가 높으며 IC/spread/fee가 양수로 유지됐다."

    if test_fs is not None and test_fs >= reference["test"]["false_safe_tail_rate"]:
        return "보류 후보", "test false_safe median이 기준 C median보다 낮아지지 않아 하방 안정성 재현이 부족하다."
    if test_severe is not None and test_severe <= reference["test"]["severe_downside_recall"]:
        return "보류 후보", "test severe_downside_recall median이 기준 C median보다 높지 않아 위험 포착 재현이 부족하다."
    if test_spread is not None and test_spread <= 0:
        return "탈락 후보", "test spread median이 양수를 유지하지 못했다."
    return "보류 후보", "핵심 지표 일부는 유지됐지만 Stage 4-5로 올릴 만큼의 validation/test 동시 안정성은 부족하다."


def _run_one(
    candidate: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    amp_dtype: str,
    force: bool,
) -> dict[str, Any]:
    experiment = _candidate_experiment(candidate, seed)
    run_dir = LOG_DIR / experiment["experiment_id"]
    metrics_path = run_dir / "run_meta.json"
    process_path = run_dir / "train_process.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    process_meta = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": candidate["candidate_id"],
        "seed": seed,
        "device": device,
        "save_run": False,
        "db_write": False,
        "inference_save": False,
    }
    _write_json(process_path, process_meta)
    _append_progress(run_dir, "run_start", candidate_id=candidate["candidate_id"], seed=seed)

    try:
        precomputed, feature_names = s41._build_exact_feature_splits(
            extra_names=list(experiment["extra_features"]),
            device=device,
            batch_size=batch_size,
        )
        train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
        config = _make_config(
            experiment,
            candidate,
            epochs=epochs,
            seed=seed,
            batch_size=batch_size,
            device=device,
            feature_names=feature_names,
        )
        _append_progress(
            run_dir,
            "training_start",
            patch_len=config.patch_len,
            patch_stride=config.patch_stride,
            lr=config.lr,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            lambda_direction_recorded=config.lambda_direction,
            lambda_direction_interpretation="PatchTST direction head 비활성이므로 성능 원인으로 해석하지 않음",
        )
        result = train_mod.run_training(
            config,
            save_run=False,
            precomputed_bundles=precomputed,
            enable_compile=False,
            local_log=True,
            local_log_dir=run_dir / "train_local_logs",
            wandb_group="cp148_stage4_4",
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
            "display_name": candidate["display_name"],
            "role": candidate["role"],
            "experiment_id": experiment["experiment_id"],
            "extra_features": list(experiment["extra_features"]),
            "seed": seed,
            "seed_policy": "재현 가능한 고정 seed 42, 7, 123",
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "amp_dtype": amp_dtype,
            "checkpoint_selection": "line_gate",
            "line_gate_meaning": "기본 생존 조건이며 단독 제품 통과 기준이 아님",
            "line_gate_meaning_changed": False,
            "lambda_direction_recorded": float(candidate["lambda_direction"]),
            "lambda_direction_interpretation": "PatchTST direction head 비활성이므로 성능 원인으로 해석하지 않음",
            "params": {
                "patch_len": int(candidate["patch_len"]),
                "patch_stride": int(candidate["patch_stride"]),
                "lr": float(candidate["lr"]),
                "weight_decay": float(candidate["weight_decay"]),
                "dropout": float(candidate["dropout"]),
                "lambda_direction": float(candidate["lambda_direction"]),
                "use_direction_head": False,
            },
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
                "model_structure_changed": False,
                "direction_head_added": False,
                "core_feature_contract_changed": False,
                "beta": 2.0,
                "line_gate_meaning_changed": False,
            },
        }
        _write_json(metrics_path, record)
        process_meta.update(
            {
                "status": "completed",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": record["elapsed_seconds"],
                "run_meta_path": str(metrics_path),
                "checkpoint_path": record.get("checkpoint_path"),
            }
        )
        _write_json(process_path, process_meta)
        _append_progress(
            run_dir,
            "run_completed",
            validation_false_safe=record["validation"].get("false_safe_tail_rate"),
            validation_severe=record["validation"].get("severe_downside_recall"),
            test_false_safe=record["test"].get("false_safe_tail_rate"),
            test_severe=record["test"].get("severe_downside_recall"),
        )
        return record
    except Exception as exc:
        process_meta.update({"status": "failed", "ended_at": datetime.now().isoformat(timespec="seconds"), "error": repr(exc)})
        _write_json(process_path, process_meta)
        _append_progress(run_dir, "run_failed", error=repr(exc))
        raise


def _aggregate_by_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["candidate_id"], []).append(record)
    aggregates: dict[str, Any] = {}
    for candidate_id, candidate_records in grouped.items():
        aggregate = _aggregate_records(candidate_records)
        classification, reason = _classify_candidate(aggregate, candidate_records)
        candidate = next(item for item in CANDIDATES if item["candidate_id"] == candidate_id)
        aggregates[candidate_id] = {
            "candidate_id": candidate_id,
            "display_name": candidate["display_name"],
            "classification": classification,
            "reason": reason,
            "all_line_gate_pass": all(bool(record.get("line_gate_pass")) for record in candidate_records),
            "params": {
                "patch_len": candidate["patch_len"],
                "patch_stride": candidate["patch_stride"],
                "lr": candidate["lr"],
                "weight_decay": candidate["weight_decay"],
                "dropout": candidate["dropout"],
                "lambda_direction_recorded": candidate["lambda_direction"],
                "lambda_direction_interpretation": "PatchTST direction head 비활성이므로 성능 원인으로 해석하지 않음",
            },
            "stage4_3_reference": candidate["stage4_3_reference"],
            "aggregate": aggregate,
        }
    return aggregates


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
                    "stat": "raw",
                    "seed": record["seed"],
                    "line_gate_pass": record["line_gate_pass"],
                    "ic_mean": summary.get("ic_mean"),
                    "long_short_spread": summary.get("long_short_spread"),
                    "fee_adjusted_return": summary.get("fee_adjusted_return"),
                    "false_safe_tail_rate": summary.get("false_safe_tail_rate"),
                    "severe_downside_recall": summary.get("severe_downside_recall"),
                    "conservative_bias": summary.get("conservative_bias"),
                    "upside_sacrifice": summary.get("upside_sacrifice"),
                    "direction_accuracy": summary.get("direction_accuracy"),
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
        for split in ["validation", "test"]:
            metrics = payload["aggregate"].get(split) or {}
            for stat in ["median", "mean", "std", "min", "max"]:
                rows.append(
                    {
                        "row_type": "aggregate",
                        "candidate_id": candidate_id,
                        "classification": payload["classification"],
                        "split": split,
                        "stat": stat,
                        "seed": "42,7,123",
                        "line_gate_pass": payload.get("all_line_gate_pass"),
                        "ic_mean": (metrics.get("ic_mean") or {}).get(stat),
                        "long_short_spread": (metrics.get("long_short_spread") or {}).get(stat),
                        "fee_adjusted_return": (metrics.get("fee_adjusted_return") or {}).get(stat),
                        "false_safe_tail_rate": (metrics.get("false_safe_tail_rate") or {}).get(stat),
                        "severe_downside_recall": (metrics.get("severe_downside_recall") or {}).get(stat),
                        "conservative_bias": (metrics.get("conservative_bias") or {}).get(stat),
                        "upside_sacrifice": (metrics.get("upside_sacrifice") or {}).get(stat),
                        "direction_accuracy": (metrics.get("direction_accuracy") or {}).get(stat),
                        "h1_false_safe_tail_rate": (metrics.get("h1_false_safe_tail_rate") or {}).get(stat),
                        "h1_severe_downside_recall": (metrics.get("h1_severe_downside_recall") or {}).get(stat),
                        "h2_h3_false_safe_tail_rate": (metrics.get("h2_h3_false_safe_tail_rate") or {}).get(stat),
                        "h2_h3_severe_downside_recall": (metrics.get("h2_h3_severe_downside_recall") or {}).get(stat),
                        "h4_h5_false_safe_tail_rate": (metrics.get("h4_h5_false_safe_tail_rate") or {}).get(stat),
                        "h4_h5_severe_downside_recall": (metrics.get("h4_h5_severe_downside_recall") or {}).get(stat),
                        "stress_false_safe_tail_rate": (metrics.get("stress_false_safe_tail_rate") or {}).get(stat),
                        "stress_severe_downside_recall": (metrics.get("stress_severe_downside_recall") or {}).get(stat),
                        "vix_rising_false_safe_tail_rate": (metrics.get("vix_rising_false_safe_tail_rate") or {}).get(stat),
                        "vix_rising_severe_downside_recall": (metrics.get("vix_rising_severe_downside_recall") or {}).get(stat),
                        "breadth_worsening_false_safe_tail_rate": (metrics.get("breadth_worsening_false_safe_tail_rate") or {}).get(stat),
                        "breadth_worsening_severe_downside_recall": (metrics.get("breadth_worsening_severe_downside_recall") or {}).get(stat),
                        "checkpoint_path": "",
                    }
                )
    return rows


def _write_summary_csv(records: list[dict[str, Any]], aggregates: dict[str, Any]) -> None:
    rows = _summary_rows(records, aggregates)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
                _fmt((metrics.get("upside_sacrifice") or {}).get("median")),
            ]
        )
    return rows


def _run_table_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for record in records:
        for split in ["validation", "test"]:
            summary = _extract_split_summary(record, split)
            rows.append(
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
                    _fmt(summary.get("conservative_bias")),
                    _fmt(summary.get("upside_sacrifice")),
                ]
            )
    return rows


def _detail_table_rows(aggregates: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for candidate_id, payload in aggregates.items():
        for split in ["validation", "test"]:
            metrics = payload["aggregate"].get(split) or {}
            for key in ["h1", "h2_h3", "h4_h5", "stress", "vix_rising", "breadth_worsening"]:
                rows.append(
                    [
                        candidate_id,
                        split,
                        key,
                        _fmt((metrics.get(f"{key}_false_safe_tail_rate") or {}).get("median")),
                        _fmt((metrics.get(f"{key}_severe_downside_recall") or {}).get("median")),
                        _fmt((metrics.get(f"{key}_long_short_spread") or {}).get("median")),
                        _fmt((metrics.get(f"{key}_fee_adjusted_return") or {}).get("median")),
                    ]
                )
    return rows


def _reference_rows() -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, payload in REFERENCES.items():
        for split, metrics in payload.items():
            rows.append(
                [
                    name,
                    split,
                    _fmt(metrics.get("ic_mean")),
                    _fmt(metrics.get("long_short_spread")),
                    _fmt(metrics.get("fee_adjusted_return")),
                    _fmt(metrics.get("false_safe_tail_rate")),
                    _fmt(metrics.get("severe_downside_recall")),
                ]
            )
    for candidate in CANDIDATES:
        for split, metrics in candidate["stage4_3_reference"].items():
            rows.append(
                [
                    f"stage4_3_{candidate['candidate_id']}_single_seed",
                    split,
                    _fmt(metrics.get("ic_mean")),
                    _fmt(metrics.get("long_short_spread")),
                    _fmt(metrics.get("fee_adjusted_return")),
                    _fmt(metrics.get("false_safe_tail_rate")),
                    _fmt(metrics.get("severe_downside_recall")),
                ]
            )
    return rows


def _walk_forward_candidates(aggregates: dict[str, Any]) -> list[str]:
    return [
        candidate_id
        for candidate_id, payload in aggregates.items()
        if payload.get("classification") == "Stage 4-5 walk-forward 후보"
    ]


def _write_report(records: list[dict[str, Any]], aggregates: dict[str, Any]) -> None:
    walk_forward = _walk_forward_candidates(aggregates)
    conclusion = (
        f"Stage 4-5 walk-forward 후보 있음: {', '.join(walk_forward)}"
        if walk_forward
        else "Stage 4-5 walk-forward 후보 없음"
    )
    lines = [
        "# CP148-LM-1D Stage 4-4 seed stability 보고서",
        "",
        f"- 한 줄 결론: {conclusion}",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "- 목적: Stage 4-3 Optuna에서 뽑힌 C feature 기반 후보 2개가 단일 seed 우연인지 확인한다.",
        "- 범위: EODHD 500 local parquet, 1D h5, PatchTST line_model, C feature pack, seed 42/7/123",
        "- 금지 준수: product save-run 없음, DB write 없음, inference 저장 없음, live fetch 없음, band/composite 실험 없음",
        "- lambda_direction 정정: artifact에는 기록하지만 PatchTST direction head가 비활성이므로 성능 원인으로 해석하지 않는다.",
        "",
        "## 1. 후보별 설정",
        "",
    ]
    lines.extend(
        _table(
            ["후보", "역할", "patch", "lr", "weight_decay", "dropout", "lambda_direction 해석"],
            [
                [
                    item["candidate_id"],
                    item["role"],
                    f"{item['patch_len']}/{item['patch_stride']}",
                    item["lr"],
                    item["weight_decay"],
                    item["dropout"],
                    "기록만 함, PatchTST에서는 비활성 해석",
                ]
                for item in CANDIDATES
            ],
        )
    )
    lines.extend(["", "## 2. 지표 설명", ""])
    lines.extend(
        _table(
            ["지표", "방향성", "의미"],
            [[info["label"], info["direction"], info["meaning"]] for info in METRIC_INFO.values()]
            + [["line_gate", "기본 생존 조건", "단독 제품 통과 기준이 아니다."]],
        )
    )
    lines.extend(["", "## 3. 비교 기준", ""])
    lines.extend(_table(["기준", "split", "IC", "spread", "fee", "false_safe", "severe"], _reference_rows()))
    lines.extend(["", "## 4. Validation seed 집계", ""])
    lines.extend(
        _table(
            ["후보", "판정", "IC med", "spread med", "fee med", "FS med", "FS std", "severe med", "severe std", "upside med"],
            _aggregate_table_rows(aggregates, "validation"),
        )
    )
    lines.extend(["", "## 5. Test seed 집계", ""])
    lines.extend(
        _table(
            ["후보", "판정", "IC med", "spread med", "fee med", "FS med", "FS std", "severe med", "severe std", "upside med"],
            _aggregate_table_rows(aggregates, "test"),
        )
    )
    lines.extend(["", "## 6. Seed별 raw metrics", ""])
    lines.extend(
        _table(
            ["후보", "split", "seed", "line_gate", "IC", "spread", "fee", "FS", "severe", "bias", "upside"],
            _run_table_rows(records),
        )
    )
    lines.extend(["", "## 7. Horizon / regime median", ""])
    lines.extend(
        _table(
            ["후보", "split", "구간", "FS med", "severe med", "spread med", "fee med"],
            _detail_table_rows(aggregates),
        )
    )
    lines.extend(["", "## 8. 후보 판단", ""])
    for candidate_id, payload in aggregates.items():
        lines.extend(
            [
                f"### {candidate_id}",
                "",
                f"- 판정: {payload['classification']}",
                f"- 이유: {payload['reason']}",
                f"- Stage 4-3 단일 seed 대비: seed 3개 median을 기준으로 재현성을 확인했다.",
                "",
            ]
        )
    lines.extend(
        [
            "## 9. 다음 액션",
            "",
            f"- 최종 판단: {conclusion}",
            "- Stage 4-5로 넘어가는 경우 product save-run이 아니라 3-fold walk-forward 검증으로만 진행한다.",
            "- 둘 다 흔들린 경우 product candidate save-run은 금지하고 Stage 4-3 후보 재선정으로 되돌린다.",
            "",
            "## 10. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- logs/meta: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
            "- script: `ai/cp148_lm_1d_stage4_4_seed_stability.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    alias = s41._configure_environment()
    s41.LOG_DIR = LOG_DIR
    s41._patch_training_for_experiment()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            record = _run_one(
                candidate,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                amp_dtype=args.amp_dtype,
                force=args.force,
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
                        "test_false_safe": record["test"].get("false_safe_tail_rate"),
                        "test_severe": record["test"].get("severe_downside_recall"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    aggregates = _aggregate_by_candidate(records)
    payload = {
        "cp": "CP148-LM-1D-Stage4-4",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_alias_dir": str(alias),
        "seed_policy": {
            "seeds": SEEDS,
            "reason": "Stage 4-2와 비교 가능하도록 42, 7, 123을 사용한다.",
        },
        "references": REFERENCES,
        "metric_dictionary": METRIC_INFO,
        "lambda_direction_note": "artifact에는 기록하지만 PatchTST direction head가 비활성이므로 성능 원인으로 해석하지 않는다.",
        "scope_compliance": {
            "product_save": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "model_structure_changed": False,
            "direction_head_added": False,
            "core_feature_contract_changed": False,
            "beta": 2.0,
            "eodhd_local_parquet": True,
        },
        "records": records,
        "aggregates": aggregates,
        "stage4_5_walk_forward_candidates": _walk_forward_candidates(aggregates),
        "process_check": {
            "status": "deferred_external_check",
            "note": "최종 외부 검증에서 python/pythonw 및 CUDA Python 학습 프로세스를 확인한다.",
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(records, aggregates)
    _write_report(records, aggregates)
    print(json.dumps({"status": "PASS", "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-4 seed stability")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
