from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
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
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ai.train as train_mod  # noqa: E402
from ai import cp148_lm_1d_stage4_1_risk_feature_abcd as s41  # noqa: E402
from ai import cp148_lm_1d_stage4_4_seed_stability as s44  # noqa: E402
from ai import cp148_lm_1d_stage4_4f_failure_analysis as f4  # noqa: E402
from ai import cp148_lm_1d_stage4_4g_overextension_feature_preflight as g4  # noqa: E402
from ai import cp148_lm_1d_stage4_0_risk_aware_rerank as r0  # noqa: E402
from ai.inference import load_checkpoint  # noqa: E402
from ai.preprocessing import MODEL_FEATURE_COLUMNS, SequenceDataset, normalize_sequence_splits  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4h_overextension_trial006_seed_stability_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4h_overextension_trial006_seed_stability_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4h_overextension_trial006_seed_stability_summary.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "cp148_lm_1d_stage4_4h_overextension_trial006_seed_stability"
STAGE4_4_METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_4_seed_stability_metrics.json"

SEEDS = [42, 7, 123]
C_FEATURES = ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"]
OVEREXTENSION_FEATURES = ["runup_20d", "runup_20d_xs_z", "ma60_extension_pos"]
EXTRA_FEATURES = [*C_FEATURES, *OVEREXTENSION_FEATURES]

CANDIDATE = {
    "candidate_id": "trial006_overextension",
    "base_candidate_id": "trial006_c_balanced",
    "display_name": "trial006 overextension 3-feature seed stability",
    "patch_len": 32,
    "patch_stride": 16,
    "lr": 0.0007362816234925851,
    "weight_decay": 0.00008143270337695065,
    "dropout": 0.10,
    "lambda_direction": 0.20,
}

REFERENCES = {
    "stage4_3_trial006_single_seed": {
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
}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
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
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _append_progress(run_dir: Path, message: str, **payload: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), "message": message, **payload}
    with (run_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def _load_stage4_4_references() -> dict[str, Any]:
    if not STAGE4_4_METRICS_PATH.exists():
        return {}
    payload = json.loads(STAGE4_4_METRICS_PATH.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload.get("records", [])
        if record.get("candidate_id") == CANDIDATE["base_candidate_id"]
    ]
    aggregate = ((payload.get("aggregates") or {}).get(CANDIDATE["base_candidate_id"]) or {}).get("aggregate") or {}
    return {"records": records, "aggregate": aggregate}


def _load_extra_feature_frame() -> pd.DataFrame:
    c_frame = s41._load_extra_feature_frame().copy()
    over_frame = g4._load_overextension_feature_frame().copy()
    combined = c_frame[C_FEATURES].join(over_frame[OVEREXTENSION_FEATURES], how="left")
    for column in EXTRA_FEATURES:
        combined[column] = combined[column].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return combined


def _aligned_extra_values(ticker: str, dates: np.ndarray, extra_frame: pd.DataFrame) -> np.ndarray:
    date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
    index = pd.MultiIndex.from_arrays([[ticker] * len(date_strings), date_strings], names=["ticker", "asof_date"])
    return extra_frame.reindex(index)[EXTRA_FEATURES].fillna(0.0).to_numpy(dtype="float32")


def _build_overextension_splits(*, device: str, batch_size: int) -> tuple[tuple, list[str]]:
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = s41._base_dataset(device=device, batch_size=batch_size)
    extra_frame = _load_extra_feature_frame()
    feature_names = [*s41.NO_FUND_FEATURES, *EXTRA_FEATURES]
    base_indices = [MODEL_FEATURE_COLUMNS.index(column) for column in s41.NO_FUND_FEATURES]
    ticker_arrays: dict[str, dict[str, Any]] = {}
    for ticker, arrays in train_bundle.ticker_arrays.items():
        base_values = arrays["features"][:, base_indices].astype("float32", copy=False)
        extra_values = _aligned_extra_values(ticker, arrays["dates"], extra_frame)
        copied = dict(arrays)
        copied["features"] = np.concatenate([base_values, extra_values], axis=1).astype("float32", copy=False)
        ticker_arrays[ticker] = copied

    def remake(bundle: SequenceDataset) -> SequenceDataset:
        return SequenceDataset(
            ticker_arrays=ticker_arrays,
            sample_refs=list(bundle.sample_refs),
            metadata=bundle.metadata.copy(),
            seq_len=bundle.seq_len,
            horizon=bundle.horizon,
            mean=None,
            std=None,
            include_future_covariate=bundle.include_future_covariate,
            line_target_type=bundle.line_target_type,
            band_target_type=bundle.band_target_type,
        )

    exact_train, exact_val, exact_test, mean, std = normalize_sequence_splits(
        remake(train_bundle),
        remake(val_bundle),
        remake(test_bundle),
    )
    return (exact_train, exact_val, exact_test, mean, std, plan), feature_names


def _experiment_for_seed(seed: int) -> dict[str, Any]:
    return {
        "experiment_id": f"stage4_4h_trial006_overextension_seed{seed}",
        "label": "C_stress_delta_overextension_3",
        "question": "과열 상승 종목의 quiet-tail reversal feature blind를 줄일 수 있는가?",
        "extra_features": list(EXTRA_FEATURES),
    }


def _make_config(
    experiment: dict[str, Any],
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
    config.patch_len = int(CANDIDATE["patch_len"])
    config.patch_stride = int(CANDIDATE["patch_stride"])
    config.lr = float(CANDIDATE["lr"])
    config.weight_decay = float(CANDIDATE["weight_decay"])
    config.dropout = float(CANDIDATE["dropout"])
    config.lambda_direction = float(CANDIDATE["lambda_direction"])
    config.use_direction_head = False
    config.feature_set = "C_stress_delta_overextension_3"
    config.checkpoint_selection = "line_gate"
    config.use_wandb = False
    config.compile_model = False
    config.market_data_provider = "eodhd"
    config.explicit_cuda_cleanup = True
    return config


def _false_safe_margin_diagnostic(
    *,
    checkpoint_path: str | Path,
    bundle: SequenceDataset,
    batch_size: int,
    device: str,
    amp_dtype: str,
) -> dict[str, Any]:
    model, _checkpoint = load_checkpoint(checkpoint_path)
    resolved_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        line, raw = r0._predict_bundle(
            model=model,
            bundle=bundle,
            device=resolved_device,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
    except (RuntimeError, TypeError) as exc:
        if "BFloat16" not in str(exc):
            raise
        line, raw = r0._predict_bundle(
            model=model,
            bundle=bundle,
            device=resolved_device,
            batch_size=batch_size,
            amp_dtype="fp32",
        )
    finally:
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    frame = f4._attach_tail_flags(f4._flat_event_frame(bundle.metadata, line, raw))
    false_safe_mask = frame["false_safe_tail"].to_numpy(dtype=bool)
    score = frame["score"].to_numpy(dtype=np.float64)
    buckets = f4._margin_buckets(score, false_safe_mask)
    return {
        "false_safe_margin_buckets": buckets,
        "strong_positive_false_safe_share_0p005_plus": (buckets.get("0p005_plus") or {}).get("share"),
        "near_zero_false_safe_share_0_to_0p005": sum(
            ((buckets.get(key) or {}).get("share") or 0.0)
            for key in ["0_to_0p001", "0p001_to_0p003", "0p003_to_0p005"]
        ),
    }


def _run_one(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    amp_dtype: str,
    force: bool,
    precomputed: tuple,
    feature_names: list[str],
) -> dict[str, Any]:
    experiment = _experiment_for_seed(seed)
    run_dir = LOG_DIR / experiment["experiment_id"]
    metrics_path = run_dir / "run_meta.json"
    process_path = run_dir / "train_process.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    started = time.perf_counter()
    process_meta = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": CANDIDATE["candidate_id"],
        "seed": seed,
        "device": device,
        "save_run": False,
        "db_write": False,
        "inference_save": False,
    }
    _write_json(process_path, process_meta)
    _append_progress(run_dir, "run_start", seed=seed)

    try:
        config = _make_config(
            experiment,
            epochs=epochs,
            seed=seed,
            batch_size=batch_size,
            device=device,
            feature_names=feature_names,
        )
        _append_progress(
            run_dir,
            "training_start",
            feature_dim=len(feature_names),
            extra_features=EXTRA_FEATURES,
            patch_len=config.patch_len,
            patch_stride=config.patch_stride,
            lr=config.lr,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
        )
        result = train_mod.run_training(
            config,
            save_run=False,
            precomputed_bundles=precomputed,
            enable_compile=False,
            local_log=True,
            local_log_dir=run_dir / "train_local_logs",
            wandb_group="cp148_stage4_4h",
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
        margin = _false_safe_margin_diagnostic(
            checkpoint_path=result["checkpoint_path"],
            bundle=test_bundle,
            batch_size=batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        record = {
            "candidate_id": CANDIDATE["candidate_id"],
            "base_candidate_id": CANDIDATE["base_candidate_id"],
            "display_name": CANDIDATE["display_name"],
            "experiment_id": experiment["experiment_id"],
            "feature_pack": "C_stress_delta_overextension_3",
            "extra_features": list(EXTRA_FEATURES),
            "overextension_features": list(OVEREXTENSION_FEATURES),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "amp_dtype": amp_dtype,
            "checkpoint_selection": "line_gate",
            "line_gate_meaning": "기본 생존 조건이며 단독 제품 통과 기준이 아님",
            "line_gate_meaning_changed": False,
            "lambda_direction_recorded": float(CANDIDATE["lambda_direction"]),
            "lambda_direction_interpretation": "PatchTST direction head 비활성이므로 성능 원인으로 해석하지 않음",
            "params": {
                "patch_len": int(CANDIDATE["patch_len"]),
                "patch_stride": int(CANDIDATE["patch_stride"]),
                "lr": float(CANDIDATE["lr"]),
                "weight_decay": float(CANDIDATE["weight_decay"]),
                "dropout": float(CANDIDATE["dropout"]),
                "lambda_direction": float(CANDIDATE["lambda_direction"]),
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
            "test_false_safe_margin": margin,
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
                "optuna": False,
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
            test_false_safe=record["test"].get("false_safe_tail_rate"),
            test_severe=record["test"].get("severe_downside_recall"),
            test_spread=record["test"].get("long_short_spread"),
        )
        return record
    except Exception as exc:
        process_meta.update(
            {
                "status": "failed",
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json(process_path, process_meta)
        _append_progress(run_dir, "run_failed", error=repr(exc))
        raise


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return s44._aggregate_records(records)


def _agg(aggregate: dict[str, Any], split: str, metric: str, stat: str = "median") -> float | None:
    return ((aggregate.get(split) or {}).get(metric) or {}).get(stat)


def _seed_record(records: list[dict[str, Any]], seed: int) -> dict[str, Any] | None:
    return next((record for record in records if int(record.get("seed")) == seed), None)


def _reference_seed_record(references: dict[str, Any], seed: int) -> dict[str, Any] | None:
    return next((record for record in references.get("records", []) if int(record.get("seed")) == seed), None)


def _judge(records: list[dict[str, Any]], aggregate: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    seed42 = _seed_record(records, 42)
    old_seed42 = _reference_seed_record(references, 42)
    old_aggregate = references.get("aggregate") or {}
    if seed42 is None or old_seed42 is None:
        return {"decision": "보류", "reason": "seed42 또는 기준 Stage 4-4 참조를 찾지 못함"}

    seed42_test = seed42.get("test") or {}
    old_seed42_test = old_seed42.get("test") or {}
    old_median_test = old_aggregate.get("test") or {}
    seed42_fs = _safe_float(seed42_test.get("false_safe_tail_rate"))
    seed42_severe = _safe_float(seed42_test.get("severe_downside_recall"))
    seed42_ic = _safe_float(seed42_test.get("ic_mean"))
    seed42_spread = _safe_float(seed42_test.get("long_short_spread"))
    old_seed42_fs = _safe_float(old_seed42_test.get("false_safe_tail_rate"))
    old_seed42_severe = _safe_float(old_seed42_test.get("severe_downside_recall"))
    old_seed42_strong = 0.8980560055542699
    new_seed42_strong = _safe_float((seed42.get("test_false_safe_margin") or {}).get("strong_positive_false_safe_share_0p005_plus"))

    median_fs = _agg(aggregate, "test", "false_safe_tail_rate")
    median_severe = _agg(aggregate, "test", "severe_downside_recall")
    median_ic = _agg(aggregate, "test", "ic_mean")
    median_spread = _agg(aggregate, "test", "long_short_spread")
    old_median_fs = ((old_median_test.get("false_safe_tail_rate") or {}).get("median"))
    old_median_severe = ((old_median_test.get("severe_downside_recall") or {}).get("median"))

    seed42_success = (
        seed42_fs is not None
        and seed42_severe is not None
        and seed42_ic is not None
        and seed42_spread is not None
        and seed42_fs <= 0.31
        and seed42_severe >= 0.69
        and seed42_ic > 0.0
        and seed42_spread > 0.0
    )
    strong_positive_reduced = (
        new_seed42_strong is not None
        and new_seed42_strong < old_seed42_strong
    )
    median_maintained = (
        median_fs is not None
        and median_severe is not None
        and median_ic is not None
        and median_spread is not None
        and old_median_fs is not None
        and old_median_severe is not None
        and median_fs <= old_median_fs + 0.01
        and median_severe >= old_median_severe - 0.01
        and median_ic > 0.0
        and median_spread > 0.0
    )
    other_seed_degrade = []
    for seed in [7, 123]:
        new_record = _seed_record(records, seed)
        old_record = _reference_seed_record(references, seed)
        if not new_record or not old_record:
            continue
        new_test = new_record.get("test") or {}
        old_test = old_record.get("test") or {}
        new_fs = _safe_float(new_test.get("false_safe_tail_rate"))
        old_fs = _safe_float(old_test.get("false_safe_tail_rate"))
        new_severe = _safe_float(new_test.get("severe_downside_recall"))
        old_severe = _safe_float(old_test.get("severe_downside_recall"))
        degraded = (
            new_fs is not None
            and old_fs is not None
            and new_severe is not None
            and old_severe is not None
            and (new_fs > old_fs + 0.03 or new_severe < old_severe - 0.03)
        )
        other_seed_degrade.append({"seed": seed, "degraded": degraded, "new_false_safe": new_fs, "old_false_safe": old_fs, "new_severe": new_severe, "old_severe": old_severe})
    no_other_seed_big_degrade = not any(item["degraded"] for item in other_seed_degrade)

    if seed42_success and median_maintained and no_other_seed_big_degrade:
        decision = "trial024로 확장"
        reason = "seed42 위험 오판이 기준선 이하로 개선됐고 3-seed median risk도 기존 trial006 수준을 유지했다."
    elif (old_seed42_fs is not None and seed42_fs is not None and seed42_fs < old_seed42_fs) or (
        old_seed42_severe is not None and seed42_severe is not None and seed42_severe > old_seed42_severe
    ):
        decision = "보류"
        reason = "seed42 일부 개선은 있으나 성공 기준 또는 median 안정성 조건을 모두 만족하지 못했다."
    else:
        decision = "실패"
        reason = "seed42 false_safe/severe가 의미 있게 개선되지 않았거나 median risk가 악화됐다."

    return {
        "decision": decision,
        "reason": reason,
        "seed42_success": seed42_success,
        "median_maintained": median_maintained,
        "no_other_seed_big_degrade": no_other_seed_big_degrade,
        "strong_positive_reduced": strong_positive_reduced,
        "seed42_comparison": {
            "new_false_safe": seed42_fs,
            "old_false_safe": old_seed42_fs,
            "new_severe": seed42_severe,
            "old_severe": old_seed42_severe,
            "new_ic": seed42_ic,
            "new_spread": seed42_spread,
            "new_strong_positive_false_safe_share": new_seed42_strong,
            "old_strong_positive_false_safe_share": old_seed42_strong,
        },
        "median_comparison": {
            "new_false_safe": median_fs,
            "old_false_safe": old_median_fs,
            "new_severe": median_severe,
            "old_severe": old_median_severe,
            "new_ic": median_ic,
            "new_spread": median_spread,
        },
        "other_seed_degrade": other_seed_degrade,
    }


def _summary_rows(records: list[dict[str, Any]], aggregate: dict[str, Any], references: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for split in ["validation", "test"]:
            summary = s44._extract_split_summary(record, split)
            rows.append(
                {
                    "row_type": "run",
                    "candidate_id": record["candidate_id"],
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
                    "h1_false_safe_tail_rate": summary.get("h1_false_safe_tail_rate"),
                    "h1_severe_downside_recall": summary.get("h1_severe_downside_recall"),
                    "h2_h3_false_safe_tail_rate": summary.get("h2_h3_false_safe_tail_rate"),
                    "h2_h3_severe_downside_recall": summary.get("h2_h3_severe_downside_recall"),
                    "h4_h5_false_safe_tail_rate": summary.get("h4_h5_false_safe_tail_rate"),
                    "h4_h5_severe_downside_recall": summary.get("h4_h5_severe_downside_recall"),
                    "stress_false_safe_tail_rate": summary.get("stress_false_safe_tail_rate"),
                    "stress_severe_downside_recall": summary.get("stress_severe_downside_recall"),
                    "strong_positive_false_safe_share_0p005_plus": (record.get("test_false_safe_margin") or {}).get("strong_positive_false_safe_share_0p005_plus") if split == "test" else "",
                    "checkpoint_path": record.get("checkpoint_path"),
                }
            )
    for split in ["validation", "test"]:
        metrics = aggregate.get(split) or {}
        for stat in ["median", "mean", "std", "min", "max"]:
            rows.append(
                {
                    "row_type": "aggregate",
                    "candidate_id": CANDIDATE["candidate_id"],
                    "split": split,
                    "stat": stat,
                    "seed": "",
                    "line_gate_pass": all(bool(record.get("line_gate_pass")) for record in records),
                    "ic_mean": (metrics.get("ic_mean") or {}).get(stat),
                    "long_short_spread": (metrics.get("long_short_spread") or {}).get(stat),
                    "fee_adjusted_return": (metrics.get("fee_adjusted_return") or {}).get(stat),
                    "false_safe_tail_rate": (metrics.get("false_safe_tail_rate") or {}).get(stat),
                    "severe_downside_recall": (metrics.get("severe_downside_recall") or {}).get(stat),
                    "conservative_bias": (metrics.get("conservative_bias") or {}).get(stat),
                    "upside_sacrifice": (metrics.get("upside_sacrifice") or {}).get(stat),
                }
            )
    for record in references.get("records", []):
        for split in ["validation", "test"]:
            summary = s44._extract_split_summary(record, split)
            rows.append(
                {
                    "row_type": "reference_stage4_4_trial006",
                    "candidate_id": record.get("candidate_id"),
                    "split": split,
                    "stat": "raw",
                    "seed": record.get("seed"),
                    "ic_mean": summary.get("ic_mean"),
                    "long_short_spread": summary.get("long_short_spread"),
                    "fee_adjusted_return": summary.get("fee_adjusted_return"),
                    "false_safe_tail_rate": summary.get("false_safe_tail_rate"),
                    "severe_downside_recall": summary.get("severe_downside_recall"),
                    "conservative_bias": summary.get("conservative_bias"),
                    "upside_sacrifice": summary.get("upside_sacrifice"),
                }
            )
    return rows


def _write_summary_csv(records: list[dict[str, Any]], aggregate: dict[str, Any], references: dict[str, Any]) -> None:
    rows = _summary_rows(records, aggregate, references)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _json_safe(value) for key, value in row.items()} for row in rows])


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _run_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for record in sorted(records, key=lambda item: int(item["seed"])):
        test = record.get("test") or {}
        margin = record.get("test_false_safe_margin") or {}
        rows.append(
            [
                record["seed"],
                record["line_gate_pass"],
                _fmt(test.get("ic_mean")),
                _fmt(test.get("long_short_spread")),
                _fmt(test.get("fee_adjusted_return")),
                _fmt(test.get("false_safe_tail_rate")),
                _fmt(test.get("severe_downside_recall")),
                _fmt(test.get("conservative_bias")),
                _fmt(margin.get("strong_positive_false_safe_share_0p005_plus")),
            ]
        )
    return rows


def _aggregate_rows(aggregate: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for split in ["validation", "test"]:
        for stat in ["median", "mean", "std"]:
            rows.append(
                [
                    split,
                    stat,
                    _fmt(_agg(aggregate, split, "ic_mean", stat)),
                    _fmt(_agg(aggregate, split, "long_short_spread", stat)),
                    _fmt(_agg(aggregate, split, "fee_adjusted_return", stat)),
                    _fmt(_agg(aggregate, split, "false_safe_tail_rate", stat)),
                    _fmt(_agg(aggregate, split, "severe_downside_recall", stat)),
                ]
            )
    return rows


def _write_report(records: list[dict[str, Any]], aggregate: dict[str, Any], references: dict[str, Any], judgement: dict[str, Any]) -> None:
    seed42 = judgement["seed42_comparison"]
    median = judgement["median_comparison"]
    lines: list[str] = [
        "# CP148-LM-1D Stage 4-4H overextension trial006 seed stability 보고서",
        "",
        f"- 작성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "- 범위: trial006_c_balanced 설정 + Stage 4-4G 추천 overextension 3개 피처, seed 42/7/123 3회",
        "- 이 실험은 기준 완화가 아니라 feature blind 문제 개선 실험이다.",
        "- 금지 작업 준수: product save-run, DB write, inference 저장, live fetch, band/composite, 모델 구조 변경, direction head 추가, core feature contract 전역 변경, Optuna 모두 미실행",
        "",
        "## 1. 핵심 결론",
        "",
        f"최종 판단: `{judgement['decision']}`. {judgement['reason']}",
        "",
        "## 2. seed42 개선 여부",
        "",
        f"- false_safe_tail_rate: 기존 `{_fmt(seed42.get('old_false_safe'))}` -> 신규 `{_fmt(seed42.get('new_false_safe'))}`",
        f"- severe_downside_recall: 기존 `{_fmt(seed42.get('old_severe'))}` -> 신규 `{_fmt(seed42.get('new_severe'))}`",
        f"- IC/spread: 신규 IC `{_fmt(seed42.get('new_ic'))}`, 신규 spread `{_fmt(seed42.get('new_spread'))}`",
        f"- strong positive false-safe 0.005+ 비중: 기존 `{_fmt(seed42.get('old_strong_positive_false_safe_share'))}` -> 신규 `{_fmt(seed42.get('new_strong_positive_false_safe_share'))}`",
        f"- seed42 성공 기준 충족: `{judgement['seed42_success']}`",
        "",
        "false_safe는 실제 하위 꼬리인데 line이 0 이상으로 안전하다고 본 비율이다. severe recall은 심한 하락을 음수 위험으로 잡아낸 비율이다.",
        "",
        "## 3. seed별 test 결과",
        "",
    ]
    lines.extend(
        _table(
            ["seed", "line_gate", "IC", "spread", "fee", "false_safe", "severe", "bias", "FS 0.005+"],
            _run_rows(records),
        )
    )
    lines.extend(["", "## 4. 3-seed aggregate", ""])
    lines.extend(
        _table(
            ["split", "stat", "IC", "spread", "fee", "false_safe", "severe"],
            _aggregate_rows(aggregate),
        )
    )
    lines.extend(
        [
            "",
            "## 5. 기준 비교",
            "",
            f"- Stage 4-4 trial006 test median false_safe `{_fmt(median.get('old_false_safe'))}` / severe `{_fmt(median.get('old_severe'))}`",
            f"- 신규 test median false_safe `{_fmt(median.get('new_false_safe'))}` / severe `{_fmt(median.get('new_severe'))}`",
            f"- Stage 4-3 trial006 단일 seed test false_safe `{_fmt(REFERENCES['stage4_3_trial006_single_seed']['test']['false_safe_tail_rate'])}` / severe `{_fmt(REFERENCES['stage4_3_trial006_single_seed']['test']['severe_downside_recall'])}`",
            f"- Stage 4-2 C median test false_safe `{_fmt(REFERENCES['stage4_2_c_stress_delta_seed_median']['test']['false_safe_tail_rate'])}` / severe `{_fmt(REFERENCES['stage4_2_c_stress_delta_seed_median']['test']['severe_downside_recall'])}`",
            "",
            "## 6. feature pack",
            "",
            "- C_stress_delta 유지: atr_ratio, vix_change_5d, credit_spread_change_20d, ma200_pct_change_20d",
            "- 추가: runup_20d, runup_20d_xs_z, ma60_extension_pos",
            "- 제외 유지: runup_20d_xs_rank, ma20_extension_pos, max_down_day_20d, pullback_from_20d_high",
            "",
            "## 7. lambda_direction 메모",
            "",
            "lambda_direction=0.20은 artifact에 기록하지만 PatchTST direction head가 비활성이므로 성능 원인으로 해석하지 않는다.",
            "",
            "## 8. 과적합 위험",
            "",
            "- overextension feature는 Stage 4-4F의 blind spot을 직접 겨냥하므로 해석은 명확하지만, runup/ma extension 계열끼리 중복될 수 있다.",
            "- 이번 결과가 성공이어도 trial024 확장 전에는 product 후보나 save-run 후보로 부르지 않는다.",
            "",
            "## 9. 다음 액션",
            "",
            f"- 판단: `{judgement['decision']}`",
            "- `trial024로 확장`이면 같은 feature pack을 trial024 설정에 3-seed로 붙여 재검증한다.",
            "- `보류`이면 피처는 유지 후보로 두되 spread/IC 또는 seed 악화 원인을 다시 본다.",
            "- `실패`이면 overextension 피처로 feature blind를 해결하지 못한 것으로 정리한다.",
            "",
            "## 10. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
            "- script: `ai/cp148_lm_1d_stage4_4h_overextension_trial006_seed_stability.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["WANDB_MODE"] = "disabled"
    alias = s41._configure_environment()
    s41.LOG_DIR = LOG_DIR
    s41._patch_training_for_experiment()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    precomputed, feature_names = _build_overextension_splits(device=args.device, batch_size=args.batch_size)

    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        record = _run_one(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            force=args.force,
            precomputed=precomputed,
            feature_names=feature_names,
        )
        records.append(record)

    aggregate = _aggregate_records(records)
    references = _load_stage4_4_references()
    judgement = _judge(records, aggregate, references)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope_compliance": {
            "product_save": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "model_structure_changed": False,
            "direction_head_added": False,
            "core_feature_contract_changed": False,
            "optuna": False,
            "beta": 2.0,
        },
        "data": {
            "provider": "eodhd",
            "backend": "local_parquet",
            "timeframe": "1D",
            "horizon": 5,
            "alias_dir": str(alias),
            "source_data_hash": records[0].get("source_data_hash") if records else None,
            "eligible_ticker_count": records[0].get("eligible_ticker_count") if records else None,
        },
        "candidate": CANDIDATE,
        "feature_pack": {
            "base": "C_stress_delta",
            "extra_features": EXTRA_FEATURES,
            "overextension_features": OVEREXTENSION_FEATURES,
            "global_feature_contract_changed": False,
        },
        "references": {
            "stage4_4_trial006": references,
            **REFERENCES,
        },
        "records": records,
        "aggregate": aggregate,
        "judgement": judgement,
        "final_decision": judgement.get("decision"),
        "lambda_direction_note": "artifact에는 기록하지만 PatchTST direction head가 비활성이므로 성능 원인으로 해석하지 않는다.",
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(records, aggregate, references)
    _write_report(records, aggregate, references, judgement)
    print(json.dumps({"status": "PASS", "decision": judgement.get("decision"), "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-4H overextension trial006 seed stability")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
