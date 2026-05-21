from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import warnings

from ai import cp148_lm_1d_stage0_2 as s2
from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage3_false_safe_sweep_logs"
ALIAS_DIR = LOG_DIR / "snapshot_alias"
ARCHIVE_LOG_DIR = PROJECT_ROOT / "docs" / "cp_archive" / "run_logs" / "cp148_lm_1d_stage3_false_safe_sweep_logs"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


STAGE2_METRICS_PATH = _first_existing(
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_metrics.json",
    PROJECT_ROOT / "docs" / "cp_archive" / "model_line" / "cp148_lm_1d_stage0_2_metrics.json",
)
STAGE2_SUMMARY_CSV_PATH = _first_existing(
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_summary.csv",
    PROJECT_ROOT / "docs" / "cp_archive" / "model_line" / "cp148_lm_1d_stage0_2_summary.csv",
)
STAGE2_PREFLIGHT_PATH = _first_existing(
    PROJECT_ROOT / "docs" / "cp148_lm_1d_stage0_2_logs" / "preflight.json",
    PROJECT_ROOT / "docs" / "cp_archive" / "run_logs" / "cp148_lm_1d_stage0_2_logs" / "preflight.json",
)

STAGE2_BEST_FALSE_SAFE = 0.30866056266521946
STAGE2_BEST_SEVERE_RECALL = 0.6850190785352241
FALSE_SAFE_TARGET = 0.27
PRODUCT_FALSE_SAFE_TARGET = 0.20
PRODUCT_SEVERE_TARGET = 0.75

LINE_KEYS = [
    "ic_mean",
    "ic_std",
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
    "mae",
    "smape",
]

BASE_SPECS = {
    "patchtst_no_fund_p32_s16": {
        "candidate": "patchtst_no_fund_p32_s16",
        "model": "patchtst",
        "feature_set": "no_fundamentals",
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "stage2_candidate": "cp148_s2_patchtst_no_fund_p32_s16",
    },
    "patchtst_pvv_p16_s8": {
        "candidate": "patchtst_pvv_p16_s8",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 16,
        "patch_stride": 8,
        "stage2_candidate": "cp148_s2_patchtst_pvv_p16_s8",
    },
    "patchtst_pvv_p32_s16_reference": {
        "candidate": "patchtst_pvv_p32_s16_reference",
        "model": "patchtst",
        "feature_set": "price_volatility_volume",
        "seq_len": 252,
        "patch_len": 32,
        "patch_stride": 16,
        "stage2_candidate": "cp148_s2_patchtst_pvv_p32_s16",
    },
}

TRIALS = [
    {
        "trial_id": "s3_no_fund_d15_wd01_lr1e4",
        "base": "patchtst_no_fund_p32_s16",
        "dropout": 0.15,
        "weight_decay": 0.01,
        "lr": 1e-4,
        "priority": "core",
    },
    {
        "trial_id": "s3_no_fund_d25_wd05_lr1e4",
        "base": "patchtst_no_fund_p32_s16",
        "dropout": 0.25,
        "weight_decay": 0.05,
        "lr": 1e-4,
        "priority": "core",
    },
    {
        "trial_id": "s3_no_fund_d35_wd10_lr5e5",
        "base": "patchtst_no_fund_p32_s16",
        "dropout": 0.35,
        "weight_decay": 0.10,
        "lr": 5e-5,
        "priority": "core",
    },
    {
        "trial_id": "s3_no_fund_d25_wd10_lr2e4",
        "base": "patchtst_no_fund_p32_s16",
        "dropout": 0.25,
        "weight_decay": 0.10,
        "lr": 2e-4,
        "priority": "extra",
    },
    {
        "trial_id": "s3_pvv_p16_d15_wd01_lr1e4",
        "base": "patchtst_pvv_p16_s8",
        "dropout": 0.15,
        "weight_decay": 0.01,
        "lr": 1e-4,
        "priority": "core",
    },
    {
        "trial_id": "s3_pvv_p16_d25_wd05_lr1e4",
        "base": "patchtst_pvv_p16_s8",
        "dropout": 0.25,
        "weight_decay": 0.05,
        "lr": 1e-4,
        "priority": "core",
    },
    {
        "trial_id": "s3_pvv_p16_d35_wd10_lr5e5",
        "base": "patchtst_pvv_p16_s8",
        "dropout": 0.35,
        "weight_decay": 0.10,
        "lr": 5e-5,
        "priority": "core",
    },
    {
        "trial_id": "s3_pvv_p32_ref_d25_wd05_lr1e4",
        "base": "patchtst_pvv_p32_s16_reference",
        "dropout": 0.25,
        "weight_decay": 0.05,
        "lr": 1e-4,
        "priority": "reference",
    },
]

WEIGHTS = {
    "ic_mean": 0.20,
    "long_short_spread": 0.15,
    "fee_adjusted_return": 0.15,
    "false_safe_tail_rate": 0.25,
    "severe_downside_recall": 0.20,
    "upside_sacrifice": 0.05,
}

FEATURE_COLUMNS_BY_SET = {
    "price_volatility_volume": [
        "log_return",
        "open_ratio",
        "high_ratio",
        "low_ratio",
        "vol_change",
        "ma_5_ratio",
        "ma_20_ratio",
        "ma_60_ratio",
        "rsi",
        "macd_ratio",
        "bb_position",
    ],
    "no_fundamentals": [
        "log_return",
        "open_ratio",
        "high_ratio",
        "low_ratio",
        "vol_change",
        "ma_5_ratio",
        "ma_20_ratio",
        "ma_60_ratio",
        "rsi",
        "macd_ratio",
        "bb_position",
        "us10y",
        "yield_spread",
        "vix_close",
        "credit_spread_hy",
        "nh_nl_index",
        "ma200_pct",
        "regime_calm",
        "regime_neutral",
        "regime_stress",
        "has_macro",
        "has_breadth",
        "has_fundamentals",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_month_end",
        "is_quarter_end",
        "is_opex_friday",
    ],
}


def _json_safe(value: Any) -> Any:
    return s2._json_safe(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_float(value: Any) -> float | None:
    return s2._safe_float(value)


def _fmt(value: Any, digits: int = 6) -> str:
    return s2._fmt(value, digits=digits)


def _log(stage: str, payload: dict[str, Any] | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **(payload or {}),
    }
    with (LOG_DIR / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(record), ensure_ascii=False), flush=True)


def _bind_stage3_paths() -> None:
    s2.LOG_DIR = LOG_DIR
    s2.ALIAS_DIR = ALIAS_DIR
    s2.PREFLIGHT_PATH = STAGE2_PREFLIGHT_PATH
    s2.cp146.LOG_DIR = LOG_DIR
    s2.cp146.ALIAS_DIR = ALIAS_DIR
    s2.cp146.PREFLIGHT_PATH = STAGE2_PREFLIGHT_PATH


def _set_env() -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["MARKET_DATA_PROVIDER"] = "eodhd"
    os.environ["LENS_DATA_BACKEND"] = "local"
    os.environ["LENS_USE_LOCAL_SNAPSHOTS"] = "1"
    os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
    os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(ALIAS_DIR)


def _trial_dir(trial_id: str) -> Path:
    current = LOG_DIR / trial_id
    if current.exists():
        return current
    archived = ARCHIVE_LOG_DIR / trial_id
    if archived.exists():
        return archived
    return current


def _current_trial_dir(trial_id: str) -> Path:
    return LOG_DIR / trial_id


def _trial_process_path(trial_id: str) -> Path:
    return _trial_dir(trial_id) / "trial_process.json"


def _trial_payload_path(trial_id: str) -> Path:
    return _trial_dir(trial_id) / "trial_metrics.json"


def _latest_summary(trial_id: str) -> dict[str, Any]:
    base = _trial_dir(trial_id) / "train_local_logs"
    if not base.exists():
        return _summary_from_console(trial_id)
    run_dirs = [path for path in base.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not run_dirs:
        return _summary_from_console(trial_id)
    latest = sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    payload = _read_json(latest / "summary.json")
    payload["_run_dir"] = str(latest)
    config_path = latest / "config.json"
    if config_path.exists():
        payload["_config"] = _read_json(config_path).get("config", {})
    return payload


def _summary_from_console(trial_id: str) -> dict[str, Any]:
    console_path = _trial_dir(trial_id) / "console.log"
    if not console_path.exists():
        return {}
    epochs: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    with console_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if "checkpoint_selection" in record:
                selection = record
            if "epoch" in record and ("val/line_metrics" in record or "val/ic_mean" in record):
                epochs.append(record)
    if not epochs:
        return {}
    selected_epoch = selection.get("selected_epoch")
    selected = None
    if selected_epoch is not None:
        selected = next((record for record in epochs if record.get("epoch") == selected_epoch), None)
    selected = selected or epochs[-1]
    selected["selection"] = selection
    selected["_run_dir"] = str(console_path)
    return selected


def _metrics_from_summary(summary: dict[str, Any], split: str) -> dict[str, Any]:
    if split == "val" and isinstance(summary.get("best_metrics"), dict):
        return {key: summary["best_metrics"].get(key) for key in LINE_KEYS if summary["best_metrics"].get(key) is not None}
    if split == "test" and isinstance(summary.get("test_metrics"), dict):
        return {key: summary["test_metrics"].get(key) for key in LINE_KEYS if summary["test_metrics"].get(key) is not None}
    prefix = f"{split}/"
    line = summary.get(f"{split}/line_metrics")
    if isinstance(line, dict):
        return {key: line.get(key) for key in LINE_KEYS if line.get(key) is not None}
    return {key: summary.get(prefix + key) for key in LINE_KEYS if summary.get(prefix + key) is not None}


def _stage2_records() -> dict[str, dict[str, Any]]:
    if not STAGE2_METRICS_PATH.exists():
        return {}
    payload = _read_json(STAGE2_METRICS_PATH)
    records = payload.get("stage2_candidates") or []
    return {str(row.get("candidate")): row for row in records if row.get("candidate")}


def _build_config(trial: dict[str, Any], *, epochs: int, seed: int, device: str, batch_size: int) -> Any:
    base = BASE_SPECS[str(trial["base"])].copy()
    base["candidate"] = str(trial["trial_id"])
    config = s2._make_config(
        candidate=base,
        epochs=epochs,
        seed=seed,
        limit_tickers=None,
        device=device,
        batch_size=batch_size,
    )
    config.lr = float(trial["lr"])
    config.dropout = float(trial["dropout"])
    config.weight_decay = float(trial["weight_decay"])
    config.feature_columns = list(FEATURE_COLUMNS_BY_SET[str(base["feature_set"])])
    config.n_features = len(config.feature_columns)
    config.use_wandb = False
    config.wandb_project = "lens-cp148-stage3-local"
    config.checkpoint_selection = "line_gate"
    return config


def run_trial(trial_id: str, *, epochs: int, seed: int, device: str, batch_size: int, force: bool = False) -> dict[str, Any]:
    _bind_stage3_paths()
    s2.cp146.ensure_snapshot_alias()
    _set_env()
    trial = next((item for item in TRIALS if item["trial_id"] == trial_id), None)
    if trial is None:
        raise KeyError(trial_id)
    process_path = _trial_process_path(trial_id)
    if process_path.exists() and not force:
        process = _read_json(process_path)
        if process.get("exit_code") == 0:
            _log("trial_skip_completed", {"trial": trial_id, "exit_code": process.get("exit_code")})
            return process
        _log("trial_rerun_failed_previous", {"trial": trial_id, "previous_exit_code": process.get("exit_code")})

    trial_dir = _current_trial_dir(trial_id)
    local_log_dir = trial_dir / "train_local_logs"
    console_log = trial_dir / "console.log"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(trial, epochs=epochs, seed=seed, device=device, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    error = None
    result: dict[str, Any] | None = None
    exit_code = 0
    _log(
        "trial_start",
        {
            "trial": trial_id,
            "base": trial["base"],
            "epochs": epochs,
            "seed": seed,
            "dropout": trial["dropout"],
            "weight_decay": trial["weight_decay"],
            "lr": trial["lr"],
        },
    )
    try:
        with console_log.open("a", encoding="utf-8") as handle, redirect_stdout(handle), redirect_stderr(handle):
            precomputed = s2._prepare_dataset_splits_with_progress(config, trial_dir)
            result = s2.run_training(
                config,
                save_run=False,
                precomputed_bundles=precomputed,
                local_log=True,
                local_log_dir=local_log_dir,
                wandb_name=f"cp148_stage3_{trial_id}_seed{seed}",
                wandb_group="cp148_lm_1d_stage3",
                wandb_config_override={
                    **asdict(config),
                    "cp": "CP148-LM-1D-Stage3",
                    "trial_id": trial_id,
                    "save_run": False,
                    "db_write": False,
                    "inference_save": False,
                },
            )
    except Exception as exc:
        exit_code = 1
        error = str(exc)
        _log("trial_failed", {"trial": trial_id, "error": error})
    elapsed = time.perf_counter() - started
    if torch.cuda.is_available():
        peak_vram = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    else:
        peak_vram = None
    gc.collect()
    process = {
        "trial_id": trial_id,
        "base": trial["base"],
        "priority": trial["priority"],
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "epochs": epochs,
        "seed": seed,
        "save_run": False,
        "db_write": False,
        "inference_save": False,
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "dropout": trial["dropout"],
        "weight_decay": trial["weight_decay"],
        "lr": trial["lr"],
        "result_run_id": (result or {}).get("run_id") if isinstance(result, dict) else None,
        "checkpoint_path": (result or {}).get("checkpoint_path") if isinstance(result, dict) else None,
        "peak_vram_bytes": peak_vram,
        "local_log_dir": str(local_log_dir),
        "console_log": str(console_log),
        "error": error,
    }
    _write_json(process_path, process)
    summary = _latest_summary(trial_id)
    if summary:
        _write_json(
            _trial_payload_path(trial_id),
            {
                "trial": trial,
                "process": process,
                "validation": _metrics_from_summary(summary, "val"),
                "test": _metrics_from_summary(summary, "test"),
                "line_gate_pass": bool(
                    (summary.get("selection") or {}).get("line_gate_pass")
                    or (summary.get("best_metrics") or {}).get("line_gate_pass")
                    or _metrics_from_summary(summary, "val").get("line_gate_pass")
                ),
                "summary_path": summary.get("_run_dir"),
            },
        )
    if exit_code != 0:
        raise RuntimeError(error or f"{trial_id} failed")
    _log("trial_done", {"trial": trial_id, "elapsed_seconds": round(elapsed, 3), "run_id": process.get("result_run_id")})
    return process


def _rank_values(records: list[dict[str, Any]], key: str, *, lower_better: bool = False) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for record in records:
        value = _safe_float((record.get("validation") or {}).get(key))
        if value is not None:
            values.append((str(record["trial_id"]), value))
    values.sort(key=lambda item: item[1], reverse=not lower_better)
    n = len(values)
    if n <= 1:
        return {trial_id: 1.0 for trial_id, _ in values}
    return {trial_id: 1.0 - rank / (n - 1) for rank, (trial_id, _) in enumerate(values)}


def _score_trials(records: list[dict[str, Any]]) -> None:
    ranks = {
        "ic_mean": _rank_values(records, "ic_mean"),
        "long_short_spread": _rank_values(records, "long_short_spread"),
        "fee_adjusted_return": _rank_values(records, "fee_adjusted_return"),
        "false_safe_tail_rate": _rank_values(records, "false_safe_tail_rate", lower_better=True),
        "severe_downside_recall": _rank_values(records, "severe_downside_recall"),
        "upside_sacrifice": _rank_values(records, "upside_sacrifice", lower_better=True),
    }
    for record in records:
        trial_id = str(record["trial_id"])
        parts = {key: ranks[key].get(trial_id, 0.0) for key in WEIGHTS}
        record["composite_rank_parts"] = parts
        record["composite_score"] = sum(parts[key] * weight for key, weight in WEIGHTS.items())


def _failure_types(record: dict[str, Any]) -> list[str]:
    metrics = record.get("validation") or {}
    failures = []
    if not record.get("line_gate_pass"):
        failures.append("line_gate_fail")
    if (_safe_float(metrics.get("ic_mean")) or -999.0) <= 0:
        failures.append("ranking_fail")
    if (_safe_float(metrics.get("long_short_spread")) or -999.0) <= 0:
        failures.append("spread_fail")
    if (_safe_float(metrics.get("fee_adjusted_return")) or -999.0) <= 0:
        failures.append("fee_negative_fail")
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    if false_safe is None or false_safe >= STAGE2_BEST_FALSE_SAFE:
        failures.append("false_safe_not_improved")
    if false_safe is None or false_safe > FALSE_SAFE_TARGET:
        failures.append("false_safe_target_miss")
    if severe is None or severe <= STAGE2_BEST_SEVERE_RECALL:
        failures.append("severe_recall_not_improved")
    if severe is None or severe < 0.70:
        failures.append("severe_recall_0p70_miss")
    return failures


def _classify(record: dict[str, Any]) -> str:
    failures = set(record.get("failure_types") or [])
    metrics = record.get("validation") or {}
    positive_core = not {"ranking_fail", "spread_fail", "fee_negative_fail", "line_gate_fail"} & failures
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    if positive_core and false_safe is not None and severe is not None:
        if false_safe < STAGE2_BEST_FALSE_SAFE and severe > STAGE2_BEST_SEVERE_RECALL:
            return "false-safe 개선 후보"
        if false_safe < STAGE2_BEST_FALSE_SAFE or severe > STAGE2_BEST_SEVERE_RECALL:
            return "부분 개선 후보"
    if false_safe is not None and severe is not None and false_safe < STAGE2_BEST_FALSE_SAFE and severe > STAGE2_BEST_SEVERE_RECALL:
        return "risk-only 탈락"
    return "탈락 후보"


def collect_results() -> list[dict[str, Any]]:
    stage2 = _stage2_records()
    records = []
    for trial in TRIALS:
        trial_id = str(trial["trial_id"])
        payload_path = _trial_payload_path(trial_id)
        process_path = _trial_process_path(trial_id)
        process = _read_json(process_path) if process_path.exists() else {}
        payload = _read_json(payload_path) if payload_path.exists() else {}
        if (not payload or not payload.get("validation")) and process.get("exit_code") == 0:
            summary = _latest_summary(trial_id)
            if summary:
                payload = {
                    "trial": trial,
                    "process": process,
                    "validation": _metrics_from_summary(summary, "val"),
                    "test": _metrics_from_summary(summary, "test"),
                    "line_gate_pass": bool(
                        (summary.get("selection") or {}).get("line_gate_pass")
                        or (summary.get("best_metrics") or {}).get("line_gate_pass")
                    ),
                    "summary_path": summary.get("_run_dir"),
                }
                _write_json(payload_path, payload)
        validation = payload.get("validation") or {}
        test = payload.get("test") or {}
        base_spec = BASE_SPECS[str(trial["base"])]
        stage2_record = stage2.get(str(base_spec["stage2_candidate"]), {})
        stage2_validation = stage2_record.get("validation") or {}
        false_safe = _safe_float(validation.get("false_safe_tail_rate"))
        severe = _safe_float(validation.get("severe_downside_recall"))
        stage2_false_safe = _safe_float(stage2_validation.get("false_safe_tail_rate"))
        stage2_severe = _safe_float(stage2_validation.get("severe_downside_recall"))
        record = {
            "trial_id": trial_id,
            "base": trial["base"],
            "stage2_candidate": base_spec["stage2_candidate"],
            "model": base_spec["model"],
            "feature_set": base_spec["feature_set"],
            "seq_len": base_spec["seq_len"],
            "patch_len": base_spec["patch_len"],
            "patch_stride": base_spec["patch_stride"],
            "dropout": trial["dropout"],
            "weight_decay": trial["weight_decay"],
            "lr": trial["lr"],
            "priority": trial["priority"],
            "status": "PASS" if process.get("exit_code") == 0 and validation else "MISSING_OR_FAILED",
            "process": process,
            "validation": validation,
            "test": test,
            "line_gate_pass": bool(payload.get("line_gate_pass") or validation.get("line_gate_pass")),
            "stage2_validation": stage2_validation,
            "false_safe_improvement_vs_stage2_base": (stage2_false_safe - false_safe) if false_safe is not None and stage2_false_safe is not None else None,
            "severe_recall_improvement_vs_stage2_base": (severe - stage2_severe) if severe is not None and stage2_severe is not None else None,
            "false_safe_improvement_vs_stage2_best": (STAGE2_BEST_FALSE_SAFE - false_safe) if false_safe is not None else None,
            "severe_recall_improvement_vs_stage2_best": (severe - STAGE2_BEST_SEVERE_RECALL) if severe is not None else None,
        }
        record["failure_types"] = _failure_types(record)
        record["classification"] = _classify(record)
        records.append(record)
    _score_trials(records)
    return sorted(records, key=lambda item: item.get("composite_score") or 0.0, reverse=True)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _bind_stage3_paths()
    _set_env()
    _log(
        "sweep_start",
        {
            "trial_count": len(TRIALS),
            "epochs": args.epochs,
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "wandb_mode": os.environ.get("WANDB_MODE"),
        },
    )
    completed = []
    for trial in TRIALS:
        process = run_trial(
            str(trial["trial_id"]),
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            force=args.force,
        )
        completed.append(process)
    payload = write_report()
    _log("sweep_done", {"completed": len(completed), "report": str(REPORT_PATH)})
    return payload


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_없음_"
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _label, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = _fmt(value)
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _trial_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        val = record.get("validation") or {}
        rows.append(
            {
                "trial": record["trial_id"],
                "base": record["base"],
                "hp": f"d={record['dropout']}, wd={record['weight_decay']}, lr={record['lr']}",
                "line_gate": record.get("line_gate_pass"),
                "score": record.get("composite_score"),
                "ic": val.get("ic_mean"),
                "spread": val.get("long_short_spread"),
                "fee": val.get("fee_adjusted_return"),
                "false_safe": val.get("false_safe_tail_rate"),
                "fs_delta_best": record.get("false_safe_improvement_vs_stage2_best"),
                "severe": val.get("severe_downside_recall"),
                "sev_delta_best": record.get("severe_recall_improvement_vs_stage2_best"),
                "class": record.get("classification"),
                "failure": ", ".join(record.get("failure_types") or []),
            }
        )
    return rows


def _top_seed_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in records
        if row.get("classification") == "false-safe 개선 후보"
        and bool(row.get("line_gate_pass"))
        and (_safe_float((row.get("validation") or {}).get("ic_mean")) or -999.0) > 0
        and (_safe_float((row.get("validation") or {}).get("long_short_spread")) or -999.0) > 0
        and (_safe_float((row.get("validation") or {}).get("fee_adjusted_return")) or -999.0) > 0
    ]
    return sorted(eligible, key=lambda item: item.get("composite_score") or 0.0, reverse=True)[:2]


def _process_check() -> dict[str, Any]:
    current_pid = os.getpid()
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Where-Object { "
            "($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') "
            f"-and $_.ProcessId -ne {current_pid} "
            "-and ($_.CommandLine -notmatch 'cp148_lm_1d_stage3_false_safe_sweep\\s+report') "
            "} | "
            "Select-Object @{Name='Id';Expression={$_.ProcessId}},"
            "@{Name='ProcessName';Expression={$_.Name}},"
            "CreationDate,ExecutablePath,CommandLine | "
            "ConvertTo-Json -Depth 3"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {"status": "check_failed", "error": str(exc)}
    text = completed.stdout.strip()
    if not text:
        return {"status": "none_visible", "processes": []}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "raw", "stdout": text}
    processes = parsed if isinstance(parsed, list) else [parsed]
    return {"status": "visible", "processes": processes}


def _write_summary_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "trial_id",
        "base",
        "stage2_candidate",
        "feature_set",
        "patch_len",
        "patch_stride",
        "dropout",
        "weight_decay",
        "lr",
        "line_gate_pass",
        "classification",
        "composite_score",
        "false_safe_improvement_vs_stage2_best",
        "severe_recall_improvement_vs_stage2_best",
        *[f"val_{key}" for key in LINE_KEYS],
        "failure_types",
    ]
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            val = record.get("validation") or {}
            row = {
                "trial_id": record["trial_id"],
                "base": record["base"],
                "stage2_candidate": record["stage2_candidate"],
                "feature_set": record["feature_set"],
                "patch_len": record["patch_len"],
                "patch_stride": record["patch_stride"],
                "dropout": record["dropout"],
                "weight_decay": record["weight_decay"],
                "lr": record["lr"],
                "line_gate_pass": record.get("line_gate_pass"),
                "classification": record.get("classification"),
                "composite_score": record.get("composite_score"),
                "false_safe_improvement_vs_stage2_best": record.get("false_safe_improvement_vs_stage2_best"),
                "severe_recall_improvement_vs_stage2_best": record.get("severe_recall_improvement_vs_stage2_best"),
                "failure_types": ",".join(record.get("failure_types") or []),
            }
            for key in LINE_KEYS:
                row[f"val_{key}"] = val.get(key)
            writer.writerow(row)


def _write_report_markdown(payload: dict[str, Any]) -> None:
    records = payload.get("trials") or []
    rows = _trial_rows(records)
    top_seed = payload.get("top_seed_recheck_candidates") or []
    top_rows = _trial_rows(top_seed)
    failed_rows = [row for row in rows if row.get("class") not in {"false-safe 개선 후보", "부분 개선 후보"}]
    lines = [
        "# CP148-LM-1D Stage 3 false-safe narrow sweep 보고서",
        "",
        f"- 생성 시각: {payload.get('generated_at')}",
        "- 범위: EODHD 500 local parquet 기준 1D h5 PatchTST line_model Stage 3 narrow sweep",
        "- 금지 준수: save-run 없음, DB write 없음, inference 저장 없음, product promotion 없음, band/composite 없음, beta 변경 없음, live fetch 없음",
        "- beta/alpha/delta: 2.0 / 1.0 / 1.0 유지",
        "- 목표: Stage 2 best false_safe_tail_rate 0.308661보다 낮추고, severe_downside_recall 0.685019보다 높이는 후보 탐색",
        "",
        "## 1. Stage 2 대비 기준",
        "",
        "- Stage 2 false_safe best: `cp148_s2_patchtst_no_fund_p32_s16` = 0.308661",
        "- Stage 2 severe_recall best: `cp148_s2_patchtst_no_fund_p32_s16` = 0.685019",
        "- Stage 3 목표 관찰선: false_safe_tail_rate <= 0.27, severe_downside_recall >= 0.70",
        "- product target은 false_safe_tail_rate <= 0.20, severe_downside_recall >= 0.75로 유지하지만 이번 보고서는 제품 결론을 내리지 않는다.",
        "",
        "## 2. Trial 결과",
        "",
        _markdown_table(
            rows,
            [
                ("trial", "trial"),
                ("base", "base"),
                ("hp", "hp"),
                ("line_gate", "line_gate"),
                ("score", "score"),
                ("IC", "ic"),
                ("spread", "spread"),
                ("fee", "fee"),
                ("false_safe", "false_safe"),
                ("FS 개선", "fs_delta_best"),
                ("severe", "severe"),
                ("severe 개선", "sev_delta_best"),
                ("분류", "class"),
            ],
        ),
        "",
        "## 3. IC/spread/fee 희생 여부",
        "",
        "Stage 3 후보는 false-safe만 낮아져도 IC, spread, fee_adjusted_return 중 하나가 무너지면 seed 재평가 후보로 넘기지 않는다.",
        "",
        _markdown_table(
            [
                {
                    "trial": row.get("trial"),
                    "core_alive": (
                        (_safe_float(row.get("ic")) or -999) > 0
                        and (_safe_float(row.get("spread")) or -999) > 0
                        and (_safe_float(row.get("fee")) or -999) > 0
                    ),
                    "failure": row.get("failure"),
                }
                for row in rows
            ],
            [("trial", "trial"), ("IC/spread/fee 양수", "core_alive"), ("failure", "failure")],
        ),
        "",
        "## 4. Top 2 seed 재평가 후보",
        "",
        _markdown_table(
            top_rows,
            [
                ("trial", "trial"),
                ("base", "base"),
                ("hp", "hp"),
                ("score", "score"),
                ("false_safe", "false_safe"),
                ("FS 개선", "fs_delta_best"),
                ("severe", "severe"),
                ("severe 개선", "sev_delta_best"),
                ("분류", "class"),
            ],
        ),
        "",
        "## 5. 탈락 후보와 이유",
        "",
        _markdown_table(
            failed_rows,
            [("trial", "trial"), ("base", "base"), ("분류", "class"), ("failure", "failure")],
        ),
        "",
        "## 6. 다음 단계 추천",
        "",
    ]
    if top_seed:
        lines.extend(
            [
                "- 위 Top 2를 seed 3개 재평가 후보로 넘긴다.",
                "- seed 3개 median에서도 false_safe/severe 개선과 IC/spread/fee 양수가 유지될 때 Stage 4 후보로 남긴다.",
            ]
        )
    else:
        lines.extend(
            [
                "- seed 재평가 후보 없음.",
                "- 이번 축의 dropout/weight_decay/lr 조정만으로는 false-safe와 severe recall을 동시에 개선하지 못한 것으로 기록한다.",
                "- 다음 LM은 beta 변경 없이 false-safe-aware selector, downside sample weighting, feature 추가 축소를 별도 CP로 검토한다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 7. 잔여 python/pythonw 프로세스 확인",
            "",
            "```json",
            json.dumps(_json_safe(payload.get("process_check")), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 8. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary csv: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report() -> dict[str, Any]:
    records = collect_results()
    top_seed = _top_seed_candidates(records)
    payload = {
        "cp": "CP148-LM-1D-Stage3",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if records else "IN_PROGRESS",
        "scope_compliance": {
            "line_model_only": True,
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "product_promotion": False,
            "band_or_composite_experiment": False,
            "beta_changed": False,
            "live_fetch": False,
        },
        "stage2_thresholds": {
            "best_false_safe_tail_rate": STAGE2_BEST_FALSE_SAFE,
            "best_severe_downside_recall": STAGE2_BEST_SEVERE_RECALL,
            "false_safe_stage3_target": FALSE_SAFE_TARGET,
            "product_false_safe_target_reference_only": PRODUCT_FALSE_SAFE_TARGET,
            "product_severe_recall_target_reference_only": PRODUCT_SEVERE_TARGET,
        },
        "trials": records,
        "top_seed_recheck_candidates": top_seed,
        "process_check": _process_check(),
    }
    _write_json(METRICS_PATH, payload)
    _write_summary_csv(records)
    _write_report_markdown(payload)
    _log("report_done", {"report": str(REPORT_PATH), "metrics": str(METRICS_PATH)})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="CP148-LM-1D Stage 3 false-safe narrow sweep")
    sub = parser.add_subparsers(dest="command", required=True)
    run_one = sub.add_parser("run-trial")
    run_one.add_argument("--trial", required=True)
    run_one.add_argument("--epochs", type=int, default=5)
    run_one.add_argument("--seed", type=int, default=42)
    run_one.add_argument("--device", default="cuda")
    run_one.add_argument("--batch-size", type=int, default=256)
    run_one.add_argument("--force", action="store_true")

    run_all = sub.add_parser("run-sweep")
    run_all.add_argument("--epochs", type=int, default=5)
    run_all.add_argument("--seed", type=int, default=42)
    run_all.add_argument("--device", default="cuda")
    run_all.add_argument("--batch-size", type=int, default=256)
    run_all.add_argument("--force", action="store_true")

    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "run-trial":
        run_trial(args.trial, epochs=args.epochs, seed=args.seed, device=args.device, batch_size=args.batch_size, force=args.force)
    elif args.command == "run-sweep":
        run_sweep(args)
    elif args.command == "report":
        payload = write_report()
        print(json.dumps({"status": payload["status"], "report": str(REPORT_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
