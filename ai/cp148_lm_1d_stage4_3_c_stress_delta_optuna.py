from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ai.train as train_mod  # noqa: E402
from ai import cp148_lm_1d_stage4_1_risk_feature_abcd as s41  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_3_c_stress_delta_optuna_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_3_c_stress_delta_optuna_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_3_c_stress_delta_optuna_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp148_lm_1d_stage4_3_c_stress_delta_optuna_logs"
STUDY_DB_PATH = LOG_DIR / "cp148_stage4_3_c_stress_delta_optuna.db"
STUDY_EXPORT_PATH = LOG_DIR / "cp148_stage4_3_c_stress_delta_optuna_study_export.json"

C_FEATURES = ["atr_ratio", "vix_change_5d", "credit_spread_change_20d", "ma200_pct_change_20d"]
PATCH_GEOMETRY = {
    "p32_s16": (32, 16),
    "p24_s12": (24, 12),
    "p16_s8": (16, 8),
}
BASELINE_C = {
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
}
BASELINE_H1_FALSE_SAFE = 0.272696
BASELINE_STRESS_FALSE_SAFE = 0.197640

SUMMARY_KEYS = [
    "ic_mean",
    "long_short_spread",
    "fee_adjusted_return",
    "false_safe_tail_rate",
    "severe_downside_recall",
    "conservative_bias",
    "upside_sacrifice",
    "direction_accuracy",
]


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


def _clip(value: float, lower: float = 0.0, upper: float = 2.0) -> float:
    if not math.isfinite(value):
        return lower
    return min(max(value, lower), upper)


def _trial_params(trial: optuna.Trial) -> dict[str, Any]:
    patch_geometry = trial.suggest_categorical("patch_geometry", list(PATCH_GEOMETRY.keys()))
    patch_len, patch_stride = PATCH_GEOMETRY[patch_geometry]
    return {
        "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 8e-4, log=True),
        "dropout": trial.suggest_categorical("dropout", [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]),
        "patch_geometry": patch_geometry,
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "lambda_direction": trial.suggest_categorical("lambda_direction", [0.05, 0.10, 0.15, 0.20, 0.30]),
    }


def _experiment_for_trial(trial_number: int, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": f"stage4_3_c_stress_delta_trial{trial_number:03d}_{params['patch_geometry']}",
        "label": "C_stress_delta_optuna",
        "question": "C_stress_delta 하방 안정성을 유지하면서 ranking/spread/fee를 회복할 수 있는가?",
        "extra_features": C_FEATURES,
    }


def _make_config(
    experiment: dict[str, Any],
    *,
    params: dict[str, Any],
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
    config.patch_len = int(params["patch_len"])
    config.patch_stride = int(params["patch_stride"])
    config.lr = float(params["lr"])
    config.weight_decay = float(params["weight_decay"])
    config.dropout = float(params["dropout"])
    config.lambda_direction = float(params["lambda_direction"])
    config.lr_schedule = "cosine"
    config.warmup_frac = 0.05
    config.grad_clip = 1.0
    config.alpha = 1.0
    config.beta = 2.0
    config.delta = 1.0
    config.lambda_line = 1.0
    config.lambda_band = 2.0
    config.lambda_width = 0.1
    config.lambda_cross = 1.0
    config.band_mode = "direct"
    config.checkpoint_selection = "line_gate"
    config.feature_set = "C_stress_delta"
    config.use_wandb = False
    return config


def _line_metrics(source: dict[str, Any]) -> dict[str, Any]:
    metrics = s41._line_metrics(source)
    return {key: metrics.get(key) for key in SUMMARY_KEYS if metrics.get(key) is not None}


def _validation_score(record: dict[str, Any]) -> tuple[float, list[str]]:
    metrics = record.get("validation") or {}
    regimes = record.get("validation_regime_metrics") or {}
    buckets = record.get("validation_horizon_bucket_metrics") or {}
    warnings: list[str] = []

    line_gate = bool(record.get("line_gate_pass"))
    ic = _safe_float(metrics.get("ic_mean")) or float("-inf")
    spread = _safe_float(metrics.get("long_short_spread")) or float("-inf")
    fee = _safe_float(metrics.get("fee_adjusted_return")) or float("-inf")
    false_safe = _safe_float(metrics.get("false_safe_tail_rate")) or float("inf")
    severe = _safe_float(metrics.get("severe_downside_recall")) or float("-inf")
    h1_false_safe = _safe_float((buckets.get("h1") or {}).get("false_safe_tail_rate"))
    stress_false_safe = _safe_float((regimes.get("stress") or {}).get("false_safe_tail_rate"))
    breadth_false_safe = _safe_float((regimes.get("breadth_worsening") or {}).get("false_safe_tail_rate"))

    if not line_gate:
        return -100.0, ["line_gate_fail"]
    hard_penalty = 0.0
    if ic <= 0.0:
        hard_penalty -= 20.0
        warnings.append("ic_non_positive")
    if spread <= 0.0:
        hard_penalty -= 20.0
        warnings.append("spread_non_positive")
    if fee <= 0.0:
        hard_penalty -= 30.0
        warnings.append("fee_non_positive")
    if false_safe > 0.32:
        hard_penalty -= 10.0
        warnings.append("false_safe_guard")
    if severe < 0.67:
        hard_penalty -= 10.0
        warnings.append("severe_guard")
    if spread < 0.005:
        hard_penalty -= 8.0
        warnings.append("spread_guard")
    if h1_false_safe is not None and h1_false_safe > BASELINE_H1_FALSE_SAFE + 0.03:
        warnings.append("h1_false_safe_worse")
    if stress_false_safe is not None and stress_false_safe > BASELINE_STRESS_FALSE_SAFE + 0.03:
        warnings.append("stress_false_safe_worse")
    if breadth_false_safe is not None and breadth_false_safe > BASELINE_C["validation"]["false_safe_tail_rate"] + 0.03:
        warnings.append("breadth_false_safe_worse")

    normalized_spread = _clip(spread / 0.008)
    normalized_fee = _clip(fee / 20.0)
    normalized_ic = _clip(ic / 0.05)
    normalized_severe = _clip((severe - 0.67) / 0.06)
    normalized_false_safe_inverse = _clip((0.32 - false_safe) / 0.06)
    score = (
        0.30 * normalized_spread
        + 0.20 * normalized_fee
        + 0.15 * normalized_ic
        + 0.20 * normalized_severe
        + 0.15 * normalized_false_safe_inverse
        + hard_penalty
    )
    return score, warnings


def _classify_trial(record: dict[str, Any]) -> str:
    metrics = record.get("validation") or {}
    ic = _safe_float(metrics.get("ic_mean")) or float("-inf")
    spread = _safe_float(metrics.get("long_short_spread")) or float("-inf")
    fee = _safe_float(metrics.get("fee_adjusted_return")) or float("-inf")
    false_safe = _safe_float(metrics.get("false_safe_tail_rate")) or float("inf")
    severe = _safe_float(metrics.get("severe_downside_recall")) or float("-inf")
    if not record.get("line_gate_pass") or fee <= 0.0:
        return "탈락 후보"
    baseline = BASELINE_C["validation"]
    risk_ok = false_safe <= baseline["false_safe_tail_rate"] and severe >= baseline["severe_downside_recall"]
    alpha_ok = (
        (spread >= 0.008 or spread > baseline["long_short_spread"] * 1.05)
        and fee > baseline["fee_adjusted_return"]
        and ic > baseline["ic_mean"]
    )
    risk_not_broken = false_safe <= baseline["false_safe_tail_rate"] + 0.025 and severe >= baseline["severe_downside_recall"] - 0.025
    if risk_ok and alpha_ok:
        return "C-balanced 후보"
    if risk_ok and spread >= baseline["long_short_spread"] * 0.8 and fee > 0.0:
        return "C-risk 후보"
    if alpha_ok and risk_not_broken:
        return "C-alpha 후보"
    return "탈락 후보"


def _run_trial(
    *,
    trial_number: int,
    params: dict[str, Any],
    epochs: int,
    seed: int,
    batch_size: int,
    device: str,
    amp_dtype: str,
    force: bool,
) -> dict[str, Any]:
    experiment = _experiment_for_trial(trial_number, params)
    trial_dir = LOG_DIR / experiment["experiment_id"]
    metrics_path = trial_dir / "run_meta.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    precomputed, feature_names = s41._build_exact_feature_splits(
        extra_names=C_FEATURES,
        device=device,
        batch_size=batch_size,
    )
    train_bundle, val_bundle, test_bundle, _mean, _std, plan = precomputed
    config = _make_config(
        experiment,
        params=params,
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
        local_log_dir=trial_dir / "train_local_logs",
        wandb_group="cp148_stage4_3",
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
        "trial_number": trial_number,
        "candidate_family": "C_stress_delta",
        "params": params,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "amp_dtype": amp_dtype,
        "feature_set": "C_stress_delta",
        "extra_features": C_FEATURES,
        "checkpoint_selection": "line_gate",
        "selector_policy": "risk_aware_line_gate_sort",
        "line_gate_meaning_changed": False,
        "run_id": result.get("run_id"),
        "checkpoint_path": result.get("checkpoint_path"),
        "line_gate_pass": bool((result.get("best_metrics") or {}).get("line_gate_pass")),
        "validation": _line_metrics(result.get("best_metrics") or {}),
        "test": _line_metrics(result.get("test_metrics") or {}),
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
            "test_used_in_objective": False,
        },
    }
    score, warnings = _validation_score(record)
    record["objective_score"] = score
    record["warnings"] = warnings
    record["classification"] = _classify_trial(record)
    _write_json(metrics_path, record)
    return record


def _trial_to_summary(record: dict[str, Any], *, row_type: str = "trial") -> dict[str, Any]:
    validation = record.get("validation") or {}
    test = record.get("test") or {}
    val_regimes = record.get("validation_regime_metrics") or {}
    val_buckets = record.get("validation_horizon_bucket_metrics") or {}
    params = record.get("params") or {}
    return {
        "row_type": row_type,
        "trial_number": record.get("trial_number"),
        "classification": record.get("classification"),
        "objective_score": record.get("objective_score"),
        "seed": record.get("seed"),
        "patch_geometry": params.get("patch_geometry"),
        "lr": params.get("lr"),
        "weight_decay": params.get("weight_decay"),
        "dropout": params.get("dropout"),
        "lambda_direction": params.get("lambda_direction"),
        "line_gate_pass": record.get("line_gate_pass"),
        "val_ic": validation.get("ic_mean"),
        "val_spread": validation.get("long_short_spread"),
        "val_fee": validation.get("fee_adjusted_return"),
        "val_false_safe": validation.get("false_safe_tail_rate"),
        "val_severe": validation.get("severe_downside_recall"),
        "val_h1_false_safe": (val_buckets.get("h1") or {}).get("false_safe_tail_rate"),
        "val_stress_false_safe": (val_regimes.get("stress") or {}).get("false_safe_tail_rate"),
        "val_breadth_false_safe": (val_regimes.get("breadth_worsening") or {}).get("false_safe_tail_rate"),
        "test_ic": test.get("ic_mean"),
        "test_spread": test.get("long_short_spread"),
        "test_fee": test.get("fee_adjusted_return"),
        "test_false_safe": test.get("false_safe_tail_rate"),
        "test_severe": test.get("severe_downside_recall"),
        "warnings": ",".join(record.get("warnings") or []),
        "checkpoint_path": record.get("checkpoint_path"),
    }


def _reference_row() -> dict[str, Any]:
    return {
        "row_type": "reference",
        "trial_number": "C_stage4_2_median",
        "classification": "기존 C reference",
        "objective_score": "",
        "seed": "42,7,123 median",
        "patch_geometry": "p32_s16",
        "lr": "stage4_2_reference",
        "weight_decay": "stage4_2_reference",
        "dropout": "stage4_2_reference",
        "lambda_direction": "stage4_2_reference",
        "line_gate_pass": True,
        "val_ic": BASELINE_C["validation"]["ic_mean"],
        "val_spread": BASELINE_C["validation"]["long_short_spread"],
        "val_fee": BASELINE_C["validation"]["fee_adjusted_return"],
        "val_false_safe": BASELINE_C["validation"]["false_safe_tail_rate"],
        "val_severe": BASELINE_C["validation"]["severe_downside_recall"],
        "val_h1_false_safe": BASELINE_H1_FALSE_SAFE,
        "val_stress_false_safe": BASELINE_STRESS_FALSE_SAFE,
        "val_breadth_false_safe": "",
        "test_ic": BASELINE_C["test"]["ic_mean"],
        "test_spread": BASELINE_C["test"]["long_short_spread"],
        "test_fee": BASELINE_C["test"]["fee_adjusted_return"],
        "test_false_safe": BASELINE_C["test"]["false_safe_tail_rate"],
        "test_severe": BASELINE_C["test"]["severe_downside_recall"],
        "warnings": "",
        "checkpoint_path": "stage4_2_reference_median",
    }


def _write_summary(records: list[dict[str, Any]]) -> None:
    rows = [_reference_row()] + [_trial_to_summary(record) for record in sorted(records, key=lambda item: item.get("objective_score", -999), reverse=True)]
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _select_stage4_4(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in ["C-balanced 후보", "C-risk 후보", "C-alpha 후보"]:
        for record in sorted(records, key=lambda item: item.get("objective_score", -999), reverse=True):
            if record.get("classification") == label and record not in selected:
                selected.append(record)
                break
            if len(selected) >= 2:
                break
        if len(selected) >= 2:
            break
    return selected[:2]


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _write_report(records: list[dict[str, Any]], selected: list[dict[str, Any]], n_trials: int) -> None:
    top = sorted(records, key=lambda item: item.get("objective_score", -999), reverse=True)[:5]
    lines = [
        "# CP148-LM-1D Stage 4-3 C_stress_delta Optuna narrow sweep 보고서",
        "",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        f"- trial 수: {len(records)} / 요청 {n_trials}",
        "- seed: sweep 단계 고정 seed 123",
        "- 기준 reference: Stage 4-2 C_stress_delta seed 3개 median",
        "- 금지 준수: product save 없음, DB write 없음, inference 저장 없음, live fetch 없음, band/composite 없음, beta=2 유지",
        "- objective는 validation만 사용했고 test는 기록/확인용으로만 남겼다.",
        "",
        "## 1. 기존 C reference",
        "",
    ]
    lines.extend(
        _table(
            ["split", "IC", "spread", "fee", "false_safe", "severe"],
            [
                [
                    "validation",
                    BASELINE_C["validation"]["ic_mean"],
                    BASELINE_C["validation"]["long_short_spread"],
                    BASELINE_C["validation"]["fee_adjusted_return"],
                    BASELINE_C["validation"]["false_safe_tail_rate"],
                    BASELINE_C["validation"]["severe_downside_recall"],
                ],
                [
                    "test",
                    BASELINE_C["test"]["ic_mean"],
                    BASELINE_C["test"]["long_short_spread"],
                    BASELINE_C["test"]["fee_adjusted_return"],
                    BASELINE_C["test"]["false_safe_tail_rate"],
                    BASELINE_C["test"]["severe_downside_recall"],
                ],
            ],
        )
    )
    lines.extend(["", "## 2. Top 5 trial", ""])
    lines.extend(
        _table(
            ["trial", "분류", "score", "geom", "lr", "wd", "dropout", "dir", "val_spread", "val_fee", "val_FS", "val_severe", "test_FS", "test_severe"],
            [
                [
                    record.get("trial_number"),
                    record.get("classification"),
                    _fmt(record.get("objective_score")),
                    (record.get("params") or {}).get("patch_geometry"),
                    _fmt((record.get("params") or {}).get("lr"), 8),
                    _fmt((record.get("params") or {}).get("weight_decay"), 8),
                    (record.get("params") or {}).get("dropout"),
                    (record.get("params") or {}).get("lambda_direction"),
                    _fmt((record.get("validation") or {}).get("long_short_spread")),
                    _fmt((record.get("validation") or {}).get("fee_adjusted_return")),
                    _fmt((record.get("validation") or {}).get("false_safe_tail_rate")),
                    _fmt((record.get("validation") or {}).get("severe_downside_recall")),
                    _fmt((record.get("test") or {}).get("false_safe_tail_rate")),
                    _fmt((record.get("test") or {}).get("severe_downside_recall")),
                ]
                for record in top
            ],
        )
    )
    lines.extend(["", "## 3. Stage 4-4 seed 재평가 후보", ""])
    if selected:
        for record in selected:
            lines.append(
                f"- trial {record.get('trial_number')}: `{record.get('classification')}`, params={record.get('params')}"
            )
    else:
        lines.append("- tuning 실패: 기존 C reference를 넘겨 Stage 4-4로 보낼 후보가 없다.")
    lines.extend(
        [
            "",
            "## 4. 산출물",
            "",
            f"- metrics: `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- summary: `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- study DB: `{STUDY_DB_PATH.relative_to(PROJECT_ROOT)}`",
            f"- study export: `{STUDY_EXPORT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
            "- script: `ai/cp148_lm_1d_stage4_3_c_stress_delta_optuna.py`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_study(study: optuna.Study) -> None:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    _write_json(STUDY_EXPORT_PATH, {"study_name": study.study_name, "trials": rows})


def run(args: argparse.Namespace) -> dict[str, Any]:
    alias = s41._configure_environment()
    s41.LOG_DIR = LOG_DIR
    s41._patch_training_for_experiment()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{STUDY_DB_PATH.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    study = optuna.create_study(
        study_name="cp148_stage4_3_c_stress_delta",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    records_by_trial: dict[int, dict[str, Any]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = _trial_params(trial)
        record = _run_trial(
            trial_number=trial.number,
            params=params,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            amp_dtype=args.amp_dtype,
            force=args.force,
        )
        records_by_trial[trial.number] = record
        trial.set_user_attr("classification", record.get("classification"))
        trial.set_user_attr("checkpoint_path", record.get("checkpoint_path"))
        trial.set_user_attr("validation", record.get("validation"))
        trial.set_user_attr("test", record.get("test"))
        trial.set_user_attr("warnings", record.get("warnings"))
        return float(record["objective_score"])

    if args.n_trials > 0:
        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    records = []
    for path in sorted(LOG_DIR.glob("stage4_3_c_stress_delta_trial*/run_meta.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    records = sorted(records, key=lambda item: item.get("objective_score", -999), reverse=True)
    selected = _select_stage4_4(records)
    payload = {
        "cp": "CP148-LM-1D-Stage4-3",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_alias_dir": str(alias),
        "study_name": study.study_name,
        "study_db_path": str(STUDY_DB_PATH),
        "baseline_reference": BASELINE_C,
        "objective_policy": {
            "test_used_in_objective": False,
            "line_gate_fail_penalty": True,
            "hard_guards": {
                "validation_false_safe_tail_rate_max": 0.32,
                "validation_severe_downside_recall_min": 0.67,
                "validation_spread_min": 0.005,
                "validation_fee_positive": True,
            },
        },
        "scope_compliance": {
            "product_save": False,
            "db_write": False,
            "inference_save": False,
            "live_fetch": False,
            "band_or_composite_experiment": False,
            "beta": 2.0,
            "line_gate_meaning_changed": False,
            "tail_loss_added": False,
            "test_used_in_objective": False,
        },
        "records": records,
        "top5": records[:5],
        "stage4_4_candidates": selected,
        "process_check": {
            "status": "deferred_external_check",
            "note": "최종 외부 검증에서 python/pythonw 및 CUDA Python 학습 프로세스를 확인한다.",
        },
    }
    _write_json(METRICS_PATH, payload)
    _write_summary(records)
    _export_study(study)
    _write_report(records, selected, args.n_trials)
    print(json.dumps({"status": "PASS", "metrics_path": str(METRICS_PATH), "report_path": str(REPORT_PATH)}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148 Stage 4-3 C_stress_delta Optuna narrow sweep")
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sampler-seed", type=int, default=14843)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
