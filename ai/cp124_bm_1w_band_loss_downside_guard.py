from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.cp121_bm_1w_band_verified_stability import (  # noqa: E402
    BAND_METRIC_KEYS,
    REGIME_NAMES,
    STABILITY_KEYS,
    _aggregate_regime_metrics,
    _band_metrics,
    _epoch_summary,
    _extract_train_json,
    _fmt,
    _json_safe,
    _mean_std,
    _safe_float,
    _table,
    _validation_regime_metrics,
)


REPORT_PATH = PROJECT_ROOT / "docs" / "cp124_bm_1w_band_loss_downside_guard_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp124_bm_1w_band_loss_downside_guard_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp124_bm_1w_band_loss_downside_guard_registry.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp124_bm_1w_band_loss_downside_guard_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp124_bm_1w_band_loss_downside_guard_logs"
CP121_METRICS_PATH = PROJECT_ROOT / "docs" / "cp121_bm_1w_band_verified_stability_metrics.json"


@dataclass(frozen=True)
class Experiment:
    name: str
    base_candidate: str
    model: str
    feature_set: str
    q_low: float
    q_high: float
    band_mode: str
    lambda_band: float
    lower_band_loss_weight: float
    upper_band_loss_weight: float
    requested_target: str
    implementation_status: str
    note: str
    reuse_cp121_baseline: bool = False
    runnable: bool = True


EXPERIMENTS = [
    Experiment(
        "tide_pvv_q15_param_baseline",
        "tide_pvv_q15_param",
        "tide",
        "price_volatility_volume",
        0.15,
        0.85,
        "param",
        2.0,
        1.0,
        1.0,
        "baseline",
        "reused_cp121",
        "CP121 recommended_default 기준선을 validation 중심으로 재사용",
        reuse_cp121_baseline=True,
    ),
    Experiment(
        "tide_pvv_q15_param_lower_guard_w1p5",
        "tide_pvv_q15_param",
        "tide",
        "price_volatility_volume",
        0.15,
        0.85,
        "param",
        2.0,
        1.5,
        1.0,
        "lower_breach_penalty",
        "implemented_lower_quantile_weight",
        "lower quantile pinball loss만 1.5배로 강화",
    ),
    Experiment(
        "tide_pvv_q15_param_asym_guard_w2p0",
        "tide_pvv_q15_param",
        "tide",
        "price_volatility_volume",
        0.15,
        0.85,
        "param",
        2.0,
        2.0,
        1.0,
        "asymmetric_interval",
        "implemented_lower_quantile_weight",
        "asymmetric interval score의 하방 가중 원칙에 맞춰 lower quantile loss를 2.0배로 강화",
    ),
    Experiment(
        "tide_pvv_q15_param_width_alignment",
        "tide_pvv_q15_param",
        "tide",
        "price_volatility_volume",
        0.15,
        0.85,
        "param",
        2.0,
        1.0,
        1.0,
        "width_alignment",
        "design_needed",
        "현재 lambda_width는 레거시 호환용이며 손실 계산에 사용되지 않아 이번 CP에서는 실행하지 않음",
        runnable=False,
    ),
    Experiment(
        "cnn_full_q10_direct_baseline",
        "cnn_full_q10_direct",
        "cnn_lstm",
        "full_features",
        0.10,
        0.90,
        "direct",
        2.0,
        1.0,
        1.0,
        "baseline",
        "reused_cp121",
        "CP121 selectable_verified 기준선을 validation 중심으로 재사용",
        reuse_cp121_baseline=True,
    ),
    Experiment(
        "cnn_full_q10_direct_lower_guard_w1p5",
        "cnn_full_q10_direct",
        "cnn_lstm",
        "full_features",
        0.10,
        0.90,
        "direct",
        2.0,
        1.5,
        1.0,
        "lower_breach_penalty",
        "implemented_lower_quantile_weight",
        "CNN full_features direct 후보의 lower quantile pinball loss를 1.5배로 강화",
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _completed(run: dict[str, Any]) -> bool:
    return run.get("execution_status") in {"PASS", "REUSED_CP121"}


def _command_for_run(experiment: Experiment, seed: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        experiment.model,
        "--timeframe",
        "1W",
        "--horizon",
        "4",
        "--seq-len",
        "104",
        "--feature-set",
        experiment.feature_set,
        "--line-target-type",
        "raw_future_return",
        "--band-target-type",
        "raw_future_return",
        "--q-low",
        str(experiment.q_low),
        "--q-high",
        str(experiment.q_high),
        "--lambda-band",
        str(experiment.lambda_band),
        "--lower-band-loss-weight",
        str(experiment.lower_band_loss_weight),
        "--upper-band-loss-weight",
        str(experiment.upper_band_loss_weight),
        "--band-mode",
        experiment.band_mode,
        "--checkpoint-selection",
        "band_gate",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--amp-dtype",
        args.amp_dtype,
        "--no-compile",
        "--no-wandb",
        "--num-workers",
        "0",
        "--explicit-cuda-cleanup",
        "--market-data-provider",
        "yfinance",
        "--seed",
        str(seed),
        "--local-log-dir",
        str(LOG_DIR / "local_runs" / experiment.name / f"seed_{seed}"),
    ]
    if experiment.model == "cnn_lstm":
        command.extend(["--fp32-modules", "lstm,heads"])
    if experiment.model == "tide":
        command.append("--use-future-covariate")
    return command


def _run_training(experiment: Experiment, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{experiment.name}_seed{seed}.stdout.log"
    command = _command_for_run(experiment, seed, args)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCHDYNAMO_DISABLE": "1",
            "WANDB_MODE": "disabled",
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(PROJECT_ROOT / "data" / "parquet"),
        }
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        execution_status = "PASS" if exit_code == 0 else "FAIL"
        timeout_expired = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        exit_code = None
        execution_status = "TIMEOUT"
        timeout_expired = True
    elapsed = time.perf_counter() - started
    ended_at = datetime.now().isoformat(timespec="seconds")
    log_path.write_text(
        "\n".join(
            [
                "# command: " + " ".join(command),
                "# environment: yfinance local snapshots, WANDB_MODE=disabled, save-run=false",
                "# stdout",
                stdout,
                "# stderr",
                stderr,
                f"# exit_code: {exit_code}",
                f"# elapsed_seconds: {elapsed:.4f}",
                f"# timeout_expired: {timeout_expired}",
            ]
        ),
        encoding="utf-8",
    )
    record: dict[str, Any] = {
        "experiment": experiment.name,
        "candidate": experiment.name,
        "base_candidate": experiment.base_candidate,
        "seed": seed,
        "candidate_config": asdict(experiment),
        "command": command,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "execution_status": execution_status,
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "timeout_expired": timeout_expired,
        "epoch_summary": _epoch_summary(stdout),
    }
    if execution_status != "PASS":
        record["failed_with_reason"] = "timeout" if timeout_expired else "nonzero_exit"
        record["stderr_tail"] = stderr[-4000:]
        return record
    try:
        result = _extract_train_json(stdout)
    except Exception as exc:
        record["execution_status"] = "FAIL"
        record["failed_with_reason"] = f"result_json_parse_failed: {exc}"
        record["stderr_tail"] = stderr[-4000:]
        return record
    best_metrics = result.get("best_metrics") if isinstance(result.get("best_metrics"), dict) else {}
    test_metrics = result.get("test_metrics") if isinstance(result.get("test_metrics"), dict) else {}
    record.update(
        {
            "run_id": result.get("run_id"),
            "checkpoint_path": result.get("checkpoint_path"),
            "source_data_hash": result.get("source_data_hash"),
            "local_log_dir": result.get("local_log_dir"),
            "feature_version": result.get("feature_version"),
            "n_features": result.get("n_features"),
            "feature_columns": result.get("feature_columns"),
            "validation_band_metrics": _band_metrics(best_metrics),
            "test_band_metrics_read_only": _band_metrics(test_metrics),
            "test_metrics_present": bool(test_metrics),
            "gate": {
                "checkpoint_selection": best_metrics.get("checkpoint_selection"),
                "gate_type": best_metrics.get("gate_type"),
                "gate_failed": best_metrics.get("gate_failed"),
                "band_gate_pass": best_metrics.get("band_gate_pass"),
                "role": best_metrics.get("role"),
                "best_epoch": best_metrics.get("best_epoch"),
                "selected_reason": best_metrics.get("selected_reason"),
            },
            "wandb_status": result.get("wandb_status"),
            "dataset_plan": result.get("dataset_plan"),
        }
    )
    return record


def _cp121_baseline_runs(experiment: Experiment, args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = _read_json(CP121_METRICS_PATH)
    for summary in payload.get("candidate_summaries", []):
        if summary.get("candidate") != experiment.base_candidate:
            continue
        runs = []
        for run in summary.get("runs", []):
            copied = dict(run)
            copied["experiment"] = experiment.name
            copied["candidate"] = experiment.name
            copied["base_candidate"] = experiment.base_candidate
            copied["candidate_config"] = asdict(experiment)
            copied["execution_status"] = "REUSED_CP121" if run.get("execution_status") == "PASS" else run.get("execution_status")
            copied["source_experiment_id"] = experiment.base_candidate
            copied["baseline_reuse_policy"] = "checkpoint과 train validation metric은 재사용하고, regime metric은 현재 yfinance local snapshot으로 재평가"
            if copied.get("execution_status") == "REUSED_CP121":
                try:
                    copied["validation_regime_metrics"] = _validation_regime_metrics(copied, args)
                except Exception as exc:
                    copied["validation_regime_metrics"] = {"error": str(exc)}
            runs.append(copied)
        return runs
    return []


def _existing_runs() -> dict[tuple[str, int], dict[str, Any]]:
    payload = _read_json(METRICS_PATH)
    result = {}
    for summary in payload.get("experiment_summaries", []):
        for run in summary.get("runs", []):
            if run.get("experiment") and run.get("seed") is not None and _completed(run):
                result[(str(run.get("experiment")), int(run.get("seed")))] = run
    return result


def _aggregate_experiment(experiment: Experiment, runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if _completed(run)]
    metric_stats: dict[str, dict[str, float | None]] = {}
    for key in STABILITY_KEYS:
        values = [
            _safe_float((run.get("validation_band_metrics") or {}).get(key))
            for run in completed
        ]
        metric_stats[key] = _mean_std([value for value in values if value is not None])
    gate_pass_count = sum(1 for run in completed if (run.get("gate") or {}).get("band_gate_pass") and not (run.get("gate") or {}).get("gate_failed"))
    test_exposure_count = sum(1 for run in completed if run.get("test_metrics_present"))
    regime_stats = _aggregate_regime_metrics(completed)
    return {
        "experiment": experiment.name,
        "base_candidate": experiment.base_candidate,
        "candidate_config": asdict(experiment),
        "runs": runs,
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "band_gate_pass_count": gate_pass_count,
        "band_gate_pass_rate": gate_pass_count / len(completed) if completed else 0.0,
        "validation_metric_stats": metric_stats,
        "validation_regime_metric_stats": regime_stats,
        "validation_regime_warnings": _regime_warnings(regime_stats),
        "test_exposure_count": test_exposure_count,
        "test_metric_policy": "ai.train 결과 JSON의 test_metrics는 read-only count만 기록하고 후보 선정에는 사용하지 않음",
    }


def _design_needed_summary(experiment: Experiment) -> dict[str, Any]:
    return {
        "experiment": experiment.name,
        "base_candidate": experiment.base_candidate,
        "candidate_config": asdict(experiment),
        "runs": [
            {
                "experiment": experiment.name,
                "candidate": experiment.name,
                "base_candidate": experiment.base_candidate,
                "execution_status": "DESIGN_NEEDED",
                "failed_with_reason": experiment.note,
                "seed": None,
                "candidate_config": asdict(experiment),
            }
        ],
        "run_count": 0,
        "completed_run_count": 0,
        "band_gate_pass_count": 0,
        "band_gate_pass_rate": 0.0,
        "validation_metric_stats": {key: _mean_std([]) for key in STABILITY_KEYS},
        "validation_regime_metric_stats": {},
        "validation_regime_warnings": [experiment.note],
        "test_exposure_count": 0,
        "test_metric_policy": "실행하지 않음",
        "decision": {
            "category": "design_needed",
            "checks": {},
            "failures": ["width_alignment_loss_not_implemented"],
        },
    }


def _regime_warnings(regime_stats: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    falling_lower = _safe_float((((regime_stats.get("falling") or {}).get("lower_breach_rate") or {}).get("mean")))
    if falling_lower is not None and falling_lower > 0.20:
        warnings.append(f"falling lower_breach_rate={falling_lower:.6f}")
    narrow_cov = _safe_float((((regime_stats.get("narrow_band") or {}).get("coverage_abs_error") or {}).get("mean")))
    if narrow_cov is not None and narrow_cov > 0.10:
        warnings.append(f"narrow_band coverage_abs_error={narrow_cov:.6f}")
    low_bwic = None
    for regime_name, stats in regime_stats.items():
        value = _safe_float(((stats.get("band_width_ic") or {}).get("mean")))
        if value is not None and (low_bwic is None or value < low_bwic[1]):
            low_bwic = (regime_name, value)
    if low_bwic is not None and low_bwic[1] < 0.10:
        warnings.append(f"{low_bwic[0]} band_width_ic={low_bwic[1]:.6f}")
    return warnings


def _baseline_map(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for summary in summaries:
        config = summary.get("candidate_config") or {}
        if config.get("requested_target") == "baseline":
            result[str(summary.get("base_candidate"))] = summary
    return result


def _metric_mean(summary: dict[str, Any], key: str) -> float | None:
    return _safe_float((((summary.get("validation_metric_stats") or {}).get(key) or {}).get("mean")))


def _regime_metric_mean(summary: dict[str, Any], regime: str, key: str) -> float | None:
    return _safe_float(((((summary.get("validation_regime_metric_stats") or {}).get(regime) or {}).get(key) or {}).get("mean")))


def _classify_experiments(summaries: list[dict[str, Any]]) -> None:
    baselines = _baseline_map(summaries)
    for summary in summaries:
        if (summary.get("decision") or {}).get("category") == "design_needed":
            continue
        config = summary.get("candidate_config") or {}
        requested = config.get("requested_target")
        if requested == "baseline":
            category = "baseline_reference"
            failures: list[str] = []
            checks = {"baseline_reference": True}
        else:
            baseline = baselines.get(str(summary.get("base_candidate"))) or {}
            baseline_falling = _regime_metric_mean(baseline, "falling", "lower_breach_rate")
            baseline_interval = _metric_mean(baseline, "asymmetric_interval_score")
            falling_lower = _regime_metric_mean(summary, "falling", "lower_breach_rate")
            interval = _metric_mean(summary, "asymmetric_interval_score")
            checks = {
                "validation_band_gate_pass": float(summary.get("band_gate_pass_rate") or 0.0) >= 1.0,
                "coverage_abs_error": (_metric_mean(summary, "coverage_abs_error") or float("inf")) <= 0.05,
                "lower_breach_rate": (_metric_mean(summary, "lower_breach_rate") or float("inf")) <= 0.18,
                "falling_lower_breach_improved": (
                    falling_lower is not None and baseline_falling is not None and falling_lower < baseline_falling
                ),
                "interval_score_not_much_worse": (
                    interval is not None
                    and baseline_interval is not None
                    and interval <= baseline_interval * 1.10
                ),
                "band_width_ic": (_metric_mean(summary, "band_width_ic") or -float("inf")) > 0.15,
                "downside_width_ic": (_metric_mean(summary, "downside_width_ic") or -float("inf")) >= 0.0,
            }
            failures = [key for key, passed in checks.items() if not passed]
            category = "selectable_verified" if not failures else "experiment_record"
        summary["decision"] = {
            "category": category,
            "checks": checks,
            "failures": failures,
        }
    selectable = [summary for summary in summaries if (summary.get("decision") or {}).get("category") == "selectable_verified"]
    if selectable:
        best = sorted(
            selectable,
            key=lambda item: (
                _regime_metric_mean(item, "falling", "lower_breach_rate") or float("inf"),
                _metric_mean(item, "coverage_abs_error") or float("inf"),
                _metric_mean(item, "asymmetric_interval_score") or float("inf"),
            ),
        )[0]
        best["decision"]["category"] = "recommended_default"


def _summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        stats = summary.get("validation_metric_stats") or {}
        decision = summary.get("decision") or {}
        rows.append(
            {
                "experiment": summary.get("experiment"),
                "category": decision.get("category"),
                "base_candidate": summary.get("base_candidate"),
                "completed": summary.get("completed_run_count"),
                "gate_rate": summary.get("band_gate_pass_rate"),
                "test_exposure": summary.get("test_exposure_count"),
                "cov_abs": (stats.get("coverage_abs_error") or {}).get("mean"),
                "lower": (stats.get("lower_breach_rate") or {}).get("mean"),
                "falling_lower": _regime_metric_mean(summary, "falling", "lower_breach_rate"),
                "interval": (stats.get("asymmetric_interval_score") or {}).get("mean"),
                "bw_ic": (stats.get("band_width_ic") or {}).get("mean"),
                "down_ic": (stats.get("downside_width_ic") or {}).get("mean"),
                "p90_width": (stats.get("p90_band_width") or {}).get("mean"),
                "failures": ",".join(decision.get("failures") or []),
            }
        )
    return rows


def _run_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for run in summary.get("runs", []):
            metrics = run.get("validation_band_metrics") or {}
            rows.append(
                {
                    "experiment": summary.get("experiment"),
                    "seed": run.get("seed"),
                    "status": run.get("execution_status"),
                    "exit_code": run.get("exit_code"),
                    "gate": (run.get("gate") or {}).get("band_gate_pass"),
                    "cov_abs": metrics.get("coverage_abs_error"),
                    "lower": metrics.get("lower_breach_rate"),
                    "interval": metrics.get("asymmetric_interval_score"),
                    "bw_ic": metrics.get("band_width_ic"),
                    "down_ic": metrics.get("downside_width_ic"),
                    "epoch_seconds_mean": (run.get("epoch_summary") or {}).get("epoch_seconds_mean"),
                    "vram_peak_mb": (run.get("epoch_summary") or {}).get("vram_peak_allocated_mb"),
                    "run_id": run.get("run_id"),
                }
            )
    return rows


def _regime_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        regime_stats = summary.get("validation_regime_metric_stats") or {}
        for regime_name in REGIME_NAMES:
            stats = regime_stats.get(regime_name) or {}
            rows.append(
                {
                    "experiment": summary.get("experiment"),
                    "regime": regime_name,
                    "samples": (stats.get("sample_count") or {}).get("mean"),
                    "cov_abs": (stats.get("coverage_abs_error") or {}).get("mean"),
                    "lower": (stats.get("lower_breach_rate") or {}).get("mean"),
                    "interval": (stats.get("asymmetric_interval_score") or {}).get("mean"),
                    "bw_ic": (stats.get("band_width_ic") or {}).get("mean"),
                    "down_ic": (stats.get("downside_width_ic") or {}).get("mean"),
                    "p90_width": (stats.get("p90_band_width") or {}).get("mean"),
                }
            )
    return rows


def _registry_entry(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("candidate_config") or {}
    stats = summary.get("validation_metric_stats") or {}
    decision = summary.get("decision") or {}
    category = decision.get("category") or "experiment_record"
    return {
        "category": category,
        "display_name": f"1W {config.get('model')} {summary.get('experiment')}",
        "role": "band_model",
        "timeframe": "1W",
        "horizon": 4,
        "model_family": config.get("model"),
        "feature_set": config.get("feature_set"),
        "target_type": "raw_future_return",
        "band_mode": config.get("band_mode"),
        "loss_adjustment": {
            "lambda_band": config.get("lambda_band"),
            "lower_band_loss_weight": config.get("lower_band_loss_weight"),
            "upper_band_loss_weight": config.get("upper_band_loss_weight"),
            "implementation_status": config.get("implementation_status"),
        },
        "strength_summary": (
            f"val cov_abs={_fmt((stats.get('coverage_abs_error') or {}).get('mean'))}, "
            f"falling lower={_fmt(_regime_metric_mean(summary, 'falling', 'lower_breach_rate'))}, "
            f"bw_ic={_fmt((stats.get('band_width_ic') or {}).get('mean'))}"
        ),
        "weakness_summary": ", ".join(decision.get("failures") or summary.get("validation_regime_warnings") or ["기준 통과"]),
        "best_use_case": "1W raw AI band downside guard 검증",
        "why_not_default": "현재 기본값" if category == "recommended_default" else ", ".join(decision.get("failures") or ["기준선 또는 실험 기록"]),
        "key_metrics": {
            key: {
                "validation_mean": (stats.get(key) or {}).get("mean"),
                "validation_std": (stats.get(key) or {}).get("std"),
            }
            for key in STABILITY_KEYS
        },
        "falling_regime_lower_breach_rate": _regime_metric_mean(summary, "falling", "lower_breach_rate"),
        "band_gate_pass_rate": summary.get("band_gate_pass_rate"),
        "test_exposure_count": summary.get("test_exposure_count"),
        "test_metric_policy": summary.get("test_metric_policy"),
        "raw_run_ids": [run.get("run_id") for run in summary.get("runs", []) if run.get("run_id")],
        "source_experiment_id": summary.get("experiment"),
        "base_candidate": summary.get("base_candidate"),
    }


def _write_summary_csv(summaries: list[dict[str, Any]]) -> None:
    rows = _summary_rows(summaries)
    fieldnames = [
        "experiment",
        "category",
        "base_candidate",
        "completed",
        "gate_rate",
        "test_exposure",
        "cov_abs",
        "lower",
        "falling_lower",
        "interval",
        "bw_ic",
        "down_ic",
        "p90_width",
        "failures",
    ]
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(payload: dict[str, Any]) -> None:
    summaries = payload.get("experiment_summaries") or []
    report = [
        "# CP124-BM 1W Band Loss / Downside Guard 제한 실험",
        "",
        "## 1. 결론",
        "",
        f"- 상태: {payload.get('status')}",
        f"- final_recommended_default: `{payload.get('final_recommended_default')}`",
        "- save-run, DB write, inference 저장, W&B, composite, 프론트 수정은 수행하지 않았다.",
        "- 후보 선택은 validation metric과 validation falling regime metric만 사용했다.",
        "- CP121 기준선은 학습을 재실행하지 않고 checkpoint를 재사용했으며, regime metric은 현재 yfinance local snapshot 기준으로 다시 계산했다.",
        "- test metric은 ai.train 결과 구조상 생성될 수 있으나 read-only count만 기록했고 선택에는 쓰지 않았다.",
        "- lower_breach_penalty/asymmetric_interval 강화는 lower quantile pinball loss weight로만 작게 구현했다.",
        "- width_alignment는 현재 lambda_width가 실제 손실에 연결되지 않아 design_needed로 기록했다.",
        "",
        "## 2. 손실 지원 범위",
        "",
        "- 기본 ForecastCompositeLoss: line Huber + band pinball + direct 교차 패널티 + direction 보조 손실.",
        "- CP124 추가: `--lower-band-loss-weight`, `--upper-band-loss-weight` CLI. 기본값은 1.0/1.0이라 기존 동작은 유지된다.",
        "- lower guard: lower quantile loss weight 1.5.",
        "- asymmetric guard: lower quantile loss weight 2.0, upper는 1.0 유지.",
        "- width alignment: 큰 구조 변경 없이 지원할 수 없어 이번 CP에서는 실행하지 않았다.",
        "",
        "## 3. 실험 요약",
        "",
        _table(
            _summary_rows(summaries),
            [
                ("experiment", "experiment"),
                ("category", "category"),
                ("base", "base_candidate"),
                ("runs", "completed"),
                ("gate", "gate_rate"),
                ("test_exp", "test_exposure"),
                ("cov_abs", "cov_abs"),
                ("lower", "lower"),
                ("falling_lower", "falling_lower"),
                ("interval", "interval"),
                ("bw_ic", "bw_ic"),
                ("down_ic", "down_ic"),
                ("p90_w", "p90_width"),
                ("failures", "failures"),
            ],
        ),
        "",
        "## 4. Seed별 실행 결과",
        "",
        _table(
            _run_rows(summaries),
            [
                ("experiment", "experiment"),
                ("seed", "seed"),
                ("status", "status"),
                ("exit", "exit_code"),
                ("gate", "gate"),
                ("cov_abs", "cov_abs"),
                ("lower", "lower"),
                ("interval", "interval"),
                ("bw_ic", "bw_ic"),
                ("down_ic", "down_ic"),
                ("epoch_s", "epoch_seconds_mean"),
                ("vram_mb", "vram_peak_mb"),
            ],
        ),
        "",
        "## 5. Regime별 Validation 평균",
        "",
        _table(
            _regime_rows(summaries),
            [
                ("experiment", "experiment"),
                ("regime", "regime"),
                ("samples", "samples"),
                ("cov_abs", "cov_abs"),
                ("lower", "lower"),
                ("interval", "interval"),
                ("bw_ic", "bw_ic"),
                ("down_ic", "down_ic"),
                ("p90_w", "p90_width"),
            ],
        ),
        "",
        "## 6. 판정 기준",
        "",
        "- validation band_gate pass.",
        "- coverage_abs_error <= 0.05.",
        "- lower_breach_rate <= 0.18.",
        "- falling regime lower_breach_rate가 같은 baseline보다 개선.",
        "- asymmetric_interval_score가 baseline 대비 10% 초과 악화되지 않음.",
        "- band_width_ic > 0.15.",
        "- downside_width_ic >= 0.",
        "",
        "## 7. 산출물",
        "",
        f"- `{REPORT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{REGISTRY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def _write_outputs(summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    _classify_experiments(summaries)
    registry = [_registry_entry(summary) for summary in summaries]
    replacement = next((row.get("source_experiment_id") for row in registry if row.get("category") == "recommended_default"), None)
    final_default = replacement or "tide_pvv_q15_param"
    status = "PASS" if all(summary.get("completed_run_count", 0) > 0 or (summary.get("decision") or {}).get("category") == "design_needed" for summary in summaries) else "PARTIAL_OR_FAIL"
    payload = {
        "cp": "CP124-BM",
        "status": status,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution_policy": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "disabled",
            "composite": False,
            "selection_uses_test_metrics": False,
            "local_snapshot_required": True,
        },
        "environment": {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(PROJECT_ROOT / "data" / "parquet"),
            "WANDB_MODE": "disabled",
        },
        "criteria": {
            "validation_coverage_abs_error_mean_max": 0.05,
            "validation_lower_breach_rate_mean_max": 0.18,
            "falling_lower_breach_rate_must_improve_baseline": True,
            "asymmetric_interval_score_max_baseline_multiplier": 1.10,
            "validation_band_width_ic_mean_min": 0.15,
            "validation_downside_width_ic_mean_min": 0.0,
        },
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "final_recommended_default": final_default,
        "experiment_summaries": summaries,
        "candidate_registry": registry,
    }
    _write_json(METRICS_PATH, payload)
    _write_json(REGISTRY_PATH, registry)
    _write_summary_csv(summaries)
    _write_report(payload)


def run(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = _existing_runs()
    selected = [experiment for experiment in EXPERIMENTS if not args.only or experiment.name in set(args.only)]
    summaries: list[dict[str, Any]] = []
    for experiment in selected:
        if not experiment.runnable:
            summary = _design_needed_summary(experiment)
            summaries.append(summary)
            _write_outputs(summaries, args)
            print(f"[CP124] 설계 필요: {experiment.name}", flush=True)
            continue
        runs: list[dict[str, Any]]
        if experiment.reuse_cp121_baseline:
            runs = _cp121_baseline_runs(experiment, args)
            print(f"[CP124] CP121 기준선 재사용: {experiment.name} runs={len(runs)}", flush=True)
        else:
            runs = []
            for seed in args.seeds:
                cached = existing.get((experiment.name, seed))
                if cached and not args.force:
                    runs.append(cached)
                    continue
                print(f"[CP124] 시작: {experiment.name} seed={seed}", flush=True)
                run_record = _run_training(experiment, seed, args)
                if run_record.get("execution_status") == "PASS":
                    try:
                        run_record["validation_regime_metrics"] = _validation_regime_metrics(run_record, args)
                    except Exception as exc:
                        run_record["validation_regime_metrics"] = {"error": str(exc)}
                print(
                    f"[CP124] 종료: {experiment.name} seed={seed} status={run_record.get('execution_status')} "
                    f"exit={run_record.get('exit_code')} elapsed={_fmt(run_record.get('elapsed_seconds'), 2)}",
                    flush=True,
                )
                runs.append(run_record)
        summaries.append(_aggregate_experiment(experiment, runs))
        _write_outputs(summaries, args)
    _write_outputs(summaries, args)
    print(json.dumps({"status": "done", "metrics_path": str(METRICS_PATH), "registry_path": str(REGISTRY_PATH)}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
