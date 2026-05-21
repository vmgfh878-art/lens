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

from ai.train import resolve_feature_columns  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp119_bm_1w_band_feature_group_experiment_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp119_bm_1w_band_feature_group_experiment_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp119_bm_1w_band_candidate_registry.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp119_bm_1w_band_experiment_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp119_bm_1w_band_feature_group_experiment_logs"
CP113_METRICS_PATH = PROJECT_ROOT / "docs" / "cp113_bm_1w_band_limited_validation_metrics.json"
CP114_METRICS_PATH = PROJECT_ROOT / "docs" / "cp114_bm_1w_band_candidate_expansion_metrics.json"

BAND_METRIC_KEYS = [
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "avg_band_width",
    "median_band_width",
    "p90_band_width",
    "asymmetric_interval_score",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_realized_vol_ratio",
    "width_bucket_downside_rate_ratio",
    "squeeze_breakout_rate",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: str
    q_low: float
    q_high: float
    note: str


EXPERIMENTS = [
    Experiment("pvv_q10_direct", "price_volatility_volume", 0.10, 0.90, "CP114 recommended 후보 재현"),
    Experiment("pvv_q15_direct", "price_volatility_volume", 0.15, 0.85, "q10 과보수 여부 비교"),
    Experiment("no_fundamentals_q10_direct", "no_fundamentals", 0.10, 0.90, "CP114 val gate 약점 재확인"),
    Experiment("technical_only_q10_direct", "technical_only", 0.10, 0.90, "기술지표 중심 feature 확인"),
    Experiment("full_features_q10_direct", "full_features", 0.10, 0.90, "전체 피처 noise 여부 확인"),
    Experiment("price_return_only_q10_direct", "price_return_only", 0.10, 0.90, "최소 가격/수익률 ablation"),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_train_json(stdout: str) -> dict[str, Any]:
    marker = "[EXIT-MARKER {\"step\": \"before_result_json\""
    marker_index = stdout.rfind(marker)
    search_start = stdout.find("\n", marker_index) + 1 if marker_index >= 0 else 0
    decoder = json.JSONDecoder()
    index = search_start
    while index < len(stdout):
        if stdout[index] != "{":
            index += 1
            continue
        try:
            parsed, end_index = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(parsed, dict) and "run_id" in parsed and "test_metrics" in parsed:
            return parsed
        index += max(end_index, 1)
    raise ValueError("ai.train stdout에서 결과 JSON을 찾지 못했습니다.")


def _band_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    source = metrics or {}
    if isinstance(source.get("band_metrics"), dict):
        source = source["band_metrics"]
    return {key: source.get(key) for key in BAND_METRIC_KEYS}


def _epoch_summary(stdout: str) -> dict[str, Any]:
    epoch_seconds: list[float] = []
    vram_peak_values: list[float] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        epoch_value = _safe_float(payload.get("epoch_seconds"))
        if epoch_value is not None:
            epoch_seconds.append(epoch_value)
        vram_value = _safe_float(payload.get("vram_peak_allocated_mb"))
        if vram_value is not None:
            vram_peak_values.append(vram_value)
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "vram_peak_allocated_mb": max(vram_peak_values) if vram_peak_values else None,
    }


def _load_references() -> dict[str, Any]:
    cp113 = _read_json(CP113_METRICS_PATH)
    cp114 = _read_json(CP114_METRICS_PATH)
    cp113_metrics = cp113.get("test_band_metrics") or {}
    cp114_experiments = {
        row.get("name"): row
        for row in cp114.get("experiments", [])
        if isinstance(row, dict) and row.get("name")
    }
    cp114_q10 = cp114_experiments.get("cnn_h4_q10_pvv_direct", {})
    return {
        "cp113_q15_pvv_direct": {
            "source": str(CP113_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "run_id": (cp113.get("execution") or {}).get("run_id"),
            "band_metrics": cp113_metrics,
        },
        "cp114_q10_pvv_direct": {
            "source": str(CP114_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "run_id": cp114_q10.get("run_id"),
            "band_metrics": cp114_q10.get("test_band_metrics") or {},
        },
    }


def _criteria(references: dict[str, Any]) -> dict[str, Any]:
    cp113_metrics = (references.get("cp113_q15_pvv_direct") or {}).get("band_metrics") or {}
    interval_limit = _safe_float(cp113_metrics.get("asymmetric_interval_score")) or 0.317112
    p90_reference = _safe_float(cp113_metrics.get("p90_band_width")) or 0.333164
    return {
        "coverage_abs_error_max": 0.05,
        "lower_breach_rate_max": 0.18,
        "asymmetric_interval_score_max": interval_limit,
        "band_width_ic_min": 0.15,
        "downside_width_ic_min": 0.0,
        "p90_band_width_max": p90_reference * 1.15,
        "p90_reference_source": "CP113 q15 PVV p90 * 1.15",
    }


def _metric_delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    left = _safe_float(current.get(key))
    right = _safe_float(reference.get(key))
    if left is None or right is None:
        return None
    return left - right


def _classify_record(record: dict[str, Any], criteria: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    if record.get("execution_status") == "SKIPPED":
        return {"category": "experiment_record", "verified": False, "reasons": ["feature_set 미구현으로 실행하지 않음"]}
    if record.get("execution_status") != "PASS":
        return {"category": "rejected", "verified": False, "reasons": [record.get("failed_with_reason") or "실행 실패"]}

    metrics = record.get("band_metrics") or {}
    gate = record.get("gate") or {}
    band_gate_pass = bool(gate.get("band_gate_pass")) and not bool(gate.get("gate_failed"))
    checks = {
        "band_gate_pass": band_gate_pass,
        "coverage_abs_error": (_safe_float(metrics.get("coverage_abs_error")) or float("inf")) <= criteria["coverage_abs_error_max"],
        "lower_breach_rate": (_safe_float(metrics.get("lower_breach_rate")) or float("inf")) <= criteria["lower_breach_rate_max"],
        "asymmetric_interval_score": (_safe_float(metrics.get("asymmetric_interval_score")) or float("inf")) <= criteria["asymmetric_interval_score_max"],
        "band_width_ic": (_safe_float(metrics.get("band_width_ic")) or -float("inf")) > criteria["band_width_ic_min"],
        "downside_width_ic": (_safe_float(metrics.get("downside_width_ic")) or -float("inf")) >= criteria["downside_width_ic_min"],
        "p90_band_width": (_safe_float(metrics.get("p90_band_width")) or float("inf")) <= criteria["p90_band_width_max"],
    }
    reasons = [key for key, ok in checks.items() if ok]
    failures = [key for key, ok in checks.items() if not ok]
    cp113_ref = ((references.get("cp113_q15_pvv_direct") or {}).get("band_metrics") or {})
    cp114_ref = ((references.get("cp114_q10_pvv_direct") or {}).get("band_metrics") or {})
    comparison = {
        "coverage_abs_error_delta_vs_cp113_q15": _metric_delta(metrics, cp113_ref, "coverage_abs_error"),
        "asymmetric_interval_score_delta_vs_cp113_q15": _metric_delta(metrics, cp113_ref, "asymmetric_interval_score"),
        "coverage_abs_error_delta_vs_cp114_q10": _metric_delta(metrics, cp114_ref, "coverage_abs_error"),
        "asymmetric_interval_score_delta_vs_cp114_q10": _metric_delta(metrics, cp114_ref, "asymmetric_interval_score"),
        "p90_band_width_delta_vs_cp114_q10": _metric_delta(metrics, cp114_ref, "p90_band_width"),
        "band_width_ic_delta_vs_cp114_q10": _metric_delta(metrics, cp114_ref, "band_width_ic"),
    }
    verified = all(checks.values())
    if verified:
        category = "selectable_verified"
    elif band_gate_pass or (_safe_float(metrics.get("band_width_ic")) or 0.0) > 0.0:
        category = "experiment_record"
    else:
        category = "rejected"
    return {
        "category": category,
        "verified": verified,
        "checks": checks,
        "reasons": reasons,
        "failures": failures,
        "comparison": comparison,
    }


def _select_recommended(records: list[dict[str, Any]]) -> str | None:
    verified = [row for row in records if (row.get("decision") or {}).get("category") == "selectable_verified"]
    if not verified:
        return None
    def feature_priority(row: dict[str, Any]) -> int:
        feature_set = (row.get("experiment") or {}).get("feature_set")
        name = str(row.get("name") or "")
        if feature_set == "price_volatility_volume" and "q10" in name:
            return 0
        if feature_set == "price_volatility_volume":
            return 1
        if feature_set == "technical_only":
            return 2
        if feature_set == "full_features":
            return 3
        return 4

    return sorted(
        verified,
        key=lambda row: (
            _safe_float((row.get("band_metrics") or {}).get("coverage_abs_error")) or float("inf"),
            _safe_float((row.get("band_metrics") or {}).get("asymmetric_interval_score")) or float("inf"),
            -(_safe_float((row.get("band_metrics") or {}).get("band_width_ic")) or -999.0),
            feature_priority(row),
        ),
    )[0].get("name")


def _apply_recommended(records: list[dict[str, Any]]) -> None:
    recommended_name = _select_recommended(records)
    for record in records:
        decision = record.setdefault("decision", {})
        if decision.get("category") == "selectable_verified" and record.get("name") == recommended_name:
            decision["category"] = "recommended_default"


def _feature_set_columns(feature_set: str) -> list[str]:
    return resolve_feature_columns(feature_set)


def _command_for_experiment(experiment: Experiment, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        "cnn_lstm",
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
        "2.0",
        "--band-mode",
        "direct",
        "--checkpoint-selection",
        "band_gate",
        "--fp32-modules",
        "lstm,heads",
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
        "--local-log-dir",
        str(LOG_DIR / "local_runs" / experiment.name),
    ]
    return command


def _run_experiment(experiment: Experiment, args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{experiment.name}.stdout.log"
    try:
        feature_columns = _feature_set_columns(experiment.feature_set)
    except Exception as exc:
        return {
            "name": experiment.name,
            "experiment": asdict(experiment),
            "execution_status": "SKIPPED",
            "failed_with_reason": f"feature_set 미구현: {exc}",
            "feature_columns": [],
            "n_features": 0,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        }

    command = _command_for_experiment(experiment, args)
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
                "# environment: yfinance local snapshots, WANDB_MODE=disabled",
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
        "name": experiment.name,
        "experiment": asdict(experiment),
        "command": command,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "execution_status": execution_status,
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "timeout_expired": timeout_expired,
        "epoch_summary": _epoch_summary(stdout),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
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
            "n_features": result.get("n_features") or len(feature_columns),
            "feature_columns": result.get("feature_columns") or feature_columns,
            "validation_band_metrics": _band_metrics(best_metrics),
            "band_metrics": _band_metrics(test_metrics),
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


def _log_footer_value(text: str, key: str) -> str | None:
    prefix = f"# {key}:"
    for line in reversed(text.splitlines()):
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _recover_record_from_log(experiment: Experiment) -> dict[str, Any] | None:
    log_path = LOG_DIR / f"{experiment.name}.stdout.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        result = _extract_train_json(text)
    except Exception:
        return None
    try:
        feature_columns = _feature_set_columns(experiment.feature_set)
    except Exception:
        feature_columns = result.get("feature_columns") or []
    best_metrics = result.get("best_metrics") if isinstance(result.get("best_metrics"), dict) else {}
    test_metrics = result.get("test_metrics") if isinstance(result.get("test_metrics"), dict) else {}
    exit_code = _safe_float(_log_footer_value(text, "exit_code"))
    elapsed = _safe_float(_log_footer_value(text, "elapsed_seconds"))
    return {
        "name": experiment.name,
        "experiment": asdict(experiment),
        "command": result.get("command"),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "execution_status": "PASS",
        "exit_code": int(exit_code) if exit_code is not None else 0,
        "started_at": None,
        "ended_at": None,
        "elapsed_seconds": elapsed,
        "timeout_expired": False,
        "epoch_summary": _epoch_summary(text),
        "run_id": result.get("run_id"),
        "checkpoint_path": result.get("checkpoint_path"),
        "source_data_hash": result.get("source_data_hash"),
        "local_log_dir": result.get("local_log_dir"),
        "feature_version": result.get("feature_version"),
        "n_features": result.get("n_features") or len(feature_columns),
        "feature_columns": result.get("feature_columns") or feature_columns,
        "validation_band_metrics": _band_metrics(best_metrics),
        "band_metrics": _band_metrics(test_metrics),
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
        "recovered_from_log": True,
    }


def _existing_records() -> list[dict[str, Any]]:
    payload = _read_json(METRICS_PATH)
    records = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
    return [row for row in records if isinstance(row, dict)]


def _registry_entry(record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("decision") or {}
    metrics = record.get("band_metrics") or {}
    category = decision.get("category") or "experiment_record"
    feature_set = (record.get("experiment") or {}).get("feature_set")
    weakness = "검증 기준 통과" if category in {"recommended_default", "selectable_verified"} else "검증 기준 미달"
    if decision.get("failures"):
        weakness = ", ".join(decision["failures"])
    elif feature_set == "technical_only":
        weakness = "price_volatility_volume와 동일 11개 컬럼이라 별도 UI 항목으로는 중복"
    elif feature_set == "full_features":
        weakness = "PVV q10보다 coverage_abs_error가 큼"
    elif record.get("name") == "pvv_q15_direct":
        weakness = "PVV q10보다 coverage_abs_error와 lower/upper breach가 큼"
    return {
        "category": category,
        "display_name": f"1W CNN-LSTM {record.get('name')}",
        "role": "band_model",
        "timeframe": "1W",
        "horizon": 4,
        "model_family": "cnn_lstm",
        "feature_set": feature_set,
        "target_type": "raw_future_return",
        "band_mode": "direct",
        "strength_summary": _strength_summary(record),
        "weakness_summary": weakness,
        "best_use_case": _best_use_case(record),
        "why_not_default": "현재 기본값" if category == "recommended_default" else weakness,
        "key_metrics": {key: metrics.get(key) for key in BAND_METRIC_KEYS},
        "raw_run_id": record.get("run_id"),
        "source_experiment_id": record.get("name"),
        "execution_status": record.get("execution_status"),
    }


def _strength_summary(record: dict[str, Any]) -> str:
    metrics = record.get("band_metrics") or {}
    cov = _fmt(metrics.get("coverage_abs_error"))
    interval = _fmt(metrics.get("asymmetric_interval_score"))
    bwic = _fmt(metrics.get("band_width_ic"))
    return f"coverage_abs_error={cov}, interval={interval}, band_width_ic={bwic}"


def _best_use_case(record: dict[str, Any]) -> str:
    name = str(record.get("name") or "")
    if "full_features" in name:
        return "전체 36개 피처 기준선 확인"
    if "no_fundamentals" in name:
        return "fundamentals 제거 안정성 확인"
    if "technical_only" in name:
        return "기술지표 중심 후보 확인"
    if "price_return_only" in name:
        return "최소 가격/수익률 ablation"
    if "q15" in name:
        return "q10 과보수 의심 시 비교 대안"
    return "기본 1W band 후보"


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record.get("band_metrics") or {}
        gate = record.get("gate") or {}
        decision = record.get("decision") or {}
        rows.append(
            {
                "name": record.get("name"),
                "feature_set": (record.get("experiment") or {}).get("feature_set"),
                "q_low": (record.get("experiment") or {}).get("q_low"),
                "q_high": (record.get("experiment") or {}).get("q_high"),
                "status": record.get("execution_status"),
                "exit_code": record.get("exit_code"),
                "category": decision.get("category"),
                "band_gate_pass": gate.get("band_gate_pass"),
                "n_features": record.get("n_features"),
                "run_id": record.get("run_id"),
                **{key: metrics.get(key) for key in BAND_METRIC_KEYS},
            }
        )
    return rows


def _write_summary_csv(records: list[dict[str, Any]]) -> None:
    rows = _summary_rows(records)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "feature_set",
        "q_low",
        "q_high",
        "status",
        "exit_code",
        "category",
        "band_gate_pass",
        "n_features",
        "run_id",
        *BAND_METRIC_KEYS,
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_fmt(value))
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _write_report(payload: dict[str, Any]) -> None:
    records = payload.get("experiments") or []
    rows = _summary_rows(records)
    registry = payload.get("candidate_registry") or []
    report = [
        "# CP119-BM 1W Band Feature Group 제한 실험",
        "",
        "## 1. 결론",
        "",
        f"- 상태: {payload.get('status')}",
        "- 새 save-run, DB write, inference 저장, W&B, composite, Supabase 대량 read, 프론트 수정은 수행하지 않았다.",
        "- yfinance local 1W parquet snapshot만 사용했다.",
        f"- recommended_default: `{payload.get('recommended_default')}`",
        "- 검증 기준을 통과하지 못한 watch/research 성격 후보는 제품 UI 후보가 아니라 experiment_record로만 남겼다.",
        "",
        "## 2. 실행 조건",
        "",
        _table(
            [
                {"item": "timeframe", "value": "1W"},
                {"item": "horizon", "value": 4},
                {"item": "seq_len", "value": 104},
                {"item": "model", "value": "cnn_lstm"},
                {"item": "band_mode", "value": "direct"},
                {"item": "epochs", "value": payload.get("epochs")},
                {"item": "batch_size", "value": payload.get("batch_size")},
                {"item": "checkpoint_selection", "value": "band_gate"},
                {"item": "W&B", "value": "disabled"},
                {"item": "save-run", "value": "false"},
            ],
            [("항목", "item"), ("값", "value")],
        ),
        "",
        "## 3. 실험 결과",
        "",
        _table(
            rows,
            [
                ("실험", "name"),
                ("feature_set", "feature_set"),
                ("q", "q_low"),
                ("상태", "status"),
                ("분류", "category"),
                ("gate", "band_gate_pass"),
                ("cov_abs", "coverage_abs_error"),
                ("lower", "lower_breach_rate"),
                ("upper", "upper_breach_rate"),
                ("p90_w", "p90_band_width"),
                ("interval", "asymmetric_interval_score"),
                ("bw_ic", "band_width_ic"),
                ("down_ic", "downside_width_ic"),
            ],
        ),
        "",
        "## 4. 검증 기준",
        "",
        _table(
            [{"item": key, "value": value} for key, value in (payload.get("criteria") or {}).items()],
            [("항목", "item"), ("값", "value")],
        ),
        "",
        "## 5. Candidate Registry",
        "",
        _table(
            [
                {
                    "category": row.get("category"),
                    "display_name": row.get("display_name"),
                    "feature_set": row.get("feature_set"),
                    "run_id": row.get("raw_run_id"),
                    "strength": row.get("strength_summary"),
                    "weakness": row.get("weakness_summary"),
                }
                for row in registry
            ],
            [
                ("category", "category"),
                ("display_name", "display_name"),
                ("feature_set", "feature_set"),
                ("run_id", "run_id"),
                ("strength", "strength"),
                ("weakness", "weakness"),
            ],
        ),
        "",
        "## 6. 해석",
        "",
        "- `price_volatility_volume` q10은 CP114 기준 후보 재현성 확인 대상이다. 새 실험이 기준을 넘으면 recommended_default를 새 run으로 갱신한다.",
        "- `price_volatility_volume` q15는 q10/q90이 과보수일 때의 비교 대안이다. 기준을 넘더라도 q10보다 균형이 약하면 selectable_verified로 둔다.",
        "- `technical_only`는 현재 CP63 정의상 `price_volatility_volume`과 같은 11개 컬럼이라 metric도 동일하다. 검증 기준은 통과했지만 별도 UI 항목으로는 중복이므로 PVV로 통합 관리하는 편이 낫다.",
        "- `full_features` q10은 검증 기준을 통과했다. PVV보다 coverage_abs_error는 크지만 interval, p90 width, downside_width_ic는 더 좋아 selectable_verified 대안으로 남긴다.",
        "- `no_fundamentals`는 coverage만 보면 좋지만 interval, dynamic width, p90 width 기준을 놓쳐 experiment_record다.",
        "- `price_return_only`는 metric 일부가 좋지만 band_gate를 통과하지 못해 제품 UI 후보가 아니다.",
        "- calibration 전제 후보는 verified로 올리지 않는다. 이번 판정은 raw band 기준이다.",
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


def _write_outputs(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    order = {experiment.name: index for index, experiment in enumerate(EXPERIMENTS)}
    records.sort(key=lambda row: order.get(str(row.get("name")), 999))
    references = _load_references()
    criteria = _criteria(references)
    for record in records:
        record["decision"] = _classify_record(record, criteria, references)
    _apply_recommended(records)
    registry = [_registry_entry(record) for record in records]
    recommended = next((row.get("source_experiment_id") for row in registry if row.get("category") == "recommended_default"), None)
    payload = {
        "cp": "CP119-BM",
        "status": "PASS" if all(row.get("execution_status") in {"PASS", "SKIPPED"} for row in records) else "PARTIAL_OR_FAIL",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution_policy": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "disabled",
            "composite": False,
            "local_snapshot_required": True,
        },
        "environment": {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(PROJECT_ROOT / "data" / "parquet"),
            "WANDB_MODE": "disabled",
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "criteria": criteria,
        "references": references,
        "recommended_default": recommended,
        "experiments": records,
        "candidate_registry": registry,
    }
    _write_json(METRICS_PATH, payload)
    _write_json(REGISTRY_PATH, registry)
    _write_summary_csv(records)
    _write_report(payload)


def run(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    selected = [item for item in EXPERIMENTS if not args.only or item.name in set(args.only)]
    existing_by_name = {row.get("name"): row for row in _existing_records()}
    records: list[dict[str, Any]] = []
    for experiment in selected:
        existing = existing_by_name.get(experiment.name)
        if existing and existing.get("execution_status") in {"PASS", "SKIPPED"} and not args.force:
            records.append(existing)
            continue
        print(f"[CP119] 시작: {experiment.name}", flush=True)
        record = _run_experiment(experiment, args)
        print(
            f"[CP119] 종료: {experiment.name} status={record.get('execution_status')} exit={record.get('exit_code')} elapsed={_fmt(record.get('elapsed_seconds'), 2)}",
            flush=True,
        )
        records.append(record)
        _write_outputs(records, args)
    record_names = {row.get("name") for row in records}
    for experiment in EXPERIMENTS:
        if experiment.name in record_names:
            continue
        existing = existing_by_name.get(experiment.name)
        recovered = existing or _recover_record_from_log(experiment)
        if recovered:
            records.append(recovered)
            record_names.add(experiment.name)
    _write_outputs(records, args)
    print(json.dumps({"status": "done", "metrics_path": str(METRICS_PATH), "registry_path": str(REGISTRY_PATH)}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32", "off"], default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
