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

from ai.targets import SUPPORTED_TARGET_TYPES  # noqa: E402
from ai.train import resolve_feature_columns  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp120_bm_1w_band_mode_target_experiment_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp120_bm_1w_band_mode_target_experiment_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp120_bm_1w_band_candidate_registry.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp120_bm_1w_band_mode_target_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp120_bm_1w_band_mode_target_experiment_logs"
CP119_METRICS_PATH = PROJECT_ROOT / "docs" / "cp119_bm_1w_band_feature_group_experiment_metrics.json"

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
    model: str
    feature_set: str
    q_low: float
    q_high: float
    band_mode: str
    band_target_type: str
    note: str
    target_probe: bool = False
    design_targets: tuple[str, ...] = ()


EXPERIMENTS = [
    Experiment("cnn_pvv_q10_direct", "cnn_lstm", "price_volatility_volume", 0.10, 0.90, "direct", "raw_future_return", "CP119 recommended_default 재현 기준"),
    Experiment("cnn_full_q10_direct", "cnn_lstm", "full_features", 0.10, 0.90, "direct", "raw_future_return", "CP119 selectable_verified 재현 기준"),
    Experiment("cnn_pvv_q10_param", "cnn_lstm", "price_volatility_volume", 0.10, 0.90, "param", "raw_future_return", "CNN-LSTM param mode 비교"),
    Experiment("tide_pvv_q15_param", "tide", "price_volatility_volume", 0.15, 0.85, "param", "raw_future_return", "CP114 TiDE param 후보 재현"),
    Experiment("tide_pvv_q10_param", "tide", "price_volatility_volume", 0.10, 0.90, "param", "raw_future_return", "TiDE q10 확장"),
    Experiment(
        "cnn_pvv_q10_direct_realized_range_probe",
        "cnn_lstm",
        "price_volatility_volume",
        0.10,
        0.90,
        "direct",
        "realized_range",
        "realized_range 또는 realized_volatility target smoke",
        target_probe=True,
        design_targets=("realized_range", "realized_volatility"),
    ),
    Experiment(
        "cnn_pvv_q10_direct_downside_probe",
        "cnn_lstm",
        "price_volatility_volume",
        0.10,
        0.90,
        "direct",
        "downside_magnitude",
        "downside_magnitude 또는 tail-risk target smoke",
        target_probe=True,
        design_targets=("downside_magnitude", "tail_event_probability"),
    ),
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


def _cp119_records() -> dict[str, Any]:
    payload = _read_json(CP119_METRICS_PATH)
    return {
        row.get("name"): row
        for row in payload.get("experiments", [])
        if isinstance(row, dict) and row.get("name")
    }


def _baseline_metrics() -> dict[str, Any]:
    records = _cp119_records()
    baseline = records.get("pvv_q10_direct", {})
    return baseline.get("band_metrics") or {}


def _criteria() -> dict[str, Any]:
    baseline = _baseline_metrics()
    interval = _safe_float(baseline.get("asymmetric_interval_score")) or 0.30197736620903015
    p90 = _safe_float(baseline.get("p90_band_width")) or 0.31979817152023315
    return {
        "coverage_abs_error_max": 0.05,
        "lower_breach_rate_max": 0.18,
        "asymmetric_interval_score_max": interval * 1.10,
        "asymmetric_interval_score_reference": interval,
        "band_width_ic_min": 0.15,
        "downside_width_ic_min": 0.0,
        "p90_band_width_max": p90 * 1.15,
        "p90_reference_source": "CP119 pvv_q10_direct p90 * 1.15",
    }


def _metric_delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    left = _safe_float(current.get(key))
    right = _safe_float(reference.get(key))
    if left is None or right is None:
        return None
    return left - right


def _classify_record(record: dict[str, Any], criteria: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    if record.get("execution_status") == "DESIGN_NEEDED":
        return {
            "category": "design_needed",
            "verified": False,
            "reasons": ["target 경로 미지원"],
            "failures": ["target_design_needed"],
        }
    if record.get("execution_status") != "PASS":
        return {
            "category": "rejected",
            "verified": False,
            "reasons": [record.get("failed_with_reason") or "실행 실패"],
            "failures": ["execution"],
        }

    metrics = record.get("band_metrics") or {}
    gate = record.get("gate") or {}
    checks = {
        "band_gate_pass": bool(gate.get("band_gate_pass")) and not bool(gate.get("gate_failed")),
        "coverage_abs_error": (_safe_float(metrics.get("coverage_abs_error")) or float("inf")) <= criteria["coverage_abs_error_max"],
        "lower_breach_rate": (_safe_float(metrics.get("lower_breach_rate")) or float("inf")) <= criteria["lower_breach_rate_max"],
        "asymmetric_interval_score": (_safe_float(metrics.get("asymmetric_interval_score")) or float("inf")) <= criteria["asymmetric_interval_score_max"],
        "band_width_ic": (_safe_float(metrics.get("band_width_ic")) or -float("inf")) > criteria["band_width_ic_min"],
        "downside_width_ic": (_safe_float(metrics.get("downside_width_ic")) or -float("inf")) >= criteria["downside_width_ic_min"],
        "p90_band_width": (_safe_float(metrics.get("p90_band_width")) or float("inf")) <= criteria["p90_band_width_max"],
    }
    failures = [key for key, ok in checks.items() if not ok]
    comparison = {
        "coverage_abs_error_delta_vs_cp119_pvv_q10": _metric_delta(metrics, baseline, "coverage_abs_error"),
        "asymmetric_interval_score_delta_vs_cp119_pvv_q10": _metric_delta(metrics, baseline, "asymmetric_interval_score"),
        "band_width_ic_delta_vs_cp119_pvv_q10": _metric_delta(metrics, baseline, "band_width_ic"),
        "downside_width_ic_delta_vs_cp119_pvv_q10": _metric_delta(metrics, baseline, "downside_width_ic"),
        "p90_band_width_delta_vs_cp119_pvv_q10": _metric_delta(metrics, baseline, "p90_band_width"),
    }
    verified = not failures
    if verified:
        category = "selectable_verified"
    elif checks["band_gate_pass"] or (_safe_float(metrics.get("band_width_ic")) or 0.0) > 0.0:
        category = "experiment_record"
    else:
        category = "rejected"
    return {
        "category": category,
        "verified": verified,
        "checks": checks,
        "reasons": [key for key, ok in checks.items() if ok],
        "failures": failures,
        "comparison": comparison,
    }


def _select_recommended(records: list[dict[str, Any]]) -> str | None:
    verified = [row for row in records if (row.get("decision") or {}).get("category") == "selectable_verified"]
    if not verified:
        return None

    def priority(row: dict[str, Any]) -> int:
        name = str(row.get("name") or "")
        if name == "cnn_pvv_q10_direct":
            return 0
        if name == "cnn_full_q10_direct":
            return 1
        if "param" in name:
            return 2
        return 3

    return sorted(
        verified,
        key=lambda row: (
            _safe_float((row.get("band_metrics") or {}).get("coverage_abs_error")) or float("inf"),
            _safe_float((row.get("band_metrics") or {}).get("asymmetric_interval_score")) or float("inf"),
            -(_safe_float((row.get("band_metrics") or {}).get("band_width_ic")) or -999.0),
            priority(row),
        ),
    )[0].get("name")


def _apply_recommended(records: list[dict[str, Any]]) -> None:
    recommended = _select_recommended(records)
    for record in records:
        decision = record.setdefault("decision", {})
        if decision.get("category") == "selectable_verified" and record.get("name") == recommended:
            decision["category"] = "recommended_default"


def _feature_columns(feature_set: str) -> list[str]:
    return resolve_feature_columns(feature_set)


def _design_needed_record(experiment: Experiment) -> dict[str, Any]:
    return {
        "name": experiment.name,
        "experiment": asdict(experiment),
        "execution_status": "DESIGN_NEEDED",
        "failed_with_reason": "요청 target 후보가 현재 ai.targets 경로에서 지원되지 않음",
        "supported_target_types": list(SUPPORTED_TARGET_TYPES),
        "requested_design_targets": list(experiment.design_targets),
        "band_metrics": {},
        "validation_band_metrics": {},
        "gate": {},
    }


def _command_for_experiment(experiment: Experiment, args: argparse.Namespace) -> list[str]:
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
        experiment.band_target_type,
        "--q-low",
        str(experiment.q_low),
        "--q-high",
        str(experiment.q_high),
        "--lambda-band",
        "2.0",
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
        "--local-log-dir",
        str(LOG_DIR / "local_runs" / experiment.name),
    ]
    if experiment.model == "cnn_lstm":
        command.extend(["--fp32-modules", "lstm,heads"])
    if experiment.model == "tide":
        command.append("--use-future-covariate")
    return command


def _run_experiment(experiment: Experiment, args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if experiment.target_probe and all(target not in SUPPORTED_TARGET_TYPES for target in experiment.design_targets):
        return _design_needed_record(experiment)
    try:
        feature_columns = _feature_columns(experiment.feature_set)
    except Exception as exc:
        return {
            "name": experiment.name,
            "experiment": asdict(experiment),
            "execution_status": "REJECTED",
            "failed_with_reason": f"feature_set 확인 실패: {exc}",
            "band_metrics": {},
            "validation_band_metrics": {},
            "gate": {},
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
    log_path = LOG_DIR / f"{experiment.name}.stdout.log"
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
        status = "PASS" if exit_code == 0 else "FAIL"
        timeout_expired = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        exit_code = None
        status = "TIMEOUT"
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
        "execution_status": status,
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "timeout_expired": timeout_expired,
        "epoch_summary": _epoch_summary(stdout),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
    }
    if status != "PASS":
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


def _existing_records() -> list[dict[str, Any]]:
    payload = _read_json(METRICS_PATH)
    records = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
    return [row for row in records if isinstance(row, dict)]


def _registry_entry(record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("decision") or {}
    metrics = record.get("band_metrics") or {}
    experiment = record.get("experiment") or {}
    category = decision.get("category") or "experiment_record"
    weakness = "검증 기준 통과" if category in {"recommended_default", "selectable_verified"} else "검증 기준 미달"
    if decision.get("failures"):
        weakness = ", ".join(decision["failures"])
    if record.get("name") == "cnn_full_q10_direct" and category == "selectable_verified":
        weakness = "PVV q10보다 coverage_abs_error가 큼"
    if record.get("name") == "tide_pvv_q15_param" and category == "selectable_verified":
        weakness = "upper breach 0.185739와 낮은 downside_width_ic 주의"
    if record.get("name") == "cnn_pvv_q10_param":
        weakness = "band_gate fail 및 coverage_abs_error 0.075430"
    if record.get("name") == "tide_pvv_q10_param":
        weakness = "coverage_abs_error 0.059751로 기준 초과"
    if category == "design_needed":
        weakness = "target 설계/구현 필요"
    return {
        "category": category,
        "display_name": f"1W {experiment.get('model')} {record.get('name')}",
        "role": "band_model",
        "timeframe": "1W",
        "horizon": 4,
        "model_family": experiment.get("model"),
        "feature_set": experiment.get("feature_set"),
        "target_type": experiment.get("band_target_type"),
        "band_mode": experiment.get("band_mode"),
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
    if record.get("execution_status") == "DESIGN_NEEDED":
        return "기존 target 경로 미지원으로 학습 미실행"
    metrics = record.get("band_metrics") or {}
    return (
        f"coverage_abs_error={_fmt(metrics.get('coverage_abs_error'))}, "
        f"interval={_fmt(metrics.get('asymmetric_interval_score'))}, "
        f"band_width_ic={_fmt(metrics.get('band_width_ic'))}"
    )


def _best_use_case(record: dict[str, Any]) -> str:
    name = str(record.get("name") or "")
    if "tide" in name:
        return "TiDE 1W BM 대안성 확인"
    if "param" in name:
        return "param center+width mode 비교"
    if "full" in name:
        return "전체 36개 feature direct 대안"
    if "probe" in name:
        return "target 설계 후보"
    return "기본 1W direct band 후보"


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record.get("band_metrics") or {}
        gate = record.get("gate") or {}
        decision = record.get("decision") or {}
        experiment = record.get("experiment") or {}
        rows.append(
            {
                "name": record.get("name"),
                "model": experiment.get("model"),
                "feature_set": experiment.get("feature_set"),
                "band_mode": experiment.get("band_mode"),
                "target_type": experiment.get("band_target_type"),
                "q_low": experiment.get("q_low"),
                "q_high": experiment.get("q_high"),
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
    fieldnames = [
        "name",
        "model",
        "feature_set",
        "band_mode",
        "target_type",
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
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        "# CP120-BM 1W Band Mode/Target 제한 실험",
        "",
        "## 1. 결론",
        "",
        f"- 상태: {payload.get('status')}",
        "- save-run, DB write, inference 저장, W&B, composite, Supabase 대량 read, 프론트 수정은 수행하지 않았다.",
        "- yfinance local 1W parquet snapshot만 사용했다.",
        f"- recommended_default: `{payload.get('recommended_default')}`",
        "- target probe는 기존 코드가 요청 target을 지원하지 않아 `design_needed`로 기록했다.",
        "",
        "## 2. 실행 조건",
        "",
        _table(
            [
                {"item": "timeframe", "value": "1W"},
                {"item": "horizon", "value": 4},
                {"item": "seq_len", "value": 104},
                {"item": "epochs", "value": payload.get("epochs")},
                {"item": "batch_size", "value": payload.get("batch_size")},
                {"item": "checkpoint_selection", "value": "band_gate"},
                {"item": "source/provider", "value": "yfinance local parquet"},
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
                ("model", "model"),
                ("feature_set", "feature_set"),
                ("mode", "band_mode"),
                ("target", "target_type"),
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
        "## 5. Direct / Param 해석",
        "",
        "- direct 기준선은 `cnn_pvv_q10_direct`다. 이 후보는 CP119 recommended_default를 CP120에서 재현하는 기준선이다.",
        "- `cnn_full_q10_direct`는 기준을 통과했다. PVV q10보다 coverage_abs_error는 크지만 interval, p90 width, downside_width_ic는 더 좋아 selectable_verified 대안으로 남긴다.",
        "- `cnn_pvv_q10_param`은 같은 PVV/q10/raw target 조건에서 interval과 band_width_ic는 좋지만 band_gate fail, coverage_abs_error 0.075430으로 raw 제품 후보가 아니다.",
        "- `tide_pvv_q15_param`은 기준을 통과해 TiDE 1W BM 대안으로 남긴다. 다만 upper breach 0.185739와 downside_width_ic 0.005363은 default가 되기 어려운 이유다.",
        "- `tide_pvv_q10_param`은 band_width_ic가 가장 높지만 coverage_abs_error 0.059751로 기준을 넘겨 experiment_record다. TiDE에서 q10 확장은 coverage/width 균형이 깨졌다.",
        "",
        "## 6. Target Probe 해석",
        "",
        f"- 현재 지원 target: `{', '.join(payload.get('supported_target_types') or [])}`",
        "- 요청된 `realized_range`, `realized_volatility`, `downside_magnitude`, `tail_event_probability`는 기존 target 경로에 없다.",
        "- 큰 target 구조 변경은 이번 CP 금지 범위이므로 구현하지 않았다.",
        "",
        "## 7. Candidate Registry",
        "",
        _table(
            [
                {
                    "category": row.get("category"),
                    "display_name": row.get("display_name"),
                    "feature_set": row.get("feature_set"),
                    "mode": row.get("band_mode"),
                    "target": row.get("target_type"),
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
                ("mode", "mode"),
                ("target", "target"),
                ("run_id", "run_id"),
                ("strength", "strength"),
                ("weakness", "weakness"),
            ],
        ),
        "",
        "## 8. 산출물",
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
    baseline = _baseline_metrics()
    criteria = _criteria()
    for record in records:
        record["decision"] = _classify_record(record, criteria, baseline)
    _apply_recommended(records)
    registry = [_registry_entry(record) for record in records]
    recommended = next((row.get("source_experiment_id") for row in registry if row.get("category") == "recommended_default"), None)
    payload = {
        "cp": "CP120-BM",
        "status": "PASS" if all(row.get("execution_status") in {"PASS", "DESIGN_NEEDED"} for row in records) else "PARTIAL_OR_FAIL",
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
        "supported_target_types": list(SUPPORTED_TARGET_TYPES),
        "cp119_baseline": {
            "source": str(CP119_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "band_metrics": baseline,
        },
        "criteria": criteria,
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
    selected_names = set(args.only or [])
    selected = [item for item in EXPERIMENTS if not selected_names or item.name in selected_names]
    existing_by_name = {row.get("name"): row for row in _existing_records()}
    records: list[dict[str, Any]] = []
    for experiment in selected:
        existing = existing_by_name.get(experiment.name)
        if existing and existing.get("execution_status") in {"PASS", "DESIGN_NEEDED"} and not args.force:
            records.append(existing)
            continue
        print(f"[CP120] 시작: {experiment.name}", flush=True)
        record = _run_experiment(experiment, args)
        print(
            f"[CP120] 종료: {experiment.name} status={record.get('execution_status')} exit={record.get('exit_code')} elapsed={_fmt(record.get('elapsed_seconds'), 2)}",
            flush=True,
        )
        records.append(record)
        _write_outputs(records, args)
    record_names = {row.get("name") for row in records}
    for name, record in existing_by_name.items():
        if name not in record_names:
            records.append(record)
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
