from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPORT_PATH = PROJECT_ROOT / "docs" / "cp64_bm_feature_group_smoke_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp64_bm_feature_group_smoke_metrics.json"
LOG_DIR = PROJECT_ROOT / "docs" / "cp64_bm_feature_group_smoke_logs"
CP62_METRICS_PATH = PROJECT_ROOT / "docs" / "cp62_cp61_schema_candidate_regrade_metrics.json"

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

BASELINE_NAMES = [
    "rolling_bollinger_return_band_w60_k1",
    "rolling_historical_quantile_band_w252",
    "constant_width_train_quantile",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    feature_set: str
    q_low: float
    q_high: float
    band_mode: str
    lambda_band: float = 2.0
    seq_len: int = 60
    epochs: int = 3
    limit_tickers: int = 50
    batch_size: int = 256


EXPERIMENTS = [
    Experiment("cnn_s60_q15_b2_direct_price_volatility", "cnn_lstm", "price_volatility", 0.15, 0.85, "direct"),
    Experiment("cnn_s60_q20_b2_direct_price_volatility", "cnn_lstm", "price_volatility", 0.20, 0.80, "direct"),
    Experiment("cnn_s60_q15_b2_direct_price_volatility_volume", "cnn_lstm", "price_volatility_volume", 0.15, 0.85, "direct"),
    Experiment("cnn_s60_q15_b2_direct_no_fundamentals", "cnn_lstm", "no_fundamentals", 0.15, 0.85, "direct"),
    Experiment("tide_q10_b2_param_technical_only", "tide", "technical_only", 0.10, 0.90, "param"),
    Experiment("tide_q10_b2_direct_price_volatility", "tide", "price_volatility", 0.10, 0.90, "direct"),
]

FULL_FEATURE_REFERENCE_BY_NAME = {
    "cnn_s60_q15_b2_direct_price_volatility": "s60_q15_b2_direct",
    "cnn_s60_q20_b2_direct_price_volatility": "s60_q20_b2_direct",
    "cnn_s60_q15_b2_direct_price_volatility_volume": "s60_q15_b2_direct",
    "cnn_s60_q15_b2_direct_no_fundamentals": "s60_q15_b2_direct",
    "tide_q10_b2_param_technical_only": "tide_param_scalar_width",
    "tide_q10_b2_direct_price_volatility": "tide_direct_original",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _epoch_summary(stdout: str) -> dict[str, Any]:
    epoch_seconds: list[float] = []
    vram_peak_mb: list[float] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if "epoch_seconds" in payload:
            value = _safe_float(payload.get("epoch_seconds"))
            if value is not None:
                epoch_seconds.append(value)
        if "vram_peak_allocated_mb" in payload:
            value = _safe_float(payload.get("vram_peak_allocated_mb"))
            if value is not None:
                vram_peak_mb.append(value)
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "vram_peak_allocated_mb": max(vram_peak_mb) if vram_peak_mb else None,
    }


def _epoch_summary_from_log(record: dict[str, Any]) -> dict[str, Any]:
    log_path_value = record.get("log_path")
    if not log_path_value:
        return record.get("epoch_summary", {})
    log_path = PROJECT_ROOT / str(log_path_value)
    if not log_path.exists():
        return record.get("epoch_summary", {})
    return _epoch_summary(log_path.read_text(encoding="utf-8", errors="replace"))


def _command_for_experiment(experiment: Experiment, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        experiment.model,
        "--timeframe",
        "1D",
        "--horizon",
        "5",
        "--seq-len",
        str(experiment.seq_len),
        "--epochs",
        str(experiment.epochs),
        "--batch-size",
        str(experiment.batch_size),
        "--limit-tickers",
        str(experiment.limit_tickers),
        "--q-low",
        str(experiment.q_low),
        "--q-high",
        str(experiment.q_high),
        "--lambda-band",
        str(experiment.lambda_band),
        "--band-mode",
        experiment.band_mode,
        "--feature-set",
        experiment.feature_set,
        "--checkpoint-selection",
        "band_gate",
        "--line-target-type",
        "raw_future_return",
        "--band-target-type",
        "raw_future_return",
        "--device",
        args.device,
        "--num-workers",
        "0",
        "--no-wandb",
        "--no-compile",
        "--amp-dtype",
        args.amp_dtype,
        "--early-stop-patience",
        str(args.early_stop_patience),
    ]
    if experiment.model == "cnn_lstm":
        command.extend(["--fp32-modules", "lstm,heads"])
    if experiment.model == "tide":
        command.append("--use-future-covariate")
    return command


def _run_experiment(experiment: Experiment, args: argparse.Namespace) -> dict[str, Any]:
    command = _command_for_experiment(experiment, args)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{experiment.name}.log"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
        status = "PASS" if return_code == 0 else "FAIL"
        timeout_expired = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        return_code = None
        status = "TIMEOUT"
        timeout_expired = True
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "\n".join(
            [
                "# command: " + " ".join(command),
                "# stdout",
                stdout,
                "# stderr",
                stderr,
                f"# exit_code: {return_code}",
                f"# elapsed_seconds: {elapsed:.4f}",
                f"# timeout_expired: {timeout_expired}",
            ]
        ),
        encoding="utf-8",
    )
    record: dict[str, Any] = {
        "name": experiment.name,
        "experiment": experiment.__dict__,
        "command": command,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "exit_code": return_code,
        "elapsed_seconds": elapsed,
        "epoch_summary": _epoch_summary(stdout),
        "execution_status": status,
    }
    if status != "PASS":
        record["status"] = "failed"
        record["failed_with_reason"] = "timeout" if timeout_expired else "nonzero_exit"
        record["stderr_tail"] = stderr[-4000:]
        return record

    result = _extract_train_json(stdout)
    record["status"] = "completed"
    record["train_result"] = result
    record["band_metrics"] = _band_metrics(result.get("test_metrics", {}))
    record["validation_band_metrics"] = _band_metrics(result.get("best_metrics", {}))
    record["n_features"] = result.get("n_features")
    record["feature_columns"] = result.get("feature_columns")
    return record


def _band_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    source = metrics.get("band_metrics") if isinstance(metrics.get("band_metrics"), dict) else metrics
    return {key: source.get(key) for key in BAND_METRIC_KEYS}


def _comparison_references() -> dict[str, Any]:
    cp62 = _read_json(CP62_METRICS_PATH)
    full_features = {
        row["candidate_name"]: row
        for row in cp62.get("band_candidates", [])
        if row.get("candidate_name")
    }
    baselines = {
        row["name"]: row
        for row in cp62.get("band_baselines", [])
        if row.get("name") in BASELINE_NAMES
    }
    return {
        "source": str(CP62_METRICS_PATH.relative_to(PROJECT_ROOT)),
        "full_features": {
            name: {
                "candidate_name": name,
                "model": row.get("model"),
                "verdict": row.get("verdict"),
                "band_metrics": _band_metrics(row.get("band_metrics", {})),
            }
            for name, row in full_features.items()
        },
        "baselines": {
            name: {
                "name": name,
                "category": row.get("category"),
                "band_metrics": _band_metrics(row.get("band_metrics", {})),
            }
            for name, row in baselines.items()
        },
    }


def _metric_delta(current: dict[str, Any], reference: dict[str, Any], key: str) -> float | None:
    left = _safe_float(current.get(key))
    right = _safe_float(reference.get(key))
    if left is None or right is None:
        return None
    return left - right


def _best_baseline_metrics(references: dict[str, Any]) -> dict[str, Any]:
    baselines = list(references.get("baselines", {}).values())
    if not baselines:
        return {}
    return min(
        (row["band_metrics"] for row in baselines),
        key=lambda metrics: (
            _safe_float(metrics.get("asymmetric_interval_score")) or float("inf"),
            _safe_float(metrics.get("coverage_abs_error")) or float("inf"),
        ),
    )


def _baseline_win_summary(metrics: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    wins: list[str] = []
    best_cov_delta: float | None = None
    best_interval_delta: float | None = None
    for name, row in references.get("baselines", {}).items():
        baseline_metrics = row.get("band_metrics", {})
        cov_delta = _metric_delta(metrics, baseline_metrics, "coverage_abs_error")
        interval_delta = _metric_delta(metrics, baseline_metrics, "asymmetric_interval_score")
        if cov_delta is not None:
            best_cov_delta = cov_delta if best_cov_delta is None else min(best_cov_delta, cov_delta)
        if interval_delta is not None:
            best_interval_delta = interval_delta if best_interval_delta is None else min(best_interval_delta, interval_delta)
        if (cov_delta is not None and cov_delta < 0.0) or (interval_delta is not None and interval_delta < 0.0):
            wins.append(str(name))
    return {
        "baseline_win_names": wins,
        "baseline_win_count": len(wins),
        "best_coverage_abs_error_delta_vs_named_baselines": best_cov_delta,
        "best_asymmetric_interval_score_delta_vs_named_baselines": best_interval_delta,
        "baseline_outcome": "WIN" if wins else "LOSE",
    }


def _classify(record: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("band_metrics", {})
    full_name = FULL_FEATURE_REFERENCE_BY_NAME.get(record["name"])
    full_ref = references.get("full_features", {}).get(full_name, {})
    full_metrics = full_ref.get("band_metrics", {})
    baseline_metrics = _best_baseline_metrics(references)

    coverage_delta_baseline = _metric_delta(metrics, baseline_metrics, "coverage_abs_error")
    score_delta_baseline = _metric_delta(metrics, baseline_metrics, "asymmetric_interval_score")
    coverage_delta_full = _metric_delta(metrics, full_metrics, "coverage_abs_error")
    score_delta_full = _metric_delta(metrics, full_metrics, "asymmetric_interval_score")
    width_ic = _safe_float(metrics.get("band_width_ic"))
    downside_ic = _safe_float(metrics.get("downside_width_ic"))
    lower = _safe_float(metrics.get("lower_breach_rate"))
    upper = _safe_float(metrics.get("upper_breach_rate"))
    breach_imbalance = abs((lower or 0.0) - (upper or 0.0)) if lower is not None and upper is not None else None
    breach_skew_excessive = breach_imbalance is None or breach_imbalance > 0.15 or max(lower or 0.0, upper or 0.0) > 0.30
    baseline_improved = (
        coverage_delta_baseline is not None
        and coverage_delta_baseline < 0.0
    ) or (
        score_delta_baseline is not None
        and score_delta_baseline < 0.0
    )
    dynamic_positive = (width_ic is not None and width_ic > 0.0) or (downside_ic is not None and downside_ic > 0.0)
    full_improved = (
        coverage_delta_full is not None
        and coverage_delta_full < 0.0
    ) or (
        score_delta_full is not None
        and score_delta_full < 0.0
    )
    worse_than_full = (
        coverage_delta_full is not None
        and score_delta_full is not None
        and coverage_delta_full > 0.0
        and score_delta_full > 0.0
    )
    much_worse_than_baseline = (
        coverage_delta_baseline is not None
        and score_delta_baseline is not None
        and coverage_delta_baseline > 0.05
        and score_delta_baseline > 0.02
    )

    if baseline_improved and dynamic_positive and not breach_skew_excessive:
        verdict = "band_survive"
    elif (full_improved or dynamic_positive) and not worse_than_full:
        verdict = "band_watch"
    elif worse_than_full or much_worse_than_baseline:
        verdict = "band_fail"
    else:
        verdict = "band_watch"

    return {
        "full_feature_reference": full_name,
        **_baseline_win_summary(metrics, references),
        "coverage_abs_error_delta_vs_full": coverage_delta_full,
        "asymmetric_interval_score_delta_vs_full": score_delta_full,
        "coverage_abs_error_delta_vs_best_baseline": coverage_delta_baseline,
        "asymmetric_interval_score_delta_vs_best_baseline": score_delta_baseline,
        "breach_imbalance": breach_imbalance,
        "breach_skew_excessive": breach_skew_excessive,
        "dynamic_positive": dynamic_positive,
        "baseline_improved": baseline_improved,
        "full_features_improved": full_improved,
        "verdict": verdict,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _build_report(payload: dict[str, Any]) -> str:
    execution_rows = []
    count_rows_by_feature_set: dict[str, dict[str, Any]] = {}
    calibration_rows = []
    width_rows = []
    for record in payload["experiments"]:
        metrics = record.get("band_metrics", {})
        comparison = record.get("comparison", {})
        epoch_summary = record.get("epoch_summary", {})
        feature_set = record["experiment"]["feature_set"]
        count_rows_by_feature_set.setdefault(
            feature_set,
            {
                "feature_set": feature_set,
                "n_features": record.get("n_features", ""),
                "columns": ", ".join(record.get("feature_columns") or []),
            },
        )
        execution_rows.append(
            {
                "name": record["name"],
                "status": record.get("execution_status", record.get("status", "")),
                "exit": record.get("exit_code", ""),
                "seconds": _fmt(record.get("elapsed_seconds"), 1),
                "epoch_mean": _fmt(epoch_summary.get("epoch_seconds_mean"), 2),
                "epochs": ",".join(_fmt(value, 1) for value in epoch_summary.get("epoch_seconds", [])),
                "vram": _fmt(epoch_summary.get("vram_peak_allocated_mb"), 1),
                "reuse": record.get("skip_reason", ""),
                "log": record.get("log_path", ""),
            }
        )
        calibration_rows.append(
            {
                "name": record["name"],
                "model": record["experiment"]["model"],
                "feature_set": feature_set,
                "n_features": record.get("n_features", ""),
                "nominal": _fmt(metrics.get("nominal_coverage")),
                "empirical": _fmt(metrics.get("empirical_coverage")),
                "cov_err": _fmt(metrics.get("coverage_abs_error")),
                "lower": _fmt(metrics.get("lower_breach_rate")),
                "upper": _fmt(metrics.get("upper_breach_rate")),
                "verdict": comparison.get("verdict", record.get("execution_status", record.get("status"))),
            }
        )
        width_rows.append(
            {
                "name": record["name"],
                "avg_width": _fmt(metrics.get("avg_band_width")),
                "median_width": _fmt(metrics.get("median_band_width")),
                "p90_width": _fmt(metrics.get("p90_band_width")),
                "interval": _fmt(metrics.get("asymmetric_interval_score")),
                "lower_penalty": _fmt(metrics.get("interval_lower_penalty")),
                "upper_penalty": _fmt(metrics.get("interval_upper_penalty")),
                "width_ic": _fmt(metrics.get("band_width_ic")),
                "downside_ic": _fmt(metrics.get("downside_width_ic")),
                "vol_ratio": _fmt(metrics.get("width_bucket_realized_vol_ratio")),
                "downside_ratio": _fmt(metrics.get("width_bucket_downside_rate_ratio")),
                "squeeze": _fmt(metrics.get("squeeze_breakout_rate")),
            }
        )

    baseline_rows = []
    for name, row in payload["references"]["baselines"].items():
        metrics = row["band_metrics"]
        baseline_rows.append(
            {
                "name": name,
                "cov_err": _fmt(metrics.get("coverage_abs_error")),
                "interval": _fmt(metrics.get("asymmetric_interval_score")),
                "width_ic": _fmt(metrics.get("band_width_ic")),
                "downside_ic": _fmt(metrics.get("downside_width_ic")),
            }
        )

    full_rows = []
    baseline_outcome_rows = []
    for record in payload["experiments"]:
        comparison = record.get("comparison", {})
        full_name = comparison.get("full_feature_reference")
        full_ref = payload["references"]["full_features"].get(full_name, {})
        full_metrics = full_ref.get("band_metrics", {})
        full_rows.append(
            {
                "name": record["name"],
                "full": full_name,
                "improved": "YES" if comparison.get("full_features_improved") else "NO",
                "cov_delta": _fmt(comparison.get("coverage_abs_error_delta_vs_full")),
                "score_delta": _fmt(comparison.get("asymmetric_interval_score_delta_vs_full")),
                "full_cov_err": _fmt(full_metrics.get("coverage_abs_error")),
                "full_score": _fmt(full_metrics.get("asymmetric_interval_score")),
            }
        )
        baseline_outcome_rows.append(
            {
                "name": record["name"],
                "outcome": comparison.get("baseline_outcome", ""),
                "wins": ", ".join(comparison.get("baseline_win_names", [])),
                "best_cov_delta": _fmt(comparison.get("best_coverage_abs_error_delta_vs_named_baselines")),
                "best_interval_delta": _fmt(comparison.get("best_asymmetric_interval_score_delta_vs_named_baselines")),
            }
        )

    verdict_counts: dict[str, int] = {}
    for record in payload["experiments"]:
        verdict = record.get("comparison", {}).get("verdict", record.get("execution_status", record.get("status", "unknown")))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    cnn_records = [row for row in payload["experiments"] if row["experiment"]["model"] == "cnn_lstm"]
    tide_records = [row for row in payload["experiments"] if row["experiment"]["model"] == "tide"]
    completed_cnn_records = [row for row in cnn_records if row.get("status") == "completed"]
    completed_tide_records = [row for row in tide_records if row.get("status") == "completed"]
    cnn_best = min(
        completed_cnn_records,
        key=lambda row: _safe_float(row.get("band_metrics", {}).get("asymmetric_interval_score")) or float("inf"),
    ) if completed_cnn_records else None
    tide_best = min(
        completed_tide_records,
        key=lambda row: _safe_float(row.get("band_metrics", {}).get("asymmetric_interval_score")) or float("inf"),
    ) if completed_tide_records else None
    cnn_best_name = cnn_best["name"] if cnn_best is not None else "완료 후보 없음"
    tide_best_name = tide_best["name"] if tide_best is not None else "완료 후보 없음"
    feature_count_rows = list(count_rows_by_feature_set.values())

    return f"""# CP64-BM CNN-LSTM/TiDE band feature group smoke

## 1. 실행 계약

이번 CP는 AI band layer 전용 smoke다. line 실험, composite/overlay 지표, 통계 baseline residual/scale 보정은 수행하지 않았다. `ai.train`에는 `--feature-set`과 alias `--feature-columns-preset`을 추가했고, CP63의 `docs/cp63_bm_feature_set_plan.json`에서 feature set을 로드한다. `full_features`는 기존 36개 기본 동작을 유지한다.

feature set 적용 경로는 다음과 같다.

1. CLI에서 `--feature-set`을 수신한다.
2. CP63 JSON에서 columns를 로드한다.
3. `MODEL_FEATURE_COLUMNS`의 부분집합인지 검증한다.
4. indicator-only 또는 contract 변경 필요 set은 차단한다.
5. train/val/test feature tensor와 mean/std를 같은 순서로 축소한다.
6. `TrainConfig.n_features`를 모델 생성에 반영한다.

공통 조건은 `feature_version=v3_adjusted_ohlc`, `role=band_model`, `band_target_type=raw_future_return`, `checkpoint_selection=band_gate`, `limit_tickers=50`, `epochs=3`, `batch_size=256`, `save-run=false`, `wandb=off`다.

## 2. feature_set별 실제 column 수

{_table(feature_count_rows, [("feature_set", "feature_set"), ("n_features", "n_features"), ("columns", "columns")])}

## 3. 실행 상태

기존 완료 결과가 있는 실험은 재실행하지 않고 재사용했다. PASS는 해당 실험 학습 명령이 exit code 0으로 끝났다는 뜻이며, SKIPPED가 있으면 이미 완료된 결과를 재사용했다는 뜻이다.

{_table(execution_rows, [("실험", "name"), ("상태", "status"), ("exit", "exit"), ("초", "seconds"), ("epoch 평균", "epoch_mean"), ("epoch 초", "epochs"), ("VRAM MB", "vram"), ("재사용", "reuse"), ("로그", "log")])}

## 4. band_metrics: coverage와 breach

{_table(calibration_rows, [("실험", "name"), ("모델", "model"), ("feature_set", "feature_set"), ("n", "n_features"), ("nominal", "nominal"), ("empirical", "empirical"), ("cov err", "cov_err"), ("lower breach", "lower"), ("upper breach", "upper"), ("판정", "verdict")])}

## 5. band_metrics: width와 dynamic signal

{_table(width_rows, [("실험", "name"), ("avg width", "avg_width"), ("median width", "median_width"), ("p90 width", "p90_width"), ("interval", "interval"), ("lower penalty", "lower_penalty"), ("upper penalty", "upper_penalty"), ("width IC", "width_ic"), ("downside IC", "downside_ic"), ("vol ratio", "vol_ratio"), ("downside ratio", "downside_ratio"), ("squeeze", "squeeze")])}

## 6. full_features 대비

delta는 CP64 feature set smoke에서 기존 full_features 후보 값을 뺀 값이다. `coverage_abs_error`와 `asymmetric_interval_score`는 낮을수록 좋다.

{_table(full_rows, [("실험", "name"), ("full 기준", "full"), ("개선", "improved"), ("cov err delta", "cov_delta"), ("interval delta", "score_delta"), ("full cov err", "full_cov_err"), ("full interval", "full_score")])}

## 7. 통계 baseline 대비

비교 baseline은 CP62 산출물의 `rolling_bollinger_return_band_w60_k1`, `rolling_historical_quantile_band_w252`, `constant_width_train_quantile`만 사용했다. 통계 baseline은 비교 기준이며 보정 모델로 쓰지 않았다.

{_table(baseline_rows, [("baseline", "name"), ("cov err", "cov_err"), ("interval", "interval"), ("width IC", "width_ic"), ("downside IC", "downside_ic")])}

실험별 baseline 승패는 다음과 같다. WIN은 지정 baseline 중 하나 이상에서 `coverage_abs_error` 또는 `asymmetric_interval_score`가 개선됐다는 뜻이다.

{_table(baseline_outcome_rows, [("실험", "name"), ("승패", "outcome"), ("이긴 baseline", "wins"), ("최선 cov delta", "best_cov_delta"), ("최선 interval delta", "best_interval_delta")])}

판정 집계는 `{verdict_counts}`다.

## 8. BM 우선순위 재판정

CNN-LSTM 최선 smoke는 `{cnn_best_name}`이고, TiDE 최선 smoke는 `{tide_best_name}`다. 이번 smoke 기준으로도 BM 후보 우선순위는 CNN-LSTM 1순위, TiDE 2순위를 유지한다. 다만 survive 판정은 smoke 조건에서의 생존권이며 제품 band 확정이 아니다. PatchTST band는 이번 CP에서 참고 후순위로 유지한다.

## 9. ATR/daily range feature 승격 여부

`atr_ratio`와 `intraday_range_ratio`는 이번 smoke에서 모델 feature로 추가하지 않았다. CP63 proxy상 band에 유리할 가능성은 있으므로 승격 검토 가치는 있다. 다만 승격하려면 feature contract, cache digest, feature_version, checkpoint 호환성 분리를 함께 바꿔야 하므로 CP64-BM에서는 승격 금지로 닫고, 별도 feature contract CP에서 다룬다.

## 10. 다음 CP 추천

- CNN-LSTM은 `price_volatility`와 `price_volatility_volume`을 중심으로 q/lambda를 좁게 재탐색한다.
- `no_fundamentals`는 dynamic width가 양수지만 interval_score가 약하므로 fundamentals 제거는 보조 가설로만 둔다.
- TiDE는 `technical_only` param을 watch/survive 경계 후보로 두고, direct는 구조 개선 없이는 우선순위를 낮춘다.
- baseline-aware 보정은 Phase 1.5 후보로만 기록한다.
- 다음 순서는 CP64-D indicator full backfill이다.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    existing_records: dict[str, dict[str, Any]] = {}
    if METRICS_PATH.exists():
        existing_payload = _read_json(METRICS_PATH)
        existing_records = {
            str(record.get("name")): record
            for record in existing_payload.get("experiments", [])
            if record.get("name")
        }
    requested = set(args.only or [experiment.name for experiment in EXPERIMENTS])
    records = []
    for experiment in EXPERIMENTS:
        if experiment.name not in requested:
            if experiment.name in existing_records:
                records.append(existing_records[experiment.name])
            continue
        if experiment.name in existing_records:
            existing = existing_records[experiment.name]
            existing["execution_status"] = existing.get("execution_status", "SKIPPED")
            existing["skip_reason"] = "existing_result_reused"
            records.append(existing)
            print(json.dumps({"cp64": "skip_existing", "experiment": experiment.name}, ensure_ascii=False), flush=True)
            continue
        print(json.dumps({"cp64": "start", "experiment": experiment.name}, ensure_ascii=False), flush=True)
        record = _run_experiment(experiment, args)
        print(
            json.dumps(
                {
                    "cp64": "done",
                    "experiment": experiment.name,
                    "status": record.get("status"),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        records.append(record)

    references = _comparison_references()
    for record in records:
        record["epoch_summary"] = _epoch_summary_from_log(record)
        if record.get("status") == "completed":
            record["comparison"] = _classify(record, references)

    payload = {
        "cp": "CP64-BM",
        "rules": {
            "full_473_training": False,
            "save_run": False,
            "db_write": False,
            "db_schema_change": False,
            "frontend_ui_backend_change": False,
            "line_metrics_used_for_band_verdict": False,
            "composite_overlay_metrics_used": False,
            "statistical_baseline_residual_scale_model": False,
            "atr_intraday_range_as_model_feature": False,
        },
        "feature_set_cli": {
            "argument": "--feature-set",
            "alias": "--feature-columns-preset",
            "source_plan": "docs/cp63_bm_feature_set_plan.json",
            "default": "full_features",
        },
        "experiments": records,
        "references": references,
    }
    _write_json(METRICS_PATH, payload)
    REPORT_PATH.write_text(_build_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP64 BM feature group smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--early-stop-patience", type=int, default=-1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--only", nargs="*", choices=[experiment.name for experiment in EXPERIMENTS], default=None)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "cp": payload["cp"],
                "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
                "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
                "experiment_count": len(payload["experiments"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
