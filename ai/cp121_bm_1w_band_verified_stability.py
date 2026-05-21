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

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402

from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.inference import _select_bundle_features, load_checkpoint, resolve_bundle, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.models.tide import TiDE  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.train import make_loader, resolve_device  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp121_bm_1w_band_verified_stability_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp121_bm_1w_band_verified_stability_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp121_bm_1w_band_verified_registry.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp121_bm_1w_band_verified_stability_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp121_bm_1w_band_verified_stability_logs"

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

STABILITY_KEYS = [
    "coverage_abs_error",
    "lower_breach_rate",
    "asymmetric_interval_score",
    "band_width_ic",
    "downside_width_ic",
    "p90_band_width",
]

REGIME_NAMES = [
    "high_volatility",
    "low_volatility",
    "rising",
    "falling",
    "wide_band",
    "narrow_band",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    feature_set: str
    q_low: float
    q_high: float
    band_mode: str
    note: str


CANDIDATES = [
    Candidate("cnn_pvv_q10_direct", "cnn_lstm", "price_volatility_volume", 0.10, 0.90, "direct", "CP119/CP120 recommended_default"),
    Candidate("cnn_full_q10_direct", "cnn_lstm", "full_features", 0.10, 0.90, "direct", "CP119/CP120 selectable_verified"),
    Candidate("tide_pvv_q15_param", "tide", "price_volatility_volume", 0.15, 0.85, "param", "CP120 selectable_verified TiDE 대안"),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
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
    vram_values: list[float] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        epoch_seconds_value = _safe_float(payload.get("epoch_seconds"))
        if epoch_seconds_value is not None:
            epoch_seconds.append(epoch_seconds_value)
        vram_value = _safe_float(payload.get("vram_peak_allocated_mb"))
        if vram_value is not None:
            vram_values.append(vram_value)
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "vram_peak_allocated_mb": max(vram_values) if vram_values else None,
    }


def _command_for_run(candidate: Candidate, seed: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        candidate.model,
        "--timeframe",
        "1W",
        "--horizon",
        "4",
        "--seq-len",
        "104",
        "--feature-set",
        candidate.feature_set,
        "--line-target-type",
        "raw_future_return",
        "--band-target-type",
        "raw_future_return",
        "--q-low",
        str(candidate.q_low),
        "--q-high",
        str(candidate.q_high),
        "--lambda-band",
        "2.0",
        "--band-mode",
        candidate.band_mode,
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
        str(LOG_DIR / "local_runs" / candidate.name / f"seed_{seed}"),
    ]
    if candidate.model == "cnn_lstm":
        command.extend(["--fp32-modules", "lstm,heads"])
    if candidate.model == "tide":
        command.append("--use-future-covariate")
    return command


def _run_training(candidate: Candidate, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{candidate.name}_seed{seed}.stdout.log"
    command = _command_for_run(candidate, seed, args)
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
        "candidate": candidate.name,
        "seed": seed,
        "candidate_config": asdict(candidate),
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


def _slice_metrics(
    *,
    name: str,
    mask: torch.Tensor,
    metadata: Any,
    line_predictions: torch.Tensor,
    lower_predictions: torch.Tensor,
    upper_predictions: torch.Tensor,
    line_targets: torch.Tensor,
    band_targets: torch.Tensor,
    raw_future_returns: torch.Tensor,
    q_low: float,
    q_high: float,
    severe_downside_threshold: float | None,
    squeeze_breakout_threshold: float | None,
) -> dict[str, Any]:
    indices = mask.nonzero(as_tuple=False).flatten()
    sample_count = int(indices.numel())
    if sample_count < 10:
        return {"sample_count": sample_count, "band_metrics": {}}
    metadata_slice = metadata.iloc[indices.detach().cpu().numpy()].copy() if metadata is not None else None
    metrics = summarize_forecast_metrics(
        metadata=metadata_slice,
        line_predictions=line_predictions.index_select(0, indices),
        lower_predictions=lower_predictions.index_select(0, indices),
        upper_predictions=upper_predictions.index_select(0, indices),
        line_targets=line_targets.index_select(0, indices),
        band_targets=band_targets.index_select(0, indices),
        raw_future_returns=raw_future_returns.index_select(0, indices),
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        q_low=q_low,
        q_high=q_high,
        severe_downside_threshold=severe_downside_threshold,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
    )
    return {
        "sample_count": sample_count,
        "band_metrics": _band_metrics(metrics),
    }


def _validation_regime_metrics(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = record.get("checkpoint_path")
    if not checkpoint_path:
        return {"error": "checkpoint_missing"}
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()
    registry = resolve_checkpoint_ticker_registry(config, "1W")
    bundle = resolve_bundle(
        split_name="val",
        timeframe="1W",
        seq_len=int(config.get("seq_len") or 104),
        horizon=int(config.get("horizon") or 4),
        tickers=config.get("tickers"),
        limit_tickers=config.get("limit_tickers"),
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        ticker_registry=registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    bundle = _select_bundle_features(bundle, list(config.get("feature_columns") or []))
    loader = make_loader(bundle, batch_size=int(config.get("batch_size") or args.batch_size), shuffle=False, device=device, num_workers=0)
    metadata = bundle.metadata.reset_index(drop=True)
    line_predictions: list[torch.Tensor] = []
    lower_predictions: list[torch.Tensor] = []
    upper_predictions: list[torch.Tensor] = []
    line_targets: list[torch.Tensor] = []
    band_targets: list[torch.Tensor] = []
    raw_targets: list[torch.Tensor] = []
    use_future_covariate = bool(config.get("use_future_covariate", config.get("model") == "tide"))
    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_ids, future_covariates in loader:
            features = features.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            if isinstance(getattr(model, "_orig_mod", model), TiDE) and use_future_covariate:
                output = model(features, ticker_id=ticker_ids, future_covariate=future_covariates)
            else:
                output = model(features, ticker_id=ticker_ids)
            line_pred, lower_pred, upper_pred = apply_band_postprocess(
                output.line.detach().cpu(),
                output.lower_band.detach().cpu(),
                output.upper_band.detach().cpu(),
            )
            line_predictions.append(line_pred)
            lower_predictions.append(lower_pred)
            upper_predictions.append(upper_pred)
            line_targets.append(line_target.detach().cpu())
            band_targets.append(band_target.detach().cpu())
            raw_targets.append(raw_future_returns.detach().cpu())
    line_tensor = torch.cat(line_predictions, dim=0)
    lower_tensor = torch.cat(lower_predictions, dim=0)
    upper_tensor = torch.cat(upper_predictions, dim=0)
    line_target_tensor = torch.cat(line_targets, dim=0)
    band_target_tensor = torch.cat(band_targets, dim=0)
    raw_tensor = torch.cat(raw_targets, dim=0)
    width = (upper_tensor - lower_tensor).mean(dim=1)
    realized_volatility = raw_tensor.std(dim=1, unbiased=False)
    realized_h4_return = raw_tensor[:, -1]
    high_vol_threshold = torch.median(realized_volatility)
    width_threshold = torch.median(width)
    masks = {
        "high_volatility": realized_volatility >= high_vol_threshold,
        "low_volatility": realized_volatility < high_vol_threshold,
        "rising": realized_h4_return >= 0.0,
        "falling": realized_h4_return < 0.0,
        "wide_band": width >= width_threshold,
        "narrow_band": width < width_threshold,
    }
    q_low = float(config.get("q_low") or (record.get("candidate_config") or {}).get("q_low") or 0.1)
    q_high = float(config.get("q_high") or (record.get("candidate_config") or {}).get("q_high") or 0.9)
    regimes = {
        name: _slice_metrics(
            name=name,
            mask=mask,
            metadata=metadata,
            line_predictions=line_tensor,
            lower_predictions=lower_tensor,
            upper_predictions=upper_tensor,
            line_targets=line_target_tensor,
            band_targets=band_target_tensor,
            raw_future_returns=raw_tensor,
            q_low=q_low,
            q_high=q_high,
            severe_downside_threshold=config.get("severe_downside_threshold"),
            squeeze_breakout_threshold=config.get("squeeze_breakout_threshold"),
        )
        for name, mask in masks.items()
    }
    return {
        "split": "val",
        "sample_count": int(raw_tensor.shape[0]),
        "regime_thresholds": {
            "realized_volatility_median": float(high_vol_threshold.item()),
            "band_width_median": float(width_threshold.item()),
        },
        "regimes": regimes,
    }


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _aggregate_candidate(candidate: Candidate, runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("execution_status") == "PASS"]
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
        "candidate": candidate.name,
        "candidate_config": asdict(candidate),
        "runs": runs,
        "run_count": len(runs),
        "completed_run_count": len(completed),
        "band_gate_pass_count": gate_pass_count,
        "band_gate_pass_rate": gate_pass_count / len(completed) if completed else 0.0,
        "validation_metric_stats": metric_stats,
        "validation_regime_metric_stats": regime_stats,
        "validation_regime_warnings": _regime_warnings(regime_stats),
        "test_exposure_count": test_exposure_count,
        "test_metric_policy": "ai.train 결과 JSON에 test_metrics가 포함되어 read-only count만 기록했고 후보 선정에는 사용하지 않음",
    }


def _aggregate_regime_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime_name in REGIME_NAMES:
        regime_result: dict[str, Any] = {}
        sample_counts: list[float] = []
        for key in STABILITY_KEYS:
            values: list[float] = []
            for run in runs:
                regime_payload = (((run.get("validation_regime_metrics") or {}).get("regimes") or {}).get(regime_name) or {})
                sample_count = _safe_float(regime_payload.get("sample_count"))
                if sample_count is not None:
                    sample_counts.append(sample_count)
                metrics = regime_payload.get("band_metrics") or {}
                value = _safe_float(metrics.get(key))
                if value is not None:
                    values.append(value)
            regime_result[key] = _mean_std(values)
        regime_result["sample_count"] = _mean_std(sample_counts)
        result[regime_name] = regime_result
    return result


def _regime_warnings(regime_stats: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    max_cov: tuple[str, float] | None = None
    max_lower: tuple[str, float] | None = None
    min_bwic: tuple[str, float] | None = None
    for regime_name, stats in regime_stats.items():
        cov = _safe_float((stats.get("coverage_abs_error") or {}).get("mean"))
        lower = _safe_float((stats.get("lower_breach_rate") or {}).get("mean"))
        bwic = _safe_float((stats.get("band_width_ic") or {}).get("mean"))
        if cov is not None and (max_cov is None or cov > max_cov[1]):
            max_cov = (regime_name, cov)
        if lower is not None and (max_lower is None or lower > max_lower[1]):
            max_lower = (regime_name, lower)
        if bwic is not None and (min_bwic is None or bwic < min_bwic[1]):
            min_bwic = (regime_name, bwic)
    if max_cov is not None and max_cov[1] > 0.10:
        warnings.append(f"{max_cov[0]} coverage_abs_error={max_cov[1]:.6f}")
    if max_lower is not None and max_lower[1] > 0.20:
        warnings.append(f"{max_lower[0]} lower_breach_rate={max_lower[1]:.6f}")
    if min_bwic is not None and min_bwic[1] < 0.10:
        warnings.append(f"{min_bwic[0]} band_width_ic={min_bwic[1]:.6f}")
    return warnings


def _criteria() -> dict[str, Any]:
    return {
        "band_gate_pass_rate_min": 1.0,
        "validation_coverage_abs_error_mean_max": 0.05,
        "validation_lower_breach_rate_mean_max": 0.18,
        "validation_asymmetric_interval_score_std_max": 0.05,
        "validation_asymmetric_interval_score_range_max": 0.10,
        "validation_band_width_ic_mean_min": 0.15,
        "validation_downside_width_ic_mean_min": 0.0,
    }


def _classify_candidate(summary: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    stats = summary.get("validation_metric_stats") or {}
    interval_stats = stats.get("asymmetric_interval_score") or {}
    interval_range = None
    if interval_stats.get("min") is not None and interval_stats.get("max") is not None:
        interval_range = float(interval_stats["max"]) - float(interval_stats["min"])
    checks = {
        "band_gate_mostly_pass": float(summary.get("band_gate_pass_rate") or 0.0) >= criteria["band_gate_pass_rate_min"],
        "validation_coverage_abs_error": _safe_float((stats.get("coverage_abs_error") or {}).get("mean")) is not None
        and float((stats.get("coverage_abs_error") or {}).get("mean")) <= criteria["validation_coverage_abs_error_mean_max"],
        "validation_lower_breach_rate": _safe_float((stats.get("lower_breach_rate") or {}).get("mean")) is not None
        and float((stats.get("lower_breach_rate") or {}).get("mean")) <= criteria["validation_lower_breach_rate_mean_max"],
        "validation_interval_stability": (
            _safe_float(interval_stats.get("std")) is not None
            and float(interval_stats["std"]) <= criteria["validation_asymmetric_interval_score_std_max"]
            and interval_range is not None
            and interval_range <= criteria["validation_asymmetric_interval_score_range_max"]
        ),
        "validation_band_width_ic": _safe_float((stats.get("band_width_ic") or {}).get("mean")) is not None
        and float((stats.get("band_width_ic") or {}).get("mean")) > criteria["validation_band_width_ic_mean_min"],
        "validation_downside_width_ic": _safe_float((stats.get("downside_width_ic") or {}).get("mean")) is not None
        and float((stats.get("downside_width_ic") or {}).get("mean")) >= criteria["validation_downside_width_ic_mean_min"],
    }
    failures = [key for key, ok in checks.items() if not ok]
    if not failures:
        category = "selectable_verified"
    elif any(key in failures for key in ("band_gate_mostly_pass", "validation_interval_stability")) and len(failures) <= 2:
        category = "unstable_verified"
    elif summary.get("completed_run_count", 0) > 0:
        category = "experiment_record"
    else:
        category = "rejected"
    return {
        "category": category,
        "checks": checks,
        "failures": failures,
        "interval_score_range": interval_range,
    }


def _select_default(candidate_summaries: list[dict[str, Any]]) -> str | None:
    verified = [item for item in candidate_summaries if (item.get("decision") or {}).get("category") == "selectable_verified"]
    if not verified:
        return None

    def priority(item: dict[str, Any]) -> int:
        name = str(item.get("candidate") or "")
        if name == "cnn_pvv_q10_direct":
            return 0
        if name == "cnn_full_q10_direct":
            return 1
        if name == "tide_pvv_q15_param":
            return 2
        return 3

    return sorted(
        verified,
        key=lambda item: (
            float(((item.get("validation_metric_stats") or {}).get("coverage_abs_error") or {}).get("mean") or float("inf")),
            float(((item.get("validation_metric_stats") or {}).get("asymmetric_interval_score") or {}).get("mean") or float("inf")),
            -float(((item.get("validation_metric_stats") or {}).get("band_width_ic") or {}).get("mean") or -999.0),
            priority(item),
        ),
    )[0].get("candidate")


def _apply_default(candidate_summaries: list[dict[str, Any]]) -> None:
    selected = _select_default(candidate_summaries)
    for item in candidate_summaries:
        decision = item.setdefault("decision", {})
        if decision.get("category") == "selectable_verified" and item.get("candidate") == selected:
            decision["category"] = "recommended_default"


def _registry_entry(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("candidate_config") or {}
    stats = summary.get("validation_metric_stats") or {}
    decision = summary.get("decision") or {}
    category = decision.get("category") or "experiment_record"
    weakness = "검증 기준 통과" if category in {"recommended_default", "selectable_verified"} else ", ".join(decision.get("failures") or ["검증 기준 미달"])
    regime_warnings = summary.get("validation_regime_warnings") or []
    if regime_warnings:
        weakness = (weakness + "; regime warning: " + ", ".join(regime_warnings)).strip()
    return {
        "category": category,
        "display_name": f"1W {config.get('model')} {summary.get('candidate')}",
        "role": "band_model",
        "timeframe": "1W",
        "horizon": 4,
        "model_family": config.get("model"),
        "feature_set": config.get("feature_set"),
        "target_type": "raw_future_return",
        "band_mode": config.get("band_mode"),
        "strength_summary": (
            f"val cov_abs mean={_fmt((stats.get('coverage_abs_error') or {}).get('mean'))}, "
            f"val interval mean={_fmt((stats.get('asymmetric_interval_score') or {}).get('mean'))}, "
            f"val bw_ic mean={_fmt((stats.get('band_width_ic') or {}).get('mean'))}"
        ),
        "weakness_summary": weakness,
        "best_use_case": _best_use_case(summary),
        "why_not_default": "현재 기본값" if category == "recommended_default" else weakness,
        "key_metrics": {
            key: {
                "validation_mean": (stats.get(key) or {}).get("mean"),
                "validation_std": (stats.get(key) or {}).get("std"),
            }
            for key in STABILITY_KEYS
        },
        "seed_count": summary.get("run_count"),
        "band_gate_pass_rate": summary.get("band_gate_pass_rate"),
        "test_exposure_count": summary.get("test_exposure_count"),
        "test_metric_policy": summary.get("test_metric_policy"),
        "raw_run_ids": [run.get("run_id") for run in summary.get("runs", []) if run.get("run_id")],
        "source_experiment_id": summary.get("candidate"),
    }


def _best_use_case(summary: dict[str, Any]) -> str:
    name = str(summary.get("candidate") or "")
    if name == "cnn_full_q10_direct":
        return "PVV보다 넓은 feature context가 필요한 1W band 대안"
    if name == "tide_pvv_q15_param":
        return "좁은 폭과 높은 dynamic width가 필요한 1W band 대안"
    return "기본 1W AI band 후보"


def _summary_rows(candidate_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidate_summaries:
        stats = item.get("validation_metric_stats") or {}
        decision = item.get("decision") or {}
        rows.append(
            {
                "candidate": item.get("candidate"),
                "category": decision.get("category"),
                "run_count": item.get("run_count"),
                "band_gate_pass_count": item.get("band_gate_pass_count"),
                "band_gate_pass_rate": item.get("band_gate_pass_rate"),
                "test_exposure_count": item.get("test_exposure_count"),
                "coverage_abs_error_mean": (stats.get("coverage_abs_error") or {}).get("mean"),
                "coverage_abs_error_std": (stats.get("coverage_abs_error") or {}).get("std"),
                "lower_breach_rate_mean": (stats.get("lower_breach_rate") or {}).get("mean"),
                "asymmetric_interval_score_mean": (stats.get("asymmetric_interval_score") or {}).get("mean"),
                "asymmetric_interval_score_std": (stats.get("asymmetric_interval_score") or {}).get("std"),
                "band_width_ic_mean": (stats.get("band_width_ic") or {}).get("mean"),
                "downside_width_ic_mean": (stats.get("downside_width_ic") or {}).get("mean"),
                "p90_band_width_mean": (stats.get("p90_band_width") or {}).get("mean"),
                "failures": ",".join(decision.get("failures") or []),
                "regime_warnings": ",".join(item.get("validation_regime_warnings") or []),
            }
        )
    return rows


def _write_summary_csv(candidate_summaries: list[dict[str, Any]]) -> None:
    rows = _summary_rows(candidate_summaries)
    fieldnames = [
        "candidate",
        "category",
        "run_count",
        "band_gate_pass_count",
        "band_gate_pass_rate",
        "test_exposure_count",
        "coverage_abs_error_mean",
        "coverage_abs_error_std",
        "lower_breach_rate_mean",
        "asymmetric_interval_score_mean",
        "asymmetric_interval_score_std",
        "band_width_ic_mean",
        "downside_width_ic_mean",
        "p90_band_width_mean",
        "failures",
        "regime_warnings",
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


def _run_rows(candidate_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidate_summaries:
        for run in item.get("runs", []):
            metrics = run.get("validation_band_metrics") or {}
            rows.append(
                {
                    "candidate": item.get("candidate"),
                    "seed": run.get("seed"),
                    "status": run.get("execution_status"),
                    "gate": (run.get("gate") or {}).get("band_gate_pass"),
                    "run_id": run.get("run_id"),
                    "cov_abs": metrics.get("coverage_abs_error"),
                    "lower": metrics.get("lower_breach_rate"),
                    "interval": metrics.get("asymmetric_interval_score"),
                    "bw_ic": metrics.get("band_width_ic"),
                    "down_ic": metrics.get("downside_width_ic"),
                    "p90_w": metrics.get("p90_band_width"),
                }
            )
    return rows


def _regime_rows(candidate_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidate_summaries:
        regime_stats = item.get("validation_regime_metric_stats") or {}
        for regime_name in REGIME_NAMES:
            stats = regime_stats.get(regime_name) or {}
            rows.append(
                {
                    "candidate": item.get("candidate"),
                    "regime": regime_name,
                    "samples": (stats.get("sample_count") or {}).get("mean"),
                    "cov_abs": (stats.get("coverage_abs_error") or {}).get("mean"),
                    "lower": (stats.get("lower_breach_rate") or {}).get("mean"),
                    "interval": (stats.get("asymmetric_interval_score") or {}).get("mean"),
                    "bw_ic": (stats.get("band_width_ic") or {}).get("mean"),
                    "down_ic": (stats.get("downside_width_ic") or {}).get("mean"),
                    "p90_w": (stats.get("p90_band_width") or {}).get("mean"),
                }
            )
    return rows


def _write_report(payload: dict[str, Any]) -> None:
    summaries = payload.get("candidate_summaries") or []
    registry = payload.get("candidate_registry") or []
    report = [
        "# CP121-BM 1W Band Verified 후보 안정성 검증",
        "",
        "## 1. 결론",
        "",
        f"- 상태: {payload.get('status')}",
        "- save-run, DB write, inference 저장, W&B, composite, Supabase 대량 read, 프론트 수정은 수행하지 않았다.",
        "- 후보 선정은 validation metric과 validation regime metric만 사용했다.",
        "- `ai.train` 결과 JSON 특성상 test_metrics는 생성되었지만 read-only count만 기록했고 후보 선정에는 쓰지 않았다.",
        f"- recommended_default: `{payload.get('recommended_default')}`",
        "",
        "## 2. 후보별 안정성 요약",
        "",
        _table(
            _summary_rows(summaries),
            [
                ("candidate", "candidate"),
                ("category", "category"),
                ("runs", "run_count"),
                ("gate", "band_gate_pass_rate"),
                ("test_exposure", "test_exposure_count"),
                ("cov_mean", "coverage_abs_error_mean"),
                ("cov_std", "coverage_abs_error_std"),
                ("lower_mean", "lower_breach_rate_mean"),
                ("interval_mean", "asymmetric_interval_score_mean"),
                ("interval_std", "asymmetric_interval_score_std"),
                ("bw_ic_mean", "band_width_ic_mean"),
                ("down_ic_mean", "downside_width_ic_mean"),
                ("p90_mean", "p90_band_width_mean"),
                ("failures", "failures"),
                ("regime_warnings", "regime_warnings"),
            ],
        ),
        "",
        "## 3. Seed별 Validation 결과",
        "",
        _table(
            _run_rows(summaries),
            [
                ("candidate", "candidate"),
                ("seed", "seed"),
                ("status", "status"),
                ("gate", "gate"),
                ("run_id", "run_id"),
                ("cov_abs", "cov_abs"),
                ("lower", "lower"),
                ("interval", "interval"),
                ("bw_ic", "bw_ic"),
                ("down_ic", "down_ic"),
                ("p90_w", "p90_w"),
            ],
        ),
        "",
        "## 4. Regime 평가 방식",
        "",
        "- high/low volatility: validation raw h4 realized volatility의 median split.",
        "- rising/falling: validation h4 최종 raw return 부호.",
        "- wide/narrow band: 예측 band 평균 폭의 median split.",
        "- regime metric은 후보 선정 보조 안정성 확인용이며 test set은 사용하지 않았다.",
        "",
        "## 5. Regime별 Validation 평균",
        "",
        _table(
            _regime_rows(summaries),
            [
                ("candidate", "candidate"),
                ("regime", "regime"),
                ("samples", "samples"),
                ("cov_abs", "cov_abs"),
                ("lower", "lower"),
                ("interval", "interval"),
                ("bw_ic", "bw_ic"),
                ("down_ic", "down_ic"),
                ("p90_w", "p90_w"),
            ],
        ),
        "",
        "## 6. Candidate Registry",
        "",
        _table(
            [
                {
                    "category": row.get("category"),
                    "display_name": row.get("display_name"),
                    "feature_set": row.get("feature_set"),
                    "mode": row.get("band_mode"),
                    "test_exposure": row.get("test_exposure_count"),
                    "runs": ",".join(row.get("raw_run_ids") or []),
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
                ("test_exposure", "test_exposure"),
                ("runs", "runs"),
                ("strength", "strength"),
                ("weakness", "weakness"),
            ],
        ),
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


def _existing_runs() -> dict[tuple[str, int], dict[str, Any]]:
    payload = _read_json(METRICS_PATH)
    result = {}
    for item in payload.get("candidate_summaries", []):
        for run in item.get("runs", []):
            if run.get("candidate") and run.get("seed") is not None:
                result[(str(run.get("candidate")), int(run.get("seed")))] = run
    return result


def _write_outputs(candidate_summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    criteria = _criteria()
    for item in candidate_summaries:
        item["decision"] = _classify_candidate(item, criteria)
    _apply_default(candidate_summaries)
    registry = [_registry_entry(item) for item in candidate_summaries]
    recommended = next((row.get("source_experiment_id") for row in registry if row.get("category") == "recommended_default"), None)
    payload = {
        "cp": "CP121-BM",
        "status": "PASS" if all(item.get("completed_run_count", 0) > 0 for item in candidate_summaries) else "PARTIAL_OR_FAIL",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution_policy": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "disabled",
            "composite": False,
            "local_snapshot_required": True,
            "selection_uses_test_metrics": False,
        },
        "environment": {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(PROJECT_ROOT / "data" / "parquet"),
            "WANDB_MODE": "disabled",
        },
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "criteria": criteria,
        "recommended_default": recommended,
        "candidate_summaries": candidate_summaries,
        "candidate_registry": registry,
    }
    _write_json(METRICS_PATH, payload)
    _write_json(REGISTRY_PATH, registry)
    _write_summary_csv(candidate_summaries)
    _write_report(payload)


def run(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [candidate for candidate in CANDIDATES if not args.only or candidate.name in set(args.only)]
    existing = _existing_runs()
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            cached = existing.get((candidate.name, seed))
            if cached and cached.get("execution_status") == "PASS" and not args.force:
                runs.append(cached)
                continue
            print(f"[CP121] 시작: {candidate.name} seed={seed}", flush=True)
            run_record = _run_training(candidate, seed, args)
            if run_record.get("execution_status") == "PASS":
                try:
                    run_record["validation_regime_metrics"] = _validation_regime_metrics(run_record, args)
                except Exception as exc:
                    run_record["validation_regime_metrics"] = {"error": str(exc)}
            print(
                f"[CP121] 종료: {candidate.name} seed={seed} status={run_record.get('execution_status')} exit={run_record.get('exit_code')} elapsed={_fmt(run_record.get('elapsed_seconds'), 2)}",
                flush=True,
            )
            runs.append(run_record)
        summaries.append(_aggregate_candidate(candidate, runs))
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
