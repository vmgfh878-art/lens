from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.band_calibration import (  # noqa: E402
    PredictionSet,
    apply_scalar_width_calibration,
    fit_scalar_width_calibration,
    summarize_predictions,
)
from ai.inference import _select_bundle_features, load_checkpoint, resolve_bundle, resolve_checkpoint_ticker_registry  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.train import autocast_context, forward_model, make_loader, resolve_device  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "cp125_bm_1w_band_calibration_regime_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp125_bm_1w_band_calibration_regime_metrics.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "cp125_bm_1w_band_final_registry.json"
REGIME_CSV_PATH = PROJECT_ROOT / "docs" / "cp125_bm_1w_band_regime_summary.csv"
CP124_METRICS_PATH = PROJECT_ROOT / "docs" / "cp124_bm_1w_band_loss_downside_guard_metrics.json"

BAND_METRIC_KEYS = [
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "asymmetric_interval_score",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "avg_band_width",
    "median_band_width",
    "p90_band_width",
    "band_width_ic",
    "downside_width_ic",
    "squeeze_breakout_rate",
]

DECISION_KEYS = [
    "coverage_abs_error",
    "lower_breach_rate",
    "asymmetric_interval_score",
    "p90_band_width",
    "band_width_ic",
    "downside_width_ic",
]

REGIME_NAMES = [
    "rising_market",
    "falling_market",
    "high_volatility",
    "low_volatility",
    "high_atr",
    "low_atr",
    "wide_band",
    "narrow_band",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate: str
    source_experiment: str
    display_name: str
    role: str
    model_family: str
    feature_set: str
    band_mode: str
    q_low: float
    q_high: float
    strength_summary: str
    weakness_summary: str
    best_use_case: str


CANDIDATES = [
    CandidateSpec(
        candidate="cnn_full_q10_direct_lower_guard_w1p5",
        source_experiment="cnn_full_q10_direct_lower_guard_w1p5",
        display_name="1W CNN-LSTM full q10 direct lower guard",
        role="band_model",
        model_family="cnn_lstm",
        feature_set="full_features",
        band_mode="direct",
        q_low=0.10,
        q_high=0.90,
        strength_summary="CP124에서 falling regime lower breach를 낮춘 하방 방어형 후보",
        weakness_summary="full_features와 q10/q90 특성상 TiDE보다 폭이 넓을 수 있음",
        best_use_case="1W 제품 기본 AI band 후보",
    ),
    CandidateSpec(
        candidate="tide_pvv_q15_param",
        source_experiment="tide_pvv_q15_param_baseline",
        display_name="1W TiDE PVV q15 param",
        role="band_model",
        model_family="tide",
        feature_set="price_volatility_volume",
        band_mode="param",
        q_low=0.15,
        q_high=0.85,
        strength_summary="CP121에서 coverage와 dynamic width가 좋았던 변동성 민감형 후보",
        weakness_summary="falling regime lower breach가 약점으로 관찰됨",
        best_use_case="1W 변동성 민감형 보조 band 후보",
    ),
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


def _band_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    source = summary.get("band_metrics") if isinstance(summary.get("band_metrics"), dict) else summary
    return {key: source.get(key) for key in BAND_METRIC_KEYS}


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


def _metric_stats(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = []
    for record in records:
        source = record.get("metrics") if isinstance(record.get("metrics"), dict) else record
        value = _safe_float(source.get(key))
        if value is not None:
            values.append(value)
    return _mean_std(values)


def _collect_candidate_runs() -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(CP124_METRICS_PATH)
    result: dict[str, list[dict[str, Any]]] = {}
    for summary in payload.get("experiment_summaries", []):
        experiment = str(summary.get("experiment") or "")
        runs = []
        for run in summary.get("runs", []):
            if run.get("checkpoint_path") and run.get("execution_status") in {"PASS", "REUSED_CP121"}:
                runs.append(run)
        if runs:
            result[experiment] = runs
    return result


def _collect_predictions(
    *,
    checkpoint_path: Path,
    split: str,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
) -> tuple[PredictionSet, dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    device = resolve_device(device_name)
    model = model.to(device)
    model.eval()
    registry = resolve_checkpoint_ticker_registry(config, str(config["timeframe"]))
    bundle = resolve_bundle(
        split_name=split,
        timeframe=str(config["timeframe"]),
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        tickers=config.get("tickers"),
        limit_tickers=config.get("limit_tickers"),
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        ticker_registry=registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    bundle = _select_bundle_features(bundle, list(config.get("feature_columns") or []))
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    line_chunks: list[torch.Tensor] = []
    lower_chunks: list[torch.Tensor] = []
    upper_chunks: list[torch.Tensor] = []
    line_target_chunks: list[torch.Tensor] = []
    band_target_chunks: list[torch.Tensor] = []
    raw_target_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_ids, future_covariates in loader:
            features = features.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, amp_dtype=amp_dtype):
                output = forward_model(model, features, ticker_ids, future_covariates)
            line, lower, upper = apply_band_postprocess(
                output.line.detach().cpu(),
                output.lower_band.detach().cpu(),
                output.upper_band.detach().cpu(),
            )
            line_chunks.append(line)
            lower_chunks.append(lower)
            upper_chunks.append(upper)
            line_target_chunks.append(line_target.detach().cpu())
            band_target_chunks.append(band_target.detach().cpu())
            raw_target_chunks.append(raw_future_returns.detach().cpu())

    predictions = PredictionSet(
        line=torch.cat(line_chunks, dim=0),
        lower=torch.cat(lower_chunks, dim=0),
        upper=torch.cat(upper_chunks, dim=0),
        line_target=torch.cat(line_target_chunks, dim=0),
        band_target=torch.cat(band_target_chunks, dim=0),
        raw_future_returns=torch.cat(raw_target_chunks, dim=0),
        metadata=bundle.metadata.reset_index(drop=True),
    )
    return predictions, {"config": config}


def _slice_predictions(predictions: PredictionSet, mask_or_indices: torch.Tensor) -> PredictionSet:
    if mask_or_indices.dtype == torch.bool:
        indices = mask_or_indices.nonzero(as_tuple=False).flatten()
    else:
        indices = mask_or_indices.flatten().to(dtype=torch.long)
    metadata = predictions.metadata.iloc[indices.detach().cpu().numpy()].reset_index(drop=True)
    return PredictionSet(
        line=predictions.line.index_select(0, indices),
        lower=predictions.lower.index_select(0, indices),
        upper=predictions.upper.index_select(0, indices),
        line_target=predictions.line_target.index_select(0, indices),
        band_target=predictions.band_target.index_select(0, indices),
        raw_future_returns=predictions.raw_future_returns.index_select(0, indices),
        metadata=metadata,
    )


def _summarize(predictions: PredictionSet, line: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    return _band_metrics(
        summarize_predictions(
            predictions,
            line=line,
            lower=lower,
            upper=upper,
            q_low=float(config.get("q_low", 0.1)),
            q_high=float(config.get("q_high", 0.9)),
            line_target_type=str(config.get("line_target_type", "raw_future_return")),
            band_target_type=str(config.get("band_target_type", "raw_future_return")),
        )
    )


def _validation_halves(predictions: PredictionSet) -> tuple[PredictionSet, PredictionSet, dict[str, Any]]:
    dates = pd.to_datetime(predictions.metadata["asof_date"])
    unique_dates = sorted(pd.Series(dates).dropna().unique())
    if len(unique_dates) < 4:
        raise ValueError("validation calibration split에 필요한 asof_date가 부족합니다.")
    cutoff = unique_dates[len(unique_dates) // 2 - 1]
    fit_mask = torch.tensor((dates <= cutoff).to_numpy(), dtype=torch.bool)
    eval_mask = torch.tensor((dates > cutoff).to_numpy(), dtype=torch.bool)
    return (
        _slice_predictions(predictions, fit_mask),
        _slice_predictions(predictions, eval_mask),
        {
            "calibration_fit_max_date": pd.Timestamp(cutoff).date().isoformat(),
            "fit_sample_count": int(fit_mask.sum().item()),
            "eval_sample_count": int(eval_mask.sum().item()),
            "unique_validation_dates": len(unique_dates),
        },
    )


def _atr_values(metadata: pd.DataFrame) -> torch.Tensor:
    path = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1W.parquet"
    if not path.exists():
        return torch.full((len(metadata),), float("nan"), dtype=torch.float32)
    indicators = pd.read_parquet(path, columns=["ticker", "date", "atr_ratio", "source", "provider"])
    indicators = indicators[(indicators["provider"] == "yfinance") & (indicators["source"] == "yfinance")].copy()
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    lookup = {
        (str(row.ticker).upper(), row.date): float(row.atr_ratio)
        for row in indicators.itertuples(index=False)
        if pd.notna(row.atr_ratio)
    }
    values = []
    for row in metadata.itertuples(index=False):
        ticker = str(getattr(row, "ticker")).upper()
        date = pd.to_datetime(getattr(row, "asof_date")).date()
        values.append(lookup.get((ticker, date), float("nan")))
    return torch.tensor(values, dtype=torch.float32)


def _regime_masks(predictions: PredictionSet, *, lower: torch.Tensor, upper: torch.Tensor) -> dict[str, torch.Tensor]:
    width = (upper - lower).mean(dim=1)
    realized_volatility = predictions.raw_future_returns.std(dim=1, unbiased=False)
    realized_h4_return = predictions.raw_future_returns[:, -1]
    atr = _atr_values(predictions.metadata)
    high_vol_threshold = torch.median(realized_volatility)
    width_threshold = torch.median(width)
    finite_atr = atr[torch.isfinite(atr)]
    if finite_atr.numel() > 0:
        atr_threshold = torch.median(finite_atr)
        high_atr = torch.isfinite(atr) & (atr >= atr_threshold)
        low_atr = torch.isfinite(atr) & (atr < atr_threshold)
    else:
        high_atr = torch.zeros_like(width, dtype=torch.bool)
        low_atr = torch.zeros_like(width, dtype=torch.bool)
    return {
        "rising_market": realized_h4_return >= 0.0,
        "falling_market": realized_h4_return < 0.0,
        "high_volatility": realized_volatility >= high_vol_threshold,
        "low_volatility": realized_volatility < high_vol_threshold,
        "high_atr": high_atr,
        "low_atr": low_atr,
        "wide_band": width >= width_threshold,
        "narrow_band": width < width_threshold,
    }


def _regime_metrics(
    predictions: PredictionSet,
    *,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    masks = _regime_masks(predictions, lower=lower, upper=upper)
    for name in REGIME_NAMES:
        mask = masks[name]
        count = int(mask.sum().item())
        if count < 10:
            regimes[name] = {"sample_count": count, "metrics": {}}
            continue
        sliced = _slice_predictions(predictions, mask)
        indices = mask.nonzero(as_tuple=False).flatten()
        regimes[name] = {
            "sample_count": count,
            "metrics": _summarize(
                sliced,
                line.index_select(0, indices),
                lower.index_select(0, indices),
                upper.index_select(0, indices),
                config,
            ),
        }
    return regimes


def _aggregate_metric_block(runs: list[dict[str, Any]], block: str) -> dict[str, Any]:
    records = [run[block] for run in runs if isinstance(run.get(block), dict)]
    return {key: _metric_stats(records, key) for key in BAND_METRIC_KEYS}


def _aggregate_regime_block(runs: list[dict[str, Any]], block: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in REGIME_NAMES:
        regime_result: dict[str, Any] = {}
        sample_counts: list[float] = []
        for key in BAND_METRIC_KEYS:
            values: list[float] = []
            for run in runs:
                payload = (((run.get(block) or {}).get(regime) or {}))
                sample_count = _safe_float(payload.get("sample_count"))
                if sample_count is not None:
                    sample_counts.append(sample_count)
                value = _safe_float((payload.get("metrics") or {}).get(key))
                if value is not None:
                    values.append(value)
            regime_result[key] = _mean_std(values)
        regime_result["sample_count"] = _mean_std(sample_counts)
        result[regime] = regime_result
    return result


def _metric_mean(summary: dict[str, Any], block: str, key: str) -> float | None:
    return _safe_float((((summary.get(block) or {}).get(key) or {}).get("mean")))


def _regime_mean(summary: dict[str, Any], block: str, regime: str, key: str) -> float | None:
    return _safe_float(((((summary.get(block) or {}).get(regime) or {}).get(key) or {}).get("mean")))


def _evaluate_seed_run(spec: CandidateSpec, source_run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = PROJECT_ROOT / str(source_run["checkpoint_path"])
    predictions, context = _collect_predictions(
        checkpoint_path=checkpoint_path,
        split="val",
        device_name=args.device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
    )
    config = context["config"]
    fit_predictions, eval_predictions, split_info = _validation_halves(predictions)
    raw_full_metrics = _summarize(predictions, predictions.line, predictions.lower, predictions.upper, config)
    raw_eval_metrics = _summarize(eval_predictions, eval_predictions.line, eval_predictions.lower, eval_predictions.upper, config)
    raw_full_regimes = _regime_metrics(
        predictions,
        line=predictions.line,
        lower=predictions.lower,
        upper=predictions.upper,
        config=config,
    )
    raw_eval_regimes = _regime_metrics(
        eval_predictions,
        line=eval_predictions.line,
        lower=eval_predictions.lower,
        upper=eval_predictions.upper,
        config=config,
    )
    target_coverage = float(config.get("q_high", spec.q_high)) - float(config.get("q_low", spec.q_low))
    calibration = fit_scalar_width_calibration(fit_predictions, target_coverage=target_coverage)
    cal_line, cal_lower, cal_upper = apply_scalar_width_calibration(eval_predictions, calibration)
    calibrated_eval_metrics = _summarize(eval_predictions, cal_line, cal_lower, cal_upper, config)
    calibrated_eval_regimes = _regime_metrics(
        eval_predictions,
        line=cal_line,
        lower=cal_lower,
        upper=cal_upper,
        config=config,
    )
    raw_width = _safe_float(raw_eval_metrics.get("avg_band_width"))
    calibrated_width = _safe_float(calibrated_eval_metrics.get("avg_band_width"))
    width_increase_ratio = None
    if raw_width is not None and calibrated_width is not None and raw_width != 0:
        width_increase_ratio = (calibrated_width / raw_width) - 1.0
    width_multiplier = (float(calibration["lower_scale"]) + float(calibration["upper_scale"])) / 2.0
    return {
        "candidate": spec.candidate,
        "seed": source_run.get("seed"),
        "run_id": source_run.get("run_id"),
        "checkpoint_path": source_run.get("checkpoint_path"),
        "source_execution_status": source_run.get("execution_status"),
        "source_gate": source_run.get("gate"),
        "config": {
            "model": config.get("model"),
            "timeframe": config.get("timeframe"),
            "horizon": config.get("horizon"),
            "seq_len": config.get("seq_len"),
            "feature_set": config.get("feature_set"),
            "n_features": config.get("n_features"),
            "q_low": config.get("q_low"),
            "q_high": config.get("q_high"),
            "band_mode": config.get("band_mode"),
            "lambda_band": config.get("lambda_band"),
            "lower_band_loss_weight": config.get("lower_band_loss_weight"),
            "upper_band_loss_weight": config.get("upper_band_loss_weight"),
            "checkpoint_selection": config.get("checkpoint_selection"),
        },
        "validation_split_policy": split_info,
        "raw_validation_full": {"metrics": raw_full_metrics},
        "raw_validation_eval": {"metrics": raw_eval_metrics},
        "scalar_calibrated_validation_eval": {"metrics": calibrated_eval_metrics},
        "raw_validation_full_regimes": raw_full_regimes,
        "raw_validation_eval_regimes": raw_eval_regimes,
        "scalar_calibrated_validation_eval_regimes": calibrated_eval_regimes,
        "scalar_width_calibration": {
            **calibration,
            "calibration_width_multiplier": width_multiplier,
            "calibration_width_increase_ratio": width_increase_ratio,
            "fit_split": "validation_first_half_by_asof_date",
            "eval_split": "validation_second_half_by_asof_date",
        },
        "test_metrics_opened_in_cp125": False,
    }


def _aggregate_candidate(spec: CandidateSpec, runs: list[dict[str, Any]], source_summary: dict[str, Any]) -> dict[str, Any]:
    gate_pass_count = sum(
        1
        for run in runs
        if (run.get("source_gate") or {}).get("band_gate_pass") and not (run.get("source_gate") or {}).get("gate_failed")
    )
    calibration_values = [run.get("scalar_width_calibration") or {} for run in runs]
    summary = {
        "candidate": spec.candidate,
        "source_experiment": spec.source_experiment,
        "candidate_config": spec.__dict__,
        "runs": runs,
        "run_count": len(runs),
        "raw_validation_band_gate_pass_rate": gate_pass_count / len(runs) if runs else 0.0,
        "test_exposure_count": source_summary.get("test_exposure_count", 0),
        "test_exposure_policy": "CP125에서는 test split을 새로 평가하지 않고 이전 CP의 read-only exposure count만 이월",
        "raw_validation_full_stats": _aggregate_metric_block([run["raw_validation_full"] for run in runs], "metrics"),
        "raw_validation_eval_stats": _aggregate_metric_block([run["raw_validation_eval"] for run in runs], "metrics"),
        "scalar_calibrated_validation_eval_stats": _aggregate_metric_block(
            [run["scalar_calibrated_validation_eval"] for run in runs],
            "metrics",
        ),
        "raw_validation_full_regime_stats": _aggregate_regime_block(runs, "raw_validation_full_regimes"),
        "raw_validation_eval_regime_stats": _aggregate_regime_block(runs, "raw_validation_eval_regimes"),
        "scalar_calibrated_validation_eval_regime_stats": _aggregate_regime_block(
            runs,
            "scalar_calibrated_validation_eval_regimes",
        ),
        "calibration_stats": {
            "calibration_width_multiplier": _mean_std(
                [
                    float(item["calibration_width_multiplier"])
                    for item in calibration_values
                    if _safe_float(item.get("calibration_width_multiplier")) is not None
                ]
            ),
            "calibration_width_increase_ratio": _mean_std(
                [
                    float(item["calibration_width_increase_ratio"])
                    for item in calibration_values
                    if _safe_float(item.get("calibration_width_increase_ratio")) is not None
                ]
            ),
            "lower_scale": _mean_std(
                [float(item["lower_scale"]) for item in calibration_values if _safe_float(item.get("lower_scale")) is not None]
            ),
            "upper_scale": _mean_std(
                [float(item["upper_scale"]) for item in calibration_values if _safe_float(item.get("upper_scale")) is not None]
            ),
        },
    }
    return summary


def _source_summary_map() -> dict[str, dict[str, Any]]:
    payload = _read_json(CP124_METRICS_PATH)
    return {
        str(summary.get("experiment")): summary
        for summary in payload.get("experiment_summaries", [])
    }


def _raw_checks(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "raw_validation_band_gate_pass": float(summary.get("raw_validation_band_gate_pass_rate") or 0.0) >= 1.0,
        "raw_coverage_abs_error": (_metric_mean(summary, "raw_validation_full_stats", "coverage_abs_error") or float("inf")) <= 0.05,
        "raw_lower_breach_rate": (_metric_mean(summary, "raw_validation_full_stats", "lower_breach_rate") or float("inf")) <= 0.18,
        "raw_falling_lower_breach_rate": (
            _regime_mean(summary, "raw_validation_full_regime_stats", "falling_market", "lower_breach_rate") or float("inf")
        )
        <= 0.20,
        "raw_high_vol_lower_breach_rate": (
            _regime_mean(summary, "raw_validation_full_regime_stats", "high_volatility", "lower_breach_rate") or float("inf")
        )
        <= 0.20,
        "raw_band_width_ic": (_metric_mean(summary, "raw_validation_full_stats", "band_width_ic") or -float("inf")) > 0.15,
        "raw_downside_width_ic": (_metric_mean(summary, "raw_validation_full_stats", "downside_width_ic") or -float("inf")) >= 0.0,
        "raw_p90_width_not_exploded": (_metric_mean(summary, "raw_validation_full_stats", "p90_band_width") or float("inf")) <= 0.35,
    }


def _calibration_checks(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "calibrated_coverage_abs_error": (
            _metric_mean(summary, "scalar_calibrated_validation_eval_stats", "coverage_abs_error") or float("inf")
        )
        <= 0.05,
        "calibrated_lower_breach_rate": (
            _metric_mean(summary, "scalar_calibrated_validation_eval_stats", "lower_breach_rate") or float("inf")
        )
        <= 0.18,
        "calibrated_falling_lower_breach_rate": (
            _regime_mean(
                summary,
                "scalar_calibrated_validation_eval_regime_stats",
                "falling_market",
                "lower_breach_rate",
            )
            or float("inf")
        )
        <= 0.20,
        "calibrated_high_vol_lower_breach_rate": (
            _regime_mean(
                summary,
                "scalar_calibrated_validation_eval_regime_stats",
                "high_volatility",
                "lower_breach_rate",
            )
            or float("inf")
        )
        <= 0.20,
        "calibrated_band_width_ic": (
            _metric_mean(summary, "scalar_calibrated_validation_eval_stats", "band_width_ic") or -float("inf")
        )
        > 0.15,
        "calibrated_width_not_extreme": (
            _safe_float(((summary.get("calibration_stats") or {}).get("calibration_width_increase_ratio") or {}).get("mean"))
            or 0.0
        )
        <= 0.50,
    }


def _apply_decisions(summaries: list[dict[str, Any]]) -> None:
    for summary in summaries:
        raw_checks = _raw_checks(summary)
        calibration_checks = _calibration_checks(summary)
        raw_failures = [key for key, ok in raw_checks.items() if not ok]
        cal_failures = [key for key, ok in calibration_checks.items() if not ok]
        if not raw_failures:
            category = "selectable_verified"
        elif raw_failures and not cal_failures:
            category = "calibration_only_candidate"
        else:
            category = "unstable_or_rejected"
        summary["decision"] = {
            "category": category,
            "raw_checks": raw_checks,
            "raw_failures": raw_failures,
            "calibration_checks": calibration_checks,
            "calibration_failures": cal_failures,
        }
    verified = [summary for summary in summaries if (summary.get("decision") or {}).get("category") == "selectable_verified"]
    if verified:
        best = sorted(
            verified,
            key=lambda item: (
                _regime_mean(item, "raw_validation_full_regime_stats", "falling_market", "lower_breach_rate") or float("inf"),
                _metric_mean(item, "raw_validation_full_stats", "coverage_abs_error") or float("inf"),
                _metric_mean(item, "raw_validation_full_stats", "asymmetric_interval_score") or float("inf"),
                _metric_mean(item, "raw_validation_full_stats", "p90_band_width") or float("inf"),
            ),
        )[0]
        best["decision"]["category"] = "recommended_default"


def _registry_entry(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("candidate_config") or {}
    decision = summary.get("decision") or {}
    raw_stats = summary.get("raw_validation_full_stats") or {}
    cal_stats = summary.get("scalar_calibrated_validation_eval_stats") or {}
    category = decision.get("category") or "unstable_or_rejected"
    return {
        "category": category,
        "display_name": config.get("display_name"),
        "role": "band_model",
        "timeframe": "1W",
        "horizon": 4,
        "model_family": config.get("model_family"),
        "feature_set": config.get("feature_set"),
        "target_type": "raw_future_return",
        "band_mode": config.get("band_mode"),
        "strength_summary": config.get("strength_summary"),
        "weakness_summary": config.get("weakness_summary"),
        "best_use_case": config.get("best_use_case"),
        "why_not_default": "현재 기본 후보" if category == "recommended_default" else ", ".join(decision.get("raw_failures") or ["기본 후보 아님"]),
        "raw_key_metrics": {
            key: (raw_stats.get(key) or {}).get("mean")
            for key in DECISION_KEYS
        },
        "raw_regime_metrics": {
            "falling_regime_lower_breach_rate": _regime_mean(summary, "raw_validation_full_regime_stats", "falling_market", "lower_breach_rate"),
            "high_vol_regime_lower_breach_rate": _regime_mean(summary, "raw_validation_full_regime_stats", "high_volatility", "lower_breach_rate"),
            "high_atr_regime_lower_breach_rate": _regime_mean(summary, "raw_validation_full_regime_stats", "high_atr", "lower_breach_rate"),
        },
        "scalar_calibrated_key_metrics": {
            key: (cal_stats.get(key) or {}).get("mean")
            for key in DECISION_KEYS
        },
        "calibration": summary.get("calibration_stats"),
        "calibration_dependency": category == "calibration_only_candidate",
        "test_exposure_count": summary.get("test_exposure_count"),
        "test_metric_policy": summary.get("test_exposure_policy"),
        "raw_run_ids": [run.get("run_id") for run in summary.get("runs", []) if run.get("run_id")],
        "source_experiment_id": summary.get("candidate"),
    }


def _summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        decision = summary.get("decision") or {}
        rows.append(
            {
                "candidate": summary.get("candidate"),
                "category": decision.get("category"),
                "raw_cov_abs": _metric_mean(summary, "raw_validation_full_stats", "coverage_abs_error"),
                "raw_lower": _metric_mean(summary, "raw_validation_full_stats", "lower_breach_rate"),
                "raw_upper": _metric_mean(summary, "raw_validation_full_stats", "upper_breach_rate"),
                "raw_falling_lower": _regime_mean(summary, "raw_validation_full_regime_stats", "falling_market", "lower_breach_rate"),
                "raw_high_vol_lower": _regime_mean(summary, "raw_validation_full_regime_stats", "high_volatility", "lower_breach_rate"),
                "raw_high_atr_lower": _regime_mean(summary, "raw_validation_full_regime_stats", "high_atr", "lower_breach_rate"),
                "raw_interval": _metric_mean(summary, "raw_validation_full_stats", "asymmetric_interval_score"),
                "raw_p90_width": _metric_mean(summary, "raw_validation_full_stats", "p90_band_width"),
                "raw_bw_ic": _metric_mean(summary, "raw_validation_full_stats", "band_width_ic"),
                "raw_down_ic": _metric_mean(summary, "raw_validation_full_stats", "downside_width_ic"),
                "cal_cov_abs": _metric_mean(summary, "scalar_calibrated_validation_eval_stats", "coverage_abs_error"),
                "cal_lower": _metric_mean(summary, "scalar_calibrated_validation_eval_stats", "lower_breach_rate"),
                "cal_falling_lower": _regime_mean(summary, "scalar_calibrated_validation_eval_regime_stats", "falling_market", "lower_breach_rate"),
                "cal_width_multiplier": _safe_float(((summary.get("calibration_stats") or {}).get("calibration_width_multiplier") or {}).get("mean")),
                "cal_width_increase": _safe_float(((summary.get("calibration_stats") or {}).get("calibration_width_increase_ratio") or {}).get("mean")),
                "raw_failures": ",".join(decision.get("raw_failures") or []),
                "cal_failures": ",".join(decision.get("calibration_failures") or []),
            }
        )
    return rows


def _regime_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for basis, block in (
            ("raw_validation_full", "raw_validation_full_regime_stats"),
            ("scalar_calibrated_validation_eval", "scalar_calibrated_validation_eval_regime_stats"),
        ):
            for regime in REGIME_NAMES:
                stats = ((summary.get(block) or {}).get(regime) or {})
                rows.append(
                    {
                        "candidate": summary.get("candidate"),
                        "basis": basis,
                        "regime": regime,
                        "sample_count": (stats.get("sample_count") or {}).get("mean"),
                        "coverage_abs_error": (stats.get("coverage_abs_error") or {}).get("mean"),
                        "lower_breach_rate": (stats.get("lower_breach_rate") or {}).get("mean"),
                        "upper_breach_rate": (stats.get("upper_breach_rate") or {}).get("mean"),
                        "asymmetric_interval_score": (stats.get("asymmetric_interval_score") or {}).get("mean"),
                        "avg_band_width": (stats.get("avg_band_width") or {}).get("mean"),
                        "p90_band_width": (stats.get("p90_band_width") or {}).get("mean"),
                        "band_width_ic": (stats.get("band_width_ic") or {}).get("mean"),
                        "downside_width_ic": (stats.get("downside_width_ic") or {}).get("mean"),
                    }
                )
    return rows


def _write_regime_csv(summaries: list[dict[str, Any]]) -> None:
    rows = _regime_rows(summaries)
    fieldnames = [
        "candidate",
        "basis",
        "regime",
        "sample_count",
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "asymmetric_interval_score",
        "avg_band_width",
        "p90_band_width",
        "band_width_ic",
        "downside_width_ic",
    ]
    with REGIME_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
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
    summaries = payload.get("candidate_summaries") or []
    registry = payload.get("candidate_registry") or []
    report = [
        "# CP125-BM 1W Band Calibration / Regime Final Evaluation",
        "",
        "## 1. 결론",
        "",
        f"- 상태: {payload.get('status')}",
        f"- 기본 후보: `{payload.get('default_candidate')}`",
        f"- 대안 후보: `{payload.get('alternative_candidate')}`",
        f"- 저장 전 추가 확인 필요: {payload.get('pre_save_check_needed')}",
        "- save-run, DB write, inference 저장, W&B, composite, 프론트 수정, 새 후보 탐색, 새 target 구현은 수행하지 않았다.",
        "- test split은 CP125에서 새로 열지 않았고, 이전 CP의 test_exposure_count만 registry에 이월했다.",
        "- raw 평가는 전체 validation 기준이고, calibration 평가는 validation 날짜 앞 절반 fit / 뒤 절반 eval 기준이다.",
        "",
        "## 2. 후보 요약",
        "",
        _table(
            _summary_rows(summaries),
            [
                ("candidate", "candidate"),
                ("category", "category"),
                ("raw_cov", "raw_cov_abs"),
                ("raw_lower", "raw_lower"),
                ("raw_falling", "raw_falling_lower"),
                ("raw_high_vol", "raw_high_vol_lower"),
                ("raw_high_atr", "raw_high_atr_lower"),
                ("raw_interval", "raw_interval"),
                ("raw_p90", "raw_p90_width"),
                ("raw_bw_ic", "raw_bw_ic"),
                ("raw_down_ic", "raw_down_ic"),
                ("cal_cov", "cal_cov_abs"),
                ("cal_lower", "cal_lower"),
                ("cal_falling", "cal_falling_lower"),
                ("cal_mult", "cal_width_multiplier"),
                ("cal_width_inc", "cal_width_increase"),
                ("raw_failures", "raw_failures"),
            ],
        ),
        "",
        "## 3. Raw / Calibration 분리 원칙",
        "",
        "- raw 기준 성능은 모델 원본 band 자체의 성능으로 해석한다.",
        "- scalar calibration은 validation 앞 절반에서 fit하고 뒤 절반에서만 평가했다.",
        "- calibration으로 개선된 수치는 raw 모델 성능으로 해석하지 않는다.",
        "- calibration 없이는 regime 기준을 넘지 못하는 후보는 기본 후보가 아니라 calibration_only_candidate 이하로 분류한다.",
        "",
        "## 4. Candidate Registry",
        "",
        _table(
            [
                {
                    "category": row.get("category"),
                    "display_name": row.get("display_name"),
                    "feature_set": row.get("feature_set"),
                    "band_mode": row.get("band_mode"),
                    "test_exposure": row.get("test_exposure_count"),
                    "why": row.get("why_not_default"),
                }
                for row in registry
            ],
            [
                ("category", "category"),
                ("display_name", "display_name"),
                ("feature_set", "feature_set"),
                ("mode", "band_mode"),
                ("test_exp", "test_exposure"),
                ("why_not_default", "why"),
            ],
        ),
        "",
        "## 5. Regime CSV",
        "",
        f"- regime별 상세 수치는 `{REGIME_CSV_PATH.relative_to(PROJECT_ROOT)}`에 저장했다.",
        "",
        "## 6. 산출물",
        "",
        f"- `{REPORT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{REGISTRY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{REGIME_CSV_PATH.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def _write_outputs(summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    _apply_decisions(summaries)
    registry = [_registry_entry(summary) for summary in summaries]
    default_candidate = next((row.get("source_experiment_id") for row in registry if row.get("category") == "recommended_default"), None)
    alternative_candidate = next(
        (
            row.get("source_experiment_id")
            for row in registry
            if row.get("category") in {"selectable_verified", "calibration_only_candidate"} and row.get("source_experiment_id") != default_candidate
        ),
        None,
    )
    payload = {
        "cp": "CP125-BM",
        "status": "PASS" if summaries and default_candidate else "PARTIAL_OR_WARN",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution_policy": {
            "save_run": False,
            "db_write": False,
            "inference_save": False,
            "wandb": "disabled",
            "composite": False,
            "frontend": False,
            "new_candidate_search": False,
            "new_target_implementation": False,
            "selection_uses_test_metrics": False,
            "cp125_opened_test_split": False,
        },
        "environment": {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(PROJECT_ROOT / "data" / "parquet"),
            "WANDB_MODE": "disabled",
        },
        "criteria": {
            "raw_validation_coverage_abs_error_max": 0.05,
            "raw_validation_lower_breach_rate_max": 0.18,
            "raw_regime_lower_breach_rate_max": 0.20,
            "raw_band_width_ic_min": 0.15,
            "raw_downside_width_ic_min": 0.0,
            "raw_p90_band_width_max": 0.35,
            "calibration_width_increase_ratio_max": 0.50,
        },
        "calibration_policy": {
            "method": "scalar_width",
            "fit_split": "validation_first_half_by_asof_date",
            "eval_split": "validation_second_half_by_asof_date",
            "target_coverage": "candidate q_high - q_low",
        },
        "default_candidate": default_candidate,
        "alternative_candidate": alternative_candidate,
        "pre_save_check_needed": bool(default_candidate),
        "pre_save_check_note": "제품 저장 전 CP126에서 선택된 기본 후보만 save-run 재학습 또는 checkpoint 재현 정책을 확정해야 함",
        "candidate_summaries": summaries,
        "candidate_registry": registry,
    }
    _write_json(METRICS_PATH, payload)
    _write_json(REGISTRY_PATH, registry)
    _write_regime_csv(summaries)
    _write_report(payload)


def run(args: argparse.Namespace) -> None:
    os.environ.update(
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
    source_runs = _collect_candidate_runs()
    source_summaries = _source_summary_map()
    summaries: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        runs = []
        for source_run in source_runs.get(spec.source_experiment, []):
            print(f"[CP125] 평가 시작: {spec.candidate} seed={source_run.get('seed')}", flush=True)
            runs.append(_evaluate_seed_run(spec, source_run, args))
            print(f"[CP125] 평가 종료: {spec.candidate} seed={source_run.get('seed')}", flush=True)
        summaries.append(_aggregate_candidate(spec, runs, source_summaries.get(spec.source_experiment, {})))
        _write_outputs(summaries, args)
    _write_outputs(summaries, args)
    print(json.dumps({"status": "done", "metrics_path": str(METRICS_PATH), "registry_path": str(REGISTRY_PATH)}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
