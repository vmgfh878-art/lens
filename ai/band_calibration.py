from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ai.evaluation import summarize_forecast_metrics
from ai.inference import load_checkpoint, resolve_checkpoint_ticker_registry
from ai.postprocess import apply_band_postprocess
from ai.preprocessing import prepare_dataset_splits
from ai.train import autocast_context, forward_model, make_loader, resolve_device


@dataclass
class PredictionSet:
    line: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    line_target: torch.Tensor
    band_target: torch.Tensor
    raw_future_returns: torch.Tensor
    metadata: Any


def _quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(torch.quantile(finite.to(torch.float32), q).item())


def _band_loss(
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    actual: torch.Tensor,
    q_low: float,
    q_high: float,
) -> dict[str, float]:
    low_error = actual - lower
    high_error = actual - upper
    low_loss = torch.maximum(q_low * low_error, (q_low - 1.0) * low_error)
    high_loss = torch.maximum(q_high * high_error, (q_high - 1.0) * high_error)
    cross_loss = torch.relu(lower - upper)
    return {
        "band_loss": float((low_loss + high_loss).mean().item()),
        "cross_loss": float(cross_loss.mean().item()),
    }


def summarize_predictions(
    predictions: PredictionSet,
    *,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    q_low: float,
    q_high: float,
    line_target_type: str,
    band_target_type: str,
) -> dict[str, float | None]:
    summary = summarize_forecast_metrics(
        metadata=predictions.metadata,
        line_predictions=line,
        lower_predictions=lower,
        upper_predictions=upper,
        line_targets=predictions.line_target,
        band_targets=predictions.band_target,
        raw_future_returns=predictions.raw_future_returns,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
        q_low=q_low,
        q_high=q_high,
    )
    summary.update(
        _band_loss(
            lower=lower,
            upper=upper,
            actual=predictions.band_target,
            q_low=q_low,
            q_high=q_high,
        )
    )
    return summary


def collect_predictions(
    *,
    checkpoint_path: Path,
    split: str,
    device_name: str,
    batch_size: int,
    num_workers: int | str,
    amp_dtype: str,
) -> tuple[PredictionSet, dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    device = resolve_device(device_name)
    model = model.to(device)
    model.eval()
    ticker_registry = resolve_checkpoint_ticker_registry(config, str(config["timeframe"]))
    train_bundle, val_bundle, test_bundle, _, _, plan = prepare_dataset_splits(
        timeframe=str(config["timeframe"]),
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        tickers=config.get("tickers"),
        limit_tickers=config.get("limit_tickers"),
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type=str(config.get("line_target_type", "raw_future_return")),
        band_target_type=str(config.get("band_target_type", "raw_future_return")),
        ticker_registry=ticker_registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    bundle = {"train": train_bundle, "val": val_bundle, "test": test_bundle}[split]
    loader = make_loader(bundle, batch_size=batch_size, shuffle=False, device=device, num_workers=num_workers)

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
    return predictions, {"config": config, "plan": plan}


def fit_scalar_width_calibration(
    predictions: PredictionSet,
    *,
    target_coverage: float,
) -> dict[str, float]:
    target_breach = max((1.0 - target_coverage) / 2.0, 1e-4)
    eps = 1e-6
    lower_width = torch.clamp(predictions.line - predictions.lower, min=eps)
    upper_width = torch.clamp(predictions.upper - predictions.line, min=eps)
    actual = predictions.band_target

    # 음수 residual은 lower breach, 양수 residual은 upper breach를 만든다.
    lower_score = (predictions.line - actual) / lower_width
    upper_score = (actual - predictions.line) / upper_width
    lower_scale = _quantile(lower_score, 1.0 - target_breach)
    upper_scale = _quantile(upper_score, 1.0 - target_breach)
    return {
        "target_coverage": float(target_coverage),
        "target_each_tail_breach": float(target_breach),
        "lower_scale": float(max(lower_scale, 0.01)) if math.isfinite(lower_scale) else 1.0,
        "upper_scale": float(max(upper_scale, 0.01)) if math.isfinite(upper_scale) else 1.0,
    }


def apply_scalar_width_calibration(
    predictions: PredictionSet,
    calibration: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    line = predictions.line
    lower_width = torch.clamp(line - predictions.lower, min=1e-6)
    upper_width = torch.clamp(predictions.upper - line, min=1e-6)
    lower = line - lower_width * float(calibration["lower_scale"])
    upper = line + upper_width * float(calibration["upper_scale"])
    return line, lower, upper


def fit_conformal_residual_calibration(
    predictions: PredictionSet,
    *,
    target_coverage: float,
) -> dict[str, float]:
    lower_q = max((1.0 - target_coverage) / 2.0, 1e-4)
    upper_q = min(1.0 - lower_q, 0.9999)
    residual = predictions.band_target - predictions.line
    return {
        "target_coverage": float(target_coverage),
        "lower_q": float(lower_q),
        "upper_q": float(upper_q),
        "lower_offset": _quantile(residual, lower_q),
        "upper_offset": _quantile(residual, upper_q),
    }


def apply_conformal_residual_calibration(
    predictions: PredictionSet,
    calibration: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    line = predictions.line
    lower = line + float(calibration["lower_offset"])
    upper = line + float(calibration["upper_offset"])
    return line, lower, upper


def diagnose_gap(val_metrics: dict[str, float | None], test_metrics: dict[str, float | None]) -> dict[str, Any]:
    val_cov = float(val_metrics.get("coverage") or float("nan"))
    test_cov = float(test_metrics.get("coverage") or float("nan"))
    val_width = float(val_metrics.get("avg_band_width") or float("nan"))
    test_width = float(test_metrics.get("avg_band_width") or float("nan"))
    test_upper = float(test_metrics.get("upper_breach_rate") or float("nan"))
    test_lower = float(test_metrics.get("lower_breach_rate") or float("nan"))
    width_ratio = test_width / val_width if math.isfinite(val_width) and val_width != 0 else float("nan")
    if test_upper > test_lower * 1.25:
        asymmetry = "upper_breach_dominant"
    elif test_lower > test_upper * 1.25:
        asymmetry = "lower_breach_dominant"
    else:
        asymmetry = "balanced"
    if test_cov < val_cov - 0.10 and math.isfinite(width_ratio) and 0.85 <= width_ratio <= 1.15:
        cause = "center_or_distribution_shift"
    elif test_cov < val_cov - 0.10 and math.isfinite(width_ratio) and width_ratio < 0.85:
        cause = "test_band_narrower"
    elif test_cov < 0.75:
        cause = "test_undercoverage"
    else:
        cause = "gap_moderate"
    return {
        "coverage_gap": val_cov - test_cov,
        "width_ratio_test_to_val": width_ratio,
        "test_breach_asymmetry": asymmetry,
        "diagnosis": cause,
    }


def evaluate_candidate(
    *,
    name: str,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
    num_workers: int | str,
    amp_dtype: str,
    target_coverage: float,
) -> dict[str, Any]:
    val_predictions, context = collect_predictions(
        checkpoint_path=checkpoint_path,
        split="val",
        device_name=device,
        batch_size=batch_size,
        num_workers=num_workers,
        amp_dtype=amp_dtype,
    )
    test_predictions, _ = collect_predictions(
        checkpoint_path=checkpoint_path,
        split="test",
        device_name=device,
        batch_size=batch_size,
        num_workers=num_workers,
        amp_dtype=amp_dtype,
    )
    config = context["config"]
    q_low = float(config.get("q_low", 0.1))
    q_high = float(config.get("q_high", 0.9))
    line_target_type = str(config.get("line_target_type", "raw_future_return"))
    band_target_type = str(config.get("band_target_type", "raw_future_return"))

    def _summarize(predictions: PredictionSet, line: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        return summarize_predictions(
            predictions,
            line=line,
            lower=lower,
            upper=upper,
            q_low=q_low,
            q_high=q_high,
            line_target_type=line_target_type,
            band_target_type=band_target_type,
        )

    original_val = _summarize(val_predictions, val_predictions.line, val_predictions.lower, val_predictions.upper)
    original_test = _summarize(test_predictions, test_predictions.line, test_predictions.lower, test_predictions.upper)

    scalar = fit_scalar_width_calibration(val_predictions, target_coverage=target_coverage)
    scalar_val = _summarize(val_predictions, *apply_scalar_width_calibration(val_predictions, scalar))
    scalar_test = _summarize(test_predictions, *apply_scalar_width_calibration(test_predictions, scalar))

    conformal = fit_conformal_residual_calibration(val_predictions, target_coverage=target_coverage)
    conformal_val = _summarize(val_predictions, *apply_conformal_residual_calibration(val_predictions, conformal))
    conformal_test = _summarize(test_predictions, *apply_conformal_residual_calibration(test_predictions, conformal))

    return {
        "name": name,
        "checkpoint_path": str(checkpoint_path),
        "run_id": checkpoint_path.stem.split("_")[-1],
        "config": {
            "model": config.get("model"),
            "timeframe": config.get("timeframe"),
            "seq_len": config.get("seq_len"),
            "horizon": config.get("horizon"),
            "q_low": q_low,
            "q_high": q_high,
            "band_mode": config.get("band_mode"),
            "checkpoint_selection": config.get("checkpoint_selection"),
            "feature_version": config.get("feature_version"),
        },
        "original": {
            "val": original_val,
            "test": original_test,
            "gap_diagnosis": diagnose_gap(original_val, original_test),
        },
        "scalar_width": {
            "calibration": scalar,
            "val": scalar_val,
            "test": scalar_test,
            "gap_diagnosis": diagnose_gap(scalar_val, scalar_test),
        },
        "conformal_residual": {
            "calibration": conformal,
            "val": conformal_val,
            "test": conformal_test,
            "gap_diagnosis": diagnose_gap(conformal_val, conformal_test),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="밴드 calibration 후처리 smoke")
    parser.add_argument("--candidate", action="append", required=True, help="name=checkpoint_path 형식")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--target-coverage", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: dict[str, Path] = {}
    for raw in args.candidate:
        if "=" not in raw:
            raise ValueError("--candidate는 name=checkpoint_path 형식이어야 합니다.")
        name, path = raw.split("=", 1)
        candidates[name] = Path(path)

    result = {
        "target_coverage": args.target_coverage,
        "calibration_methods": ["scalar_width", "conformal_residual"],
        "candidates": {},
    }
    for name, checkpoint_path in candidates.items():
        result["candidates"][name] = evaluate_candidate(
            name=name,
            checkpoint_path=checkpoint_path,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp_dtype=args.amp_dtype,
            target_coverage=args.target_coverage,
        )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "candidate_count": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
