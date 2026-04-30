from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.composite_inference import (  # noqa: E402
    _apply_scalar_width,
    _json_safe,
    _repo_path,
    _select_latest_common_indices,
    collect_predictions,
)
from ai.evaluation import summarize_forecast_metrics  # noqa: E402


def _finite_quantile(values: torch.Tensor, q: float) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 1.0
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
    return {"band_loss": float((low_loss + high_loss).mean().item())}


def _line_inside_ratio(line: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    inside = (lower <= line) & (line <= upper)
    return float(inside.all(dim=1).to(torch.float32).mean().item())


def _width_metrics(
    *,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    base_lower: torch.Tensor,
    base_upper: torch.Tensor,
) -> dict[str, float | bool]:
    downside_width = torch.clamp(line - lower, min=0.0)
    upside_width = torch.clamp(upper - line, min=0.0)
    base_width = torch.clamp(base_upper - base_lower, min=1e-6)
    width = upper - lower
    lower_delta = lower - base_lower
    return {
        "downside_width": float(downside_width.mean().item()),
        "upside_width": float(upside_width.mean().item()),
        "avg_band_width": float(width.mean().item()),
        "width_increase_ratio": float((width.mean() / base_width.mean()).item()),
        "line_inside_band_ratio": _line_inside_ratio(line, lower, upper),
        "lower_le_upper_all": bool(torch.all(lower <= upper).item()),
        "conservative_series_changed": bool(torch.any(torch.abs(lower_delta) > 1e-8).item()),
        "lower_less_conservative": bool(torch.any(lower > base_lower + 1e-8).item()),
    }


def _policy_bounds(
    *,
    policy: str,
    line: torch.Tensor,
    base_lower: torch.Tensor,
    base_upper: torch.Tensor,
    fit_params: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    base_downside = torch.clamp(line - base_lower, min=1e-6)
    base_upside = torch.clamp(base_upper - line, min=1e-6)
    if policy == "risk_first_lower_preserve":
        return base_lower, base_upper
    if policy == "risk_first_upper_buffer_1.10":
        return base_lower, line + base_upside * 1.10
    if policy == "risk_first_upper_buffer_1.25":
        return base_lower, line + base_upside * 1.25
    if policy == "asymmetric_quantile_expand":
        return base_lower, line + base_upside * float(fit_params["upper_expand_scale"])
    if policy == "symmetric_width_expand":
        scale = float(fit_params["symmetric_expand_scale"])
        return line - base_downside * scale, line + base_upside * scale
    raise ValueError(f"지원하지 않는 CP46 정책입니다: {policy}")


def _fit_policy_params(
    *,
    line: torch.Tensor,
    base_lower: torch.Tensor,
    base_upper: torch.Tensor,
    actual: torch.Tensor,
    target_upper_breach: float,
    target_coverage: float,
) -> dict[str, float]:
    base_downside = torch.clamp(line - base_lower, min=1e-6)
    base_upside = torch.clamp(base_upper - line, min=1e-6)
    upper_score = (actual - line) / base_upside
    lower_score = (line - actual) / base_downside
    symmetric_score = torch.maximum(upper_score, lower_score)
    return {
        "upper_expand_scale": max(_finite_quantile(upper_score, 1.0 - target_upper_breach), 1.0),
        "symmetric_expand_scale": max(_finite_quantile(symmetric_score, target_coverage), 1.0),
        "target_upper_breach": float(target_upper_breach),
        "target_coverage": float(target_coverage),
    }


def _select_tensors(
    *,
    line_checkpoint: Path,
    band_checkpoint: Path,
    split: str,
    limit_tickers: int,
    max_rows: int,
    device: str,
    batch_size: int,
    amp_dtype: str,
    lower_scale: float,
    upper_scale: float,
) -> dict[str, Any]:
    line_predictions = collect_predictions(
        checkpoint_path=line_checkpoint,
        split=split,
        tickers=None,
        limit_tickers=limit_tickers,
        device_name=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    band_predictions = collect_predictions(
        checkpoint_path=band_checkpoint,
        split=split,
        tickers=None,
        limit_tickers=limit_tickers,
        device_name=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    selected = _select_latest_common_indices(line_predictions, band_predictions, max_rows=max_rows)
    line_indices = [line_index for line_index, _ in selected]
    band_indices = [band_index for _, band_index in selected]

    line = line_predictions.line[line_indices]
    band_line = band_predictions.line[band_indices]
    calibrated_lower, calibrated_upper = _apply_scalar_width(
        line=band_line,
        lower=band_predictions.lower[band_indices],
        upper=band_predictions.upper[band_indices],
        lower_scale=lower_scale,
        upper_scale=upper_scale,
    )
    base_lower = torch.minimum(calibrated_lower, line)
    base_upper = torch.maximum(calibrated_upper, line)
    return {
        "line": line,
        "base_lower": base_lower,
        "base_upper": base_upper,
        "line_targets": line_predictions.line_target[line_indices],
        "band_targets": line_predictions.band_target[line_indices],
        "raw_future_returns": line_predictions.raw_future_returns[line_indices],
        "metadata": line_predictions.metadata.iloc[line_indices].reset_index(drop=True),
        "line_config": line_predictions.config,
        "band_config": band_predictions.config,
        "row_count": len(selected),
    }


def _summarize_policy(
    *,
    tensors: dict[str, Any],
    policy: str,
    fit_params: dict[str, float],
) -> dict[str, Any]:
    lower, upper = _policy_bounds(
        policy=policy,
        line=tensors["line"],
        base_lower=tensors["base_lower"],
        base_upper=tensors["base_upper"],
        fit_params=fit_params,
    )
    line_config = tensors["line_config"]
    band_config = tensors["band_config"]
    summary = summarize_forecast_metrics(
        metadata=tensors["metadata"],
        line_predictions=tensors["line"],
        lower_predictions=lower,
        upper_predictions=upper,
        line_targets=tensors["line_targets"],
        band_targets=tensors["band_targets"],
        raw_future_returns=tensors["raw_future_returns"],
        line_target_type=str(line_config.get("line_target_type", "raw_future_return")),
        band_target_type=str(line_config.get("band_target_type", "raw_future_return")),
        q_low=float(band_config.get("q_low", 0.1)),
        q_high=float(band_config.get("q_high", 0.9)),
        severe_downside_threshold=line_config.get("severe_downside_threshold"),
        squeeze_breakout_threshold=band_config.get("squeeze_breakout_threshold") or line_config.get("squeeze_breakout_threshold"),
    )
    summary.update(
        _band_loss(
            lower=lower,
            upper=upper,
            actual=tensors["band_targets"],
            q_low=float(band_config.get("q_low", 0.1)),
            q_high=float(band_config.get("q_high", 0.9)),
        )
    )
    summary.update(
        _width_metrics(
            line=tensors["line"],
            lower=lower,
            upper=upper,
            base_lower=tensors["base_lower"],
            base_upper=tensors["base_upper"],
        )
    )
    summary["pass_flags"] = {
        "coverage": 0.75 <= float(summary["coverage"]) <= 0.90,
        "upper_breach_rate": float(summary["upper_breach_rate"]) <= 0.15,
        "lower_breach_rate": float(summary["lower_breach_rate"]) <= 0.12,
        "line_inside_band_ratio": float(summary["line_inside_band_ratio"]) >= 0.95,
        "lower_not_less_conservative": not bool(summary["lower_less_conservative"]),
    }
    summary["all_pass"] = all(summary["pass_flags"].values())
    return summary


def evaluate_upper_calibration(
    *,
    line_checkpoint: Path,
    band_checkpoint: Path,
    limit_tickers: int,
    max_rows: int,
    device: str,
    batch_size: int,
    amp_dtype: str,
    lower_scale: float,
    upper_scale: float,
    output_json: Path,
) -> dict[str, Any]:
    val_tensors = _select_tensors(
        line_checkpoint=line_checkpoint,
        band_checkpoint=band_checkpoint,
        split="val",
        limit_tickers=limit_tickers,
        max_rows=max_rows,
        device=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        lower_scale=lower_scale,
        upper_scale=upper_scale,
    )
    test_tensors = _select_tensors(
        line_checkpoint=line_checkpoint,
        band_checkpoint=band_checkpoint,
        split="test",
        limit_tickers=limit_tickers,
        max_rows=max_rows,
        device=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        lower_scale=lower_scale,
        upper_scale=upper_scale,
    )
    fit_params = _fit_policy_params(
        line=val_tensors["line"],
        base_lower=val_tensors["base_lower"],
        base_upper=val_tensors["base_upper"],
        actual=val_tensors["band_targets"],
        target_upper_breach=0.125,
        target_coverage=0.85,
    )
    policies = [
        "risk_first_lower_preserve",
        "risk_first_upper_buffer_1.10",
        "risk_first_upper_buffer_1.25",
        "asymmetric_quantile_expand",
        "symmetric_width_expand",
    ]
    result = {
        "cp": "CP46-M",
        "line_checkpoint": _repo_path(line_checkpoint),
        "band_checkpoint": _repo_path(band_checkpoint),
        "limit_tickers": limit_tickers,
        "max_rows": max_rows,
        "val_row_count": val_tensors["row_count"],
        "test_row_count": test_tensors["row_count"],
        "band_scalar_calibration": {
            "lower_scale": float(lower_scale),
            "upper_scale": float(upper_scale),
        },
        "fit_params": fit_params,
        "policies": {
            policy: {
                "validation": _summarize_policy(tensors=val_tensors, policy=policy, fit_params=fit_params),
                "test": _summarize_policy(tensors=test_tensors, policy=policy, fit_params=fit_params),
            }
            for policy in policies
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP46 composite upper calibration 비교")
    parser.add_argument("--line-checkpoint", required=True)
    parser.add_argument("--band-checkpoint", required=True)
    parser.add_argument("--limit-tickers", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--lower-scale", type=float, required=True)
    parser.add_argument("--upper-scale", type=float, required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_upper_calibration(
        line_checkpoint=Path(args.line_checkpoint),
        band_checkpoint=Path(args.band_checkpoint),
        limit_tickers=args.limit_tickers,
        max_rows=args.max_rows,
        device=args.device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
        lower_scale=args.lower_scale,
        upper_scale=args.upper_scale,
        output_json=Path(args.output_json),
    )
    table = {
        policy: {
            "coverage": values["test"]["coverage"],
            "lower_breach_rate": values["test"]["lower_breach_rate"],
            "upper_breach_rate": values["test"]["upper_breach_rate"],
            "avg_band_width": values["test"]["avg_band_width"],
            "width_increase_ratio": values["test"]["width_increase_ratio"],
            "line_inside_band_ratio": values["test"]["line_inside_band_ratio"],
            "all_pass": values["test"]["all_pass"],
        }
        for policy, values in result["policies"].items()
    }
    print(json.dumps(_json_safe({"test_row_count": result["test_row_count"], "policies": table}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
