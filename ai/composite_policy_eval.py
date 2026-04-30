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
from ai.inference import load_checkpoint_config  # noqa: E402


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


def _line_inside_metrics(
    *,
    line: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, float | bool]:
    inside = (lower <= line) & (line <= upper)
    return {
        "line_inside_band_ratio": float(inside.all(dim=1).to(torch.float32).mean().item()),
        "line_inside_band_point_ratio": float(inside.to(torch.float32).mean().item()),
        "line_inside_band_count": int(inside.all(dim=1).sum().item()),
        "lower_le_upper_all": bool(torch.all(lower <= upper).item()),
    }


def _policy_bounds(
    *,
    policy: str,
    patch_line: torch.Tensor,
    band_line: torch.Tensor,
    calibrated_lower: torch.Tensor,
    calibrated_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    downside_width = torch.clamp(band_line - calibrated_lower, min=1e-6)
    upside_width = torch.clamp(calibrated_upper - band_line, min=1e-6)

    if policy == "raw_composite":
        return calibrated_lower, calibrated_upper, "CP42 방식 그대로: PatchTST line과 CNN-LSTM calibrated band를 중심 재정렬 없이 조합"
    if policy == "include_line_clamp":
        return (
            torch.minimum(calibrated_lower, patch_line),
            torch.maximum(calibrated_upper, patch_line),
            "line이 항상 밴드 안에 들어오도록 하방/상방을 필요한 만큼 확장",
        )
    if policy == "line_centered_asymmetric":
        return (
            patch_line - downside_width,
            patch_line + upside_width,
            "CNN-LSTM의 보정된 하방폭/상방폭은 유지하고 중심만 PatchTST line으로 이동",
        )
    if policy == "risk_first_lower_preserve":
        return (
            torch.minimum(calibrated_lower, patch_line),
            torch.maximum(calibrated_upper, patch_line),
            "하방은 더 보수적인 쪽을 채택하고 상방도 line 포함을 보장",
        )
    raise ValueError(f"알 수 없는 composite policy입니다: {policy}")


def _passes_cp43(metrics: dict[str, Any]) -> dict[str, bool]:
    coverage = float(metrics["coverage"])
    lower_breach = float(metrics["lower_breach_rate"])
    upper_breach = float(metrics["upper_breach_rate"])
    inside = float(metrics["line_inside_band_ratio"])
    width_ratio = float(metrics["width_ratio_vs_raw"])
    return {
        "line_inside_band_ratio": inside >= 0.95,
        "coverage": 0.75 <= coverage <= 0.90,
        "lower_breach_rate": lower_breach <= 0.12,
        "upper_breach_rate": upper_breach <= 0.15,
        "width_not_excessive": width_ratio <= 1.25,
    }


def evaluate_composite_policies(
    *,
    line_checkpoint: Path,
    band_checkpoint: Path,
    split: str,
    tickers: list[str] | None,
    limit_tickers: int | None,
    max_rows: int,
    device_name: str,
    batch_size: int,
    amp_dtype: str,
    lower_scale: float,
    upper_scale: float,
    output_json: Path,
) -> dict[str, Any]:
    line_config = load_checkpoint_config(line_checkpoint)
    band_config = load_checkpoint_config(band_checkpoint)
    line_predictions = collect_predictions(
        checkpoint_path=line_checkpoint,
        split=split,
        tickers=tickers,
        limit_tickers=limit_tickers,
        device_name=device_name,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    band_predictions = collect_predictions(
        checkpoint_path=band_checkpoint,
        split=split,
        tickers=tickers,
        limit_tickers=limit_tickers,
        device_name=device_name,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )

    selected_indices = _select_latest_common_indices(line_predictions, band_predictions, max_rows=max_rows)
    line_indices = [line_index for line_index, _ in selected_indices]
    band_indices = [band_index for _, band_index in selected_indices]

    patch_line = line_predictions.line[line_indices]
    line_targets = line_predictions.line_target[line_indices]
    band_targets = line_predictions.band_target[line_indices]
    raw_future_returns = line_predictions.raw_future_returns[line_indices]
    metadata = line_predictions.metadata.iloc[line_indices].reset_index(drop=True)

    band_line = band_predictions.line[band_indices]
    calibrated_lower, calibrated_upper = _apply_scalar_width(
        line=band_line,
        lower=band_predictions.lower[band_indices],
        upper=band_predictions.upper[band_indices],
        lower_scale=lower_scale,
        upper_scale=upper_scale,
    )

    q_low = float(band_config.get("q_low", 0.1))
    q_high = float(band_config.get("q_high", 0.9))
    policies = [
        "raw_composite",
        "include_line_clamp",
        "line_centered_asymmetric",
        "risk_first_lower_preserve",
    ]

    policy_results: dict[str, Any] = {}
    raw_width: float | None = None
    for policy in policies:
        lower, upper, description = _policy_bounds(
            policy=policy,
            patch_line=patch_line,
            band_line=band_line,
            calibrated_lower=calibrated_lower,
            calibrated_upper=calibrated_upper,
        )
        summary = summarize_forecast_metrics(
            metadata=metadata,
            line_predictions=patch_line,
            lower_predictions=lower,
            upper_predictions=upper,
            line_targets=line_targets,
            band_targets=band_targets,
            raw_future_returns=raw_future_returns,
            line_target_type=str(line_config.get("line_target_type", "raw_future_return")),
            band_target_type=str(line_config.get("band_target_type", "raw_future_return")),
            q_low=q_low,
            q_high=q_high,
            severe_downside_threshold=line_config.get("severe_downside_threshold"),
            squeeze_breakout_threshold=band_config.get("squeeze_breakout_threshold") or line_config.get("squeeze_breakout_threshold"),
        )
        summary.update(
            _band_loss(
                lower=lower,
                upper=upper,
                actual=band_targets,
                q_low=q_low,
                q_high=q_high,
            )
        )
        summary.update(_line_inside_metrics(line=patch_line, lower=lower, upper=upper))
        if policy == "raw_composite":
            raw_width = float(summary["avg_band_width"])
        width_ratio = float(summary["avg_band_width"]) / raw_width if raw_width else 1.0
        summary["width_ratio_vs_raw"] = width_ratio
        summary["description"] = description
        summary["cp43_pass_flags"] = _passes_cp43(summary)
        summary["cp43_all_pass"] = all(summary["cp43_pass_flags"].values())
        policy_results[policy] = summary

    result = {
        "cp": "CP43-M",
        "scope": "PatchTST line + CNN-LSTM calibrated band composite policy comparison",
        "line_checkpoint": _repo_path(line_checkpoint),
        "band_checkpoint": _repo_path(band_checkpoint),
        "split": split,
        "limit_tickers": limit_tickers,
        "max_rows": max_rows,
        "row_count": len(metadata),
        "calibration": {
            "method": "scalar_width",
            "lower_scale": float(lower_scale),
            "upper_scale": float(upper_scale),
        },
        "policy_results": policy_results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP43 composite band 정책 비교")
    parser.add_argument("--line-checkpoint", required=True)
    parser.add_argument("--band-checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit-tickers", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp-dtype", default="off", choices=["off", "bf16", "fp16"])
    parser.add_argument("--lower-scale", type=float, required=True)
    parser.add_argument("--upper-scale", type=float, required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_composite_policies(
        line_checkpoint=Path(args.line_checkpoint),
        band_checkpoint=Path(args.band_checkpoint),
        split=args.split,
        tickers=args.tickers,
        limit_tickers=args.limit_tickers,
        max_rows=args.max_rows,
        device_name=args.device,
        batch_size=args.batch_size,
        amp_dtype=args.amp_dtype,
        lower_scale=args.lower_scale,
        upper_scale=args.upper_scale,
        output_json=Path(args.output_json),
    )
    table = {
        name: {
            key: metrics.get(key)
            for key in (
                "line_inside_band_ratio",
                "coverage",
                "lower_breach_rate",
                "upper_breach_rate",
                "avg_band_width",
                "width_ratio_vs_raw",
                "spearman_ic",
                "long_short_spread",
                "fee_adjusted_return",
                "cp43_all_pass",
            )
        }
        for name, metrics in result["policy_results"].items()
    }
    print(json.dumps(_json_safe({"row_count": result["row_count"], "policies": table}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
