from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import pandas as pd


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_forecast_metrics(
    *,
    metadata: pd.DataFrame | None,
    line_predictions: torch.Tensor,
    lower_predictions: torch.Tensor,
    upper_predictions: torch.Tensor,
    line_targets: torch.Tensor,
    band_targets: torch.Tensor,
    raw_future_returns: torch.Tensor,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
    top_k_frac: float = 0.1,
    fee_bps: float = 10.0,
) -> dict[str, float | None]:
    line_pred = line_predictions.detach().cpu().to(torch.float32)
    lower = lower_predictions.detach().cpu().to(torch.float32)
    upper = upper_predictions.detach().cpu().to(torch.float32)
    line_actual = line_targets.detach().cpu().to(torch.float32)
    band_actual = band_targets.detach().cpu().to(torch.float32)
    raw_actual = raw_future_returns.detach().cpu().to(torch.float32)

    absolute_error = torch.abs(line_pred - line_actual)
    smape = 2.0 * absolute_error / (line_pred.abs() + line_actual.abs() + 1e-6)
    if line_target_type == "direction_label":
        direction_threshold = 0.5
    else:
        direction_threshold = 0.0

    summary: dict[str, float | None] = {
        "coverage": float(((band_actual >= lower) & (band_actual <= upper)).float().mean().item()),
        "avg_band_width": float((upper - lower).mean().item()),
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
        "direction_accuracy": float(
            ((line_pred[:, -1] >= direction_threshold) == (line_actual[:, -1] >= direction_threshold)).float().mean().item()
        ),
    }
    if line_target_type == "raw_future_return":
        signed_error = line_pred - raw_actual
        over_mask = signed_error > 0
        summary["mean_signed_error"] = float(signed_error.mean().item())
        summary["overprediction_rate"] = float(over_mask.float().mean().item())
        summary["mean_overprediction"] = float(signed_error[over_mask].mean().item()) if bool(over_mask.any()) else 0.0
    else:
        summary["mean_signed_error"] = None
        summary["overprediction_rate"] = None
        summary["mean_overprediction"] = None

    if metadata is None or metadata.empty:
        summary.update(
            {
                "spearman_ic": None,
                "top_k_long_spread": None,
                "top_k_short_spread": None,
                "long_short_spread": None,
                "fee_adjusted_return": None,
                "fee_adjusted_sharpe": None,
                "fee_adjusted_turnover": None,
            }
        )
        return summary

    eval_frame = metadata.reset_index(drop=True).copy()
    eval_frame["score"] = line_pred[:, -1].numpy()
    eval_frame["actual_return"] = raw_actual[:, -1].numpy()
    eval_frame["band_width"] = (upper[:, -1] - lower[:, -1]).numpy()

    ic_values: list[float] = []
    long_spreads: list[float] = []
    short_spreads: list[float] = []
    long_short_spreads: list[float] = []
    net_returns: list[float] = []
    turnover_values: list[float] = []
    previous_weights: dict[str, float] = {}
    fee_rate = float(fee_bps) / 10000.0

    for asof_date, group in eval_frame.groupby("asof_date", sort=True):
        del asof_date
        if len(group) >= 2:
            ic = group["score"].corr(group["actual_return"], method="spearman")
            if pd.notna(ic):
                ic_values.append(float(ic))

        if group.empty:
            continue

        top_k = max(int(math.ceil(len(group) * top_k_frac)), 1)
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        long_group = ranked.head(top_k)
        short_group = ranked.tail(top_k)

        long_mean = float(long_group["actual_return"].mean())
        short_mean = float(short_group["actual_return"].mean())
        long_short = long_mean - short_mean

        long_spreads.append(long_mean)
        short_spreads.append(short_mean)
        long_short_spreads.append(long_short)

        weights: dict[str, float] = {}
        long_weight = 1.0 / max(len(long_group), 1)
        short_weight = -1.0 / max(len(short_group), 1)
        for _, row in long_group.iterrows():
            weights[str(row["ticker"])] = weights.get(str(row["ticker"]), 0.0) + long_weight
        for _, row in short_group.iterrows():
            weights[str(row["ticker"])] = weights.get(str(row["ticker"]), 0.0) + short_weight

        turnover = sum(abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in set(weights) | set(previous_weights))
        gross_return = 0.0
        for _, row in group.iterrows():
            gross_return += weights.get(str(row["ticker"]), 0.0) * float(row["actual_return"])
        net_returns.append(gross_return - (turnover * fee_rate))
        turnover_values.append(turnover)
        previous_weights = weights

    net_return_array = np.asarray(net_returns, dtype=np.float64)
    if net_return_array.size > 0:
        cumulative = float(np.prod(1.0 + net_return_array) - 1.0)
        volatility = float(np.std(net_return_array, ddof=0))
        sharpe = float(np.mean(net_return_array) / volatility) if volatility > 0 else 0.0
    else:
        cumulative = None
        sharpe = None

    summary.update(
        {
            "spearman_ic": _safe_mean(ic_values),
            "top_k_long_spread": _safe_mean(long_spreads),
            "top_k_short_spread": _safe_mean(short_spreads),
            "long_short_spread": _safe_mean(long_short_spreads),
            "fee_adjusted_return": cumulative,
            "fee_adjusted_sharpe": sharpe,
            "fee_adjusted_turnover": _safe_mean(turnover_values),
        }
    )
    return summary


def build_single_sample_evaluation(
    *,
    actual_series: list[float],
    line_series: list[float],
    lower_series: list[float],
    upper_series: list[float],
    line_target_type: str = "raw_future_return",
) -> dict[str, float]:
    actual_tensor = torch.tensor(actual_series, dtype=torch.float32)
    line_tensor = torch.tensor(line_series, dtype=torch.float32)
    lower_tensor = torch.tensor(lower_series, dtype=torch.float32)
    upper_tensor = torch.tensor(upper_series, dtype=torch.float32)
    absolute_error = torch.abs(line_tensor - actual_tensor)
    smape = 2.0 * absolute_error / (line_tensor.abs() + actual_tensor.abs() + 1e-6)
    if line_target_type == "direction_label":
        direction_threshold = 0.5
    else:
        direction_threshold = 0.0
    return {
        "coverage": float(((actual_tensor >= lower_tensor) & (actual_tensor <= upper_tensor)).float().mean().item()),
        "avg_band_width": float((upper_tensor - lower_tensor).mean().item()),
        "direction_accuracy": float(((line_tensor >= direction_threshold) == (actual_tensor >= direction_threshold)).float().mean().item()),
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
    }
