from __future__ import annotations

import math
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()
import numpy as np
import pandas as pd


BAND_METRIC_NAMES = (
    "nominal_coverage",
    "empirical_coverage",
    "coverage",
    "coverage_error",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "avg_band_width",
    "lower_breach_error",
    "lower_breach_abs_error",
    "upper_breach_error",
    "upper_breach_abs_error",
    "median_band_width",
    "p90_band_width",
    "asymmetric_interval_score",
    "interval_score",
    "interval_width_component",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "empirical_q_low",
    "empirical_q_high",
    "empirical_p10",
    "empirical_p25",
    "empirical_p50",
    "empirical_p75",
    "empirical_p90",
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_low_realized_vol",
    "width_bucket_mid_realized_vol",
    "width_bucket_high_realized_vol",
    "width_bucket_low_downside_rate",
    "width_bucket_mid_downside_rate",
    "width_bucket_high_downside_rate",
    "width_bucket_realized_vol_ratio",
    "width_bucket_downside_rate_ratio",
    "squeeze_breakout_rate",
)

LINE_METRIC_NAMES = (
    "spearman_ic",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "top_k_long_spread",
    "top_k_short_spread",
    "long_short_spread",
    "spread_mean",
    "spread_std",
    "spread_ir",
    "spread_t_stat",
    "direction_accuracy",
    "mae",
    "smape",
    "mean_signed_error",
    "overprediction_rate",
    "mean_overprediction",
    "underprediction_rate",
    "mean_underprediction",
    "false_safe_rate",
    "false_safe_negative_rate",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
    "fee_adjusted_return",
    "fee_adjusted_sharpe",
    "fee_adjusted_turnover",
)

RISK_METRIC_NAMES = (
    "risk_threshold",
    "risk_decision_threshold",
    "risk_prevalence",
    "risk_recall",
    "risk_precision",
    "risk_false_alarm_rate",
    "risk_accuracy",
    "risk_roc_auc",
    "risk_pr_auc",
    "risk_calibration_error",
    "dual_head_false_safe_rate",
    "line_positive_risk_low_tail_rate",
    "line_positive_risk_high_tail_rate",
    "risk_separation_gap",
)

REGIME_METRIC_NAMES = (
    "regime_accuracy",
    "regime_adjacent_accuracy",
    "regime_ordinal_mae",
    "regime_ordinal_mse",
    "regime_macro_f1",
    "strong_down_recall",
    "strong_down_precision",
    "strong_up_precision",
    "regime_return_monotonicity",
    "regime_return_monotonic_non_decreasing",
    "line_positive_false_safe_rate",
    "line_top_decile_false_safe_rate",
    "regime_filtered_false_safe_reduction",
    "spread_retention",
    "line_positive_regime_safe_actual_down_rate",
    "line_positive_regime_risky_actual_down_rate",
    "regime_risk_line_corr_pearson",
    "regime_risk_line_corr_spearman",
    "regime_expected_class_line_corr_pearson",
    "regime_expected_class_line_corr_spearman",
    "regime_risk_residual_severe_auc",
    "top_false_safe_date_max_share",
    "top_false_safe_ticker_max_share",
)

LEGACY_OVERLAY_DIAGNOSTIC_NAMES = (
    "line_inside_band_ratio",
    "line_inside_band_point_ratio",
    "product_display_warning_rate",
    "conservative_series_false_safe_rate",
)


def _is_legacy_overlay_metric(key: str) -> bool:
    return any(key == name or key.endswith(f"_{name}") for name in LEGACY_OVERLAY_DIAGNOSTIC_NAMES)


def _metric_layer_payload(summary: dict[str, Any], names: tuple[str, ...], *, kind: str) -> dict[str, Any]:
    payload = {key: summary.get(key) for key in names if key in summary}
    for key, value in summary.items():
        if kind == "line" and (
            key.startswith(("all_horizon_", "h1_h5_", "h6_h10_", "h11_h20_"))
            and "_band_" not in key
            and not _is_legacy_overlay_metric(key)
        ):
            payload[key] = value
        if kind == "band" and (
            key.startswith(("all_horizon_band_", "h1_h5_band_", "h6_h10_band_", "h11_h20_band_"))
            and not _is_legacy_overlay_metric(key)
        ):
            payload[key] = value
    return payload


def _attach_metric_layers(
    summary: dict[str, Any],
    *,
    include_legacy_overlay_diagnostics: bool,
) -> dict[str, Any]:
    legacy = {key: value for key, value in summary.items() if _is_legacy_overlay_metric(key)}
    if not include_legacy_overlay_diagnostics:
        for key in legacy:
            summary.pop(key, None)
    summary["line_metrics"] = _metric_layer_payload(summary, LINE_METRIC_NAMES, kind="line")
    summary["band_metrics"] = _metric_layer_payload(summary, BAND_METRIC_NAMES, kind="band")
    summary["risk_metrics"] = {key: summary.get(key) for key in RISK_METRIC_NAMES if key in summary}
    summary["regime_metrics"] = {key: summary.get(key) for key in REGIME_METRIC_NAMES if key in summary}
    if include_legacy_overlay_diagnostics:
        summary["legacy_overlay_diagnostics"] = legacy
    return summary


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _safe_ir(mean_value: float | None, std_value: float | None) -> float | None:
    if mean_value is None or std_value is None or std_value <= 0:
        return None
    return float(mean_value / std_value)


def _safe_t_stat(mean_value: float | None, std_value: float | None, count: int) -> float | None:
    if mean_value is None or std_value is None or std_value <= 0 or count < 2:
        return None
    return float(mean_value / (std_value / math.sqrt(count)))


def _safe_rate(mask: torch.Tensor, denominator: torch.Tensor) -> float | None:
    count = int(denominator.sum().item())
    if count == 0:
        return None
    return float((mask & denominator).float().sum().item() / count)


def _severe_downside_threshold(start: int, end: int) -> float:
    if end <= 5:
        return 0.05
    if start < 10 and end <= 10:
        return 0.08
    return 0.12


def _cross_section_line_risk_metrics(
    *,
    metadata: pd.DataFrame | None,
    score: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, Any]:
    if metadata is None or metadata.empty or len(metadata) != int(score.numel()):
        return {"false_safe_tail_rate": None, "downside_capture_rate": None}

    frame = metadata.reset_index(drop=True).copy()
    if "asof_date" not in frame.columns:
        return {"false_safe_tail_rate": None, "downside_capture_rate": None}
    frame["score"] = score.detach().cpu().to(torch.float32).reshape(-1).numpy()
    frame["actual"] = actual.detach().cpu().to(torch.float32).reshape(-1).numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "actual", "asof_date"])
    if frame.empty:
        return {"false_safe_tail_rate": None, "downside_capture_rate": None}

    false_safe_count = 0
    capture_count = 0
    tail_count = 0
    for _, group in frame.groupby("asof_date", sort=True):
        if len(group) < 2:
            continue
        actual_tail_cutoff = group["actual"].quantile(0.20)
        score_tail_cutoff = group["score"].quantile(0.20)
        tail = group["actual"] <= actual_tail_cutoff
        group_tail_count = int(tail.sum())
        if group_tail_count == 0:
            continue
        tail_count += group_tail_count
        false_safe_count += int(((group["score"] >= 0.0) & tail).sum())
        capture_count += int(((group["score"] <= score_tail_cutoff) & tail).sum())

    if tail_count == 0:
        return {"false_safe_tail_rate": None, "downside_capture_rate": None}
    return {
        "false_safe_tail_rate": float(false_safe_count / tail_count),
        "downside_capture_rate": float(capture_count / tail_count),
    }


def _conservative_line_metrics(
    *,
    score: torch.Tensor,
    actual: torch.Tensor,
    severe_threshold: float,
    metadata: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    score = score.detach().cpu().to(torch.float32).reshape(-1)
    actual = actual.detach().cpu().to(torch.float32).reshape(-1)
    finite = torch.isfinite(score) & torch.isfinite(actual)
    if not bool(finite.any()):
        return {
            "overprediction_rate": None,
            "mean_overprediction": None,
            "underprediction_rate": None,
            "mean_underprediction": None,
            "downside_capture_rate": None,
            "severe_downside_recall": None,
            "false_safe_rate": None,
            "false_safe_negative_rate": None,
            "false_safe_tail_rate": None,
            "false_safe_severe_rate": None,
            "conservative_bias": None,
            "upside_sacrifice": None,
        }

    score = score[finite]
    actual = actual[finite]
    signed_error = score - actual
    over_mask = signed_error > 0
    under_mask = signed_error < 0
    actual_q20 = torch.quantile(actual, 0.20)
    actual_q80 = torch.quantile(actual, 0.80)
    score_q20 = torch.quantile(score, 0.20)
    downside_mask = actual <= actual_q20
    negative_mask = actual < 0
    risky_mask = downside_mask | negative_mask
    severe_mask = actual <= float(severe_threshold)
    upside_mask = actual >= actual_q80
    cross_section = _cross_section_line_risk_metrics(metadata=metadata, score=score, actual=actual)
    false_safe_tail_rate = cross_section["false_safe_tail_rate"]
    downside_capture_rate = cross_section["downside_capture_rate"]
    if false_safe_tail_rate is None:
        false_safe_tail_rate = _safe_rate(score >= 0, downside_mask)
    if downside_capture_rate is None:
        downside_capture_rate = _safe_rate(score <= score_q20, downside_mask)

    return {
        "overprediction_rate": float(over_mask.float().mean().item()),
        "mean_overprediction": float(signed_error[over_mask].mean().item()) if bool(over_mask.any()) else 0.0,
        "underprediction_rate": float(under_mask.float().mean().item()),
        "mean_underprediction": float(signed_error[under_mask].mean().item()) if bool(under_mask.any()) else 0.0,
        "downside_capture_rate": downside_capture_rate,
        "severe_downside_recall": _safe_rate(score < 0, severe_mask),
        "false_safe_rate": false_safe_tail_rate,
        "false_safe_negative_rate": _safe_rate(score >= 0, negative_mask),
        "false_safe_tail_rate": false_safe_tail_rate,
        "false_safe_severe_rate": _safe_rate(score >= 0, severe_mask),
        "conservative_bias": float(signed_error.mean().item()),
        "upside_sacrifice": float((actual[upside_mask] - score[upside_mask]).mean().item()) if bool(upside_mask.any()) else None,
    }


def _investment_metrics_for_score(
    *,
    metadata: pd.DataFrame | None,
    score: torch.Tensor,
    actual_return: torch.Tensor,
    top_k_frac: float,
    fee_bps: float,
) -> dict[str, float | None]:
    if metadata is None or metadata.empty:
        return {
            "spearman_ic": None,
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "ic_t_stat": None,
            "top_k_long_spread": None,
            "top_k_short_spread": None,
            "long_short_spread": None,
            "spread_mean": None,
            "spread_std": None,
            "spread_ir": None,
            "spread_t_stat": None,
            "fee_adjusted_return": None,
            "fee_adjusted_sharpe": None,
            "fee_adjusted_turnover": None,
        }

    eval_frame = metadata.reset_index(drop=True).copy()
    eval_frame["score"] = score.detach().cpu().to(torch.float32).numpy()
    eval_frame["actual_return"] = actual_return.detach().cpu().to(torch.float32).numpy()

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

        turnover = sum(
            abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
            for ticker in set(weights) | set(previous_weights)
        )
        gross_return = 0.0
        for _, row in group.iterrows():
            gross_return += weights.get(str(row["ticker"]), 0.0) * float(row["actual_return"])
        net_returns.append(gross_return - (turnover * fee_rate))
        turnover_values.append(turnover)
        previous_weights = weights

    net_return_array = np.asarray(net_returns, dtype=np.float64)
    ic_mean = _safe_mean(ic_values)
    ic_std = _safe_std(ic_values)
    spread_mean = _safe_mean(long_short_spreads)
    spread_std = _safe_std(long_short_spreads)
    if net_return_array.size > 0:
        cumulative = float(np.prod(1.0 + net_return_array) - 1.0)
        volatility = float(np.std(net_return_array, ddof=0))
        sharpe = float(np.mean(net_return_array) / volatility) if volatility > 0 else 0.0
    else:
        cumulative = None
        sharpe = None

    return {
        "spearman_ic": ic_mean,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": _safe_ir(ic_mean, ic_std),
        "ic_t_stat": _safe_t_stat(ic_mean, ic_std, len(ic_values)),
        "top_k_long_spread": _safe_mean(long_spreads),
        "top_k_short_spread": _safe_mean(short_spreads),
        "long_short_spread": spread_mean,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "spread_ir": _safe_ir(spread_mean, spread_std),
        "spread_t_stat": _safe_t_stat(spread_mean, spread_std, len(long_short_spreads)),
        "fee_adjusted_return": cumulative,
        "fee_adjusted_sharpe": sharpe,
        "fee_adjusted_turnover": _safe_mean(turnover_values),
    }


def _prefixed(prefix: str, values: dict[str, float | None]) -> dict[str, float | None]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _spearman_corr(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().cpu().to(torch.float32).reshape(-1)
    right = right.detach().cpu().to(torch.float32).reshape(-1)
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum().item()) < 2:
        return None
    frame = pd.DataFrame({"left": left[finite].numpy(), "right": right[finite].numpy()})
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method="spearman")
    return float(value) if pd.notna(value) else None


def _pearson_corr(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().cpu().to(torch.float32).reshape(-1)
    right = right.detach().cpu().to(torch.float32).reshape(-1)
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum().item()) < 2:
        return None
    frame = pd.DataFrame({"left": left[finite].numpy(), "right": right[finite].numpy()})
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method="pearson")
    return float(value) if pd.notna(value) else None


def _linear_residual_auc(score: torch.Tensor, control: torch.Tensor, target: torch.Tensor) -> float | None:
    score = score.detach().cpu().to(torch.float32).reshape(-1)
    control = control.detach().cpu().to(torch.float32).reshape(-1)
    target = target.detach().cpu().to(torch.float32).reshape(-1)
    finite = torch.isfinite(score) & torch.isfinite(control) & torch.isfinite(target)
    if int(finite.sum().item()) < 10:
        return None
    x = control[finite]
    y = score[finite]
    x_centered = x - x.mean()
    denom = torch.sum(x_centered * x_centered)
    if float(denom.item()) <= 1e-12:
        residual = y - y.mean()
    else:
        slope = torch.sum(x_centered * (y - y.mean())) / denom
        intercept = y.mean() - slope * x.mean()
        residual = y - ((slope * x) + intercept)
    return _binary_roc_auc(residual, target[finite])


def _safe_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    selected = values[mask]
    if selected.numel() == 0:
        return None
    return float(selected.mean().item())


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


def _band_metric_defaults(prefix: str | None = None) -> dict[str, float | None]:
    if prefix:
        return {f"{prefix}_{name}": None for name in BAND_METRIC_NAMES}
    return {name: None for name in BAND_METRIC_NAMES}


def _band_interval_metrics(
    *,
    line_pred: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    band_actual: torch.Tensor,
    raw_actual: torch.Tensor,
    q_low: float,
    q_high: float,
    lower_penalty_weight: float,
    upper_penalty_weight: float,
    squeeze_breakout_threshold: float | None = None,
    prefix: str | None = None,
) -> dict[str, float | None]:
    line_flat = line_pred.detach().cpu().to(torch.float32).reshape(-1)
    lower_flat = lower.detach().cpu().to(torch.float32).reshape(-1)
    upper_flat = upper.detach().cpu().to(torch.float32).reshape(-1)
    band_flat = band_actual.detach().cpu().to(torch.float32).reshape(-1)
    raw_flat = raw_actual.detach().cpu().to(torch.float32).reshape(-1)
    finite = (
        torch.isfinite(line_flat)
        & torch.isfinite(lower_flat)
        & torch.isfinite(upper_flat)
        & torch.isfinite(band_flat)
        & torch.isfinite(raw_flat)
    )
    if not bool(finite.any()):
        return _band_metric_defaults(prefix)

    line_flat = line_flat[finite]
    lower_flat = lower_flat[finite]
    upper_flat = upper_flat[finite]
    band_flat = band_flat[finite]
    raw_flat = raw_flat[finite]
    nominal = float(max(min(q_high - q_low, 1.0), 0.0))
    alpha = max(1.0 - nominal, 1e-6)
    covered = (band_flat >= lower_flat) & (band_flat <= upper_flat)
    lower_breach = band_flat < lower_flat
    upper_breach = band_flat > upper_flat
    width = upper_flat - lower_flat
    nonnegative_width = torch.clamp(width, min=0.0)
    lower_excess = torch.relu(lower_flat - band_flat)
    upper_excess = torch.relu(band_flat - upper_flat)
    lower_penalty = lower_penalty_weight * (2.0 / alpha) * lower_excess
    upper_penalty = upper_penalty_weight * (2.0 / alpha) * upper_excess
    empirical = float(covered.float().mean().item())
    downside_width = torch.clamp(line_flat - lower_flat, min=0.0)
    realized_abs = raw_flat.abs()
    downside_realized = torch.relu(-raw_flat)

    q33 = torch.quantile(nonnegative_width, 1.0 / 3.0)
    q66 = torch.quantile(nonnegative_width, 2.0 / 3.0)
    low_bucket = nonnegative_width <= q33
    high_bucket = nonnegative_width >= q66
    mid_bucket = ~(low_bucket | high_bucket)
    low_vol = _safe_tensor_mean(realized_abs, low_bucket)
    mid_vol = _safe_tensor_mean(realized_abs, mid_bucket)
    high_vol = _safe_tensor_mean(realized_abs, high_bucket)
    low_downside = _safe_tensor_mean((raw_flat < 0).to(torch.float32), low_bucket)
    mid_downside = _safe_tensor_mean((raw_flat < 0).to(torch.float32), mid_bucket)
    high_downside = _safe_tensor_mean((raw_flat < 0).to(torch.float32), high_bucket)
    squeeze_threshold = torch.quantile(nonnegative_width, 0.20)
    if squeeze_breakout_threshold is None:
        breakout_threshold = torch.quantile(realized_abs, 0.80)
    else:
        breakout_threshold = torch.tensor(float(squeeze_breakout_threshold), dtype=realized_abs.dtype)
    squeeze_bucket = nonnegative_width <= squeeze_threshold
    inside = (lower_flat <= line_flat) & (line_flat <= upper_flat)
    risky_mask = (raw_flat <= torch.quantile(raw_flat, 0.20)) | (raw_flat < 0)

    metrics: dict[str, float | None] = {
        "nominal_coverage": nominal,
        "empirical_coverage": empirical,
        "coverage_error": empirical - nominal,
        "coverage_abs_error": abs(empirical - nominal),
        "lower_breach_error": float(lower_breach.float().mean().item()) - float(q_low),
        "lower_breach_abs_error": abs(float(lower_breach.float().mean().item()) - float(q_low)),
        "upper_breach_error": float(upper_breach.float().mean().item()) - float(1.0 - q_high),
        "upper_breach_abs_error": abs(float(upper_breach.float().mean().item()) - float(1.0 - q_high)),
        "median_band_width": float(torch.quantile(width, 0.50).item()),
        "p90_band_width": float(torch.quantile(width, 0.90).item()),
        "asymmetric_interval_score": float((nonnegative_width + lower_penalty + upper_penalty).mean().item()),
        "interval_score": float((nonnegative_width + lower_penalty + upper_penalty).mean().item()),
        "interval_width_component": float(nonnegative_width.mean().item()),
        "interval_lower_penalty": float(lower_penalty.mean().item()),
        "interval_upper_penalty": float(upper_penalty.mean().item()),
        "empirical_q_low": float(lower_breach.float().mean().item()),
        "empirical_q_high": float((band_flat <= upper_flat).float().mean().item()),
        "empirical_p10": None,
        "empirical_p25": None,
        "empirical_p50": None,
        "empirical_p75": None,
        "empirical_p90": None,
        "band_width_ic": _spearman_corr(nonnegative_width, realized_abs),
        "downside_width_ic": _spearman_corr(downside_width, downside_realized),
        "width_bucket_low_realized_vol": low_vol,
        "width_bucket_mid_realized_vol": mid_vol,
        "width_bucket_high_realized_vol": high_vol,
        "width_bucket_low_downside_rate": low_downside,
        "width_bucket_mid_downside_rate": mid_downside,
        "width_bucket_high_downside_rate": high_downside,
        "width_bucket_realized_vol_ratio": _safe_ratio(high_vol, low_vol),
        "width_bucket_downside_rate_ratio": _safe_ratio(high_downside, low_downside),
        "squeeze_breakout_rate": _safe_tensor_mean((realized_abs > breakout_threshold).to(torch.float32), squeeze_bucket),
        "line_inside_band_ratio": float(inside.reshape(line_pred.shape).all(dim=1).to(torch.float32).mean().item()),
        "line_inside_band_point_ratio": float(inside.to(torch.float32).mean().item()),
        "product_display_warning_rate": float((~inside | (upper_flat < lower_flat)).to(torch.float32).mean().item()),
        "conservative_series_false_safe_rate": _safe_rate(lower_flat >= 0, risky_mask),
    }
    for quantile, value in ((q_low, metrics["empirical_q_low"]), (q_high, metrics["empirical_q_high"])):
        for canonical in (0.10, 0.25, 0.50, 0.75, 0.90):
            if abs(float(quantile) - canonical) < 1e-8:
                metrics[f"empirical_p{int(canonical * 100)}"] = value
    if prefix:
        return _prefixed(prefix, metrics)
    return metrics


def _band_segment_metrics(
    *,
    prefix: str,
    start: int,
    end: int,
    line_pred: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    band_actual: torch.Tensor,
    raw_actual: torch.Tensor,
    q_low: float,
    q_high: float,
    lower_penalty_weight: float,
    upper_penalty_weight: float,
    squeeze_breakout_threshold: float | None = None,
) -> dict[str, float | None]:
    horizon = int(lower.shape[1])
    if start >= horizon:
        return _band_metric_defaults(prefix)
    stop = min(end, horizon)
    return _band_interval_metrics(
        line_pred=line_pred[:, start:stop],
        lower=lower[:, start:stop],
        upper=upper[:, start:stop],
        band_actual=band_actual[:, start:stop],
        raw_actual=raw_actual[:, start:stop],
        q_low=q_low,
        q_high=q_high,
        lower_penalty_weight=lower_penalty_weight,
        upper_penalty_weight=upper_penalty_weight,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        prefix=prefix,
    )


def _line_segment_metrics(
    *,
    prefix: str,
    start: int,
    end: int,
    metadata: pd.DataFrame | None,
    line_pred: torch.Tensor,
    line_actual: torch.Tensor,
    raw_actual: torch.Tensor,
    severe_downside_threshold: float | None,
    top_k_frac: float,
    fee_bps: float,
) -> dict[str, float | None]:
    horizon = int(line_pred.shape[1])
    if start >= horizon:
        metric_names = (
            "spearman_ic",
            "ic_mean",
            "ic_std",
            "ic_ir",
            "ic_t_stat",
            "top_k_long_spread",
            "top_k_short_spread",
            "long_short_spread",
            "spread_mean",
            "spread_std",
            "spread_ir",
            "spread_t_stat",
            "mae",
            "smape",
            "overprediction_rate",
            "mean_overprediction",
            "underprediction_rate",
            "mean_underprediction",
            "downside_capture_rate",
            "severe_downside_recall",
            "false_safe_rate",
            "false_safe_negative_rate",
            "false_safe_tail_rate",
            "false_safe_severe_rate",
            "conservative_bias",
            "upside_sacrifice",
        )
        return {f"{prefix}_{name}": None for name in metric_names}

    stop = min(end, horizon)
    segment_pred = line_pred[:, start:stop].mean(dim=1)
    segment_actual = line_actual[:, start:stop].mean(dim=1)
    segment_raw = raw_actual[:, start:stop].mean(dim=1)
    absolute_error = torch.abs(segment_pred - segment_actual)
    smape = 2.0 * absolute_error / (segment_pred.abs() + segment_actual.abs() + 1e-6)
    investment = _investment_metrics_for_score(
        metadata=metadata,
        score=segment_pred,
        actual_return=segment_raw,
        top_k_frac=top_k_frac,
        fee_bps=fee_bps,
    )
    conservative = _conservative_line_metrics(
        score=segment_pred,
        actual=segment_raw,
        severe_threshold=(
            float(severe_downside_threshold)
            if severe_downside_threshold is not None
            else -abs(_severe_downside_threshold(start, stop))
        ),
        metadata=metadata,
    )
    return _prefixed(
        prefix,
        {
            **investment,
            "mae": float(absolute_error.mean().item()),
            "smape": float(smape.mean().item()),
            **conservative,
        },
    )


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
    q_low: float = 0.1,
    q_high: float = 0.9,
    interval_lower_penalty_weight: float = 2.0,
    interval_upper_penalty_weight: float = 1.0,
    severe_downside_threshold: float | None = None,
    squeeze_breakout_threshold: float | None = None,
    top_k_frac: float = 0.1,
    fee_bps: float = 10.0,
    include_legacy_overlay_diagnostics: bool = False,
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
        "lower_breach_rate": float((band_actual < lower).float().mean().item()),
        "upper_breach_rate": float((band_actual > upper).float().mean().item()),
        "avg_band_width": float((upper - lower).mean().item()),
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
        "direction_accuracy": float(
            ((line_pred[:, -1] >= direction_threshold) == (line_actual[:, -1] >= direction_threshold)).float().mean().item()
        ),
    }
    summary.update(
        _band_interval_metrics(
            line_pred=line_pred,
            lower=lower,
            upper=upper,
            band_actual=band_actual,
            raw_actual=raw_actual,
            q_low=q_low,
            q_high=q_high,
            lower_penalty_weight=interval_lower_penalty_weight,
            upper_penalty_weight=interval_upper_penalty_weight,
            squeeze_breakout_threshold=squeeze_breakout_threshold,
        )
    )
    if line_target_type == "raw_future_return":
        conservative = _conservative_line_metrics(
            score=line_pred,
            actual=raw_actual,
            severe_threshold=(
                float(severe_downside_threshold)
                if severe_downside_threshold is not None
                else -abs(_severe_downside_threshold(0, int(line_pred.shape[1])))
            ),
            metadata=None,
        )
        summary["mean_signed_error"] = conservative["conservative_bias"]
        summary.update(conservative)
    else:
        for key in (
            "mean_signed_error",
            "overprediction_rate",
            "mean_overprediction",
            "underprediction_rate",
            "mean_underprediction",
            "downside_capture_rate",
            "severe_downside_recall",
            "false_safe_rate",
            "false_safe_negative_rate",
            "false_safe_tail_rate",
            "false_safe_severe_rate",
            "conservative_bias",
            "upside_sacrifice",
        ):
            summary[key] = None

    summary.update(
        _line_segment_metrics(
            prefix="all_horizon",
            start=0,
            end=int(line_pred.shape[1]),
            metadata=metadata,
            line_pred=line_pred,
            line_actual=line_actual,
            raw_actual=raw_actual,
            severe_downside_threshold=severe_downside_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        )
    )
    summary.update(
        _band_segment_metrics(
            prefix="all_horizon_band",
            start=0,
            end=int(line_pred.shape[1]),
            line_pred=line_pred,
            lower=lower,
            upper=upper,
            band_actual=band_actual,
            raw_actual=raw_actual,
            q_low=q_low,
            q_high=q_high,
            lower_penalty_weight=interval_lower_penalty_weight,
            upper_penalty_weight=interval_upper_penalty_weight,
            squeeze_breakout_threshold=squeeze_breakout_threshold,
        )
    )
    for prefix, start, end in (
        ("h1_h5", 0, 5),
        ("h6_h10", 5, 10),
        ("h11_h20", 10, 20),
    ):
        summary.update(
            _line_segment_metrics(
                prefix=prefix,
                start=start,
                end=end,
                metadata=metadata,
                line_pred=line_pred,
                line_actual=line_actual,
                raw_actual=raw_actual,
                severe_downside_threshold=severe_downside_threshold,
                top_k_frac=top_k_frac,
                fee_bps=fee_bps,
            )
        )
        summary.update(
            _band_segment_metrics(
                prefix=f"{prefix}_band",
                start=start,
                end=end,
                line_pred=line_pred,
                lower=lower,
                upper=upper,
                band_actual=band_actual,
                raw_actual=raw_actual,
                q_low=q_low,
                q_high=q_high,
                lower_penalty_weight=interval_lower_penalty_weight,
                upper_penalty_weight=interval_upper_penalty_weight,
                squeeze_breakout_threshold=squeeze_breakout_threshold,
            )
        )

    if metadata is None or metadata.empty:
        summary.update(
            {
                "spearman_ic": None,
                "ic_mean": None,
                "ic_std": None,
                "ic_ir": None,
                "ic_t_stat": None,
                "top_k_long_spread": None,
                "top_k_short_spread": None,
                "long_short_spread": None,
                "spread_mean": None,
                "spread_std": None,
                "spread_ir": None,
                "spread_t_stat": None,
                "fee_adjusted_return": None,
                "fee_adjusted_sharpe": None,
                "fee_adjusted_turnover": None,
            }
        )
        return _attach_metric_layers(
            summary,
            include_legacy_overlay_diagnostics=include_legacy_overlay_diagnostics,
        )

    summary.update(
        _investment_metrics_for_score(
            metadata=metadata,
            score=line_pred[:, -1],
            actual_return=raw_actual[:, -1],
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        )
    )
    return _attach_metric_layers(
        summary,
        include_legacy_overlay_diagnostics=include_legacy_overlay_diagnostics,
    )


def _binary_roc_auc(score: torch.Tensor, target: torch.Tensor) -> float | None:
    finite = torch.isfinite(score) & torch.isfinite(target)
    if not bool(finite.any()):
        return None
    score = score[finite]
    target = (target[finite] > 0.5).to(torch.float32)
    pos_count = int(target.sum().item())
    neg_count = int((target <= 0.5).sum().item())
    if pos_count == 0 or neg_count == 0:
        return None
    order = torch.argsort(score)
    ranks = torch.empty_like(score, dtype=torch.float32)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float32, device=score.device)
    pos_rank_sum = ranks[target > 0.5].sum()
    auc = (pos_rank_sum - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)
    return float(auc.item())


def _binary_average_precision(score: torch.Tensor, target: torch.Tensor) -> float | None:
    finite = torch.isfinite(score) & torch.isfinite(target)
    if not bool(finite.any()):
        return None
    score = score[finite]
    target = (target[finite] > 0.5).to(torch.float32)
    pos_count = float(target.sum().item())
    if pos_count <= 0:
        return None
    order = torch.argsort(score, descending=True)
    sorted_target = target[order]
    cumulative_tp = torch.cumsum(sorted_target, dim=0)
    ranks = torch.arange(1, sorted_target.numel() + 1, dtype=torch.float32, device=score.device)
    precision_at_k = cumulative_tp / ranks
    return float((precision_at_k * sorted_target).sum().item() / pos_count)


def _risk_calibration_error(probability: torch.Tensor, target: torch.Tensor, *, bins: int = 10) -> float | None:
    finite = torch.isfinite(probability) & torch.isfinite(target)
    if not bool(finite.any()):
        return None
    probability = probability[finite].clamp(0.0, 1.0)
    target = (target[finite] > 0.5).to(torch.float32)
    total = float(probability.numel())
    error = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (probability >= low) & (probability <= high)
        else:
            mask = (probability >= low) & (probability < high)
        count = int(mask.sum().item())
        if count == 0:
            continue
        error += (count / total) * abs(float(probability[mask].mean().item()) - float(target[mask].mean().item()))
    return float(error)


def _line_v2_risk_metrics(
    *,
    line_pred: torch.Tensor,
    risk_prob: torch.Tensor,
    raw_actual: torch.Tensor,
    risk_threshold: float,
    risk_decision_threshold: float,
) -> dict[str, float | None]:
    score = line_pred.detach().cpu().to(torch.float32).reshape(-1)
    prob = risk_prob.detach().cpu().to(torch.float32).reshape(-1)
    actual = raw_actual.detach().cpu().to(torch.float32).reshape(-1)
    finite = torch.isfinite(score) & torch.isfinite(prob) & torch.isfinite(actual)
    if not bool(finite.any()):
        return {name: None for name in RISK_METRIC_NAMES}

    score = score[finite]
    prob = prob[finite]
    actual = actual[finite]
    risk_target = (actual <= float(risk_threshold)).to(torch.float32)
    risky_pred = prob >= float(risk_decision_threshold)
    tail = risk_target > 0.5
    non_tail = ~tail
    line_positive = score > 0.0
    risk_low = prob < float(risk_decision_threshold)
    risk_high = prob >= float(risk_decision_threshold)
    separation_gap = None
    if bool(tail.any()) and bool(non_tail.any()):
        separation_gap = float(prob[tail].mean().item() - prob[non_tail].mean().item())

    return {
        "risk_threshold": float(risk_threshold),
        "risk_decision_threshold": float(risk_decision_threshold),
        "risk_prevalence": float(tail.to(torch.float32).mean().item()),
        "risk_recall": _safe_rate(risky_pred, tail),
        "risk_precision": _safe_rate(tail, risky_pred),
        "risk_false_alarm_rate": _safe_rate(risky_pred, non_tail),
        "risk_accuracy": float((risky_pred == tail).to(torch.float32).mean().item()),
        "risk_roc_auc": _binary_roc_auc(prob, risk_target),
        "risk_pr_auc": _binary_average_precision(prob, risk_target),
        "risk_calibration_error": _risk_calibration_error(prob, risk_target),
        "dual_head_false_safe_rate": _safe_rate(line_positive & risk_low, tail),
        "line_positive_risk_low_tail_rate": _safe_rate(line_positive & risk_low, tail),
        "line_positive_risk_high_tail_rate": _safe_rate(line_positive & risk_high, tail),
        "risk_separation_gap": separation_gap,
    }


def summarize_line_v2_metrics(
    *,
    metadata: pd.DataFrame | None,
    line_predictions: torch.Tensor,
    risk_logits: torch.Tensor,
    line_targets: torch.Tensor,
    raw_future_returns: torch.Tensor,
    line_target_type: str = "raw_future_return",
    severe_downside_threshold: float | None = None,
    risk_decision_threshold: float = 0.5,
    top_k_frac: float = 0.1,
    fee_bps: float = 10.0,
) -> dict[str, float | None]:
    line_pred = line_predictions.detach().cpu().to(torch.float32)
    risk_prob = torch.sigmoid(risk_logits.detach().cpu().to(torch.float32))
    line_actual = line_targets.detach().cpu().to(torch.float32)
    raw_actual = raw_future_returns.detach().cpu().to(torch.float32)
    absolute_error = torch.abs(line_pred - line_actual)
    smape = 2.0 * absolute_error / (line_pred.abs() + line_actual.abs() + 1e-6)
    direction_threshold = 0.5 if line_target_type == "direction_label" else 0.0
    risk_threshold = (
        float(severe_downside_threshold)
        if severe_downside_threshold is not None
        else -abs(_severe_downside_threshold(0, int(line_pred.shape[1])))
    )

    summary: dict[str, float | None] = {
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
        "direction_accuracy": float(
            ((line_pred[:, -1] >= direction_threshold) == (line_actual[:, -1] >= direction_threshold)).float().mean().item()
        ),
    }
    if line_target_type == "raw_future_return":
        conservative = _conservative_line_metrics(
            score=line_pred,
            actual=raw_actual,
            severe_threshold=risk_threshold,
            metadata=None,
        )
        summary["mean_signed_error"] = conservative["conservative_bias"]
        summary.update(conservative)

    summary.update(
        _line_segment_metrics(
            prefix="all_horizon",
            start=0,
            end=int(line_pred.shape[1]),
            metadata=metadata,
            line_pred=line_pred,
            line_actual=line_actual,
            raw_actual=raw_actual,
            severe_downside_threshold=risk_threshold,
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        )
    )
    for prefix, start, end in (
        ("h1_h5", 0, 5),
        ("h6_h10", 5, 10),
        ("h11_h20", 10, 20),
    ):
        summary.update(
            _line_segment_metrics(
                prefix=prefix,
                start=start,
                end=end,
                metadata=metadata,
                line_pred=line_pred,
                line_actual=line_actual,
                raw_actual=raw_actual,
                severe_downside_threshold=risk_threshold,
                top_k_frac=top_k_frac,
                fee_bps=fee_bps,
            )
        )

    if metadata is None or metadata.empty:
        summary.update(
            {
                "spearman_ic": None,
                "ic_mean": None,
                "ic_std": None,
                "ic_ir": None,
                "ic_t_stat": None,
                "top_k_long_spread": None,
                "top_k_short_spread": None,
                "long_short_spread": None,
                "spread_mean": None,
                "spread_std": None,
                "spread_ir": None,
                "spread_t_stat": None,
                "fee_adjusted_return": None,
                "fee_adjusted_sharpe": None,
                "fee_adjusted_turnover": None,
            }
        )
    else:
        summary.update(
            _investment_metrics_for_score(
                metadata=metadata,
                score=line_pred[:, -1],
                actual_return=raw_actual[:, -1],
                top_k_frac=top_k_frac,
                fee_bps=fee_bps,
            )
        )

    summary.update(
        _line_v2_risk_metrics(
            line_pred=line_pred,
            risk_prob=risk_prob,
            raw_actual=raw_actual,
            risk_threshold=risk_threshold,
            risk_decision_threshold=risk_decision_threshold,
        )
    )
    return _attach_metric_layers(summary, include_legacy_overlay_diagnostics=False)


def _regime_targets_from_thresholds(raw_actual: torch.Tensor, thresholds: tuple[float, float, float, float] | list[float]) -> torch.Tensor:
    h5_actual = raw_actual.detach().cpu().to(torch.float32)[:, -1]
    q = h5_actual.new_tensor([float(value) for value in thresholds])
    target = torch.zeros_like(h5_actual, dtype=torch.long)
    target = target + (h5_actual > q[0]).to(torch.long)
    target = target + (h5_actual > q[1]).to(torch.long)
    target = target + (h5_actual > q[2]).to(torch.long)
    target = target + (h5_actual > q[3]).to(torch.long)
    return target


def _macro_f1(predicted: torch.Tensor, target: torch.Tensor, class_count: int = 5) -> tuple[float | None, dict[str, float | None], dict[str, float | None]]:
    f1_values: list[float] = []
    precision_by_class: dict[str, float | None] = {}
    recall_by_class: dict[str, float | None] = {}
    for class_index in range(class_count):
        pred_mask = predicted == class_index
        target_mask = target == class_index
        tp = float((pred_mask & target_mask).sum().item())
        fp = float((pred_mask & ~target_mask).sum().item())
        fn = float((~pred_mask & target_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tp / (tp + fn) if tp + fn > 0 else None
        precision_by_class[f"class_{class_index}_precision"] = precision
        recall_by_class[f"class_{class_index}_recall"] = recall
        if precision is not None and recall is not None and precision + recall > 0:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return (_safe_mean(f1_values), precision_by_class, recall_by_class)


def _line_regime_filter_metrics(
    *,
    metadata: pd.DataFrame | None,
    line_score: torch.Tensor,
    actual: torch.Tensor,
    predicted_class: torch.Tensor,
    target_class: torch.Tensor,
    top_k_frac: float,
) -> dict[str, float | None]:
    line_np = line_score.detach().cpu().to(torch.float32).numpy()
    actual_np = actual.detach().cpu().to(torch.float32).numpy()
    pred_np = predicted_class.detach().cpu().numpy()
    target_np = target_class.detach().cpu().numpy()
    strong_down = target_np == 0
    line_positive = line_np > 0.0
    risky_regime = pred_np <= 1
    safe_regime = pred_np >= 2
    positive_strong_down = line_positive & strong_down
    positive_count = int(line_positive.sum())
    line_positive_false_safe_rate = (
        float(positive_strong_down.sum() / positive_count) if positive_count else None
    )
    safe_positive = line_positive & safe_regime
    risky_positive = line_positive & risky_regime
    safe_down_rate = float((actual_np[safe_positive] < 0.0).mean()) if safe_positive.any() else None
    risky_down_rate = float((actual_np[risky_positive] < 0.0).mean()) if risky_positive.any() else None

    if metadata is None or metadata.empty:
        return {
            "line_positive_false_safe_rate": line_positive_false_safe_rate,
            "line_top_decile_false_safe_rate": None,
            "regime_filtered_false_safe_reduction": None,
            "spread_retention": None,
            "line_positive_regime_safe_actual_down_rate": safe_down_rate,
            "line_positive_regime_risky_actual_down_rate": risky_down_rate,
        }

    frame = metadata.reset_index(drop=True).copy()
    frame["line_score"] = line_np
    frame["actual"] = actual_np
    frame["predicted_class"] = pred_np
    frame["target_class"] = target_np
    base_tail_rates: list[float] = []
    filtered_tail_rates: list[float] = []
    base_spreads: list[float] = []
    filtered_spreads: list[float] = []
    for _, group in frame.groupby("asof_date", sort=True):
        if len(group) < 4:
            continue
        top_k = max(int(math.ceil(len(group) * top_k_frac)), 1)
        ranked = group.sort_values("line_score", ascending=False).reset_index(drop=True)
        top = ranked.head(top_k)
        bottom = ranked.tail(top_k)
        base_tail_rates.append(float((top["target_class"] == 0).mean()))
        base_spread = float(top["actual"].mean() - bottom["actual"].mean())
        base_spreads.append(base_spread)
        filtered_top = top[top["predicted_class"] >= 2]
        if not filtered_top.empty:
            filtered_tail_rates.append(float((filtered_top["target_class"] == 0).mean()))
            filtered_spreads.append(float(filtered_top["actual"].mean() - bottom["actual"].mean()))
    base_tail = _safe_mean(base_tail_rates)
    filtered_tail = _safe_mean(filtered_tail_rates)
    base_spread_mean = _safe_mean(base_spreads)
    filtered_spread_mean = _safe_mean(filtered_spreads)
    return {
        "line_positive_false_safe_rate": line_positive_false_safe_rate,
        "line_top_decile_false_safe_rate": base_tail,
        "regime_filtered_false_safe_reduction": (
            (base_tail - filtered_tail) / base_tail
            if base_tail is not None and filtered_tail is not None and base_tail > 0
            else None
        ),
        "spread_retention": (
            filtered_spread_mean / base_spread_mean
            if base_spread_mean is not None and filtered_spread_mean is not None and abs(base_spread_mean) > 1e-12
            else None
        ),
        "line_positive_regime_safe_actual_down_rate": safe_down_rate,
        "line_positive_regime_risky_actual_down_rate": risky_down_rate,
    }


def summarize_line_regime_metrics(
    *,
    metadata: pd.DataFrame | None,
    line_predictions: torch.Tensor,
    regime_logits: torch.Tensor,
    line_targets: torch.Tensor,
    raw_future_returns: torch.Tensor,
    line_target_type: str = "raw_future_return",
    regime_thresholds: tuple[float, float, float, float] | list[float] | None = None,
    top_k_frac: float = 0.1,
    fee_bps: float = 10.0,
) -> dict[str, float | None]:
    line_pred = line_predictions.detach().cpu().to(torch.float32)
    logits = regime_logits.detach().cpu().to(torch.float32)
    line_actual = line_targets.detach().cpu().to(torch.float32)
    raw_actual = raw_future_returns.detach().cpu().to(torch.float32)
    if regime_thresholds is None:
        h5 = raw_actual[:, -1]
        regime_thresholds = tuple(float(torch.quantile(h5, q).item()) for q in (0.10, 0.35, 0.65, 0.90))
    regime_thresholds = tuple(float(value) for value in regime_thresholds)
    absolute_error = torch.abs(line_pred - line_actual)
    smape = 2.0 * absolute_error / (line_pred.abs() + line_actual.abs() + 1e-6)
    direction_threshold = 0.5 if line_target_type == "direction_label" else 0.0
    h5_line = line_pred[:, -1]
    h5_actual = raw_actual[:, -1]
    target_class = _regime_targets_from_thresholds(raw_actual, regime_thresholds)
    predicted_class = torch.argmax(logits, dim=-1).to(torch.long)
    probability = torch.softmax(logits, dim=-1)
    class_axis = torch.arange(5, dtype=probability.dtype, device=probability.device).view(1, 5)
    regime_risk_score = probability[:, 0] + probability[:, 1]
    regime_expected_class = (probability * class_axis).sum(dim=-1)
    severe_target = (target_class == 0).to(torch.float32)
    class_error = (predicted_class - target_class).abs().to(torch.float32)
    macro_f1, precision_by_class, recall_by_class = _macro_f1(predicted_class, target_class)
    mean_return_by_predicted_class: dict[str, float | None] = {}
    class_indices: list[float] = []
    class_returns: list[float] = []
    for class_index in range(5):
        mask = predicted_class == class_index
        value = float(h5_actual[mask].mean().item()) if bool(mask.any()) else None
        mean_return_by_predicted_class[f"predicted_class_{class_index}_mean_return"] = value
        if value is not None:
            class_indices.append(float(class_index))
            class_returns.append(value)
    monotonicity = None
    if len(class_indices) >= 2:
        monotonicity = float(pd.Series(class_indices).corr(pd.Series(class_returns), method="spearman"))
    non_decreasing = None
    ordered_values = [mean_return_by_predicted_class[f"predicted_class_{idx}_mean_return"] for idx in range(5)]
    present_values = [value for value in ordered_values if value is not None]
    if len(present_values) >= 2:
        non_decreasing = float(all(left <= right for left, right in zip(present_values, present_values[1:])))

    conservative = _conservative_line_metrics(
        score=line_pred,
        actual=raw_actual,
        severe_threshold=regime_thresholds[0],
        metadata=None,
    )
    summary: dict[str, float | None] = {
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
        "direction_accuracy": float(
            ((line_pred[:, -1] >= direction_threshold) == (line_actual[:, -1] >= direction_threshold)).float().mean().item()
        ),
        "mean_signed_error": conservative["conservative_bias"],
        **conservative,
    }
    summary.update(
        _line_segment_metrics(
            prefix="all_horizon",
            start=0,
            end=int(line_pred.shape[1]),
            metadata=metadata,
            line_pred=line_pred,
            line_actual=line_actual,
            raw_actual=raw_actual,
            severe_downside_threshold=regime_thresholds[0],
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        )
    )
    summary.update(
        _line_segment_metrics(
            prefix="h1_h5",
            start=0,
            end=min(5, int(line_pred.shape[1])),
            metadata=metadata,
            line_pred=line_pred,
            line_actual=line_actual,
            raw_actual=raw_actual,
            severe_downside_threshold=regime_thresholds[0],
            top_k_frac=top_k_frac,
            fee_bps=fee_bps,
        )
    )
    if metadata is None or metadata.empty:
        summary.update(
            {
                "spearman_ic": None,
                "ic_mean": None,
                "ic_std": None,
                "ic_ir": None,
                "ic_t_stat": None,
                "top_k_long_spread": None,
                "top_k_short_spread": None,
                "long_short_spread": None,
                "spread_mean": None,
                "spread_std": None,
                "spread_ir": None,
                "spread_t_stat": None,
                "fee_adjusted_return": None,
                "fee_adjusted_sharpe": None,
                "fee_adjusted_turnover": None,
            }
        )
    else:
        summary.update(
            _investment_metrics_for_score(
                metadata=metadata,
                score=h5_line,
                actual_return=h5_actual,
                top_k_frac=top_k_frac,
                fee_bps=fee_bps,
            )
        )

    summary.update(
        {
            "regime_threshold_q10": regime_thresholds[0],
            "regime_threshold_q35": regime_thresholds[1],
            "regime_threshold_q65": regime_thresholds[2],
            "regime_threshold_q90": regime_thresholds[3],
            "regime_accuracy": float((predicted_class == target_class).to(torch.float32).mean().item()),
            "regime_adjacent_accuracy": float((class_error <= 1.0).to(torch.float32).mean().item()),
            "regime_ordinal_mae": float(class_error.mean().item()),
            "regime_ordinal_mse": float(class_error.pow(2).mean().item()),
            "regime_macro_f1": macro_f1,
            "strong_down_recall": recall_by_class.get("class_0_recall"),
            "strong_down_precision": precision_by_class.get("class_0_precision"),
            "strong_up_precision": precision_by_class.get("class_4_precision"),
            "regime_return_monotonicity": monotonicity,
            "regime_return_monotonic_non_decreasing": non_decreasing,
            **precision_by_class,
            **recall_by_class,
            **mean_return_by_predicted_class,
        }
    )
    summary.update(
        _line_regime_filter_metrics(
            metadata=metadata,
            line_score=h5_line,
            actual=h5_actual,
            predicted_class=predicted_class,
            target_class=target_class,
            top_k_frac=top_k_frac,
        )
    )
    false_safe_mask = (h5_line >= 0) & (target_class == 0)
    top_date_share = None
    top_ticker_share = None
    if metadata is not None and not metadata.empty and bool(false_safe_mask.any()):
        false_safe_meta = metadata.reset_index(drop=True).loc[false_safe_mask.detach().cpu().numpy()].copy()
        if "asof_date" in false_safe_meta.columns and len(false_safe_meta) > 0:
            date_counts = false_safe_meta["asof_date"].astype(str).value_counts()
            top_date_share = float(date_counts.iloc[0] / len(false_safe_meta)) if len(date_counts) else None
        if "ticker" in false_safe_meta.columns and len(false_safe_meta) > 0:
            ticker_counts = false_safe_meta["ticker"].astype(str).value_counts()
            top_ticker_share = float(ticker_counts.iloc[0] / len(false_safe_meta)) if len(ticker_counts) else None
    summary.update(
        {
            "regime_risk_line_corr_pearson": _pearson_corr(regime_risk_score, h5_line),
            "regime_risk_line_corr_spearman": _spearman_corr(regime_risk_score, h5_line),
            "regime_expected_class_line_corr_pearson": _pearson_corr(regime_expected_class, h5_line),
            "regime_expected_class_line_corr_spearman": _spearman_corr(regime_expected_class, h5_line),
            "regime_risk_residual_severe_auc": _linear_residual_auc(regime_risk_score, h5_line, severe_target),
            "top_false_safe_date_max_share": top_date_share,
            "top_false_safe_ticker_max_share": top_ticker_share,
        }
    )
    return _attach_metric_layers(summary, include_legacy_overlay_diagnostics=False)


def summarize_band_metrics(
    *,
    lower_predictions: torch.Tensor,
    upper_predictions: torch.Tensor,
    band_targets: torch.Tensor,
    raw_future_returns: torch.Tensor,
    q_low: float = 0.1,
    q_high: float = 0.9,
    interval_lower_penalty_weight: float = 2.0,
    interval_upper_penalty_weight: float = 1.0,
    squeeze_breakout_threshold: float | None = None,
) -> dict[str, float | None]:
    lower = lower_predictions.detach().cpu().to(torch.float32)
    upper = upper_predictions.detach().cpu().to(torch.float32)
    band_actual = band_targets.detach().cpu().to(torch.float32)
    raw_actual = raw_future_returns.detach().cpu().to(torch.float32)
    midpoint = (lower + upper) / 2.0
    summary: dict[str, float | None] = {
        "coverage": float(((band_actual >= lower) & (band_actual <= upper)).float().mean().item()),
        "lower_breach_rate": float((band_actual < lower).float().mean().item()),
        "upper_breach_rate": float((band_actual > upper).float().mean().item()),
        "avg_band_width": float((upper - lower).mean().item()),
    }
    summary.update(
        _band_interval_metrics(
            line_pred=midpoint,
            lower=lower,
            upper=upper,
            band_actual=band_actual,
            raw_actual=raw_actual,
            q_low=q_low,
            q_high=q_high,
            lower_penalty_weight=interval_lower_penalty_weight,
            upper_penalty_weight=interval_upper_penalty_weight,
            squeeze_breakout_threshold=squeeze_breakout_threshold,
        )
    )
    summary.update(
        _band_segment_metrics(
            prefix="all_horizon_band",
            start=0,
            end=int(lower.shape[1]),
            line_pred=midpoint,
            lower=lower,
            upper=upper,
            band_actual=band_actual,
            raw_actual=raw_actual,
            q_low=q_low,
            q_high=q_high,
            lower_penalty_weight=interval_lower_penalty_weight,
            upper_penalty_weight=interval_upper_penalty_weight,
            squeeze_breakout_threshold=squeeze_breakout_threshold,
        )
    )
    for prefix, start, end in (
        ("h1_h5_band", 0, 5),
        ("h6_h10_band", 5, 10),
        ("h11_h20_band", 10, 20),
    ):
        summary.update(
            _band_segment_metrics(
                prefix=prefix,
                start=start,
                end=end,
                line_pred=midpoint,
                lower=lower,
                upper=upper,
                band_actual=band_actual,
                raw_actual=raw_actual,
                q_low=q_low,
                q_high=q_high,
                lower_penalty_weight=interval_lower_penalty_weight,
                upper_penalty_weight=interval_upper_penalty_weight,
                squeeze_breakout_threshold=squeeze_breakout_threshold,
            )
        )
    return _attach_metric_layers(summary, include_legacy_overlay_diagnostics=False)


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
        "lower_breach_rate": float((actual_tensor < lower_tensor).float().mean().item()),
        "upper_breach_rate": float((actual_tensor > upper_tensor).float().mean().item()),
        "avg_band_width": float((upper_tensor - lower_tensor).mean().item()),
        "direction_accuracy": float(((line_tensor >= direction_threshold) == (actual_tensor >= direction_threshold)).float().mean().item()),
        "mae": float(absolute_error.mean().item()),
        "smape": float(smape.mean().item()),
    }
