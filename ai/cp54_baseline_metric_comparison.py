from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.preprocessing import SequenceDataset, SequenceDatasetBundle, prepare_dataset_splits  # noqa: E402
from ai.train import estimate_train_risk_thresholds  # noqa: E402


LINE_METRIC_KEYS = (
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_t_stat",
    "spearman_ic",
    "long_short_spread",
    "spread_mean",
    "spread_ir",
    "spread_t_stat",
    "fee_adjusted_sharpe",
    "direction_accuracy",
    "false_safe_negative_rate",
    "false_safe_tail_rate",
    "false_safe_severe_rate",
    "severe_downside_recall",
    "downside_capture_rate",
    "conservative_bias",
    "upside_sacrifice",
    "mae",
    "smape",
)

BAND_METRIC_KEYS = (
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "empirical_q_low",
    "empirical_q_high",
    "lower_breach_rate",
    "upper_breach_rate",
    "avg_band_width",
    "median_band_width",
    "p90_band_width",
    "interval_score",
    "interval_lower_penalty",
    "interval_upper_penalty",
    "band_width_ic",
    "downside_width_ic",
    "width_bucket_realized_vol_ratio",
    "width_bucket_downside_rate_ratio",
    "squeeze_breakout_rate",
)

COMPOSITE_METRIC_KEYS = (
    "line_inside_band_ratio",
    "line_inside_band_point_ratio",
    "product_display_warning_rate",
    "conservative_series_false_safe_rate",
    "composite_width_increase_ratio",
    "coverage",
    "lower_breach_rate",
    "upper_breach_rate",
    "avg_band_width",
    "interval_score",
    "band_width_ic",
    "downside_width_ic",
)

AI_LINE_NAMES = {
    "h5_longer_context_seq252_p32_s16",
    "h5_baseline_seq252_p16_s8",
    "h5_dense_overlap_seq252_p16_s4",
}
AI_BAND_NAMES = {
    "s60_q15_b2_direct_188",
    "s60_q15_b2_direct",
    "tide_param_scalar_width",
    "tide_direct_original",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _repo_path(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _metric_subset(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in keys}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


class SplitData:
    def __init__(self, *, bundle: SequenceDataset | SequenceDatasetBundle) -> None:
        self.bundle = bundle
        self.metadata = bundle.metadata.reset_index(drop=True).copy()
        self.raw_future_returns = _extract_raw_returns(bundle)
        self.horizon = int(self.raw_future_returns.shape[1])


def _extract_raw_returns(bundle: SequenceDataset | SequenceDatasetBundle) -> np.ndarray:
    if isinstance(bundle, SequenceDatasetBundle):
        return bundle.raw_future_returns.detach().cpu().numpy().astype("float32", copy=True)

    values = np.empty((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        anchor_close = float(closes[end_idx])
        future_start = end_idx + 1
        future_end = future_start + bundle.horizon
        values[row_idx] = (closes[future_start:future_end] / max(anchor_close, 1e-6)) - 1.0
    return values


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype="float32"))


def _evaluate_predictions(
    *,
    data: SplitData,
    line: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_low: float,
    q_high: float,
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    raw = data.raw_future_returns.astype("float32", copy=False)
    return summarize_forecast_metrics(
        metadata=data.metadata,
        line_predictions=_as_tensor(line),
        lower_predictions=_as_tensor(lower),
        upper_predictions=_as_tensor(upper),
        line_targets=_as_tensor(raw),
        band_targets=_as_tensor(raw),
        raw_future_returns=_as_tensor(raw),
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        q_low=q_low,
        q_high=q_high,
        severe_downside_threshold=thresholds.get("severe_downside_threshold"),
        squeeze_breakout_threshold=thresholds.get("squeeze_breakout_threshold"),
    )


def _wide_band(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(shape, -1_000_000.0, dtype="float32")
    upper = np.full(shape, 1_000_000.0, dtype="float32")
    return lower, upper


def _past_horizon_returns(closes: np.ndarray, *, end_idx: int, horizon_step: int, window: int) -> np.ndarray:
    latest_anchor = end_idx - horizon_step
    if latest_anchor < 0:
        return np.empty(0, dtype="float32")
    first_anchor = max(0, latest_anchor - window + 1)
    anchors = np.arange(first_anchor, latest_anchor + 1, dtype=np.int64)
    if anchors.size == 0:
        return np.empty(0, dtype="float32")
    values = (closes[anchors + horizon_step] / np.clip(closes[anchors], 1e-6, None)) - 1.0
    values = np.asarray(values, dtype="float32")
    return values[np.isfinite(values)]


def _trailing_momentum(closes: np.ndarray, *, end_idx: int, horizon_step: int) -> float:
    start_idx = end_idx - horizon_step
    if start_idx < 0:
        return 0.0
    value = (float(closes[end_idx]) / max(float(closes[start_idx]), 1e-6)) - 1.0
    return float(value) if np.isfinite(value) else 0.0


def _build_historical_mean_line(bundle: SequenceDataset, *, window: int) -> np.ndarray:
    output = np.zeros((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        for horizon_idx in range(bundle.horizon):
            values = _past_horizon_returns(closes, end_idx=end_idx, horizon_step=horizon_idx + 1, window=window)
            output[row_idx, horizon_idx] = float(values.mean()) if values.size else 0.0
    return output


def _build_momentum_line(bundle: SequenceDataset, *, sign: float = 1.0) -> np.ndarray:
    output = np.zeros((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        for horizon_idx in range(bundle.horizon):
            output[row_idx, horizon_idx] = sign * _trailing_momentum(
                closes,
                end_idx=end_idx,
                horizon_step=horizon_idx + 1,
            )
    return output


def _shuffle_by_date(scores: np.ndarray, metadata: pd.DataFrame, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = scores.copy()
    if "asof_date" not in metadata.columns:
        order = np.arange(scores.shape[0])
        rng.shuffle(order)
        return scores[order]
    for _, index in metadata.groupby("asof_date", sort=True).groups.items():
        indices = np.asarray(list(index), dtype=np.int64)
        if indices.size <= 1:
            continue
        source = indices.copy()
        rng.shuffle(source)
        shuffled[indices] = scores[source]
    return shuffled


def _build_constant_train_quantiles(train_raw: np.ndarray, shape: tuple[int, int], *, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
    lower_by_h = np.quantile(train_raw, q_low, axis=0).astype("float32")
    upper_by_h = np.quantile(train_raw, q_high, axis=0).astype("float32")
    lower = np.tile(lower_by_h.reshape(1, -1), (shape[0], 1)).astype("float32")
    upper = np.tile(upper_by_h.reshape(1, -1), (shape[0], 1)).astype("float32")
    return lower, upper


def _build_rolling_quantile_band(
    bundle: SequenceDataset,
    *,
    window: int,
    q_low: float,
    q_high: float,
    fallback_lower: np.ndarray,
    fallback_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(bundle), bundle.horizon), dtype="float32")
    upper = np.empty((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        for horizon_idx in range(bundle.horizon):
            values = _past_horizon_returns(closes, end_idx=end_idx, horizon_step=horizon_idx + 1, window=window)
            if values.size:
                lower[row_idx, horizon_idx] = float(np.quantile(values, q_low))
                upper[row_idx, horizon_idx] = float(np.quantile(values, q_high))
            else:
                lower[row_idx, horizon_idx] = float(fallback_lower[horizon_idx])
                upper[row_idx, horizon_idx] = float(fallback_upper[horizon_idx])
    return lower, upper


def _build_bollinger_return_band(
    bundle: SequenceDataset,
    *,
    window: int,
    k_std: float,
    fallback_mean: np.ndarray,
    fallback_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(bundle), bundle.horizon), dtype="float32")
    upper = np.empty((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        for horizon_idx in range(bundle.horizon):
            values = _past_horizon_returns(closes, end_idx=end_idx, horizon_step=horizon_idx + 1, window=window)
            if values.size >= 2:
                center = float(values.mean())
                spread = float(values.std(ddof=0))
            else:
                center = float(fallback_mean[horizon_idx])
                spread = float(fallback_std[horizon_idx])
            lower[row_idx, horizon_idx] = center - (k_std * spread)
            upper[row_idx, horizon_idx] = center + (k_std * spread)
    return lower, upper


def _recent_daily_vol(closes: np.ndarray, *, end_idx: int, window: int) -> float:
    first = max(0, end_idx - window)
    if end_idx <= first:
        return 0.0
    history = closes[first : end_idx + 1]
    daily = np.diff(history) / np.clip(history[:-1], 1e-6, None)
    daily = daily[np.isfinite(daily)]
    return float(daily.std(ddof=0)) if daily.size else 0.0


def _build_volatility_scaled_constant_band(
    bundle: SequenceDataset,
    *,
    window: int,
    train_raw: np.ndarray,
    q_low: float,
    q_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_lower = np.quantile(train_raw, q_low, axis=0).astype("float32")
    train_upper = np.quantile(train_raw, q_high, axis=0).astype("float32")
    center = ((train_lower + train_upper) / 2.0).astype("float32")
    half_width = np.maximum((train_upper - train_lower) / 2.0, 1e-6).astype("float32")
    train_step_std = np.maximum(train_raw.std(axis=0), 1e-6).astype("float32")

    lower = np.empty((len(bundle), bundle.horizon), dtype="float32")
    upper = np.empty((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        daily_vol = _recent_daily_vol(closes, end_idx=end_idx, window=window)
        for horizon_idx in range(bundle.horizon):
            horizon_step = horizon_idx + 1
            realized_vol_proxy = daily_vol * float(np.sqrt(horizon_step))
            scale = np.clip(realized_vol_proxy / float(train_step_std[horizon_idx]), 0.25, 4.0)
            lower[row_idx, horizon_idx] = float(center[horizon_idx] - half_width[horizon_idx] * scale)
            upper[row_idx, horizon_idx] = float(center[horizon_idx] + half_width[horizon_idx] * scale)
    return lower, upper


def _require_lazy(bundle: SequenceDataset | SequenceDatasetBundle, *, label: str) -> SequenceDataset:
    if not isinstance(bundle, SequenceDataset):
        raise TypeError(f"{label} split이 lazy SequenceDataset이 아닙니다. rolling baseline 계산을 중단합니다.")
    return bundle


def _evaluate_line_baselines(
    *,
    train_bundle: SequenceDataset,
    val_data: SplitData,
    test_data: SplitData,
    q_low: float,
    q_high: float,
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    shape_val = val_data.raw_future_returns.shape
    shape_test = test_data.raw_future_returns.shape
    wide_val = _wide_band(shape_val)
    wide_test = _wide_band(shape_test)

    val_bundle = _require_lazy(val_data.bundle, label="validation")
    test_bundle = _require_lazy(test_data.bundle, label="test")
    val_momentum = _build_momentum_line(val_bundle)
    test_momentum = _build_momentum_line(test_bundle)
    builders: list[tuple[str, str, np.ndarray, np.ndarray, dict[str, Any]]] = [
        ("zero_line", "line", np.zeros(shape_val, dtype="float32"), np.zeros(shape_test, dtype="float32"), {}),
        (
            "historical_mean_line_w20",
            "line",
            _build_historical_mean_line(val_bundle, window=20),
            _build_historical_mean_line(test_bundle, window=20),
            {"window": 20},
        ),
        (
            "historical_mean_line_w60",
            "line",
            _build_historical_mean_line(val_bundle, window=60),
            _build_historical_mean_line(test_bundle, window=60),
            {"window": 60},
        ),
        ("momentum_line_horizon", "line", val_momentum, test_momentum, {"lookback": "horizon_step"}),
        ("reversal_line_horizon", "line", -val_momentum, -test_momentum, {"lookback": "horizon_step", "sign": -1}),
        (
            "random_or_shuffled_score",
            "diagnostic_line",
            _shuffle_by_date(val_momentum, val_data.metadata),
            _shuffle_by_date(test_momentum, test_data.metadata),
            {"source": "datewise_shuffled_momentum", "seed": 42},
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, category, val_line, test_line, config in builders:
        val_metrics = _evaluate_predictions(
            data=val_data,
            line=val_line,
            lower=wide_val[0],
            upper=wide_val[1],
            q_low=q_low,
            q_high=q_high,
            thresholds=thresholds,
        )
        test_metrics = _evaluate_predictions(
            data=test_data,
            line=test_line,
            lower=wide_test[0],
            upper=wide_test[1],
            q_low=q_low,
            q_high=q_high,
            thresholds=thresholds,
        )
        rows.append(
            {
                "name": name,
                "category": category,
                "config": config,
                "validation": _metric_subset(val_metrics, LINE_METRIC_KEYS),
                "test": _metric_subset(test_metrics, LINE_METRIC_KEYS),
                "notes": _line_baseline_notes(name, test_metrics),
            }
        )
    return rows


def _evaluate_band_baselines(
    *,
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    q_low: float,
    q_high: float,
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    train_raw = train_data.raw_future_returns
    fallback_lower = np.quantile(train_raw, q_low, axis=0).astype("float32")
    fallback_upper = np.quantile(train_raw, q_high, axis=0).astype("float32")
    fallback_mean = train_raw.mean(axis=0).astype("float32")
    fallback_std = np.maximum(train_raw.std(axis=0), 1e-6).astype("float32")
    val_bundle = _require_lazy(val_data.bundle, label="validation")
    test_bundle = _require_lazy(test_data.bundle, label="test")

    band_specs: list[tuple[str, str, dict[str, Any], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = []
    val_constant = _build_constant_train_quantiles(train_raw, val_data.raw_future_returns.shape, q_low=q_low, q_high=q_high)
    test_constant = _build_constant_train_quantiles(train_raw, test_data.raw_future_returns.shape, q_low=q_low, q_high=q_high)
    band_specs.append(("constant_width_train_quantile", "band", {"q_low": q_low, "q_high": q_high}, val_constant, test_constant))

    for window in (60, 120, 252):
        band_specs.append(
            (
                f"rolling_historical_quantile_band_w{window}",
                "band",
                {"window": window, "q_low": q_low, "q_high": q_high},
                _build_rolling_quantile_band(
                    val_bundle,
                    window=window,
                    q_low=q_low,
                    q_high=q_high,
                    fallback_lower=fallback_lower,
                    fallback_upper=fallback_upper,
                ),
                _build_rolling_quantile_band(
                    test_bundle,
                    window=window,
                    q_low=q_low,
                    q_high=q_high,
                    fallback_lower=fallback_lower,
                    fallback_upper=fallback_upper,
                ),
            )
        )

    for window in (20, 60):
        for k_std in (1.0, 1.5, 2.0):
            band_specs.append(
                (
                    f"rolling_bollinger_return_band_w{window}_k{k_std:g}",
                    "band",
                    {"window": window, "k_std": k_std},
                    _build_bollinger_return_band(
                        val_bundle,
                        window=window,
                        k_std=k_std,
                        fallback_mean=fallback_mean,
                        fallback_std=fallback_std,
                    ),
                    _build_bollinger_return_band(
                        test_bundle,
                        window=window,
                        k_std=k_std,
                        fallback_mean=fallback_mean,
                        fallback_std=fallback_std,
                    ),
                )
            )

    for window in (20, 60):
        band_specs.append(
            (
                f"volatility_scaled_constant_band_w{window}",
                "band",
                {"window": window, "base": "train_quantile_width", "scale_clip": [0.25, 4.0]},
                _build_volatility_scaled_constant_band(val_bundle, window=window, train_raw=train_raw, q_low=q_low, q_high=q_high),
                _build_volatility_scaled_constant_band(test_bundle, window=window, train_raw=train_raw, q_low=q_low, q_high=q_high),
            )
        )

    rows: list[dict[str, Any]] = []
    for name, category, config, val_band, test_band in band_specs:
        val_lower, val_upper = val_band
        test_lower, test_upper = test_band
        val_line = ((val_lower + val_upper) / 2.0).astype("float32")
        test_line = ((test_lower + test_upper) / 2.0).astype("float32")
        val_metrics = _evaluate_predictions(
            data=val_data,
            line=val_line,
            lower=val_lower,
            upper=val_upper,
            q_low=q_low,
            q_high=q_high,
            thresholds=thresholds,
        )
        test_metrics = _evaluate_predictions(
            data=test_data,
            line=test_line,
            lower=test_lower,
            upper=test_upper,
            q_low=q_low,
            q_high=q_high,
            thresholds=thresholds,
        )
        rows.append(
            {
                "name": name,
                "category": category,
                "config": config,
                "validation": _metric_subset(val_metrics, BAND_METRIC_KEYS),
                "test": _metric_subset(test_metrics, BAND_METRIC_KEYS),
                "notes": _band_baseline_notes(name, test_metrics),
            }
        )
    return rows


def _line_baseline_notes(name: str, metrics: dict[str, Any]) -> str:
    ic = _safe_float(metrics.get("ic_mean"))
    spread = _safe_float(metrics.get("long_short_spread"))
    false_safe_tail = _safe_float(metrics.get("false_safe_tail_rate"))
    severe_recall = _safe_float(metrics.get("severe_downside_recall"))
    if name == "zero_line":
        return "상수 0 기준선입니다. IC는 의미가 약하고 false-safe 해석 기준으로 사용합니다."
    if ic is not None and spread is not None and ic > 0 and spread > 0:
        if false_safe_tail is not None and false_safe_tail < 0.15 and severe_recall is not None and severe_recall >= 0.80:
            return "라인 기준선 중 강한 후보권입니다."
        return "IC/spread는 양수지만 보수 line 기준은 별도 확인이 필요합니다."
    return "AI 후보가 반드시 이겨야 하는 약한 기준선입니다."


def _band_baseline_notes(name: str, metrics: dict[str, Any]) -> str:
    coverage_error = _safe_float(metrics.get("coverage_abs_error"))
    interval_score = _safe_float(metrics.get("interval_score"))
    width_ic = _safe_float(metrics.get("band_width_ic"))
    downside_ic = _safe_float(metrics.get("downside_width_ic"))
    if coverage_error is not None and coverage_error <= 0.08:
        calibration = "calibration 후보권"
    elif coverage_error is not None and coverage_error <= 0.15:
        calibration = "calibration 생존권"
    else:
        calibration = "calibration 약함"
    dynamic = "동적 폭 양호" if (width_ic or 0) > 0 and (downside_ic or 0) > 0 else "동적 폭 근거 약함"
    return f"{calibration}, {dynamic}, interval_score={interval_score}"


def _select_ai_candidates(cp53: dict[str, Any]) -> dict[str, Any]:
    line = [candidate for candidate in cp53.get("line_candidates", []) if candidate.get("name") in AI_LINE_NAMES]
    band = [candidate for candidate in cp53.get("band_candidates", []) if candidate.get("name") in AI_BAND_NAMES]
    composite_source = (cp53.get("composite_policies") or {}).get("policies") or {}
    composite = {
        name: {
            "test": _metric_subset((block.get("test") or {}), COMPOSITE_METRIC_KEYS),
            "validation": _metric_subset((block.get("validation") or {}), COMPOSITE_METRIC_KEYS),
            "status": block.get("status"),
        }
        for name, block in composite_source.items()
        if name in {"risk_first_upper_buffer_1.10", "risk_first_lower_preserve", "raw_composite"}
    }
    return {
        "line_candidates": line,
        "band_candidates": band,
        "composite_policies": composite,
    }


def _best_by(rows: list[dict[str, Any]], metric: str, *, lower_is_better: bool = False) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_value: float | None = None
    for row in rows:
        value = _safe_float((row.get("test") or {}).get(metric))
        if value is None:
            continue
        if best is None:
            best = row
            best_value = value
            continue
        assert best_value is not None
        if (lower_is_better and value < best_value) or (not lower_is_better and value > best_value):
            best = row
            best_value = value
    return best


def _answer_questions(payload: dict[str, Any]) -> dict[str, Any]:
    ai_line = {
        candidate["name"]: candidate
        for candidate in payload["ai_reference"]["line_candidates"]
    }
    ai_band = {
        candidate["name"]: candidate
        for candidate in payload["ai_reference"]["band_candidates"]
    }
    line_baselines = payload["line_baselines"]
    band_baselines = payload["band_baselines"]

    patch = ai_line.get("h5_longer_context_seq252_p32_s16")
    best_line_ic = _best_by(line_baselines, "ic_mean")
    best_line_false_safe = _best_by(line_baselines, "false_safe_tail_rate", lower_is_better=True)
    cnn = ai_band.get("s60_q15_b2_direct_188") or ai_band.get("s60_q15_b2_direct")
    best_band_interval = _best_by(band_baselines, "interval_score", lower_is_better=True)
    best_band_calibration = _best_by(band_baselines, "coverage_abs_error", lower_is_better=True)
    best_dynamic_width = _best_by(band_baselines, "band_width_ic")

    patch_test = (patch or {}).get("test") or {}
    cnn_test = (cnn or {}).get("test") or {}
    answers = {
        "patchtst_line_vs_simple_baselines": _compare_patchtst_line(
            patch_test=patch_test,
            best_line_ic=best_line_ic,
            best_line_false_safe=best_line_false_safe,
        ),
        "cnn_lstm_band_vs_baselines": _compare_cnn_band(
            cnn_test=cnn_test,
            best_band_interval=best_band_interval,
            best_band_calibration=best_band_calibration,
        ),
        "dynamic_width_vs_volatility_baseline": _compare_dynamic_width(
            cnn_test=cnn_test,
            best_dynamic_width=best_dynamic_width,
        ),
        "wide_band_or_dynamic_risk": _wide_band_answer(cnn_test=cnn_test),
        "h5_h20_branch_policy": "h5와 h20은 계속 분리합니다. CP53 기준 h20 후보는 별도 branch 성격이며, CP54 baseline도 h5 raw-return 비교판으로 제한했습니다.",
    }
    return answers


def _compare_patchtst_line(
    *,
    patch_test: dict[str, Any],
    best_line_ic: dict[str, Any] | None,
    best_line_false_safe: dict[str, Any] | None,
) -> str:
    patch_ic = _safe_float(patch_test.get("ic_mean"))
    patch_spread = _safe_float(patch_test.get("long_short_spread"))
    baseline_ic = _safe_float(((best_line_ic or {}).get("test") or {}).get("ic_mean"))
    baseline_name = (best_line_ic or {}).get("name", "없음")
    patch_tail = _safe_float(patch_test.get("false_safe_tail_rate"))
    best_tail = _safe_float(((best_line_false_safe or {}).get("test") or {}).get("false_safe_tail_rate"))
    tail_name = (best_line_false_safe or {}).get("name", "없음")
    verdict = "PatchTST가 line 기준선보다 우세합니다."
    if baseline_ic is not None and patch_ic is not None and patch_ic <= baseline_ic:
        verdict = "PatchTST가 IC 기준선 우위를 충분히 확보하지 못했습니다."
    if best_tail is not None and patch_tail is not None and patch_tail > best_tail:
        verdict += " false-safe 기준도 추가 점검이 필요합니다."
    return (
        f"{verdict} PatchTST h5 longer-context test IC={patch_ic}, spread={patch_spread}. "
        f"최고 IC baseline은 {baseline_name} IC={baseline_ic}. "
        f"false_safe_tail은 PatchTST={patch_tail}, 최저 baseline {tail_name}={best_tail}입니다."
    )


def _compare_cnn_band(
    *,
    cnn_test: dict[str, Any],
    best_band_interval: dict[str, Any] | None,
    best_band_calibration: dict[str, Any] | None,
) -> str:
    cnn_error = _safe_float(cnn_test.get("coverage_abs_error"))
    cnn_interval = _safe_float(cnn_test.get("interval_score"))
    interval_name = (best_band_interval or {}).get("name", "없음")
    interval_value = _safe_float(((best_band_interval or {}).get("test") or {}).get("interval_score"))
    cal_name = (best_band_calibration or {}).get("name", "없음")
    cal_value = _safe_float(((best_band_calibration or {}).get("test") or {}).get("coverage_abs_error"))
    verdict = "CNN-LSTM band가 baseline을 이기지 못했습니다."
    if (
        cnn_error is not None
        and cal_value is not None
        and cnn_interval is not None
        and interval_value is not None
        and cnn_error <= cal_value
        and cnn_interval <= interval_value
    ):
        verdict = "CNN-LSTM band가 calibration/interval 기준에서 baseline을 이겼습니다."
    return (
        f"{verdict} CNN-LSTM band coverage_abs_error={cnn_error}, interval_score={cnn_interval}. "
        f"최저 interval baseline은 {interval_name}={interval_value}, "
        f"최저 coverage error baseline은 {cal_name}={cal_value}입니다."
    )


def _compare_dynamic_width(*, cnn_test: dict[str, Any], best_dynamic_width: dict[str, Any] | None) -> str:
    cnn_width_ic = _safe_float(cnn_test.get("band_width_ic"))
    cnn_downside_ic = _safe_float(cnn_test.get("downside_width_ic"))
    baseline_name = (best_dynamic_width or {}).get("name", "없음")
    baseline_width_ic = _safe_float(((best_dynamic_width or {}).get("test") or {}).get("band_width_ic"))
    verdict = "CNN-LSTM의 동적 폭 신호는 양수지만, 최고 단순 baseline보다 약합니다."
    if baseline_width_ic is not None and cnn_width_ic is not None and cnn_width_ic >= baseline_width_ic:
        verdict = "CNN-LSTM의 동적 폭 신호가 단순 baseline보다 강합니다."
    return (
        f"{verdict} CNN-LSTM band_width_ic={cnn_width_ic}, downside_width_ic={cnn_downside_ic}. "
        f"최고 width baseline은 {baseline_name} band_width_ic={baseline_width_ic}입니다."
    )


def _wide_band_answer(*, cnn_test: dict[str, Any]) -> str:
    coverage_error = _safe_float(cnn_test.get("coverage_abs_error"))
    width_ic = _safe_float(cnn_test.get("band_width_ic"))
    downside_ic = _safe_float(cnn_test.get("downside_width_ic"))
    if (coverage_error is not None and coverage_error <= 0.15) and (width_ic or 0) > 0 and (downside_ic or 0) > 0:
        return "단순히 넓어서 맞추는 후보로만 보기는 어렵지만, baseline 대비 우위는 아직 부족합니다. nominal 대비 오차는 생존권이고 폭-위험 상관은 양수입니다."
    return "동적 위험 폭 근거가 약하거나 calibration 오차가 커서, 넓어서 맞추는 후보로 봐야 합니다."


def _format_float(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _line_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 후보 | IC | spread | IC_IR | false_safe_tail | severe_recall | downside_capture | sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("test") or {}
        lines.append(
            "| {name} | {ic} | {spread} | {ic_ir} | {tail} | {recall} | {capture} | {sharpe} |".format(
                name=row.get("name"),
                ic=_format_float(metrics.get("ic_mean")),
                spread=_format_float(metrics.get("long_short_spread")),
                ic_ir=_format_float(metrics.get("ic_ir")),
                tail=_format_float(metrics.get("false_safe_tail_rate")),
                recall=_format_float(metrics.get("severe_downside_recall")),
                capture=_format_float(metrics.get("downside_capture_rate")),
                sharpe=_format_float(metrics.get("fee_adjusted_sharpe")),
            )
        )
    return "\n".join(lines)


def _band_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 후보 | nominal | empirical | abs_error | width | interval | width_ic | downside_ic | squeeze |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row.get("test") or {}
        lines.append(
            "| {name} | {nominal} | {empirical} | {error} | {width} | {interval} | {width_ic} | {downside_ic} | {squeeze} |".format(
                name=row.get("name"),
                nominal=_format_float(metrics.get("nominal_coverage")),
                empirical=_format_float(metrics.get("empirical_coverage")),
                error=_format_float(metrics.get("coverage_abs_error")),
                width=_format_float(metrics.get("avg_band_width")),
                interval=_format_float(metrics.get("interval_score")),
                width_ic=_format_float(metrics.get("band_width_ic")),
                downside_ic=_format_float(metrics.get("downside_width_ic")),
                squeeze=_format_float(metrics.get("squeeze_breakout_rate")),
            )
        )
    return "\n".join(lines)


def _render_report(payload: dict[str, Any]) -> str:
    ai_line_rows = [
        {"name": row.get("name"), "test": row.get("test") or {}, "status": row.get("status")}
        for row in payload["ai_reference"]["line_candidates"]
    ]
    ai_band_rows = [
        {"name": row.get("name"), "test": row.get("test") or {}, "status": row.get("status")}
        for row in payload["ai_reference"]["band_candidates"]
    ]
    composite_rows = payload["ai_reference"]["composite_policies"]
    answer_lines = "\n".join(f"- {key}: {value}" for key, value in payload["answers"].items())
    return f"""# CP54 기준선 지표 비교 보고서

CP54는 새 학습이 아니라 CP52/CP53 표준 지표판을 단순 baseline에도 적용해 AI 후보가 실제 기준선을 이겼는지 확인하는 CP다.

## Executive Summary
- 평가 공간: raw_future_return, timeframe={payload['scope']['timeframe']}, horizon={payload['scope']['horizon']}, limit_tickers={payload['scope']['limit_tickers']}.
- 기준 quantile: q_low={payload['scope']['q_low']}, q_high={payload['scope']['q_high']}, nominal_coverage={payload['scope']['nominal_coverage']}.
- Line baseline은 zero, historical mean, momentum, reversal, shuffled score를 계산했다.
- Band baseline은 constant train quantile, rolling historical quantile, return-space Bollinger, volatility-scaled constant band를 계산했다.
- AI 후보 수: line {len(ai_line_rows)}개, band {len(ai_band_rows)}개, composite policy {len(composite_rows)}개.
- 결론: PatchTST line은 단순 line 기준선을 대체로 이겼지만, CNN-LSTM band는 rolling historical quantile/Bollinger baseline을 interval_score와 coverage_abs_error에서 아직 못 이겼다.

## 판정 질문 답변
{answer_lines}

## Line Baseline
{_line_table(payload['line_baselines'])}

## AI Line 후보
{_line_table(ai_line_rows)}

## Band Baseline
{_band_table(payload['band_baselines'])}

## AI Band 후보
{_band_table(ai_band_rows)}

## Composite 정책 참고
| 정책 | line_inside | warning | conservative_false_safe | width_increase | coverage | upper_breach |
|---|---:|---:|---:|---:|---:|---:|
{_composite_table_rows(composite_rows)}

## 기존 판정 변화
- CP53의 PatchTST h5 longer-context 후보는 line 기준선과 나란히 비교한다. baseline이 일부 지표에서 더 좋으면 AI가 아직 못 이긴 영역으로 기록한다.
- CNN-LSTM band 후보는 nominal 대비 coverage_abs_error와 interval_score를 baseline과 직접 비교한다.
- Composite는 모델 성능 판정이 아니라 제품 표시/정책 지표로 유지한다.

## 남은 리스크
- CP54는 기존 CP53 AI 산출물을 재사용하고 baseline만 새로 계산했다. AI checkpoint 재추론을 새로 돌리지 않았다.
- h20 branch는 h5 baseline 표와 직접 경쟁시키지 않았다.
- random_or_shuffled_score는 후보가 아니라 downside_capture_rate 무작위 기준 확인용이다.

## 다음 CP 추천
- AI line이 momentum/reversal보다 약한 지표는 feature 또는 objective 재점검 대상으로 둔다.
- AI band가 Bollinger/rolling quantile보다 interval_score에서 밀리면 CNN-LSTM band sweep 대신 baseline-aware calibration을 먼저 해야 한다.
- h5와 h20은 계속 branch를 분리해서 보고한다.
"""


def _composite_table_rows(composite_rows: dict[str, Any]) -> str:
    rows: list[str] = []
    for name, block in composite_rows.items():
        metrics = block.get("test") or {}
        rows.append(
            "| {name} | {inside} | {warning} | {false_safe} | {increase} | {coverage} | {upper} |".format(
                name=name,
                inside=_format_float(metrics.get("line_inside_band_ratio")),
                warning=_format_float(metrics.get("product_display_warning_rate")),
                false_safe=_format_float(metrics.get("conservative_series_false_safe_rate")),
                increase=_format_float(metrics.get("composite_width_increase_ratio")),
                coverage=_format_float(metrics.get("coverage")),
                upper=_format_float(metrics.get("upper_breach_rate")),
            )
        )
    return "\n".join(rows)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    cp53 = _read_json(PROJECT_ROOT / args.cp53_json)
    train_bundle, val_bundle, test_bundle, _, _, plan = prepare_dataset_splits(
        timeframe=args.timeframe,
        seq_len=args.seq_len,
        horizon=args.horizon,
        tickers=None,
        limit_tickers=args.limit_tickers,
        include_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
    )
    train_bundle = _require_lazy(train_bundle, label="train")
    val_bundle = _require_lazy(val_bundle, label="validation")
    test_bundle = _require_lazy(test_bundle, label="test")
    train_data = SplitData(bundle=train_bundle)
    val_data = SplitData(bundle=val_bundle)
    test_data = SplitData(bundle=test_bundle)
    thresholds = estimate_train_risk_thresholds(train_bundle)

    line_baselines = _evaluate_line_baselines(
        train_bundle=train_bundle,
        val_data=val_data,
        test_data=test_data,
        q_low=args.q_low,
        q_high=args.q_high,
        thresholds=thresholds,
    )
    band_baselines = _evaluate_band_baselines(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        q_low=args.q_low,
        q_high=args.q_high,
        thresholds=thresholds,
    )
    payload: dict[str, Any] = {
        "cp": "CP54-M",
        "purpose": "CP52/CP53 표준 지표판을 단순 baseline에도 적용해 AI 후보의 의미를 검증한다.",
        "rules": {
            "new_deep_learning_training": False,
            "full_473_training": False,
            "save_run": False,
            "db_schema_change": False,
            "ui_change": False,
            "fake_data": False,
            "model_structure_change": False,
        },
        "scope": {
            "timeframe": args.timeframe,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "limit_tickers": args.limit_tickers,
            "q_low": args.q_low,
            "q_high": args.q_high,
            "nominal_coverage": args.q_high - args.q_low,
            "feature_version": "v3_adjusted_ohlc",
            "eligible_ticker_count": plan.num_tickers,
            "train_samples": len(train_bundle),
            "validation_samples": len(val_bundle),
            "test_samples": len(test_bundle),
            "risk_thresholds": thresholds,
        },
        "line_baselines": line_baselines,
        "band_baselines": band_baselines,
        "ai_reference_source": _repo_path(PROJECT_ROOT / args.cp53_json),
        "ai_reference": _select_ai_candidates(cp53),
        "limitations": [
            "AI 후보는 CP53 재채점 결과를 참조했고, CP54에서 새 딥러닝 추론/학습을 실행하지 않았다.",
            "baseline은 h5 1D raw_future_return 기준으로 계산했다. h20 후보는 별도 branch로 유지한다.",
        ],
    }
    payload["answers"] = _answer_questions(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP54 기준선 지표 비교")
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--seq-len", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--limit-tickers", type=int, default=50)
    parser.add_argument("--q-low", type=float, default=0.15)
    parser.add_argument("--q-high", type=float, default=0.85)
    parser.add_argument("--cp53-json", default="docs/cp53_existing_candidate_regrade_metrics.json")
    parser.add_argument("--output-json", default="docs/cp54_baseline_metric_comparison_metrics.json")
    parser.add_argument("--output-report", default="docs/cp54_baseline_metric_comparison_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    output_json = PROJECT_ROOT / args.output_json
    output_report = PROJECT_ROOT / args.output_report
    _write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": _repo_path(output_json),
                "output_report": _repo_path(output_report),
                "line_baselines": len(payload["line_baselines"]),
                "band_baselines": len(payload["band_baselines"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
