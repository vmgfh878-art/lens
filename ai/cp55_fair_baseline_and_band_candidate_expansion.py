from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from ai.inference import load_checkpoint_config, resolve_checkpoint_ticker_registry  # noqa: E402
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

CORE_BAND_COMPARE = (
    ("interval_score", "lower"),
    ("coverage_abs_error", "lower"),
    ("band_width_ic", "higher"),
    ("downside_width_ic", "higher"),
)

_CHECKPOINT_CONFIG_CACHE: dict[str, dict[str, Any]] = {}
_SPLIT_CACHE: dict[str, tuple[Any, Any, Any, dict[str, float | None]]] = {}
_BASELINE_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class Candidate:
    name: str
    role: str
    family: str
    source: str
    checkpoint_path: str | None
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    config_hint: dict[str, Any]
    calibration_method: str = "none"
    notes: str = ""


@dataclass
class SplitData:
    bundle: SequenceDataset
    metadata: pd.DataFrame
    raw_future_returns: np.ndarray
    horizon: int


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


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _repo_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _metric_subset(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in keys}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _checkpoint_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    repo_path = _repo_path(path)
    if repo_path is None:
        return {}
    if repo_path not in _CHECKPOINT_CONFIG_CACHE:
        full_path = PROJECT_ROOT / repo_path
        if not full_path.exists():
            _CHECKPOINT_CONFIG_CACHE[repo_path] = {}
        else:
            _CHECKPOINT_CONFIG_CACHE[repo_path] = load_checkpoint_config(full_path)
    return dict(_CHECKPOINT_CONFIG_CACHE[repo_path])


def _merge_config(candidate: Candidate) -> dict[str, Any]:
    config = _checkpoint_config(candidate.checkpoint_path)
    merged = {**candidate.config_hint, **{key: value for key, value in config.items() if value is not None}}
    if "model" not in merged:
        merged["model"] = candidate.family
    if "timeframe" not in merged:
        merged["timeframe"] = "1D"
    if "horizon" not in merged:
        merged["horizon"] = 5
    if "seq_len" not in merged:
        merged["seq_len"] = 252
    if "q_low" not in merged:
        merged["q_low"] = 0.15
    if "q_high" not in merged:
        merged["q_high"] = 0.85
    if "line_target_type" not in merged:
        merged["line_target_type"] = "raw_future_return"
    if "band_target_type" not in merged:
        merged["band_target_type"] = "raw_future_return"
    if "feature_version" not in merged:
        merged["feature_version"] = "v3_adjusted_ohlc"
    if "limit_tickers" not in merged:
        merged["limit_tickers"] = candidate.config_hint.get("limit_tickers", 50)
    return merged


def _normalize_band_metrics(metrics: dict[str, Any], *, q_low: float, q_high: float) -> dict[str, Any]:
    normalized = dict(metrics or {})
    nominal = float(q_high) - float(q_low)
    if normalized.get("nominal_coverage") is None:
        normalized["nominal_coverage"] = nominal
    if normalized.get("empirical_coverage") is None and normalized.get("coverage") is not None:
        normalized["empirical_coverage"] = normalized.get("coverage")
    if normalized.get("coverage_abs_error") is None and normalized.get("empirical_coverage") is not None:
        normalized["coverage_abs_error"] = abs(float(normalized["empirical_coverage"]) - nominal)
    if normalized.get("empirical_q_low") is None and normalized.get("lower_breach_rate") is not None:
        normalized["empirical_q_low"] = normalized.get("lower_breach_rate")
    if normalized.get("empirical_q_high") is None and normalized.get("upper_breach_rate") is not None:
        normalized["empirical_q_high"] = 1.0 - float(normalized["upper_breach_rate"])
    return normalized


def _band_metrics_complete(metrics: dict[str, Any]) -> bool:
    return all(metrics.get(key) is not None for key in ("coverage_abs_error", "interval_score", "band_width_ic", "downside_width_ic"))


def _candidate_family(path: str | None, hint: dict[str, Any]) -> str:
    model = str(hint.get("model") or "").lower()
    if model:
        return model
    if path:
        name = Path(path).name.lower()
        if name.startswith("patchtst"):
            return "patchtst"
        if name.startswith("cnn_lstm"):
            return "cnn_lstm"
        if name.startswith("tide"):
            return "tide"
    return "unknown"


def _add_candidate(
    candidates: list[Candidate],
    *,
    name: str,
    role: str,
    source: str,
    checkpoint_path: str | None,
    validation: dict[str, Any] | None,
    test: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
    calibration_method: str = "none",
    notes: str = "",
) -> None:
    hint = dict(config or {})
    family = _candidate_family(checkpoint_path, hint)
    candidates.append(
        Candidate(
            name=name,
            role=role,
            family=family,
            source=source,
            checkpoint_path=_repo_path(checkpoint_path),
            validation_metrics=dict(validation or {}),
            test_metrics=dict(test or {}),
            config_hint=hint,
            calibration_method=calibration_method,
            notes=notes,
        )
    )


def _collect_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    cp53 = _read_json("docs/cp53_existing_candidate_regrade_metrics.json")
    for item in cp53.get("line_candidates", []):
        _add_candidate(
            candidates,
            name=str(item.get("name")),
            role="line",
            source="CP53 line",
            checkpoint_path=item.get("checkpoint_path"),
            validation=item.get("validation"),
            test=item.get("test"),
            config={"limit_tickers": item.get("limit_tickers")},
            notes="CP53 CP52 지표판 재채점 line 후보",
        )
        _add_candidate(
            candidates,
            name=f"patchtst_band_reference::{item.get('name')}",
            role="band",
            source="CP53 PatchTST band reference",
            checkpoint_path=item.get("checkpoint_path"),
            validation={},
            test=item.get("test_band_reference"),
            config={"limit_tickers": item.get("limit_tickers")},
            notes="PatchTST line checkpoint의 band head 참고 평가",
        )

    for item in cp53.get("band_candidates", []):
        _add_candidate(
            candidates,
            name=str(item.get("name")),
            role="band",
            source="CP53 band",
            checkpoint_path=item.get("checkpoint_path"),
            validation=item.get("validation"),
            test=item.get("test"),
            config={"limit_tickers": item.get("limit_tickers")},
            calibration_method="scalar_width" if "scalar" in str(item.get("name", "")) else "source_reported",
            notes="CP53 CP52 지표판 재채점 band 후보",
        )

    for source_path in (
        "docs/cp32_patchtst_clean_feature_revalidation_metrics.json",
        "docs/cp33_patchtst_revin_ablation_metrics.json",
    ):
        data = _read_json(source_path)
        for name, item in (data.get("runs") or {}).items():
            config = dict(item.get("config_summary") or item.get("config") or {})
            config["limit_tickers"] = (item.get("dataset_plan") or {}).get("input_ticker_count") or 50
            _add_candidate(
                candidates,
                name=f"{Path(source_path).stem}::{name}",
                role="band",
                source=Path(source_path).stem,
                checkpoint_path=item.get("checkpoint_path"),
                validation=item.get("validation"),
                test=item.get("test"),
                config=config,
                notes="PatchTST q/revin 후보. 기존 산출물은 CP52 전체 band 지표가 부족할 수 있음",
            )

    for source_path in (
        "docs/cp35_tide_cnn_band_rescue_smoke_metrics.json",
        "docs/cp37_role_based_model_recheck_metrics.json",
    ):
        data = _read_json(source_path)
        common = data.get("common_config") or data.get("common") or {}
        for name, item in (data.get("runs") or {}).items():
            config = {**common, **dict(item.get("config") or {})}
            config["limit_tickers"] = (item.get("dataset_plan") or {}).get("input_ticker_count") or common.get("limit_tickers") or 50
            role = "line" if str(config.get("role_requested", "")).startswith("line") else "band"
            _add_candidate(
                candidates,
                name=f"{Path(source_path).stem}::{name}",
                role=role,
                source=Path(source_path).stem,
                checkpoint_path=item.get("checkpoint_path"),
                validation=item.get("validation"),
                test=item.get("test"),
                config=config,
                notes="역할 분리/구조별 smoke 후보. 기존 산출물은 CP52 전체 지표가 부족할 수 있음",
            )

    cp38 = _read_json("docs/cp38_band_calibration_gap_metrics.json")
    for name, item in (cp38.get("candidates") or {}).items():
        for block_name in ("original", "scalar_width"):
            block = item.get(block_name) or {}
            if not block:
                continue
            config = dict(item.get("config") or {})
            config["limit_tickers"] = 50
            _add_candidate(
                candidates,
                name=f"cp38::{name}::{block_name}",
                role="band",
                source="cp38_band_calibration_gap",
                checkpoint_path=item.get("checkpoint_path"),
                validation=block.get("val"),
                test=block.get("test"),
                config=config,
                calibration_method=block_name,
                notes="CP38 calibration 후보. CP52 전체 band 지표가 부족할 수 있음",
            )

    for source_path in (
        "docs/cp45_cnn_lstm_band_sweep_metrics.json",
        "logs/cp45/cp45_actual_metrics.json",
        "logs/cp45/cp45_188_confirm_metrics.json",
    ):
        data = _read_json(source_path)
        for record in data.get("records", []):
            candidate = dict(record.get("candidate") or {})
            train = dict(record.get("train") or {})
            calibration = dict(record.get("band_calibration") or {})
            config = {**candidate, "model": "cnn_lstm", "limit_tickers": data.get("limit_tickers", 100)}
            for block_name in ("original", "scalar_width"):
                block = calibration.get(block_name) or {}
                if not block:
                    continue
                _add_candidate(
                    candidates,
                    name=f"{Path(source_path).stem}::{candidate.get('name')}::{block_name}",
                    role="band",
                    source=Path(source_path).stem,
                    checkpoint_path=train.get("checkpoint_path"),
                    validation=block.get("val"),
                    test=block.get("test"),
                    config=config,
                    calibration_method=block_name,
                    notes="CP45 CNN-LSTM sweep 후보. CP52 전체 band 지표가 부족할 수 있음",
                )

    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    deduped: dict[tuple[str, str, str], Candidate] = {}
    for candidate in candidates:
        key = (str(candidate.checkpoint_path), candidate.role, candidate.calibration_method)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            continue
        existing_complete = _band_metrics_complete(existing.test_metrics) if existing.role == "band" else bool(existing.test_metrics.get("ic_mean") is not None)
        candidate_complete = _band_metrics_complete(candidate.test_metrics) if candidate.role == "band" else bool(candidate.test_metrics.get("ic_mean") is not None)
        if candidate_complete and not existing_complete:
            deduped[key] = candidate
        elif candidate_complete == existing_complete and candidate.source.startswith("CP53"):
            deduped[key] = candidate
    return list(deduped.values())


def _extract_raw_returns(bundle: SequenceDataset | SequenceDatasetBundle) -> np.ndarray:
    if isinstance(bundle, SequenceDatasetBundle):
        return bundle.raw_future_returns.detach().cpu().numpy().astype("float32", copy=True)
    values = np.empty((len(bundle), bundle.horizon), dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        closes = bundle.ticker_arrays[ticker]["closes"]
        anchor_close = max(float(closes[end_idx]), 1e-6)
        future_start = end_idx + 1
        future_end = future_start + bundle.horizon
        values[row_idx] = (closes[future_start:future_end] / anchor_close) - 1.0
    return values


def _require_lazy(bundle: SequenceDataset | SequenceDatasetBundle, *, label: str) -> SequenceDataset:
    if not isinstance(bundle, SequenceDataset):
        raise TypeError(f"{label} split이 lazy SequenceDataset이 아닙니다.")
    return bundle


def _split_key(config: dict[str, Any], *, limit_tickers: int) -> str:
    registry_path = config.get("ticker_registry_path")
    return json.dumps(
        {
            "timeframe": config.get("timeframe", "1D"),
            "seq_len": int(config.get("seq_len", 252)),
            "horizon": int(config.get("horizon", 5)),
            "limit_tickers": int(limit_tickers),
            "registry_path": str(registry_path) if registry_path else None,
            "future": bool(config.get("use_future_covariate", config.get("model") == "tide")),
        },
        sort_keys=True,
    )


def _prepare_splits(config: dict[str, Any], *, limit_tickers: int) -> tuple[SplitData, SplitData, SplitData, dict[str, float | None]]:
    key = _split_key(config, limit_tickers=limit_tickers)
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    timeframe = str(config.get("timeframe", "1D"))
    ticker_registry = None
    try:
        ticker_registry = resolve_checkpoint_ticker_registry(config, timeframe)
    except Exception:
        ticker_registry = None
    train_bundle, val_bundle, test_bundle, _, _, _ = prepare_dataset_splits(
        timeframe=timeframe,
        seq_len=int(config.get("seq_len", 252)),
        horizon=int(config.get("horizon", 5)),
        tickers=None,
        limit_tickers=limit_tickers,
        include_future_covariate=bool(config.get("use_future_covariate", config.get("model") == "tide")),
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry=ticker_registry,
        ticker_registry_path=config.get("ticker_registry_path"),
    )
    train_bundle = _require_lazy(train_bundle, label="train")
    val_bundle = _require_lazy(val_bundle, label="validation")
    test_bundle = _require_lazy(test_bundle, label="test")
    thresholds = estimate_train_risk_thresholds(train_bundle)
    result = (
        SplitData(train_bundle, train_bundle.metadata.reset_index(drop=True).copy(), _extract_raw_returns(train_bundle), train_bundle.horizon),
        SplitData(val_bundle, val_bundle.metadata.reset_index(drop=True).copy(), _extract_raw_returns(val_bundle), val_bundle.horizon),
        SplitData(test_bundle, test_bundle.metadata.reset_index(drop=True).copy(), _extract_raw_returns(test_bundle), test_bundle.horizon),
        thresholds,
    )
    _SPLIT_CACHE[key] = result
    return result


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype="float32"))


def _evaluate(
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


def _rolling_stats_for_bundle(bundle: SequenceDataset, *, windows: tuple[int, ...], q_low: float, q_high: float) -> dict[tuple[str, int, int, str], np.ndarray]:
    cache: dict[tuple[str, int, int, str], np.ndarray] = {}
    for ticker, arrays in bundle.ticker_arrays.items():
        closes = arrays["closes"].astype("float64", copy=False)
        for horizon_step in range(1, bundle.horizon + 1):
            if len(closes) <= horizon_step:
                returns = np.empty(0, dtype="float64")
            else:
                returns = (closes[horizon_step:] / np.clip(closes[:-horizon_step], 1e-6, None)) - 1.0
            series = pd.Series(returns)
            for window in windows:
                if returns.size == 0:
                    for stat in ("mean", "std", "q_low", "q_high"):
                        cache[(ticker, horizon_step, window, stat)] = np.full(len(closes), np.nan, dtype="float32")
                    continue
                mean = series.rolling(window=window, min_periods=1).mean().to_numpy(dtype="float32")
                std = series.rolling(window=window, min_periods=2).std(ddof=0).fillna(0.0).to_numpy(dtype="float32")
                low = series.rolling(window=window, min_periods=1).quantile(q_low).to_numpy(dtype="float32")
                high = series.rolling(window=window, min_periods=1).quantile(q_high).to_numpy(dtype="float32")
                padded = len(closes)
                for stat, values in (("mean", mean), ("std", std), ("q_low", low), ("q_high", high)):
                    arr = np.full(padded, np.nan, dtype="float32")
                    arr[: len(values)] = values
                    cache[(ticker, horizon_step, window, stat)] = arr
    return cache


def _lookup_stat(
    stats: dict[tuple[str, int, int, str], np.ndarray],
    *,
    ticker: str,
    end_idx: int,
    horizon_step: int,
    window: int,
    stat: str,
    fallback: float,
) -> float:
    latest_anchor = end_idx - horizon_step
    if latest_anchor < 0:
        return fallback
    values = stats.get((ticker, horizon_step, window, stat))
    if values is None or latest_anchor >= len(values):
        return fallback
    value = float(values[latest_anchor])
    return value if np.isfinite(value) else fallback


def _historical_mean_line(data: SplitData, stats: dict[tuple[str, int, int, str], np.ndarray], *, window: int) -> np.ndarray:
    output = np.zeros_like(data.raw_future_returns, dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(data.bundle.sample_refs):
        for horizon_idx in range(data.horizon):
            output[row_idx, horizon_idx] = _lookup_stat(
                stats,
                ticker=ticker,
                end_idx=end_idx,
                horizon_step=horizon_idx + 1,
                window=window,
                stat="mean",
                fallback=0.0,
            )
    return output


def _momentum_line(data: SplitData, *, sign: float = 1.0) -> np.ndarray:
    output = np.zeros_like(data.raw_future_returns, dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(data.bundle.sample_refs):
        closes = data.bundle.ticker_arrays[ticker]["closes"]
        for horizon_idx in range(data.horizon):
            step = horizon_idx + 1
            start_idx = end_idx - step
            if start_idx < 0:
                continue
            value = (float(closes[end_idx]) / max(float(closes[start_idx]), 1e-6)) - 1.0
            output[row_idx, horizon_idx] = sign * value if np.isfinite(value) else 0.0
    return output


def _shuffle_by_date(scores: np.ndarray, metadata: pd.DataFrame, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = scores.copy()
    for _, index in metadata.groupby("asof_date", sort=True).groups.items():
        indices = np.asarray(list(index), dtype=np.int64)
        if indices.size <= 1:
            continue
        source = indices.copy()
        rng.shuffle(source)
        output[indices] = scores[source]
    return output


def _constant_band(train_raw: np.ndarray, shape: tuple[int, int], *, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
    lower_by_h = np.quantile(train_raw, q_low, axis=0).astype("float32")
    upper_by_h = np.quantile(train_raw, q_high, axis=0).astype("float32")
    return (
        np.tile(lower_by_h.reshape(1, -1), (shape[0], 1)).astype("float32"),
        np.tile(upper_by_h.reshape(1, -1), (shape[0], 1)).astype("float32"),
    )


def _rolling_quantile_band(
    data: SplitData,
    stats: dict[tuple[str, int, int, str], np.ndarray],
    *,
    window: int,
    fallback_lower: np.ndarray,
    fallback_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty_like(data.raw_future_returns, dtype="float32")
    upper = np.empty_like(data.raw_future_returns, dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(data.bundle.sample_refs):
        for horizon_idx in range(data.horizon):
            step = horizon_idx + 1
            lower[row_idx, horizon_idx] = _lookup_stat(stats, ticker=ticker, end_idx=end_idx, horizon_step=step, window=window, stat="q_low", fallback=float(fallback_lower[horizon_idx]))
            upper[row_idx, horizon_idx] = _lookup_stat(stats, ticker=ticker, end_idx=end_idx, horizon_step=step, window=window, stat="q_high", fallback=float(fallback_upper[horizon_idx]))
    return lower, upper


def _bollinger_band(
    data: SplitData,
    stats: dict[tuple[str, int, int, str], np.ndarray],
    *,
    window: int,
    k_std: float,
    fallback_mean: np.ndarray,
    fallback_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty_like(data.raw_future_returns, dtype="float32")
    upper = np.empty_like(data.raw_future_returns, dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(data.bundle.sample_refs):
        for horizon_idx in range(data.horizon):
            step = horizon_idx + 1
            mean = _lookup_stat(stats, ticker=ticker, end_idx=end_idx, horizon_step=step, window=window, stat="mean", fallback=float(fallback_mean[horizon_idx]))
            std = _lookup_stat(stats, ticker=ticker, end_idx=end_idx, horizon_step=step, window=window, stat="std", fallback=float(fallback_std[horizon_idx]))
            lower[row_idx, horizon_idx] = mean - (k_std * std)
            upper[row_idx, horizon_idx] = mean + (k_std * std)
    return lower, upper


def _recent_daily_vol(closes: np.ndarray, *, end_idx: int, window: int) -> float:
    start = max(0, end_idx - window)
    if end_idx <= start:
        return 0.0
    history = closes[start : end_idx + 1]
    daily = np.diff(history) / np.clip(history[:-1], 1e-6, None)
    daily = daily[np.isfinite(daily)]
    return float(daily.std(ddof=0)) if daily.size else 0.0


def _volatility_scaled_band(data: SplitData, train_raw: np.ndarray, *, window: int, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
    train_lower = np.quantile(train_raw, q_low, axis=0).astype("float32")
    train_upper = np.quantile(train_raw, q_high, axis=0).astype("float32")
    center = ((train_lower + train_upper) / 2.0).astype("float32")
    half_width = np.maximum((train_upper - train_lower) / 2.0, 1e-6).astype("float32")
    train_std = np.maximum(train_raw.std(axis=0), 1e-6).astype("float32")
    lower = np.empty_like(data.raw_future_returns, dtype="float32")
    upper = np.empty_like(data.raw_future_returns, dtype="float32")
    for row_idx, (ticker, end_idx) in enumerate(data.bundle.sample_refs):
        closes = data.bundle.ticker_arrays[ticker]["closes"]
        daily_vol = _recent_daily_vol(closes, end_idx=end_idx, window=window)
        for horizon_idx in range(data.horizon):
            scale = np.clip((daily_vol * np.sqrt(horizon_idx + 1)) / float(train_std[horizon_idx]), 0.25, 4.0)
            lower[row_idx, horizon_idx] = center[horizon_idx] - (half_width[horizon_idx] * scale)
            upper[row_idx, horizon_idx] = center[horizon_idx] + (half_width[horizon_idx] * scale)
    return lower, upper


def _baseline_pack(config: dict[str, Any], *, limit_tickers: int, q_low: float, q_high: float) -> dict[str, Any]:
    key = json.dumps(
        {
            "split": _split_key(config, limit_tickers=limit_tickers),
            "q_low": q_low,
            "q_high": q_high,
        },
        sort_keys=True,
    )
    if key in _BASELINE_CACHE:
        return _BASELINE_CACHE[key]

    train, validation, test, thresholds = _prepare_splits(config, limit_tickers=limit_tickers)
    windows = (20, 60, 120, 252)
    val_stats = _rolling_stats_for_bundle(validation.bundle, windows=windows, q_low=q_low, q_high=q_high)
    test_stats = _rolling_stats_for_bundle(test.bundle, windows=windows, q_low=q_low, q_high=q_high)
    fallback_lower = np.quantile(train.raw_future_returns, q_low, axis=0).astype("float32")
    fallback_upper = np.quantile(train.raw_future_returns, q_high, axis=0).astype("float32")
    fallback_mean = train.raw_future_returns.mean(axis=0).astype("float32")
    fallback_std = np.maximum(train.raw_future_returns.std(axis=0), 1e-6).astype("float32")

    shape_val = validation.raw_future_returns.shape
    shape_test = test.raw_future_returns.shape
    wide_val = (np.full(shape_val, -1_000_000.0, dtype="float32"), np.full(shape_val, 1_000_000.0, dtype="float32"))
    wide_test = (np.full(shape_test, -1_000_000.0, dtype="float32"), np.full(shape_test, 1_000_000.0, dtype="float32"))

    val_momentum = _momentum_line(validation)
    test_momentum = _momentum_line(test)
    line_specs: list[tuple[str, dict[str, Any], np.ndarray, np.ndarray]] = [
        ("zero_line", {}, np.zeros(shape_val, dtype="float32"), np.zeros(shape_test, dtype="float32")),
        ("historical_mean_line_w20", {"window": 20}, _historical_mean_line(validation, val_stats, window=20), _historical_mean_line(test, test_stats, window=20)),
        ("historical_mean_line_w60", {"window": 60}, _historical_mean_line(validation, val_stats, window=60), _historical_mean_line(test, test_stats, window=60)),
        ("momentum_line_horizon", {"lookback": "horizon_step"}, val_momentum, test_momentum),
        ("reversal_line_horizon", {"lookback": "horizon_step", "sign": -1}, -val_momentum, -test_momentum),
    ]
    for seed in range(30):
        line_specs.append(
            (
                f"random_or_shuffled_score_seed_{seed}",
                {"source": "datewise_shuffled_momentum", "seed": seed},
                _shuffle_by_date(val_momentum, validation.metadata, seed=seed),
                _shuffle_by_date(test_momentum, test.metadata, seed=seed),
            )
        )

    line_baselines = []
    for name, baseline_config, val_line, test_line in line_specs:
        line_baselines.append(
            {
                "name": name,
                "config": baseline_config,
                "validation": _metric_subset(
                    _evaluate(data=validation, line=val_line, lower=wide_val[0], upper=wide_val[1], q_low=q_low, q_high=q_high, thresholds=thresholds),
                    LINE_METRIC_KEYS,
                ),
                "test": _metric_subset(
                    _evaluate(data=test, line=test_line, lower=wide_test[0], upper=wide_test[1], q_low=q_low, q_high=q_high, thresholds=thresholds),
                    LINE_METRIC_KEYS,
                ),
            }
        )

    band_specs: list[tuple[str, dict[str, Any], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = []
    band_specs.append(
        (
            "constant_width_train_quantile",
            {"q_low": q_low, "q_high": q_high},
            _constant_band(train.raw_future_returns, shape_val, q_low=q_low, q_high=q_high),
            _constant_band(train.raw_future_returns, shape_test, q_low=q_low, q_high=q_high),
        )
    )
    for window in (60, 120, 252):
        band_specs.append(
            (
                f"rolling_historical_quantile_band_w{window}",
                {"window": window, "q_low": q_low, "q_high": q_high},
                _rolling_quantile_band(validation, val_stats, window=window, fallback_lower=fallback_lower, fallback_upper=fallback_upper),
                _rolling_quantile_band(test, test_stats, window=window, fallback_lower=fallback_lower, fallback_upper=fallback_upper),
            )
        )
    for window in (20, 60):
        for k_std in (1.0, 1.5, 2.0):
            band_specs.append(
                (
                    f"rolling_bollinger_return_band_w{window}_k{k_std:g}",
                    {"window": window, "k_std": k_std},
                    _bollinger_band(validation, val_stats, window=window, k_std=k_std, fallback_mean=fallback_mean, fallback_std=fallback_std),
                    _bollinger_band(test, test_stats, window=window, k_std=k_std, fallback_mean=fallback_mean, fallback_std=fallback_std),
                )
            )
    for window in (20, 60):
        band_specs.append(
            (
                f"volatility_scaled_constant_band_w{window}",
                {"window": window, "base": "train_quantile_width", "scale_clip": [0.25, 4.0]},
                _volatility_scaled_band(validation, train.raw_future_returns, window=window, q_low=q_low, q_high=q_high),
                _volatility_scaled_band(test, train.raw_future_returns, window=window, q_low=q_low, q_high=q_high),
            )
        )

    band_baselines = []
    for name, baseline_config, val_band, test_band in band_specs:
        val_lower, val_upper = val_band
        test_lower, test_upper = test_band
        val_line = ((val_lower + val_upper) / 2.0).astype("float32")
        test_line = ((test_lower + test_upper) / 2.0).astype("float32")
        band_baselines.append(
            {
                "name": name,
                "config": baseline_config,
                "validation": _metric_subset(
                    _evaluate(data=validation, line=val_line, lower=val_lower, upper=val_upper, q_low=q_low, q_high=q_high, thresholds=thresholds),
                    BAND_METRIC_KEYS,
                ),
                "test": _metric_subset(
                    _evaluate(data=test, line=test_line, lower=test_lower, upper=test_upper, q_low=q_low, q_high=q_high, thresholds=thresholds),
                    BAND_METRIC_KEYS,
                ),
            }
        )

    pack = {
        "scope": {
            "timeframe": config.get("timeframe", "1D"),
            "seq_len": int(config.get("seq_len", 252)),
            "horizon": int(config.get("horizon", 5)),
            "limit_tickers": int(limit_tickers),
            "q_low": q_low,
            "q_high": q_high,
            "nominal_coverage": q_high - q_low,
            "train_samples": len(train.bundle),
            "validation_samples": len(validation.bundle),
            "test_samples": len(test.bundle),
            "risk_thresholds": thresholds,
        },
        "line_baselines": line_baselines,
        "band_baselines": band_baselines,
    }
    _BASELINE_CACHE[key] = pack
    return pack


def _best_baseline(rows: list[dict[str, Any]], *, metric: str, direction: str) -> dict[str, Any] | None:
    best = None
    best_value = None
    for row in rows:
        value = _safe_float((row.get("test") or {}).get(metric))
        if value is None:
            continue
        if best is None:
            best = row
            best_value = value
            continue
        assert best_value is not None
        if (direction == "lower" and value < best_value) or (direction == "higher" and value > best_value):
            best = row
            best_value = value
    return best


def _compare_band(ai_metrics: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    baseline_metrics = (baseline or {}).get("test") or {}
    ai_wins = 0
    baseline_wins = 0
    ties_or_missing = 0
    per_metric = {}
    for metric, direction in CORE_BAND_COMPARE:
        ai_value = _safe_float(ai_metrics.get(metric))
        baseline_value = _safe_float(baseline_metrics.get(metric))
        if ai_value is None or baseline_value is None:
            ties_or_missing += 1
            per_metric[metric] = "missing"
            continue
        if abs(ai_value - baseline_value) < 1e-12:
            ties_or_missing += 1
            per_metric[metric] = "tie"
        elif (direction == "lower and ai" and False):
            pass
        elif (direction == "lower" and ai_value < baseline_value) or (direction == "higher" and ai_value > baseline_value):
            ai_wins += 1
            per_metric[metric] = "ai"
        else:
            baseline_wins += 1
            per_metric[metric] = "baseline"
    return {
        "ai_wins": ai_wins,
        "baseline_wins": baseline_wins,
        "ties_or_missing": ties_or_missing,
        "per_metric": per_metric,
    }


def _band_verdict(ai_metrics: dict[str, Any], comparison: dict[str, Any], *, complete: bool) -> str:
    if not complete:
        return "재실험 필요"
    ai_wins = int(comparison["ai_wins"])
    baseline_wins = int(comparison["baseline_wins"])
    width_ic = _safe_float(ai_metrics.get("band_width_ic")) or 0.0
    downside_ic = _safe_float(ai_metrics.get("downside_width_ic")) or 0.0
    coverage_error = _safe_float(ai_metrics.get("coverage_abs_error"))
    if ai_wins >= 3:
        return "후보"
    if ai_wins >= 1:
        return "생존"
    if baseline_wins >= 4:
        return "탈락"
    if coverage_error is not None and coverage_error <= 0.15 and (width_ic > 0 or downside_ic > 0):
        return "구조 보류"
    return "보류"


def _random_summary(line_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    random_rows = [row for row in line_baselines if str(row.get("name", "")).startswith("random_or_shuffled_score_seed_")]
    values = [(_safe_float((row.get("test") or {}).get("ic_mean")), row) for row in random_rows]
    finite = [(value, row) for value, row in values if value is not None]
    if not finite:
        return {"seed_count": len(random_rows), "ic_mean_avg": None, "ic_mean_best": None, "best_seed": None}
    best_value, best_row = max(finite, key=lambda item: item[0])
    return {
        "seed_count": len(random_rows),
        "ic_mean_avg": float(np.mean([value for value, _ in finite])),
        "ic_mean_best": best_value,
        "best_seed": best_row.get("name"),
    }


def _analyze_line_candidate(candidate: Candidate, config: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(candidate.test_metrics)
    best_ic_baseline = _best_baseline(pack["line_baselines"], metric="ic_mean", direction="higher")
    random_summary = _random_summary(pack["line_baselines"])
    return {
        "name": candidate.name,
        "family": candidate.family,
        "source": candidate.source,
        "checkpoint_path": candidate.checkpoint_path,
        "config": _config_summary(config),
        "ai_metrics": _metric_subset(metrics, LINE_METRIC_KEYS),
        "best_line_baseline_by_ic": {
            "name": (best_ic_baseline or {}).get("name"),
            "metrics": _metric_subset((best_ic_baseline or {}).get("test") or {}, LINE_METRIC_KEYS),
        },
        "random_or_shuffled_score": random_summary,
        "verdict": _line_verdict(metrics, best_ic_baseline),
    }


def _line_verdict(metrics: dict[str, Any], best_baseline: dict[str, Any] | None) -> str:
    ai_ic = _safe_float(metrics.get("ic_mean"))
    ai_spread = _safe_float(metrics.get("long_short_spread"))
    baseline_ic = _safe_float(((best_baseline or {}).get("test") or {}).get("ic_mean"))
    false_safe = _safe_float(metrics.get("false_safe_tail_rate"))
    severe = _safe_float(metrics.get("severe_downside_recall"))
    if ai_ic is None:
        return "재실험 필요"
    if ai_ic > 0 and (ai_spread or 0) > 0 and (baseline_ic is None or ai_ic >= baseline_ic):
        if (false_safe is not None and false_safe < 0.15) and (severe is not None and severe >= 0.80):
            return "후보"
        return "생존"
    if ai_ic > 0 and (ai_spread or 0) > 0:
        return "보류"
    return "탈락"


def _analyze_band_candidate(candidate: Candidate, config: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    q_low = float(config.get("q_low", 0.15))
    q_high = float(config.get("q_high", 0.85))
    metrics = _normalize_band_metrics(candidate.test_metrics, q_low=q_low, q_high=q_high)
    complete = _band_metrics_complete(metrics)
    best_interval_baseline = _best_baseline(pack["band_baselines"], metric="interval_score", direction="lower")
    comparison = _compare_band(metrics, best_interval_baseline)
    return {
        "name": candidate.name,
        "family": candidate.family,
        "source": candidate.source,
        "checkpoint_path": candidate.checkpoint_path,
        "calibration_method": candidate.calibration_method,
        "config": _config_summary(config),
        "ai_metrics": _metric_subset(metrics, BAND_METRIC_KEYS),
        "cp52_metric_complete": complete,
        "best_statistical_baseline": {
            "name": (best_interval_baseline or {}).get("name"),
            "metrics": _metric_subset((best_interval_baseline or {}).get("test") or {}, BAND_METRIC_KEYS),
        },
        "comparison": comparison,
        "verdict": _band_verdict(metrics, comparison, complete=complete),
        "notes": candidate.notes,
    }


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "role",
        "timeframe",
        "horizon",
        "seq_len",
        "q_low",
        "q_high",
        "lambda_band",
        "band_mode",
        "line_target_type",
        "band_target_type",
        "feature_version",
        "limit_tickers",
        "patch_len",
        "patch_stride",
        "use_revin",
        "use_future_covariate",
        "ticker_registry_path",
    )
    return {key: config.get(key) for key in keys if config.get(key) is not None}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _collect_candidates()
    line_results: list[dict[str, Any]] = []
    band_results: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for candidate in candidates:
        config = _merge_config(candidate)
        limit_tickers = int(config.get("limit_tickers") or 50)
        if args.max_limit_tickers and limit_tickers > args.max_limit_tickers:
            unsupported.append(
                {
                    "name": candidate.name,
                    "role": candidate.role,
                    "reason": f"limit_tickers={limit_tickers}가 max_limit_tickers={args.max_limit_tickers}보다 큼",
                    "checkpoint_path": candidate.checkpoint_path,
                }
            )
            continue
        try:
            pack = _baseline_pack(
                config,
                limit_tickers=limit_tickers,
                q_low=float(config.get("q_low", 0.15)),
                q_high=float(config.get("q_high", 0.85)),
            )
        except Exception as exc:
            unsupported.append(
                {
                    "name": candidate.name,
                    "role": candidate.role,
                    "reason": f"baseline 계산 실패: {type(exc).__name__}: {exc}",
                    "checkpoint_path": candidate.checkpoint_path,
                    "config": _config_summary(config),
                }
            )
            continue
        if candidate.role == "line":
            line_results.append(_analyze_line_candidate(candidate, config, pack))
        else:
            band_results.append(_analyze_band_candidate(candidate, config, pack))

    payload = {
        "cp": "CP55-M",
        "purpose": "후보별 동일 조건 통계 baseline 비교와 band 후보군 재확장",
        "rules": {
            "new_deep_learning_training": False,
            "save_run": False,
            "full_473_training": False,
            "ui_change": False,
            "db_schema_change": False,
            "model_structure_change": False,
            "fake_data": False,
        },
        "baseline_scope_note": "이번 baseline은 DLinear/NLinear/LightGBM/CatBoost 같은 학습형 baseline이 아니라 통계 baseline이다.",
        "baseline_definitions": {
            "line": [
                "zero_line",
                "historical_mean_line_w20/w60",
                "momentum_line_horizon",
                "reversal_line_horizon",
                "random_or_shuffled_score_seed_0..29",
            ],
            "band": [
                "constant_width_train_quantile",
                "rolling_historical_quantile_band_w60/w120/w252",
                "rolling_bollinger_return_band_w20/w60_k1/k1.5/k2",
                "volatility_scaled_constant_band_w20/w60",
            ],
        },
        "line_candidates": line_results,
        "band_candidates": band_results,
        "unsupported_or_reexperiment_required": unsupported,
        "summary": _build_summary(line_results, band_results, unsupported),
    }
    return payload


def _build_summary(line_results: list[dict[str, Any]], band_results: list[dict[str, Any]], unsupported: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for row in band_results:
        status = str(row.get("verdict"))
        family = str(row.get("family"))
        by_status[status] = by_status.get(status, 0) + 1
        by_family.setdefault(family, {})
        by_family[family][status] = by_family[family].get(status, 0) + 1
    best_band = sorted(
        [
            row
            for row in band_results
            if _safe_float((row.get("ai_metrics") or {}).get("interval_score")) is not None
        ],
        key=lambda row: float((row.get("ai_metrics") or {}).get("interval_score")),
    )
    return {
        "line_candidate_count": len(line_results),
        "band_candidate_count": len(band_results),
        "unsupported_or_reexperiment_required_count": len(unsupported),
        "band_verdict_counts": by_status,
        "band_verdict_by_family": by_family,
        "best_ai_band_by_interval_score": [
            {
                "name": row.get("name"),
                "family": row.get("family"),
                "interval_score": (row.get("ai_metrics") or {}).get("interval_score"),
                "best_baseline": ((row.get("best_statistical_baseline") or {}).get("name")),
                "baseline_interval_score": (((row.get("best_statistical_baseline") or {}).get("metrics") or {}).get("interval_score")),
                "verdict": row.get("verdict"),
            }
            for row in best_band[:5]
        ],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _band_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| AI 후보 | family | baseline | AI interval | baseline interval | AI cov err | baseline cov err | AI width_ic | baseline width_ic | AI 승 | baseline 승 | 판정 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ai = row.get("ai_metrics") or {}
        base = (row.get("best_statistical_baseline") or {}).get("metrics") or {}
        comp = row.get("comparison") or {}
        lines.append(
            "| {name} | {family} | {baseline} | {ai_interval} | {base_interval} | {ai_error} | {base_error} | {ai_width} | {base_width} | {ai_wins} | {base_wins} | {verdict} |".format(
                name=row.get("name"),
                family=row.get("family"),
                baseline=(row.get("best_statistical_baseline") or {}).get("name"),
                ai_interval=_fmt(ai.get("interval_score")),
                base_interval=_fmt(base.get("interval_score")),
                ai_error=_fmt(ai.get("coverage_abs_error")),
                base_error=_fmt(base.get("coverage_abs_error")),
                ai_width=_fmt(ai.get("band_width_ic")),
                base_width=_fmt(base.get("band_width_ic")),
                ai_wins=comp.get("ai_wins"),
                base_wins=comp.get("baseline_wins"),
                verdict=row.get("verdict"),
            )
        )
    return "\n".join(lines)


def _line_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| AI 후보 | family | AI IC | best baseline | baseline IC | AI spread | false_safe_tail | severe_recall | 판정 |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ai = row.get("ai_metrics") or {}
        base = row.get("best_line_baseline_by_ic") or {}
        lines.append(
            "| {name} | {family} | {ai_ic} | {baseline} | {base_ic} | {spread} | {false_safe} | {severe} | {verdict} |".format(
                name=row.get("name"),
                family=row.get("family"),
                ai_ic=_fmt(ai.get("ic_mean")),
                baseline=base.get("name"),
                base_ic=_fmt((base.get("metrics") or {}).get("ic_mean")),
                spread=_fmt(ai.get("long_short_spread")),
                false_safe=_fmt(ai.get("false_safe_tail_rate")),
                severe=_fmt(ai.get("severe_downside_recall")),
                verdict=row.get("verdict"),
            )
        )
    return "\n".join(lines)


def _render_report(payload: dict[str, Any]) -> str:
    band_sorted = sorted(
        payload["band_candidates"],
        key=lambda row: (
            {"후보": 0, "생존": 1, "구조 보류": 2, "보류": 3, "재실험 필요": 4, "탈락": 5}.get(str(row.get("verdict")), 9),
            row.get("family", ""),
            row.get("name", ""),
        ),
    )
    unsupported_lines = "\n".join(
        f"- {row.get('name')} ({row.get('role')}): {row.get('reason')}" for row in payload["unsupported_or_reexperiment_required"][:40]
    )
    if not unsupported_lines:
        unsupported_lines = "- 없음"
    return f"""# CP55 후보별 공정 baseline 비교 및 band 후보군 재확장 보고서

CP55는 새 학습이 아니라, 기존 checkpoint와 기존 산출물을 후보별 동일 조건의 통계 baseline과 비교하는 CP다.

## Executive Summary
- 이번 baseline은 모델 baseline이 아니라 통계 baseline이다. DLinear/NLinear/LightGBM/CatBoost 같은 학습형 baseline은 아직 아니다.
- line baseline은 zero, historical mean, momentum, reversal, random/shuffled 30 seeds를 포함했다.
- band baseline은 train quantile, rolling historical quantile, return-space Bollinger, volatility-scaled constant band를 포함했다.
- line 후보 {payload['summary']['line_candidate_count']}개, band 후보 {payload['summary']['band_candidate_count']}개를 정리했다.
- CP52 전체 band 지표가 없는 과거 후보는 새 추론을 돌리지 않고 `재실험 필요` 또는 부분 비교로 표시했다.

## Line 후보 공정 비교
{_line_table(payload['line_candidates'])}

## Band 후보 공정 비교
{_band_table(band_sorted)}

## Band 판정 요약
- 상태별: {json.dumps(payload['summary']['band_verdict_counts'], ensure_ascii=False)}
- family별: {json.dumps(payload['summary']['band_verdict_by_family'], ensure_ascii=False)}

## 핵심 해석
- PatchTST는 line 후보로 유지한다. band head는 과보수/통계 baseline 대비 약한 후보가 많아 line 전용 해석이 맞다.
- CNN-LSTM은 일부 dynamic width 신호가 있지만, 후보별 fair baseline에서 rolling quantile/Bollinger를 안정적으로 이겼다고 보기 어렵다.
- TiDE는 checkpoint는 남아 있으나 CP52 전체 지표가 부족한 산출물이 많다. 기존 regrade 기준으로도 downside_width_ic가 약해 주력 band 후보는 아니다.
- 통계 baseline이 강하므로 다음 band CP는 AI가 직접 lower/upper를 예측하는 방식보다 baseline-aware residual/scale band를 우선 검토해야 한다.

## 재실험 또는 미지원 목록
{unsupported_lines}

## 다음 단계 제안
- A안: rolling historical quantile 또는 Bollinger return band 위에 residual scale을 학습하는 baseline-aware band로 전환한다.
- B안: TiDE branch는 CP52 full regrade를 먼저 한 뒤 dynamic width가 살아나는 경우에만 재개한다.
- C안: CNN-LSTM은 direct lower/upper sweep보다 baseline 대비 residual correction으로 좁힌다.
- D안: PatchTST는 line 전용으로 유지하고 band 후보 경쟁에서는 분리한다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP55 후보별 공정 통계 baseline 비교")
    parser.add_argument("--output-json", default="docs/cp55_fair_baseline_and_band_candidate_expansion_metrics.json")
    parser.add_argument("--output-report", default="docs/cp55_fair_baseline_and_band_candidate_expansion_report.md")
    parser.add_argument("--max-limit-tickers", type=int, default=200)
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
                "line_candidates": len(payload["line_candidates"]),
                "band_candidates": len(payload["band_candidates"]),
                "unsupported": len(payload["unsupported_or_reexperiment_required"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
