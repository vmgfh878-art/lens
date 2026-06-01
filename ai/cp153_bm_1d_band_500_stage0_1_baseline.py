from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from statistics import NormalDist
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 이 CP는 로컬 yfinance snapshot만 사용한다.
os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.evaluation import summarize_forecast_metrics  # noqa: E402
from ai.inference import resolve_checkpoint_ticker_registry  # noqa: E402
from ai.models.common import BandOutput, ForecastOutput  # noqa: E402
from ai.postprocess import apply_band_postprocess  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    FUTURE_COVARIATE_DIM,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    SequenceDataset,
    append_calendar_features,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import build_registry  # noqa: E402
from ai.train import (  # noqa: E402
    MODEL_REGISTRY,
    apply_feature_columns_to_splits,
    autocast_context,
    forward_model,
    make_loader,
    resolve_device,
    resolve_feature_columns,
)  # noqa: E402


TIMEFRAME = "1D"
HORIZON = 5
SEQ_LEN = 60
TARGET_TYPE = "raw_future_return"
PROVIDER = "yfinance"
FEATURE_SET = "price_volatility_volume"

PRICE_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_yfinance_500.parquet"
PRICE_MANIFEST_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_yfinance_500.manifest.json"
INDICATOR_PATH = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1D_500.parquet"
INDICATOR_MANIFEST_PATH = PROJECT_ROOT / "data" / "parquet" / "indicators_yfinance_1D_500.manifest.json"
EODHD_PRICE_PATH = PROJECT_ROOT / "data" / "parquet" / "price_data_eodhd_500.parquet"
BACKFILL_STATE_PATH = PROJECT_ROOT / "data" / "parquet" / "yfinance_500_backfill_state.json"
CP72_METRICS_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cp_archive"
    / "model_band"
    / "cp72_bm_1d_full_band_product_candidate_metrics.json"
)
FEATURE_SET_PLAN_PATH = PROJECT_ROOT / "docs" / "cp63_bm_feature_set_plan.json"
ARCHIVED_FEATURE_SET_PLAN_PATH = PROJECT_ROOT / "docs" / "cp_archive" / "model_band" / "cp63_bm_feature_set_plan.json"

REPORT_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage0_1_baseline_report.md"
METRICS_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage0_1_baseline_metrics.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage0_1_baseline_summary.csv"
LOG_DIR = PROJECT_ROOT / "docs" / "cp153_bm_1d_band_500_stage0_1_baseline_logs"
OVERLAY_DIR = LOG_DIR / "snapshot_overlay"

Q_PAIRS = [
    ("q15_q85", 0.15, 0.85, "stage2_primary"),
    ("q10_q90", 0.10, 0.90, "stage2_wide"),
    ("q05_q95", 0.05, 0.95, "conservative_reference"),
]

BASELINE_METHODS = [
    "constant_width_train_quantile",
    "rolling_historical_quantile_band",
    "rolling_bollinger_return_band",
]

BAND_METRIC_KEYS = [
    "nominal_coverage",
    "empirical_coverage",
    "coverage_abs_error",
    "lower_breach_rate",
    "upper_breach_rate",
    "lower_breach_abs_error",
    "upper_breach_abs_error",
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


@dataclass(frozen=True)
class SplitPayload:
    plan: Any
    dataset: SequenceDataset
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    registry: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _sha16(payload: Any) -> str:
    raw = json.dumps(_clean_json(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return None
    x = x[finite]
    y = y[finite]
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return None
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    corr = np.corrcoef(xr, yr)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else None


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    return prepared


def load_source_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    price_manifest = _read_json(PRICE_MANIFEST_PATH)
    indicator_manifest = _read_json(INDICATOR_MANIFEST_PATH)
    price = _prepare_frame(pd.read_parquet(PRICE_PATH))
    indicators = _prepare_frame(pd.read_parquet(INDICATOR_PATH))
    indicators = indicators[indicators["timeframe"].astype(str).str.upper() == TIMEFRAME].copy()
    return price, indicators, price_manifest, indicator_manifest


def ensure_feature_set_plan_available() -> dict[str, Any]:
    if FEATURE_SET_PLAN_PATH.exists():
        return {
            "status": "present",
            "path": str(FEATURE_SET_PLAN_PATH),
            "source": str(FEATURE_SET_PLAN_PATH),
            "copied_from_archive": False,
        }
    if not ARCHIVED_FEATURE_SET_PLAN_PATH.exists():
        return {
            "status": "missing",
            "path": str(FEATURE_SET_PLAN_PATH),
            "source": str(ARCHIVED_FEATURE_SET_PLAN_PATH),
            "copied_from_archive": False,
        }
    FEATURE_SET_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARCHIVED_FEATURE_SET_PLAN_PATH, FEATURE_SET_PLAN_PATH)
    return {
        "status": "restored_from_archive",
        "path": str(FEATURE_SET_PLAN_PATH),
        "source": str(ARCHIVED_FEATURE_SET_PLAN_PATH),
        "copied_from_archive": True,
    }


def build_split_payload(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    source_data_hash: str,
    tickers: list[str] | None = None,
    ticker_registry: dict[str, Any] | None = None,
    ticker_registry_path: str | None = None,
) -> SplitPayload:
    feature_frame = indicators.copy()
    price_frame = price.copy()
    if tickers is not None:
        ticker_set = {ticker.upper() for ticker in tickers}
        feature_frame = feature_frame[feature_frame["ticker"].isin(ticker_set)].copy()
        price_frame = price_frame[price_frame["ticker"].isin(ticker_set)].copy()

    plan = build_dataset_plan(
        feature_frame,
        timeframe=TIMEFRAME,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        ticker_registry=ticker_registry,
        ticker_registry_path=ticker_registry_path,
        market_data_provider=PROVIDER,
        source_data_hash=source_data_hash,
    )
    active_registry = ticker_registry or build_registry(plan.eligible_tickers, TIMEFRAME)
    eligible = set(plan.eligible_tickers)
    dataset = build_lazy_sequence_dataset(
        feature_df=feature_frame[feature_frame["ticker"].isin(eligible)].copy(),
        price_df=price_frame[price_frame["ticker"].isin(eligible)].copy(),
        timeframe=TIMEFRAME,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        ticker_registry=active_registry,
        include_future_covariate=True,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(
        dataset,
        split_specs=plan.split_specs,
    )
    return SplitPayload(
        plan=plan,
        dataset=dataset,
        train=train_bundle,
        val=val_bundle,
        test=test_bundle,
        registry=active_registry,
    )


def collect_targets(bundle: SequenceDataset) -> np.ndarray:
    targets = np.empty((len(bundle.sample_refs), bundle.horizon), dtype=np.float32)
    refs_by_ticker: dict[str, list[tuple[int, int]]] = {}
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        refs_by_ticker.setdefault(str(ticker), []).append((row_idx, int(end_idx)))

    for ticker, refs in refs_by_ticker.items():
        closes = np.asarray(bundle.ticker_arrays[ticker]["closes"], dtype=np.float64)
        for row_idx, end_idx in refs:
            anchor = float(closes[end_idx])
            future = closes[end_idx + 1 : end_idx + 1 + bundle.horizon]
            targets[row_idx] = ((future / anchor) - 1.0).astype(np.float32)
    return targets


def split_overlap_summary(train: SequenceDataset, val: SequenceDataset, test: SequenceDataset) -> dict[str, Any]:
    def keys(bundle: SequenceDataset) -> set[tuple[str, int]]:
        return {(str(ticker), int(end_idx)) for ticker, end_idx in bundle.sample_refs}

    train_keys = keys(train)
    val_keys = keys(val)
    test_keys = keys(test)
    train_meta = train.metadata.copy()
    val_meta = val.metadata.copy()
    test_meta = test.metadata.copy()
    return {
        "train_val_sample_overlap": len(train_keys & val_keys),
        "train_test_sample_overlap": len(train_keys & test_keys),
        "val_test_sample_overlap": len(val_keys & test_keys),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_date_min": str(train_meta["asof_date"].min()),
        "train_date_max": str(train_meta["asof_date"].max()),
        "val_date_min": str(val_meta["asof_date"].min()),
        "val_date_max": str(val_meta["asof_date"].max()),
        "test_date_min": str(test_meta["asof_date"].min()),
        "test_date_max": str(test_meta["asof_date"].max()),
    }


def feature_quality_summary(indicators: pd.DataFrame, targets_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    enriched = indicators.copy()
    if "has_fundamentals" not in enriched.columns:
        enriched["has_fundamentals"] = False
    for column in ["revenue", "net_income", "equity", "eps", "roe", "debt_ratio"]:
        if column not in enriched.columns:
            enriched[column] = np.nan
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    enriched["has_fundamentals"] = enriched["has_fundamentals"].fillna(False).astype(bool)
    enriched = append_calendar_features(enriched)

    missing_columns = [column for column in MODEL_FEATURE_COLUMNS if column not in enriched.columns]
    nonfinite_counts: dict[str, int] = {}
    total_nonfinite = 0
    if not missing_columns:
        for column in MODEL_FEATURE_COLUMNS:
            values = pd.to_numeric(enriched[column], errors="coerce").to_numpy(dtype=np.float64)
            count = int((~np.isfinite(values)).sum())
            if count:
                nonfinite_counts[column] = count
                total_nonfinite += count

    target_nonfinite_by_split = {
        split: int((~np.isfinite(values)).sum())
        for split, values in targets_by_split.items()
    }
    return {
        "feature_version": FEATURE_CONTRACT_VERSION,
        "model_feature_column_count": len(MODEL_FEATURE_COLUMNS),
        "missing_model_feature_columns": missing_columns,
        "feature_nonfinite_count_after_contract_impute": total_nonfinite,
        "feature_nonfinite_counts_after_contract_impute": nonfinite_counts,
        "target_nonfinite_by_split": target_nonfinite_by_split,
        "target_nonfinite_total": int(sum(target_nonfinite_by_split.values())),
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "intraday_range_ratio_in_model_features": "intraday_range_ratio" in MODEL_FEATURE_COLUMNS,
    }


def duplicate_summary(price: pd.DataFrame, indicators: pd.DataFrame) -> dict[str, Any]:
    return {
        "price_duplicate_ticker_date_rows": int(price.duplicated(["ticker", "date"]).sum()),
        "indicator_duplicate_ticker_timeframe_date_rows": int(
            indicators.duplicated(["ticker", "timeframe", "date"]).sum()
        ),
    }


def ctra_hubb_status(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    plan: Any,
) -> dict[str, Any]:
    state = _read_json(BACKFILL_STATE_PATH) if BACKFILL_STATE_PATH.exists() else {}
    ticker_state = state.get("tickers") or {}
    payload: dict[str, Any] = {}
    for ticker in ["CTRA", "HUBB"]:
        info = dict(ticker_state.get(ticker) or {})
        payload[ticker] = {
            "in_backfill_state": ticker in ticker_state,
            "backfill_state": info,
            "price_rows": int((price["ticker"] == ticker).sum()),
            "indicator_rows": int((indicators["ticker"] == ticker).sum()),
            "eligible": ticker in set(plan.eligible_tickers),
            "excluded_reason": plan.excluded_reasons.get(ticker),
        }
    return payload


def provider_spot_check(price: pd.DataFrame) -> dict[str, Any]:
    if not EODHD_PRICE_PATH.exists():
        return {
            "status": "PROVIDER_CONFOUNDED",
            "reason": "local_eodhd_price_parquet_missing",
            "rows": [],
            "max_diff_pct": None,
            "failed_count": None,
        }
    eodhd = _prepare_frame(pd.read_parquet(EODHD_PRICE_PATH))
    y_index = price.set_index(["ticker", "date"]).sort_index()
    e_index = eodhd.set_index(["ticker", "date"]).sort_index()
    samples = {
        "AAPL": ["2020-08-28", "2020-08-31", "2020-09-01"],
        "TSLA": ["2020-08-28", "2020-08-31", "2022-08-25"],
        "NVDA": ["2021-07-20", "2024-06-10", "2024-06-11"],
        "AMZN": ["2022-06-03", "2022-06-06", "2022-06-07"],
        "GOOGL": ["2022-07-15", "2022-07-18", "2022-07-19"],
    }
    columns = ["open", "high", "low", "close", "adjusted_close"]
    rows: list[dict[str, Any]] = []
    max_diff = 0.0
    failed_count = 0
    missing_count = 0

    for ticker, dates in samples.items():
        for raw_date in dates:
            date = pd.Timestamp(raw_date)
            row: dict[str, Any] = {
                "ticker": ticker,
                "sample_date": raw_date,
                "resolved_date": raw_date,
                "status": "ok",
                "column_diffs_pct": {},
                "max_diff_pct": None,
            }
            key = (ticker, date)
            if key not in y_index.index or key not in e_index.index:
                row["status"] = "missing_in_local_provider"
                missing_count += 1
                rows.append(row)
                continue

            y_row = y_index.loc[key]
            e_row = e_index.loc[key]
            diffs: dict[str, float | None] = {}
            row_max = 0.0
            for column in columns:
                y_value = _safe_float(y_row[column])
                e_value = _safe_float(e_row[column])
                if y_value is None or e_value is None:
                    diffs[column] = None
                    continue
                denominator = max(abs(e_value), 1e-9)
                diff_pct = abs(y_value - e_value) / denominator
                diffs[column] = float(diff_pct)
                row_max = max(row_max, float(diff_pct))
            row["column_diffs_pct"] = diffs
            row["max_diff_pct"] = row_max
            max_diff = max(max_diff, row_max)
            if row_max >= 0.001:
                row["status"] = "diff_over_0p1pct"
                failed_count += 1
            rows.append(row)

    status = "PASS" if failed_count == 0 and missing_count == 0 else "PROVIDER_CONFOUNDED"
    return {
        "status": status,
        "threshold_diff_pct": 0.001,
        "max_diff_pct": max_diff,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "rows": rows,
    }


def build_rolling_cache(dataset: SequenceDataset, quantiles: list[float]) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    for ticker, arrays in dataset.ticker_arrays.items():
        closes = np.asarray(arrays["closes"], dtype=np.float64)
        n_rows = len(closes)
        returns = np.full((dataset.horizon, n_rows), np.nan, dtype=np.float32)
        for horizon_idx in range(dataset.horizon):
            step = horizon_idx + 1
            valid = n_rows - step
            if valid <= 0:
                continue
            returns[horizon_idx, :valid] = (closes[step:] / closes[:-step] - 1.0).astype(np.float32)

        quantile_payload: dict[str, np.ndarray] = {}
        for quantile in quantiles:
            values = np.full_like(returns, np.nan, dtype=np.float32)
            for horizon_idx in range(dataset.horizon):
                step = horizon_idx + 1
                series = pd.Series(returns[horizon_idx].astype(np.float64))
                values[horizon_idx] = (
                    series.shift(step)
                    .rolling(window=252, min_periods=60)
                    .quantile(float(quantile))
                    .to_numpy(dtype=np.float32)
                )
            quantile_payload[f"{quantile:.2f}"] = values

        mean60 = np.full_like(returns, np.nan, dtype=np.float32)
        std60 = np.full_like(returns, np.nan, dtype=np.float32)
        for horizon_idx in range(dataset.horizon):
            step = horizon_idx + 1
            series = pd.Series(returns[horizon_idx].astype(np.float64))
            shifted = series.shift(step)
            mean60[horizon_idx] = shifted.rolling(window=60, min_periods=30).mean().to_numpy(dtype=np.float32)
            std60[horizon_idx] = shifted.rolling(window=60, min_periods=30).std(ddof=0).to_numpy(dtype=np.float32)

        cache[str(ticker)] = {
            "quantiles": quantile_payload,
            "mean60": mean60,
            "std60": std60,
        }
    return cache


def _bundle_end_indices(bundle: SequenceDataset) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row_idx, (ticker, end_idx) in enumerate(bundle.sample_refs):
        grouped.setdefault(str(ticker), []).append((row_idx, int(end_idx)))
    return {
        ticker: (
            np.asarray([row_idx for row_idx, _ in refs], dtype=np.int64),
            np.asarray([end_idx for _, end_idx in refs], dtype=np.int64),
        )
        for ticker, refs in grouped.items()
    }


def rolling_quantile_predictions(
    bundle: SequenceDataset,
    cache: dict[str, Any],
    *,
    q_low: float,
    q_high: float,
    fallback_low: np.ndarray,
    fallback_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    upper = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    lower_key = f"{q_low:.2f}"
    upper_key = f"{q_high:.2f}"
    for ticker, (rows, end_indices) in _bundle_end_indices(bundle).items():
        ticker_cache = cache[ticker]["quantiles"]
        low_values = ticker_cache[lower_key][:, end_indices].T
        high_values = ticker_cache[upper_key][:, end_indices].T
        lower[rows] = np.where(np.isfinite(low_values), low_values, fallback_low.reshape(1, -1))
        upper[rows] = np.where(np.isfinite(high_values), high_values, fallback_high.reshape(1, -1))
    return lower, upper


def bollinger_predictions(
    bundle: SequenceDataset,
    cache: dict[str, Any],
    *,
    q_high: float,
    fallback_mean: np.ndarray,
    fallback_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k_value = float(NormalDist().inv_cdf(q_high))
    lower = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    upper = np.empty((len(bundle), bundle.horizon), dtype=np.float32)
    for ticker, (rows, end_indices) in _bundle_end_indices(bundle).items():
        ticker_cache = cache[ticker]
        mean_values = ticker_cache["mean60"][:, end_indices].T
        std_values = ticker_cache["std60"][:, end_indices].T
        mean_values = np.where(np.isfinite(mean_values), mean_values, fallback_mean.reshape(1, -1))
        std_values = np.where(np.isfinite(std_values), std_values, fallback_std.reshape(1, -1))
        lower[rows] = mean_values - (k_value * std_values)
        upper[rows] = mean_values + (k_value * std_values)
    return lower, upper


def _date_cross_sectional_width_ic(
    *,
    metadata: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    frame = metadata[["asof_date"]].reset_index(drop=True).copy()
    width = np.maximum(upper - lower, 0.0)
    center = (lower + upper) / 2.0
    downside_width = np.maximum(center - lower, 0.0)
    realized_abs = np.abs(actual)
    downside = np.maximum(-actual, 0.0)
    band_values: list[float] = []
    downside_values: list[float] = []
    for asof_date, group in frame.groupby("asof_date", sort=True):
        del asof_date
        indices = group.index.to_numpy(dtype=np.int64)
        band_corr = _spearman_corr(width[indices].reshape(-1), realized_abs[indices].reshape(-1))
        downside_corr = _spearman_corr(downside_width[indices].reshape(-1), downside[indices].reshape(-1))
        if band_corr is not None:
            band_values.append(band_corr)
        if downside_corr is not None:
            downside_values.append(downside_corr)

    return {
        "band_width_ic_date_cs_mean": float(np.mean(band_values)) if band_values else None,
        "band_width_ic_date_cs_std": float(np.std(band_values, ddof=1)) if len(band_values) > 1 else None,
        "band_width_ic_date_cs_count": len(band_values),
        "downside_width_ic_date_cs_mean": float(np.mean(downside_values)) if downside_values else None,
        "downside_width_ic_date_cs_std": float(np.std(downside_values, ddof=1)) if len(downside_values) > 1 else None,
        "downside_width_ic_date_cs_count": len(downside_values),
    }


def summarize_band_prediction(
    *,
    name: str,
    split_name: str,
    q_label: str,
    q_low: float,
    q_high: float,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    metadata: pd.DataFrame,
    squeeze_breakout_threshold: float,
) -> dict[str, Any]:
    lower_t = torch.from_numpy(lower.astype(np.float32))
    upper_t = torch.from_numpy(upper.astype(np.float32))
    actual_t = torch.from_numpy(actual.astype(np.float32))
    line_t = (lower_t + upper_t) / 2.0
    summary = summarize_forecast_metrics(
        metadata=metadata,
        line_predictions=line_t,
        lower_predictions=torch.minimum(lower_t, upper_t),
        upper_predictions=torch.maximum(lower_t, upper_t),
        line_targets=actual_t,
        band_targets=actual_t,
        raw_future_returns=actual_t,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        q_low=q_low,
        q_high=q_high,
        interval_lower_penalty_weight=2.0,
        interval_upper_penalty_weight=1.0,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        include_legacy_overlay_diagnostics=False,
    )
    metrics = {key: _safe_float(summary.get(key)) for key in BAND_METRIC_KEYS if key in summary}
    date_cs = _date_cross_sectional_width_ic(
        metadata=metadata,
        lower=np.minimum(lower, upper),
        upper=np.maximum(lower, upper),
        actual=actual,
    )
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    metrics.update(
        {
            "baseline": name,
            "split": split_name,
            "q_label": q_label,
            "q_low": q_low,
            "q_high": q_high,
            "nominal_coverage": q_high - q_low,
        }
    )
    return metrics


def compute_baselines(payload: SplitPayload) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_targets = collect_targets(payload.train)
    val_targets = collect_targets(payload.val)
    test_targets = collect_targets(payload.test)
    train_abs_p80 = float(np.nanquantile(np.abs(train_targets.reshape(-1)), 0.80))
    quantiles = sorted({q for _, q_low, q_high, _ in Q_PAIRS for q in (q_low, q_high)})
    rolling_cache = build_rolling_cache(payload.dataset, quantiles)
    rows: list[dict[str, Any]] = []

    for q_label, q_low, q_high, q_note in Q_PAIRS:
        global_low = float(np.nanquantile(train_targets.reshape(-1), q_low))
        global_high = float(np.nanquantile(train_targets.reshape(-1), q_high))
        horizon_low = np.nanquantile(train_targets, q_low, axis=0).astype(np.float32)
        horizon_high = np.nanquantile(train_targets, q_high, axis=0).astype(np.float32)
        horizon_mean = np.nanmean(train_targets, axis=0).astype(np.float32)
        horizon_std = np.nanstd(train_targets, axis=0).astype(np.float32)

        for split_name, bundle, actual in (
            ("val", payload.val, val_targets),
            ("test", payload.test, test_targets),
        ):
            constant_lower = np.full_like(actual, global_low, dtype=np.float32)
            constant_upper = np.full_like(actual, global_high, dtype=np.float32)
            rows.append(
                {
                    **summarize_band_prediction(
                        name="constant_width_train_quantile",
                        split_name=split_name,
                        q_label=q_label,
                        q_low=q_low,
                        q_high=q_high,
                        lower=constant_lower,
                        upper=constant_upper,
                        actual=actual,
                        metadata=bundle.metadata,
                        squeeze_breakout_threshold=train_abs_p80,
                    ),
                    "q_note": q_note,
                    "baseline_detail": "global_train_quantile_repeated_all_horizons",
                }
            )

            rolling_lower, rolling_upper = rolling_quantile_predictions(
                bundle,
                rolling_cache,
                q_low=q_low,
                q_high=q_high,
                fallback_low=horizon_low,
                fallback_high=horizon_high,
            )
            rows.append(
                {
                    **summarize_band_prediction(
                        name="rolling_historical_quantile_band",
                        split_name=split_name,
                        q_label=q_label,
                        q_low=q_low,
                        q_high=q_high,
                        lower=rolling_lower,
                        upper=rolling_upper,
                        actual=actual,
                        metadata=bundle.metadata,
                        squeeze_breakout_threshold=train_abs_p80,
                    ),
                    "q_note": q_note,
                    "baseline_detail": "w252_shifted_horizon_specific_quantile_min_periods_60",
                }
            )

            boll_lower, boll_upper = bollinger_predictions(
                bundle,
                rolling_cache,
                q_high=q_high,
                fallback_mean=horizon_mean,
                fallback_std=horizon_std,
            )
            rows.append(
                {
                    **summarize_band_prediction(
                        name="rolling_bollinger_return_band",
                        split_name=split_name,
                        q_label=q_label,
                        q_low=q_low,
                        q_high=q_high,
                        lower=boll_lower,
                        upper=boll_upper,
                        actual=actual,
                        metadata=bundle.metadata,
                        squeeze_breakout_threshold=train_abs_p80,
                    ),
                    "q_note": q_note,
                    "baseline_detail": "w60_shifted_mean_std_gaussian_k_from_q_high_min_periods_30",
                }
            )

    target_checks = {
        "train": int((~np.isfinite(train_targets)).sum()),
        "val": int((~np.isfinite(val_targets)).sum()),
        "test": int((~np.isfinite(test_targets)).sum()),
        "train_abs_return_p80_for_squeeze": train_abs_p80,
    }
    return rows, target_checks


def baseline_sota_and_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    val_rows = [row for row in rows if row.get("split") == "val"]
    low_is_good = [
        "coverage_abs_error",
        "lower_breach_abs_error",
        "upper_breach_abs_error",
        "asymmetric_interval_score",
        "median_band_width",
        "p90_band_width",
        "squeeze_breakout_rate",
    ]
    high_is_good = [
        "band_width_ic",
        "downside_width_ic",
        "width_bucket_realized_vol_ratio",
    ]
    by_q: dict[str, Any] = {}
    for q_label, q_low, q_high, _ in Q_PAIRS:
        subset = [row for row in val_rows if row.get("q_label") == q_label]
        sota: dict[str, Any] = {}
        for metric in low_is_good:
            finite = [row for row in subset if _safe_float(row.get(metric)) is not None]
            if finite:
                best = min(finite, key=lambda row: float(row[metric]))
                sota[metric] = {"value": float(best[metric]), "baseline": best["baseline"], "direction": "lower"}
        for metric in high_is_good:
            finite = [row for row in subset if _safe_float(row.get(metric)) is not None]
            if finite:
                best = max(finite, key=lambda row: float(row[metric]))
                sota[metric] = {"value": float(best[metric]), "baseline": best["baseline"], "direction": "higher"}

        coverage_pass_rows = [
            row
            for row in subset
            if (_safe_float(row.get("coverage_abs_error")) is not None and float(row["coverage_abs_error"]) <= 0.05)
            and _safe_float(row.get("p90_band_width")) is not None
        ]
        p90_base_rows = coverage_pass_rows or [row for row in subset if _safe_float(row.get("p90_band_width")) is not None]
        p90_min = min(float(row["p90_band_width"]) for row in p90_base_rows) if p90_base_rows else None
        p90_overwide = (p90_min * 1.25) if p90_min is not None else None

        by_q[q_label] = {
            "q_low": q_low,
            "q_high": q_high,
            "baseline_sota_validation": sota,
            "hard_gate_lock": {
                "coverage_abs_error_max": min(0.05, float(sota.get("coverage_abs_error", {}).get("value", 0.05)) * 1.10),
                "lower_breach_rate_max": q_low + 0.03,
                "lower_breach_abs_error_max": (
                    float(sota.get("lower_breach_abs_error", {}).get("value", 0.05)) + 0.005
                ),
                "upper_breach_abs_error_max": (
                    float(sota.get("upper_breach_abs_error", {}).get("value", 0.05)) + 0.005
                ),
                "asymmetric_interval_score_max": sota.get("asymmetric_interval_score", {}).get("value"),
                "band_width_ic_min": max(0.15, float(sota.get("band_width_ic", {}).get("value", 0.0))),
                "downside_width_ic_min": max(0.0, float(sota.get("downside_width_ic", {}).get("value", 0.0))),
                "width_bucket_realized_vol_ratio_min": 1.0,
                "squeeze_breakout_rate_max": sota.get("squeeze_breakout_rate", {}).get("value"),
                "p90_band_width_overwide_threshold": p90_overwide,
                "p90_band_width_overwide_basis": (
                    "coverage_pass_baseline_min_p90_x1p25" if coverage_pass_rows else "all_baseline_min_p90_x1p25"
                ),
            },
        }

    return {
        "aggregation_lock": {
            "band_width_ic": "date_cross_sectional_mean",
            "downside_width_ic": "date_cross_sectional_mean",
            "flatten_values_retained_as_diagnostics": True,
        },
        "metric_direction_lock": {
            "lower_is_better": low_is_good,
            "higher_is_better": high_is_good,
            "coverage_interpreted_by_abs_error": True,
        },
        "band_selection_score_lock": {
            "formula": (
                "0.30*rank_pct_inverse(asymmetric_interval_score)"
                "+0.20*rank_pct_inverse(coverage_abs_error)"
                "+0.15*rank_pct_inverse(lower_breach_abs_error)"
                "+0.15*rank_pct(band_width_ic)"
                "+0.10*rank_pct(downside_width_ic)"
                "+0.05*rank_pct_inverse(p90_band_width)"
                "+0.05*rank_pct_inverse(squeeze_breakout_rate)"
            ),
            "tie_breakers": [
                "asymmetric_interval_score",
                "coverage_abs_error",
                "downside_width_ic",
                "model_family_diversity",
            ],
            "line_metrics_used": False,
            "composite_used": False,
        },
        "by_q": by_q,
    }


def _load_checkpoint_for_replay(checkpoint_path: Path) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config") or {})
    raw_role = str(config.get("model_role") or config.get("output_role") or config.get("role") or "legacy").lower()
    if raw_role == "band_model":
        raw_role = "band"
        config["model_role"] = "band"
    if raw_role == "line_model":
        raw_role = "line_v2"
        config["model_role"] = "line_v2"
    model_cls = MODEL_REGISTRY[config["model"]]
    feature_columns = list(config.get("feature_columns") or MODEL_FEATURE_COLUMNS)
    n_features = int(config.get("n_features") or len(feature_columns) or MODEL_N_FEATURES)
    model_kwargs = {
        "n_features": n_features,
        "seq_len": int(config["seq_len"]),
        "horizon": int(config["horizon"]),
        "dropout": float(config.get("dropout", 0.2)),
        "band_mode": config.get("band_mode", "direct"),
        "num_tickers": int(config.get("num_tickers", 0)),
        "ticker_emb_dim": int(config.get("ticker_emb_dim", 32)),
        "output_role": raw_role,
    }
    if config["model"] == "cnn_lstm":
        model_kwargs["use_direction_head"] = bool(config.get("use_direction_head", False))
        model_kwargs["fp32_modules"] = str(config.get("fp32_modules", "none"))
    if config["model"] == "tide":
        use_future_covariate = bool(config.get("use_future_covariate", True))
        model_kwargs["future_cov_dim"] = config.get("future_cov_dim", FUTURE_COVARIATE_DIM) if use_future_covariate else 0
    if config["model"] == "patchtst":
        model_kwargs["use_revin"] = bool(config.get("use_revin", True))
        model_kwargs["ci_aggregate"] = config.get("ci_aggregate", "target")
        model_kwargs["target_channel_idx"] = int(config.get("target_channel_idx", 0))
        model_kwargs["ci_target_fast"] = bool(config.get("ci_target_fast", False))
        model_kwargs["patch_len"] = int(config.get("patch_len", 16))
        model_kwargs["stride"] = int(config.get("patch_stride", config.get("stride", 8)))
        model_kwargs["d_model"] = int(config.get("patchtst_d_model", 128))
        model_kwargs["n_heads"] = int(config.get("patchtst_n_heads", 8))
        model_kwargs["n_layers"] = int(config.get("patchtst_n_layers", 3))
    model = model_cls(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint["config"] = config
    return model, checkpoint


def run_cp72_replay(
    *,
    price: pd.DataFrame,
    indicators: pd.DataFrame,
    source_data_hash: str,
    squeeze_breakout_threshold: float,
) -> dict[str, Any]:
    if not CP72_METRICS_PATH.exists():
        return {"status": "skipped", "reason": "cp72_archive_metrics_missing"}
    cp72 = _read_json(CP72_METRICS_PATH)
    checkpoint_path = PROJECT_ROOT / str(
        cp72.get("storage_verification", {}).get("checkpoint_path", "")
    )
    if not checkpoint_path.exists():
        return {"status": "skipped", "reason": "checkpoint_missing", "checkpoint_path": str(checkpoint_path)}
    try:
        model, checkpoint = _load_checkpoint_for_replay(checkpoint_path)
        config = dict(checkpoint.get("config") or {})
        registry = resolve_checkpoint_ticker_registry(config, TIMEFRAME)
        if registry is None:
            return {"status": "skipped", "reason": "checkpoint_registry_missing"}
        known_tickers = sorted(set(registry.get("mapping") or {}).intersection(set(indicators["ticker"].unique())))
        replay_payload = build_split_payload(
            price=price,
            indicators=indicators,
            source_data_hash=source_data_hash,
            tickers=known_tickers,
            ticker_registry=registry,
            ticker_registry_path=str(config.get("ticker_registry_path") or ""),
        )
        train_norm, val_norm, test_norm, mean, std = normalize_sequence_splits(
            replay_payload.train,
            replay_payload.val,
            replay_payload.test,
        )
        del train_norm
        feature_columns = list(config.get("feature_columns") or resolve_feature_columns(FEATURE_SET))
        _, val_selected, test_selected, _, _ = apply_feature_columns_to_splits(
            replay_payload.train,
            val_norm,
            test_norm,
            mean,
            std,
            feature_columns,
        )
        del _
        device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        metrics_by_split: dict[str, Any] = {}
        for split_name, bundle in (("val", val_selected), ("test", test_selected)):
            metrics_by_split[split_name] = evaluate_model_bundle_for_band(
                model=model,
                bundle=bundle,
                device=device,
                config=config,
                squeeze_breakout_threshold=squeeze_breakout_threshold,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "status": "completed",
            "checkpoint_path": str(checkpoint_path),
            "source_experiment_id": cp72.get("final_product_candidate", {}).get("run_id"),
            "replay_ticker_count": len(known_tickers),
            "yfinance_eligible_ticker_count": len(replay_payload.plan.eligible_tickers),
            "split_rows": {
                "val": len(val_selected),
                "test": len(test_selected),
            },
            "feature_columns": feature_columns,
            "metrics_by_split": metrics_by_split,
            "cp72_original_band_metrics": cp72.get("band_metrics", {}),
            "provider_note": "forward_only_replay_on_yfinance_500_overlap_tickers_no_save",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "reason": type(exc).__name__,
            "message": str(exc),
        }


def evaluate_model_bundle_for_band(
    *,
    model: Any,
    bundle: SequenceDataset,
    device: torch.device,
    config: dict[str, Any],
    squeeze_breakout_threshold: float,
) -> dict[str, Any]:
    loader = make_loader(bundle, batch_size=512, shuffle=False, device=device, num_workers=0)
    lower_predictions: list[torch.Tensor] = []
    upper_predictions: list[torch.Tensor] = []
    raw_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for features, line_target, band_target, raw_future_returns, ticker_id, future_covariates in loader:
            del line_target, band_target
            features = features.to(device, non_blocking=True)
            ticker_id = ticker_id.to(device, non_blocking=True)
            future_covariates = future_covariates.to(device, non_blocking=True)
            with autocast_context(device, str(config.get("amp_dtype", "bf16"))):
                output = forward_model(model, features, ticker_id, future_covariates)
            if isinstance(output, ForecastOutput):
                _, lower, upper = apply_band_postprocess(
                    output.line.detach().cpu(),
                    output.lower_band.detach().cpu(),
                    output.upper_band.detach().cpu(),
                )
            elif isinstance(output, BandOutput):
                lower = torch.minimum(output.lower_band.detach().cpu(), output.upper_band.detach().cpu())
                upper = torch.maximum(output.lower_band.detach().cpu(), output.upper_band.detach().cpu())
            else:
                raise TypeError(f"band replay에서 지원하지 않는 출력입니다: {type(output).__name__}")
            lower_predictions.append(lower)
            upper_predictions.append(upper)
            raw_targets.append(raw_future_returns.detach().cpu())

    lower_t = torch.cat(lower_predictions, dim=0)
    upper_t = torch.cat(upper_predictions, dim=0)
    actual_t = torch.cat(raw_targets, dim=0)
    line_t = (lower_t + upper_t) / 2.0
    summary = summarize_forecast_metrics(
        metadata=bundle.metadata,
        line_predictions=line_t,
        lower_predictions=lower_t,
        upper_predictions=upper_t,
        line_targets=actual_t,
        band_targets=actual_t,
        raw_future_returns=actual_t,
        line_target_type=TARGET_TYPE,
        band_target_type=TARGET_TYPE,
        q_low=float(config.get("q_low", 0.15)),
        q_high=float(config.get("q_high", 0.85)),
        interval_lower_penalty_weight=2.0,
        interval_upper_penalty_weight=1.0,
        squeeze_breakout_threshold=squeeze_breakout_threshold,
        include_legacy_overlay_diagnostics=False,
    )
    lower_np = lower_t.numpy()
    upper_np = upper_t.numpy()
    actual_np = actual_t.numpy()
    date_cs = _date_cross_sectional_width_ic(
        metadata=bundle.metadata,
        lower=lower_np,
        upper=upper_np,
        actual=actual_np,
    )
    metrics = {key: _safe_float(summary.get(key)) for key in BAND_METRIC_KEYS if key in summary}
    metrics["band_width_ic_flatten"] = metrics.get("band_width_ic")
    metrics["downside_width_ic_flatten"] = metrics.get("downside_width_ic")
    metrics["band_width_ic"] = date_cs["band_width_ic_date_cs_mean"]
    metrics["downside_width_ic"] = date_cs["downside_width_ic_date_cs_mean"]
    metrics.update(date_cs)
    return metrics


def prepare_snapshot_overlay() -> dict[str, Any]:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    links = [
        (PRICE_PATH, OVERLAY_DIR / "price_data_yfinance_1D.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data_yfinance_1D.manifest.json"),
        (PRICE_PATH, OVERLAY_DIR / "price_data_yfinance.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data_yfinance.manifest.json"),
        (PRICE_PATH, OVERLAY_DIR / "price_data.parquet"),
        (PRICE_MANIFEST_PATH, OVERLAY_DIR / "price_data.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators_yfinance_1D.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators_yfinance_1D.manifest.json"),
        (INDICATOR_PATH, OVERLAY_DIR / "indicators.parquet"),
        (INDICATOR_MANIFEST_PATH, OVERLAY_DIR / "indicators.manifest.json"),
    ]
    results: list[dict[str, Any]] = []
    for source, target in links:
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
            mode = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            mode = "copy"
        results.append({"source": str(source), "target": str(target), "mode": mode})
    return {
        "overlay_dir": str(OVERLAY_DIR),
        "entries": results,
    }


def run_timing_smoke() -> dict[str, Any]:
    overlay = prepare_snapshot_overlay()
    stdout_path = LOG_DIR / "timing_smoke_cnn_s60_q15_direct.stdout.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        "cnn_lstm",
        "--model-role",
        "band",
        "--timeframe",
        TIMEFRAME,
        "--horizon",
        str(HORIZON),
        "--seq-len",
        str(SEQ_LEN),
        "--feature-set",
        FEATURE_SET,
        "--line-target-type",
        TARGET_TYPE,
        "--band-target-type",
        TARGET_TYPE,
        "--q-low",
        "0.15",
        "--q-high",
        "0.85",
        "--lambda-band",
        "2.0",
        "--band-mode",
        "direct",
        "--checkpoint-selection",
        "band_gate",
        "--fp32-modules",
        "lstm,heads",
        "--epochs",
        "1",
        "--batch-size",
        "256",
        "--limit-tickers",
        "100",
        "--device",
        device,
        "--amp-dtype",
        "bf16",
        "--no-compile",
        "--no-wandb",
        "--num-workers",
        "0",
        "--local-log",
        "--local-log-dir",
        str(LOG_DIR / "ai_train_local_logs"),
        "--explicit-cuda-cleanup",
    ]
    env = os.environ.copy()
    env.update(
        {
            "MARKET_DATA_PROVIDER": "yfinance",
            "LENS_USE_LOCAL_SNAPSHOTS": "1",
            "LENS_REQUIRE_LOCAL_SNAPSHOTS": "1",
            "LENS_LOCAL_SNAPSHOT_DIR": str(OVERLAY_DIR),
            "WANDB_MODE": "disabled",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    start = time.perf_counter()
    timed_out = False
    output_lines: list[str] = []
    timeout_seconds = 1800
    with stdout_path.open("w", encoding="utf-8", newline="") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
                output_lines.append(line.rstrip("\n"))
            if proc.poll() is not None:
                break
            if time.perf_counter() - start > timeout_seconds:
                timed_out = True
                proc.kill()
                break
        for line in proc.stdout:
            print(line, end="")
            log_handle.write(line)
            output_lines.append(line.rstrip("\n"))
        exit_code = proc.wait()
    elapsed = time.perf_counter() - start
    result_payload = _parse_last_json_line(output_lines)
    vram_peak = _extract_number_from_lines(output_lines, r"vram_peak_allocated_mb[=: ]+([0-9.]+)")
    epoch_seconds = _extract_number_from_lines(output_lines, r"epoch(?:_seconds)?[=: ]+([0-9.]+)")
    if result_payload:
        epoch_seconds = _safe_float(result_payload.get("epoch_1_seconds_from_stdout")) or epoch_seconds
        execution = result_payload.get("execution") or {}
        vram_peak = _safe_float(execution.get("vram_peak_allocated_mb")) or vram_peak
    return {
        "status": "TIMEOUT" if timed_out else ("PASS" if exit_code == 0 else "FAIL"),
        "exit_code": int(exit_code),
        "elapsed_seconds": round(elapsed, 3),
        "timeout_seconds": timeout_seconds,
        "command": cmd,
        "stdout_path": str(stdout_path),
        "overlay": overlay,
        "parsed_result": result_payload,
        "epoch_seconds": epoch_seconds,
        "vram_peak_allocated_mb": vram_peak,
        "stage2_stage3_estimate": _estimate_stage_time(epoch_seconds, elapsed, vram_peak),
    }


def _parse_last_json_line(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_number_from_lines(lines: list[str], pattern: str) -> float | None:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    for line in reversed(lines):
        match = compiled.search(line)
        if not match:
            continue
        value = _safe_float(match.group(1))
        if value is not None:
            return value
    return None


def _estimate_stage_time(epoch_seconds: float | None, elapsed_seconds: float, vram_peak: float | None) -> dict[str, Any]:
    base_epoch = epoch_seconds or elapsed_seconds
    full_epoch_low = base_epoch * 5.0 * 1.10
    full_epoch_high = base_epoch * 5.0 * 1.50
    return {
        "basis": "100_ticker_1epoch_smoke_scaled_to_500_tickers",
        "base_epoch_seconds": base_epoch,
        "full_500_one_epoch_seconds_low": round(full_epoch_low, 1),
        "full_500_one_epoch_seconds_high": round(full_epoch_high, 1),
        "stage2_100ticker_8_candidates_1epoch_minutes": round((base_epoch * 8) / 60.0, 1),
        "stage2_100ticker_8_candidates_3epoch_minutes": round((base_epoch * 8 * 3) / 60.0, 1),
        "stage3_500ticker_3_candidates_3epoch_minutes_low": round((full_epoch_low * 3 * 3) / 60.0, 1),
        "stage3_500ticker_3_candidates_3epoch_minutes_high": round((full_epoch_high * 3 * 3) / 60.0, 1),
        "estimated_vram_peak_mb": vram_peak,
        "estimated_vram_note": "500 ticker는 배치 크기가 같으면 VRAM보다 wall time 증가가 주효하다고 본다.",
    }


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "q_label",
        "q_low",
        "q_high",
        "baseline",
        "empirical_coverage",
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "lower_breach_abs_error",
        "upper_breach_abs_error",
        "avg_band_width",
        "median_band_width",
        "p90_band_width",
        "asymmetric_interval_score",
        "interval_lower_penalty",
        "interval_upper_penalty",
        "band_width_ic",
        "downside_width_ic",
        "band_width_ic_flatten",
        "downside_width_ic_flatten",
        "width_bucket_realized_vol_ratio",
        "width_bucket_downside_rate_ratio",
        "squeeze_breakout_rate",
        "baseline_detail",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _metric_table(rows: list[dict[str, Any]], *, split: str, q_label: str) -> str:
    subset = [row for row in rows if row.get("split") == split and row.get("q_label") == q_label]
    header = (
        "| baseline | coverage_abs_error | lower_breach | upper_breach | "
        "interval_score | p90_width | band_width_ic | downside_width_ic | squeeze |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in subset:
        lines.append(
            "| {baseline} | {coverage_abs_error} | {lower_breach_rate} | {upper_breach_rate} | "
            "{asymmetric_interval_score} | {p90_band_width} | {band_width_ic} | {downside_width_ic} | "
            "{squeeze_breakout_rate} |".format(
                baseline=row.get("baseline"),
                coverage_abs_error=_fmt(row.get("coverage_abs_error")),
                lower_breach_rate=_fmt(row.get("lower_breach_rate")),
                upper_breach_rate=_fmt(row.get("upper_breach_rate")),
                asymmetric_interval_score=_fmt(row.get("asymmetric_interval_score")),
                p90_band_width=_fmt(row.get("p90_band_width")),
                band_width_ic=_fmt(row.get("band_width_ic")),
                downside_width_ic=_fmt(row.get("downside_width_ic")),
                squeeze_breakout_rate=_fmt(row.get("squeeze_breakout_rate")),
            )
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def write_report(metrics: dict[str, Any]) -> None:
    baseline_rows = metrics["stage1"]["baseline_rows"]
    status = metrics["final_status"]
    stage0 = metrics["stage0"]
    gates = metrics["stage1"]["gates"]
    timing = metrics["stage0"]["timing_smoke"]
    provider_status = stage0["provider_spot_check"]["status"]
    cp72_status = stage0["cp72_forward_replay"]["status"]
    plan_candidates = [
        "cnn_s60_q15_direct",
        "cnn_s60_q10_direct",
        "cnn_s60_q15_lower_guard",
        "tcn_s60_q15_direct",
        "tcn_s120_q15_direct",
        "tcn_s60_q10_direct",
        "tide_s104_q15_param",
        "patch_s252_q15_direct",
    ]
    lines = [
        "# CP153-BM 1D Band 500 Stage 0/1 Baseline Report",
        "",
        f"- 판정: **{status}**",
        f"- 기준 문서: `docs/cp153_bm_1d_band_500_plan.md` REV2",
        f"- 생성 시각 UTC: `{metrics['created_at_utc']}`",
        f"- role 해석: `band_model` 전용, line/composite 지표 미사용",
        f"- target: `{TARGET_TYPE}`, timeframe `{TIMEFRAME}`, horizon `{HORIZON}`",
        "",
        "## Stage 0 계약",
        "",
        f"- yfinance price parquet: `{PRICE_PATH}`",
        f"- yfinance indicator parquet: `{INDICATOR_PATH}`",
        f"- price source_data_hash: `{stage0['manifests']['price'].get('source_data_hash')}`",
        f"- indicator source_data_hash: `{stage0['manifests']['indicator'].get('source_data_hash')}`",
        f"- CP153 계약 hash: `{stage0['contract_hash']}`",
        f"- feature_version: `{stage0['feature_quality']['feature_version']}`",
        f"- feature_set `{FEATURE_SET}` column 수: `{stage0['feature_set']['column_count']}`",
        f"- feature_set plan 상태: `{stage0['feature_set']['plan_file']['status']}`",
        f"- eligible ticker 수: `{stage0['dataset_plan']['eligible_ticker_count']}`",
        f"- 제외 ticker 수: `{stage0['dataset_plan']['excluded_ticker_count']}`",
        f"- feature NaN/Inf after contract impute: `{stage0['feature_quality']['feature_nonfinite_count_after_contract_impute']}`",
        f"- target NaN/Inf total: `{stage0['feature_quality']['target_nonfinite_total']}`",
        f"- duplicate rows: price `{stage0['duplicates']['price_duplicate_ticker_date_rows']}`, indicator `{stage0['duplicates']['indicator_duplicate_ticker_timeframe_date_rows']}`",
        f"- split sample overlap: train/val `{stage0['split_overlap']['train_val_sample_overlap']}`, train/test `{stage0['split_overlap']['train_test_sample_overlap']}`, val/test `{stage0['split_overlap']['val_test_sample_overlap']}`",
        f"- CTRA/HUBB 상태: `{json.dumps(stage0['ctra_hubb_status'], ensure_ascii=False)}`",
        "",
        "## Provider Spot-Check",
        "",
        f"- 상태: **{provider_status}**",
        f"- 최대 diff pct: `{_fmt(stage0['provider_spot_check'].get('max_diff_pct'))}`",
        f"- 실패 rows: `{stage0['provider_spot_check'].get('failed_count')}`, missing rows: `{stage0['provider_spot_check'].get('missing_count')}`",
        "",
        "diff pct가 0.1% 이상이면 `PROVIDER_CONFOUNDED`로 기록하며, 이 경우 CP72 대비 개선 표현은 금지한다.",
        "",
        "## CP72 Forward-Only Replay",
        "",
        f"- 상태: **{cp72_status}**",
        f"- 설명: `{stage0['cp72_forward_replay'].get('reason') or stage0['cp72_forward_replay'].get('provider_note')}`",
        "",
        "## Timing Smoke",
        "",
        f"- 상태: **{timing['status']}**, exit code `{timing['exit_code']}`, wall `{_fmt(timing['elapsed_seconds'])}`초",
        f"- epoch_seconds: `{_fmt(timing.get('epoch_seconds'))}`",
        f"- vram_peak_allocated_mb: `{_fmt(timing.get('vram_peak_allocated_mb'))}`",
        f"- stdout log: `{timing['stdout_path']}`",
        "",
        "## Stage 1 Baseline",
        "",
        "band_width_ic/downside_width_ic 집계 방식은 `date_cross_sectional_mean`으로 잠근다. flatten 값은 진단용으로만 보존한다.",
        "",
        "### q15/q85 validation",
        "",
        _metric_table(baseline_rows, split="val", q_label="q15_q85"),
        "",
        "### q10/q90 validation",
        "",
        _metric_table(baseline_rows, split="val", q_label="q10_q90"),
        "",
        "### q05/q95 validation",
        "",
        _metric_table(baseline_rows, split="val", q_label="q05_q95"),
        "",
        "## Gate Lock",
        "",
        "Stage 2 이후 후보는 validation 기준 hard gate를 먼저 통과하고, 통과 후보 안에서 band_selection_score로 정렬한다.",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        "## Stage 2 추천 후보와 예상 시간",
        "",
        "제품 후보 판단은 하지 않는다. Stage 2 model zoo smoke는 REV2 후보군을 그대로 사용한다.",
        "",
        *[f"- {name}" for name in plan_candidates],
        "",
        f"- 100티커 8후보 1epoch 예상: `{timing['stage2_stage3_estimate']['stage2_100ticker_8_candidates_1epoch_minutes']}`분",
        f"- 100티커 8후보 3epoch 예상: `{timing['stage2_stage3_estimate']['stage2_100ticker_8_candidates_3epoch_minutes']}`분",
        f"- 500티커 3후보 3epoch 예상: `{timing['stage2_stage3_estimate']['stage3_500ticker_3_candidates_3epoch_minutes_low']}`~`{timing['stage2_stage3_estimate']['stage3_500ticker_3_candidates_3epoch_minutes_high']}`분",
        "",
        "## 금지사항 준수",
        "",
        "- product save-run 없음",
        "- DB write 없음",
        "- inference 저장 없음",
        "- live fetch 없음",
        "- EODHD fallback 없음, local EODHD parquet spot-check만 수행",
        "- composite/overlay 지표 미사용",
        "- 제품 후보 판단 없음",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics(metrics: dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(_clean_json(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def determine_status(stage0: dict[str, Any]) -> str:
    contract_fail_reasons: list[str] = []
    if not PRICE_PATH.exists() or not INDICATOR_PATH.exists():
        contract_fail_reasons.append("missing_yfinance_500_parquet")
    if stage0["feature_quality"]["feature_nonfinite_count_after_contract_impute"] != 0:
        contract_fail_reasons.append("feature_nonfinite")
    if stage0["feature_quality"]["target_nonfinite_total"] != 0:
        contract_fail_reasons.append("target_nonfinite")
    if any(
        stage0["split_overlap"][key] != 0
        for key in ["train_val_sample_overlap", "train_test_sample_overlap", "val_test_sample_overlap"]
    ):
        contract_fail_reasons.append("split_overlap")
    if stage0["dataset_plan"]["eligible_ticker_count"] <= 0:
        contract_fail_reasons.append("no_eligible_ticker")
    if stage0["timing_smoke"]["status"] not in {"PASS"}:
        contract_fail_reasons.append("timing_smoke_failed")
    if contract_fail_reasons:
        stage0["contract_fail_reasons"] = contract_fail_reasons
        return "FAIL_CONTRACT"
    if stage0["provider_spot_check"]["status"] != "PASS":
        return "WARN_PROVIDER_CONFOUNDED"
    return "PASS_BASELINE_READY"


def main() -> None:
    parser = argparse.ArgumentParser(description="CP153 1D band 500 Stage 0/1 baseline 감사")
    parser.add_argument("--skip-timing-smoke", action="store_true")
    parser.add_argument("--skip-cp72-replay", action="store_true")
    parser.add_argument("--timing-only-update", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.timing_only_update:
        metrics = _read_json(METRICS_PATH)
        metrics["stage0"]["timing_smoke"] = run_timing_smoke()
        metrics["final_status"] = determine_status(metrics["stage0"])
        metrics["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_summary_csv(metrics["stage1"]["baseline_rows"])
        write_metrics(metrics)
        write_report(metrics)
        print(
            json.dumps(
                {
                    "status": metrics["final_status"],
                    "report": str(REPORT_PATH),
                    "metrics": str(METRICS_PATH),
                    "summary_csv": str(SUMMARY_CSV_PATH),
                    "timing_smoke_status": metrics["stage0"]["timing_smoke"]["status"],
                    "provider_spot_check": metrics["stage0"]["provider_spot_check"]["status"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return

    feature_set_plan_status = ensure_feature_set_plan_available()
    price, indicators, price_manifest, indicator_manifest = load_source_frames()
    contract_hash = _sha16(
        {
            "price_manifest": price_manifest,
            "indicator_manifest": indicator_manifest,
            "timeframe": TIMEFRAME,
            "horizon": HORIZON,
            "target": TARGET_TYPE,
            "feature_version": FEATURE_CONTRACT_VERSION,
        }
    )
    source_data_hash = str(indicator_manifest.get("source_data_hash") or price_manifest.get("source_data_hash") or contract_hash)
    payload = build_split_payload(price=price, indicators=indicators, source_data_hash=source_data_hash)
    train_targets = collect_targets(payload.train)
    val_targets = collect_targets(payload.val)
    test_targets = collect_targets(payload.test)
    target_checks = {"train": train_targets, "val": val_targets, "test": test_targets}
    feature_columns = resolve_feature_columns(FEATURE_SET)
    duplicates = duplicate_summary(price, indicators)
    split_overlap = split_overlap_summary(payload.train, payload.val, payload.test)
    feature_quality = feature_quality_summary(indicators, target_checks)
    ctra_hubb = ctra_hubb_status(price=price, indicators=indicators, plan=payload.plan)
    provider_check = provider_spot_check(price)

    baseline_rows, baseline_target_checks = compute_baselines(payload)
    gates = baseline_sota_and_gates(baseline_rows)
    squeeze_threshold = float(baseline_target_checks["train_abs_return_p80_for_squeeze"])
    cp72_replay = (
        {"status": "skipped", "reason": "user_requested_skip"}
        if args.skip_cp72_replay
        else run_cp72_replay(
            price=price,
            indicators=indicators,
            source_data_hash=source_data_hash,
            squeeze_breakout_threshold=squeeze_threshold,
        )
    )
    timing_smoke = (
        {
            "status": "SKIPPED",
            "exit_code": None,
            "elapsed_seconds": None,
            "stdout_path": None,
            "stage2_stage3_estimate": _estimate_stage_time(None, 0.0, None),
        }
        if args.skip_timing_smoke
        else run_timing_smoke()
    )

    stage0 = {
        "manifests": {
            "price": price_manifest,
            "indicator": indicator_manifest,
        },
        "contract_hash": contract_hash,
        "paths": {
            "price": str(PRICE_PATH),
            "price_manifest": str(PRICE_MANIFEST_PATH),
            "indicator": str(INDICATOR_PATH),
            "indicator_manifest": str(INDICATOR_MANIFEST_PATH),
        },
        "feature_set": {
            "name": FEATURE_SET,
            "plan_file": feature_set_plan_status,
            "column_count": len(feature_columns),
            "columns": feature_columns,
        },
        "dataset_plan": {
            "input_ticker_count": int(payload.plan.input_ticker_count),
            "eligible_ticker_count": len(payload.plan.eligible_tickers),
            "excluded_ticker_count": len(payload.plan.excluded_reasons),
            "excluded_reasons": payload.plan.excluded_reasons,
            "date_min": payload.plan.date_min,
            "date_max": payload.plan.date_max,
            "seq_len": payload.plan.seq_len,
            "horizon": payload.plan.horizon,
            "h_max": payload.plan.h_max,
            "source_data_hash": payload.plan.source_data_hash,
            "ticker_registry_path": payload.plan.ticker_registry_path,
            "num_tickers": payload.plan.num_tickers,
        },
        "split_overlap": split_overlap,
        "duplicates": duplicates,
        "feature_quality": feature_quality,
        "target_checks_from_baseline": baseline_target_checks,
        "ctra_hubb_status": ctra_hubb,
        "provider_spot_check": provider_check,
        "cp72_forward_replay": cp72_replay,
        "timing_smoke": timing_smoke,
        "model_role_interpretation": "band_model_only",
        "line_metrics_used_for_decision": False,
        "composite_used": False,
    }
    final_status = determine_status(stage0)
    metrics = {
        "cp": "CP153-BM",
        "stage": "Stage 0/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_status": final_status,
        "stage0": stage0,
        "stage1": {
            "baseline_methods": BASELINE_METHODS,
            "q_pairs": [
                {"label": label, "q_low": q_low, "q_high": q_high, "note": note}
                for label, q_low, q_high, note in Q_PAIRS
            ],
            "baseline_rows": baseline_rows,
            "gates": gates,
        },
        "outputs": {
            "report": str(REPORT_PATH),
            "metrics": str(METRICS_PATH),
            "summary_csv": str(SUMMARY_CSV_PATH),
            "logs": str(LOG_DIR),
        },
    }
    write_summary_csv(baseline_rows)
    write_metrics(metrics)
    write_report(metrics)
    print(
        json.dumps(
            {
                "status": final_status,
                "report": str(REPORT_PATH),
                "metrics": str(METRICS_PATH),
                "summary_csv": str(SUMMARY_CSV_PATH),
                "timing_smoke_status": timing_smoke["status"],
                "provider_spot_check": provider_check["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
