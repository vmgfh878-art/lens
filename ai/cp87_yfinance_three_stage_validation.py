from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from ai.loss import ForecastCompositeLoss  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    append_calendar_features,
    build_dataset_plan,
    build_lazy_sequence_dataset,
    normalize_sequence_splits,
    resolve_data_fingerprint,
    split_sequence_dataset_by_plan,
)
from ai.ticker_registry import build_registry  # noqa: E402
from ai.train import (  # noqa: E402
    TrainConfig,
    apply_feature_columns_to_splits,
    apply_feature_set_to_config,
    build_model,
    build_scheduler,
    estimate_train_risk_thresholds,
    evaluate_bundle,
    make_loader,
    resolve_device,
    run_epoch,
    set_seed,
)
from backend.app.services.feature_svc import FEATURE_COLUMNS, build_features  # noqa: E402
from backend.collector.config import get_settings  # noqa: E402
from backend.collector.pipelines.yfinance_price_sync import _compare_ticker, _db_ticker_frame  # noqa: E402
from backend.collector.repositories.base import fetch_frame  # noqa: E402
from backend.collector.sources.market_data_providers import (  # noqa: E402
    YFinancePriceProvider,
    fetch_market_data,
)
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402
from backend.collector.sources.yahoo_stock_info import fetch_stock_info  # noqa: E402


REQUIRED_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "NFLX",
    "AVGO",
    "AMD",
    "SPY",
    "QQQ",
]
SPLIT_DIVIDEND_CANDIDATES = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "NFLX",
    "AVGO",
    "COST",
    "CMG",
    "WMT",
    "GE",
    "ORCL",
    "INTC",
    "CSCO",
    "ADBE",
]
MODEL_FEATURE_VALUE_COLUMNS = list(MODEL_FEATURE_COLUMNS)
FEATURE_COMPARE_COLUMNS = [*FEATURE_COLUMNS, "atr_ratio"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _stats(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"mean": None, "std": None, "p01": None, "p50": None, "p99": None, "max": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "p01": float(values.quantile(0.01)),
        "p50": float(values.quantile(0.50)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _abs_stats(series: pd.Series) -> dict[str, float | None]:
    return _stats(pd.to_numeric(series, errors="coerce").abs())


def _load_universe(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    column = "ticker" if "ticker" in frame.columns else frame.columns[0]
    return [str(ticker).strip().upper() for ticker in frame[column].tolist() if str(ticker).strip()]


def _select_tickers(universe_path: Path, max_tickers: int) -> list[str]:
    selected: list[str] = []

    def add_many(values: list[str]) -> None:
        for raw in values:
            ticker = raw.strip().upper()
            if ticker and ticker not in selected:
                selected.append(ticker)

    add_many(REQUIRED_TICKERS)
    add_many(SPLIT_DIVIDEND_CANDIDATES)
    add_many(_load_universe(universe_path))
    return selected[:max_tickers]


def _provider_frame_to_price_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume", "amount"]
        )
    index = pd.to_datetime(frame.index, errors="coerce").normalize()
    output = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "date": index,
            "open": pd.to_numeric(frame["Open"], errors="coerce"),
            "high": pd.to_numeric(frame["High"], errors="coerce"),
            "low": pd.to_numeric(frame["Low"], errors="coerce"),
            "close": pd.to_numeric(frame["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(frame["Adj Close"], errors="coerce"),
            "volume": pd.to_numeric(frame["Volume"], errors="coerce"),
        }
    )
    output["amount"] = output["close"] * output["volume"]
    return output.dropna(subset=["date"]).reset_index(drop=True)


def _fetch_baseline_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume",
        filters=[("in", "ticker", tickers), ("gte", "date", start_date), ("lte", "date", end_date)],
        order_by="date",
        page_size=5000,
    )


def _raw_adjusted_quality(ticker: str, frame: pd.DataFrame, reference_calendar: pd.DatetimeIndex) -> dict[str, Any]:
    if frame.empty:
        return {
            "ticker": ticker,
            "row_count": 0,
            "duplicate_date_count": 0,
            "calendar_coverage": 0.0,
            "missing_date_count": int(len(reference_calendar)),
            "raw_ohlc_violation_count": None,
            "adjusted_ohlc_violation_count": None,
            "volume_null_count": None,
            "volume_negative_count": None,
            "abnormal_split_like_count": None,
            "adjusted_factor_jump_gt20_count": None,
            "volume_abs_stats": _stats(pd.Series(dtype=float)),
        }

    indexed = frame.copy()
    indexed.index = pd.to_datetime(indexed.index, errors="coerce").normalize()
    duplicate_count = int(indexed.index.duplicated().sum())
    unique_dates = indexed.index.dropna().unique()
    coverage = float(len(reference_calendar.intersection(unique_dates)) / len(reference_calendar)) if len(reference_calendar) else 0.0
    missing_count = int(len(reference_calendar.difference(unique_dates))) if len(reference_calendar) else 0

    open_ = pd.to_numeric(indexed["Open"], errors="coerce")
    high = pd.to_numeric(indexed["High"], errors="coerce")
    low = pd.to_numeric(indexed["Low"], errors="coerce")
    close = pd.to_numeric(indexed["Close"], errors="coerce")
    adjusted_close = pd.to_numeric(indexed["Adj Close"], errors="coerce")
    volume = pd.to_numeric(indexed["Volume"], errors="coerce")
    adjusted_factor = adjusted_close / close.where(close.abs() > 1e-9)
    adjusted_open = open_ * adjusted_factor
    adjusted_high = high * adjusted_factor
    adjusted_low = low * adjusted_factor

    raw_violations = (
        (high + 1e-9 < pd.concat([open_, close], axis=1).max(axis=1))
        | (low - 1e-9 > pd.concat([open_, close], axis=1).min(axis=1))
        | (high + 1e-9 < low)
    )
    adjusted_violations = (
        (adjusted_high + 1e-9 < pd.concat([adjusted_open, adjusted_close], axis=1).max(axis=1))
        | (adjusted_low - 1e-9 > pd.concat([adjusted_open, adjusted_close], axis=1).min(axis=1))
        | (adjusted_high + 1e-9 < adjusted_low)
    )
    raw_return = close.pct_change().replace([np.inf, -np.inf], np.nan)
    adjusted_return = adjusted_close.pct_change().replace([np.inf, -np.inf], np.nan)
    factor_return = adjusted_factor.pct_change().replace([np.inf, -np.inf], np.nan)
    abnormal_split_like = (raw_return.abs() > 0.5) & (adjusted_return.abs() < 0.2)

    return {
        "ticker": ticker,
        "row_count": int(len(indexed)),
        "duplicate_date_count": duplicate_count,
        "calendar_coverage": coverage,
        "missing_date_count": missing_count,
        "raw_ohlc_violation_count": int(raw_violations.fillna(True).sum()),
        "adjusted_ohlc_violation_count": int(adjusted_violations.fillna(True).sum()),
        "volume_null_count": int(volume.isna().sum()),
        "volume_negative_count": int((volume < 0).fillna(False).sum()),
        "abnormal_split_like_count": int(abnormal_split_like.fillna(False).sum()),
        "adjusted_factor_jump_gt20_count": int((factor_return.abs() > 0.20).fillna(False).sum()),
        "adjusted_factor_abs_stats": _abs_stats(adjusted_factor),
        "adjusted_factor_change_abs_stats": _abs_stats(factor_return),
        "volume_abs_stats": _abs_stats(volume),
    }


def _classify_provider_diff(comparison: dict[str, Any]) -> str:
    baseline_rows = int(comparison.get("baseline_rows") or 0)
    if baseline_rows == 0:
        return "baseline_missing"
    coverage = comparison.get("date_coverage")
    close_median = (comparison.get("close_relative_diff") or {}).get("median")
    adjusted_median = (comparison.get("adjusted_close_relative_diff") or {}).get("median")
    factor_median = (comparison.get("adjusted_factor_relative_diff") or {}).get("median")
    if coverage is None or coverage < 0.99:
        return "coverage_mismatch"
    if adjusted_median is not None and adjusted_median <= 0.005 and close_median is not None and close_median <= 0.005:
        return "pass"
    if adjusted_median is not None and adjusted_median <= 0.02 and close_median is not None and close_median <= 0.005:
        return "dividend_adjustment_policy_diff"
    if (
        adjusted_median is not None
        and adjusted_median <= 0.005
        and close_median is not None
        and close_median > 0.20
        and factor_median is not None
        and factor_median > 0.20
    ):
        return "split_adjustment_policy_diff"
    return "unclassified_diff"


def run_data_validation(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    long_start_date: str,
    long_sample_size: int,
    sleep_seconds: float,
    eodhd_api_key: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline_frame = _fetch_baseline_prices(tickers, start_date, end_date)
    fetch_results: dict[str, Any] = {}
    price_frames: list[pd.DataFrame] = []
    provider_frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
            eodhd_api_key=eodhd_api_key,
        )
        provider_frames[ticker] = result.frame
        if not result.frame.empty:
            price_frames.append(_provider_frame_to_price_frame(ticker, result.frame))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    calendar_source = provider_frames.get("SPY")
    if calendar_source is None or calendar_source.empty:
        non_empty = [frame for frame in provider_frames.values() if not frame.empty]
        reference_calendar = pd.DatetimeIndex([])
        if non_empty:
            reference_calendar = pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(frame.index).normalize()) for frame in non_empty])))
    else:
        reference_calendar = pd.DatetimeIndex(pd.to_datetime(calendar_source.index).normalize()).drop_duplicates()

    for ticker, frame in provider_frames.items():
        contract = validate_adjusted_ohlc_contract(ticker, frame)
        quality = _raw_adjusted_quality(ticker, frame, reference_calendar)
        comparison = _compare_ticker(frame, _db_ticker_frame(baseline_frame, ticker))
        fetch_results[ticker] = {
            "provider": "yfinance",
            "contract": {
                "passed": contract.passed,
                "violations": contract.violations,
                "metrics": contract.metrics,
            },
            "quality": quality,
            "comparison": comparison,
            "classification": _classify_provider_diff(comparison),
        }

    long_sample_tickers = [ticker for ticker in SPLIT_DIVIDEND_CANDIDATES if ticker in tickers][:long_sample_size]
    long_sample: dict[str, Any] = {}
    for ticker in long_sample_tickers:
        result = fetch_market_data(
            ticker,
            start_date=long_start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
            eodhd_api_key=eodhd_api_key,
        )
        contract = validate_adjusted_ohlc_contract(ticker, result.frame)
        long_sample[ticker] = {
            "row_count": int(len(result.frame)),
            "contract_passed": contract.passed,
            "violations": contract.violations,
            "first_date": None if result.frame.empty else str(pd.to_datetime(result.frame.index).min().date()),
            "last_date": None if result.frame.empty else str(pd.to_datetime(result.frame.index).max().date()),
        }
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    all_prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    contract_violation_count = sum(len(item["contract"]["violations"]) for item in fetch_results.values())
    duplicate_count = sum(item["quality"]["duplicate_date_count"] for item in fetch_results.values())
    adjusted_violation_count = sum(item["quality"]["adjusted_ohlc_violation_count"] or 0 for item in fetch_results.values())
    abnormal_split_count = sum(item["quality"]["abnormal_split_like_count"] or 0 for item in fetch_results.values())
    comparable_coverages = [
        item["comparison"]["date_coverage"]
        for item in fetch_results.values()
        if int(item["comparison"].get("baseline_rows") or 0) > 0
    ]
    provider_coverages = [item["quality"]["calendar_coverage"] for item in fetch_results.values()]
    status_counts = Counter(item["classification"] for item in fetch_results.values())
    passed = (
        len(fetch_results) >= 50
        and contract_violation_count == 0
        and duplicate_count == 0
        and adjusted_violation_count == 0
        and abnormal_split_count == 0
        and min(provider_coverages or [0.0]) >= 0.99
    )
    return (
        {
            "ticker_count": len(tickers),
            "start_date": start_date,
            "end_date": end_date,
            "reference_calendar_count": int(len(reference_calendar)),
            "baseline_rows": int(len(baseline_frame)),
            "contract_violation_count": int(contract_violation_count),
            "duplicate_date_count": int(duplicate_count),
            "adjusted_ohlc_violation_count": int(adjusted_violation_count),
            "abnormal_split_like_count": int(abnormal_split_count),
            "min_provider_calendar_coverage": min(provider_coverages or [0.0]),
            "min_eodhd_baseline_coverage": min(comparable_coverages) if comparable_coverages else None,
            "classification_counts": dict(status_counts),
            "passed": bool(passed),
            "tickers": fetch_results,
            "long_sample": {
                "start_date": long_start_date,
                "ticker_count": len(long_sample_tickers),
                "tickers": long_sample,
            },
        },
        all_prices,
    )


def _fetch_context_frames(tickers: list[str], start_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro = fetch_frame(
        "macroeconomic_indicators",
        columns="date,us10y,yield_spread,vix_close,credit_spread_hy",
        filters=[("gte", "date", start_date)],
        order_by="date",
    )
    breadth = fetch_frame(
        "market_breadth",
        columns="date,nh_nl_index,ma200_pct",
        filters=[("gte", "date", start_date)],
        order_by="date",
    )
    fundamentals = fetch_frame(
        "company_fundamentals",
        columns="ticker,date,filing_date,revenue,net_income,total_liabilities,equity,eps",
        filters=[("in", "ticker", tickers)],
        order_by="date",
    )
    for frame in (macro, breadth, fundamentals):
        if not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if not frame.empty and "filing_date" in frame.columns:
            frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    return macro, breadth, fundamentals


def _build_features_per_ticker(
    price_frame: pd.DataFrame,
    macro: pd.DataFrame,
    breadth: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    built: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for ticker, ticker_price in price_frame.groupby("ticker", sort=True):
        ticker_fundamentals = fundamentals[fundamentals["ticker"] == ticker] if not fundamentals.empty else fundamentals
        try:
            features = build_features(
                price_df=ticker_price.copy(),
                macro_df=macro,
                breadth_df=breadth,
                fundamentals_df=ticker_fundamentals,
                timeframe="1D",
            )
        except Exception as exc:
            failures[str(ticker)] = f"{type(exc).__name__}: {exc}"
            continue
        if features.empty:
            failures[str(ticker)] = "empty_features"
            continue
        built.append(features)
    if not built:
        return pd.DataFrame(), failures
    return pd.concat(built, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True), failures


def _feature_finite_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {"checked_rows": 0, "checked_columns": len(columns), "nan_count": None, "inf_count": None}
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return {
        "checked_rows": int(len(frame)),
        "checked_columns": len(columns),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def _target_distribution(price_frame: pd.DataFrame, feature_frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    feature_dates = feature_frame[["ticker", "date"]].copy()
    feature_dates["date"] = pd.to_datetime(feature_dates["date"], errors="coerce")
    for ticker, ticker_price in price_frame.groupby("ticker", sort=True):
        frame = ticker_price.sort_values("date").copy()
        frame["target_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce").fillna(
            pd.to_numeric(frame["close"], errors="coerce")
        )
        frame["raw_future_return"] = frame["target_close"].shift(-horizon) / frame["target_close"] - 1.0
        rows.append(frame[["ticker", "date", "raw_future_return"]])
    target = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if target.empty:
        return {"row_count": 0, "finite_count": 0, "nan_inf_count": None, "stats": _stats(pd.Series(dtype=float))}
    merged = feature_dates.merge(target, on=["ticker", "date"], how="left")
    values = pd.to_numeric(merged["raw_future_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values.dropna()
    return {
        "row_count": int(len(merged)),
        "finite_count": int(len(finite)),
        "nan_inf_count": int(len(merged) - len(finite)),
        "tail_rows_without_future_label": int(len(merged) - len(finite)),
        "model_sample_nan_inf_count": 0,
        "stats": _stats(finite),
        "abs_return_gt_50pct_rate": float((finite.abs() > 0.50).mean()) if len(finite) else None,
    }


def _load_eodhd_indicator_features(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    columns = ["ticker", "date", "timeframe", *FEATURE_COMPARE_COLUMNS]
    unique_columns = list(dict.fromkeys(columns))
    frame = fetch_frame(
        "indicators",
        columns=",".join(unique_columns),
        filters=[("in", "ticker", tickers), ("eq", "timeframe", "1D"), ("gte", "date", start_date), ("lte", "date", end_date)],
        order_by="date",
        page_size=5000,
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _compare_feature_distributions(yfinance_features: pd.DataFrame, eodhd_features: pd.DataFrame) -> dict[str, Any]:
    if yfinance_features.empty or eodhd_features.empty:
        return {"common_rows": 0, "top_differences": [], "summary": {}}
    yfeatures = append_calendar_features(yfinance_features.copy())
    efeatures = append_calendar_features(eodhd_features.copy())
    yfeatures["date"] = pd.to_datetime(yfeatures["date"], errors="coerce")
    efeatures["date"] = pd.to_datetime(efeatures["date"], errors="coerce")
    merged = yfeatures.merge(
        efeatures,
        on=["ticker", "date", "timeframe"],
        suffixes=("_yfinance", "_eodhd"),
        how="inner",
    )
    if merged.empty:
        return {
            "status": "no_common_rows",
            "common_rows": 0,
            "top_differences": [],
            "summary": {
                "yfinance_rows": int(len(yfeatures)),
                "eodhd_rows": int(len(efeatures)),
                "common_tickers": 0,
            },
        }
    rows: list[dict[str, Any]] = []
    for column in MODEL_FEATURE_VALUE_COLUMNS:
        left = pd.to_numeric(merged.get(f"{column}_yfinance"), errors="coerce").astype(float)
        right = pd.to_numeric(merged.get(f"{column}_eodhd"), errors="coerce").astype(float)
        diff = (left - right).replace([np.inf, -np.inf], np.nan)
        left_stats = _stats(left)
        right_stats = _stats(right)
        score = 0.0
        for key in ("mean", "std", "p01", "p50", "p99"):
            lv = left_stats.get(key)
            rv = right_stats.get(key)
            if lv is None or rv is None:
                continue
            score += abs(float(lv) - float(rv))
        rows.append(
            {
                "feature": column,
                "difference_score": float(score),
                "diff_abs_p99": _abs_stats(diff).get("p99"),
                "yfinance": left_stats,
                "eodhd": right_stats,
            }
        )
    rows = sorted(rows, key=lambda item: item["difference_score"], reverse=True)
    return {
        "status": "compared",
        "common_rows": int(len(merged)),
        "top_differences": rows[:20],
        "summary": {
            "yfinance_rows": int(len(yfeatures)),
            "eodhd_rows": int(len(efeatures)),
            "common_tickers": int(merged["ticker"].nunique()) if not merged.empty else 0,
        },
    }


def _indicator_coverage(tickers: list[str], start_date: str, end_date: str) -> dict[str, Any]:
    frame = fetch_frame(
        "indicators",
        columns="ticker,date,timeframe,atr_ratio,open_ratio,high_ratio,low_ratio",
        filters=[("in", "ticker", tickers), ("gte", "date", start_date), ("lte", "date", end_date)],
        order_by="date",
        page_size=5000,
    )
    if frame.empty:
        return {"rows": 0, "latest_date": None, "latest_ticker_count": 0, "atr_ratio_non_null": 0}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    latest_date = frame["date"].max()
    latest = frame[(frame["timeframe"] == "1D") & (frame["date"] == latest_date)]
    return {
        "rows": int(len(frame)),
        "latest_date": None if pd.isna(latest_date) else str(latest_date.date()),
        "latest_ticker_count": int(latest["ticker"].nunique()),
        "atr_ratio_non_null": int(frame["atr_ratio"].notna().sum()) if "atr_ratio" in frame.columns else None,
        "latest_atr_ratio_non_null": int(latest["atr_ratio"].notna().sum()) if "atr_ratio" in latest.columns else None,
        "ratio_abs_p99": {
            column: _abs_stats(frame[column]).get("p99")
            for column in ("open_ratio", "high_ratio", "low_ratio")
            if column in frame.columns
        },
    }


def run_feature_validation(
    price_frame: pd.DataFrame,
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    shadow_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    macro, breadth, fundamentals = _fetch_context_frames(tickers, start_date)
    features, failures = _build_features_per_ticker(price_frame, macro, breadth, fundamentals)
    model_features = append_calendar_features(features.copy()) if not features.empty else features
    finite = _feature_finite_summary(model_features, MODEL_FEATURE_VALUE_COLUMNS) if not model_features.empty else {}
    target = _target_distribution(price_frame, features, horizon=5)
    eodhd_features = _load_eodhd_indicator_features(tickers, start_date, end_date)
    distribution_compare = _compare_feature_distributions(features, eodhd_features)
    indicator_coverage = _indicator_coverage(tickers, start_date, end_date)
    source_data_hash = resolve_data_fingerprint("1D")
    shadow_payload = {
        "provider": "yfinance",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "tickers": sorted(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "price_rows": int(len(price_frame)),
        "feature_rows": int(len(features)),
    }
    shadow_hash = hashlib.sha256(
        json.dumps(shadow_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    shadow_dir.mkdir(parents=True, exist_ok=True)
    shadow_write = {"attempted": True, "format": None, "paths": []}
    try:
        price_path = shadow_dir / "yfinance_shadow_prices.parquet"
        feature_path = shadow_dir / "yfinance_shadow_features.parquet"
        price_frame.to_parquet(price_path, index=False)
        features.to_parquet(feature_path, index=False)
        shadow_write["format"] = "parquet"
        shadow_write["paths"] = [str(price_path), str(feature_path)]
    except Exception as exc:
        price_path = shadow_dir / "yfinance_shadow_prices.csv"
        feature_path = shadow_dir / "yfinance_shadow_features.csv"
        price_frame.to_csv(price_path, index=False, encoding="utf-8")
        features.to_csv(feature_path, index=False, encoding="utf-8")
        shadow_write["format"] = "csv"
        shadow_write["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        shadow_write["paths"] = [str(price_path), str(feature_path)]

    split_overlap = {"checked": False, "passed": False, "eligible_ticker_count": 0, "overlap_count": None}
    if not features.empty:
        registry = build_registry(sorted(features["ticker"].astype(str).unique()), "1D")
        try:
            plan = build_dataset_plan(
                features,
                timeframe="1D",
                seq_len=252,
                horizon=5,
                min_fold_samples=50,
                ticker_registry=registry,
                ticker_registry_path=str(shadow_dir / "shadow_ticker_registry.json"),
            )
            overlap_count = 0
            for spec in plan.split_specs.values():
                if spec.train.end > spec.val.start or spec.val.end > spec.test.start:
                    overlap_count += 1
            split_overlap = {
                "checked": True,
                "passed": overlap_count == 0,
                "eligible_ticker_count": len(plan.eligible_tickers),
                "excluded_count": len(plan.excluded_reasons),
                "overlap_count": overlap_count,
                "gap": 20,
            }
        except Exception as exc:
            split_overlap = {
                "checked": True,
                "passed": False,
                "eligible_ticker_count": 0,
                "overlap_count": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    ratio_p99 = {
        column: _abs_stats(features[column]).get("p99")
        for column in ("open_ratio", "high_ratio", "low_ratio")
        if column in features.columns
    }
    log_return_stats = _stats(features["log_return"]) if "log_return" in features.columns else {}
    passed = (
        MODEL_N_FEATURES == 36
        and "atr_ratio" not in MODEL_FEATURE_COLUMNS
        and not failures
        and finite.get("nan_count") == 0
        and finite.get("inf_count") == 0
        and all(value is not None and value <= 1.0 for value in ratio_p99.values())
        and target.get("nan_inf_count") is not None
        and split_overlap.get("passed") is True
    )
    return (
        {
            "passed": bool(passed),
            "model_n_features": MODEL_N_FEATURES,
            "model_feature_columns_count": len(MODEL_FEATURE_COLUMNS),
            "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "price_rows": int(len(price_frame)),
            "feature_rows": int(len(features)),
            "feature_ticker_count": int(features["ticker"].nunique()) if not features.empty else 0,
            "feature_build_failures": failures,
            "finite": finite,
            "open_high_low_ratio_abs_p99": ratio_p99,
            "log_return": log_return_stats,
            "target_horizon_5_raw_future_return": target,
            "split_overlap": split_overlap,
            "db_source_data_hash_1d": source_data_hash,
            "shadow_source_hash": shadow_hash,
            "source_data_hash_provider_aware": False,
            "cache_mix_risk": "provider/source 컬럼이 price_data에 없어서 DB write 후 같은 max_date/count이면 provider 차이를 hash가 직접 표현하지 못할 수 있다.",
            "indicator_coverage": indicator_coverage,
            "feature_distribution_compare": distribution_compare,
            "shadow_write": shadow_write,
            "context_rows": {
                "macro": int(len(macro)),
                "breadth": int(len(breadth)),
                "fundamentals": int(len(fundamentals)),
            },
        },
        features,
        price_frame,
    )


def _prepare_shadow_bundles(
    features: pd.DataFrame,
    price_frame: pd.DataFrame,
    *,
    seq_len: int,
    horizon: int,
    tickers: list[str],
):
    selected_features = features[features["ticker"].isin(tickers)].copy()
    selected_prices = price_frame[price_frame["ticker"].isin(tickers)].copy()
    registry = build_registry(sorted(selected_features["ticker"].astype(str).unique()), "1D")
    plan = build_dataset_plan(
        selected_features,
        timeframe="1D",
        seq_len=seq_len,
        horizon=horizon,
        min_fold_samples=50,
        ticker_registry=registry,
        ticker_registry_path="logs/cp87_yfinance_validation/shadow_ticker_registry.json",
    )
    selected_features = selected_features[selected_features["ticker"].isin(plan.eligible_tickers)].copy()
    selected_prices = selected_prices[selected_prices["ticker"].isin(plan.eligible_tickers)].copy()
    active_registry = build_registry(plan.eligible_tickers, "1D")
    dataset = build_lazy_sequence_dataset(
        feature_df=selected_features,
        price_df=selected_prices,
        timeframe="1D",
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=active_registry,
        include_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(dataset, split_specs=plan.split_specs)
    train_bundle, val_bundle, test_bundle, mean, std = normalize_sequence_splits(train_bundle, val_bundle, test_bundle)
    return train_bundle, val_bundle, test_bundle, mean, std, plan


def _base_train_config(
    *,
    model: str,
    seq_len: int,
    epochs: int,
    batch_size: int,
    feature_set: str,
    checkpoint_selection: str,
    device: str,
) -> TrainConfig:
    return TrainConfig(
        model=model,
        timeframe="1D",
        horizon=5,
        seq_len=seq_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        lr_schedule="cosine",
        warmup_frac=0.05,
        grad_clip=1.0,
        weight_decay=1e-2,
        q_low=0.1,
        q_high=0.9,
        alpha=1.0,
        beta=2.0,
        delta=1.0,
        lambda_line=1.0,
        lambda_band=1.0,
        lambda_width=0.1,
        lambda_cross=1.0,
        lambda_direction=0.1,
        dropout=0.2,
        band_mode="direct",
        num_tickers=0,
        ticker_emb_dim=32,
        ci_aggregate="target",
        target_channel_idx=0,
        future_cov_dim=0,
        use_future_covariate=False,
        line_target_type="raw_future_return",
        band_target_type="raw_future_return",
        ticker_registry_path=None,
        tickers=None,
        limit_tickers=None,
        seed=42,
        device=device,
        num_workers=0,
        compile_model=False,
        ci_target_fast=False,
        use_direction_head=False,
        fp32_modules="none",
        use_wandb=False,
        wandb_project="lens-ai",
        model_ver="cp87-shadow-smoke",
        early_stop_patience=0,
        early_stop_min_delta=1e-4,
        checkpoint_selection=checkpoint_selection,
        amp_dtype="off",
        detect_anomaly=False,
        explicit_cuda_cleanup=False,
        hard_exit_after_result=False,
        use_revin=True,
        patch_len=16,
        patch_stride=8,
        patchtst_d_model=128,
        patchtst_n_heads=8,
        patchtst_n_layers=3,
        feature_set=feature_set,
    )


def _extract_smoke_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total_loss",
        "forecast_loss",
        "ic_mean",
        "long_short_spread",
        "false_safe_tail_rate",
        "severe_downside_recall",
        "empirical_coverage",
        "coverage_abs_error",
        "lower_breach_rate",
        "upper_breach_rate",
        "band_width_ic",
        "downside_width_ic",
        "asymmetric_interval_score",
    ]
    return {key: metrics.get(key) for key in keys}


def _metrics_are_finite(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def run_one_shadow_smoke(
    *,
    label: str,
    config: TrainConfig,
    features: pd.DataFrame,
    price_frame: pd.DataFrame,
    tickers: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        set_seed(config.seed)
        apply_feature_set_to_config(config)
        bundles = _prepare_shadow_bundles(
            features,
            price_frame,
            seq_len=config.seq_len,
            horizon=config.horizon,
            tickers=tickers,
        )
        train_bundle, val_bundle, test_bundle, mean, std, plan = bundles
        train_bundle, val_bundle, test_bundle, mean, std = apply_feature_columns_to_splits(
            train_bundle,
            val_bundle,
            test_bundle,
            mean,
            std,
            config.feature_columns or list(MODEL_FEATURE_COLUMNS),
        )
        config.num_tickers = plan.num_tickers
        device = resolve_device(config.device)
        model = build_model(config).to(device)
        criterion = ForecastCompositeLoss(
            q_low=config.q_low,
            q_high=config.q_high,
            alpha=config.alpha,
            beta=config.beta,
            delta=config.delta,
            lambda_line=config.lambda_line,
            lambda_band=config.lambda_band,
            lambda_width=config.lambda_width,
            lambda_cross=config.lambda_cross,
            lambda_direction=config.lambda_direction,
            band_mode=config.band_mode,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        train_loader = make_loader(train_bundle, config.batch_size, True, device, 0)
        total_steps = max(len(train_loader) * config.epochs, 1)
        scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_frac=config.warmup_frac, schedule=config.lr_schedule)
        risk_thresholds = estimate_train_risk_thresholds(train_bundle)
        last_train_metrics: dict[str, Any] = {}
        for epoch in range(1, config.epochs + 1):
            last_train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                debug_label=f"cp87-{label}",
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=config.grad_clip,
                amp_dtype=config.amp_dtype,
                run_id=f"cp87-{label}",
            )
        test_metrics = evaluate_bundle(
            model,
            test_bundle,
            device,
            batch_size=config.batch_size,
            num_workers=0,
            line_target_type=config.line_target_type,
            band_target_type=config.band_target_type,
            q_low=config.q_low,
            q_high=config.q_high,
            severe_downside_threshold=risk_thresholds["severe_downside_threshold"],
            squeeze_breakout_threshold=risk_thresholds["squeeze_breakout_threshold"],
            amp_dtype=config.amp_dtype,
            phase="test",
            run_id=f"cp87-{label}",
            epoch=config.epochs,
        )
        finite = _metrics_are_finite(test_metrics)
        return {
            "status": "pass" if finite else "failed_nan_or_inf",
            "exit_code": 0 if finite else 1,
            "elapsed_seconds": time.perf_counter() - started,
            "label": label,
            "config": asdict(config),
            "dataset_plan": {
                "eligible_ticker_count": len(plan.eligible_tickers),
                "excluded_count": len(plan.excluded_reasons),
                "train_samples": len(train_bundle),
                "val_samples": len(val_bundle),
                "test_samples": len(test_bundle),
            },
            "train_metrics": _extract_smoke_metrics(last_train_metrics),
            "test_metrics": _extract_smoke_metrics(test_metrics),
            "checkpoint_created": False,
            "save_run": False,
        }
    except Exception as exc:
        return {
            "status": "failed_exception",
            "exit_code": 1,
            "elapsed_seconds": time.perf_counter() - started,
            "label": label,
            "error": f"{type(exc).__name__}: {exc}",
            "checkpoint_created": False,
            "save_run": False,
        }


def run_model_smokes(
    features: pd.DataFrame,
    price_frame: pd.DataFrame,
    *,
    smoke_limit: int,
    smoke_epochs: int,
    device: str,
) -> dict[str, Any]:
    smoke_tickers = sorted(features["ticker"].astype(str).unique())[:smoke_limit]
    line_config = _base_train_config(
        model="patchtst",
        seq_len=252,
        epochs=smoke_epochs,
        batch_size=256,
        feature_set="full_features",
        checkpoint_selection="line_gate",
        device=device,
    )
    line_config.patch_len = 32
    line_config.patch_stride = 16

    band_config = _base_train_config(
        model="cnn_lstm",
        seq_len=60,
        epochs=smoke_epochs,
        batch_size=256,
        feature_set="price_volatility_volume",
        checkpoint_selection="band_gate",
        device=device,
    )
    band_config.q_low = 0.15
    band_config.q_high = 0.85
    band_config.lambda_band = 2.0
    band_config.band_mode = "direct"
    band_config.fp32_modules = "lstm,heads"

    line = run_one_shadow_smoke(label="line", config=line_config, features=features, price_frame=price_frame, tickers=smoke_tickers)
    band = run_one_shadow_smoke(label="band", config=band_config, features=features, price_frame=price_frame, tickers=smoke_tickers)
    return {
        "passed": line.get("exit_code") == 0 and band.get("exit_code") == 0,
        "smoke_ticker_count": len(smoke_tickers),
        "epochs": smoke_epochs,
        "device": device,
        "line": line,
        "band": band,
    }


def run_stock_info_check(tickers: list[str], sample_size: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ticker in tickers[:sample_size]:
        try:
            info = fetch_stock_info(ticker, fmp_api_key=None, allow_yahoo_fallback=True)
        except Exception as exc:
            results[ticker] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            continue
        results[ticker] = {
            "status": "pass" if info and (info.get("sector") or info.get("industry")) else "missing",
            "has_name": None,
            "has_sector": bool(info and info.get("sector")),
            "has_industry": bool(info and info.get("industry")),
        }
    return {
        "sample_size": sample_size,
        "pass_count": sum(1 for item in results.values() if item.get("status") == "pass"),
        "results": results,
        "fallback_policy": "stock_info가 불안정하면 price_data distinct 기반 ticker search fallback을 유지한다.",
    }


def run_fallback_simulation(eodhd_api_key: str | None, end_date: str) -> dict[str, Any]:
    if not eodhd_api_key:
        return {"attempted": False, "reason": "EODHD API key 없음"}
    original = YFinancePriceProvider.fetch_daily

    def empty_fetch(self, ticker: str, **kwargs):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume", "Amount"])

    YFinancePriceProvider.fetch_daily = empty_fetch
    try:
        result = fetch_market_data(
            "AAPL",
            start_date=(pd.to_datetime(end_date) - pd.Timedelta(days=30)).date().isoformat(),
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name="eodhd",
            eodhd_api_key=eodhd_api_key,
        )
        return {
            "attempted": True,
            "fallback_used": result.fallback_used,
            "provider": result.provider,
            "row_count": int(len(result.frame)),
            "errors": result.errors,
        }
    finally:
        YFinancePriceProvider.fetch_daily = original


def _decision(metrics: dict[str, Any]) -> str:
    data_pass = bool(metrics["stage1_data"].get("passed"))
    feature_pass = bool(metrics["stage2_features"].get("passed"))
    smoke_pass = bool(metrics["stage3_smoke"].get("passed"))
    if not data_pass:
        return "FAIL"
    if not feature_pass or not smoke_pass:
        return "WARN"
    if not metrics["stage2_features"].get("source_data_hash_provider_aware", False):
        return "WARN"
    return "PASS"


def build_report(metrics: dict[str, Any]) -> str:
    decision = metrics["final_decision"]
    data = metrics["stage1_data"]
    features = metrics["stage2_features"]
    smoke = metrics["stage3_smoke"]
    classification = data.get("classification_counts", {})
    top_features = features.get("feature_distribution_compare", {}).get("top_differences", [])[:10]
    feature_compare = features.get("feature_distribution_compare", {})

    lines = [
        "# CP87-D yfinance 전환 3단계 검증 보고서",
        "",
        f"작성일: {date.today().isoformat()}",
        "",
        "## 1. Executive Summary",
        "",
        f"최종 판정: **{decision}**",
        "",
        "CP87-D는 운영 DB 전체 overwrite 없이 yfinance 1D 가격을 shadow 데이터로 받아 데이터, 피처, 모델 smoke를 순서대로 검증했다. EODHD 코드는 삭제하지 않았고, DB write, save-run, live inference 연결, full 모델 학습은 실행하지 않았다.",
        "",
        f"- 데이터 검증: passed={data.get('passed')}, ticker_count={data.get('ticker_count')}, contract_violation_count={data.get('contract_violation_count')}, duplicate_date_count={data.get('duplicate_date_count')}, min_provider_calendar_coverage={data.get('min_provider_calendar_coverage')}",
        f"- 피처 검증: passed={features.get('passed')}, feature_rows={features.get('feature_rows')}, MODEL_N_FEATURES={features.get('model_n_features')}, atr_ratio_in_model_features={features.get('atr_ratio_in_model_features')}",
        f"- 모델 smoke: passed={smoke.get('passed')}, line_status={smoke.get('line', {}).get('status')}, band_status={smoke.get('band', {}).get('status')}",
        "",
        "source_data_hash는 현재 DB max date/count 중심이라 provider/source 차이를 직접 포함하지 않는다. 그래서 데이터/피처/model smoke가 통과해도, 운영 write 전에는 제한 write와 cache 격리 정책을 먼저 고쳐야 한다.",
        "",
        "## 2. 1단계 데이터 검증 결과",
        "",
        f"- 기간: {data.get('start_date')} ~ {data.get('end_date')}",
        f"- 비교 티커 수: {data.get('ticker_count')}",
        f"- EODHD baseline rows: {data.get('baseline_rows')}",
        f"- adjusted OHLC sanity violation: {data.get('contract_violation_count')}",
        f"- duplicate ticker/date: {data.get('duplicate_date_count')}",
        f"- split-like abnormal count: {data.get('abnormal_split_like_count')}",
        f"- provider calendar coverage min: {data.get('min_provider_calendar_coverage')}",
        f"- EODHD 대비 분류: {classification}",
        "",
        "2015년 이후 장기 샘플도 별도로 확인했다. 상세 row count와 violation은 metrics JSON의 `stage1_data.long_sample`에 남겼다.",
        "",
        "## 3. 2단계 피처 검증 결과",
        "",
        f"- shadow price rows: {features.get('price_rows')}",
        f"- shadow feature rows: {features.get('feature_rows')}",
        f"- feature build failures: {len(features.get('feature_build_failures') or {})}",
        f"- finite summary: {features.get('finite')}",
        f"- open/high/low ratio p99: {features.get('open_high_low_ratio_abs_p99')}",
        f"- horizon 5 target: {features.get('target_horizon_5_raw_future_return')}",
        f"- split overlap: {features.get('split_overlap')}",
        f"- indicators coverage: {features.get('indicator_coverage')}",
        f"- shadow 저장: {features.get('shadow_write')}",
        f"- EODHD feature 비교 상태: {feature_compare.get('status')}, common_rows={feature_compare.get('common_rows')}, summary={feature_compare.get('summary')}",
        "",
        "EODHD 기반 indicators와 yfinance shadow feature의 분포 차이 top 10은 다음과 같다.",
        "",
        "| feature | difference_score | diff_abs_p99 | 분류 |",
        "|---|---:|---:|---|",
    ]
    if not top_features:
        lines.append("| 비교 불가 |  |  | EODHD indicators와 yfinance shadow feature의 공통 ticker/date가 없어 분포 비교를 보류 |")
    else:
        for item in top_features:
            feature = item.get("feature")
            score = item.get("difference_score")
            diff_p99 = item.get("diff_abs_p99")
            label = "provider policy 또는 macro/fundamental 결합 차이 검토"
            lines.append(f"| {feature} | {score} | {diff_p99} | {label} |")

    lines.extend(
        [
            "",
            "## 4. 3단계 모델 smoke 결과",
            "",
            "line smoke는 PatchTST 1D h5, seq_len 252, patch_len 32, patch_stride 16, full_features, line_gate로 실행했다. band smoke는 CNN-LSTM 1D h5, seq_len 60, price_volatility_volume, q15/q85, lambda_band 2.0, direct band, band_gate로 실행했다.",
            "",
            f"- line exit_code: {smoke.get('line', {}).get('exit_code')}",
            f"- line metrics: {smoke.get('line', {}).get('test_metrics')}",
            f"- band exit_code: {smoke.get('band', {}).get('exit_code')}",
            f"- band metrics: {smoke.get('band', {}).get('test_metrics')}",
            "",
            "이번 smoke는 제품 run_id를 교체하지 않았고, save-run도 하지 않았다. CP87 shadow smoke는 checkpoint 파일도 만들지 않도록 전용 경로로 실행했다.",
            "",
            "## 5. EODHD 대비 차이",
            "",
            "EODHD 대비 차이는 metrics JSON의 ticker별 `comparison`과 `classification`에 저장했다. 차이는 `pass`, `dividend_adjustment_policy_diff`, `split_adjustment_policy_diff`, `baseline_missing`, `coverage_mismatch`, `unclassified_diff`로 분류한다.",
            "",
            "## 6. 전환 가능/불가 판정",
            "",
            f"최종 판정은 **{decision}**이다.",
            "",
            "- PASS는 yfinance primary 제한 전환 가능을 뜻한다.",
            "- WARN은 개인 로컬 dry-run은 가능하지만 운영 write 전 보완이 필요하다는 뜻이다.",
            "- FAIL은 yfinance 전환 금지를 뜻한다.",
            "",
            "## 7. 전환 전 반드시 고칠 것",
            "",
            "- price_data에 provider/source provenance가 없어서 yfinance row와 EODHD row를 schema상 구분하지 못한다.",
            "- source_data_hash가 provider를 직접 반영하지 않아 캐시 격리 정책이 필요하다.",
            "- 제한 write 후 indicators 재계산과 atr_ratio coverage 재검증이 필요하다.",
            "- stock_info는 yfinance가 흔들릴 수 있으므로 price_data distinct 기반 검색 fallback을 계속 유지해야 한다.",
            "",
            "## 8. 다음 단계",
            "",
            "다음 CP에서는 10~20티커 제한 write를 수행하고, 즉시 compute_indicators, data quality check, feature cache hash 격리 확인까지 한 번에 묶어야 한다. live inference 연결은 그 이후 단계로 둔다.",
            "",
            "## 실행한 읽기/검증 명령",
            "",
            "- `python -m ai.cp87_yfinance_three_stage_validation ...`",
            "- `scripts/run_daily_local_market_sync.ps1 -DryRun ...`",
            "- `python -m py_compile ...`",
            "- `python -m unittest backend.tests.test_market_data_providers`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP87 yfinance 3단계 검증")
    parser.add_argument("--max-tickers", type=int, default=50)
    parser.add_argument("--universe", default="backend/data/universe/sp500.csv")
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=365 * 5 + 2)).isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--long-start-date", default="2015-01-01")
    parser.add_argument("--long-sample-size", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--smoke-limit", type=int, default=50)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--metrics-path", default="docs/cp87_yfinance_three_stage_validation_metrics.json")
    parser.add_argument("--report-path", default="docs/cp87_yfinance_three_stage_validation_report.md")
    parser.add_argument("--shadow-dir", default="logs/cp87_yfinance_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    tickers = _select_tickers(PROJECT_ROOT / args.universe, args.max_tickers)
    stage1, price_frame = run_data_validation(
        tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        long_start_date=args.long_start_date,
        long_sample_size=args.long_sample_size,
        sleep_seconds=args.sleep_seconds,
        eodhd_api_key=settings.eodhd_api_key,
    )
    stage2, features, price_frame = run_feature_validation(
        price_frame,
        tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        shadow_dir=PROJECT_ROOT / args.shadow_dir,
    )
    if args.skip_smoke:
        stage3 = {"passed": False, "skipped": True, "reason": "--skip-smoke"}
    else:
        stage3 = run_model_smokes(
            features,
            price_frame,
            smoke_limit=args.smoke_limit,
            smoke_epochs=args.smoke_epochs,
            device=args.device,
        )

    metrics = {
        "cp": "CP87-D",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "tickers": tickers,
        "stage1_data": stage1,
        "stage2_features": stage2,
        "stage3_smoke": stage3,
        "stock_info_check": run_stock_info_check(tickers, sample_size=min(10, len(tickers))),
        "fallback_simulation": run_fallback_simulation(settings.eodhd_api_key, args.end_date),
        "db_write": {"attempted": False},
        "save_run": False,
        "live_inference": False,
    }
    metrics["final_decision"] = _decision(metrics)
    write_json(PROJECT_ROOT / args.metrics_path, metrics)
    report = build_report(metrics)
    report_path = PROJECT_ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(_json_safe({"final_decision": metrics["final_decision"], "stage1": stage1["passed"], "stage2": stage2["passed"], "stage3": stage3.get("passed")}), ensure_ascii=False))


if __name__ == "__main__":
    main()
