from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    prepare_dataset_splits,
    resolve_data_fingerprint,
)
from ai.train import resolve_feature_columns  # noqa: E402
from backend.app.services.feature_svc import build_features  # noqa: E402
from backend.collector.repositories.base import fetch_frame  # noqa: E402
from backend.collector.sources.market_data_providers import (  # noqa: E402
    fetch_market_data,
    provider_adjustment_policy,
)
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402


CP95_METRICS = ROOT_DIR / "docs" / "cp95_yfinance_100ticker_long_history_validation_metrics.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_LOG_DIR = ROOT_DIR / "logs" / "cp98_local_parquet_snapshot_bootstrap"
REQUIRED_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if pd.isna(value) if value is not None and not isinstance(value, (str, bytes, list, dict, tuple)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_cp95_tickers(limit: int | None) -> list[str]:
    payload = json.loads(CP95_METRICS.read_text(encoding="utf-8"))
    tickers = [str(ticker).upper() for ticker in payload.get("final_tickers", []) if str(ticker).strip()]
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def fetch_stock_info_snapshot(tickers: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    filters: list[tuple[str, str, object]] = [("in", "ticker", tickers)]
    frame = fetch_frame("stock_info", columns="*", filters=filters, order_by="ticker", limit=1000)
    if not frame.empty and "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame = frame[frame["ticker"].isin(tickers)].copy()
        frame = frame.drop_duplicates(subset=["ticker"], keep="last").sort_values("ticker").reset_index(drop=True)
    missing = sorted(set(tickers) - set(frame["ticker"].tolist() if "ticker" in frame.columns else []))
    return frame, {
        "rows": int(len(frame)),
        "missing_tickers": missing,
        "missing_count": len(missing),
    }


def price_frame_to_records(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    now = datetime.utcnow().isoformat() + "Z"
    policy = provider_adjustment_policy("yfinance")
    working = frame.copy()
    working.index = pd.to_datetime(working.index)
    records = pd.DataFrame(
        {
            "ticker": ticker,
            "date": working.index.date.astype(str),
            "open": pd.to_numeric(working["Open"], errors="coerce"),
            "high": pd.to_numeric(working["High"], errors="coerce"),
            "low": pd.to_numeric(working["Low"], errors="coerce"),
            "close": pd.to_numeric(working["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(working["Adj Close"], errors="coerce"),
            "volume": pd.to_numeric(working["Volume"], errors="coerce").fillna(0).astype("int64"),
            "amount": pd.to_numeric(working.get("Amount", working["Close"] * working["Volume"]), errors="coerce"),
            "source": "yfinance",
            "provider": "yfinance",
            "provider_adjustment_policy": policy,
            "updated_at": now,
        }
    )
    records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
    return records


def build_price_snapshot(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    chunks: list[pd.DataFrame] = []
    ticker_metrics: dict[str, Any] = {}
    failed: dict[str, str] = {}

    for ticker in tickers:
        result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
        )
        if result.frame.empty:
            failed[ticker] = "source_empty"
            ticker_metrics[ticker] = {
                "rows": 0,
                "passed": False,
                "violations": ["source_empty"],
                "source_errors": result.errors,
            }
            continue
        contract = validate_adjusted_ohlc_contract(ticker, result.frame)
        ticker_metrics[ticker] = {
            "rows": int(len(result.frame)),
            "passed": bool(contract.passed),
            "violations": contract.violations,
            "metrics": contract.metrics,
        }
        if not contract.passed:
            failed[ticker] = "adjusted_ohlc_contract_failed"
            continue
        chunks.append(price_frame_to_records(ticker, result.frame))

    if chunks:
        price_df = pd.concat(chunks, ignore_index=True)
        price_df = price_df.sort_values(["ticker", "date"]).drop_duplicates(
            subset=["ticker", "date", "source"],
            keep="last",
        )
    else:
        price_df = pd.DataFrame()

    duplicate_count = 0
    if not price_df.empty:
        duplicate_count = int(price_df.duplicated(subset=["ticker", "date", "source"]).sum())

    return price_df.reset_index(drop=True), {
        "requested_tickers": len(tickers),
        "processed_tickers": int(price_df["ticker"].nunique()) if "ticker" in price_df.columns else 0,
        "row_count": int(len(price_df)),
        "date_min": None if price_df.empty else str(price_df["date"].min()),
        "date_max": None if price_df.empty else str(price_df["date"].max()),
        "duplicate_ticker_date_source": duplicate_count,
        "failed": failed,
        "ticker_metrics": ticker_metrics,
    }


def build_indicator_snapshot(price_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = build_features(price_df=price_df, timeframe="1D")
    if features.empty:
        return features, {
            "row_count": 0,
            "ticker_count": 0,
            "error": "empty_features",
        }
    features = features.copy()
    features["source"] = "yfinance"
    features["provider"] = "yfinance"
    features["updated_at"] = datetime.utcnow().isoformat() + "Z"
    features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")
    features = features.sort_values(["ticker", "timeframe", "date", "source"]).drop_duplicates(
        subset=["ticker", "timeframe", "date", "source"],
        keep="last",
    )

    numeric_columns = [column for column in MODEL_FEATURE_COLUMNS if column in features.columns]
    feature_values = features[numeric_columns].to_numpy(dtype=float)
    finite_failures = int((~np.isfinite(feature_values)).sum())
    duplicate_count = int(features.duplicated(subset=["ticker", "timeframe", "date", "source"]).sum())
    atr_non_null = int(features["atr_ratio"].notna().sum()) if "atr_ratio" in features.columns else 0
    atr_coverage = float(atr_non_null / len(features)) if len(features) else 0.0

    return features.reset_index(drop=True), {
        "row_count": int(len(features)),
        "ticker_count": int(features["ticker"].nunique()),
        "date_min": str(features["date"].min()),
        "date_max": str(features["date"].max()),
        "duplicate_ticker_timeframe_date_source": duplicate_count,
        "feature_non_finite_count": finite_failures,
        "atr_ratio_non_null": atr_non_null,
        "atr_ratio_coverage": atr_coverage,
    }


def checksum_frame(frame: pd.DataFrame, columns: list[str]) -> str | None:
    if frame.empty:
        return None
    existing = [column for column in columns if column in frame.columns]
    payload = frame[existing].sort_values(existing[: min(3, len(existing))]).to_dict(orient="records")
    return hashlib.sha256(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def verify_local_split_gate(tickers: list[str]) -> dict[str, Any]:
    try:
        train, val, test, _, _, plan = prepare_dataset_splits(
            timeframe="1D",
            seq_len=60,
            horizon=5,
            tickers=tickers,
            market_data_provider="yfinance",
        )
        return {
            "status": "PASS",
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "source_data_hash": plan.source_data_hash,
            "feature_version": FEATURE_CONTRACT_VERSION,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP98 local parquet snapshot bootstrap")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=(date.today() + timedelta(days=1)).isoformat())
    parser.add_argument("--limit-tickers", type=int, default=100)
    parser.add_argument("--metrics-path", default=str(ROOT_DIR / "docs" / "cp98_local_parquet_snapshot_bootstrap_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tickers = load_cp95_tickers(args.limit_tickers)
    for ticker in REQUIRED_TICKERS:
        if ticker not in tickers:
            tickers.insert(0, ticker)
    tickers = list(dict.fromkeys(tickers))

    metrics: dict[str, Any] = {
        "cp": "CP98-D",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scope": {
            "provider": "yfinance",
            "source": "yfinance",
            "timeframe": "1D",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "ticker_count": len(tickers),
            "tickers": tickers,
        },
        "forbidden_actions_observed": {
            "supabase_price_data_read": False,
            "supabase_indicators_read": False,
            "supabase_price_data_write": False,
            "supabase_indicators_write": False,
            "model_training": False,
            "inference_save": False,
            "product_run_replacement": False,
            "eodhd_delete": False,
            "one_week_or_one_month_processing": False,
        },
    }

    stock_info, stock_metrics = fetch_stock_info_snapshot(tickers)
    stock_info_path = output_dir / "stock_info.parquet"
    stock_info.to_parquet(stock_info_path, index=False)
    metrics["stock_info_snapshot"] = {
        **stock_metrics,
        "path": str(stock_info_path),
        "bytes": stock_info_path.stat().st_size,
    }

    price_df, price_metrics = build_price_snapshot(tickers, start_date=args.start_date, end_date=args.end_date)
    price_path = output_dir / "price_data_yfinance.parquet"
    price_df.to_parquet(price_path, index=False)
    metrics["price_snapshot"] = {
        **price_metrics,
        "path": str(price_path),
        "bytes": price_path.stat().st_size,
        "checksum": checksum_frame(price_df, ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume", "source"]),
        "provider_adjustment_policy": provider_adjustment_policy("yfinance"),
    }

    indicators_df, indicator_metrics = build_indicator_snapshot(price_df)
    indicators_path = output_dir / "indicators_yfinance_1D.parquet"
    indicators_df.to_parquet(indicators_path, index=False)
    metrics["indicator_snapshot"] = {
        **indicator_metrics,
        "path": str(indicators_path),
        "bytes": indicators_path.stat().st_size,
        "checksum": checksum_frame(indicators_df, ["ticker", "timeframe", "date", *MODEL_FEATURE_COLUMNS, "atr_ratio", "source"]),
    }

    metrics["feature_contract"] = {
        "MODEL_N_FEATURES": MODEL_N_FEATURES,
        "FEATURE_CONTRACT_VERSION": FEATURE_CONTRACT_VERSION,
        "atr_ratio_in_MODEL_FEATURE_COLUMNS": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "band_feature_set": "price_volatility_volume",
        "band_feature_set_columns": resolve_feature_columns("price_volatility_volume"),
    }

    split_tickers = [ticker for ticker in REQUIRED_TICKERS if ticker in set(price_df.get("ticker", []))]
    metrics["local_split_gate"] = verify_local_split_gate(split_tickers)

    try:
        metrics["source_data_hash"] = resolve_data_fingerprint(
            "1D",
            tickers=split_tickers,
            market_data_provider="yfinance",
        )
    except Exception as exc:
        metrics["source_data_hash"] = None
        metrics["source_data_hash_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    price_pass = (
        price_metrics["row_count"] > 0
        and price_metrics["duplicate_ticker_date_source"] == 0
        and not price_metrics["failed"]
    )
    indicator_pass = (
        indicator_metrics.get("row_count", 0) > 0
        and indicator_metrics.get("duplicate_ticker_timeframe_date_source", 1) == 0
        and indicator_metrics.get("feature_non_finite_count", 1) == 0
    )
    stock_warn = stock_metrics["missing_count"] > 0
    split_pass = metrics["local_split_gate"].get("status") == "PASS"
    metrics["final_decision"] = {
        "status": "PASS" if price_pass and indicator_pass and split_pass and not stock_warn else "WARN" if price_pass and indicator_pass and split_pass else "FAIL",
        "price_pass": price_pass,
        "indicator_pass": indicator_pass,
        "stock_info_warn": stock_warn,
        "local_split_gate_pass": split_pass,
    }

    write_json(Path(args.metrics_path), metrics)
    write_json(log_dir / "cp98_snapshot_bootstrap_metrics.json", metrics)
    print(json.dumps(_json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
