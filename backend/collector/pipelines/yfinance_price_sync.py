from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.collector.config import get_settings  # noqa: E402
from backend.collector.jobs.sync_prices import run as run_prices  # noqa: E402
from backend.collector.repositories.base import fetch_frame  # noqa: E402
from backend.collector.repositories.local_snapshots import local_snapshots_required, read_snapshot_frame  # noqa: E402
from backend.collector.sources.market_data_providers import fetch_market_data, normalize_provider_name  # noqa: E402
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402
from backend.collector.universe import load_tickers_from_csv  # noqa: E402


DEFAULT_COMPARE_TICKERS = [
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
ADJUSTED_CLOSE_MEDIAN_DIFF_LIMIT = 0.005
CLOSE_MEDIAN_DIFF_LIMIT = 0.005
LOCAL_PRIMARY_ADJUSTED_CLOSE_WARN_LIMIT = 0.02
MIN_DATE_COVERAGE = 0.99


def _default_start_date() -> str:
    return (date.today() - timedelta(days=730)).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def _relative_diff(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = right.abs().where(right.abs() > 1e-9, 1.0)
    return ((left - right).abs() / denominator).replace([np.inf, -np.inf], np.nan)


def _series_stats(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"median": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _db_ticker_frame(db_frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if db_frame.empty:
        return pd.DataFrame()
    ticker_frame = db_frame[db_frame["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if ticker_frame.empty:
        return ticker_frame
    ticker_frame["date"] = pd.to_datetime(ticker_frame["date"], errors="coerce")
    ticker_frame = ticker_frame.dropna(subset=["date"]).sort_values("date").set_index("date")
    return pd.DataFrame(
        {
            "Open": pd.to_numeric(ticker_frame["open"], errors="coerce"),
            "High": pd.to_numeric(ticker_frame["high"], errors="coerce"),
            "Low": pd.to_numeric(ticker_frame["low"], errors="coerce"),
            "Close": pd.to_numeric(ticker_frame["close"], errors="coerce"),
            "Adj Close": pd.to_numeric(ticker_frame["adjusted_close"], errors="coerce"),
            "Volume": pd.to_numeric(ticker_frame["volume"], errors="coerce"),
        },
        index=ticker_frame.index,
    )


def _compare_ticker(provider_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> dict:
    if baseline_frame.empty:
        return {
            "baseline_rows": 0,
            "provider_rows": int(len(provider_frame)),
            "date_coverage": 0.0,
            "missing_date_count": None,
            "close_relative_diff": _series_stats(pd.Series(dtype=float)),
            "adjusted_close_relative_diff": _series_stats(pd.Series(dtype=float)),
            "adjusted_factor_relative_diff": _series_stats(pd.Series(dtype=float)),
            "volume_relative_diff": _series_stats(pd.Series(dtype=float)),
        }

    provider = provider_frame.copy()
    baseline = baseline_frame.copy()
    provider.index = pd.to_datetime(provider.index).normalize()
    baseline.index = pd.to_datetime(baseline.index).normalize()
    provider = provider[~provider.index.duplicated(keep="last")]
    baseline = baseline[~baseline.index.duplicated(keep="last")]

    common_dates = provider.index.intersection(baseline.index)
    date_coverage = float(len(common_dates) / len(baseline.index)) if len(baseline.index) else 0.0
    missing_dates = baseline.index.difference(provider.index)

    if len(common_dates) == 0:
        return {
            "baseline_rows": int(len(baseline)),
            "provider_rows": int(len(provider)),
            "date_coverage": date_coverage,
            "missing_date_count": int(len(missing_dates)),
            "close_relative_diff": _series_stats(pd.Series(dtype=float)),
            "adjusted_close_relative_diff": _series_stats(pd.Series(dtype=float)),
            "adjusted_factor_relative_diff": _series_stats(pd.Series(dtype=float)),
            "volume_relative_diff": _series_stats(pd.Series(dtype=float)),
        }

    provider_aligned = provider.loc[common_dates]
    baseline_aligned = baseline.loc[common_dates]
    provider_factor = provider_aligned["Adj Close"] / provider_aligned["Close"].where(
        provider_aligned["Close"].abs() > 1e-9
    )
    baseline_factor = baseline_aligned["Adj Close"] / baseline_aligned["Close"].where(
        baseline_aligned["Close"].abs() > 1e-9
    )
    return {
        "baseline_rows": int(len(baseline)),
        "provider_rows": int(len(provider)),
        "common_rows": int(len(common_dates)),
        "date_coverage": date_coverage,
        "missing_date_count": int(len(missing_dates)),
        "close_relative_diff": _series_stats(_relative_diff(provider_aligned["Close"], baseline_aligned["Close"])),
        "adjusted_close_relative_diff": _series_stats(
            _relative_diff(provider_aligned["Adj Close"], baseline_aligned["Adj Close"])
        ),
        "adjusted_factor_relative_diff": _series_stats(_relative_diff(provider_factor, baseline_factor)),
        "volume_relative_diff": _series_stats(_relative_diff(provider_aligned["Volume"], baseline_aligned["Volume"])),
    }


def load_baseline_price_frame(tickers: list[str], start_date: str, end_date: str | None) -> pd.DataFrame:
    filters: list[tuple[str, str, object]] = [
        ("in", "ticker", tickers),
        ("gte", "date", start_date),
    ]
    if end_date:
        filters.append(("lte", "date", end_date))
    local_frame = read_snapshot_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume,source,provider",
        filters=filters,
        order_by="date",
        provider="eodhd",
    )
    if local_frame is not None:
        return local_frame
    if local_snapshots_required():
        raise RuntimeError("로컬 snapshot 모드에서는 yfinance 비교 기준 price_data를 Supabase에서 읽지 않습니다.")
    return fetch_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume",
        filters=filters,
        order_by="date",
        page_size=5000,
    )


def run_dry_run_compare(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str | None,
    provider: str,
    fallback_provider: str | None,
    eodhd_api_key: str | None,
) -> dict:
    provider_name = normalize_provider_name(provider)
    fallback_name = normalize_provider_name(fallback_provider) if fallback_provider else None
    baseline_frame = load_baseline_price_frame(tickers, start_date, end_date)
    ticker_results: dict[str, dict] = {}

    for ticker in tickers:
        fetch_result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name=provider_name,
            fallback_provider_name=fallback_name,
            eodhd_api_key=eodhd_api_key,
        )
        contract = validate_adjusted_ohlc_contract(ticker, fetch_result.frame)
        comparison = _compare_ticker(fetch_result.frame, _db_ticker_frame(baseline_frame, ticker))
        close_diff_median = comparison["close_relative_diff"]["median"]
        adjusted_diff_median = comparison["adjusted_close_relative_diff"]["median"]
        factor_diff_median = comparison["adjusted_factor_relative_diff"]["median"]
        baseline_missing = comparison["baseline_rows"] == 0
        coverage_ok = comparison["date_coverage"] >= MIN_DATE_COVERAGE
        raw_close_ok = close_diff_median is not None and close_diff_median <= CLOSE_MEDIAN_DIFF_LIMIT
        adjusted_close_ok = (
            adjusted_diff_median is not None and adjusted_diff_median <= LOCAL_PRIMARY_ADJUSTED_CLOSE_WARN_LIMIT
        )
        strict_adjusted_close_ok = (
            adjusted_diff_median is not None and adjusted_diff_median <= ADJUSTED_CLOSE_MEDIAN_DIFF_LIMIT
        )
        split_policy_diff = (
            strict_adjusted_close_ok
            and close_diff_median is not None
            and close_diff_median > 0.20
            and factor_diff_median is not None
            and factor_diff_median > 0.20
        )
        comparable_pass = (
            not baseline_missing
            and coverage_ok
            and adjusted_close_ok
            and (raw_close_ok or split_policy_diff)
        )
        ticker_pass = (
            contract.passed
            and not fetch_result.fallback_used
            and (baseline_missing or comparable_pass)
        )
        if baseline_missing:
            status = "baseline_missing_contract_only"
        elif split_policy_diff:
            status = "split_adjustment_policy_diff"
        elif comparable_pass and not strict_adjusted_close_ok:
            status = "dividend_adjustment_policy_diff"
        elif comparable_pass:
            status = "pass"
        else:
            status = "comparison_failed"
        ticker_results[ticker] = {
            "passed": ticker_pass,
            "status": status,
            "strict_adjusted_close_diff_pass": strict_adjusted_close_ok,
            "split_adjustment_policy_diff": split_policy_diff,
            "provider": fetch_result.provider,
            "requested_provider": fetch_result.requested_provider,
            "fallback_provider": fetch_result.fallback_provider,
            "fallback_used": fetch_result.fallback_used,
            "source_errors": fetch_result.errors,
            "contract": {
                "passed": contract.passed,
                "violations": contract.violations,
                "metrics": contract.metrics,
            },
            "comparison": comparison,
        }

    overall_pass = bool(ticker_results) and all(result["passed"] for result in ticker_results.values())
    return {
        "mode": "dry_run_compare",
        "provider": provider_name,
        "fallback_provider": fallback_name,
        "start_date": start_date,
        "end_date": end_date,
        "baseline_rows": int(len(baseline_frame)),
        "ticker_count": len(tickers),
        "overall_pass": overall_pass,
        "pass_criteria": {
            "adjusted_ohlc_sanity_violation_count": 0,
            "min_date_coverage": MIN_DATE_COVERAGE,
            "adjusted_close_median_relative_diff_max": ADJUSTED_CLOSE_MEDIAN_DIFF_LIMIT,
            "local_primary_adjusted_close_warn_limit": LOCAL_PRIMARY_ADJUSTED_CLOSE_WARN_LIMIT,
            "raw_close_median_relative_diff_max": CLOSE_MEDIAN_DIFF_LIMIT,
            "baseline_missing_policy": "DB baseline이 없으면 provider contract만 검사하고 write gate에서는 경고로 분류한다.",
            "split_policy_diff_policy": "adjusted_close가 일치하고 raw close/factor만 크게 다르면 split raw/adjusted 정책 차이로 분류한다.",
            "fallback_used": False,
        },
        "tickers": ticker_results,
    }


def write_metrics(path: str | Path, metrics: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="yfinance 로컬 primary 가격 수집 dry-run/write 파이프라인")
    parser.add_argument("--provider", default="yfinance", help="primary market data provider")
    parser.add_argument("--fallback-provider", default="eodhd", help="fallback market data provider. 빈 값이면 비활성")
    parser.add_argument("--tickers", nargs="*", help="write 대상 ticker. dry-run만 실행할 때는 기본 비교 샘플을 사용")
    parser.add_argument("--compare-tickers", nargs="*", default=DEFAULT_COMPARE_TICKERS, help="dry-run 비교 ticker")
    parser.add_argument("--universe", help="write 대상 universe CSV 경로")
    parser.add_argument("--start-date", default=_default_start_date())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--write", action="store_true", help="dry-run PASS 후 price_data에 upsert")
    parser.add_argument("--batch-limit", type=int, default=80)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument(
        "--metrics-path",
        default="docs/cp86_yfinance_local_primary_migration_metrics.json",
    )
    parser.add_argument("--allow-fail", action="store_true", help="dry-run 실패여도 metrics를 남기고 0으로 종료")
    return parser.parse_args()


def _resolve_write_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [ticker.strip().upper() for ticker in args.tickers if ticker.strip()]
    if args.universe:
        return load_tickers_from_csv(args.universe)
    return []


def main() -> None:
    args = parse_args()
    settings = get_settings()
    fallback_provider = args.fallback_provider.strip() if args.fallback_provider else None
    if fallback_provider == "":
        fallback_provider = None

    compare_tickers = [ticker.strip().upper() for ticker in args.compare_tickers if ticker.strip()]
    metrics = run_dry_run_compare(
        compare_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        provider=args.provider,
        fallback_provider=fallback_provider,
        eodhd_api_key=settings.eodhd_api_key,
    )
    write_result = None

    if args.write:
        if not metrics["overall_pass"]:
            metrics["write"] = {
                "attempted": False,
                "reason": "dry_run_compare_failed",
            }
            write_metrics(args.metrics_path, metrics)
            raise SystemExit("dry-run 비교가 PASS가 아니므로 write를 중단했습니다.")

        write_tickers = _resolve_write_tickers(args)
        if not write_tickers:
            metrics["write"] = {
                "attempted": False,
                "reason": "write_requires_tickers_or_universe",
            }
            write_metrics(args.metrics_path, metrics)
            raise SystemExit("write 모드는 --tickers 또는 --universe가 필요합니다.")

        write_result = run_prices(
            write_tickers,
            default_start=args.start_date,
            lookback_days=args.lookback_days,
            repair_mode=False,
            eodhd_api_key=settings.eodhd_api_key,
            batch_limit=args.batch_limit,
            sleep_seconds=args.sleep_seconds,
            allow_yahoo_fallback=False,
            require_fundamentals=False,
            provider=args.provider,
            fallback_provider=fallback_provider,
            force_start_date=True,
        )
        metrics["write"] = {
            "attempted": True,
            "result": write_result,
            "indicators_recompute_required": True,
        }
    else:
        metrics["write"] = {
            "attempted": False,
            "reason": "dry_run_only",
        }

    write_metrics(args.metrics_path, metrics)
    print(json.dumps(_json_safe({"overall_pass": metrics["overall_pass"], "write": metrics["write"]}), ensure_ascii=False))
    if not metrics["overall_pass"] and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
