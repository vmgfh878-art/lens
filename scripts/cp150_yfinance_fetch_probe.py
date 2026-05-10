from __future__ import annotations

import argparse
import contextlib
from datetime import date, datetime, timedelta
import io
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import requests
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.sources.market_data_providers import (  # noqa: E402
    _build_yahoo_chart_params,
    _yahoo_chart_headers,
    fetch_market_data,
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
PRICE_PATH = ROOT_DIR / "data" / "parquet" / "price_data_yfinance.parquet"
EODHD_500_PATH = ROOT_DIR / "data" / "parquet" / "price_data_eodhd_500.parquet"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def package_info() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "yfinance_version": getattr(yf, "__version__", None),
        "yfinance_path": getattr(yf, "__file__", None),
        "requests_version": getattr(requests, "__version__", None),
    }


def yfinance_debug_download(ticker: str) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    rows = 0
    columns: list[str] = []
    error = None
    try:
        if hasattr(yf, "enable_debug_mode"):
            yf.enable_debug_mode()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            frame = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, actions=False, progress=False, threads=False)
        rows = int(len(frame))
        columns = [str(column) for column in frame.columns]
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    return {
        "ticker": ticker,
        "rows": rows,
        "columns": columns,
        "error": error,
        "stdout_tail": stdout.getvalue()[-2000:],
        "stderr_tail": stderr.getvalue()[-4000:],
    }


def probe_chart_endpoint(
    ticker: str,
    *,
    host: str,
    params: dict[str, Any],
    browser_headers: bool,
    trust_env: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = trust_env
    url = f"https://{host}/v8/finance/chart/{ticker}"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "ticker": ticker,
        "host": host,
        "browser_headers": browser_headers,
        "trust_env": trust_env,
        "url": url,
        "params": params,
    }
    try:
        response = session.get(url, params=params, headers=_yahoo_chart_headers(browser_headers), timeout=timeout_seconds)
        elapsed = time.perf_counter() - started
        text = response.text
        result.update(
            {
                "elapsed_seconds": round(elapsed, 3),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "text_prefix_500": text[:500],
            }
        )
        try:
            payload = response.json()
            chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
            chart_result = chart.get("result") or []
            result["json_parse_ok"] = True
            result["chart_error"] = chart.get("error")
            result["chart_result_count"] = len(chart_result)
            if chart_result:
                result["timestamp_count"] = len(chart_result[0].get("timestamp") or [])
        except ValueError as exc:
            result["json_parse_ok"] = False
            result["json_error"] = f"{type(exc).__name__}:{exc}"
    except Exception as exc:
        elapsed = time.perf_counter() - started
        result.update({"elapsed_seconds": round(elapsed, 3), "error": f"{type(exc).__name__}:{exc}"})
    return result


def load_sample_universe(limit: int) -> list[str]:
    if PRICE_PATH.exists():
        frame = pd.read_parquet(PRICE_PATH, columns=["ticker"])
        tickers = frame["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    else:
        tickers = DEFAULT_TICKERS
    ordered = []
    for ticker in [*DEFAULT_TICKERS, *tickers]:
        if ticker not in ordered:
            ordered.append(ticker)
    return ordered[:limit]


def measure_fetch_sample(tickers: list[str], *, end_date: str, lookback_days: int, sleep_seconds: float) -> dict[str, Any]:
    start_date = (pd.to_datetime(end_date).date() - timedelta(days=lookback_days)).isoformat()
    rows = []
    started = time.perf_counter()
    for ticker in tickers:
        ticker_started = time.perf_counter()
        result = fetch_market_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name="yfinance",
            fallback_provider_name=None,
            eodhd_api_key=None,
        )
        elapsed = time.perf_counter() - ticker_started
        rows.append(
            {
                "ticker": ticker,
                "success": not result.frame.empty,
                "row_count": int(len(result.frame)),
                "elapsed_seconds": round(elapsed, 3),
                "provider": result.provider,
                "fallback_used": bool(result.fallback_used),
                "errors": result.errors,
                "fetch_method": result.frame.attrs.get("fetch_method") if not result.frame.empty else None,
                "yahoo_chart_host": result.frame.attrs.get("yahoo_chart_host") if not result.frame.empty else None,
            }
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)
    total_elapsed = time.perf_counter() - started
    success_count = sum(1 for item in rows if item["success"])
    failed_count = len(rows) - success_count
    total_rows = sum(int(item["row_count"]) for item in rows)
    empty_or_json_errors = sum(
        1
        for item in rows
        if (not item["success"]) and any("empty" in error.lower() or "json" in error.lower() for error in item["errors"])
    )
    ticker_per_hour = len(rows) / total_elapsed * 3600 if total_elapsed > 0 else None
    return {
        "sample_size": len(rows),
        "start_date": start_date,
        "end_date": end_date,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "success_ticker_count": success_count,
        "failed_ticker_count": failed_count,
        "total_row_count": total_rows,
        "avg_fetch_seconds_per_ticker": round(total_elapsed / len(rows), 3) if rows else None,
        "ticker_per_hour": round(ticker_per_hour, 2) if ticker_per_hour else None,
        "empty_json_error_ratio": empty_or_json_errors / len(rows) if rows else None,
        "fallback_used_count": sum(1 for item in rows if item["fallback_used"]),
        "tickers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CP150 yfinance/Yahoo chart fetch probe")
    parser.add_argument("--output", default="docs/cp150_dg_yfinance_fetch_probe_metrics.json")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[5])
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    args = parser.parse_args()

    end_date = pd.to_datetime(args.end_date).date().isoformat()
    start_date = (pd.to_datetime(end_date).date() - timedelta(days=args.lookback_days)).isoformat()
    range_params = _build_yahoo_chart_params(period="10d")
    period_params = _build_yahoo_chart_params(start_date=start_date, end_date=end_date)

    endpoint_probes = []
    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        for params_name, params in [("range_10d", range_params), ("period1_period2", period_params)]:
            for browser_headers in [False, True]:
                for trust_env in [True, False]:
                    probe = probe_chart_endpoint(
                        args.ticker,
                        host=host,
                        params=params,
                        browser_headers=browser_headers,
                        trust_env=trust_env,
                        timeout_seconds=args.timeout_seconds,
                    )
                    probe["params_name"] = params_name
                    endpoint_probes.append(probe)

    sample_metrics = []
    for size in args.sample_sizes:
        tickers = load_sample_universe(size)
        sample_metrics.append(
            measure_fetch_sample(
                tickers,
                end_date=end_date,
                lookback_days=args.lookback_days,
                sleep_seconds=args.sleep_seconds,
            )
        )

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "package_info": package_info(),
        "yf_download_debug": yfinance_debug_download(args.ticker),
        "endpoint_probes": endpoint_probes,
        "sample_metrics": sample_metrics,
        "forbidden_actions_observed": {
            "eodhd_fallback": False,
            "supabase_bulk_read_write": False,
            "db_write": False,
            "model_training": False,
            "inference_save": False,
        },
    }
    output = ROOT_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_status": sample_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
