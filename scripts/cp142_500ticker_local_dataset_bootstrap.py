from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
CONTEXT_DIR = SNAPSHOT_DIR / "context"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp142_500ticker_local_dataset_bootstrap"
UNIVERSE_PATH = ROOT_DIR / "backend" / "data" / "universe" / "sp500.csv"

PRICE_500_PATH = SNAPSHOT_DIR / "price_data_yfinance_500.parquet"
INDICATOR_1D_500_PATH = SNAPSHOT_DIR / "indicators_yfinance_1D_500.parquet"
INDICATOR_1W_500_PATH = SNAPSHOT_DIR / "indicators_yfinance_1W_500.parquet"
PRICE_1W_500_PATH = SNAPSHOT_DIR / "price_data_yfinance_1W_500.parquet"
STOCK_INFO_500_PATH = SNAPSHOT_DIR / "stock_info_yfinance_500.parquet"
CONTEXT_500_DIR = CONTEXT_DIR / "cp142_500"

DEFAULT_PREFLIGHT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX"]
FRED_SERIES = {
    "DGS10": "us10y",
    "DGS2": "us2y",
    "VIXCLS": "vix_close",
    "BAMLH0A0HYM2": "credit_spread_hy",
}
MACRO_COLUMNS = ["us10y", "yield_spread", "vix_close", "credit_spread_hy"]

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)
os.environ["WANDB_MODE"] = "disabled"

for _proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    _proxy_value = os.environ.get(_proxy_key, "")
    if "127.0.0.1:9" in _proxy_value:
        os.environ.pop(_proxy_key, None)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import FEATURE_COLUMNS, build_features, resample_price_frame  # noqa: E402
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402
from backend.collector.utils.network import sanitize_proxy_env  # noqa: E402
from scripts.cp133_local_full_features_context_backfill import (  # noqa: E402
    build_market_breadth_context,
    build_sector_context,
    fetch_edgar_fundamentals_context,
    frame_checksum,
    indicator_value_checksum,
)

sanitize_proxy_env()


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if value is not None and not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_sp500_universe() -> pd.DataFrame:
    frame = pd.read_csv(UNIVERSE_PATH)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["yfinance_symbol"] = frame["ticker"].str.replace(".", "-", regex=False)
    frame = frame.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
    return frame


def direct_yahoo_probe(ticker: str) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    started = datetime.utcnow()
    try:
        response = session.get(
            url,
            params={"range": "10d", "interval": "1d", "events": "div,splits"},
            timeout=20,
        )
        elapsed = (datetime.utcnow() - started).total_seconds()
        text_head = response.text[:160].replace("\n", " ")
        json_ok = False
        timestamp_count = None
        json_error = None
        try:
            payload = response.json()
            result = payload.get("chart", {}).get("result") or []
            timestamp_count = len(result[0].get("timestamp", [])) if result else 0
            json_ok = True
        except Exception as exc:
            json_error = f"{type(exc).__name__}: {exc}"
        return {
            "ticker": ticker,
            "status_code": int(response.status_code),
            "content_type": response.headers.get("content-type"),
            "text_head": text_head,
            "json_ok": json_ok,
            "json_error": json_error,
            "timestamp_count": timestamp_count,
            "elapsed_seconds": round(elapsed, 3),
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "status_code": None,
            "content_type": None,
            "text_head": None,
            "json_ok": False,
            "json_error": f"{type(exc).__name__}: {exc}",
            "timestamp_count": None,
            "elapsed_seconds": None,
        }


def preflight_yahoo(tickers: list[str]) -> dict[str, Any]:
    probes = {ticker: direct_yahoo_probe(ticker) for ticker in tickers}
    status_codes = [probe.get("status_code") for probe in probes.values()]
    all_429 = bool(status_codes) and all(code == 429 for code in status_codes)
    any_429 = any(code == 429 for code in status_codes)
    any_json_ok = any(bool(probe.get("json_ok")) and int(probe.get("timestamp_count") or 0) > 0 for probe in probes.values())
    if all_429:
        status = "FAIL_429"
        reason = "Yahoo chart API preflight가 모든 티커에서 429 Too Many Requests를 반환했다."
    elif any_429 and not any_json_ok:
        status = "FAIL_PARTIAL_429"
        reason = "Yahoo chart API preflight에 429가 포함됐고 정상 JSON 응답이 없다."
    elif any_json_ok:
        status = "PASS"
        reason = "Yahoo chart API preflight에서 정상 JSON 응답을 확인했다."
    else:
        status = "FAIL_NO_VALID_JSON"
        reason = "Yahoo chart API preflight에서 정상 가격 JSON을 확인하지 못했다."
    return {
        "status": status,
        "reason": reason,
        "probes": probes,
        "any_json_ok": any_json_ok,
        "any_429": any_429,
        "all_429": all_429,
    }


def fetch_yfinance_batch(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        " ".join(symbols),
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False,
        threads=False,
        progress=False,
        timeout=30,
        group_by="ticker",
    )


def extract_symbol_frame(batch: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame()
    if isinstance(batch.columns, pd.MultiIndex):
        if symbol not in batch.columns.get_level_values(0):
            return pd.DataFrame()
        frame = batch[symbol].copy()
    else:
        frame = batch.copy()
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return pd.DataFrame()
    return frame[required].dropna(how="all")


def price_frame_to_records(ticker: str, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    policy = provider_adjustment_policy("yfinance")
    working = frame.copy()
    working.index = pd.to_datetime(working.index, errors="coerce").normalize()
    records = pd.DataFrame(
        {
            "ticker": ticker,
            "yfinance_symbol": symbol,
            "date": working.index,
            "open": pd.to_numeric(working["Open"], errors="coerce"),
            "high": pd.to_numeric(working["High"], errors="coerce"),
            "low": pd.to_numeric(working["Low"], errors="coerce"),
            "close": pd.to_numeric(working["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(working["Adj Close"], errors="coerce"),
            "volume": pd.to_numeric(working["Volume"], errors="coerce").fillna(0).astype("int64"),
            "amount": pd.to_numeric(working["Close"], errors="coerce") * pd.to_numeric(working["Volume"], errors="coerce").fillna(0),
            "source": "yfinance",
            "provider": "yfinance",
            "provider_adjustment_policy": policy,
            "updated_at": utc_now_iso(),
        }
    )
    return records.dropna(subset=["date"]).reset_index(drop=True)


def fetch_prices(
    universe: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    current_date: date,
    batch_size: int,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    chunks: list[pd.DataFrame] = []
    failed_rows: list[dict[str, Any]] = []
    ticker_metrics: dict[str, Any] = {}
    aborted = False
    abort_reason = None

    for batch_index, start in enumerate(range(0, len(universe), batch_size), start=1):
        batch_universe = universe.iloc[start : start + batch_size].copy()
        symbols = batch_universe["yfinance_symbol"].tolist()
        try:
            batch_frame = fetch_yfinance_batch(symbols, start_date, end_date)
        except Exception as exc:
            reason = f"batch_exception:{type(exc).__name__}:{str(exc)[:160]}"
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                aborted = True
                abort_reason = reason
            for row in batch_universe.to_dict(orient="records"):
                failed_rows.append({**row, "stage": "price_fetch", "reason": reason})
            if aborted:
                remaining = universe.iloc[start + batch_size :].copy()
                for row in remaining.to_dict(orient="records"):
                    failed_rows.append({**row, "stage": "price_fetch", "reason": "not_attempted_after_abort"})
                break
            continue

        if batch_frame.empty:
            probes = preflight_yahoo(symbols[: min(2, len(symbols))])
            reason = "batch_empty"
            if probes["any_429"]:
                reason = "batch_empty_after_yahoo_429"
                aborted = True
                abort_reason = reason
            for row in batch_universe.to_dict(orient="records"):
                failed_rows.append({**row, "stage": "price_fetch", "reason": reason})
            if aborted:
                remaining = universe.iloc[start + batch_size :].copy()
                for row in remaining.to_dict(orient="records"):
                    failed_rows.append({**row, "stage": "price_fetch", "reason": "not_attempted_after_abort"})
                break
            continue

        for row in batch_universe.to_dict(orient="records"):
            ticker = str(row["ticker"])
            symbol = str(row["yfinance_symbol"])
            symbol_frame = extract_symbol_frame(batch_frame, symbol)
            if symbol_frame.empty:
                reason = "symbol_empty_in_batch"
                failed_rows.append({**row, "stage": "price_fetch", "reason": reason})
                ticker_metrics[ticker] = {"status": "FAIL", "reason": reason, "rows": 0}
                continue
            contract = validate_adjusted_ohlc_contract(ticker, symbol_frame)
            if not contract.passed:
                reason = "adjusted_ohlc_contract_failed"
                failed_rows.append({**row, "stage": "price_contract", "reason": reason, "violations": ";".join(contract.violations)})
                ticker_metrics[ticker] = {"status": "FAIL", "reason": reason, "rows": len(symbol_frame), "violations": contract.violations}
                continue
            records = price_frame_to_records(ticker, symbol, symbol_frame)
            records = records[pd.to_datetime(records["date"], errors="coerce").dt.date < current_date].copy()
            if records.empty:
                reason = "only_partial_or_no_completed_rows"
                failed_rows.append({**row, "stage": "price_contract", "reason": reason})
                ticker_metrics[ticker] = {"status": "FAIL", "reason": reason, "rows": 0}
                continue
            chunks.append(records)
            ticker_metrics[ticker] = {
                "status": "PASS",
                "rows": int(len(records)),
                "date_min": records["date"].min().date().isoformat(),
                "date_max": records["date"].max().date().isoformat(),
            }
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    price = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not price.empty:
        price = price.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date", "source"], keep="last")
        price["date"] = pd.to_datetime(price["date"]).dt.strftime("%Y-%m-%d")
    failed = pd.DataFrame(failed_rows)
    return price.reset_index(drop=True), failed.reset_index(drop=True), {
        "aborted": aborted,
        "abort_reason": abort_reason,
        "ticker_metrics": ticker_metrics,
        "batch_size": batch_size,
        "price_rows": int(len(price)),
        "price_ticker_count": int(price["ticker"].nunique()) if not price.empty else 0,
    }


def fetch_fred_macro(start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    series_ids = ",".join(FRED_SERIES.keys())
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_ids}&cosd={start_date}&coed={end_date}"
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        content = response.content
        if content.startswith(b"PK"):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                with archive.open(csv_names[0]) as handle:
                    raw = pd.read_csv(handle)
        else:
            raw = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        return pd.DataFrame(columns=["date", *MACRO_COLUMNS]), {"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}
    if raw.empty or "observation_date" not in raw.columns:
        return pd.DataFrame(columns=["date", *MACRO_COLUMNS]), {"status": "FAIL", "error": "empty_or_missing_observation_date"}
    raw = raw.rename(columns={"observation_date": "date", **FRED_SERIES})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for column in ["us10y", "us2y", "vix_close", "credit_spread_hy"]:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column].replace(".", np.nan), errors="coerce")
    raw["yield_spread"] = raw["us10y"] - raw["us2y"] if {"us10y", "us2y"}.issubset(raw.columns) else np.nan
    frame = raw[["date", *MACRO_COLUMNS]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["source"] = "fred"
    frame["provider"] = "fred"
    frame["asof_date"] = frame["date"]
    frame["release_policy"] = "fred_observation_date_close_available"
    frame["context_version"] = "cp142_500_context_v1"
    return frame, {
        "status": "PASS",
        "rows": int(len(frame)),
        "date_min": frame["date"].min().date().isoformat() if not frame.empty else None,
        "date_max": frame["date"].max().date().isoformat() if not frame.empty else None,
    }


def checksum_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_parquet(path: Path, duplicate_keys: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(ROOT_DIR))}
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT_DIR)),
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "date_min": frame["date"].min().date().isoformat() if "date" in frame.columns and not frame.empty else None,
        "date_max": frame["date"].max().date().isoformat() if "date" in frame.columns and not frame.empty else None,
        "duplicate_count": int(frame.duplicated(duplicate_keys).sum()) if all(key in frame.columns for key in duplicate_keys) else None,
        "bytes": int(path.stat().st_size),
        "sha256": checksum_file(path),
    }


def validate_feature_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"status": "FAIL", "reason": "empty"}
    model_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    values = frame[model_columns].to_numpy(dtype=float)
    non_finite = int((~np.isfinite(values)).sum())
    ratio_metrics: dict[str, Any] = {}
    for column in ["open_ratio", "high_ratio", "low_ratio"]:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").abs().dropna()
            ratio_metrics[column] = {
                "p99": None if series.empty else float(series.quantile(0.99)),
                "max": None if series.empty else float(series.max()),
            }
    return {
        "status": "PASS" if non_finite == 0 else "FAIL",
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "feature_non_finite_count": non_finite,
        "ratio_sanity": ratio_metrics,
        "atr_ratio_coverage": float(frame["atr_ratio"].notna().mean()) if "atr_ratio" in frame.columns else None,
        "has_macro_rate": float(frame["has_macro"].mean()) if "has_macro" in frame.columns else None,
        "has_breadth_rate": float(frame["has_breadth"].mean()) if "has_breadth" in frame.columns else None,
        "has_fundamentals_rate": float(frame["has_fundamentals"].mean()) if "has_fundamentals" in frame.columns else None,
    }


def write_failed_tickers(path: Path, failed: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if failed.empty:
        pd.DataFrame(columns=["ticker", "yfinance_symbol", "company_name", "sector", "industry", "stage", "reason"]).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        failed.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    final = metrics["final_decision"]
    report = f"""# CP142-D 500티커 local yfinance dataset bootstrap 보고서

## 1. 요약

판정은 **{final["status"]}**이다.

{final["summary"]}

## 2. universe

- 입력 파일: `backend/data/universe/sp500.csv`
- 요청 ticker 수: {metrics["universe"]["ticker_count"]}
- yfinance symbol 변환 예시: `BRK.B -> BRK-B`

## 3. yfinance preflight

| 항목 | 값 |
|---|---|
| status | {metrics["preflight"]["status"]} |
| reason | {metrics["preflight"]["reason"]} |
| any_429 | {metrics["preflight"]["any_429"]} |
| any_json_ok | {metrics["preflight"]["any_json_ok"]} |

## 4. 생성 파일

| 파일 | exists | rows | tickers | date_min | date_max |
|---|---:|---:|---:|---|---|
| price 1D 500 | {metrics["files"]["price_1d"].get("exists")} | {metrics["files"]["price_1d"].get("rows")} | {metrics["files"]["price_1d"].get("ticker_count")} | {metrics["files"]["price_1d"].get("date_min")} | {metrics["files"]["price_1d"].get("date_max")} |
| indicators 1D 500 | {metrics["files"]["indicators_1d"].get("exists")} | {metrics["files"]["indicators_1d"].get("rows")} | {metrics["files"]["indicators_1d"].get("ticker_count")} | {metrics["files"]["indicators_1d"].get("date_min")} | {metrics["files"]["indicators_1d"].get("date_max")} |
| indicators 1W 500 | {metrics["files"]["indicators_1w"].get("exists")} | {metrics["files"]["indicators_1w"].get("rows")} | {metrics["files"]["indicators_1w"].get("ticker_count")} | {metrics["files"]["indicators_1w"].get("date_min")} | {metrics["files"]["indicators_1w"].get("date_max")} |

## 5. 검증

- adjusted OHLC violation: {metrics["validation"].get("adjusted_ohlc_violation_count")}
- duplicate price: {metrics["validation"].get("price_duplicate_count")}
- 1D feature status: {metrics["validation"].get("feature_1d", {}).get("status")}
- 1W feature status: {metrics["validation"].get("feature_1w", {}).get("status")}
- 1D split gate: {metrics["validation"].get("split_gate_1d", {}).get("status")}
- 1W split gate: {metrics["validation"].get("split_gate_1w", {}).get("status")}

## 6. 실패 ticker

- 실패 ticker 수: {metrics["failed_tickers"]["count"]}
- 실패 CSV: `docs/cp142_500ticker_failed_tickers.csv`
- 대표 실패 사유: {metrics["failed_tickers"].get("reason_counts")}

## 7. 금지 작업 미발생 확인

DB write, Supabase price_data/indicators 대량 read/write, EODHD 호출, 모델 학습/inference, W&B, 프론트 수정은 수행하지 않았다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP142 500 ticker local yfinance dataset bootstrap")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=(date.today() + timedelta(days=1)).isoformat())
    parser.add_argument("--current-date", default=date.today().isoformat())
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--preflight-tickers", nargs="*", default=DEFAULT_PREFLIGHT_TICKERS)
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp142_500ticker_local_dataset_bootstrap_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp142_500ticker_local_dataset_bootstrap_report.md"))
    parser.add_argument("--failed-tickers-path", default=str(DOCS_DIR / "cp142_500ticker_failed_tickers.csv"))
    parser.add_argument("--manifest-path", default=str(DOCS_DIR / "cp142_500ticker_dataset_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_sp500_universe()
    current_date = pd.to_datetime(args.current_date).date()
    preflight = preflight_yahoo([ticker.replace(".", "-") for ticker in args.preflight_tickers])

    failed = pd.DataFrame()
    price = pd.DataFrame()
    indicators_1d = pd.DataFrame()
    indicators_1w = pd.DataFrame()
    price_metrics: dict[str, Any] = {}
    context_metrics: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "adjusted_ohlc_violation_count": None,
        "price_duplicate_count": None,
        "feature_1d": {"status": "NOT_RUN"},
        "feature_1w": {"status": "NOT_RUN"},
        "split_gate_1d": {"status": "NOT_RUN"},
        "split_gate_1w": {"status": "NOT_RUN"},
    }
    files_written = False

    if preflight["status"] != "PASS":
        failed = universe.copy()
        failed["stage"] = "yfinance_preflight"
        failed["reason"] = preflight["status"]
        final_status = "FAIL"
        final_summary = "Yahoo preflight가 통과하지 못해 500티커 대량 수집을 안전 중단했다. 현재 상태에서 yfinance bootstrap을 진행하면 429를 확대할 위험이 있다."
    else:
        price, failed, price_metrics = fetch_prices(
            universe,
            start_date=args.start_date,
            end_date=args.end_date,
            current_date=current_date,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
        )
        if price_metrics.get("aborted"):
            final_status = "FAIL"
            final_summary = f"수집 도중 안전 중단했다: {price_metrics.get('abort_reason')}"
        elif price.empty:
            final_status = "FAIL"
            final_summary = "가격 parquet을 생성할 수 없어 dataset bootstrap에 실패했다."
        else:
            stock_info = universe[["ticker", "company_name", "sector", "industry"]].copy()
            stock_info["market_cap"] = np.nan
            stock_info["source"] = "sp500_csv"
            stock_info["provider"] = "local_universe"

            macro, macro_metrics = fetch_fred_macro(args.start_date, current_date.isoformat())
            breadth, breadth_metrics = build_market_breadth_context(price, min_ticker_count=max(50, int(price["ticker"].nunique() * 0.4)))
            sector, sector_metrics = build_sector_context(price, stock_info)
            fundamentals, fundamentals_metrics = fetch_edgar_fundamentals_context(
                sorted(price["ticker"].unique().tolist())[: min(100, int(price["ticker"].nunique()))],
                max_tickers=min(100, int(price["ticker"].nunique())),
                delay_seconds=0.12,
            )
            context_metrics = {
                "macro": macro_metrics,
                "market_breadth": breadth_metrics,
                "sector_returns": sector_metrics,
                "fundamentals": fundamentals_metrics,
            }
            indicators_1d = build_features(price_df=price, macro_df=macro, breadth_df=breadth, fundamentals_df=fundamentals, timeframe="1D")
            indicators_1w = build_features(price_df=price, macro_df=macro, breadth_df=breadth, fundamentals_df=fundamentals, timeframe="1W")
            for frame, timeframe in [(indicators_1d, "1D"), (indicators_1w, "1W")]:
                if not frame.empty:
                    frame["source"] = "yfinance"
                    frame["provider"] = "yfinance"
                    frame["updated_at"] = utc_now_iso()
                    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
                    frame["context_hash"] = frame_checksum(pd.concat([macro, breadth, fundamentals], ignore_index=True, sort=False), ["ticker", "date", "filing_date", "us10y", "nh_nl_index", "revenue"])

            price_1w = resample_price_frame(price, "1W")
            if not price_1w.empty:
                price_1w["source"] = "yfinance"
                price_1w["provider"] = "yfinance"
                price_1w["provider_adjustment_policy"] = provider_adjustment_policy("yfinance")
                price_1w["updated_at"] = utc_now_iso()
                price_1w["date"] = pd.to_datetime(price_1w["date"]).dt.strftime("%Y-%m-%d")

            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            CONTEXT_500_DIR.mkdir(parents=True, exist_ok=True)
            price.to_parquet(PRICE_500_PATH, index=False)
            price_1w.to_parquet(PRICE_1W_500_PATH, index=False)
            indicators_1d.to_parquet(INDICATOR_1D_500_PATH, index=False)
            indicators_1w.to_parquet(INDICATOR_1W_500_PATH, index=False)
            stock_info.to_parquet(STOCK_INFO_500_PATH, index=False)
            macro.to_parquet(CONTEXT_500_DIR / "macroeconomic_indicators_500.parquet", index=False)
            breadth.to_parquet(CONTEXT_500_DIR / "market_breadth_500.parquet", index=False)
            sector.to_parquet(CONTEXT_500_DIR / "sector_returns_500.parquet", index=False)
            fundamentals.to_parquet(CONTEXT_500_DIR / "company_fundamentals_500.parquet", index=False)
            files_written = True

            validation = {
                "adjusted_ohlc_violation_count": int(sum(1 for item in price_metrics.get("ticker_metrics", {}).values() if item.get("status") != "PASS")),
                "price_duplicate_count": int(price.duplicated(["ticker", "date", "source"]).sum()),
                "feature_1d": validate_feature_frame(indicators_1d),
                "feature_1w": validate_feature_frame(indicators_1w),
                "indicator_1d_checksum": indicator_value_checksum(indicators_1d),
                "indicator_1w_checksum": indicator_value_checksum(indicators_1w),
                "context_checksum": frame_checksum(pd.concat([macro, breadth, fundamentals], ignore_index=True, sort=False), ["ticker", "date", "filing_date", "us10y", "nh_nl_index", "revenue"]),
                "split_gate_1d": {"status": "MANUAL_REQUIRED", "reason": "500 파일은 canonical local snapshot filename과 분리되어 있어 별도 snapshot dir gate가 필요하다."},
                "split_gate_1w": {"status": "MANUAL_REQUIRED", "reason": "500 파일은 canonical local snapshot filename과 분리되어 있어 별도 snapshot dir gate가 필요하다."},
            }
            enough_tickers = int(price["ticker"].nunique()) >= 450
            features_ok = validation["feature_1d"]["status"] == "PASS" and validation["feature_1w"]["status"] == "PASS"
            if enough_tickers and features_ok and failed.empty:
                final_status = "PASS"
                final_summary = "500티커 local yfinance dataset bootstrap이 완료됐다."
            elif enough_tickers and features_ok:
                final_status = "WARN"
                final_summary = "충분한 ticker coverage와 feature sanity는 확보했지만 일부 ticker 실패가 남았다."
            else:
                final_status = "FAIL"
                final_summary = "500티커 dataset coverage 또는 feature sanity가 기준에 미달했다."

    failed_path = Path(args.failed_tickers_path)
    write_failed_tickers(failed_path, failed)

    reason_counts = {}
    if not failed.empty and "reason" in failed.columns:
        reason_counts = failed["reason"].astype(str).value_counts().to_dict()

    files = {
        "price_1d": summarize_parquet(PRICE_500_PATH, ["ticker", "date", "source"]),
        "price_1w": summarize_parquet(PRICE_1W_500_PATH, ["ticker", "date", "source"]),
        "indicators_1d": summarize_parquet(INDICATOR_1D_500_PATH, ["ticker", "timeframe", "date", "source"]),
        "indicators_1w": summarize_parquet(INDICATOR_1W_500_PATH, ["ticker", "timeframe", "date", "source"]),
        "stock_info": summarize_parquet(STOCK_INFO_500_PATH, ["ticker"]),
    }

    manifest = {
        "cp": "CP142-D",
        "created_at": utc_now_iso(),
        "status": final_status,
        "universe_path": str(UNIVERSE_PATH.relative_to(ROOT_DIR)),
        "requested_ticker_count": int(len(universe)),
        "provider": "yfinance",
        "source": "yfinance",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "current_date_gate": current_date.isoformat(),
        "files_written": files_written,
        "files": files,
        "preflight": preflight,
        "validation": validation,
        "failed_tickers_csv": str(failed_path.relative_to(ROOT_DIR)),
    }
    manifest_path = Path(args.manifest_path)
    write_json(manifest_path, manifest)

    metrics = {
        **manifest,
        "universe": {
            "ticker_count": int(len(universe)),
            "sector_count": int(universe["sector"].nunique()),
            "industry_count": int(universe["industry"].nunique()),
        },
        "price_fetch": price_metrics,
        "context": context_metrics,
        "failed_tickers": {
            "count": int(len(failed)),
            "reason_counts": reason_counts,
            "path": str(failed_path.relative_to(ROOT_DIR)),
        },
        "forbidden_actions_observed": {
            "model_training": False,
            "inference": False,
            "db_write": False,
            "supabase_price_indicators_bulk_read_write": False,
            "eodhd_call": False,
            "frontend_modify": False,
            "wandb": False,
        },
        "final_decision": {
            "status": final_status,
            "summary": final_summary,
        },
    }
    metrics_path = Path(args.metrics_path)
    report_path = Path(args.report_path)
    write_json(metrics_path, metrics)
    write_json(LOG_DIR / metrics_path.name, metrics)
    write_report(report_path, metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
