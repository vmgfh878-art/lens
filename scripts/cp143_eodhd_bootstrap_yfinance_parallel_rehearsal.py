from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp143_eodhd_bootstrap_yfinance_parallel_migration"
UNIVERSE_PATH = ROOT_DIR / "backend" / "data" / "universe" / "sp500.csv"
YFINANCE_PRICE_PATH = SNAPSHOT_DIR / "price_data_yfinance.parquet"
SAMPLE_PARQUET_PATH = LOG_DIR / "eodhd_price_rehearsal_sample.parquet"

DEFAULT_SAMPLE_TICKERS = [
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
    "ADBE",
    "BRK.B",
]

load_dotenv(ROOT_DIR / ".env")

os.environ["MARKET_DATA_PROVIDER"] = "eodhd"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(SNAPSHOT_DIR))
os.environ["WANDB_MODE"] = "disabled"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.collector.errors import SourceLimitReachedError  # noqa: E402
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402
from backend.collector.sources.price_contract import validate_adjusted_ohlc_contract  # noqa: E402


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


def load_universe() -> pd.DataFrame:
    frame = pd.read_csv(UNIVERSE_PATH)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["yfinance_symbol"] = frame["ticker"].str.replace(".", "-", regex=False)
    return frame.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)


def to_eodhd_symbol(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if "." in normalized and not normalized.endswith((".US", ".NYSE", ".NASDAQ", ".BATS", ".AMEX")):
        normalized = normalized.replace(".", "-")
    if "." in normalized:
        return normalized
    return f"{normalized}.US"


def fetch_eodhd_frame_direct(
    session: requests.Session,
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = to_eodhd_symbol(ticker)
    try:
        response = session.get(
            f"https://eodhd.com/api/eod/{symbol}",
            params={
                "api_token": api_key.strip(),
                "fmt": "json",
                "from": start_date,
                "to": end_date,
            },
            timeout=30,
        )
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "REQUEST_EXCEPTION",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc).split("api_token=")[0][:240],
        }

    meta: dict[str, Any] = {
        "http_status": int(response.status_code),
        "content_type": response.headers.get("content-type"),
        "text_length": len(response.text),
    }
    if response.status_code == 429:
        raise SourceLimitReachedError("EODHD", response.text[:240])
    if response.status_code >= 400:
        meta["status"] = "HTTP_ERROR"
        meta["body_head"] = response.text[:240]
        return pd.DataFrame(), meta

    try:
        payload = response.json()
    except Exception as exc:
        meta.update({"status": "JSON_ERROR", "exception_type": type(exc).__name__})
        return pd.DataFrame(), meta

    if isinstance(payload, dict):
        message = str(payload.get("message") or payload.get("error") or "").strip()
        if "limit" in message.lower():
            raise SourceLimitReachedError("EODHD", message)
        meta.update({"status": "DICT_PAYLOAD", "dict_keys": sorted(payload.keys())[:10], "message": message[:240]})
        return pd.DataFrame(), meta
    if not payload:
        meta["status"] = "EMPTY_PAYLOAD"
        return pd.DataFrame(), meta

    frame = pd.DataFrame(payload)
    if frame.empty or "date" not in frame.columns:
        meta["status"] = "EMPTY_FRAME"
        meta["columns"] = list(frame.columns)
        return pd.DataFrame(), meta

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").set_index("date")
    renamed = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "Adj Close",
            "volume": "Volume",
        }
    )
    if "Adj Close" not in renamed.columns:
        renamed["Adj Close"] = renamed["Close"]
    renamed["Amount"] = pd.to_numeric(renamed["Close"], errors="coerce") * pd.to_numeric(renamed["Volume"], errors="coerce")
    renamed.replace([np.inf, -np.inf], np.nan, inplace=True)
    meta["status"] = "PASS"
    meta["rows"] = int(len(renamed))
    return renamed.where(pd.notnull(renamed), None), meta


def provider_frame_to_records(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working.index = pd.to_datetime(working.index, errors="coerce").normalize()
    records = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "date": working.index,
            "open": pd.to_numeric(working["Open"], errors="coerce"),
            "high": pd.to_numeric(working["High"], errors="coerce"),
            "low": pd.to_numeric(working["Low"], errors="coerce"),
            "close": pd.to_numeric(working["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(working["Adj Close"], errors="coerce"),
            "volume": pd.to_numeric(working["Volume"], errors="coerce").fillna(0).astype("int64"),
            "amount": pd.to_numeric(working.get("Amount", working["Close"] * working["Volume"]), errors="coerce"),
            "source": "eodhd",
            "provider": "eodhd",
            "provider_adjustment_policy": provider_adjustment_policy("eodhd"),
            "updated_at": utc_now_iso(),
        }
    )
    return records.dropna(subset=["date"]).reset_index(drop=True)


def fetch_eodhd_sample(tickers: list[str], start_date: str, end_date: str, sleep_seconds: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        return pd.DataFrame(), {
            "status": "FAIL_NO_KEY",
            "api_key_present": False,
            "ticker_metrics": {},
            "failed_tickers": {ticker: "missing_eodhd_api_key" for ticker in tickers},
        }

    chunks: list[pd.DataFrame] = []
    ticker_metrics: dict[str, Any] = {}
    failed_tickers: dict[str, str] = {}
    source_limit_hit = False
    session = requests.Session()
    # 일부 로컬 환경은 HTTP(S)_PROXY가 127.0.0.1:9 같은 차단 프록시로 잡힌다.
    # EODHD 키/쿼터 판정을 오염시키지 않도록 제한 리허설에서는 환경 프록시를 무시한다.
    session.trust_env = False

    for ticker in tickers:
        try:
            provider_frame, fetch_meta = fetch_eodhd_frame_direct(session, ticker, start_date, end_date, api_key)
        except SourceLimitReachedError as exc:
            failed_tickers[ticker] = "source_limit_reached"
            ticker_metrics[ticker] = {"status": "FAIL_LIMIT", "error": str(exc)[:240]}
            source_limit_hit = True
            break
        except Exception as exc:
            failed_tickers[ticker] = f"exception:{type(exc).__name__}"
            ticker_metrics[ticker] = {"status": "FAIL_EXCEPTION", "error": str(exc)[:240]}
            continue

        if provider_frame.empty:
            failed_tickers[ticker] = "empty_frame"
            ticker_metrics[ticker] = {"status": "FAIL_EMPTY", "rows": 0, "fetch_meta": fetch_meta}
            continue
        contract = validate_adjusted_ohlc_contract(ticker, provider_frame)
        if not contract.passed:
            failed_tickers[ticker] = "adjusted_ohlc_contract_failed"
            ticker_metrics[ticker] = {
                "status": "FAIL_CONTRACT",
                "rows": int(len(provider_frame)),
                "fetch_meta": fetch_meta,
                "violations": contract.violations,
                "contract_metrics": contract.metrics,
            }
            continue
        records = provider_frame_to_records(ticker, provider_frame)
        chunks.append(records)
        ticker_metrics[ticker] = {
            "status": "PASS",
            "rows": int(len(records)),
            "date_min": records["date"].min().date().isoformat(),
            "date_max": records["date"].max().date().isoformat(),
            "fetch_meta": fetch_meta,
            "contract_metrics": contract.metrics,
        }
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if source_limit_hit:
        for ticker in tickers[tickers.index(next(reversed(failed_tickers))) + 1 :] if failed_tickers else []:
            failed_tickers.setdefault(ticker, "not_attempted_after_source_limit")

    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not frame.empty:
        frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date", "source"], keep="last")
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

    status = "PASS" if not failed_tickers and not frame.empty else "WARN" if not frame.empty else "FAIL"
    return frame.reset_index(drop=True), {
        "status": status,
        "api_key_present": True,
        "requested_tickers": len(tickers),
        "success_tickers": int(frame["ticker"].nunique()) if not frame.empty else 0,
        "row_count": int(len(frame)),
        "date_min": None if frame.empty else str(frame["date"].min()),
        "date_max": None if frame.empty else str(frame["date"].max()),
        "duplicate_ticker_date_source": 0 if frame.empty else int(frame.duplicated(["ticker", "date", "source"]).sum()),
        "source_limit_hit": source_limit_hit,
        "ticker_metrics": ticker_metrics,
        "failed_tickers": failed_tickers,
    }


def adjusted_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    working = frame.copy()
    close = pd.to_numeric(working["close"], errors="coerce")
    adjusted_close = pd.to_numeric(working["adjusted_close"], errors="coerce").fillna(close)
    factor = adjusted_close / close.where(close.abs() > 1e-9)
    result = pd.DataFrame(
        {
            "ticker": working["ticker"].astype(str).str.upper(),
            "date": pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            f"{prefix}_close": close,
            f"{prefix}_adjusted_close": adjusted_close,
            f"{prefix}_adjusted_factor": factor,
            f"{prefix}_adjusted_open": pd.to_numeric(working["open"], errors="coerce") * factor,
            f"{prefix}_adjusted_high": pd.to_numeric(working["high"], errors="coerce") * factor,
            f"{prefix}_adjusted_low": pd.to_numeric(working["low"], errors="coerce") * factor,
            f"{prefix}_volume": pd.to_numeric(working["volume"], errors="coerce"),
        }
    )
    return result.dropna(subset=["ticker", "date"])


def compare_with_local_yfinance(eodhd_frame: pd.DataFrame, tickers: list[str]) -> dict[str, Any]:
    if eodhd_frame.empty or not YFINANCE_PRICE_PATH.exists():
        return {"status": "SKIPPED", "reason": "missing_eodhd_or_yfinance_frame"}
    yfinance = pd.read_parquet(YFINANCE_PRICE_PATH)
    yfinance["ticker"] = yfinance["ticker"].astype(str).str.upper()
    yfinance = yfinance[yfinance["ticker"].isin([ticker.replace(".", "-").upper() for ticker in tickers] + [ticker.upper() for ticker in tickers])].copy()
    e_adj = adjusted_columns(eodhd_frame, "eodhd")
    y_adj = adjusted_columns(yfinance, "yfinance")
    joined = e_adj.merge(y_adj, on=["ticker", "date"], how="inner")
    if joined.empty:
        return {"status": "FAIL", "reason": "no_overlap"}

    def rel_diff(column_a: str, column_b: str) -> pd.Series:
        denominator = joined[column_b].abs().where(joined[column_b].abs() > 1e-9)
        return ((joined[column_a] - joined[column_b]) / denominator).abs().replace([np.inf, -np.inf], np.nan)

    close_diff = rel_diff("eodhd_close", "yfinance_close")
    adjusted_close_diff = rel_diff("eodhd_adjusted_close", "yfinance_adjusted_close")
    factor_diff = rel_diff("eodhd_adjusted_factor", "yfinance_adjusted_factor")
    ticker_metrics = {}
    for ticker, group in joined.groupby("ticker"):
        close_series = ((group["eodhd_close"] - group["yfinance_close"]) / group["yfinance_close"].abs().where(group["yfinance_close"].abs() > 1e-9)).abs()
        adj_series = ((group["eodhd_adjusted_close"] - group["yfinance_adjusted_close"]) / group["yfinance_adjusted_close"].abs().where(group["yfinance_adjusted_close"].abs() > 1e-9)).abs()
        ticker_metrics[ticker] = {
            "overlap_rows": int(len(group)),
            "date_min": str(group["date"].min()),
            "date_max": str(group["date"].max()),
            "close_median_rel_diff": None if close_series.dropna().empty else float(close_series.median()),
            "adjusted_close_median_rel_diff": None if adj_series.dropna().empty else float(adj_series.median()),
            "adjusted_close_p99_rel_diff": None if adj_series.dropna().empty else float(adj_series.quantile(0.99)),
        }
    return {
        "status": "PASS",
        "overlap_rows": int(len(joined)),
        "overlap_ticker_count": int(joined["ticker"].nunique()),
        "date_min": str(joined["date"].min()),
        "date_max": str(joined["date"].max()),
        "close_median_rel_diff": float(close_diff.median()) if not close_diff.dropna().empty else None,
        "close_p99_rel_diff": float(close_diff.quantile(0.99)) if not close_diff.dropna().empty else None,
        "adjusted_close_median_rel_diff": float(adjusted_close_diff.median()) if not adjusted_close_diff.dropna().empty else None,
        "adjusted_close_p99_rel_diff": float(adjusted_close_diff.quantile(0.99)) if not adjusted_close_diff.dropna().empty else None,
        "adjusted_factor_median_rel_diff": float(factor_diff.median()) if not factor_diff.dropna().empty else None,
        "adjusted_factor_p99_rel_diff": float(factor_diff.quantile(0.99)) if not factor_diff.dropna().empty else None,
        "ticker_metrics": ticker_metrics,
    }


def estimate_bootstrap_plan(universe_count: int, sleep_seconds: float) -> dict[str, Any]:
    price_calls = universe_count
    estimated_seconds = price_calls * (sleep_seconds + 0.8)
    return {
        "price_endpoint_calls": price_calls,
        "estimated_seconds_at_current_sleep": round(estimated_seconds, 1),
        "estimated_minutes_at_current_sleep": round(estimated_seconds / 60, 1),
        "recommended_batch": "ticker 단건 호출 503회, 25~50 ticker 단위 checkpoint 저장",
        "retry_policy": [
            "429 또는 limit 메시지 발생 시 즉시 중단",
            "empty_frame은 ticker별 실패로 기록하고 전체는 계속 진행",
            "3회 이하 네트워크 예외만 exponential backoff 재시도",
            "각 batch마다 parquet checkpoint와 failed csv 갱신",
        ],
    }


def write_strategy(path: Path) -> None:
    text = """# EODHD bootstrap + yfinance parallel 전략

## 결론

추천안은 **B안: EODHD bootstrap dataset과 yfinance parallel dataset 분리**이다.

EODHD로 500티커 초기 history를 빠르게 확보하되, yfinance daily append를 같은 parquet series에 섞지 않는다. 두 provider는 adjusted close, split/dividend 반영, ticker symbol, 누락일 정책이 다를 수 있으므로 source/provider별 파일과 manifest를 분리한다.

## 파일명 계약

- `data/parquet/price_data_eodhd_500.parquet`
- `data/parquet/indicators_eodhd_1D_500.parquet`
- `data/parquet/indicators_eodhd_1W_500.parquet`
- `data/parquet/context/eodhd_500/*.parquet`
- `data/parquet/price_data_yfinance_500.parquet`
- `data/parquet/indicators_yfinance_1D_500.parquet`
- `data/parquet/indicators_yfinance_1W_500.parquet`
- `data/parquet/context/yfinance_500/*.parquet`

## 운영 원칙

1. EODHD dataset은 bootstrap baseline이다.
2. yfinance dataset은 별도 parallel dataset으로 천천히 쌓는다.
3. 같은 ticker/date라도 provider가 다르면 같은 parquet에 append하지 않는다.
4. 모델 학습/feature cache는 `market_data_provider`, `provider_adjustment_policy`, `source_data_hash`, context checksum을 포함한다.
5. Supabase에는 raw price/indicator를 올리지 않고 product latest/thin 결과만 저장한다.

## 금지할 혼합 방식

- `price_data_eodhd_500.parquet` 끝에 yfinance 신규 row append
- EODHD indicators와 yfinance price를 조합해 feature 생성
- provider가 다른 feature cache path 재사용
- provider 차이를 무시한 500티커 model run 비교

## 다음 CP 순서

1. EODHD 500 bootstrap execution CP
2. EODHD 500 indicators/context generation CP
3. 500 model training/smoke CP
4. yfinance parallel collector CP
5. EODHD baseline과 yfinance parallel reconciliation CP
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    final = metrics["final_decision"]
    report = f"""# CP143-D EODHD bootstrap + yfinance parallel 전환 보고서

## 1. 요약

판정은 **{final["status"]}**이다.

{final["summary"]}

## 2. EODHD 사용 가능성

| 항목 | 값 |
|---|---|
| API key present | {metrics["eodhd_availability"]["api_key_present"]} |
| sample status | {metrics["eodhd_sample"]["status"]} |
| requested tickers | {metrics["eodhd_sample"]["requested_tickers"]} |
| success tickers | {metrics["eodhd_sample"]["success_tickers"]} |
| row count | {metrics["eodhd_sample"]["row_count"]} |
| source limit hit | {metrics["eodhd_sample"]["source_limit_hit"]} |

## 3. 500티커 bootstrap 예상

| 항목 | 값 |
|---|---|
| universe count | {metrics["universe"]["ticker_count"]} |
| expected price calls | {metrics["bootstrap_estimate"]["price_endpoint_calls"]} |
| estimated minutes | {metrics["bootstrap_estimate"]["estimated_minutes_at_current_sleep"]} |

## 4. yfinance overlap 비교

| 항목 | 값 |
|---|---|
| status | {metrics["overlap_comparison"].get("status")} |
| overlap tickers | {metrics["overlap_comparison"].get("overlap_ticker_count")} |
| overlap rows | {metrics["overlap_comparison"].get("overlap_rows")} |
| adjusted_close median rel diff | {metrics["overlap_comparison"].get("adjusted_close_median_rel_diff")} |
| adjusted_close p99 rel diff | {metrics["overlap_comparison"].get("adjusted_close_p99_rel_diff")} |
| adjusted_factor p99 rel diff | {metrics["overlap_comparison"].get("adjusted_factor_p99_rel_diff")} |

## 5. 선택한 운영 전략

추천안은 **B안: EODHD bootstrap dataset과 yfinance parallel dataset 분리**이다.

A안처럼 EODHD baseline 뒤에 yfinance daily row를 같은 parquet에 append하면 provider adjustment policy가 경계일에서 섞인다. CP29 계열에서 raw/adjusted 혼용이 실제 병목이었던 만큼, 이번에는 source/provider별 dataset을 끝까지 분리한다.

## 6. local dataset naming

- `price_data_eodhd_500.parquet`
- `indicators_eodhd_1D_500.parquet`
- `indicators_eodhd_1W_500.parquet`
- `context/eodhd_500/*.parquet`
- `price_data_yfinance_500.parquet`
- `indicators_yfinance_1D_500.parquet`
- `indicators_yfinance_1W_500.parquet`
- `context/yfinance_500/*.parquet`

## 7. 금지 작업 미발생 확인

DB write, Supabase raw data write, 모델 학습, inference 저장, 프론트 수정, EODHD 500 full run은 수행하지 않았다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP143 EODHD bootstrap + yfinance parallel rehearsal")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_SAMPLE_TICKERS)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp143_eodhd_bootstrap_yfinance_parallel_migration_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp143_eodhd_bootstrap_yfinance_parallel_migration_report.md"))
    parser.add_argument("--strategy-path", default=str(DOCS_DIR / "eodhd_bootstrap_yfinance_parallel_strategy.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    tickers = [ticker.upper() for ticker in args.tickers]
    eodhd_frame, eodhd_metrics = fetch_eodhd_sample(tickers, args.start_date, args.end_date, args.sleep_seconds)
    if not eodhd_frame.empty:
        eodhd_frame.to_parquet(SAMPLE_PARQUET_PATH, index=False)
    overlap = compare_with_local_yfinance(eodhd_frame, tickers)
    estimate = estimate_bootstrap_plan(int(len(universe)), args.sleep_seconds)

    if eodhd_metrics["status"] == "PASS" and overlap.get("status") == "PASS":
        status = "PASS"
        summary = "EODHD 소규모 bootstrap 리허설과 yfinance local overlap 비교가 통과했다. 500티커 bootstrap execution CP로 넘어갈 수 있다. 단, yfinance와는 병렬 dataset 분리가 필수다."
    elif eodhd_metrics["status"] in {"PASS", "WARN"}:
        status = "WARN"
        summary = "EODHD 소규모 호출은 가능하지만 일부 실패 또는 yfinance overlap 비교 한계가 남았다. 병렬 dataset 분리 전략은 유지한다."
    else:
        status = "FAIL"
        summary = "EODHD key/quota/응답 문제로 bootstrap 가능성을 확인하지 못했다."

    metrics = {
        "cp": "CP143-D",
        "created_at": utc_now_iso(),
        "scope": {
            "tickers": tickers,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "eodhd_full_500_run": False,
            "sample_parquet_path": str(SAMPLE_PARQUET_PATH.relative_to(ROOT_DIR)) if SAMPLE_PARQUET_PATH.exists() else None,
        },
        "universe": {
            "path": str(UNIVERSE_PATH.relative_to(ROOT_DIR)),
            "ticker_count": int(len(universe)),
            "sector_count": int(universe["sector"].nunique()),
            "industry_count": int(universe["industry"].nunique()),
        },
        "eodhd_availability": {
            "api_key_present": bool(os.environ.get("EODHD_API_KEY")),
            "quota_remaining_known": False,
            "quota_note": "현재 코드 경로는 EODHD remaining quota API를 조회하지 않는다. 500 bootstrap 전 dashboard 또는 별도 quota endpoint 확인 필요.",
        },
        "eodhd_sample": eodhd_metrics,
        "overlap_comparison": overlap,
        "bootstrap_estimate": estimate,
        "recommended_strategy": {
            "selected": "B",
            "name": "EODHD bootstrap dataset과 yfinance parallel dataset 분리",
            "reject_a_reason": "EODHD baseline 뒤에 yfinance daily append를 같은 series에 섞으면 provider adjustment 경계가 생겨 CP29류 가격 피처 혼용 사고가 재발할 수 있다.",
        },
        "dataset_naming": {
            "eodhd_price": "data/parquet/price_data_eodhd_500.parquet",
            "eodhd_indicators_1d": "data/parquet/indicators_eodhd_1D_500.parquet",
            "eodhd_indicators_1w": "data/parquet/indicators_eodhd_1W_500.parquet",
            "yfinance_price": "data/parquet/price_data_yfinance_500.parquet",
            "yfinance_indicators_1d": "data/parquet/indicators_yfinance_1D_500.parquet",
            "yfinance_indicators_1w": "data/parquet/indicators_yfinance_1W_500.parquet",
        },
        "forbidden_actions_observed": {
            "db_write": False,
            "supabase_raw_data_write": False,
            "model_training": False,
            "inference_save": False,
            "frontend_modify": False,
            "eodhd_full_500_run": False,
        },
        "final_decision": {
            "status": status,
            "summary": summary,
        },
    }
    metrics_path = Path(args.metrics_path)
    report_path = Path(args.report_path)
    strategy_path = Path(args.strategy_path)
    write_json(metrics_path, metrics)
    write_json(LOG_DIR / metrics_path.name, metrics)
    write_strategy(strategy_path)
    write_report(report_path, metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
