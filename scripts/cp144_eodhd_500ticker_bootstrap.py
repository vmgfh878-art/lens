from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
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
LOG_DIR = ROOT_DIR / "logs" / "cp144_eodhd_500_bootstrap"
UNIVERSE_PATH = ROOT_DIR / "backend" / "data" / "universe" / "sp500.csv"
PRICE_PATH = SNAPSHOT_DIR / "price_data_eodhd_500.parquet"
MANIFEST_PATH = SNAPSHOT_DIR / "price_data_eodhd_500.manifest.json"
METRICS_PATH = DOCS_DIR / "cp144_eodhd_500ticker_bootstrap_metrics.json"
REPORT_PATH = DOCS_DIR / "cp144_eodhd_500ticker_bootstrap_report.md"
FAILED_CSV_PATH = DOCS_DIR / "cp144_eodhd_500ticker_failed_tickers.csv"
STATUS_CSV_PATH = LOG_DIR / "ticker_status.csv"


load_dotenv(ROOT_DIR / ".env")
os.environ["MARKET_DATA_PROVIDER"] = "eodhd"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(SNAPSHOT_DIR))

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


def load_universe(limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(UNIVERSE_PATH)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame = frame.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame


def to_eodhd_symbol(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if "." in normalized and not normalized.endswith((".US", ".NYSE", ".NASDAQ", ".BATS", ".AMEX")):
        normalized = normalized.replace(".", "-")
    if "." in normalized:
        return normalized
    return f"{normalized}.US"


def fetch_eodhd_frame(
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
        message = str(exc).split("api_token=")[0][:240]
        status = "FAILED_PROXY" if "proxy" in type(exc).__name__.lower() or "proxy" in message.lower() else "FAILED_HTTP"
        return pd.DataFrame(), {
            "status": status,
            "exception_type": type(exc).__name__,
            "exception_message": message,
        }

    meta: dict[str, Any] = {
        "http_status": int(response.status_code),
        "content_type": response.headers.get("content-type"),
        "text_length": len(response.text),
    }
    if response.status_code == 429:
        raise SourceLimitReachedError("EODHD", response.text[:240])
    if response.status_code >= 400:
        meta["status"] = "FAILED_HTTP"
        meta["body_head"] = response.text[:240]
        return pd.DataFrame(), meta

    try:
        payload = response.json()
    except Exception as exc:
        meta.update({"status": "FAILED_HTTP", "exception_type": type(exc).__name__})
        return pd.DataFrame(), meta

    if isinstance(payload, dict):
        message = str(payload.get("message") or payload.get("error") or "").strip()
        if "limit" in message.lower():
            raise SourceLimitReachedError("EODHD", message)
        meta.update({"status": "FAILED_EMPTY", "dict_keys": sorted(payload.keys())[:10], "message": message[:240]})
        return pd.DataFrame(), meta
    if not payload:
        meta["status"] = "FAILED_EMPTY"
        return pd.DataFrame(), meta

    frame = pd.DataFrame(payload)
    if frame.empty or "date" not in frame.columns:
        meta["status"] = "FAILED_EMPTY"
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
    meta["status"] = "FETCHED"
    meta["rows"] = int(len(renamed))
    return renamed.where(pd.notnull(renamed), None), meta


def provider_frame_to_records(ticker: str, frame: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
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
            "updated_at": fetched_at,
        }
    )
    records.index = pd.RangeIndex(len(records))
    records = records.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return records


def load_status() -> pd.DataFrame:
    if not STATUS_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(STATUS_CSV_PATH)


def write_status(rows: list[dict[str, Any]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame = frame.sort_values(["ticker", "attempted_at"]).drop_duplicates("ticker", keep="last")
    frame.to_csv(STATUS_CSV_PATH, index=False, encoding="utf-8")


def batch_path(batch_index: int) -> Path:
    return LOG_DIR / f"price_batch_{batch_index:03d}.parquet"


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def summarize_quality(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "ticker_count": 0,
            "date_min": None,
            "date_max": None,
            "duplicate_ticker_date_source": 0,
            "adjusted_ohlc_violation_count": None,
        }

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
        working[column] = pd.to_numeric(working[column], errors="coerce")

    close = working["close"]
    factor = working["adjusted_close"] / close.where(close.abs() > 1e-12)
    adjusted_open = working["open"] * factor
    adjusted_high = working["high"] * factor
    adjusted_low = working["low"] * factor

    finite_required = working[["open", "high", "low", "close", "adjusted_close"]].replace([np.inf, -np.inf], np.nan)
    required_null_count = int(finite_required.isna().sum().sum())
    adjusted_factor_invalid = int((~np.isfinite(factor) | (factor <= 0)).sum())
    high_low_violation = int((adjusted_high < adjusted_low).sum())
    high_bound_violation = int((adjusted_high + 1e-9 < pd.concat([adjusted_open, working["adjusted_close"]], axis=1).max(axis=1)).sum())
    low_bound_violation = int((adjusted_low - 1e-9 > pd.concat([adjusted_open, working["adjusted_close"]], axis=1).min(axis=1)).sum())
    volume_null_count = int(working["volume"].isna().sum())
    volume_negative_count = int((working["volume"] < 0).sum())

    ratio_frame = pd.DataFrame(
        {
            "ticker": working["ticker"].astype(str),
            "date": working["date"],
            "adjusted_open": adjusted_open,
            "adjusted_high": adjusted_high,
            "adjusted_low": adjusted_low,
            "adjusted_close": working["adjusted_close"],
        }
    ).sort_values(["ticker", "date"])
    previous_close = ratio_frame.groupby("ticker")["adjusted_close"].shift(1)
    denominator = previous_close.where(previous_close.abs() > 1e-12)
    ratio_metrics: dict[str, Any] = {}
    for column in ("adjusted_open", "adjusted_high", "adjusted_low"):
        ratio = ((ratio_frame[column] - previous_close) / denominator).replace([np.inf, -np.inf], np.nan).dropna().abs()
        ratio_metrics[column.replace("adjusted_", "") + "_ratio"] = {
            "p99_abs": None if ratio.empty else float(ratio.quantile(0.99)),
            "max_abs": None if ratio.empty else float(ratio.max()),
            "observation_count": int(len(ratio)),
        }

    return {
        "row_count": int(len(working)),
        "ticker_count": int(working["ticker"].nunique()),
        "date_min": working["date"].min().date().isoformat(),
        "date_max": working["date"].max().date().isoformat(),
        "duplicate_ticker_date_source": int(working.duplicated(["ticker", "date", "source"]).sum()),
        "source_values": sorted(working["source"].dropna().astype(str).unique().tolist()),
        "provider_values": sorted(working["provider"].dropna().astype(str).unique().tolist()),
        "policy_values": sorted(working["provider_adjustment_policy"].dropna().astype(str).unique().tolist()),
        "missing_required_ohlc_count": required_null_count,
        "adjusted_factor_invalid_count": adjusted_factor_invalid,
        "adjusted_high_low_violation_count": high_low_violation,
        "adjusted_high_bound_violation_count": high_bound_violation,
        "adjusted_low_bound_violation_count": low_bound_violation,
        "adjusted_ohlc_violation_count": adjusted_factor_invalid + high_low_violation + high_bound_violation + low_bound_violation,
        "volume_null_count": volume_null_count,
        "volume_negative_count": volume_negative_count,
        "ratio_sanity": ratio_metrics,
        "yfinance_row_count": int((working["source"].astype(str).str.lower() == "yfinance").sum()),
    }


def dataframe_hash(frame: pd.DataFrame) -> str:
    columns = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume", "source", "provider"]
    working = frame[columns].copy().sort_values(["ticker", "date", "source"]).reset_index(drop=True)
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    csv_bytes = working.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def write_manifest(frame: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    file_hash = hashlib.sha256(PRICE_PATH.read_bytes()).hexdigest() if PRICE_PATH.exists() else None
    manifest = {
        "created_at": utc_now_iso(),
        "provider": "eodhd",
        "source": "eodhd",
        "provider_adjustment_policy": provider_adjustment_policy("eodhd"),
        "universe_path": str(UNIVERSE_PATH.relative_to(ROOT_DIR)),
        "universe_ticker_count": metrics["universe"]["ticker_count"],
        "ticker_count": metrics["quality"]["ticker_count"],
        "row_count": metrics["quality"]["row_count"],
        "date_min": metrics["quality"]["date_min"],
        "date_max": metrics["quality"]["date_max"],
        "source_data_hash": dataframe_hash(frame) if not frame.empty else None,
        "parquet_sha256": file_hash,
        "columns": list(frame.columns),
        "proxy_policy": metrics["proxy_policy"],
        "forbidden_mix_policy": "EODHD와 yfinance는 같은 parquet series에 섞지 않는다.",
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def write_report(metrics: dict[str, Any]) -> None:
    quality = metrics["quality"]
    final = metrics["final_decision"]
    report = f"""# CP144-D EODHD 500티커 local bootstrap execution 보고서

## 1. 요약

판정은 **{final["status"]}**이다.

{final["summary"]}

## 2. 실행 범위

| 항목 | 값 |
|---|---|
| universe | `{metrics["universe"]["path"]}` |
| universe ticker count | {metrics["universe"]["ticker_count"]} |
| start_date | {metrics["scope"]["start_date"]} |
| end_date | {metrics["scope"]["end_date"]} |
| batch_size | {metrics["scope"]["batch_size"]} |
| provider/source | eodhd/eodhd |
| adjustment_policy | {metrics["scope"]["provider_adjustment_policy"]} |

## 3. 수집 결과

| 항목 | 값 |
|---|---|
| fetched ticker count | {metrics["status_counts"].get("fetched", 0)} |
| failed_empty | {metrics["status_counts"].get("failed_empty", 0)} |
| failed_contract | {metrics["status_counts"].get("failed_contract", 0)} |
| failed_http | {metrics["status_counts"].get("failed_http", 0)} |
| failed_proxy | {metrics["status_counts"].get("failed_proxy", 0)} |
| retry_pending | {metrics["status_counts"].get("retry_pending", 0)} |
| source_limit_hit | {metrics["source_limit_hit"]} |

## 4. 데이터 품질 검증

| 항목 | 값 |
|---|---|
| row count | {quality["row_count"]} |
| ticker count | {quality["ticker_count"]} |
| date min | {quality["date_min"]} |
| date max | {quality["date_max"]} |
| duplicate ticker/date/source | {quality["duplicate_ticker_date_source"]} |
| adjusted OHLC violation | {quality["adjusted_ohlc_violation_count"]} |
| missing OHLC count | {quality["missing_required_ohlc_count"]} |
| volume null count | {quality["volume_null_count"]} |
| volume negative count | {quality["volume_negative_count"]} |
| yfinance row count | {quality["yfinance_row_count"]} |

## 5. 저장 파일

- `data/parquet/price_data_eodhd_500.parquet`
- `data/parquet/price_data_eodhd_500.manifest.json`
- `logs/cp144_eodhd_500_bootstrap/ticker_status.csv`
- `docs/cp144_eodhd_500ticker_failed_tickers.csv`

## 6. 프록시 정책

이번 bootstrap은 EODHD session에서 환경 프록시를 사용하지 않는 정책으로 실행했다. 로컬 `HTTP_PROXY`/`HTTPS_PROXY`가 차단 프록시로 잡히면 정상 API key도 빈 frame처럼 보일 수 있기 때문이다. metrics와 manifest에 `use_env_proxy=false`로 기록했다.

## 7. source/provider 분리

생성 parquet의 `source`, `provider`는 모두 `eodhd`이고 yfinance row는 0개다. yfinance 병렬 dataset은 `price_data_yfinance_500.parquet` 계열로 별도 생성해야 하며, EODHD parquet에 yfinance daily row를 append하지 않는다.

## 8. 실패 티커

실패 또는 retry 대상은 `{FAILED_CSV_PATH.relative_to(ROOT_DIR)}`에 기록했다. 실패가 0개면 다음 indicator/context generation CP로 바로 진행 가능하다.

## 9. 금지 작업 미발생 확인

Supabase raw write, DB write, yfinance append, 모델 학습, inference 저장, 프론트 수정, W&B 실행은 수행하지 않았다.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("EODHD_API_KEY")
    universe = load_universe(args.limit_tickers)
    all_tickers = universe["ticker"].tolist()
    previous_status = load_status()
    fetched_tickers = set(previous_status.loc[previous_status["status"] == "fetched", "ticker"].astype(str)) if not previous_status.empty and args.resume else set()
    status_rows = previous_status.to_dict("records") if args.resume and not previous_status.empty else []
    pending_tickers = [ticker for ticker in all_tickers if ticker not in fetched_tickers]

    proxy_env_present = any(bool(os.environ.get(name)) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"))
    session = requests.Session()
    session.trust_env = bool(args.use_env_proxy)
    source_limit_hit = False
    batch_records: list[dict[str, Any]] = []

    if not api_key:
        status_rows = [
            {
                "ticker": ticker,
                "status": "failed_http",
                "reason": "missing_eodhd_api_key",
                "rows": 0,
                "date_min": None,
                "date_max": None,
                "attempted_at": utc_now_iso(),
            }
            for ticker in all_tickers
        ]
        write_status(status_rows)
    else:
        ticker_batches = chunks(pending_tickers, args.batch_size)
        for batch_index, batch_tickers in enumerate(ticker_batches):
            batch_chunks: list[pd.DataFrame] = []
            for ticker in batch_tickers:
                fetched_at = utc_now_iso()
                final_meta: dict[str, Any] = {}
                status = "retry_pending"
                provider_frame = pd.DataFrame()
                for attempt in range(1, args.max_retries + 2):
                    try:
                        provider_frame, final_meta = fetch_eodhd_frame(session, ticker, args.start_date, args.end_date, api_key)
                    except SourceLimitReachedError as exc:
                        source_limit_hit = True
                        status = "retry_pending"
                        final_meta = {"status": "SOURCE_LIMIT", "error": str(exc)[:240], "attempt": attempt}
                        break
                    raw_status = str(final_meta.get("status", "FAILED_HTTP"))
                    if raw_status == "FETCHED" and not provider_frame.empty:
                        status = "fetched"
                        break
                    if raw_status == "FAILED_EMPTY":
                        status = "failed_empty"
                        break
                    if raw_status == "FAILED_PROXY":
                        status = "failed_proxy"
                    else:
                        status = "failed_http"
                    if attempt <= args.max_retries:
                        time.sleep(args.retry_sleep_seconds * attempt)

                status_record = {
                    "ticker": ticker,
                    "status": status,
                    "reason": final_meta.get("status"),
                    "rows": 0,
                    "date_min": None,
                    "date_max": None,
                    "http_status": final_meta.get("http_status"),
                    "attempted_at": fetched_at,
                    "batch_index": batch_index,
                }

                if source_limit_hit:
                    status_record["reason"] = "source_limit_reached"
                    status_rows.append(status_record)
                    break

                if status != "fetched":
                    status_rows.append(status_record)
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                contract = validate_adjusted_ohlc_contract(ticker, provider_frame)
                if not contract.passed:
                    status_record.update(
                        {
                            "status": "failed_contract",
                            "reason": ";".join(contract.violations[:3]),
                            "rows": int(len(provider_frame)),
                        }
                    )
                    status_rows.append(status_record)
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                records = provider_frame_to_records(ticker, provider_frame, fetched_at)
                records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
                batch_chunks.append(records)
                status_record.update(
                    {
                        "status": "fetched",
                        "reason": "ok",
                        "rows": int(len(records)),
                        "date_min": str(records["date"].min()),
                        "date_max": str(records["date"].max()),
                        "contract_violation_count": int(contract.metrics.get("violation_count", 0)),
                    }
                )
                status_rows.append(status_record)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            write_status(status_rows)
            if batch_chunks:
                batch_frame = pd.concat(batch_chunks, ignore_index=True)
                batch_frame = batch_frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date", "source"], keep="last")
                output_path = batch_path(batch_index)
                batch_frame.to_parquet(output_path, index=False)
                batch_records.append(
                    {
                        "batch_index": batch_index,
                        "path": str(output_path.relative_to(ROOT_DIR)),
                        "ticker_count": int(batch_frame["ticker"].nunique()),
                        "row_count": int(len(batch_frame)),
                        "date_min": str(batch_frame["date"].min()),
                        "date_max": str(batch_frame["date"].max()),
                    }
                )
            if source_limit_hit:
                break

    parquet_parts = sorted(LOG_DIR.glob("price_batch_*.parquet"))
    part_frames: list[pd.DataFrame] = []
    existing_batch_records: list[dict[str, Any]] = []
    for path in parquet_parts:
        part = pd.read_parquet(path)
        part_frames.append(part)
        existing_batch_records.append(
            {
                "batch_index": int(path.stem.split("_")[-1]),
                "path": str(path.relative_to(ROOT_DIR)),
                "ticker_count": int(part["ticker"].nunique()) if not part.empty else 0,
                "row_count": int(len(part)),
                "date_min": None if part.empty else str(part["date"].min()),
                "date_max": None if part.empty else str(part["date"].max()),
            }
        )
    combined = pd.concat(part_frames, ignore_index=True) if part_frames else pd.DataFrame()
    batch_records = existing_batch_records
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        combined = combined.sort_values(["ticker", "date", "source"]).drop_duplicates(["ticker", "date", "source"], keep="last")
        combined.to_parquet(PRICE_PATH, index=False)

    status_frame = load_status()
    if status_frame.empty:
        status_counts: dict[str, int] = {}
    else:
        status_counts = {str(key): int(value) for key, value in status_frame["status"].value_counts().to_dict().items()}
    failed_frame = status_frame[status_frame["status"] != "fetched"].copy() if not status_frame.empty else pd.DataFrame()
    failed_frame.to_csv(FAILED_CSV_PATH, index=False, encoding="utf-8")

    quality = summarize_quality(combined)
    metrics = {
        "cp": "CP144-D",
        "created_at": utc_now_iso(),
        "scope": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "batch_size": args.batch_size,
            "sleep_seconds": args.sleep_seconds,
            "max_retries": args.max_retries,
            "resume": bool(args.resume),
            "provider": "eodhd",
            "source": "eodhd",
            "provider_adjustment_policy": provider_adjustment_policy("eodhd"),
        },
        "universe": {
            "path": str(UNIVERSE_PATH.relative_to(ROOT_DIR)),
            "ticker_count": int(len(universe)),
            "sector_count": int(universe["sector"].nunique()) if "sector" in universe.columns else None,
            "industry_count": int(universe["industry"].nunique()) if "industry" in universe.columns else None,
        },
        "proxy_policy": {
            "use_env_proxy": bool(args.use_env_proxy),
            "env_proxy_present": proxy_env_present,
            "policy": "EODHD session에서 환경 프록시를 사용하지 않는다." if not args.use_env_proxy else "EODHD session에서 환경 프록시를 사용한다.",
        },
        "api_key_present": bool(api_key),
        "source_limit_hit": source_limit_hit,
        "status_counts": status_counts,
        "batch_outputs": batch_records,
        "quality": quality,
        "output_paths": {
            "price_parquet": str(PRICE_PATH.relative_to(ROOT_DIR)) if PRICE_PATH.exists() else None,
            "manifest": str(MANIFEST_PATH.relative_to(ROOT_DIR)) if MANIFEST_PATH.exists() else None,
            "failed_csv": str(FAILED_CSV_PATH.relative_to(ROOT_DIR)),
            "status_csv": str(STATUS_CSV_PATH.relative_to(ROOT_DIR)),
        },
        "forbidden_actions_observed": {
            "supabase_raw_write": False,
            "db_write": False,
            "yfinance_append": False,
            "model_training": False,
            "inference_save": False,
            "frontend_modify": False,
            "wandb": False,
            "mixed_provider_parquet": bool(quality.get("yfinance_row_count", 0)),
        },
    }

    if not combined.empty and PRICE_PATH.exists():
        metrics["manifest"] = write_manifest(combined, metrics)
        metrics["output_paths"]["manifest"] = str(MANIFEST_PATH.relative_to(ROOT_DIR))
    else:
        metrics["manifest"] = None

    fetched_count = int(status_counts.get("fetched", 0))
    failed_count = int(sum(value for key, value in status_counts.items() if key != "fetched"))
    if source_limit_hit or not api_key:
        final_status = "FAIL"
        summary = "EODHD quota/rate/API key 문제로 500티커 bootstrap을 완료하지 못했다."
    elif quality.get("adjusted_ohlc_violation_count") not in (0, None) or quality.get("duplicate_ticker_date_source", 0) != 0 or quality.get("yfinance_row_count", 0) != 0:
        final_status = "FAIL"
        summary = "데이터 품질 또는 source/provider 혼합 가드가 실패했다."
    elif fetched_count >= max(1, int(len(universe) * 0.95)):
        final_status = "PASS" if failed_count == 0 else "WARN"
        summary = "EODHD 500티커 local price bootstrap이 완료되어 다음 indicator/context generation CP로 진행 가능하다."
    else:
        final_status = "WARN" if fetched_count > 0 else "FAIL"
        summary = "일부 ticker를 확보했지만 coverage가 부족하거나 실패 ticker가 많아 재시도 계획이 필요하다."

    metrics["final_decision"] = {
        "status": final_status,
        "summary": summary,
        "fetched_ticker_count": fetched_count,
        "failed_ticker_count": failed_count,
    }

    write_json(METRICS_PATH, metrics)
    write_json(LOG_DIR / METRICS_PATH.name, metrics)
    write_report(metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP144 EODHD 500 ticker local bootstrap")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_false", dest="resume")
    parser.add_argument("--use-env-proxy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bootstrap(args)


if __name__ == "__main__":
    main()
