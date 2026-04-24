"""Lens raw parquet 초기 적재 스크립트."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"
CHUNK_SIZE = 500
UPSERT_MAX_ATTEMPTS = 4
UPSERT_RETRY_BASE_SECONDS = 1.5


def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[Error] .env에 SUPABASE_URL, SUPABASE_KEY가 필요합니다.")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _is_retryable_upsert_error(error: Exception) -> bool:
    """일시적인 Supabase 업서트 실패만 재시도 대상으로 본다."""
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError, json.JSONDecodeError)):
        return True

    if not isinstance(error, APIError):
        return False

    message = str(getattr(error, "message", "") or error).lower()
    details = str(getattr(error, "details", "") or "").lower()
    code = str(getattr(error, "code", "") or "")
    combined = f"{message} {details}"
    retry_keywords = ("502", "bad gateway", "json could not be generated", "cloudflare", "html")
    return code == "502" or any(keyword in combined for keyword in retry_keywords)


def _upsert_chunk_with_retry(client: Client, table: str, chunk: list[dict], on_conflict: str) -> None:
    """업서트가 잠깐 실패하면 지수 백오프로 다시 시도한다."""
    for attempt in range(1, UPSERT_MAX_ATTEMPTS + 1):
        try:
            client.table(table).upsert(chunk, on_conflict=on_conflict).execute()
            return
        except Exception as error:
            is_retryable = _is_retryable_upsert_error(error)
            is_last_attempt = attempt == UPSERT_MAX_ATTEMPTS
            if not is_retryable or is_last_attempt:
                raise

            wait_seconds = UPSERT_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"[warn] upsert 재시도 {attempt}/{UPSERT_MAX_ATTEMPTS - 1} "
                f"(table={table}, rows={len(chunk)}, wait={wait_seconds:.1f}s): {error}"
            )
            time.sleep(wait_seconds)


def chunked_upsert(client: Client, table: str, records: list[dict], on_conflict: str) -> None:
    total = len(records)
    for index in range(0, total, CHUNK_SIZE):
        chunk = records[index : index + CHUNK_SIZE]
        _upsert_chunk_with_retry(client, table, chunk, on_conflict)
        done = min(index + CHUNK_SIZE, total)
        print(f"  {done:,} / {total:,}", end="\r")
    print(f"  {total:,} rows loaded")


def safe_value(value):
    if pd.isna(value):
        return None
    return value


def read_parquet(filename: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        print(f"[Error] parquet 파일이 없습니다: {path}")
        sys.exit(1)
    return pd.read_parquet(path)


def filter_by_tickers(df: pd.DataFrame, sample_tickers: list[str] | None) -> pd.DataFrame:
    if not sample_tickers or "ticker" not in df.columns:
        return df
    return df[df["ticker"].isin(sample_tickers)].copy()


def step_stock_info(client: Client, data_dir: Path, sample_tickers: list[str] | None = None) -> None:
    print("\n[1/6] stock_info 적재 중.")
    df = filter_by_tickers(read_parquet("stock_info.parquet", data_dir).copy(), sample_tickers)
    if "industry" not in df.columns:
        df["industry"] = "Unknown"
    else:
        df["industry"] = df["industry"].fillna("Unknown")
    if "market_cap" not in df.columns:
        df["market_cap"] = None
    records = [
        {
            "ticker": row["ticker"],
            "sector": safe_value(row.get("sector")),
            "industry": safe_value(row.get("industry")),
            "market_cap": safe_value(row.get("market_cap")),
        }
        for _, row in df.iterrows()
    ]
    chunked_upsert(client, "stock_info", records, on_conflict="ticker")


def step_price_data(client: Client, data_dir: Path, sample_tickers: list[str] | None = None) -> None:
    print("\n[2/6] price_data 적재 중.")
    df = filter_by_tickers(read_parquet("price_data.parquet", data_dir).copy(), sample_tickers)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["ticker", "date", "close"])
    for optional_col in ("adjusted_close", "amount", "per", "pbr"):
        if optional_col not in df.columns:
            df[optional_col] = None
    records = [
        {
            "ticker": row["ticker"],
            "date": row["date"],
            "open": safe_value(row.get("open")),
            "high": safe_value(row.get("high")),
            "low": safe_value(row.get("low")),
            "close": safe_value(row.get("close")),
            "adjusted_close": safe_value(row.get("adjusted_close")),
            "volume": safe_value(row.get("volume")),
            "amount": safe_value(row.get("amount")),
            "per": safe_value(row.get("per")),
            "pbr": safe_value(row.get("pbr")),
        }
        for _, row in df.iterrows()
    ]
    chunked_upsert(client, "price_data", records, on_conflict="ticker,date")


def step_macro(client: Client, data_dir: Path) -> None:
    print("\n[3/6] macroeconomic_indicators 적재 중.")
    df = read_parquet("macroeconomic_indicators.parquet", data_dir).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    chunked_upsert(client, "macroeconomic_indicators", records, on_conflict="date")


def step_market_breadth(client: Client, data_dir: Path) -> None:
    print("\n[4/6] market_breadth 적재 중.")
    df = read_parquet("market_breadth.parquet", data_dir).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    chunked_upsert(client, "market_breadth", records, on_conflict="date")


def step_sector_returns(client: Client, data_dir: Path) -> None:
    print("\n[5/6] sector_returns 적재 중.")
    df = read_parquet("sector_returns.parquet", data_dir).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    subset = df[["date", "sector", "etf_ticker", "return", "close"]]
    records = subset.where(pd.notnull(subset), None).to_dict(orient="records")
    chunked_upsert(client, "sector_returns", records, on_conflict="date,sector")


def step_company_fundamentals(
    client: Client, data_dir: Path, sample_tickers: list[str] | None = None
) -> None:
    print("\n[6/6] company_fundamentals 적재 중.")
    df = filter_by_tickers(read_parquet("company_fundamentals.parquet", data_dir).copy(), sample_tickers)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for optional_col in ("shares_issued", "interest_coverage"):
        if optional_col not in df.columns:
            df[optional_col] = None
    cols = [
        "ticker",
        "date",
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "equity",
        "shares_issued",
        "eps",
        "roe",
        "debt_ratio",
        "interest_coverage",
        "operating_cash_flow",
    ]
    subset = df[cols]
    records = subset.where(pd.notnull(subset), None).to_dict(orient="records")
    chunked_upsert(client, "company_fundamentals", records, on_conflict="ticker,date")


def run_bootstrap(data_dir: Path = DEFAULT_DATA_DIR, sample_tickers: list[str] | None = None) -> None:
    print("=" * 60)
    print(" Lens raw parquet 초기 적재")
    print("=" * 60)

    supabase = get_client()
    step_stock_info(supabase, data_dir, sample_tickers)
    step_price_data(supabase, data_dir, sample_tickers)
    step_macro(supabase, data_dir)
    step_market_breadth(supabase, data_dir)
    step_sector_returns(supabase, data_dir)
    step_company_fundamentals(supabase, data_dir, sample_tickers)

    print("\n" + "=" * 60)
    print(" Raw 초기 적재 완료.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens raw parquet 적재")
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_DATA_DIR),
        help="parquet 파일이 있는 폴더 경로",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="샘플 적재할 종목 목록. 지정하지 않으면 전체를 적재합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bootstrap(Path(args.source_dir), args.tickers)
