"""
로컬 yfinance price/indicators/stock_info → backend/data/v1/ 의 작은 serving parquets.

학교 데모용. Render free tier (512MB) 적합 크기로 필터.

Output:
  backend/data/v1/market_prices_1d.parquet   (past 1y, 500 ticker)
  backend/data/v1/market_prices_1w.parquet
  backend/data/v1/market_indicators_1d.parquet (essential cols only)
  backend/data/v1/market_stock_info.parquet

Run:
    cd C:\\Users\\user\\lens
    python backend/scripts/build_v1_market_local.py
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "parquet"
DST = ROOT / "backend" / "data" / "v1"
DST.mkdir(parents=True, exist_ok=True)


def resolve_asof_date(raw: str | None) -> date:
    if raw:
        return date.fromisoformat(raw)
    return date.today()


def build_prices_1d(cutoff_1y: str):
    print("=" * 50)
    print("Prices 1D")
    print("=" * 50)
    df = pd.read_parquet(SRC / "price_data_yfinance_500.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["date"] >= cutoff_1y].copy()
    print(f"  rows: {before:,} → {len(df):,}")

    # 필수 column 만
    keep = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DST / "market_prices_1d.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def build_prices_1w(cutoff_1y: str):
    print("=" * 50)
    print("Prices 1W")
    print("=" * 50)
    df = pd.read_parquet(SRC / "price_data_yfinance_1W.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["date"] >= cutoff_1y].copy()
    print(f"  rows: {before:,} → {len(df):,}")

    keep = ["ticker", "date", "timeframe", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DST / "market_prices_1w.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def build_indicators_1d(cutoff_1y: str):
    print("=" * 50)
    print("Indicators 1D")
    print("=" * 50)
    df = pd.read_parquet(SRC / "indicators_yfinance_1D_500.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["date"] >= cutoff_1y].copy()
    print(f"  rows: {before:,} → {len(df):,}")

    # frontend 가 보통 쓰는 essential indicators 만
    keep = [
        "ticker", "date", "timeframe", "regime_label",
        "log_return", "rsi", "macd_ratio",
        "ma_5_ratio", "ma_20_ratio", "ma_60_ratio",
        "bb_position", "vol_change",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DST / "market_indicators_1d.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def build_stock_info():
    print("=" * 50)
    print("Stock info")
    print("=" * 50)
    df = pd.read_parquet(SRC / "stock_info.parquet")
    print(f"  rows: {len(df):,}")

    # 컬럼 단순화
    keep = ["ticker", "sector", "industry", "market_cap"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values("ticker").reset_index(drop=True)

    out = DST / "market_stock_info.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="로컬 v1 market parquet를 생성합니다.")
    parser.add_argument("--asof-date", default=None, help="명시한 기준일로 cutoff를 계산합니다. 생략하면 시스템 날짜를 사용합니다.")
    return parser.parse_args()


def main(asof_date: str | None = None):
    today = resolve_asof_date(asof_date)
    cutoff_1y = (today - timedelta(days=400)).isoformat()  # 약간 buffer
    print(f"\nAsof date: {today.isoformat()} (source={'argument' if asof_date else 'system_date'})")
    print(f"Cutoff: {cutoff_1y}\n")
    build_prices_1d(cutoff_1y)
    print()
    build_prices_1w(cutoff_1y)
    print()
    build_indicators_1d(cutoff_1y)
    print()
    build_stock_info()
    print()
    total = sum(p.stat().st_size for p in DST.glob("market_*.parquet")) / 1024 / 1024
    print(f"Done. Total market parquets: {total:.2f} MB at {DST}")


if __name__ == "__main__":
    main(parse_args().asof_date)
