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

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "parquet"
DST = ROOT / "backend" / "data" / "v1"
DST.mkdir(parents=True, exist_ok=True)

TODAY = date(2026, 5, 22)
CUTOFF_1Y = (TODAY - timedelta(days=400)).isoformat()  # 약간 buffer


def build_prices_1d():
    print("=" * 50)
    print("Prices 1D")
    print("=" * 50)
    df = pd.read_parquet(SRC / "price_data_yfinance_500.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["date"] >= CUTOFF_1Y].copy()
    print(f"  rows: {before:,} → {len(df):,}")

    # 필수 column 만
    keep = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DST / "market_prices_1d.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def build_prices_1w():
    print("=" * 50)
    print("Prices 1W")
    print("=" * 50)
    df = pd.read_parquet(SRC / "price_data_yfinance_1W.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["date"] >= CUTOFF_1Y].copy()
    print(f"  rows: {before:,} → {len(df):,}")

    keep = ["ticker", "date", "timeframe", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DST / "market_prices_1w.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def build_indicators_1d():
    print("=" * 50)
    print("Indicators 1D")
    print("=" * 50)
    df = pd.read_parquet(SRC / "indicators_yfinance_1D_500.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["date"] >= CUTOFF_1Y].copy()
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


def main():
    print(f"\nCutoff: {CUTOFF_1Y}\n")
    build_prices_1d()
    print()
    build_prices_1w()
    print()
    build_indicators_1d()
    print()
    build_stock_info()
    print()
    total = sum(p.stat().st_size for p in DST.glob("market_*.parquet")) / 1024 / 1024
    print(f"Done. Total market parquets: {total:.2f} MB at {DST}")


if __name__ == "__main__":
    main()
