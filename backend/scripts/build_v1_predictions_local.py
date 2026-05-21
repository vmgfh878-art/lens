"""
CP204 v1 import package → filtered local parquets for backend serving.

Output: backend/data/v1/
  - predictions_line_1d.parquet  (past 365 days)
  - predictions_band_1d.parquet  (past 365 days, 5 horizons each)
  - predictions_band_1w.parquet  (past 730 days, ensemble averaged)

Run:
    cd C:\\Users\\user\\lens
    python backend/scripts/build_v1_predictions_local.py
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "artifacts" / "cp204_v1_import_package"
DST = ROOT / "backend" / "data" / "v1"
DST.mkdir(parents=True, exist_ok=True)

TODAY = date(2026, 5, 20)
CUTOFF_1D = (TODAY - timedelta(days=365)).isoformat()
CUTOFF_1W = (TODAY - timedelta(days=730)).isoformat()


def build_line_1d():
    print("=" * 50)
    print("1D Line")
    print("=" * 50)
    df = pd.read_parquet(SRC / "cp204_1d_line_payload.parquet")
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["asof_date"] >= CUTOFF_1D].copy()
    after = len(df)
    print(f"  rows: {before:,} → {after:,} (cutoff={CUTOFF_1D})")

    # Keep only frontend-needed columns
    cols = [
        "ticker", "asof_date",
        "line_score", "safe_line_score",
        "line_rank_by_date", "safe_line_rank_by_date",
        "line_top_decile_flag", "safe_line_top_decile_flag",
        "actual_h5_return",
        "model_id", "source_cp",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.sort_values(["ticker", "asof_date"]).reset_index(drop=True)

    out = DST / "predictions_line_1d.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  tickers: {df['ticker'].nunique()}, dates: {df['asof_date'].nunique()}")


def build_band_1d():
    print("=" * 50)
    print("1D Band")
    print("=" * 50)
    df = pd.read_parquet(SRC / "cp204_1d_band_historical_payload.parquet")
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["asof_date"] >= CUTOFF_1D].copy()
    after = len(df)
    print(f"  rows: {before:,} → {after:,} (cutoff={CUTOFF_1D})")

    cols = [
        "ticker", "asof_date", "forecast_date", "horizon_step",
        "band_lower", "band_upper",
        "actual_return", "actual_return_available",
        "model_id", "source_cp",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.sort_values(["ticker", "asof_date", "horizon_step"]).reset_index(drop=True)

    out = DST / "predictions_band_1d.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  tickers: {df['ticker'].nunique()}, dates: {df['asof_date'].nunique()}, horizons: {sorted(df['horizon_step'].unique())}")


def build_band_1w():
    print("=" * 50)
    print("1W Band (ensemble averaged)")
    print("=" * 50)
    df = pd.read_parquet(SRC / "cp204_1w_band_payload.parquet")
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["asof_date"] >= CUTOFF_1W].copy()
    print(f"  rows after date filter: {before:,} → {len(df):,} (cutoff={CUTOFF_1W})")

    # Ensemble average over seeds
    agg = df.groupby(["ticker", "asof_date", "horizon_step"], as_index=False).agg({
        "band_lower": "mean",
        "band_upper": "mean",
        "actual_return": "mean",
        "model_id": "first",
        "source_cp": "first",
    })
    print(f"  after ensemble avg: {len(agg):,}")

    agg = agg.sort_values(["ticker", "asof_date", "horizon_step"]).reset_index(drop=True)

    out = DST / "predictions_band_1w.parquet"
    agg.to_parquet(out, index=False, compression="snappy")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  tickers: {agg['ticker'].nunique()}, dates: {agg['asof_date'].nunique()}, horizons: {sorted(agg['horizon_step'].unique())}")


def main():
    print(f"\nTODAY (logical): {TODAY.isoformat()}")
    print(f"Cutoffs: 1D >= {CUTOFF_1D}, 1W >= {CUTOFF_1W}\n")

    build_line_1d()
    print()
    build_band_1d()
    print()
    build_band_1w()
    print()

    total = sum(p.stat().st_size for p in DST.glob("*.parquet")) / 1024 / 1024
    print("=" * 50)
    print(f"Done. Total size: {total:.2f} MB at {DST}")


if __name__ == "__main__":
    main()
