"""
CP204 v1 import package → Supabase 3 tables.

Pre-requisite: backend/db/migrations/v1_predictions_tables.sql 이 Supabase 에 적용된 상태.

Filter:
  - 1D Line / 1D Band: asof_date >= today - 365 days
  - 1W Band: asof_date >= today - 730 days (ensemble averaged over seeds)

Run:
    cd C:\\Users\\user\\lens
    python backend/scripts/import_v1_predictions_to_supabase.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

PKG = ROOT / "data" / "artifacts" / "cp204_v1_import_package"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("SUPABASE_URL / SUPABASE_KEY 환경변수가 없습니다.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

TODAY = date(2026, 5, 20)
CUTOFF_1D = (TODAY - timedelta(days=365)).isoformat()
CUTOFF_1W = (TODAY - timedelta(days=730)).isoformat()


def _to_float(v):
    if pd.isna(v):
        return None
    return float(v)


def _to_bool(v, default=False):
    if pd.isna(v):
        return default
    return bool(v)


def batch_upsert(table: str, records: list[dict], batch_size: int = 500):
    total = len(records)
    if total == 0:
        print(f"  [{table}] empty, skipped")
        return
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table(table).upsert(batch).execute()
        except Exception as exc:
            print(f"\n  [{table}] batch {i} failed: {exc}")
            print(f"  first record sample: {batch[0]}")
            raise
        done = min(i + batch_size, total)
        pct = (done / total) * 100
        print(f"  [{table}] {done}/{total} ({pct:.1f}%)", end="\r")
    print(f"  [{table}] {total}/{total} done                    ")


def import_line_1d():
    df = pd.read_parquet(PKG / "cp204_1d_line_payload.parquet")
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    df = df[df["asof_date"] >= CUTOFF_1D].copy()
    print(f"1D Line: {len(df):,} rows after filter (cutoff={CUTOFF_1D})")

    records = []
    for _, r in df.iterrows():
        records.append({
            "ticker": str(r["ticker"]),
            "asof_date": r["asof_date"],
            "line_score": _to_float(r.get("line_score")),
            "safe_line_score": _to_float(r.get("safe_line_score")),
            "line_rank_by_date": _to_float(r.get("line_rank_by_date")),
            "safe_line_rank_by_date": _to_float(r.get("safe_line_rank_by_date")),
            "line_top_decile_flag": _to_bool(r.get("line_top_decile_flag")),
            "safe_line_top_decile_flag": _to_bool(r.get("safe_line_top_decile_flag")),
            "actual_h5_return": _to_float(r.get("actual_h5_return")),
            "model_id": str(r.get("model_id")) if pd.notna(r.get("model_id")) else None,
            "source_cp": str(r.get("source_cp")) if pd.notna(r.get("source_cp")) else None,
        })
    batch_upsert("predictions_line_1d", records)


def import_band_1d():
    df = pd.read_parquet(PKG / "cp204_1d_band_historical_payload.parquet")
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.strftime("%Y-%m-%d")
    df = df[df["asof_date"] >= CUTOFF_1D].copy()
    print(f"1D Band: {len(df):,} rows after filter (cutoff={CUTOFF_1D})")

    records = []
    for _, r in df.iterrows():
        records.append({
            "ticker": str(r["ticker"]),
            "asof_date": r["asof_date"],
            "forecast_date": r["forecast_date"],
            "horizon_step": int(r["horizon_step"]),
            "band_lower": _to_float(r.get("band_lower")),
            "band_upper": _to_float(r.get("band_upper")),
            "actual_return": _to_float(r.get("actual_return")),
            "actual_return_available": _to_bool(r.get("actual_return_available")),
            "model_id": str(r.get("model_id")) if pd.notna(r.get("model_id")) else None,
            "source_cp": str(r.get("source_cp")) if pd.notna(r.get("source_cp")) else None,
        })
    batch_upsert("predictions_band_1d", records)


def import_band_1w():
    df = pd.read_parquet(PKG / "cp204_1w_band_payload.parquet")
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")
    df = df[df["asof_date"] >= CUTOFF_1W].copy()
    print(f"1W Band raw: {len(df):,} rows after date filter (cutoff={CUTOFF_1W})")

    # Ensemble average over seeds
    agg = df.groupby(["ticker", "asof_date", "horizon_step"], as_index=False).agg({
        "band_lower": "mean",
        "band_upper": "mean",
        "actual_return": "mean",
        "model_id": "first",
        "source_cp": "first",
    })
    print(f"1W Band after ensemble avg: {len(agg):,} rows")

    records = []
    for _, r in agg.iterrows():
        records.append({
            "ticker": str(r["ticker"]),
            "asof_date": r["asof_date"],
            "horizon_step": int(r["horizon_step"]),
            "band_lower": _to_float(r.get("band_lower")),
            "band_upper": _to_float(r.get("band_upper")),
            "actual_return": _to_float(r.get("actual_return")),
            "model_id": str(r.get("model_id")) if pd.notna(r.get("model_id")) else None,
            "source_cp": str(r.get("source_cp")) if pd.notna(r.get("source_cp")) else None,
        })
    batch_upsert("predictions_band_1w", records)


def main():
    print("=" * 60)
    print("CP204 v1 Predictions → Supabase Import")
    print(f"Today (logical): {TODAY.isoformat()}")
    print(f"Cutoffs: 1D >= {CUTOFF_1D}, 1W >= {CUTOFF_1W}")
    print("=" * 60)

    print()
    import_line_1d()
    print()
    import_band_1d()
    print()
    import_band_1w()
    print()
    print("All slots imported.")


if __name__ == "__main__":
    main()
