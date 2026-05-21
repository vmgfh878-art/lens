"""
v1 line/band parquets → unified product_prediction_history_1D.parquet
(기존 frontend 의 /api/v1/stocks/{ticker}/predictions/product-history 와 호환)

Output:
  data/parquet/product_prediction_history_1D.parquet
  data/parquet/product_prediction_history_1D.manifest.json

Run:
    cd C:\\Users\\user\\lens
    python backend/scripts/rebuild_product_history_parquet.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "backend" / "data" / "v1"
OUT_DIR = ROOT / "backend" / "data" / "v1"   # Render rootDir=backend 호환
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_OUT = OUT_DIR / "product_prediction_history_1D.parquet"
MANIFEST_OUT = OUT_DIR / "product_prediction_history_1D.manifest.json"

NOW_ISO = datetime.now(timezone.utc).isoformat()


import numpy as np


def build_line_rows() -> pd.DataFrame:
    df = pd.read_parquet(V1_DIR / "predictions_line_1d.parquet")
    n = len(df)
    out = pd.DataFrame({
        "ticker": df["ticker"].astype(str).str.upper(),
        "timeframe": "1D",
        "role": "line",
        "run_id": df["model_id"].astype(str),
        "asof_date": pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d"),
        "display_horizon": 5,
        "display_date": pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d"),
        "line_value": df["safe_line_score"].astype("float64"),
        "lower_value": np.full(n, np.nan, dtype="float64"),
        "upper_value": np.full(n, np.nan, dtype="float64"),
        "source": "v1_local_parquet",
        "model_feature_hash": df["source_cp"].astype(str),
        "created_at": NOW_ISO,
    })
    return out


def build_band_rows() -> pd.DataFrame:
    df = pd.read_parquet(V1_DIR / "predictions_band_1d.parquet")
    n = len(df)
    out = pd.DataFrame({
        "ticker": df["ticker"].astype(str).str.upper(),
        "timeframe": "1D",
        "role": "band",
        "run_id": df["model_id"].astype(str),
        "asof_date": pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d"),
        "display_horizon": df["horizon_step"].astype(int),
        "display_date": pd.to_datetime(df["forecast_date"]).dt.strftime("%Y-%m-%d"),
        "line_value": np.full(n, np.nan, dtype="float64"),
        "lower_value": df["band_lower"].astype("float64"),
        "upper_value": df["band_upper"].astype("float64"),
        "source": "v1_local_parquet",
        "model_feature_hash": df["source_cp"].astype(str),
        "created_at": NOW_ISO,
    })
    return out


def main():
    print("Building unified product_prediction_history_1D...")

    line_df = build_line_rows()
    print(f"  line rows: {len(line_df):,} ({line_df['ticker'].nunique()} tickers)")

    band_df = build_band_rows()
    print(f"  band rows: {len(band_df):,} ({band_df['ticker'].nunique()} tickers)")

    unified = pd.concat([line_df, band_df], ignore_index=True)
    unified = unified.sort_values(["ticker", "role", "asof_date", "display_horizon"]).reset_index(drop=True)
    print(f"  total: {len(unified):,}")

    unified.to_parquet(PARQUET_OUT, index=False, compression="snappy")
    size_mb = PARQUET_OUT.stat().st_size / 1024 / 1024
    print(f"  → {PARQUET_OUT} ({size_mb:.2f} MB)")

    # Manifest
    line_run_id = str(line_df["run_id"].iloc[0]) if len(line_df) else None
    band_run_id = str(band_df["run_id"].iloc[0]) if len(band_df) else None
    asof_start = unified["asof_date"].min()
    asof_end = unified["asof_date"].max()

    manifest = {
        "line_run_id": line_run_id,
        "band_run_id": band_run_id,
        "asof_start": asof_start,
        "asof_end": asof_end,
        "row_count": int(len(unified)),
        "source": "v1_local_parquet",
        "built_at": NOW_ISO,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {MANIFEST_OUT}")
    print(f"  manifest: {manifest}")
    print("Done.")


if __name__ == "__main__":
    main()
