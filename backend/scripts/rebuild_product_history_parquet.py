"""
v1 serving line/band parquet를 unified product_prediction_history_1D.parquet로 재구성한다.

이 파일은 기존 frontend의 /api/v1/stocks/{ticker}/predictions/product-history
호환 계층을 위해 사용한다.
"""

from __future__ import annotations

from bisect import bisect_right
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "backend" / "data" / "v1"
OUT_DIR = V1_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
MARKET_PATH = V1_DIR / "market_prices_1d.parquet"

PARQUET_OUT = OUT_DIR / "product_prediction_history_1D.parquet"
MANIFEST_OUT = OUT_DIR / "product_prediction_history_1D.manifest.json"

NOW_ISO = datetime.now(timezone.utc).isoformat()


def add_business_days(date_value: str, days: int) -> str:
    parsed = pd.Timestamp(date_value)
    return str((parsed + pd.offsets.BDay(days)).date())


def build_band_future_calendar() -> dict[str, list[str]]:
    band = pd.read_parquet(V1_DIR / "predictions_band_1d.parquet", columns=["ticker", "asof_date", "forecast_date", "horizon_step"])
    band["ticker"] = band["ticker"].astype(str).str.upper()
    band["asof_date"] = pd.to_datetime(band["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    band["forecast_date"] = pd.to_datetime(band["forecast_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    calendars: dict[str, list[str]] = {}
    for ticker, ticker_rows in band.groupby("ticker", sort=False):
        latest_asof = ticker_rows["asof_date"].max()
        latest_rows = ticker_rows[ticker_rows["asof_date"] == latest_asof].sort_values("horizon_step")
        calendars[str(ticker)] = latest_rows["forecast_date"].dropna().astype(str).tolist()
    return calendars


def resolve_line_display_dates(line_rows: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    future_band_calendar = build_band_future_calendar()
    resolved = pd.Series(index=line_rows.index, dtype="object")

    price_groups = {
        ticker: rows["date"].dropna().astype(str).tolist()
        for ticker, rows in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False)
    }

    for ticker, rows in line_rows.groupby("ticker", sort=False):
        known_dates = price_groups.get(str(ticker), [])
        future_dates = future_band_calendar.get(str(ticker), [])
        for idx, asof_date in rows["asof_date"].items():
            if not isinstance(asof_date, str):
                resolved.loc[idx] = None
                continue
            target_date: str | None = None
            if known_dates:
                base_index = bisect_right(known_dates, asof_date) - 1
                if base_index >= 0:
                    target_index = base_index + 5
                    if target_index < len(known_dates):
                        target_date = known_dates[target_index]
                    elif future_dates:
                        sessions_to_latest = len(known_dates) - bisect_right(known_dates, asof_date)
                        future_index = len(future_dates) - 1 - sessions_to_latest
                        if 0 <= future_index < len(future_dates):
                            target_date = future_dates[future_index]
            resolved.loc[idx] = target_date or add_business_days(asof_date, 5)

    return resolved


def build_line_rows() -> pd.DataFrame:
    df = pd.read_parquet(V1_DIR / "predictions_line_1d.parquet")
    prices = pd.read_parquet(MARKET_PATH, columns=["ticker", "date", "close"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    merged = df.merge(
        prices.rename(columns={"date": "asof_date"}),
        on=["ticker", "asof_date"],
        how="left",
        validate="many_to_one",
    )
    n = len(merged)
    display_dates = resolve_line_display_dates(
        merged[["ticker", "asof_date"]].copy(),
        prices[["ticker", "date"]].copy(),
    )
    return pd.DataFrame(
        {
            "ticker": merged["ticker"].astype(str).str.upper(),
            "timeframe": "1D",
            "role": "line",
            "run_id": merged["model_id"].astype(str),
            "asof_date": pd.to_datetime(merged["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "display_horizon": 5,
            "display_date": display_dates,
            # 차트 계약을 line price-level로 맞춘다.
            "line_value": merged["close"].astype("float64") * (1.0 + merged["safe_line_score"].astype("float64")),
            "lower_value": np.full(n, np.nan, dtype="float64"),
            "upper_value": np.full(n, np.nan, dtype="float64"),
            "source": "v1_local_parquet",
            "model_feature_hash": merged["source_cp"].astype(str),
            "created_at": NOW_ISO,
        }
    )


def build_band_rows() -> pd.DataFrame:
    df = pd.read_parquet(V1_DIR / "predictions_band_1d.parquet")
    n = len(df)
    return pd.DataFrame(
        {
            "ticker": df["ticker"].astype(str).str.upper(),
            "timeframe": "1D",
            "role": "band",
            "run_id": df["model_id"].astype(str),
            "asof_date": pd.to_datetime(df["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "display_horizon": df["horizon_step"].astype(int),
            "display_date": pd.to_datetime(df["forecast_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "line_value": np.full(n, np.nan, dtype="float64"),
            "lower_value": df["band_lower"].astype("float64"),
            "upper_value": df["band_upper"].astype("float64"),
            "source": "v1_local_parquet",
            "model_feature_hash": df["source_cp"].astype(str),
            "created_at": NOW_ISO,
        }
    )


def main() -> None:
    print("Building unified product_prediction_history_1D...")

    line_df = build_line_rows()
    print(f"  line rows: {len(line_df):,} ({line_df['ticker'].nunique()} tickers)")

    band_df = build_band_rows()
    print(f"  band rows: {len(band_df):,} ({band_df['ticker'].nunique()} tickers)")

    unified = pd.concat([line_df, band_df], ignore_index=True)
    unified = unified.sort_values(["ticker", "role", "asof_date", "display_horizon"]).reset_index(drop=True)
    print(f"  total rows: {len(unified):,}")

    unified.to_parquet(PARQUET_OUT, index=False, compression="snappy")
    size_mb = PARQUET_OUT.stat().st_size / 1024 / 1024
    print(f"  wrote {PARQUET_OUT} ({size_mb:.2f} MB)")

    manifest = {
        "line_run_id": str(line_df["run_id"].iloc[0]) if len(line_df) else None,
        "band_run_id": str(band_df["run_id"].iloc[0]) if len(band_df) else None,
        "asof_start": unified["asof_date"].min(),
        "asof_end": unified["asof_date"].max(),
        "row_count": int(len(unified)),
        "source": "v1_local_parquet",
        "built_at": NOW_ISO,
        "cp": "CP212",
        "line_policy": "current_predictions_line_1d_serving_rows",
        "band_policy": "current_predictions_band_1d_serving_rows",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {MANIFEST_OUT}")
    print(f"  manifest: {manifest}")


if __name__ == "__main__":
    main()
