from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import (  # noqa: E402
    _BASE_FEATURE_COLUMNS,
    _CONTEXT_COLUMNS,
    _compute_features_for_single_ticker,
    _resample_context_frame,
    resample_price_frame,
)
from backend.collector.repositories.base import fetch_frame  # noqa: E402


def build_aapl_debug_frame(timeframe: str = "1D") -> pd.DataFrame:
    price = fetch_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume",
        filters=[("eq", "ticker", "AAPL")],
        order_by="date",
    )
    macro = fetch_frame(
        "macroeconomic_indicators",
        columns="date,us10y,yield_spread,vix_close,credit_spread_hy",
        order_by="date",
    )
    breadth = fetch_frame(
        "market_breadth",
        columns="date,nh_nl_index,ma200_pct",
        order_by="date",
    )

    price["date"] = pd.to_datetime(price["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    breadth["date"] = pd.to_datetime(breadth["date"])

    price_frame = resample_price_frame(price, timeframe)
    feature_frame = _compute_features_for_single_ticker(price_frame)
    macro_frame = _resample_context_frame(
        macro,
        timeframe,
        ("us10y", "yield_spread", "vix_close", "credit_spread_hy"),
    )
    breadth_frame = _resample_context_frame(
        breadth,
        timeframe,
        ("nh_nl_index", "ma200_pct"),
    )

    if not macro_frame.empty:
        feature_frame = feature_frame.merge(macro_frame, on="date", how="left")
    if not breadth_frame.empty:
        feature_frame = feature_frame.merge(breadth_frame, on="date", how="left")

    for column in _CONTEXT_COLUMNS:
        if column not in feature_frame.columns:
            feature_frame[column] = pd.NA

    fill_columns = [column for column in _CONTEXT_COLUMNS if column in feature_frame.columns]
    feature_frame[fill_columns] = feature_frame[fill_columns].ffill()
    return feature_frame


def summarize_first_non_null(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for column in _BASE_FEATURE_COLUMNS:
        series = frame[column]
        valid_dates = frame.loc[series.notna(), "date"]
        result[column] = {
            "first_non_null": valid_dates.min().date().isoformat() if not valid_dates.empty else None,
            "null_ratio": float(series.isna().mean()),
        }
    return result


def main() -> None:
    frame = build_aapl_debug_frame("1D")
    print(json.dumps(summarize_first_non_null(frame), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
