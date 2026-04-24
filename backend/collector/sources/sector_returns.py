from __future__ import annotations

import pandas as pd


def build_sector_returns(price_frame: pd.DataFrame, stock_info_frame: pd.DataFrame) -> pd.DataFrame:
    """보유한 주가와 섹터 메타데이터로 일별 섹터 수익률을 계산한다."""
    if price_frame.empty or stock_info_frame.empty:
        return pd.DataFrame(columns=["date", "sector", "etf_ticker", "return", "close"])

    merged = price_frame.merge(stock_info_frame[["ticker", "sector"]], on="ticker", how="left")
    merged = merged.dropna(subset=["date", "ticker", "close", "sector"]).copy()
    if merged.empty:
        return pd.DataFrame(columns=["date", "sector", "etf_ticker", "return", "close"])

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["ticker", "date"])
    merged["return"] = merged.groupby("ticker")["close"].pct_change()
    merged = merged.dropna(subset=["return"])
    if merged.empty:
        return pd.DataFrame(columns=["date", "sector", "etf_ticker", "return", "close"])

    sector_frame = (
        merged.groupby(["date", "sector"], as_index=False)
        .agg(
            return_value=("return", "mean"),
            close_value=("close", "mean"),
        )
        .rename(columns={"return_value": "return", "close_value": "close"})
    )
    sector_frame["etf_ticker"] = None

    market_frame = (
        merged.groupby("date", as_index=False)
        .agg(
            return_value=("return", "mean"),
            close_value=("close", "mean"),
        )
        .rename(columns={"return_value": "return", "close_value": "close"})
    )
    market_frame["sector"] = "Market"
    market_frame["etf_ticker"] = None

    result = pd.concat([sector_frame, market_frame], ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    return result[["date", "sector", "etf_ticker", "return", "close"]].sort_values(["date", "sector"])
