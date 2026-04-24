from __future__ import annotations

from collections import defaultdict

import pandas as pd

from backend.collector.repositories.base import fetch_frame


MINIMUM_ROWS_BY_TIMEFRAME = {
    "1D": 65,
    "1W": 64,
    # 월봉은 약 3년치만 확보돼도 표시/학습 진단에 충분하므로 24개월을 하한으로 둔다.
    "1M": 24,
}

REQUIRED_MACRO_COLUMNS = ("us10y", "yield_spread", "vix_close", "credit_spread_hy")


def summarize_indicator_counts(
    frame: pd.DataFrame,
    target_tickers: list[str],
    minimum_rows_by_timeframe: dict[str, int] | None = None,
) -> dict:
    """티커별 indicators 적재 길이를 기준으로 준비 상태를 계산한다."""
    thresholds = minimum_rows_by_timeframe or MINIMUM_ROWS_BY_TIMEFRAME
    target_set = set(target_tickers)
    counts_by_ticker: dict[str, dict[str, int]] = defaultdict(dict)

    if not frame.empty:
        grouped = (
            frame.groupby(["ticker", "timeframe"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for row in grouped.to_dict(orient="records"):
            ticker = str(row["ticker"]).upper()
            timeframe = str(row["timeframe"]).upper()
            if ticker in target_set:
                counts_by_ticker[ticker][timeframe] = int(row["count"])

    pending: list[str] = []
    ready_by_timeframe: dict[str, int] = {}
    for timeframe, threshold in thresholds.items():
        ready_by_timeframe[timeframe] = sum(
            1 for ticker in target_tickers if counts_by_ticker.get(ticker, {}).get(timeframe, 0) >= threshold
        )

    for ticker in target_tickers:
        if any(counts_by_ticker.get(ticker, {}).get(timeframe, 0) < threshold for timeframe, threshold in thresholds.items()):
            pending.append(ticker)

    return {
        "target_count": len(target_tickers),
        "ready_by_timeframe": ready_by_timeframe,
        "pending_tickers": pending,
        "counts_by_ticker": {ticker: counts_by_ticker.get(ticker, {}) for ticker in target_tickers},
    }


def get_indicator_readiness(target_tickers: list[str]) -> dict:
    """DB에서 indicators 상태를 읽어 학습 준비 상태를 계산한다."""
    frame = fetch_frame(
        "indicators",
        columns="ticker,timeframe,date",
        filters=[("in", "ticker", target_tickers)] if target_tickers else None,
        order_by="date",
    )
    return summarize_indicator_counts(frame, target_tickers)


def get_macro_readiness() -> dict:
    """필수 거시 컬럼별 적재 범위를 요약한다."""
    frame = fetch_frame(
        "macroeconomic_indicators",
        columns="date,us10y,yield_spread,vix_close,credit_spread_hy",
        order_by="date",
    )
    summary: dict[str, dict[str, object]] = {}
    if frame.empty:
        for column in REQUIRED_MACRO_COLUMNS:
            summary[column] = {"rows": 0, "min_date": None, "max_date": None}
        return summary

    for column in REQUIRED_MACRO_COLUMNS:
        series_frame = frame[["date", column]].dropna()
        summary[column] = {
            "rows": int(len(series_frame)),
            "min_date": None if series_frame.empty else str(series_frame["date"].min()),
            "max_date": None if series_frame.empty else str(series_frame["date"].max()),
        }
    return summary
