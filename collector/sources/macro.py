from __future__ import annotations

import pandas as pd
import requests
from fredapi import Fred


FRED_SERIES_MAP = {
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "ppi": "PPIACO",
    "gdp": "GDP",
    "real_gdp": "GDPC1",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "unemployment_rate": "UNRATE",
    "jolt": "JTSJOL",
    "consumer_sentiment": "UMCSENT",
    "cci": "CSCICP03USM665S",
    "interest_rate": "DFF",
    "ff_targetrate_upper": "DFEDTARU",
    "ff_targetrate_lower": "DFEDTARL",
    "us10y": "DGS10",
    "us2y": "DGS2",
    "credit_spread_hy": "BAMLH0A0HYM2",
    "trade_balance": "BOPGSTB",
    "tradebalance_goods": "BOPGTB",
    "trade_import": "IMPGS",
    "trade_export": "EXPGS",
}

FMP_MACRO_MAP = {
    "vix_close": "^VIX",
}


def fetch_fred_frame(start_date: str, api_key: str | None) -> pd.DataFrame:
    """FRED 시계열을 하나의 데이터프레임으로 병합한다."""
    if not api_key:
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames: list[pd.DataFrame] = []
    for column, series_id in FRED_SERIES_MAP.items():
        try:
            series = fred.get_series(series_id, observation_start=start_date)
            frame = pd.DataFrame(series, columns=[column])
            frames.append(frame)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def fetch_fmp_macro_frame(start_date: str, api_key: str | None) -> pd.DataFrame:
    """FMP에서 접근 가능한 보조 시장 지표를 읽는다."""
    if not api_key:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    base_url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    for column, ticker in FMP_MACRO_MAP.items():
        try:
            response = requests.get(
                f"{base_url}?symbol={ticker}&from={start_date}&apikey={api_key}",
                timeout=20,
            )
            if response.status_code != 200:
                continue
            payload = response.json()
        except Exception:
            continue

        if not payload:
            continue

        frame = pd.DataFrame(payload)
        if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[["date", "close"]].rename(columns={"close": column}).set_index("date")
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def build_macro_frame(start_date: str, fred_api_key: str | None, fmp_api_key: str | None = None) -> pd.DataFrame:
    """FRED와 FMP 보조 지표를 병합한 거시 프레임을 만든다."""
    fred_frame = fetch_fred_frame(start_date, fred_api_key)
    fmp_frame = fetch_fmp_macro_frame(start_date, fmp_api_key)

    if fred_frame.empty and fmp_frame.empty:
        return pd.DataFrame()

    combined = pd.concat([frame for frame in (fred_frame, fmp_frame) if not frame.empty], axis=1)
    combined = combined.groupby(level=0, axis=1).first()

    if "us10y" in combined.columns and "us2y" in combined.columns:
        combined["yield_spread"] = combined["us10y"] - combined["us2y"]
    else:
        combined["yield_spread"] = None

    combined = combined.sort_index()
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    return combined.reset_index()
