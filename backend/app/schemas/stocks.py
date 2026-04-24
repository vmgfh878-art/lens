from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class StockSummary(BaseModel):
    ticker: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None


class PriceBar(BaseModel):
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: int | None = None


class PriceResponseData(BaseModel):
    ticker: str
    timeframe: Literal["1D", "1W", "1M"]
    start: str
    end: str
    data: list[PriceBar]


class PredictionData(BaseModel):
    ticker: str
    model_name: Literal["patchtst", "cnn_lstm", "tide"]
    timeframe: Literal["1D", "1W"]
    horizon: int
    asof_date: str
    decision_time: str
    run_id: str
    model_ver: str
    signal: Literal["BUY", "SELL", "HOLD"]
    forecast_dates: list[str]
    upper_band_series: list[float]
    lower_band_series: list[float]
    conservative_series: list[float]
    line_series: list[float]
    band_quantile_low: float | None = None
    band_quantile_high: float | None = None
