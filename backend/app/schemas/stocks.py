from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class IndicatorPoint(BaseModel):
    date: str
    rsi: float | None = None
    macd_ratio: float | None = None
    bb_position: float | None = None
    ma_5_ratio: float | None = None
    ma_20_ratio: float | None = None
    ma_60_ratio: float | None = None
    vol_change: float | None = None
    volume: int | None = None
    atr_ratio: float | None = None
    regime_label: str | None = None


class IndicatorResponseData(BaseModel):
    ticker: str
    timeframe: Literal["1D", "1W", "1M"]
    data: list[IndicatorPoint]


class PredictionData(BaseModel):
    ticker: str
    model_name: str
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
    meta: dict[str, Any] = Field(default_factory=dict)
