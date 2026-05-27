import type { DisplayTimeframe } from "./common";

export interface IndicatorPoint {
  date: string;
  rsi: number | null;
  macd_ratio: number | null;
  bb_position: number | null;
  ma_5_ratio: number | null;
  ma_20_ratio: number | null;
  ma_60_ratio: number | null;
  vol_change: number | null;
  volume: number | null;
  atr_ratio: number | null;
  regime_label: string | null;
}

export interface IndicatorResult {
  ticker: string;
  timeframe: DisplayTimeframe;
  data: IndicatorPoint[];
}
