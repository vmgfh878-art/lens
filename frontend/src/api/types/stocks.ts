import type { DisplayTimeframe } from "./common";

export interface PriceBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

export interface StockSummary {
  ticker: string;
  sector: string | null;
  industry: string | null;
  market_cap: number | null;
}

export interface PriceResult {
  ticker: string;
  timeframe: DisplayTimeframe;
  start: string;
  end: string;
  data: PriceBar[];
}
