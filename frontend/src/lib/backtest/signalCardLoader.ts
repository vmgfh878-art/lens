// 전략 신호 카드 로더 — 한 티커의 가격/지표/예측을 fetch한 뒤 signalBuilder로 분류.
// I/O 경계 + signalBuilder 호출을 한 자리에 모아 BacktestView를 view-only로 둔다.

import {
  DisplayTimeframe,
  PredictionResult,
  fetchIndicators,
  fetchPrices,
  fetchTickers,
} from "@/api/client";
import { DEFAULT_PRICE_LOOKBACK_DAYS } from "@/lib/backtest/constants";
import {
  buildIndicatorSignals,
  buildSignals,
  classifyIndicatorSignal,
  classifySignalGroup,
  getLatestIndicatorBefore,
  normalizeRsi,
} from "@/lib/backtest/signalBuilder";
import type { StrategyId, StrategySignalCard } from "@/lib/backtest/types";
import { buildDefaultPriceWindow, sortPriceRows, sortUniqueByDate } from "@/lib/dateUtils";

export async function fetchPriceHistory(ticker: string, timeframe: DisplayTimeframe) {
  const window = buildDefaultPriceWindow(DEFAULT_PRICE_LOOKBACK_DAYS);
  const response = await fetchPrices(ticker, {
    timeframe,
    start: window.start,
    end: window.end,
  });
  return sortPriceRows(response.data.data);
}

export async function loadStrategySignalCard(ticker: string, strategyId: StrategyId): Promise<StrategySignalCard> {
  const baseCard: StrategySignalCard = {
    ticker,
    sector: null,
    group: "watch",
    signalLabel: "아직 신호 없음",
    reason:
      strategyId === "lens_balance_v1"
        ? "가격, 예측선, AI 밴드 history 중 일부가 부족해 최신 전략 신호를 만들 수 없습니다."
        : "가격 또는 보조지표가 부족해 최신 전략 신호를 만들 수 없습니다.",
    asofDate: null,
    conservativeReturn: null,
    lowerBandReturn: null,
    bandWidthReturn: null,
    bandWidthExpansion: null,
    bandWidthPercentile: null,
    ma60Ratio: null,
    ma20Ratio: null,
    macdRatio: null,
    rsi: null,
    atrRatio: null,
    hasUsableSignal: false,
  };

  try {
    const [prices, indicatorsResponse, lineResponse, bandResponse, tickerResponse] = await Promise.all([
      fetchPriceHistory(ticker, "1D"),
      fetchIndicators(ticker, { timeframe: "1D", limit: 300 }).catch(() => null),
      // legacy /predictions/history endpoint 제거됨. lens_balance_v1 은 v1 endpoint 로 옮겨야 한다.
      Promise.resolve(null as { data: PredictionResult[] } | null),
      Promise.resolve(null as { data: PredictionResult[] } | null),
      fetchTickers({ search: ticker, limit: 1 }).catch(() => null),
    ]);
    const indicators = sortUniqueByDate(indicatorsResponse?.data.data ?? []);
    const lineRows = lineResponse?.data ?? [];
    const bandRows = bandResponse?.data ?? [];
    const sector = tickerResponse?.data.find((item) => item.ticker === ticker)?.sector ?? null;

    if (prices.length === 0) {
      return { ...baseCard, sector };
    }

    const signals =
      strategyId === "lens_balance_v1"
        ? lineRows.length > 0 && bandRows.length > 0
          ? buildSignals({ priceRows: prices, lineHistory: lineRows, bandHistory: bandRows })
          : []
        : buildIndicatorSignals({ priceRows: prices, indicators });
    const latestSignal = signals.at(-1);
    if (!latestSignal) {
      return { ...baseCard, sector };
    }

    const latestIndicator = getLatestIndicatorBefore(indicators, latestSignal.date);
    const classification = strategyId === "lens_balance_v1" ? classifySignalGroup(latestSignal) : classifyIndicatorSignal(latestSignal);
    return {
      ticker,
      sector,
      group: classification.group,
      signalLabel: classification.label,
      reason: latestSignal.reason,
      asofDate: latestSignal.date,
      conservativeReturn: latestSignal.conservativeReturn,
      lowerBandReturn: latestSignal.lowerBandReturn,
      bandWidthReturn: latestSignal.bandWidthReturn,
      bandWidthExpansion: latestSignal.bandWidthExpansion,
      bandWidthPercentile: latestSignal.bandWidthPercentile,
      ma60Ratio: latestSignal.ma60Ratio ?? latestIndicator?.ma_60_ratio ?? null,
      ma20Ratio: latestSignal.ma20Ratio ?? latestIndicator?.ma_20_ratio ?? null,
      macdRatio: latestSignal.macdRatio ?? latestIndicator?.macd_ratio ?? null,
      rsi: latestSignal.rsi ?? normalizeRsi(latestIndicator?.rsi),
      atrRatio: latestSignal.atrRatio ?? latestIndicator?.atr_ratio ?? null,
      hasUsableSignal: true,
    };
  } catch {
    return baseCard;
  }
}
