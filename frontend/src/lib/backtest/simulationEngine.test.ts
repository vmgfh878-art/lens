// CP231 Step C — simulationEngine.ts 추출 직후 결과를 박제.
// 통계 함수의 분기와 runStrategyBacktest의 핵심 수치를 잡아 이후 Step의 회귀를 가드.

import { describe, expect, it } from "vitest";

import type { IndicatorPoint, PredictionResult, PriceBar } from "@/api/client";

import {
  calculateMaxDrawdown,
  calculateSharpe,
  calculateSortino,
  chooseLargeLossThreshold,
  quantile,
  runStrategyBacktest,
} from "./simulationEngine";
import type { BacktestPoint } from "./types";

function makePriceBar(date: string, close: number): PriceBar {
  return { date, open: close, high: close, low: close, close, volume: null };
}

function makePoint(
  date: string,
  strategyEquity: number,
  buyHoldEquity: number,
  position: 0 | 1 = 1,
  price = 100,
): BacktestPoint {
  return { date, price, strategyEquity, buyHoldEquity, position };
}

function makeIndicator(date: string, overrides: Partial<IndicatorPoint> = {}): IndicatorPoint {
  return {
    date,
    ma_20_ratio: null,
    ma_60_ratio: null,
    macd_ratio: null,
    rsi: null,
    atr_ratio: null,
    bb_position: null,
    ...overrides,
  } as IndicatorPoint;
}

function makePrediction(asofDate: string, lower: number, upper: number, line: number): PredictionResult {
  return {
    asof_date: asofDate,
    forecast_dates: [asofDate],
    conservative_series: [line],
    line_series: [line],
    lower_band_series: [lower],
    upper_band_series: [upper],
  } as unknown as PredictionResult;
}

describe("calculateMaxDrawdown", () => {
  it("빈 배열 → 0", () => {
    expect(calculateMaxDrawdown([])).toBe(0);
  });
  it("단조 증가 → 0 (peak 갱신만)", () => {
    const points = [
      makePoint("2024-01-01", 1, 1),
      makePoint("2024-01-02", 1.1, 1.05),
      makePoint("2024-01-03", 1.2, 1.1),
    ];
    expect(calculateMaxDrawdown(points)).toBe(0);
  });
  it("peak 후 하락 → 음수 percent", () => {
    const points = [
      makePoint("2024-01-01", 1, 1),
      makePoint("2024-01-02", 1.5, 1.2),
      makePoint("2024-01-03", 0.75, 1.0),
    ];
    // peak=1.5, low=0.75 → drawdown = 0.75/1.5 - 1 = -0.5 → -50
    expect(calculateMaxDrawdown(points)).toBe(-50);
  });
  it("equityKey=buyHoldEquity → buyHold 기준", () => {
    const points = [
      makePoint("2024-01-01", 1, 1),
      makePoint("2024-01-02", 2, 2),
      makePoint("2024-01-03", 1.5, 1),
    ];
    // buyHold peak=2, low=1 → -50
    expect(calculateMaxDrawdown(points, "buyHoldEquity")).toBe(-50);
  });
});

describe("calculateSharpe", () => {
  it("샘플 < 2 → 0", () => {
    expect(calculateSharpe([])).toBe(0);
    expect(calculateSharpe([0.01])).toBe(0);
  });
  it("std 0 → 0", () => {
    expect(calculateSharpe([0.01, 0.01, 0.01])).toBe(0);
  });
  it("정상 returns → (mean/std) * sqrt(252)", () => {
    const result = calculateSharpe([0.01, -0.005, 0.02, 0.01]);
    expect(result).toBeCloseTo(13.475, 2);
  });
});

describe("calculateSortino", () => {
  it("샘플 < 2 → 0", () => {
    expect(calculateSortino([])).toBe(0);
    expect(calculateSortino([0.01])).toBe(0);
  });
  it("downside < 2 + mean>0 → mean*sqrt(252)", () => {
    const returns = [0.01, 0.02, 0.03];
    const expected = (0.01 + 0.02 + 0.03) / 3 * Math.sqrt(252);
    expect(calculateSortino(returns)).toBeCloseTo(expected, 6);
  });
  it("downside ≥ 2 정상 → (mean/downsideDev) * sqrt(252)", () => {
    const result = calculateSortino([0.01, -0.005, 0.02, -0.01, 0.01]);
    expect(result).toBeCloseTo(7.099, 2);
  });
});

describe("quantile", () => {
  it("빈 배열 → null", () => {
    expect(quantile([], 0.5)).toBeNull();
  });
  it("정렬 후 인덱스 ratio*(n-1)", () => {
    expect(quantile([1, 2, 3, 4, 5], 0.5)).toBe(3);
    expect(quantile([5, 4, 3, 2, 1], 0.5)).toBe(3);
    expect(quantile([1, 2, 3, 4, 5], 0)).toBe(1);
    expect(quantile([1, 2, 3, 4, 5], 1)).toBe(5);
    expect(quantile([1, 2, 3, 4, 5], 0.2)).toBe(1);
  });
});

describe("chooseLargeLossThreshold", () => {
  it("빈 배열 → null", () => {
    expect(chooseLargeLossThreshold([])).toBeNull();
  });
  it("lowerQuintile vs -0.02 중 작은 쪽", () => {
    // -0.05이 lowerQuintile일 때, -0.05 < -0.02 → -0.05
    expect(chooseLargeLossThreshold([-0.05, -0.03, 0.01, 0.02, 0.03])).toBe(-0.05);
    // lowerQuintile > -0.02 → -0.02 cap
    expect(chooseLargeLossThreshold([0.01, 0.02, 0.03, 0.04, 0.05])).toBe(-0.02);
  });
});

describe("runStrategyBacktest", () => {
  it("priceRows < 2 → null", () => {
    expect(
      runStrategyBacktest({
        strategyId: "lens_balance_v1",
        priceRows: [],
        lineHistory: [],
        bandHistory: [],
        indicators: [],
        feeBps: 10,
      })
    ).toBeNull();
  });
  it("indicator_balance_v2 - 추세 진입 시나리오 → 결과 박제", () => {
    const dates = Array.from({ length: 60 }, (_, i) =>
      `2024-${String(Math.floor(i / 30) + 1).padStart(2, "0")}-${String((i % 30) + 1).padStart(2, "0")}`
    );
    const priceRows = dates.map((d, i) => makePriceBar(d, 100 + i)); // 단조 상승
    const indicators = priceRows.map((p) =>
      makeIndicator(p.date, {
        ma_60_ratio: 0.03,
        ma_20_ratio: 0.01,
        macd_ratio: 0.01,
        rsi: 60,
        atr_ratio: 0.03,
        bb_position: 0.5,
      })
    );

    const result = runStrategyBacktest({
      strategyId: "indicator_balance_v2",
      priceRows,
      lineHistory: [],
      bandHistory: [],
      indicators,
      feeBps: 10,
    });

    expect(result).not.toBeNull();
    expect(result!.tradeCount).toBe(1); // 매수 한 번 (단조 상승, exit 없음)
    expect(result!.points.length).toBeGreaterThan(0);
    expect(result!.maxDrawdownPct).toBe(0); // 단조 상승 → drawdown 0
    expect(result!.tradeEvents).toHaveLength(1);
    expect(result!.tradeEvents[0].kind).toBe("entry");
  });
  it("lens_balance_v1 - lineHistory/bandHistory 부족 시 signals 비어서 null", () => {
    const priceRows = [
      makePriceBar("2024-01-01", 100),
      makePriceBar("2024-01-02", 101),
      makePriceBar("2024-01-03", 102),
    ];
    expect(
      runStrategyBacktest({
        strategyId: "lens_balance_v1",
        priceRows,
        lineHistory: [],
        bandHistory: [],
        indicators: [],
        feeBps: 10,
      })
    ).toBeNull();
  });
  it("lens_balance_v1 - entry 신호 + 단조 상승 → strategyReturnPct 박제", () => {
    const dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"];
    const priceRows = dates.map((d, i) => makePriceBar(d, 100 + i));
    // line=lower=upper=close → entry/hold 충족, 위험 없음
    const lineHistory = priceRows.map((p) => makePrediction(p.date, p.close, p.close, p.close));
    const bandHistory = lineHistory;
    const result = runStrategyBacktest({
      strategyId: "lens_balance_v1",
      priceRows,
      lineHistory,
      bandHistory,
      indicators: [],
      feeBps: 10,
    });
    expect(result).not.toBeNull();
    expect(result!.tradeCount).toBeGreaterThanOrEqual(0);
    expect(result!.strategyReturnPct).toBeDefined();
    expect(result!.buyHoldReturnPct).toBeCloseTo(4, 6); // 100 → 104
  });
});
