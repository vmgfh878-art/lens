// CP231 Step C — signalBuilder.ts 추출 직후 결과를 박제.
// 각 분기 경계와 빈 입력을 deterministic하게 잡아 이후 Step의 회귀를 가드한다.

import { describe, expect, it } from "vitest";

import type { IndicatorPoint, PredictionResult, PriceBar } from "@/api/client";

import {
  buildIndicatorSignals,
  buildSignals,
  classifyIndicatorSignal,
  classifySignalGroup,
  getBandWidthState,
  getLatestIndicatorBefore,
  getRawTarget,
  normalizeRsi,
} from "./signalBuilder";
import type { RiskSignal } from "./types";

function makePriceBar(date: string, close: number): PriceBar {
  return { date, open: close, high: close, low: close, close, volume: null };
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

function makeRiskSignal(overrides: Partial<RiskSignal> = {}): RiskSignal {
  return {
    date: "2024-01-01",
    position: 0,
    targetPosition: 0,
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
    reason: "테스트",
    ...overrides,
  } as RiskSignal;
}

describe("normalizeRsi", () => {
  it("0~1 범위는 ×100 (proportion → percent)", () => {
    expect(normalizeRsi(0.5)).toBe(50);
    expect(normalizeRsi(0)).toBe(0);
    expect(normalizeRsi(1)).toBe(100);
  });
  it("이미 percent면 그대로", () => {
    expect(normalizeRsi(50)).toBe(50);
    expect(normalizeRsi(75.5)).toBe(75.5);
  });
  it("undefined / null / NaN → null", () => {
    expect(normalizeRsi(undefined)).toBeNull();
    expect(normalizeRsi(null)).toBeNull();
    expect(normalizeRsi(Number.NaN)).toBeNull();
  });
});

describe("getBandWidthState", () => {
  it("bandWidthReturn null → '-'", () => {
    expect(getBandWidthState({ bandWidthReturn: null, bandWidthExpansion: 2, bandWidthPercentile: 0.9 })).toBe("-");
  });
  it("expansion ≥ 1.25 → '확장'", () => {
    expect(getBandWidthState({ bandWidthReturn: 0.05, bandWidthExpansion: 1.25, bandWidthPercentile: null })).toBe("확장");
    expect(getBandWidthState({ bandWidthReturn: 0.05, bandWidthExpansion: 1.30, bandWidthPercentile: 0.5 })).toBe("확장");
  });
  it("percentile ≥ 0.75 (expansion 미만) → '넓음'", () => {
    expect(getBandWidthState({ bandWidthReturn: 0.05, bandWidthExpansion: 1.0, bandWidthPercentile: 0.75 })).toBe("넓음");
    expect(getBandWidthState({ bandWidthReturn: 0.05, bandWidthExpansion: null, bandWidthPercentile: 0.8 })).toBe("넓음");
  });
  it("둘 다 임계 미만 → '보통'", () => {
    expect(getBandWidthState({ bandWidthReturn: 0.05, bandWidthExpansion: 1.0, bandWidthPercentile: 0.5 })).toBe("보통");
  });
});

describe("getRawTarget", () => {
  it("lineEntryThreshold 통과 + 확장 없음 → target 1 (진입)", () => {
    expect(
      getRawTarget({ conservativeReturn: -0.01, lowerBandReturn: -0.02, bandWidthExpansion: 1.0 })
    ).toEqual({
      target: 1,
      reason: "보수적 기준선이 진입 기준을 충족하고 밴드 폭 급확장이 없습니다.",
    });
  });
  it("진입 아래 + hold 위 + 위험 없음 → target 1 (보유)", () => {
    expect(
      getRawTarget({ conservativeReturn: -0.04, lowerBandReturn: -0.02, bandWidthExpansion: 1.0 }).target
    ).toBe(1);
  });
  it("line 약함 + lower 위험 + 확장 동시 → target 0", () => {
    expect(
      getRawTarget({ conservativeReturn: -0.1, lowerBandReturn: -0.1, bandWidthExpansion: 1.5 })
    ).toEqual({
      target: 0,
      reason: "예측선 약화와 밴드 하단 위험, 밴드 폭 확장이 동시에 나타났습니다.",
    });
  });
  it("진입/보유 미달 → target 0 (대기)", () => {
    expect(
      getRawTarget({ conservativeReturn: -0.1, lowerBandReturn: 0, bandWidthExpansion: 1.0 }).target
    ).toBe(0);
  });
});

describe("classifySignalGroup", () => {
  it("targetPosition 1 & position 0 → buy", () => {
    expect(classifySignalGroup(makeRiskSignal({ targetPosition: 1, position: 0 }))).toEqual({
      group: "buy",
      label: "매수 후보",
    });
  });
  it("position 1 → hold", () => {
    expect(classifySignalGroup(makeRiskSignal({ targetPosition: 0, position: 1 }))).toEqual({
      group: "hold",
      label: "전략상 보유 유지",
    });
  });
  it("lower 위험 또는 확장 또는 line 약함 → risk", () => {
    expect(
      classifySignalGroup(
        makeRiskSignal({ targetPosition: 0, position: 0, lowerBandReturn: -0.1 })
      ).group
    ).toBe("risk");
    expect(
      classifySignalGroup(
        makeRiskSignal({ targetPosition: 0, position: 0, bandWidthExpansion: 1.5 })
      ).group
    ).toBe("risk");
  });
  it("위험 없음 → watch", () => {
    expect(
      classifySignalGroup(
        makeRiskSignal({
          targetPosition: 0,
          position: 0,
          conservativeReturn: 0.1,
          lowerBandReturn: 0.1,
          bandWidthExpansion: 0.5,
        })
      ).group
    ).toBe("watch");
  });
});

describe("classifyIndicatorSignal", () => {
  it("targetPosition 1 & position 0 → buy", () => {
    expect(classifyIndicatorSignal(makeRiskSignal({ targetPosition: 1, position: 0 }))).toEqual({
      group: "buy",
      label: "매수 후보",
    });
  });
  it("position 1 → hold", () => {
    expect(classifyIndicatorSignal(makeRiskSignal({ targetPosition: 0, position: 1 }))).toEqual({
      group: "hold",
      label: "보유 유지",
    });
  });
  it("ma60 또는 ma20 약함 → risk", () => {
    expect(
      classifyIndicatorSignal(
        makeRiskSignal({ targetPosition: 0, position: 0, ma60Ratio: -0.1 })
      ).group
    ).toBe("risk");
  });
  it("reason에 '변동성' 포함 → risk", () => {
    expect(
      classifyIndicatorSignal(
        makeRiskSignal({
          targetPosition: 0,
          position: 0,
          ma60Ratio: 0.05,
          ma20Ratio: 0.05,
          reason: "변동성이 커지고 단기 추세가 약해져 매도합니다.",
        })
      ).group
    ).toBe("risk");
  });
});

describe("getLatestIndicatorBefore", () => {
  const indicators = [
    makeIndicator("2024-01-01"),
    makeIndicator("2024-01-03"),
    makeIndicator("2024-01-05"),
  ];
  it("date null → 마지막", () => {
    expect(getLatestIndicatorBefore(indicators, null)?.date).toBe("2024-01-05");
  });
  it("date 일치 or 이전의 가장 최근", () => {
    expect(getLatestIndicatorBefore(indicators, "2024-01-04")?.date).toBe("2024-01-03");
    expect(getLatestIndicatorBefore(indicators, "2024-01-03")?.date).toBe("2024-01-03");
  });
  it("모든 indicator 이후 date → 마지막", () => {
    expect(getLatestIndicatorBefore(indicators, "2024-01-10")?.date).toBe("2024-01-05");
  });
  it("모든 indicator 이전 date → 마지막 (fallback)", () => {
    expect(getLatestIndicatorBefore(indicators, "2023-12-01")?.date).toBe("2024-01-05");
  });
});

describe("buildSignals", () => {
  it("빈 입력 → 빈 배열", () => {
    expect(buildSignals({ priceRows: [], lineHistory: [], bandHistory: [] })).toEqual([]);
  });
  it("정상 입력 → date 정렬 + position/targetPosition/reason 박제", () => {
    const dates = ["2024-01-01", "2024-01-02", "2024-01-03"];
    const priceRows = dates.map((d, i) => makePriceBar(d, 100 - i));
    // lower=upper=line=close → conservativeReturn=0, lowerBandReturn=0, bandWidthReturn=0
    // 0 ≥ -0.02 (lineEntryThreshold), 0 > -0.06 (lowerRiskThreshold), expansion=null → entryOk
    const lineHistory = priceRows.map((p) =>
      makePrediction(p.date, p.close, p.close, p.close)
    );
    const bandHistory = lineHistory;
    const signals = buildSignals({ priceRows, lineHistory, bandHistory });
    expect(signals).toHaveLength(3);
    expect(signals[0].date).toBe("2024-01-01");
    expect(signals[0].targetPosition).toBe(1);
    expect(signals[0].position).toBe(0);
    // reentryConfirmDays=2 → entryStreak 2일째에 진입 확정 (signals[1])
    expect(signals[1].position).toBe(1);
    expect(signals[2].position).toBe(1);
  });
});

describe("buildIndicatorSignals", () => {
  it("빈 priceRows → 빈 배열", () => {
    expect(buildIndicatorSignals({ priceRows: [], indicators: [] })).toEqual([]);
  });
  it("rsi prop 정규화 + ma60/ma20 finite 필터", () => {
    const priceRows = Array.from({ length: 60 }, (_, i) =>
      makePriceBar(
        `2024-${String(Math.floor(i / 30) + 1).padStart(2, "0")}-${String((i % 30) + 1).padStart(2, "0")}`,
        100 + i
      )
    );
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
    const signals = buildIndicatorSignals({ priceRows, indicators });
    expect(signals.length).toBeGreaterThan(0);
    // trendEntry 충족 → target 1, entryConfirmDays=2일 후 position=1로 전환
    expect(signals[0].targetPosition).toBe(1);
    expect(signals.at(-1)?.position).toBe(1);
  });
});
