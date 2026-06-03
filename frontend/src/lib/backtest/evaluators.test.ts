// CP230 characterization — backtest/evaluators.ts 의 분기 경계값 박제.
// 운영 코드 무변경. 현재 출력을 그대로 단정. 의심점은 보고서로.

import { describe, expect, it } from "vitest";

import {
  describeAvoidanceStrength,
  evaluateDrawdown,
  evaluateFollowReturn,
  evaluateLossAvoidance,
  evaluateTradeFrequency,
} from "./evaluators";
import type { BacktestSimulationResult } from "./types";

function makeResult(
  overrides: Partial<BacktestSimulationResult> = {},
): BacktestSimulationResult {
  return {
    strategyReturnPct: 0,
    buyHoldReturnPct: 0,
    buyHoldReturnRatio: 1,
    maxDrawdownImprovementPct: 0,
    largeLossAvoidanceRate: 0,
    tradeCount: 0,
    ...overrides,
  } as BacktestSimulationResult;
}

describe("describeAvoidanceStrength", () => {
  it("≥0.6 → '강함'", () => {
    expect(describeAvoidanceStrength(0.6)).toBe("강함");
    expect(describeAvoidanceStrength(1)).toBe("강함");
  });
  it("0.4~0.59 → '보통'", () => {
    expect(describeAvoidanceStrength(0.59)).toBe("보통");
    expect(describeAvoidanceStrength(0.4)).toBe("보통");
  });
  it("<0.4 → '약함'", () => {
    expect(describeAvoidanceStrength(0.39)).toBe("약함");
    expect(describeAvoidanceStrength(0)).toBe("약함");
  });
  it("null/undefined/NaN → '-'", () => {
    expect(describeAvoidanceStrength(null)).toBe("-");
    expect(describeAvoidanceStrength(undefined)).toBe("-");
    expect(describeAvoidanceStrength(Number.NaN)).toBe("-");
  });
});

describe("evaluateFollowReturn", () => {
  it("null result → '-'", () => {
    expect(evaluateFollowReturn(null)).toBe("-");
  });
  it("buyHoldReturnRatio null → '-'", () => {
    expect(
      evaluateFollowReturn(makeResult({ buyHoldReturnRatio: null as unknown as number })),
    ).toBe("-");
  });
  it("방어 우위: buyHold<0 & strategy>buyHold → '방어 우위'", () => {
    expect(
      evaluateFollowReturn(
        makeResult({
          buyHoldReturnPct: -10,
          strategyReturnPct: -3,
          buyHoldReturnRatio: 0.3,
        }),
      ),
    ).toBe("방어 우위");
  });
  it("ratio ≥0.7 → '양호'", () => {
    expect(
      evaluateFollowReturn(
        makeResult({
          buyHoldReturnPct: 5,
          strategyReturnPct: 4,
          buyHoldReturnRatio: 0.7,
        }),
      ),
    ).toBe("양호");
  });
  it("0.4 ≤ ratio < 0.7 → '보통'", () => {
    expect(
      evaluateFollowReturn(
        makeResult({
          buyHoldReturnPct: 5,
          strategyReturnPct: 2,
          buyHoldReturnRatio: 0.4,
        }),
      ),
    ).toBe("보통");
  });
  it("ratio < 0.4 → '약함'", () => {
    expect(
      evaluateFollowReturn(
        makeResult({
          buyHoldReturnPct: 5,
          strategyReturnPct: 1,
          buyHoldReturnRatio: 0.39,
        }),
      ),
    ).toBe("약함");
  });
});

describe("evaluateDrawdown", () => {
  it("null → '-'", () => {
    expect(evaluateDrawdown(null)).toBe("-");
  });
  it("≥5 → '양호'", () => {
    expect(evaluateDrawdown(makeResult({ maxDrawdownImprovementPct: 5 }))).toBe(
      "양호",
    );
  });
  it(">0 & <5 → '보통'", () => {
    expect(
      evaluateDrawdown(makeResult({ maxDrawdownImprovementPct: 0.1 })),
    ).toBe("보통");
  });
  it("≤0 → '약함'", () => {
    expect(evaluateDrawdown(makeResult({ maxDrawdownImprovementPct: 0 }))).toBe(
      "약함",
    );
    expect(
      evaluateDrawdown(makeResult({ maxDrawdownImprovementPct: -1 })),
    ).toBe("약함");
  });
});

describe("evaluateLossAvoidance", () => {
  it("describeAvoidanceStrength 위임", () => {
    expect(
      evaluateLossAvoidance(makeResult({ largeLossAvoidanceRate: 0.7 })),
    ).toBe("강함");
    expect(
      evaluateLossAvoidance(makeResult({ largeLossAvoidanceRate: 0.5 })),
    ).toBe("보통");
    expect(evaluateLossAvoidance(null)).toBe("-");
  });
});

describe("evaluateTradeFrequency", () => {
  it("null → '-'", () => {
    expect(evaluateTradeFrequency(null)).toBe("-");
  });
  it("≤20 → '적정'", () => {
    expect(evaluateTradeFrequency(makeResult({ tradeCount: 20 }))).toBe("적정");
    expect(evaluateTradeFrequency(makeResult({ tradeCount: 0 }))).toBe("적정");
  });
  it("21~40 → '많음'", () => {
    expect(evaluateTradeFrequency(makeResult({ tradeCount: 21 }))).toBe("많음");
    expect(evaluateTradeFrequency(makeResult({ tradeCount: 40 }))).toBe("많음");
  });
  it(">40 → '과도'", () => {
    expect(evaluateTradeFrequency(makeResult({ tradeCount: 41 }))).toBe("과도");
  });
});
