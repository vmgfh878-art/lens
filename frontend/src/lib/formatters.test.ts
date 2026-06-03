// CP230 characterization — lib/formatters.ts 박제.
// Intl ko-KR maximumFractionDigits 동작에 의존. 로케일 환경은
// playwright.config 의 locale 설정과 머신 환경에 따라 다를 수 있어, 단정은
// 부호/구두점 위주로 두고 천단위 구분자 등 표기 차이는 보고서에 명시.

import { describe, expect, it } from "vitest";

import {
  finiteOrNull,
  formatCompact,
  formatNumber,
  formatPercent,
  formatRatio,
  formatRatioAsPercent,
  formatUnsignedRatioAsPercent,
  isFiniteNumber,
  normalizeRsi,
  ratioToPercent,
} from "./formatters";

describe("formatNumber", () => {
  it("null/undefined/NaN → '-'", () => {
    expect(formatNumber(null)).toBe("-");
    expect(formatNumber(undefined)).toBe("-");
    expect(formatNumber(Number.NaN)).toBe("-");
  });
  it("기본 digits=2, Intl ko-KR", () => {
    // Intl maximumFractionDigits=2 → 정수는 그대로, 소수는 잘림.
    expect(formatNumber(1.234)).toMatch(/^1[.,]23$/);
    expect(formatNumber(2)).toBe("2");
  });
});

describe("formatPercent", () => {
  it("null → '-'", () => {
    expect(formatPercent(null)).toBe("-");
  });
  it("양수에 '+' 부호", () => {
    expect(formatPercent(5)).toMatch(/^\+5([.,]\d+)?%$/);
  });
  it("음수에 부호 그대로(-)", () => {
    expect(formatPercent(-3)).toMatch(/^-3([.,]\d+)?%$/);
  });
  it("0은 양수 분기 → '+'", () => {
    expect(formatPercent(0)).toMatch(/^\+0%$/);
  });
});

describe("formatRatio", () => {
  it("null → '-'", () => {
    expect(formatRatio(null)).toBe("-");
  });
  it("|value| ≤ 1 → *100 후 1자리 + '%'", () => {
    expect(formatRatio(0.5)).toMatch(/^50%$/);
    expect(formatRatio(1)).toMatch(/^100%$/);
    expect(formatRatio(-0.25)).toMatch(/^-25%$/);
  });
  it("|value| > 1 → formatNumber(value, 2) 그대로(부호 없음, % 없음)", () => {
    expect(formatRatio(2)).toBe("2");
    expect(formatRatio(2.5)).toMatch(/^2[.,]5$/);
  });
});

describe("formatRatioAsPercent", () => {
  it("0.05 → '+5.00%' (Intl 결과는 천단위 없을 때)", () => {
    expect(formatRatioAsPercent(0.05)).toMatch(/^\+5%$/);
  });
  it("null → '-'", () => {
    expect(formatRatioAsPercent(null)).toBe("-");
  });
});

describe("formatUnsignedRatioAsPercent", () => {
  it("0.05 → '5%' (부호 없음)", () => {
    expect(formatUnsignedRatioAsPercent(0.05)).toMatch(/^5%$/);
  });
  it("null → '-'", () => {
    expect(formatUnsignedRatioAsPercent(null)).toBe("-");
  });
});

describe("formatCompact", () => {
  it("null → '-'", () => {
    expect(formatCompact(null)).toBe("-");
  });
  it("compact notation 정상 동작 (정확한 표기는 머신 의존)", () => {
    const result = formatCompact(12345);
    expect(typeof result).toBe("string");
    expect(result).not.toBe("-");
  });
});

describe("normalizeRsi", () => {
  it("0~1 범위 → *100", () => {
    expect(normalizeRsi(0.5)).toBe(50);
    expect(normalizeRsi(0)).toBe(0);
    expect(normalizeRsi(1)).toBe(100);
  });
  it("1 초과 → 그대로", () => {
    expect(normalizeRsi(70)).toBe(70);
    expect(normalizeRsi(100)).toBe(100);
  });
  it("음수도 그대로 (0~1 범위 아님)", () => {
    expect(normalizeRsi(-0.5)).toBe(-0.5);
  });
});

describe("finiteOrNull", () => {
  it("finite number → value", () => {
    expect(finiteOrNull(3.14)).toBe(3.14);
    expect(finiteOrNull(0)).toBe(0);
  });
  it("NaN/Infinity → null", () => {
    expect(finiteOrNull(Number.NaN)).toBeNull();
    expect(finiteOrNull(Infinity)).toBeNull();
    expect(finiteOrNull(-Infinity)).toBeNull();
  });
  it("string/null/undefined/boolean → null", () => {
    expect(finiteOrNull("3")).toBeNull();
    expect(finiteOrNull(null)).toBeNull();
    expect(finiteOrNull(undefined)).toBeNull();
    expect(finiteOrNull(true)).toBeNull();
  });
});

describe("isFiniteNumber type guard", () => {
  it("finite number → true", () => {
    expect(isFiniteNumber(0)).toBe(true);
    expect(isFiniteNumber(-1.5)).toBe(true);
  });
  it("NaN/Infinity/string/null → false", () => {
    expect(isFiniteNumber(Number.NaN)).toBe(false);
    expect(isFiniteNumber(Infinity)).toBe(false);
    expect(isFiniteNumber("3")).toBe(false);
    expect(isFiniteNumber(null)).toBe(false);
  });
});

describe("ratioToPercent", () => {
  it("*100", () => {
    expect(ratioToPercent(0.5)).toBe(50);
    expect(ratioToPercent(0)).toBe(0);
    expect(ratioToPercent(-0.25)).toBe(-25);
  });
});
