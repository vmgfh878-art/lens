// Backtest 페이지의 전략 / 신호 그룹 / 규칙 정의.

import { PRODUCT_SLOT_BY_ID } from "@/lib/productSlots";

import type { SignalGroupId, StrategyDefinition, StrategyId } from "./types";

export const PRODUCT_LINE_RUN_ID = PRODUCT_SLOT_BY_ID["line-1d"].runId ?? "";
export const PRODUCT_BAND_RUN_ID = PRODUCT_SLOT_BY_ID["band-1d"].runId ?? "";
export const DEFAULT_PRICE_LOOKBACK_DAYS = 365;
export const PREDICTION_HISTORY_LIMIT = 200;
export const DEFAULT_FEE_BPS = 10;

export const SIGNAL_SCAN_TICKERS = [
  "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "JPM", "XOM",
];

export const LENS_BALANCE_RULE = {
  lineEntryThreshold: -0.002,
  lineHoldThreshold: -0.014,
  lowerRiskThreshold: -0.05,
  widthExpansionThreshold: 1.1,
  widthPercentileThreshold: 0.75,
  confirmDays: 2,
  reentryConfirmDays: 2,
};

export const LENS_BALANCE_VALIDATION = [
  ["전략 수익률", "3.98%"],
  ["단순 보유 수익률", "2.37%"],
  ["전략 MDD", "-5.93%"],
  ["단순 보유 MDD", "-13.08%"],
  ["손실 회피율", "55.6%"],
  ["시장 참여율", "44.7%"],
  ["강한 상승 방어율", "67.1%"],
  ["검증 티커", "95개"],
] as const;

export const INDICATOR_BASELINE_RULE = {
  ma60Entry: 0.02,
  ma20Entry: -0.02,
  ma60Exit: -0.05,
  ma20Exit: -0.05,
  macdEntry: 0,
  rsiEntryCap: 75,
  atrExit: 0.07,
  pullbackBb: 0.35,
  pullbackRsi: 55,
  entryConfirmDays: 2,
  exitConfirmDays: 3,
};

export const INDICATOR_BASELINE_VALIDATION = [
  ["전략 수익률", "22.95%"],
  ["500 동일가중 시장", "16.87%"],
  ["단순 보유 평균", "30.91%"],
  ["전략 MDD", "-20.13%"],
  ["단순 보유 MDD", "-25.95%"],
  ["MDD 개선폭", "+5.82%"],
  ["시장 참여율", "63.5%"],
  ["검증 티커", "500개"],
] as const;

export const SIGNAL_GROUPS: Array<{ id: SignalGroupId; title: string; description: string }> = [
  { id: "buy", title: "매수 후보", description: "선택한 전략 기준으로 신규 진입을 검토할 수 있는 종목입니다." },
  { id: "hold", title: "보유 유지", description: "선택한 전략상 보유 상태를 이어가는 종목입니다." },
  { id: "risk", title: "위험 확대", description: "전략 기준이 약해져 방어 확인이 필요한 종목입니다." },
  { id: "watch", title: "관망", description: "진입 기준이 아직 충분하지 않거나 신호가 부족한 종목입니다." },
];

export const SIGNAL_GROUP_DEFAULT_LIMIT: Record<SignalGroupId, number> = {
  buy: 3,
  hold: 3,
  risk: 3,
  watch: 0,
};

export const SIGNAL_GROUP_MAX_LIMIT = 10;

export const STRATEGIES: StrategyDefinition[] = [
  {
    id: "indicator_baseline_v1",
    label: "지표 기준선 v1",
    shortLabel: "지표 기준선",
    description:
      "AI 제품 슬롯과 분리해, 60일 추세, 20일 추세, MACD, RSI, ATR만으로 비교하는 연구 기준선입니다.",
    ruleRows: [
      ["사용 지표", "60일 추세, 20일 추세, MACD, RSI, ATR, Bollinger 위치"],
      ["진입", "60일 추세 +2% 이상, 20일 추세 -2% 이상, MACD 양수, RSI 75 미만"],
      ["저가 반등 진입", "60일 추세가 살아 있고 Bollinger 위치가 낮으며 RSI 55 미만"],
      ["청산", "60일 추세 -5% 이하 또는 20일 추세 -5% 이하"],
      ["변동성 방어", "ATR 7% 이상이고 20일 추세가 약하면 청산 후보"],
      ["확인일", "진입 2일 확인, 청산 3일 확인"],
      ["포지션", "단일 티커 100% 또는 현금 100%"],
      ["판정", "제품 provenance가 아닌 연구 기준선"],
    ],
    visibleFactors: ["ma60Trend", "ma20Trend", "macd", "rsi", "atr"],
    validationRows: INDICATOR_BASELINE_VALIDATION,
    scopeNote:
      "최근 1년 500티커 기준으로 동일가중 시장 proxy는 넘었지만, 각 티커 단순 보유 평균 수익률은 넘지 못했습니다. 기준선 전략으로만 봅니다.",
    usesAi: false,
  },
  {
    id: "lens_balance_v1",
    label: "Lens Balance v1",
    shortLabel: "Lens Balance",
    description:
      "1D 보수적 기준선과 자동 갱신된 1D AI 밴드를 함께 읽는 단일 티커 연구 전략입니다. 제품 provenance와 연구 기준선은 분리해서 봅니다.",
    ruleRows: [
      ["사용 지표", "보수적 기준선, 자동 갱신된 1D AI 밴드"],
      ["제품 예측선 슬롯", `${PRODUCT_SLOT_BY_ID["line-1d"].displayName} · ${PRODUCT_SLOT_BY_ID["line-1d"].sourceCp}`],
      ["제품 밴드 슬롯", `${PRODUCT_SLOT_BY_ID["band-1d"].displayName} · ${PRODUCT_SLOT_BY_ID["band-1d"].sourceCp}`],
      ["예측선 계약", "F4 β=4 ensemble v1. raw output은 safe_line_score이며 화면에서는 가격으로 환산합니다."],
      ["진입", "보수적 기준선 >= -0.2%, 밴드 폭 급확장 아님"],
      ["보유", "보수적 기준선 >= -1.4%, line 약화와 밴드 위험이 동시에 확정되지 않음"],
      ["청산", "line 약화 + 밴드 하단 위험 또는 밴드 폭 확장"],
      ["확인일", "청산 2일 확인, 재진입 2일 확인"],
      ["포지션", "단일 티커 100% 또는 현금 100%"],
      ["판정", "제품 기본 후보가 아니라 AI 지표 해석용 실험 전략 v1"],
    ],
    visibleFactors: ["conservative", "lowerBand", "bandWidth", "bandExpansion", "bandPercentile"],
    validationRows: LENS_BALANCE_VALIDATION,
    scopeNote:
      "CP109/CP110 95개 티커 holdout 기준 결과입니다. 제품 기본 후보가 아니라 AI 지표 해석용 실험 전략 v1입니다.",
    usesAi: true,
  },
];

export function getStrategyDefinition(strategyId: StrategyId) {
  return STRATEGIES.find((strategy) => strategy.id === strategyId) ?? STRATEGIES[0];
}

export function strategyNeedsLine(strategyId: StrategyId) {
  return strategyId === "lens_balance_v1";
}

export function strategyNeedsBand(strategyId: StrategyId) {
  return strategyId === "lens_balance_v1";
}
