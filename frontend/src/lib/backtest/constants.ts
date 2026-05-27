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

export const BACKEND_STRATEGY_IDS = new Set<StrategyId>([
  "indicator_balance_v2",
  "ai_balance_v2",
  "ai_band_defense_v1",
]);

export const LENS_BALANCE_RULE = {
  lineEntryThreshold: -0.02,
  lineHoldThreshold: -0.06,
  lowerRiskThreshold: -0.06,
  widthExpansionThreshold: 1.25,
  widthPercentileThreshold: 0.75,
  confirmDays: 3,
  reentryConfirmDays: 2,
};

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

export const SIGNAL_GROUPS: Array<{ id: SignalGroupId; title: string; description: string }> = [
  { id: "buy", title: "매수 후보", description: "선택한 전략 기준으로 신규 진입을 검토할 수 있는 종목입니다." },
  { id: "hold", title: "보유 유지", description: "선택한 전략상 보유 상태를 이어가는 종목입니다." },
  { id: "risk", title: "위험 확대", description: "전략 기준이 약해져 방어 확인이 필요한 종목입니다." },
  { id: "watch", title: "관망", description: "진입 기준이 아직 충분하지 않거나 신호가 부족한 종목입니다." },
];

export const SIGNAL_GROUP_DEFAULT_LIMIT: Record<SignalGroupId, number> = {
  buy: 5,
  hold: 5,
  risk: 5,
  watch: 5,
};

export const SIGNAL_GROUP_MAX_LIMIT = 30;

export const STRATEGIES: StrategyDefinition[] = [
  {
    id: "indicator_balance_v2",
    label: "지표 균형 v2",
    shortLabel: "지표 균형",
    description:
      "AI 예측 없이 60일 추세, 20일 추세, MACD, RSI, ATR, Bollinger 위치만으로 티커별 보유와 현금 대기를 판단하는 기준선 전략입니다.",
    ruleRows: [
      ["전략 단위", "각 티커별 100% 보유 또는 100% 현금"],
      ["진입", "60일 추세 +2% 이상, 20일 추세 -2% 이상, MACD 양수, RSI 75 미만"],
      ["저가 반등 진입", "60일 추세가 살아 있고 Bollinger 위치가 낮으며 RSI 55 미만"],
      ["청산", "60일 추세 -5% 이하 또는 20일 추세 -5% 이하"],
      ["변동성 방어", "ATR 7% 이상이고 20일 추세가 약하면 청산 후보"],
      ["확인일", "진입 2일 확인, 청산 3일 확인"],
      ["역할", "AI 없이 비교하기 위한 보조지표-only 기준선"],
    ],
    visibleFactors: ["ma60Trend", "ma20Trend", "macd", "rsi", "atr"],
    validationRows: [
      ["CP157 전략 수익률", "22.95%"],
      ["CP157 단순 보유 평균", "30.91%"],
      ["CP157 MDD 개선폭", "+5.82%p"],
      ["CP157 손실 회피율", "37.15%"],
      ["CP157 시장 참여율", "63.52%"],
      ["검증 티커", "500개"],
    ],
    scopeNote:
      "포트폴리오 추천이 아니라 각 티커에 같은 long/cash 룰을 독립 적용한 단일 티커 백테스트입니다. 500티커 평가는 티커별 결과의 평균으로 봅니다.",
    usesAi: false,
  },
  {
    id: "ai_balance_v2",
    label: "AI 균형 v2",
    shortLabel: "AI 균형",
    description:
      "AI line은 방향과 보유 의지로, AI band는 하방 위험과 불확실성 확인용으로 쓰는 티커별 long/cash 전략입니다.",
    ruleRows: [
      ["전략 단위", "각 티커별 100% 보유 또는 100% 현금"],
      ["사용 AI", "1D Line + 1D Band"],
      ["진입", "line_score -2% 이상, 60일 추세 0% 이상, 20일 추세 -4% 이상"],
      ["밴드 확인", "하방 밴드 -6% 이상 또는 밴드 폭 확장 1.25배 미만"],
      ["청산", "line_score -6% 미만이면서 하방 밴드나 밴드 폭 위험이 동시에 발생"],
      ["가격 방어", "20일 추세 -10% 이하 또는 ATR 급등과 단기 추세 약화"],
      ["확인일", "진입 2일 확인, 청산 3일 확인"],
    ],
    visibleFactors: ["conservative", "lowerBand", "bandWidth", "bandExpansion", "ma60Trend", "ma20Trend", "atr"],
    validationRows: [
      ["빠른 재평가 전략 수익률", "19.6%"],
      ["빠른 재평가 단순 보유", "32.9%"],
      ["빠른 재평가 MDD 개선폭", "+9.5%p"],
      ["빠른 재평가 손실 회피율", "41.9%"],
      ["빠른 재평가 시장 참여율", "56.0%"],
      ["상태", "보류/설명용 후보"],
    ],
    scopeNote:
      "AI line과 AI band를 하나의 composite 모델로 합성하지 않습니다. 두 AI 지표를 각각 읽어 티커별 보유/현금 판단에만 사용합니다.",
    usesAi: true,
  },
  {
    id: "ai_band_defense_v1",
    label: "AI 밴드 방어 v1",
    shortLabel: "밴드 방어",
    description:
      "AI 지표 중 1D band만 사용합니다. 방향성은 가격 보조지표가 담당하고, AI band는 위험 veto와 청산 확인에만 쓰는 방어 전략입니다.",
    ruleRows: [
      ["전략 단위", "각 티커별 100% 보유 또는 100% 현금"],
      ["사용 AI", "1D Band만 사용"],
      ["진입", "60일 추세 +2% 이상, 20일 추세 -3% 이상, RSI 82 미만"],
      ["반등 진입", "60일 추세가 살아 있고 Bollinger 위치가 낮으며 RSI 60 미만"],
      ["밴드 veto", "하방 밴드 -8% 미만이면서 밴드 폭 1.60배 이상이면 진입 제한"],
      ["청산", "밴드 stress, 60/20일 추세 이탈, 또는 ATR 급등"],
      ["확인일", "진입 2일 확인, 청산 3일 확인"],
    ],
    visibleFactors: ["lowerBand", "bandWidth", "bandExpansion", "ma60Trend", "ma20Trend", "rsi", "atr"],
    validationRows: [
      ["빠른 재평가 전략 수익률", "23.8%"],
      ["빠른 재평가 단순 보유", "32.9%"],
      ["빠른 재평가 MDD 개선폭", "+5.2%p"],
      ["빠른 재평가 손실 회피율", "32.0%"],
      ["빠른 재평가 시장 참여율", "67.0%"],
      ["상태", "방어 후보"],
    ],
    scopeNote:
      "AI line은 사용하지 않습니다. AI band가 가격 지표 기반 진입을 막거나 위험 확대를 확인하는 보조 역할만 합니다.",
    usesAi: true,
  },
];

export function getStrategyDefinition(strategyId: StrategyId) {
  return STRATEGIES.find((strategy) => strategy.id === strategyId) ?? STRATEGIES[0];
}

export function strategyNeedsLine(strategyId: StrategyId) {
  return strategyId === "ai_balance_v2";
}

export function strategyNeedsBand(strategyId: StrategyId) {
  return strategyId === "ai_balance_v2" || strategyId === "ai_band_defense_v1";
}

export function isBackendStrategy(strategyId: StrategyId) {
  return BACKEND_STRATEGY_IDS.has(strategyId);
}
