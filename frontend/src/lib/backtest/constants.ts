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
  "lineband_risk_guard",
  "line_defense",
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
      ["전략 수익률", "22.95%"],
      ["단순 보유 평균", "30.91%"],
      ["MDD 개선폭", "+5.82%p"],
      ["손실 회피율", "37.15%"],
      ["시장 참여율", "63.52%"],
      ["검증 티커", "500개"],
    ],
    scopeNote:
      "포트폴리오 추천이 아니라 각 티커에 같은 long/cash 룰을 독립 적용한 단일 티커 백테스트입니다. 500티커 평가는 티커별 결과의 평균으로 봅니다.",
    usesAi: false,
  },
  {
    id: "lineband_risk_guard",
    label: "라인밴드 리스크 가드",
    shortLabel: "리스크 가드",
    description:
      "모멘텀(20일 수익률·MACD 가속)으로 진입하되, AI 라인이 약하거나 AI 밴드가 위험-상태(이 종목 기준 깊은 하단/넓은 폭 = 고변동·꼬리위험)면 진입을 막고 현금으로 빠지는 보수적 리스크 가드입니다. 수익을 키우는 전략이 아니라, 위험할 때 빠져 낙폭과 대형손실을 줄이는 방어 전략입니다(참여율 낮음·현금 비중 큼).",
    ruleRows: [
      ["전략 단위", "각 티커별 100% 보유 또는 100% 현금"],
      ["사용 AI", "1D 라인 + 1D 밴드"],
      ["진입", "20일 수익률 +2% 이상 & MACD 가속 ≥ 0 & MA5 ≥ 0 & RSI 80 미만"],
      ["AI 라인 확인", "AI 라인이 이 종목 기준 중간분위(p50) 이상"],
      ["밴드 위험-상태 veto", "하방 밴드가 p10 이하로 깊거나 밴드 폭이 p90 이상으로 넓으면 진입 차단·청산"],
      ["청산", "20일 수익률 음전 또는 추세·변동성 붕괴 또는 밴드 위험-상태"],
      ["확인일", "진입 2일 확인, 청산 3일 확인"],
      ["성격", "방어 전용 · 위험할 때 현금 (수익 개선 아님)"],
    ],
    visibleFactors: ["conservative", "lowerBand", "bandWidth", "bandExpansion", "ma60Trend", "ma20Trend", "macd", "rsi", "atr"],
    validationRows: [
      ["AI 평가창", "~471티커, 2025-06~2026-06"],
      ["전략 수익률", "3.3% (단순 보유 28.7%)"],
      ["MDD 개선폭(보유 대비)", "+15.3%p"],
      ["손실 회피율", "76.8%"],
      ["시장 참여율", "23% (대부분 현금)"],
      ["held-out OOS 검증", "ΔMaxDD +6.0%p, Bonferroni 통과 (n=255)"],
    ],
    scopeNote:
      "AI는 수익엔 기여하지 못합니다(공격형 전략 OOS null, 상승장 caveat). AI 라인·밴드를 '위험-상태 게이지'로 보고 위험할 때만 현금으로 빠져 낙폭·대형손실을 줄이는 보수적 방어에만 효과가 있습니다. CP252 통제 발굴(티커 200/271 + 시간 val/test OOS 두 축 · 엄격 Bonferroni)로 검증했습니다.",
    usesAi: true,
  },
  {
    id: "line_defense",
    label: "라인 방어",
    shortLabel: "라인 방어",
    description:
      "밴드를 쓰지 않고 AI 라인만으로 위험 국면 진입을 막는 방어 전략입니다. 모멘텀 진입에 AI 라인 확인(이 종목 기준 p60 이상)을 더했습니다. 옛 AI 라인 임계가 잘못 설정돼 '라인은 쓸모없다'고 판단했던 것을, 상대 분위 기준으로 바로잡아 라인이 방어에 쓸모 있음을 보인 '라인 부활' 전략입니다. 수익 개선용은 아닙니다.",
    ruleRows: [
      ["전략 단위", "각 티커별 100% 보유 또는 100% 현금"],
      ["사용 AI", "1D 라인만 사용 (밴드 미사용)"],
      ["진입", "20일 수익률 +2% 이상 & MACD 가속 ≥ 0 & MA5 ≥ 0 & RSI 80 미만"],
      ["AI 라인 확인", "AI 라인이 이 종목 기준 p60 이상(밴드 없이 라인 단독 게이트)"],
      ["청산", "20일 수익률 음전 또는 추세·변동성 붕괴"],
      ["확인일", "진입 2일 확인, 청산 3일 확인"],
      ["성격", "라인 단독 방어 · '라인 부활' 쇼케이스 (수익 개선 아님)"],
    ],
    visibleFactors: ["conservative", "ma60Trend", "ma20Trend", "macd", "rsi", "atr"],
    validationRows: [
      ["AI 평가창", "~471티커, 2025-06~2026-06"],
      ["전략 수익률", "8.6% (단순 보유 28.7%)"],
      ["MDD 개선폭(보유 대비)", "+14.9%p"],
      ["손실 회피율", "73.7%"],
      ["시장 참여율", "25% (대부분 현금)"],
      ["held-out OOS 검증", "ΔMaxDD +4.5%p, Bonferroni 통과 (n=255)"],
    ],
    scopeNote:
      "밴드 없이 AI 라인만으로 위험 국면 노출을 줄이는 보수적 방어입니다. CP248~251에서 '라인은 죽었다'던 결론이 사실은 옛 절대임계(-2%, 진입 63% 차단)가 잘못된 탓이었고, CP252에서 상대 분위 게이트로 바꾸니 방어에 유효함을 OOS·Bonferroni로 확인했습니다. 단 수익에는 기여하지 못합니다(공격형 null).",
    usesAi: true,
  },
];

export function getStrategyDefinition(strategyId: StrategyId) {
  return STRATEGIES.find((strategy) => strategy.id === strategyId) ?? STRATEGIES[0];
}

const LINE_STRATEGIES = new Set<StrategyId>(["lineband_risk_guard", "line_defense"]);

export function strategyNeedsLine(strategyId: StrategyId) {
  return LINE_STRATEGIES.has(strategyId); // CP253 — 발굴 방어 전략 2개 모두 AI 라인 사용.
}

export function strategyNeedsBand(strategyId: StrategyId) {
  return strategyId === "lineband_risk_guard"; // CP253 — 리스크 가드만 밴드 사용.
}

export function isBackendStrategy(strategyId: StrategyId) {
  return BACKEND_STRATEGY_IDS.has(strategyId);
}
