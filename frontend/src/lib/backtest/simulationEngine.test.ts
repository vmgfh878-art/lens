// CP230 forward-stub — CP231에서 simulationEngine.ts 추출 후 import 연결 + skip 해제.
//
// 현재는 BacktestView.tsx 내 인라인 시뮬레이션 루프. CP231이 모듈로 추출하면
// 아래 describe.skip을 풀고 expect 본체를 채우면 회귀가 잡힌다.
//
// 입력 형태(예상): 전략 시뮬레이션 input(예측 시리즈, 가격, 비용/슬리피지 등)
// 출력 형태(예상): BacktestSimulationResult({strategyReturnPct, buyHoldReturnPct,
//                  buyHoldReturnRatio, maxDrawdownImprovementPct,
//                  largeLossAvoidanceRate, tradeCount, ...})

import { describe, it } from "vitest";

describe.skip("simulationEngine (CP231 추출 후 활성)", () => {
  it.todo("AAPL 1D 보수적 라인 + 가격 → strategyReturnPct/tradeCount 박제");
  it.todo("buyHoldReturnPct 음수 케이스 → 방어 우위 판정 유지");
  it.todo("tradeCount 경계 (20/40) 분류 유지");
  it.todo("maxDrawdownImprovementPct 0/5/음수 분기 유지");
});
