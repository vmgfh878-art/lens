// CP230 forward-stub — CP232에서 signalBuilder.ts 추출 후 import 연결 + skip 해제.
//
// 현재는 BacktestView.tsx:350 `buildSignals(params: {...})` 인라인. CP232가
// 모듈로 추출하면 아래 describe.skip을 풀고 expect 본체를 채우면 된다.

import { describe, it } from "vitest";

describe.skip("signalBuilder.buildSignals (CP232 추출 후 활성)", () => {
  it.todo("AAPL 보수적 라인 입력 → 매매 신호 시리즈 박제");
  it.todo("밴드 하단 돌파 시 신호 발생 분기 유지");
  it.todo("빈 입력 → 빈 신호");
  it.todo("conservative_series 우선 fallback to line_series 동작 유지");
});
