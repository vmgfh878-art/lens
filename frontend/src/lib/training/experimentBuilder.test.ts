// CP230 forward-stub — CP233에서 experimentBuilder.ts 추출 후 import 연결 + skip 해제.
//
// 현재는 TrainingView.tsx:1104 `buildExperimentCards(detail: AiRunDetail)` 인라인.
// CP233이 모듈로 추출하면 아래 describe.skip을 풀고 expect 본체를 채우면 된다.

import { describe, it } from "vitest";

describe.skip("experimentBuilder.buildExperimentCards (CP233 추출 후 활성)", () => {
  it.todo("AiRunDetail 입력 → 카드 배열 박제 (role/model/seed 등)");
  it.todo("completed/failed_nan/failed_quality_gate status 분류 유지");
  it.todo("ensemble seeds 표현 유지 (PatchTST F4 β=4 등)");
  it.todo("초기 계획 평가 테이블 행 순서 유지");
});
