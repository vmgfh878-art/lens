# ADR-0022: TrainingView 1461줄을 8모듈로 분할

Status: Accepted
Date: 2026-06-04
CP: CP232

## 결정

`frontend/src/components/TrainingView.tsx` (시작 1461줄)을 4개 lib + 4개 components로 가른다. 이전에 `lib/training/*`(constants/formatters/runUtils/detailFields/staticEvaluation/reproducibility/usageData/lineTimeline/bandTimeline)와 `components/training/*`(ExperimentTimeline/Line·Band 아카이브)로 시작된 분리 패턴을 이어간다.

- **`lib/training/cardTypes.ts`** — `GoalCardProps` + `GoalCardData` 인터페이스. GoalCard/Grid/StoredEvaluationSection/experimentBuilder가 공유.
- **`lib/training/comparisonTypes.ts`** — `ComparisonRow` 인터페이스. TrainingView 본문(buildComparisonRows/describe*/get*Verdict/getFinalJudgement) + ComparisonTable이 공유.
- **`lib/training/metricAccess.ts`** — `getMetricByKeys` 외 5 헬퍼. experimentBuilder가 호출하고 TrainingView 본문(ExperimentDetail/loadRuns)도 호출.
- **`lib/training/experimentBuilder.ts`** — 5개 빌더 (`buildLine/BandExperimentCards`, `buildExperimentCards`, `hasDisplayableExperimentMetrics`, `getExperimentFailureReason`).
- **`components/training/ProductMetricsSection.tsx`** — `PptMappingSection` + `V1ExtraIndicatorsSection` named export. slotId-accessor 패턴 (UsageData/Reproducibility와 동일).
- **`components/training/ComparisonTable.tsx`** — 실험 상세의 "비교 지표" 표.
- **`components/training/SignificanceSection.tsx`** — CP216.2 "통계 검정" 섹션 (DM/Bootstrap CI/GW regime).
- **`components/training/GoalCards.tsx`** — GoalCard 컴포넌트 + 그리드 + StoredEvaluationSection + StaticGoalCard 어댑터.

## cardTypes를 staticEvaluation.StaticGoalCard와 합치지 않은 이유

`StaticGoalCard`는 CP216 정적 평가 카드 (운영 모델 검증치 박제) 데이터형. `GoalCardData`는 실험 빌더 출력 + GoalCard 컴포넌트 prop. 의미가 다르고 lifecycle도 다름 — 정적 박제 vs 동적 빌더 결과. `staticCardsToGoalCardData` 어댑터로 일방향 변환. 같은 모듈에 두면 의미 결합이 생겨 후속 변경 (정적 → 외부 JSON) 시 builder 영향. 별도 `cardTypes.ts`로 분리.

## 추출 순서 원칙 — React 비의존 lib 우선

순수 함수(빌더 + 헬퍼) → 정적 섹션 컴포넌트 → 표 컴포넌트 → 시각 핵심 컴포넌트 → 카드 묶음. lib 먼저 안정화한 뒤 컴포넌트를 가르는 게 위험 낮음 — 컴포넌트는 JSX className/role/footnote의 1글자 변경도 시각 회귀라 가장 까다롭다.

## CP230 안전망 (Playwright + Vitest)으로 회귀 가드

지시서 §환경은 "Vitest/스냅샷 부재"로 적었으나 CP230이 Playwright 1.60 + Vitest 4.1.8 + 4 screenshot baseline을 깔아둔 뒤라 매 Step `npm run test:e2e` + `npm run test:unit`로 자동 회귀 가드. 시각 회귀 0 ("통계 검정" 영역 SignificanceSection 추출 단계 포함) + Vitest 152 passed 유지. tsc + 수동 시각 비교만 사용한다는 지시서 §검증 기준보다 강한 안전망.

## getExperimentFailureReason dead code 처리

호출자 0건 (Grep 확정) — CP231 차단 트리거 #7/CP232 §범위 ("dead code 삭제 금지, 그대로 이전만")에 따라 `experimentBuilder.ts`로 옮긴 채 caller 0인 상태로 남김. 후속 cleanup CP에서 처리.

## 결과

TrainingView 1461 → 940줄 (-36%). 8 신규 모듈, 동작 무변경 (screenshot diff 0). Vitest 152 passed 매 Step.
