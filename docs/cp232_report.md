# CP232 보고서 — TrainingView 분리

**완료일**: 2026-06-04
**선행 의존**: CP230 (Playwright + Vitest baseline, 그린), CP231 (BacktestView 분리 완료)
**커밋 범위**: `cb55b44` (CP231 끝) → `b780e26` → `7dd6ca5` → `d2d2311` → `6b0179b` → `f5f11cb` → `696179f` (6 commit)

## 요구

`frontend/src/components/TrainingView.tsx` (시작 1461줄, 지시서 진단 1560에서 CP224b dead code 제거 후 -99)을 lib 4 + components 4 = 8 모듈로 분할. 동작 무변경, 매 Step tsc + Vitest + Playwright screenshot diff로 회귀 가드.

## 한 일 (Step별)

| Step | 내용 | 커밋 | 신규 모듈 |
|---|---|---|---|
| 1 | lib 3개 신규 (공존, caller 미이전) | `b780e26` | cardTypes.ts, metricAccess.ts, experimentBuilder.ts |
| 2 | TrainingView 옛 정의 11개 삭제 (interface 2 + 헬퍼 6 + 빌더 5 + 일부 인근) + import 교체 | `7dd6ca5` | — |
| 3 | ProductMetricsSection (PptMapping + V1Extra) | `d2d2311` | `components/training/ProductMetricsSection.tsx` |
| 4 | ComparisonTable + ComparisonRow 타입 분리 | `6b0179b` | `comparisonTypes.ts`, `components/training/ComparisonTable.tsx` |
| 5 | SignificanceSection (시각 핵심, 글자 단위 복제) | `f5f11cb` | `components/training/SignificanceSection.tsx` |
| 6+7 | unused import 정리 + GoalCards 추출 (목표 ≤950 달성 위해) | `696179f` | `components/training/GoalCards.tsx` |

## 결정

- **Step 6+7 합본**: Step 6 (지시서 §Step 6 import 정리) 결과 1008줄 (목표 ≤950 살짝 초과). 1 더 추가 추출 (GoalCard/Grid/StoredEvaluation 묶음) 해서 940으로 떨어뜨림 — 단일 commit으로 묶음.
- **GoalCard 컴포넌트 export 필요**: GoalCards 모듈 내부 사용 (Grid 안)에 더해 TrainingView 본문 PendingSlotDetail JSX 한 곳에서 직접 호출 (line 437) → tsc 에러 발견 후 export 추가.
- **`metricAccess.ts` 별도 모듈**: getMetricText / getMetricNumber가 TrainingView 본문 (ExperimentDetail의 buildComparisonRows, ModelRunDetails) + experimentBuilder 양쪽에서 사용 → 헬퍼 모듈 분리. formatters/detailFields에 합치지 않음 (의미가 metric access 한 가지로 모임).
- **`comparisonTypes.ts` 별도 모듈 (cardTypes와 분리)**: ComparisonRow와 GoalCardData는 의미가 다름 (행 vs 카드). 같은 모듈에 두면 후속 변경 결합.
- **ProductMetricsSection은 named export 2개** (묶음 X): 지시서 권장. 호출부 LineModelDetail/BandModelDetail JSX 순서 (`<V1ExtraIndicatorsSection/>` → `<PptMappingSection/>`) 무변경.
- **getExperimentFailureReason dead code**: caller 0건 (Grep 확정). 지시서 §범위 (dead code 삭제 금지)에 따라 experimentBuilder로 옮긴 채 caller 0 그대로. 별도 cleanup CP 후속.
- **지시서 §검증 (tsc + 수동 시각 비교)보다 CP230 안전망 적용**: 지시서가 작성된 시점은 CP230 도입 전이라 "Vitest/스냅샷 부재"로 적었으나 실제로는 Playwright 1.60 + Vitest 4.1.8 + 4 screenshot baseline 존재. 매 Step `test:e2e` + `test:unit` 자동 회귀 가드.

## 핵심 컴포넌트 존재 체크리스트

| 항목 | 확인 |
|---|---|
| `lib/training/cardTypes.ts` 생성 | OK |
| `lib/training/comparisonTypes.ts` 생성 | OK |
| `lib/training/metricAccess.ts` 생성 | OK |
| `lib/training/experimentBuilder.ts` 생성 | OK |
| `components/training/ProductMetricsSection.tsx` 생성 | OK (named exports) |
| `components/training/ComparisonTable.tsx` 생성 | OK |
| `components/training/SignificanceSection.tsx` 생성 | OK |
| `components/training/GoalCards.tsx` 생성 | OK |
| 함수/컴포넌트 signature 보존 | OK (모든 caller 그대로) |
| className 문자열 1글자 변경 0 | OK (글자 단위 복제) |
| LineModelDetail/BandModelDetail/ExperimentDetail/메인 TrainingView 본체 로직 변경 0 | OK |
| API 응답 schema 변경 0 | OK (백엔드 무관) |
| getExperimentFailureReason dead code 보존 (`experimentBuilder.ts`에 caller 0으로 이전) | OK |

## 새 테스트 결과

CP232 자체는 신규 단위 테스트 추가하지 않았다. CP230 / CP231에서 박제된 152개 Vitest 테스트가 회귀 가드로 작동 — Step 1~7 매번 동일 152 passed | 4 todo 유지. 신규 stub 활성화는 CP233 또는 별도 cleanup CP에서.

## dry-run 결과 (Playwright screenshot diff)

매 Step 후 4 screenshot (report / stocks / backtests / training, chromium-win32) 비교. **모든 Step에서 diff 0** (`maxDiffPixelRatio` 0.01 임계 내). 시각 회귀 없음.

| Step | tsc | Vitest | Playwright (training 포함) |
|---|---|---|---|
| 1 | 0 | 152 passed \| 4 todo | (코드 변경 없음, skip) |
| 2 | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |
| 3 | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |
| 4 | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |
| 5 (시각 핵심) | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |
| 6+7 | 0 | 152 passed \| 4 todo | 4 passed, diff 0 |

## 기존 회귀 통과 건수

- `npx tsc --noEmit`: baseline 0 → 매 Step 0 → 최종 0. (Step 7 중간 GoalCard 미정의 1 error 발생 → 즉시 export 추가로 해결.)
- Vitest: baseline 152 passed | 4 todo → 매 Step 동일 → 최종 동일. 회귀 0.
- Playwright: baseline 4 passed → 매 Step 4 passed, diff 0.

## 줄수 측정 (git show 기준)

| 시점 (commit) | TrainingView.tsx 줄수 |
|---|---|
| 시작 (cb55b44) | 1461 (working tree) / 1465 (git show) |
| Step 1 (b780e26) | 1465 (TrainingView 무수정 — 신규 lib 추가만) |
| Step 2 (7dd6ca5) | 1258 |
| Step 3 (d2d2311) | 1204 |
| Step 4 (6b0179b) | 1164 |
| Step 5 (f5f11cb) | 1017 |
| Step 6+7 (696179f) | **944** |

목표 ≤950 충족. 시작 대비 -521줄 (-36%).

## 자가 점검 결과

- **[Plan v3 정합]** PASS — 사유: 구조 분리 only. 밴드 본체·fidelity·cost·검정 결과 데이터 (정적 박제) 무변경. SignificanceSection은 글자 단위 복제로 CP216.2 산출물 화면 동일 보존.
- **[구조 결함]** PASS — 사유: 순환 import 없음 (experimentBuilder → metricAccess, GoalCards → cardTypes 단방향). ProductMetricsSection은 호출 순서 유지 위해 named export 2개. getExperimentFailureReason dead code는 §범위에 따라 이전만, 별도 cleanup 보고. ComparisonRow 타입은 cardTypes 대신 comparisonTypes로 분리 (의미 결합 방지).
- **[모델 영향]** PASS (N/A 확정) — 사유: 학습·calibration·평가 산출 코드 무변경. backend/ai/ 무수정. parquet/DB write 0. AI 모델 detail 표시 화면만 분리.

## 후속 (별도 CP)

- **`getExperimentFailureReason` dead code 삭제**: 호출자 0건 Grep 확정. CP232 §범위에 따라 보존, 별도 청소 CP 후속.
- **ReproducibilitySection / UsageDataSection orphan 컴포넌트**: TrainingView에서 import/렌더 0건이고 CP224b에서 일부 정리됨. 남은 lib (reproducibility.ts / usageData.ts) 및 잔여 컴포넌트 상태 점검 별도 CP 권장.
- **시각 회귀 자동화 격상**: Playwright 4 screenshot은 viewport 1280x800 단일. SignificanceSection details 펼침 / GW interpretation / 실험 상세 ComparisonTable 등 진입 시 화면은 baseline 무존재 → 별도 CP에서 추가 screenshot baseline 권장.

## ADR

`docs/adr/0022-trainingview-split.md` 작성. 8모듈 분할 결정, cardTypes를 StaticGoalCard와 분리한 이유, 추출 순서 원칙 (React 비의존 lib 우선), CP230 안전망 활용 (지시서 §환경 정정), getExperimentFailureReason 처리 기록.
