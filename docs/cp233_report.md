# CP233 보고서 — StockView 부분 분리

**완료일**: 2026-06-04
**선행 의존**: CP230 (Playwright + Vitest baseline, 그린), CP231/CP232 완료
**커밋 범위**: `57f4959` (CP232 끝) → `aacefa6` → `895ed1c` → `bb36bd5` (3 commit)

## 요구

`frontend/src/components/StockView.tsx` (시작 1041줄, 지시서 진단 1097에서 CP224b dead code 제거 후 -56)을 lib + components 모듈로 분할. 메인 로직(loadStock/12조건 분기/신선도 라벨)은 §제외. 매 Step tsc + Vitest + Playwright screenshot diff 가드.

## 한 일 (Step별)

| Step | 내용 | 커밋 | 신규 모듈 |
|---|---|---|---|
| 1 | helpers.ts + 단위 테스트 (공존) | `aacefa6` | `lib/stock/helpers.ts`, `lib/stock/helpers.test.ts` |
| 2+3 | StockView 옛 정의 삭제 + import 정리 + dead code 제거 | `895ed1c` | — |
| 4 | IndicatorOptions.tsx 추출 | `bb36bd5` | `components/stock/IndicatorOptions.tsx` |

## 결정

- **Step 1-0 (Vitest 설치) skip**: 지시서가 작성된 시점은 CP230 도입 전이라 "vitest 미설치"로 명시했으나 실제로는 CP230에서 이미 설치됨 (Vitest 4.1.8). vitest.config.ts도 존재. 바로 단위 테스트 작성.
- **Step 2+3 합본**: 옛 정의 삭제 + dead code 제거 + import 정리를 단일 commit으로 처리. 차단 트리거 #9 ("구조 변경 + 동작 변경 혼합 금지") 위반 아님 — getLastPoint 삭제는 caller 0건 (Grep 확정)이라 동작 변경 0.
- **AiState 인터페이스 helpers로 이동**: buildAiState 반환형이라 helpers.ts에 정의 + export. StockView가 import해 재사용 (한 군데 정의로 통일).
- **`IndicatorOption` 컴포넌트는 module-local**: IndicatorOptions.tsx 안에서 `IndicatorOptionGroup`만 named export, `IndicatorOption`은 비-export (원본 비-export 패턴 보존).
- **`getExperimentFailureReason`류 dead code 처리 패턴 재적용**: getLastPoint는 caller 0건 Grep 확정 + 동명이지만 별개인 함수(predictionOverlay.ts 내 로컬)가 별도 존재 → CP233 §범위 "dead code 삭제" 항목에 포함되어 실제 삭제.
- **메인 로직 분리 보류 (ADR-0023 기록)**: loadStock + 12조건 분기 + 신선도 라벨은 React state 12개에 강하게 결합. 분리하려면 custom hook 설계가 필요한데 안전망/리뷰 필요. CP233은 저위험 추출만으로 한정 → 924줄, ≤905 19줄 초과 인정.

## 핵심 컴포넌트 존재 체크리스트

| 항목 | 확인 |
|---|---|
| `lib/stock/helpers.ts` 생성 | OK |
| `lib/stock/helpers.test.ts` 생성 (14개 신규 박제 테스트) | OK |
| `components/stock/IndicatorOptions.tsx` 생성 | OK |
| 함수 signature 보존 (`fetchPriceHistory(ticker,timeframe,fullHistory=false)` 등) | OK (바이트 동일) |
| 컴포넌트 props 인터페이스 보존 (`IndicatorOptionGroupProps`, `IndicatorOptionProps`) | OK |
| className 문자열 변경 0 | OK |
| `getLastPoint` dead code 제거 (caller 0건 Grep 확정) | OK |
| `IndicatorChartPoint` / `IndicatorChartSeries` named import 제거 | OK |
| `PRODUCT_HISTORY_LOOKBACK_DAYS` / `LINE_OVERLAY_HELD` 본문 잔존 (loadStock 의존, §범위) | OK |
| 메인 로직 (loadStock / 12조건 / 신선도 라벨) 무변경 | OK |
| API 응답 schema / 백엔드 호출 0 변경 | OK |

## 새 테스트 결과

`helpers.test.ts` — 14 passed (getLastPrice 2 / getChangePercent 3 / getLastFinite 3 / getPriceLookbackDays 2 / buildAiState 2 / fetchPriceHistory 2). Vitest 전체 152 → 166 passed | 4 todo.

`fetchPriceHistory`는 `vi.mock("@/api/client")`로 `fetchPrices`를 mock해 호출 인자(ticker, timeframe) + 정렬(sortPriceRows) 통과만 박제. 네트워크 실호출 0.

## dry-run 결과 (Playwright screenshot diff)

매 Step 후 4 screenshot 비교. **모든 Step에서 diff 0**.

| Step | tsc | Vitest | Playwright |
|---|---|---|---|
| 1 | 0 | 166 passed \| 4 todo (+14) | (코드 변경 0, skip) |
| 2+3 | 0 | 166 passed \| 4 todo | 4 passed, diff 0 (stocks 화면 가격 차트 + 오버레이 + 토글 모두 baseline) |
| 4 | 0 | 166 passed \| 4 todo | 4 passed, diff 0 (지표 선택 위젯 unchanged) |

## 기존 회귀 통과 건수

- `npx tsc --noEmit`: baseline 0 → 매 Step 0 → 최종 0.
- Vitest: baseline 152 passed | 4 todo → 최종 166 passed | 4 todo. 회귀 0.
- Playwright: baseline 4 passed → 최종 4 passed, diff 0.

## 줄수 측정 (git show 기준)

| 시점 (commit) | StockView.tsx 줄수 |
|---|---|
| 시작 (57f4959 = CP232 끝) | 1042 |
| Step 1 (aacefa6) | 1042 (StockView 무수정 — helpers.ts 신규만) |
| Step 2+3 (895ed1c) | 966 |
| Step 4 (bb36bd5) | **925** |

목표 ≤905 미달 (19줄 초과). 시작 대비 -117줄 (-11%). ADR-0023에 미달 사유 기록.

## 자가 점검 결과

- **[Plan v3 정합]** PASS — 사유: FE 구조 분리 only. 가격 차트·AI 밴드 오버레이·보수적 기준선·신선도 라벨 모두 화면 차원 무변경.
- **[구조 결함]** PASS — 사유: 순환 import 없음. helpers signature 바이트 동일 (테스트로 박제). getLastPoint dead code 제거 (Grep으로 caller 0 사전 확인). IndicatorOption은 module-local 유지 (원본 비-export 패턴 보존).
- **[모델 영향]** PASS (N/A 확정) — 사유: 예측 오버레이 동작 불변 (`checkBandOverlay` 등 predictionOverlay 호출 그대로). 학습·calibration 코드 무관.

## 후속 (별도 CP)

- **메인 로직 분리 (ADR-0023 권고)**: `loadStock` 또는 신선도 파생 라벨을 custom hook (e.g. `useStockLoader`)으로 빼면 ≤905 도달 + 재사용. state 12개 + ref/effect 의존을 hook 인터페이스로 노출하는 설계 + screenshot baseline 강화 필요.
- **시각 회귀 자동화 확대**: 현재 Playwright 4 screenshot은 viewport 1280x800 단일 + 기본 상태(AAPL/1D). 1W 전환 / 토글 ON 상태 / 1M 모드 등 추가 baseline 권장.

## ADR

`docs/adr/0023-stockview-partial-split.md` 작성. 메인 로직 분리 보류 사유 + ≤905 미달 트레이드오프 + 후속 권고.
