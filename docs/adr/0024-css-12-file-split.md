# ADR-0024: globals.css 5208줄을 12파일로 분리 (responsive 미완)

Status: Partially accepted (10 of 12 regions extracted; responsive blocked)
Date: 2026-06-04
CP: CP234

## 결정

`frontend/src/app/globals.css` (시작 5208줄)을 영역 코멘트 헤더 단위로 자른 뒤 `frontend/src/app/styles/`에 배치하고 `layout.tsx`가 cascade 순서대로 import한다. tokens.css는 불변. 사용자가 명시한 import 순서가 cascade를 결정한다:

```
base → shell → stock → backtest → training → components
   → report-model-role-card → report-archive → report-view
   → significance → reproducibility → experiment → (responsive)
```

## 분리 경계 (선택 이유)

- **신호카드는 backtest로**: `.signal-card` / `:hover` / `.is-selected` 결합 체인이 한 파일 + 원본 상대 순서를 유지해야 hover/선택 하이라이트가 보존된다. 신호카드는 "Stocks topbar" 코멘트 헤더 다음에 있었지만 의미적으로 backtest 영역의 일부 → backtest.css에 함께 옮겼다.
- **scoped `@media (max-width:920px)`는 동봉**: significance / experiment-archive 전용 scoped media는 자기 영역과 같은 파일에 본문 → media 순으로 둠. base 정의 → media override 소스 순서 보존.
- **report 3-way 분할 (사용자 합의)**: report 영역이 1009줄로 800 초과. 사용자 합의로 셋 분할:
  - `report-model-role-card.css` (19줄) — CP220 `.report-model-role-card` 본문.
  - `report-archive.css` (445줄) — CP220 `.experiment-archive` + scoped `@media (max-width:920px)`.
  - `report-view.css` (544줄) — `.report-view` / `.report-hero` / `.report-status-strip` / `.comparison-table` 등.
  - model-role-card 19줄은 자연 크기로 매우 작으나 사용자 합의로 별도 파일 유지.

## import 순서 (cascade 결정)

사용자 명시 순서 그대로. 지시서 §Sub-step 순서와는 components/training이 swap, report가 components 직후 (지시서는 report가 reproducibility 다음). 매 Step 추출 직후 Playwright 4 screenshot diff 0 확인으로 cascade 안전성 보장.

## cascade 보존 규칙

- CSS 값/selector 1글자 변경 0 (이동만, 지시서 §금지).
- `.X` base + `.X:hover` / `.X.is-on` / `.X.is-selected` / `.X[open]` / 변형 selector 모두 동일 파일 + 동일 상대 순서 유지.
- 마침내 commit 직전 view.css에 Responsive 헤더가 한 줄 잘못 포함된 사고가 있었다 (Step 7 중간 수정). 검출은 grep으로, 수정은 view.css 마지막 줄 제거 + globals.css 헤더 prepend. cascade는 영향 0이지만 정확성 위해 fix.

## 대안 — `@import` 방식 채택 안 함

`globals.css`에 `@import "./styles/base.css"; ...` 12줄 두는 안. CSS `@import`는 런타임 순차 페치 (Next 빌드에서 인라인되긴 하지만 권장 안 함). layout.tsx 다중 import가 표준.

## 미완 사유 — responsive

전역 `@media (max-width:900px)` 블록 (114줄, globals.css에 잔존)이 stock/training/report/backtest/experiment/components 클래스를 한꺼번에 override. 회귀를 잡으려면 narrow viewport screenshot baseline 필수. **CP230 baseline은 viewport 1280x800 단일이라 ≤900px 컷 부재**. 사용자 명시 차단 트리거 ("narrow viewport baseline 없으면 차단 보고") 발동 → Step 11 + Step 12 미수행. 후속 CP에서 narrow viewport baseline (375x800, 768x800) 추가 후 responsive.css 추출 + globals.css 제거 + layout.tsx 최종 정리.

## screenshot-gated Strangler

10 Step 모두 매 단계에서 `npx tsc --noEmit` 0 + Playwright 4 screenshot (chromium, 1280x800, locale ko-KR) diff 0 그린 → 1280x800 viewport 회귀 0. narrow viewport 회귀는 측정 불가 (후속 CP).
