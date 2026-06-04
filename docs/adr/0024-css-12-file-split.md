# ADR-0024: globals.css 5208줄을 12파일로 분리 (완료)

Status: Accepted
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

## responsive 마무리 — rename으로 해결

전역 `@media (max-width:900px)` 블록 (114줄)은 globals.css가 단일 영역만 남았을 때 처음에는 narrow viewport baseline 부재로 "추출" 안전성을 못 보장하는 듯 보였다. 그러나 결국 cascade가 깨질 가능성은 **이동(move)이 동반된 변경**일 때만 생긴다. 사용자 통찰: globals.css에 남은 114줄을 **다른 파일로 옮기는 게 아니라 파일명만 rename** (`globals.css` → `responsive.css`)하고 layout.tsx의 import 경로만 바꾸면 — 내용 1 bit 변경 0 + 같은 디렉토리 + import 순서상 동일 위치 — **cascade 회귀는 원천 불가능**. 따라서 narrow viewport baseline은 필요 없다. 1280x800 Playwright diff 0 + 구조적 동일성으로 충분.

## screenshot-gated Strangler

10 Step (Step 1~10) + Step 11+12 (rename) 모두 매 단계에서 `npx tsc --noEmit` 0 + Playwright 4 screenshot (chromium, 1280x800, locale ko-KR) diff 0 그린. globals.css 5208 → 0. 최종 13개 파일 (tokens.css + styles/12 + responsive.css).
