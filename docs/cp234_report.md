# CP234 보고서 — globals.css 분리 (완료)

**완료일**: 2026-06-04
**선행 의존**: CP230 (Playwright 1280x800 baseline, 그린)
**커밋 범위**: `cc44e26` (CP233 끝) → `2add1b7` → `e77c698` → `47b5dbf` → `d301b07` → `2b591e5` → `2153ce6` → `dba91ef` → `40c2661` → `75a7285` → `7e6d91d` → `5fef4cd` (report draft) → `f3027a6` (rename 마무리) (12 commit)
**마지막 그린**: `f3027a6`

## 요구

`frontend/src/app/globals.css` (시작 5208줄)을 영역별 12개 파일로 분리. tokens.css 불변. layout.tsx import 순서로 cascade 결정. 매 Step Playwright 4 screenshot diff 0.

## 한 일 (Step별)

| Step | 영역 | 커밋 | 영역 줄수 | globals.css 잔여 |
|---|---|---|---|---|
| 0 | baseline 그린 확인 + styles 디렉토리 | — | — | 5208 |
| 1 | base.css | `2add1b7` | 40 | 5167 |
| 2 | shell.css | `e77c698` | 421 | 4746 |
| 3 | stock.css (두 토막) | `47b5dbf` | 1173 | 3573 |
| 4 | backtest.css | `d301b07` | 639 | 2934 |
| 5 | training.css (사용자 순서: backtest→training→components) | `2b591e5` | 661 | 2273 |
| 6 | components.css (두 토막) | `2153ce6` | 410 | 1863 |
| 7 | report 3-way (model-role-card / archive / view) | `dba91ef` | 19 + 445 + 544 = 1008 | 856 |
| 8 | significance.css (본문 + scoped @media) | `40c2661` | 141 | 715 |
| 9 | reproducibility.css | `75a7285` | 398 | 317 |
| 10 | experiment.css | `7e6d91d` | 203 | 114 |
| 11+12 | globals.css → responsive.css **rename** + layout import 경로 변경 | `f3027a6` | 114 (그대로) | **0 (삭제)** |

## 결정 / 사용자 합의

- **사용자 import 순서 적용**: backtest → **training → components** → report → significance → reproducibility → experiment → responsive. 지시서 §Sub-step 순서와 components/training swap + report 위치 다름. 추출 순서도 동일하게 맞춤 (Step 추가 순서 = 최종 cascade 순서).
- **report 3-way 분할 합의**: report 영역 1009줄 > 800 초과 → 사용자에게 보고 → 셋 분할 합의 (model-role-card 19 / archive 445 / view 544). model-role-card는 자연 크기 작지만 사용자 합의대로 별도 파일.
- **신호카드 backtest로 통합**: 원본에서 "Stocks topbar" 다음 영역(.signal-card 등)이 의미상 backtest. backtest.css에 함께 옮겨 cascade 보존.
- **scoped @media 동봉**: significance / experiment-archive 전용 scoped @media는 자기 영역 파일에 본문 → media 순으로.
- **responsive는 추출이 아니라 rename으로 해결**: globals.css에 단일 영역(responsive)만 남았을 때 처음에는 narrow viewport baseline 부재로 차단했으나, **이동(move)이 아니라 파일명만 rename** (`globals.css` → `responsive.css`, 같은 디렉토리, layout import 경로만 변경)으로 처리하면 내용 1bit 변경 0 + 같은 위치 + 같은 import 순서라 cascade 회귀 원천 불가능. narrow viewport baseline 불필요. 1280x800 diff 0 + 구조적 동일성으로 충분.

## 핵심 보존 체크리스트

| 항목 | 확인 |
|---|---|
| `.signal-card` base + `:hover` + `.is-selected` 결합 동일 파일·순서 | OK (backtest.css) |
| `.significance__finding--{good,warn}` 변형 base 뒤 | OK (significance.css) |
| `.reproducibility[open]` state base 뒤 | OK (reproducibility.css) |
| scoped @media (max-width:920px) 자기 영역 동봉 | OK (significance.css + report-archive.css) |
| CSS 값 / selector / className 1글자 변경 0 | OK (이동만) |
| tokens.css 불변 | OK |
| @import 폰트 2 + tokens base.css 맨 위 유지 | OK |
| layout.tsx 다중 import 순서 = 사용자 명시 | OK |

## 사고 보고 (Step 도중 차단 트리거 발동 → 해결)

1. **Step 2 — 한글 `─` 문자 손상**: PowerShell `Get-Content` 기본 인코딩이 시스템 ANSI (CP949)라 U+2500 box-drawing 문자가 `?`로 손상. 차단 트리거 발동 → `git checkout` revert + 새 shell.css 파일 삭제 → `[System.IO.File]::ReadAllLines($path, [System.Text.UTF8Encoding]::new($false))` UTF8 명시 + `WriteAllLines` UTF8 노BOM으로 재시도 → 보존 OK. 이후 Step들 모두 UTF8 명시 사용.
2. **Step 7 — Responsive 헤더 view.css 잘못 포함**: report 3-way cut에서 view 영역 끝 인덱스가 Responsive 헤더 라인을 포함. 시각 회귀 0 (코멘트라 cascade 영향 없음)이지만 정확성 위해 fix: view.css 마지막 줄 제거 + globals.css에 헤더 + 빈줄 prepend.

## dry-run 결과 (Playwright screenshot diff)

매 Step 후 4 screenshot (report / stocks / backtests / training, chromium-win32, viewport 1280x800) 비교. **Step 1~10 모두 diff 0**. tsc 0 매 Step.

| Step | tsc | Playwright (4 screen, 1280x800) |
|---|---|---|
| 1 base | 0 | 4 passed, diff 0 |
| 2 shell | 0 | 4 passed, diff 0 |
| 3 stock | 0 | 4 passed, diff 0 |
| 4 backtest | 0 | 4 passed, diff 0 |
| 5 training | 0 | 4 passed, diff 0 |
| 6 components | 0 | 4 passed, diff 0 |
| 7 report (3-way + Responsive 헤더 fix) | 0 | 4 passed, diff 0 |
| 8 significance | 0 | 4 passed, diff 0 |
| 9 reproducibility | 0 | 4 passed, diff 0 |
| 10 experiment | 0 | 4 passed, diff 0 |

Vitest는 CSS 변경 영향 없음 (166 passed | 4 todo 그대로).

## 줄수 측정 (최종)

| 파일 | 줄수 | 비고 |
|---|---|---|
| tokens.css | 65 | 불변 |
| base.css | 40 | 헤더 + @import + reset |
| shell.css | 421 | App shell + Sidebar + Topbar |
| stock.css | 1173 | 두 토막 (Workspace + Stocks layout) — 자연 크기 |
| backtest.css | 639 | Search + 신호카드 |
| training.css | 661 | Training view + CP216 |
| components.css | 410 | 두 토막 (Notices + Tables) |
| report-model-role-card.css | 19 | 사용자 셋 분할 |
| report-archive.css | 445 | experiment-archive + scoped @media |
| report-view.css | 544 | report-view / report-hero 등 |
| significance.css | 141 | 본문 + scoped @media |
| reproducibility.css | 398 | 재현 매니페스트 + GW 해석 |
| experiment.css | 203 | CP218 실험 타임라인 |
| **responsive.css** | **114** | Responsive 헤더 + @media (max-width:900px) 블록. globals.css → responsive.css rename. |
| **globals.css** | **0** | 삭제 완료 (rename 결과). |

신규 모듈 12개 + rename 1개. globals.css 5208 → 0.

## 자가 점검 결과

- **[Plan v3 정합]** PASS — 사유: CSS 분리만. 밴드 본체·fidelity·cost·모델·calibration 로직 무관.
- **[구조 결함]** PASS — 사유: Step 11+12 rename으로 globals.css 0줄 = 단일 파일 제거 목표 달성. responsive는 별도 파일로 깔끔. cascade-safe (layout.tsx 마지막 import 위치 = 전역 @media가 모든 base 뒤).
- **[모델 영향]** PASS (N/A 확정) — 사유: 프론트 CSS만 변경. backend·ai·parquet 무관.

## 후속 (별도 CP)

1. **narrow viewport baseline 추가** (선택): Playwright `screens.smoke.spec.ts` 확장 (375x800 mobile + 768x800 tablet). 현재 cascade 회귀는 1280x800 + 구조적 동일성으로 보장되지만, 향후 responsive.css 안 selector 추가/변경 시 narrow 회귀 가드 필요.
2. 미사용 CSS 규칙 정리 (CP234 §제외였던 항목, 별도 CP).
3. CP216.2 통계 검정 narrow 화면 details 펼침 등 상태 컷 추가 (현재 baseline에 상태 변형 안 드러남).

## ADR

`docs/adr/0024-css-12-file-split.md` 작성. 분리 경계 / import 순서 / cascade 보존 / 사용자 합의 셋 분할 / responsive 미완 사유.
