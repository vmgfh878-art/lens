# CP234 FE/CSS 분리(고위험) (Directive)

> 런북(`docs/cp221_237_refactoring_runbook.md`)이 이 문서를 자동으로 꺼내 실행한다. 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다.
>
> **이 CP는 본 리팩토링 묶음에서 가장 위험하다.** CSS는 파일 분리만으로 specificity/cascade(소스 순서) 가 깨질 수 있고, 깨져도 빌드는 통과하고 화면만 조용히 틀어진다. 그래서 "한 영역씩 + 매 영역 screenshot diff 0"이 절대 규칙이다. 의심되면 무조건 멈춘다.

---

## 역할 고정

- **모드**: `code` (구현 + 같은 턴 자가 점검).
- **권한**: 코드 수정, 로컬 검증(tsc / next build / dev 서버 기동 / 스크린샷 비교)만.
- **금지**:
  - 새 학습(training) 실행 금지.
  - 새 calibration 실행 금지.
  - DB write 금지.
  - Supabase 호출 금지.
  - 사용자가 직접 수정한 파일 revert 금지.
  - **CSS 값 자체를 "정리/개선"하지 마라.** 이 CP는 *이동(move)*만 한다. 색·여백·selector·속성 한 글자도 바꾸지 않는다. 줄을 다른 파일로 옮길 뿐이다. 리팩토링 커밋과 기능/스타일 변경 커밋을 섞지 않는다.
- **자가 점검**: 완료 후 [Plan v3 정합] [구조 결함] [모델 영향] 보고(양식 하단).
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128) — 이 CP에선 Python 불필요(프론트 전용)이지만, 백엔드가 떠 있어야 화면이 정상 렌더되는 화면이 있으면 기동 필요.
- **백엔드 기동**: `scripts\start_demo.ps1` 또는 `python -m uvicorn ...` (포트 기본 8000 가정).
- **프론트**: `frontend` 디렉토리에서 `npm run dev` (Next.js, 기본 포트 3000).
- **포트 충돌 피하기**: 이미 떠 있는 dev 서버가 있으면 그걸 쓰고 새로 띄우지 마라. 새로 띄워야 하면 `--port`로 다른 포트 지정. 검증 끝나면 본인이 띄운 서버만 정리.
- **대상 파일**: `frontend/src/app/globals.css` (현재 **5208줄**), `frontend/src/app/layout.tsx` (현재 22줄), `frontend/src/app/tokens.css` (현재 66줄, **유지·불변**).

---

## 진단 (근거)

조사 출처: `frontend/src/app/globals.css` 전체 Grep(`^/\* ─`, `^/\* CP`, `@media`, `^\.signal-`, `^\.report-` 등) + 경계부 직접 Read. `frontend/src/app/layout.tsx` 직접 Read. `frontend/src/components/` 목록 + `frontend/package.json` scripts/deps 확인.

### 1) 단일 거대 파일

`frontend/src/app/globals.css`는 **5208줄** 단일 파일이다(`wc -l` 확인). 디자인 토큰만 `tokens.css`(66줄)로 빠져 있고 나머지 전부가 한 파일에 있다. layout.tsx는 이 한 줄로만 들고 온다:

```tsx
// frontend/src/app/layout.tsx:4
import "./globals.css";
```

`globals.css` 선두(1~4줄):

```css
/* line 1 */ /* Lens Dashboard — light, airy, indigo-accented system */
/* line 2 */ @import url("https://fonts.googleapis.com/css2?family=Inter:...&display=swap");
/* line 3 */ @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/...pretendard.css");
/* line 4 */ @import "./tokens.css";
```

→ **함의 1: `@import` 규칙은 CSS 파일 맨 위(다른 모든 규칙보다 먼저)에만 올 수 있다.** 폰트 2줄(2~3)과 토큰 import(4)는 base 파일 맨 위로 가거나, 분리하면 layout.tsx가 그것들을 *가장 먼저* import해야 한다. 순서가 틀리면 폰트가 안 먹거나 `@import`가 무시된다.

### 2) 논리 영역이 코멘트로 구획돼 있으나 서로 끼어 있음(핵심 위험)

`^/\* ─` 및 주요 코멘트 앵커(줄 번호 = 현재 globals.css):

| 줄 | 코멘트 앵커 | 논리 영역 |
|---|---|---|
| 1 | Lens Dashboard 헤더 + 폰트/토큰 @import + reset/body/button | **base** |
| 42 | App shell | shell |
| 54 | Sidebar | shell |
| 123 | Sidebar search | shell |
| 256 | legacy eyebrow | shell |
| 356 | Collapsed state | shell |
| 391 | Topbar | shell |
| 463 | Workspace | shell/stock 경계 |
| 514 | Panels & cards | (공용 패널) |
| 563 | Stocks topbar | stock |
| 812 | Search / select fields (BacktestView) | **backtest** |
| 934 | `.signal-group-panel` / `.signal-card*` (812 블록 내부) | **backtest (신호카드)** |
| 1451 | Stocks layout | stock |
| 1647 | Forecast layer groups | stock |
| 2218 | Layer toggle | stock |
| 2275 | Notices | components |
| 2334 | StatusInline (CP213) | components |
| 2428 | Metric cards | components |
| 2545 | Charts placeholders | components |
| 2562 | Status badges | components |
| 2649 | **Training view** | training |
| 3057 | CP216 사용 데이터 그룹화 | training |
| 3212 | CP216 정적 평가 카드 | training |
| 3310 | CP217.2 통계 검정 (significance) | **significance** |
| 3405 | CP220 재현 매니페스트 | **reproducibility** |
| 3608 | GW 해석 박스 | reproducibility |
| 3803 | `@media (max-width:920px)` — significance 전용 | significance(미디어) |
| 3849 | CP220 model-role-card | report |
| 3868 | CP220 experiment-archive | report |
| 4073 | `@media (max-width:920px)` — experiment-archive 전용 | report(미디어) |
| 4313 | `.report-view` / `.report-*` | **report** |
| 4857 | Tables | components |
| 4893 | **`@media (max-width:900px)` — 전역 반응형** | **responsive(교차 위험)** |
| 5006 | CP218 experiment-timeline | **experiment** |

→ **함의 2(가장 위험): 논리 영역이 파일 안에서 인터리브(끼어)돼 있다.**
- 신호카드(`.signal-card*`, 934~1107)는 *backtest* 영역이지만 stock 블록 사이에 박혀 있다.
- "Training view" 코멘트(2649) 아래에 training / 정적평가 / **significance / reproducibility / report / experiment**가 한 덩어리로 섞여 있다.
- `tables`(4857)와 전역 `@media`(4893)가 report 뒤, experiment 앞에 끼어 있다.

따라서 "comment 헤더 단위로 통째로 잘라 옮긴다"는 단순 규칙이 대체로 맞지만, **영역 경계가 코멘트 줄에 정확히 떨어지지 않는 구간이 있다.** 잘못 자르면 한 selector 그룹이 둘로 쪼개져 cascade가 깨진다.

### 3) cascade/specificity 결합 selector 실재(끊으면 깨짐)

직접 확인된, **반드시 같은 파일·같은 상대순서로 유지해야 하는** 상태/조합 selector 예:

```css
/* globals.css:992-998  — :hover 와 .is-selected 가 base .signal-card 바로 뒤에 와야 함 */
.signal-card:hover,
.signal-card.is-selected { ... }
.signal-card.is-selected { ... }
/* 1002 */ .signal-group-panel--watch .signal-card { ... }

/* significance: 변형(modifier)이 base 뒤에 와야 verdict 색이 덮임 */
.significance__finding--good .significance__finding-verdict { ... }   /* 3382 */
.significance__finding--warn .significance__finding-verdict { ... }   /* 3393 */

/* reproducibility: 속성/상태 selector */
.reproducibility[open] > summary::before { content: "▾"; }            /* 3431 */
.reproducibility[open] > summary { border-bottom: 1px solid ...; }    /* 3433 */
```

→ **함의 3: `.X`(기본) → `.X:hover`/`.X.is-on`/`.X.is-selected`/`.X[open]`/`.A .X`(변형) 가 같은 specificity면 "나중에 선언된 쪽"이 이긴다. 분리로 소스 순서가 바뀌면 색·테두리·선택 하이라이트가 조용히 틀어진다.** 이게 이 CP의 회귀가 빌드로 안 잡히는 이유다.

### 4) 전역 `@media` 블록이 교차 영역을 참조(분리 시 가장 까다로움)

`globals.css:4894-5004` 단일 `@media (max-width:900px)` 블록 하나가 **여러 영역의 클래스**를 한꺼번에 덮어쓴다(직접 Read 확인). 참조 클래스 예: `.market-layout`(stock), `.training-layout`(training), `.report-hero`/`.report-status-strip`(report), `.strategy-signal-overview`/`.backtest-toolbar`/`.rule-grid`(backtest), `.experiment-row`/`.experiment-section-grid`(experiment), `.toolbar`(components).

반면 `:920px` 두 블록(3803=significance, 4073=experiment-archive)은 각자 자기 영역 selector만 덮고 그 영역 바로 뒤에 붙어 있다 → 그 영역 파일에 같이 따라가면 된다.

→ **함의 4: scoped `@media`(3803, 4073)는 자기 영역 파일에 동봉한다. 전역 `@media`(4894)는 한 영역에 속하지 않으므로 별도 `responsive` 파일로 통째 이동하고, 그 파일을 import 순서상 거의 마지막(experiment 직전 또는 직후, 단 자신이 덮는 모든 base 규칙보다 뒤)에 둔다.** media 쿼리는 동일 specificity일 때 소스 순서 영향은 받지만, "base 정의 → 이후 media override"라는 순서만 지키면 안전하다.

### 5) 안전망 부재(선행 의존이 진짜인 이유)

`frontend/package.json` scripts는 `dev / build / start` 뿐이고, deps에 playwright/jest/vitest/storybook/percy/chromatic **전무**(확인됨). `docs/adr/` 디렉토리도 **아직 없음**(확인됨). 즉 **현재 레포에는 screenshot 비교 수단이 존재하지 않는다.** screenshot baseline은 CP230이 만든다. CP230 그린 없이 이 CP를 시작하면 "diff 0"을 측정할 방법 자체가 없으므로 **시작 불가**.

---

## 선행 의존

- **CP230 그린 (필수, hard block).** CP230이 4개 화면(Stock / Backtest / Training / Report)의 **screenshot baseline**과 그걸 다시 찍어 비교하는 절차를 확립해야 한다. 위 진단 5)대로 현재 레포엔 스크린샷 수단이 없다.
  - CP230 산출물에 baseline 이미지 저장 경로와 "재촬영 + diff" 명령이 정의돼 있어야 한다. **그 경로/명령을 그대로 이 CP의 검증에 쓴다.** 정의가 없으면 → **차단 트리거**(아래)로 즉시 보고하고 시작하지 마라.
- 그 외 선행: 없음(이 CP는 프론트 CSS 전용, 백엔드 CP223/225 계열과 독립).

---

## 범위

### 포함
- `frontend/src/app/globals.css`(5208줄)를 **12개 영역 파일 + 유지되는 tokens.css**로 분리:
  `tokens.css(유지)`, `base.css`, `shell.css`, `stock.css`, `backtest.css`, `training.css`, `components.css`, `report.css`, `significance.css`, `reproducibility.css`, `experiment.css`, `responsive.css`.
  - 목표: 각 파일 **500~800줄** 범위(아래 성공 기준 표). 일부 영역은 자연 크기가 그보다 작거나(예: significance, experiment) 클 수 있다 — **억지로 줄을 옮겨 700에 맞추지 마라.** 크기 균형보다 **cascade 보존**이 우선. 한 영역이 800을 크게 넘으면(예: report) 의미 경계(report 본문 vs model/experiment-archive)로 한 번 더 쪼개도 되지만, 그 결정은 **차단 후 사용자에게 보고하고 합의**한다(임의 분할 금지).
- `frontend/src/app/layout.tsx`의 import 라인 교체(분리 파일 순서대로).
- ADR 1장(`docs/adr/0024-css-12-file-split.md`).
- `docs/cp234_report.md`.

### 제외
- **CSS 값/selector 변경 일절 금지** (이동만).
- tokens.css 내용 변경 금지(유지).
- 새 디자인·새 클래스·미사용 규칙 삭제 금지(미사용처럼 보여도 이 CP에서 지우지 마라 — 별도 CP).
- Supabase·DB·학습·calibration 일절 무관(호출 금지).
- 사용자 직접 수정 파일 revert 금지.

---

## Sub-step (Strangler Fig, 작은 단위)

> **핵심 패턴(이 CP 버전)**: globals.css가 "옛 코드"다. 영역을 하나 *복사*해 새 파일로 만들고(옛 코드 옆 새 코드 공존), layout.tsx에 그 새 파일 import를 옛 globals.css *앞* 적절한 순서 위치에 추가하고, **globals.css에서 그 영역을 삭제**(caller 이전 = 동일 규칙을 새 파일이 제공)한다. 그 직후 **4화면 screenshot diff 0**을 확인한다. 한 Step = 한 영역 = 한 revert 단위.
>
> 매 Step 공통 마무리:
> 1. `cd frontend; npx tsc --noEmit` → 에러 0 (CSS는 tsc 영향 없지만 layout.tsx 변경 검증).
> 2. `npm run build` 성공(또는 dev 서버 무에러 컴파일).
> 3. **CP230 절차로 4화면 재촬영 + diff → 전부 0.**
> 4. diff 0이면 `git add` 후 단일 커밋(영역명 명시). diff 발생이면 → **즉시 중단·보고**(차단 트리거).

### Step 0 — 준비(분리 없음, 안전망 확인)
- CP230 그린 확인: baseline 이미지 존재 + "재촬영+diff" 명령 동작 확인. 동작 안 하면 차단·보고.
- 현재 상태 기준선: `npm run build` 성공 + 4화면 baseline 재촬영 diff 0(=현재가 baseline과 일치) 확인. 여기서부터 diff가 나면 분리 탓이 아니므로 먼저 환경을 정리.
- `frontend/src/app/styles/` 디렉토리 생성(분리 파일 보관처). 경로는 CP230/런북이 다른 위치를 지정하면 그걸 따른다.
- **커밋 없음**(준비만).

> 아래 Step 1~12는 **import 순서대로** 추출한다(순서가 cascade를 결정하므로 추출도 그 순서로 하면 사고 추적이 쉽다). 각 Step의 "원본 구간"은 현재 globals.css 줄 범위 + 코멘트 앵커로 식별하되, **줄 번호는 앞 Step에서 삭제가 일어나면 밀린다 → 매번 코멘트 앵커/대표 selector로 재확인**하고 잘라라(고정 줄 번호 맹신 금지).

### Step 1 — base 추출
- 원본 구간: globals.css **선두(1줄)~`/* ─── App shell ─── */`(현재 42) 직전**. 즉 헤더 코멘트 + `@import` 3줄(폰트2 + tokens) + `*{box-sizing}` + `html,body` + `body` + `button/input/select` (현재 1~41).
- 새 파일 `styles/base.css`로 이동. **`@import` 3줄(현재 2~4)을 이 파일 맨 위에 그대로** 둔다(폰트 → tokens 순서 유지). tokens.css는 그대로 두고 base가 `@import "../tokens.css";`로 들고 오게 한다(경로는 새 위치 기준으로 조정).
- layout.tsx: `import "./globals.css";` 를 `import "./styles/base.css";` + (이후 Step에서 한 줄씩 추가) 형태로 전환 시작. **이 Step에선 base만 추가하고, 나머지는 아직 globals.css가 제공**하므로 globals.css import 라인도 base 다음에 유지한다(공존).
  ```tsx
  import "./styles/base.css";   // 새
  import "./globals.css";        // 옛(나머지 영역, 점차 비워짐)
  ```
- globals.css에서 1~41 삭제(=base 이전 완료).
- 마무리(공통 1~4). 커밋: `refactor(css): extract base.css (CP234 step1)`.

### Step 2 — shell 추출
- 원본 구간: `/* ─── App shell ─── */`(현재 42) ~ `/* ─── Workspace ─── */`(현재 463) **직전**. 포함: App shell, Sidebar, Sidebar search, legacy eyebrow, Collapsed state, Topbar.
- `styles/shell.css`로 이동, layout.tsx에 `import "./styles/shell.css";`를 base 다음·globals 앞에 추가. globals.css에서 해당 구간 삭제.
- 마무리(공통). 커밋: `refactor(css): extract shell.css (CP234 step2)`.

### Step 3 — stock 추출
- 원본 구간(인터리브 주의): `/* ─── Workspace ─── */`(463) + `Panels & cards`(514) + `/* ─── Stocks topbar ─── */`(563)~`Search / select fields (BacktestView)`(812) **직전**, **그리고** `Stocks layout`(1451) + `Forecast layer groups`(1647) + `Layer toggle`(2218)~`Notices`(2275) **직전**.
- **주의**: 812~1451 구간(backtest, 신호카드 포함)은 stock이 아니라 Step 4로 간다. 즉 stock은 **두 토막**(463~811, 1451~2274)을 모은다. 두 토막을 stock.css 안에서 **원본에 나온 순서대로**(463 토막 먼저, 1451 토막 나중) 이어 붙인다.
- `styles/stock.css`로 이동. layout import 추가(순서: base, shell, **stock**, … globals). globals.css에서 두 토막 삭제.
- 마무리(공통). 커밋: `refactor(css): extract stock.css (CP234 step3)`.

### Step 4 — backtest 추출
- 원본 구간: `/* ─── Search / select fields (BacktestView) ─── */`(812) ~ `/* Stocks layout */`(1451) **직전**. 여기에 `.signal-group-panel`/`.signal-card*`(934~1107) 전부 포함 — **신호카드의 `:hover`/`.is-selected`(992~998) 와 base `.signal-card`(978)가 한 파일·동일 상대순서로 유지되는지 반드시 눈으로 확인.**
- `styles/backtest.css`로 이동. layout import 추가(base, shell, stock, **backtest**, … globals). globals.css에서 삭제.
- 마무리(공통). 신호 카드 선택/hover 상태가 스크린샷에 드러나도록 CP230 baseline이 "카드 선택된 상태"를 포함하는지 확인(없으면 차단 트리거: 상태 회귀를 잡을 수 없음).
- 커밋: `refactor(css): extract backtest.css (CP234 step4)`.

### Step 5 — components 추출
- 원본 구간(인터리브): `/* ─── Notices ─── */`(2275) ~ `/* ─── Training view ─── */`(2649) **직전** (Notices, StatusInline, Metric cards, Charts placeholders, Status badges) **그리고** `/* ─── Tables ─── */`(4857) ~ `/* ─── Responsive ─── */`(4893) **직전**.
- 두 토막(2275~2648, 4857~4892)을 components.css에 원본 순서대로 모은다.
- layout import 추가(base, shell, stock, backtest, **components**, … globals). globals.css에서 두 토막 삭제.
- 마무리(공통). 커밋: `refactor(css): extract components.css (CP234 step5)`.

### Step 6 — training 추출
- 원본 구간: `/* ─── Training view ─── */`(2649) ~ `/* CP217.2 — 통계 검정 ... */`(3310, significance) **직전**. 포함: Training view, `CP216 사용 데이터 그룹화`(3057), `CP216 정적 평가 카드`(3212).
- `styles/training.css`로 이동. layout import 추가(… components, **training**, … globals). globals.css에서 삭제.
- 마무리(공통). 커밋: `refactor(css): extract training.css (CP234 step6)`.

### Step 7 — significance 추출
- 원본 구간: `/* CP217.2 — 통계 검정 ... */`(3310) ~ `/* CP220 — 재현 매니페스트 ... */`(3405) **직전**, **그리고 동봉**: significance 전용 `@media (max-width:920px)` 블록(현재 3803~3847). 단 3405~3803 사이에 reproducibility/GW 내용이 끼어 있으므로 **3803 media 블록만** 골라 옮기고, 그 사이 reproducibility 본문은 Step 8로 둔다.
  - 즉 significance.css = (3310~3404 본문) + (3803~3847 media). 두 조각을 significance.css 안에서 본문 먼저, media 나중 순으로 둔다(소스 순서 보존: base 정의 → media override).
  - **`.significance__finding--good .significance__finding-verdict`(3382) 류 변형 selector가 base finding 뒤에 오는지 확인.**
- layout import 추가(… training, **significance**, … globals). globals.css에서 두 조각 삭제.
- 마무리(공통). 커밋: `refactor(css): extract significance.css (CP234 step7)`.

### Step 8 — reproducibility 추출
- 원본 구간: `/* CP220 — 재현 매니페스트 ... */`(3405) ~ significance media(3803) **직전** 중 **reproducibility 본문**(`/* GW 해석 박스 */` 3608 포함). 즉 3405~3802에서 significance media(3803~)를 뺀 reproducibility/GW 본문.
  - **`.reproducibility[open] > summary`(3431,3433) 등 `[open]` 상태 규칙이 base `.reproducibility > summary`(3412) 뒤에 오는지 확인.**
- `styles/reproducibility.css`로 이동. layout import 추가(… significance, **reproducibility**, … globals). globals.css에서 삭제.
- 마무리(공통). 커밋: `refactor(css): extract reproducibility.css (CP234 step8)`.

### Step 9 — report 추출
- 원본 구간: `/* CP220 — model-role-card ... */`(3849) ~ `/* ─── Tables ─── */`(4857) **직전**, **동봉**: experiment-archive 전용 `@media (max-width:920px)`(4073~4091). 포함: model-role-card, experiment-archive, `.report-view`/`.report-*`(4313~), comparison-table 등.
- **크기 주의**: 이 구간은 약 1000줄로 800 초과 가능. **이 Step 전에 멈추고 사용자에게 보고**: "report 영역이 800줄 초과(~1000). (a) report.css 단일 유지 / (b) report 본문 + report-archive 둘로 분할 중 택1." 합의 전 임의 분할 금지. (b) 합의 시에도 분할 경계에서 cascade 안 깨지게 같은 selector 그룹은 한 파일로.
- layout import 추가(… reproducibility, **report**(혹은 report + report-archive), … globals). globals.css에서 삭제.
- 마무리(공통). 커밋: `refactor(css): extract report.css (CP234 step9)`.

### Step 10 — responsive(전역 @media) 추출 — 최고 위험
- 원본 구간: `/* ─── Responsive ─── */`(4893) ~ `/* CP218 — ... 타임라인 */`(5006) **직전**. 즉 전역 `@media (max-width:900px)` 블록(4894~5004) 통째.
- `styles/responsive.css`로 이동. 이 블록은 stock/training/report/backtest/experiment/components 클래스를 **덮어쓰므로**, layout import에서 **그 모든 base 파일보다 뒤**에 와야 한다. 즉 import 순서상 **거의 마지막**(experiment 직전 또는 직후 모두 가능하나, 진단 4)에 따라 일관되게 experiment 다음·globals 앞 권장).
- globals.css에서 삭제.
- 마무리(공통). **이 Step은 화면 폭을 좁혀(≤900px) 찍은 baseline이 있어야 회귀를 잡는다.** CP230 baseline에 narrow viewport 컷이 없으면 → **차단 트리거**(좁은 화면 회귀 측정 불가)로 보고하고 진행 보류.
- 커밋: `refactor(css): extract responsive.css (CP234 step10)`.

### Step 11 — experiment 추출
- 원본 구간: `/* CP218 — 라인/밴드 정적 실험 타임라인 */`(5006) ~ 파일 끝(5208). `.experiment-timeline*` 전부.
- `styles/experiment.css`로 이동. layout import 추가. globals.css에서 삭제 → **이 시점 globals.css는 비거나 헤더 코멘트만 남는다.**
- 마무리(공통). 커밋: `refactor(css): extract experiment.css (CP234 step11)`.

### Step 12 — globals.css 제거 + layout 정리
- globals.css가 비었는지 확인. 비었으면 파일 삭제, layout.tsx에서 `import "./globals.css";` 제거.
- 최종 layout.tsx import 순서(고정, cascade 결정):
  ```tsx
  import "./styles/base.css";          // (@import 폰트+tokens 포함)
  import "./styles/shell.css";
  import "./styles/stock.css";
  import "./styles/backtest.css";
  import "./styles/training.css";
  import "./styles/components.css";
  import "./styles/report.css";        // (+ report-archive.css if split)
  import "./styles/significance.css";
  import "./styles/reproducibility.css";
  import "./styles/experiment.css";
  import "./styles/responsive.css";    // 전역 @media: base 규칙들 뒤
  ```
  > **주의**: 이 최종 순서는 스펙이 지정한 *논리 순서*(tokens→base→shell→stock→backtest→training→components→report→significance→reproducibility→experiment→responsive)다. 단, 진단 4)에 따라 전역 responsive는 자신이 덮는 모든 영역보다 뒤여야 하므로 **맨 끝**에 둔다. components가 significance/report보다 앞이어도, 영역 간 선택자가 겹치지 않으면(서로 다른 클래스 네임스페이스) 순서 영향이 없다 — 그래도 **Step별로 추가한 순서 = 최종 순서**가 일치하도록 유지하라.
- **대안 검토(보고용)**: globals.css를 지우는 대신 globals.css 안에 `@import "./styles/base.css"; …` 순서로 12줄을 두고 layout.tsx는 `import "./globals.css";` 한 줄을 유지하는 방법도 있다. 단 **CSS `@import`는 런타임 순차 페치라 성능 저하**(Next 빌드시 인라인되긴 하나 권장 안 함). 기본은 layout.tsx 다중 import. `@import` 방식을 쓰려면 사유와 함께 보고.
- 마무리(공통, **전체 회귀 재확인**: 4화면 + narrow 컷 전부 diff 0). 커밋: `refactor(css): drop empty globals.css, finalize import order (CP234 step12)`.

---

## 인터페이스 보존

- **layout.tsx 컴포넌트 시그니처/JSX 불변**: `RootLayout` 함수, `metadata`, `<html lang="ko">`/`<body>` 구조 그대로. 바뀌는 건 **import 라인 묶음뿐**.
- **CSS 클래스명 / selector / 속성 값 전부 불변**: 어떤 컴포넌트도 className을 바꾸지 않는다. 컴포넌트 `.tsx`는 이 CP에서 **건드리지 않는다**(`frontend/src/components/*` 무수정).
- **시각적 출력 불변**: 모든 화면이 픽셀 동일해야 한다(diff 0). 이것이 이 CP의 유일한 동작 계약이다.
- 만약 분리 중 "이 selector를 옮기면 깨질 것 같다"는 판단이 서면 → **옮기지 말고 멈추고 보고**. selector를 고치는 건 인터페이스 변경이며 이 CP 범위 밖.

---

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 |
|---|---|---|
| `globals.css` 줄 수 | 5208 | 0(삭제) 또는 헤더만 |
| 분리 파일 수 | 1(+tokens) | 11 영역 파일 + tokens.css 유지 (report 분할 합의 시 12) |
| 각 영역 파일 줄 수 | — | 500~800 목표(significance/experiment 등은 자연 크기 우선, cascade 보존 > 균형) |
| 4화면 screenshot diff | (baseline=0) | **0**(매 Step, 전 Step 누적) |
| narrow(≤900px) screenshot diff | (baseline) | **0**(Step10 이후 필수) |
| `npx tsc --noEmit` 에러 | 0 | 추가 0 |
| `npm run build` | 성공 | 성공 |
| import 순서 | n/a | Step12 고정 순서와 정확히 일치 |
| 예상 시간 | — | 3~5시간(영역 12 × 촬영/검증 오버헤드) |

> mypy/pytest 항목 해당 없음(프론트 전용).

---

## 검증

각 Step 끝 + 최종. PowerShell 기준(프론트 디렉토리에서):

```powershell
# 1) 타입/빌드 (layout.tsx 변경 검증)
cd C:\Users\user\lens\frontend
npx tsc --noEmit          # 기대: 출력 없음(에러 0)
npm run build             # 기대: "Compiled successfully" / 에러 0

# 2) dev 서버(이미 떠 있으면 재사용)
# npm run dev   # 기대: 컴파일 경고/에러 0, http://localhost:3000

# 3) screenshot diff — CP230이 정의한 명령을 그대로 사용
#    (예시 placeholder — 실제 명령/경로는 CP230 산출물에서 확인해 대체)
#    4화면: Stock / Backtest / Training / Report + narrow(≤900px) 컷
#    기대: 모든 컷 diff = 0 (변경 픽셀 없음)
```

- 줄 수 확인: `(Get-Content .\src\app\styles\<area>.css | Measure-Object -Line).Lines` 로 각 영역 500~800 확인.
- 분리 누락 확인: 모든 Step 후 `globals.css`에 남은 selector가 없는지 Grep(`^\.`/`^@media`/`^[a-z]`)로 점검. 남았는데 의도치 않았으면 어느 영역인지 식별해 옮기되, 애매하면 멈추고 보고.

---

## 차단 트리거 (중요)

> **다음 상황이면 즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지. 이 CP는 본 묶음에서 가장 위험하니, 의심만 들어도 멈춘다.**

1. **CP230 미그린 / baseline 부재**: screenshot baseline 또는 "재촬영+diff" 명령이 없거나 동작하지 않는다 → 측정 불가. **시작하지 마라.**
2. **어느 영역이든 추출 후 screenshot diff ≠ 0**: cascade/specificity가 깨진 증거. **그 Step에서 멈추고**, (a) 어느 영역 파일, (b) 어느 클래스/selector, (c) before/after 어떤 시각 변화(색/여백/상태 하이라이트/레이아웃)인지 적어 보고. 다음 영역으로 절대 넘어가지 마라.
3. **narrow viewport 컷이 baseline에 없음**(Step10 진입 시): 전역 `@media` 회귀를 잡을 수 없음 → Step10 보류·보고.
4. **상태 컷 부재**: 신호카드 선택/hover, `.reproducibility[open]`, significance 변형 등 **상태가 baseline 스크린샷에 안 드러남** → 상태 cascade 회귀가 측정 불가. 해당 영역(backtest/reproducibility/significance) Step 전에 보고.
5. **영역 경계가 코멘트에 안 떨어져 한 selector 그룹이 둘로 쪼개질 위험**: 자르는 지점에서 `.X` base와 `.X:hover`/`.X.is-*`/`.X[open]`/변형이 서로 다른 파일로 갈라진다 → 자르지 말고 보고(경계 재정의 필요).
6. **report(또는 다른) 영역 800줄 초과**로 추가 분할이 필요: 임의 분할 금지. 분할 경계안과 함께 보고·합의.
7. **tsc/build 실패**: import 경로 오타·누락 가능. 고치되, 원인이 불명하거나 동작 변경을 동반하면 보고.
8. **여러 영역을 한 Step에 묶고 싶은 유혹**: 금지. 한 Step = 한 영역. 묶으면 회귀 원인 격리가 불가능해진다.
9. **환경변수/포트 문제로 화면이 baseline과 다르게 렌더**(분리와 무관한 회귀): 먼저 환경부터 baseline 상태로 정리. 정리 안 되면 보고.

보고 형식(차단 시): 어느 Step / 어느 영역 파일 / 어느 selector / 어떤 diff(스크린샷 또는 픽셀 설명) / 추정 원인 / 제안(되돌릴지·경계 재정의할지) 5줄 이내.

---

## ADR

완료 후 `docs/adr/0024-css-12-file-split.md` 1장(200~300단어) 작성:
- **무엇을 기록**: globals.css(5208줄)를 12파일로 분리한 결정 — (1) **분리 경계**(어떤 영역이 어떤 파일로, 신호카드가 왜 backtest인지·"Training view" 코멘트가 왜 training/significance/reproducibility/report/experiment로 쪼개졌는지), (2) **import 순서**(layout.tsx 최종 순서와 왜 그 순서인지: 전역 responsive가 왜 맨 끝인지, scoped @media는 왜 자기 영역 동봉인지), (3) **cascade 보존 규칙**(base→상태/변형 selector 동일 파일·동일 상대순서, 값 무변경, 미사용 규칙 비삭제), (4) 대안(@import 방식)을 왜 안 썼는지, (5) screenshot-gated Strangler 절차로 회귀 0을 어떻게 보장했는지.

`docs/adr/` 디렉토리가 아직 없으므로 **생성**하고 작성한다.

---

## 자가 점검 결과 양식

완료 보고에 아래를 채운다(각 PASS/WARN/FAIL + 한 줄 사유):

- **[Plan v3 정합]**: PASS/WARN/FAIL — 사유: ______ (CSS 분리는 밴드/모델/fidelity와 무관해야 정상. 영향 있으면 FAIL)
- **[구조 결함]**: PASS/WARN/FAIL — 사유: ______ (인터리브 영역을 깔끔히 갈랐는지, 경계에서 selector 그룹이 안 쪼개졌는지)
- **[모델 영향]**: PASS/WARN/FAIL — 사유: ______ (없어야 정상. 프론트 CSS만 변경)

---

## 산출물

- 변경/생성 파일:
  - `frontend/src/app/styles/base.css`, `shell.css`, `stock.css`, `backtest.css`, `training.css`, `components.css`, `report.css`(필요 시 `report-archive.css`), `significance.css`, `reproducibility.css`, `experiment.css`, `responsive.css` (신규)
  - `frontend/src/app/globals.css` (삭제 또는 헤더만)
  - `frontend/src/app/layout.tsx` (import 라인 교체)
  - `frontend/src/app/tokens.css` (불변, 참조만)
  - `docs/adr/0024-css-12-file-split.md` (신규)
- `docs/cp234_report.md`: 요구 / 한 일(영역별 분리 결과·각 파일 줄 수 표) / 결정(경계·import 순서·report 분할 여부) / 후속(미사용 규칙 정리 등 별도 CP 제안). 필요한 만큼만, 장황 금지.
