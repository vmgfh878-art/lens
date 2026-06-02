# CP233 StockView.tsx 순수 helper + IndicatorOptions 분리 (Directive)

## 역할 고정 — 모드 code

- 모드: **code** (구현 + 같은 턴 자가 점검)
- 권한: 프론트 코드 수정, 로컬 dev 검증(tsc / vitest / 브라우저 screenshot)
- 금지: 새 학습, 새 calibration, DB write, Supabase 호출, **사용자가 직접 수정한 파일 revert**, 백엔드 코드 변경, API 응답 schema 변경
- 자가 점검(보고 필수): **Plan v3 정합 / 구조 결함 / 모델 영향** 3축
- 커밋 메시지 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

> 이 CP는 **구조 분리(리팩토링)** 전용이다. 동작·표현·문구·조건 분기를 1글자도 바꾸지 않는다. 추출한 함수 본문은 원본과 **바이트 동일**해야 한다(이동만).

## 환경

- 워킹 디렉토리: `C:\Users\user\lens` (이 worktree 가 아니라 메인 체크아웃. 프론트 소스는 `C:\Users\user\lens\frontend\src` 에만 있다.)
- 프론트 venv 없음. Node 툴체인. 작업 디렉토리 기준 `frontend/` 에서 npm 실행.
- 타입 별칭: `tsconfig.json` 에 `baseUrl: ./src`, `paths: { "@/*": ["./*"] }`. 즉 `@/lib/stock/helpers` → `src/lib/stock/helpers.ts`, `@/components/stock/IndicatorOptions` → `src/components/stock/IndicatorOptions.tsx`.
- 타입 체크 명령: **`npx tsc --noEmit -p tsconfig.json`** (frontend 안에서 실행). `npm run build`(=`next build`)는 느리므로 검증엔 tsc 사용. 단 최종 1회는 `npm run build` 로 next 빌드 회귀까지 확인.
- 백엔드 기동(차트가 가격을 받아야 screenshot 비교 가능): `scripts\start_demo.ps1` 또는 `python -m uvicorn ...`. dev backend default `127.0.0.1:8000`.
- 프론트 dev: `frontend` 에서 `npm run dev` (Next 기본 포트 3000). 이미 떠 있으면 새로 띄우지 말고 재사용. 포트 충돌 시 기존 프로세스 확인 후 진행(임의 kill 금지).
- **테스트 러너 미설치 주의**: 현재 `frontend/package.json` 에 vitest 없음. `node_modules/.bin` 에 `next`, `tsc` 만 있음. → Step 1 에서 vitest 설치가 선행 필요(아래 Sub-step Step 1-0 참조). 설치가 정책상 막히면 **차단 보고**.

## 진단 (근거)

조사 출처: `frontend/src/components/StockView.tsx` 전체 직접 Read + `Grep`(함수명/심볼 호출처) + `frontend/package.json` / `tsconfig.json` 직접 Read (2026-06-02).

`StockView.tsx` 현재 **1097줄** (`wc -l` 확인). 단일 컴포넌트에 (a) 모듈 상단 순수 helper, (b) 거대 `loadStock` + 파생 상태 계산, (c) JSX, (d) 파일 하단 보조 컴포넌트 2개가 한 파일에 뭉쳐 있다. 이 중 **부작용 없이 떼어낼 수 있는 순수 함수**와 **독립 프레젠테이션 컴포넌트**가 명확히 식별된다.

분리 대상 — 모듈 상단 helper (현재 줄번호):

| 심볼 | 현재 위치 | 성격 | StockView 내 호출처 |
|---|---|---|---|
| `getPriceLookbackDays(timeframe)` | L124–130 | 순수 | `fetchPriceHistory`(L181) 만 |
| `getLastPrice(rows)` | L132–134 | 순수 | L505 |
| `getChangePercent(rows)` | L136–146 | 순수 | L507 |
| `getLastFinite(values)` | L148–158 | 순수 | L575 |
| `getLastPoint(points)` | **L160–162** | 순수 | **없음 (dead code)** |
| `buildAiState(timeframe)` | L164–175 | 순수 | L348 |
| `fetchPriceHistory(ticker,timeframe,fullHistory)` | L179–200 | 얇은 fetch | L302 |
| 상수 `FULL_PRICE_HISTORY_START_YEAR=2015` | L120 | 상수 | `fetchPriceHistory`(L191) 만 |

분리 대상 — 파일 하단 컴포넌트:

| 심볼 | 현재 위치 | StockView 내 사용처 |
|---|---|---|
| `interface IndicatorOptionGroupProps` | L1053–1058 | 내부 |
| `function IndicatorOptionGroup` | L1060–1074 | JSX L1030 |
| `interface IndicatorOptionProps` | L1076–1080 | 내부 |
| `function IndicatorOption` | L1082–1097 | `IndicatorOptionGroup` 내부 L1065 |

추가로 발견한 **죽은 코드 2건** (구조 결함, 보고 대상):

1. `getLastPoint` (L160–162): StockView 안에서 호출 0회. `Grep "getLastPoint" frontend/src` 결과 — StockView 정의 1건 + `lib/predictionOverlay.ts` 의 **동명이지만 별개**인 로컬 함수(L154 정의, L180·L195 자체 사용)뿐. 즉 StockView 의 것은 어디서도 안 쓴다. → **helpers 로 옮기지 말고 삭제**한다(죽은 코드를 공유 모듈로 옮기면 죽은 코드만 퍼진다).
2. import L25 `import IndicatorPanel, { IndicatorChartPoint, IndicatorChartSeries } from "@/components/IndicatorPanel";` 의 named import 중 `IndicatorChartPoint` 는 위 `getLastPoint` 시그니처에서만 쓰이고, `IndicatorChartSeries` 는 **파일 전체에서 사용 0회**(`Grep` 확인). → `getLastPoint` 삭제 후 두 named import 모두 미사용이 되므로 L25 를 `import IndicatorPanel from "@/components/IndicatorPanel";` 로 정리.

목표: **1097줄 → 약 900줄**. 메인 로직(신선도 판정, AI 상태 12조건 분기 L423–444, 토글 disable 사유 우선순위 L631–658, 파생 라벨 계산 L505–766)은 조건부 렌더와 강하게 얽혀 있어 **이번 CP 범위 밖(유지)**.

## 선행 의존

- **CP230 (프론트 characterization 스냅샷) 그린** 이 선행. CP230 에서 만든 StockView screenshot/스냅샷 베이스라인이 그린이 아니면 이 CP 시작 금지.
- 전역 안전망 규칙: CP230 그린 전 프론트 분리(CP231~) 금지. 이 문서는 그 규칙을 따른다.
- CP230 베이스라인이 없거나 깨져 있으면 → **즉시 차단 보고** (안전망 없이 분리 진행 금지).

## 범위

### 포함

1. `frontend/src/lib/stock/helpers.ts` 신설 — 순수 helper `getPriceLookbackDays` / `getLastPrice` / `getChangePercent` / `getLastFinite` / `buildAiState` + 얇은 fetch `fetchPriceHistory` + 상수 `FULL_PRICE_HISTORY_START_YEAR` 이동. (약 60줄)
2. `frontend/src/components/stock/IndicatorOptions.tsx` 신설 — `IndicatorOptionGroup` + `IndicatorOption` (+ 두 props 인터페이스) 이동. (약 50줄)
3. `getLastPoint`(L160–162) **삭제** + L25 import 정리.
4. `StockView.tsx` 의 caller 를 새 모듈 import 로 이전, 옛 정의 제거.
5. helpers 순수 함수에 대한 **vitest 단위 테스트** 신설(+ 러너 설치).

### 제외

- 메인 로직(`loadStock`, 12조건 AI 상태 분기, 토글 disable 사유, 파생 라벨) 분해 — 유지.
- `Chart` / `IndicatorPanel` / `LayerToggle` / `StatusInline` 내부 변경 — 손대지 않음.
- API client(`@/api/client`) / 백엔드 / Supabase — 보류·미접촉.
- 동작·문구·조건·스타일 변경 — 전면 금지.
- `PRODUCT_HISTORY_LOOKBACK_DAYS`(L119) 이동 — **금지**. L362·L374·L382 에서도 쓰이므로 StockView 에 남긴다(이건 fetchPriceHistory 전용이 아님).

## Sub-step (Strangler Fig, 작은 단위)

각 Step = 한 revert 단위. "옛 코드 옆 새 코드 공존 → caller 이전 → 옛 제거" 패턴. 추출 순서는 전역 규칙대로 **순수 함수 → I/O 경계 → 컴포넌트 → 정리**.

---

### Step 1 — helpers.ts 추출 + vitest

**Step 1-0 (러너 선행, 별도 커밋):**
- `frontend` 에서 vitest 설치: `npm install -D vitest`.
- `frontend/vitest.config.ts` 생성. `@/*` 별칭을 tsconfig 와 동일하게 풀도록 alias 설정(`resolve.alias['@'] = path.resolve(__dirname, './src')`), `test.environment = 'node'`(helpers 는 DOM 불필요).
- `frontend/package.json` `scripts` 에 `"test": "vitest run"` 추가.
- 검증: `npx vitest run` 이 "테스트 0개"라도 정상 종료(exit 0)되는지 확인.
- 커밋: `chore(frontend): add vitest runner`
- **차단**: `npm install` 이 네트워크/정책으로 실패하면 멈추고 보고(설치 없이 임의 우회 금지).

**Step 1-1 (새 코드 공존):**
- `frontend/src/lib/stock/helpers.ts` 생성. 아래 심볼을 **본문 변경 없이** 이동하고 `export` 부여:
  - `FULL_PRICE_HISTORY_START_YEAR` (L120), `getPriceLookbackDays` (L124–130), `getLastPrice` (L132–134), `getChangePercent` (L136–146), `getLastFinite` (L148–158), `buildAiState` (L164–175), `fetchPriceHistory` (L179–200).
- `AiState` 타입은 `buildAiState` 반환형이라 필요. StockView 의 `AiState`(L97–100)는 **그대로 두고**, helpers.ts 는 동일 형태의 타입을 자체 정의하거나 공유 타입으로 분리. (가장 단순: helpers.ts 에 `export interface AiState {...}` 를 두고 StockView 가 이를 import 하도록. 단 이 변경이 다른 import 를 깨지 않는지 tsc 로 확인.)
- helpers.ts 가 import 해야 할 외부 심볼: `DisplayTimeframe`, `fetchPrices`, `PriceBar` (`@/api/client`), `PRICE_LOOKBACK_LIMIT_1D`, `PRICE_LOOKBACK_LIMIT_1W` (`@/lib/constants`), `buildDefaultPriceWindow`, `buildFullPriceWindows`, `sortPriceRows` (`@/lib/dateUtils`).
- 이 시점엔 StockView 는 아직 옛 로컬 정의를 쓴다(공존).
- 검증: `npx tsc --noEmit -p tsconfig.json` → 에러 0.

**Step 1-2 (테스트):**
- `frontend/src/lib/stock/__tests__/helpers.test.ts` 작성. 순수 함수 위주(부작용 없는 것만):
  - `getLastPrice([])` → null / 비어있지 않으면 마지막 bar.
  - `getChangePercent`: 길이 <2 → null, previous=0 → null, 정상 % 계산.
  - `getLastFinite`: null/undefined → null, 뒤에서 첫 finite 반환, 전부 NaN → null.
  - `getPriceLookbackDays('1W')` === `PRICE_LOOKBACK_LIMIT_1W`, `'1D'`/`'1M'` === `PRICE_LOOKBACK_LIMIT_1D`.
  - `buildAiState('1M')` → kind `disabled`, 그 외 → kind `empty` (message 문자열까지 현재 값과 일치하는지 고정).
  - `fetchPriceHistory` 는 `fetchPrices` 를 mock(`vi.mock('@/api/client')`) 해서 호출 인자(start/end/timeframe)와 `sortPriceRows` 통과만 확인. 네트워크 실호출 금지.
- 검증: `npx vitest run` → 전부 통과.

**Step 1-3 (caller 이전 + 옛 제거):**
- StockView 의 호출부를 새 import 로 교체: L302 `fetchPriceHistory`, L348 `buildAiState`, L505 `getLastPrice`, L507 `getChangePercent`, L575 `getLastFinite`. (`getPriceLookbackDays` 는 StockView 가 직접 호출 안 함 — helpers 내부에서만 쓰임.)
- StockView 상단 import 블록에 `import { buildAiState, fetchPriceHistory, getChangePercent, getLastFinite, getLastPrice } from "@/lib/stock/helpers";` 추가.
- StockView 에서 옛 로컬 정의(L120, L124–130, L132–134, L136–146, L148–158, L164–175, L179–200) **삭제**. `AiState` 를 helpers 로 옮겼다면 L97–100 정의도 제거하고 import 로 대체.
- 검증: `npx tsc --noEmit` 0 + `npx vitest run` 통과.
- 커밋: `refactor(stock): extract price/ai-state helpers to lib/stock/helpers`

---

### Step 2 — IndicatorOptions.tsx 추출

**Step 2-1 (새 코드 공존):**
- `frontend/src/components/stock/IndicatorOptions.tsx` 생성. L1053–1097 의 `IndicatorOptionGroupProps` / `IndicatorOptionGroup` / `IndicatorOptionProps` / `IndicatorOption` 를 **본문 변경 없이** 이동.
- `export function IndicatorOptionGroup` (외부에서 쓰임), `IndicatorOption` 은 같은 파일 내부에서만 쓰이면 export 불필요(원본과 동일하게 비-export 유지 가능).
- import 필요 심볼: `IndicatorDefinition`, `IndicatorId` (`@/lib/indicators`).
- 검증: `npx tsc --noEmit` 0.

**Step 2-2 (caller 이전 + 옛 제거):**
- StockView L1030 의 `<IndicatorOptionGroup .../>` 사용은 그대로 두고, 상단에 `import { IndicatorOptionGroup } from "@/components/stock/IndicatorOptions";` 추가.
- StockView 하단의 옛 정의(L1053–1097) 삭제.
- 검증: `npx tsc --noEmit` 0.
- 커밋: `refactor(stock): extract IndicatorOptions to components/stock`

---

### Step 3 — 죽은 코드 제거 + import 정리

- `getLastPoint`(L160–162) 삭제. **사전 확인 필수**: `Grep "getLastPoint" frontend/src/components/StockView.tsx` 가 정의 1건 외 0건인지(이미 Step 1 에서 주변이 빠졌으니 라인 이동 감안). 호출이 1건이라도 있으면 삭제 금지하고 보고.
- L25 import 를 `import IndicatorPanel from "@/components/IndicatorPanel";` 로 정리(`IndicatorChartPoint`, `IndicatorChartSeries` named import 제거). **사전 확인**: 두 심볼이 파일 전체에서 미사용인지 `Grep` 재확인(`getLastPoint` 삭제 후 `IndicatorChartPoint` 사용 0, `IndicatorChartSeries` 원래 0).
- 남은 미사용 import(예: 다른 helper 이동으로 떠버린 것) 점검·정리.
- 검증: `npx tsc --noEmit` 0 + `npx vitest run` 통과 + `npm run build`(next 빌드) 성공.
- 커밋: `refactor(stock): drop dead getLastPoint + tidy imports`

---

> Step 1·2·3 각각 끝에 **screenshot diff 비교**(아래 검증) 수행. 어느 Step 이라도 차트/오버레이 시각 변화가 보이면 그 Step 만 revert 하고 차단 보고.

## 인터페이스 보존

- 추출 함수 시그니처 **불변**: `getPriceLookbackDays(timeframe)`, `getLastPrice(rows)`, `getChangePercent(rows)`, `getLastFinite(values)`, `buildAiState(timeframe)`, `fetchPriceHistory(ticker, timeframe, fullHistory=false)` 인자·반환형 그대로.
- 컴포넌트 props 인터페이스 **불변**: `IndicatorOptionGroupProps`(title/options/selectedIndicators/onChange), `IndicatorOptionProps`(option/checked/onChange) 필드 그대로.
- **API 응답 schema·백엔드 호출 0 변경.** `fetchPriceHistory` 가 부르는 `fetchPrices` 인자도 동일.
- `StockView` 의 export(default)·props(없음)·렌더 출력 DOM 구조 불변.
- 위를 바꿔야 하는 상황이 생기면(예: `AiState` 공유가 타사용처를 깸) → 호출자 영향 분석 후 **차단 보고**, 임의 변경 금지.

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 |
|---|---|---|
| `StockView.tsx` 줄 수 | 1097 | **≤ 905** (목표 ~900, 허용 상한 905) |
| `lib/stock/helpers.ts` | 없음 | 신설 ~60줄 (±15) |
| `components/stock/IndicatorOptions.tsx` | 없음 | 신설 ~50줄 (±10) |
| vitest 테스트 | 0개 | helpers 케이스 전부 통과(회귀 0) |
| `npx tsc --noEmit` 추가 에러 | 0 | **0** |
| `npm run build`(next) | 성공 | 성공 유지 |
| StockView screenshot diff | — | **허용오차 내 / 차트·오버레이 픽셀 변화 0** |
| 예상 시간 | — | 약 2.5시간 |

## 검증

PowerShell(워킹 디렉토리 `C:\Users\user\lens`):

```powershell
# 타입 체크 (각 Step 후)
Set-Location frontend
npx tsc --noEmit -p tsconfig.json   # 기대: 출력 없음, exit 0

# 단위 테스트 (Step 1 이후)
npx vitest run                      # 기대: helpers 케이스 all pass

# next 빌드 회귀 (Step 3 마지막 1회)
npm run build                       # 기대: 빌드 성공

# 줄 수 확인
(Get-Content src/components/StockView.tsx | Measure-Object -Line).Lines   # 기대: <= 905
(Get-Content src/lib/stock/helpers.ts | Measure-Object -Line).Lines
(Get-Content src/components/stock/IndicatorOptions.tsx | Measure-Object -Line).Lines
```

Screenshot 회귀(각 Step 후):
- 백엔드 기동(`scripts\start_demo.ps1`) + `npm run dev` 후 `http://localhost:3000` 에서 StockView 로드.
- CP230 베이스라인과 동일 조건(기본 `AAPL` / `1D` / 캔들, 그리고 `1W` 전환, AI 밴드·보수적 기준선 토글 ON 상태)으로 캡처.
- 비교 대상: **가격 차트, AI 밴드 오버레이, 보수적 기준선, 하단 지표, 모델 정보 패널 라벨**.
- 기대: CP230 베이스라인과 픽셀 동일(렌더 타이밍 antialiasing 수준의 허용오차 내). 텍스트 라벨 1글자도 변하면 안 됨.

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **차트/예측 오버레이 screenshot diff 발생** — AI 밴드·보수적 기준선·캔들·지표 중 하나라도 시각 변화 = 동작이 바뀐 것. 해당 Step revert 후 보고.
2. **모델 정보 패널 텍스트 라벨 변화** — 신선도 배지/문구/숫자가 베이스라인과 달라지면 = 파생 계산이 깨진 것. 보고.
3. `npx tsc --noEmit` 에 **신규 에러** 발생 — 특히 `AiState`/`IndicatorChartPoint`/`IndicatorChartSeries` 타입 정리가 다른 import 를 깬 경우. 보고.
4. **vitest 가 helpers 동작 차이를 잡음** — 추출이 "이동"이 아니라 "변형"이 된 것. 보고.
5. `getLastPoint` 또는 `IndicatorChartPoint`/`IndicatorChartSeries` 에 **예상 못 한 사용처**가 grep 으로 나옴 — 삭제 전제 붕괴. 삭제 보류·보고.
6. **CP230 베이스라인이 그린이 아님/부재** — 안전망 없음. 분리 시작 금지·보고.
7. `npm install -D vitest` 또는 `npm run build` **실패**(네트워크/정책/빌드) — 우회 금지·보고.
8. `npm run dev` 포트 충돌로 기존 프로세스를 죽여야 하는 상황 — 임의 kill 금지·보고.
9. 한 커밋에 구조 변경 + 동작 변경이 섞이게 되는 정황 — 멈추고 분리·보고.

## ADR

해당 없음 (단순 추출 — 아키텍처 결정 없음). 단, 작업 중 "메인 로직(12조건 AI 상태 분기 등)을 왜 분리 안 했는가"에 대한 판단을 남길 가치가 있다고 보이면 `docs/adr/0023-stockview-partial-split.md` 에 200~300단어 1장으로 "순수/프레젠테이션만 분리, 상태 결합 로직은 후속 CP 로 보류" 결정을 기록(선택). 기본은 생략.

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다(빈칸 금지).

- [Plan v3 정합] PASS / WARN / FAIL — 사유:
- [구조 결함] PASS / WARN / FAIL — 사유: (dead code `getLastPoint` 제거 여부 포함)
- [모델 영향] PASS / WARN / FAIL — 사유: (예측 오버레이 동작 불변 확인 여부)

## 산출물

- 변경/신설 파일:
  - 신설 `frontend/src/lib/stock/helpers.ts`
  - 신설 `frontend/src/lib/stock/__tests__/helpers.test.ts`
  - 신설 `frontend/src/components/stock/IndicatorOptions.tsx`
  - 신설 `frontend/vitest.config.ts`
  - 수정 `frontend/src/components/StockView.tsx` (1097 → ≤905)
  - 수정 `frontend/package.json` (vitest devDep + test script)
- 보고서: `docs/cp233_report.md` — 요구 / 한 일(Step별 커밋) / 결정(dead code 삭제·import 정리) / 후속(메인 로직 분리 보류 사유) 를 필요한 만큼만.
