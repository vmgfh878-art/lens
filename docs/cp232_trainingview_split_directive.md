# CP232 TrainingView.tsx 분리 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지, 본문에 박힌 파일:줄번호를 기준으로 한다.

## 역할 고정

- **모드**: `code` (구현 + 자가 점검). 기획/설계 모드 아님.
- **권한**: 프론트엔드 코드 수정, 로컬 검증(`tsc`, dev server 기동, 브라우저/preview 시각 확인)만.
- **금지**:
  - 새 학습(training) 실행 금지.
  - 새 calibration / 평가 산출물 생성 금지.
  - DB write 금지. Supabase 호출/스키마 변경 금지.
  - 사용자가 직접 수정한 파일을 임의 revert 금지.
  - 동작(behavior) 변경 금지 — 이 CP는 **순수 구조 분리(behavior-neutral)** 다. 화면 출력 1px도 바뀌면 안 된다.
- **자가 점검**: 완료 후 [Plan v3 정합] [구조 결함] [모델 영향] 3축 PASS/WARN/FAIL 보고(양식 하단).
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **프론트엔드 루트**: `C:\Users\user\lens\frontend`
- **스택(확인됨, `frontend/package.json`)**: Next.js `14.2.3`, React `^18`, TypeScript `^5`.
  - `tsconfig.json`: `strict: true`, `noEmit: true`, paths `@/* → ./*`.
  - **ESLint 없음**(`eslint` / `eslint-config-next` 미설치). → `next lint` 쓰지 마라. lint 검증은 `tsc`로 대체.
  - **테스트 러너 없음**: Vitest / Jest / Playwright / Cypress **모두 미설치**, 테스트 파일 0개. → `npm test` / `vitest` 명령 만들지 마라. 회귀 검증은 `tsc` + 수동 시각 비교다(검증 섹션 참조).
- **Python venv**(이 CP에선 안 씀, 참고): `.venv` Python 3.10.0, torch 2.11.0+cu128.
- **기동**: `scripts\start_demo.ps1` (프론트 `http://127.0.0.1:3000`, 백엔드 `http://127.0.0.1:8000`). 또는 프론트만 `cd frontend; npm run dev`.
- **포트 충돌**: 3000 점유 중이면 `start_demo.ps1`은 자동 종료하지 않는다. 이미 dev server가 떠 있으면 그걸 재사용하고, 없을 때만 새로 띄운다. 새 포트를 임의로 쓰지 말 것(시각 baseline 일관성 위해 3000 고정).
- **typecheck 기준 명령**: `cd C:\Users\user\lens\frontend; npx tsc --noEmit` — **시작 시점 baseline은 EXIT 0 (clean)** 임을 본 지시서 작성 중 확인함. 이 CP 종료 시에도 EXIT 0 이어야 한다(신규 에러 0).

## 진단 (근거)

**대상 파일**: `frontend/src/components/TrainingView.tsx` — **현재 1560줄**(작성 시점 `wc -l` 확인).

이 파일은 이미 `lib/training/*`(constants/formatters/runUtils/detailFields/staticEvaluation/reproducibility/usageData/lineTimeline/bandTimeline)와 `components/training/*`(ExperimentTimeline/LineExperimentArchive/Band1d·1wExperimentArchive/ReproducibilitySection/UsageDataSection)로 **분리가 시작됐다**. CP232는 그 패턴을 이어 4덩어리를 더 빼낸다.

남아있는 분리 후보(모두 `TrainingView.tsx` 내부, 줄번호는 작성 시점 기준):

1. **순수 데이터 변환 빌더**(부작용 없음 → 1순위 추출):
   - `buildLineExperimentCards(detail)` — **975–1031** (≈57줄). `LINE_METRICS` 4개 지표 → `GoalCardData[]`.
   - `buildBandExperimentCards(detail)` — **1033–1102** (≈70줄). `BAND_METRICS` 5개 지표 → `GoalCardData[]`.
   - `buildExperimentCards(detail)` — **1104–1106** (3줄). kind 분기.
   - `hasDisplayableExperimentMetrics(detail)` — **1108–1110** (3줄). `buildExperimentCards(...).length >= 2`.
   - `getExperimentFailureReason(detail)` — **1112–1143** (≈32줄). **현 시점 호출자 없음(dead code)** — `Grep`으로 정의 외 참조 0건 확인. 처리 방침은 [범위] 참조(이 CP에서 삭제 금지, 그대로 이전만).
   - 합계 ≈ 168줄. 이들은 React/JSX 없는 순수 함수라 가장 안전한 추출 단위.
2. **정적 섹션 컴포넌트**(props 1개 `slotId`, 내부에서 `lib/training/staticEvaluation` accessor 호출, empty면 `null` 반환 — 기존 `UsageDataSection`/`ReproducibilitySection`과 동일 패턴):
   - `SignificanceSection({ slotId })` — **399–549** (≈151줄). `getStaticSignificance(slotId)` 사용. CP216.2 통계 검정 표(DM·Bootstrap CI·GW regime). **화면 핵심 보존 대상.**
   - `PptMappingSection({ slotId })` — **551–579** (≈29줄). `getPptMapping(slotId)`.
   - `V1ExtraIndicatorsSection({ slotId })` — **581–606** (≈26줄). `getV1ExtraIndicators(slotId)`.
3. **비교 표 컴포넌트**:
   - `ComparisonTable({ rows })` — **1200–1230** (31줄). `ComparisonRow[]` → `<table>`. `ExperimentDetail`(1285줄)에서만 사용.

**의존 관계(중요, 추출을 막는 지점)**:
- `GoalCardData`(인터페이스, **139–141**)는 `GoalCardProps`(**125–137**)를 extend한다. 둘 다 `TrainingView.tsx` 내부 정의. 빌더들이 `GoalCardData[]`를 반환하므로, 빌더를 `lib`로 빼면 **이 타입을 공유 가능한 위치로 옮기거나 재노출**해야 한다(인터페이스 보존 섹션 참조).
- `ComparisonRow`(인터페이스, **113–123**)도 `TrainingView.tsx` 내부 정의. `ComparisonTable` props가 이걸 받는다.
- 빌더가 쓰는 헬퍼: `getMetricNumberFromStoredEvaluation`(183–185), `getMetricNumber`(177–181), `getExperimentKind`(from `@/lib/training/runUtils`), `formatMetric`/`formatSignedNumber`/`formatSignedPctPoint`(from `@/lib/training/formatters`). 이 중 `getMetricNumber*`는 현재 `TrainingView.tsx` 내부 함수다 → 빌더와 함께 이동 필요.

**유지(분리 안 함)**: `LineModelDetail`(796–872, ≈77줄)·`BandModelDetail`(874–973, ≈100줄)은 100줄 안팎이라 그대로 둔다(스펙 명시). `ExperimentDetail`(1232–1301)·메인 `TrainingView`(1303–1560)도 이번엔 본체에 남긴다.

**스펙 대비 실태 정정(반드시 인지)**:
- 스펙은 "CP220 ReproducibilitySection 산출물 화면 영향"을 보존 대상으로 적었으나, **`ReproducibilitySection`/`UsageDataSection`은 현재 `TrainingView.tsx`에서 import/렌더되지 않는다**(`Grep` 확인: TrainingView 내 참조 0건). 즉 현재 화면에 안 뜬다(orphan 컴포넌트). 따라서 이 CP의 시각 보존 핵심은 **SignificanceSection(통계 검정) + StoredEvaluationSection(목표 대비 평가, 빌더 출력) + PptMappingSection(초기 계획 평가)** 이며, Reproducibility/UsageData는 이 CP 범위 밖이다(건드리지 마라). 이 사실을 report에 적어라.
- 스펙은 "Vitest" 검증을 적었으나 **리포지토리에 Vitest가 없다**(위 환경 확인). 따라서 검증은 `tsc --noEmit` + 수동 시각 비교로 한다. Vitest 도입은 이 CP 범위 아님(별도 CP).

**조사 출처**: `Read frontend/src/components/TrainingView.tsx`(1–1560 전체), `Read .../training/ReproducibilitySection.tsx`·`UsageDataSection.tsx`(추출 패턴 표본), `Grep`(staticEvaluation export · GoalCardData/ComparisonRow 사용처 · Reproducibility 미사용), `frontend/package.json`·`tsconfig.json`·`scripts/start_demo.ps1`, `npx tsc --noEmit`(baseline EXIT 0).

## 선행 의존

- **CP230(프론트 characterization) 그린**: 필수. CP230이 TrainingView를 포함한 프론트 화면의 시각/동작 baseline(스냅샷 또는 기준 스크린샷)을 그린으로 만들어야 한다. **CP230 그린 전에는 이 CP(CP231~ 프론트 분리)를 시작하지 마라.** 런북이 순서를 보장하지만, 시작 직전 CP230 산출물(기준 스크린샷/스냅샷)이 존재하고 비교 가능 상태인지 1차 확인하라. 없으면 [차단 트리거]로 즉시 중단·보고.
- 그 외 백엔드 CP(CP223 등)와는 무관(이 CP는 프론트 only, API 호출 schema 불변).

## 범위

**포함**:
- `experimentBuilder.ts`(순수 빌더 5개 + 그들이 의존하는 `getMetricNumber*` 헬퍼) 추출.
- `SignificanceSection.tsx` / `ProductMetricsSection.tsx`(= `PptMappingSection` + `V1ExtraIndicatorsSection` 묶음) / `ComparisonTable.tsx` 추출.
- 추출 후 `TrainingView.tsx`의 옛 정의 제거 + import 교체.
- 공유 타입(`GoalCardData`/`GoalCardProps`/`ComparisonRow`) 재배치(인터페이스 보존 섹션 규칙대로).

**제외(건드리지 마라)**:
- `ReproducibilitySection`/`UsageDataSection` 및 그 lib(현재 미렌더 orphan). 이번 CP에서 렌더 추가/삭제/수정 금지.
- `LineModelDetail`/`BandModelDetail`/`ExperimentDetail`/메인 `TrainingView` 본체 로직 변경(import 교체로 인한 줄 삭제 외 동작 수정 금지).
- `getExperimentFailureReason` **삭제 금지**(dead code지만 이 CP는 구조 이동만; 삭제는 동작/범위 변경이라 별도 cleanup CP). 그대로 `experimentBuilder.ts`로 옮겨 export만 한다. report 후속에 "dead code 삭제 후보"로 1줄 남겨라.
- Supabase/DB/백엔드/학습/calibration 일체.
- CSS 클래스명 변경 금지(스타일은 전역 CSS가 클래스로 매칭 — className 문자열 1글자도 바꾸면 시각 회귀).

## Sub-step (Strangler Fig, 작은 단위)

> 원칙: 각 Step = "옛 코드 옆 새 코드 공존 → caller 이전 → 옛 제거" + **한 Step = 한 commit = 한 revert 단위**. 추출 순서는 순수 함수 → 정적 컴포넌트 → 본체 정리. 각 Step 끝에서 `tsc --noEmit` EXIT 0 확인. Step 3·5(시각 영향 있는 섹션 이전)에서는 추가로 수동 시각 비교.

### Step 1 — 순수 빌더 신규 파일 생성(공존, caller 미이전)
- 새 파일 `frontend/src/lib/training/experimentBuilder.ts` 생성.
- 다음을 **복사**(아직 옛 코드 삭제 X)해 옮기고 `export` 부여:
  - 헬퍼: `getMetricByKeys`(154–165, 단 이미 다른 곳에서 안 쓰면 internal로), `getMetricNumber`(177–181), `getMetricNumberFromStoredEvaluation`(183–185). *주의*: `getMetricByKeys`/`getMetricText`/`getMetricNumber`가 `TrainingView` 다른 함수(`ModelRunDetails` 등)에서도 쓰이는지 `Grep`로 확인. 공유되면 **빌더 파일이 그걸 import**하도록 별도 헬퍼 모듈(`lib/training/metricAccess.ts`)로 빼거나, `formatters`/`detailFields` 기존 모듈에 합쳐라(중복 정의 2벌 만들지 말 것).
  - 빌더: `buildLineExperimentCards`, `buildBandExperimentCards`, `buildExperimentCards`, `hasDisplayableExperimentMetrics`, `getExperimentFailureReason`.
- 빌더 반환 타입 `GoalCardData`를 위해: `GoalCardProps`/`GoalCardData` 인터페이스를 **`lib/training/cardTypes.ts`(신규)** 로 이동하고 `export`. `TrainingView.tsx`와 `experimentBuilder.ts` 양쪽이 여기서 import. (대안으로 `staticEvaluation.ts`의 `StaticGoalCard`와 합치는 건 의미 변동 위험 → 별도 `cardTypes.ts` 권장.)
- import: `AiRunDetail`(`@/api/client`), `LINE_METRICS`/`BAND_METRICS`/`MetricDefinition`(`@/lib/training/constants`), `getExperimentKind`(`@/lib/training/runUtils`), `formatMetric`/`formatSignedNumber`/`formatSignedPctPoint`(`@/lib/training/formatters`).
- 이 Step에서는 `TrainingView.tsx`를 아직 안 고친다(빌더는 양쪽에 잠시 공존, 새 파일은 어디서도 import 안 됨 — 죽은 새 파일 상태 OK).
- **검증**: `npx tsc --noEmit` EXIT 0. **commit**: `refactor(fe): add experimentBuilder lib (CP232 step1)`.

### Step 2 — caller 이전 + 옛 빌더 제거
- `TrainingView.tsx`에서 위 빌더/헬퍼 정의(975–1143 영역 + 옮긴 헬퍼)를 **삭제**하고, 상단 import 블록에 `from "@/lib/training/experimentBuilder"`로 교체. `GoalCardData`/`GoalCardProps`는 `from "@/lib/training/cardTypes"`로 교체.
- `LineModelDetail`(빌더 호출 803줄 부근)·`BandModelDetail`(881줄 부근)·`loadRuns`(`hasDisplayableExperimentMetrics` 1419줄)·`experimentGroups`(`hasDisplayableComparison`→내부 `buildComparisonRows`는 그대로 둠) 호출부가 새 import를 가리키는지 확인.
- **검증**: `npx tsc --noEmit` EXIT 0. 줄 수 감소 확인(`wc -l TrainingView.tsx`). **commit**: `refactor(fe): move card builders out of TrainingView (CP232 step2)`.

### Step 3 — 정적 섹션 추출: ProductMetricsSection (PptMapping + V1Extra)
- 새 파일 `frontend/src/components/training/ProductMetricsSection.tsx`.
  - `PptMappingSection`(551–579)·`V1ExtraIndicatorsSection`(581–606)을 **그대로** 옮긴다(JSX·className 문자열 1글자도 변경 금지).
  - export 형태 2안 중 택1, 호출부 영향 최소화: (a) 두 컴포넌트를 named export로 그대로 노출, 또는 (b) `ProductMetricsSection({ slotId })`로 묶어 내부에서 둘을 순서대로 렌더. **현재 `LineModelDetail`/`BandModelDetail`은 `<V1ExtraIndicatorsSection/>` 다음 `<PptMappingSection/>` 순(863–864, 965–966)으로 둘을 개별 렌더**하므로, 순서/래퍼 div가 안 바뀌도록 (a) named export가 더 안전하다. (b)로 묶을 경우 렌더 순서를 현재와 정확히 동일(V1Extra → PptMapping)하게 하고 두 `<section>` 사이에 래퍼 추가 금지.
  - import: `getPptMapping`/`getV1ExtraIndicators`(`@/lib/training/staticEvaluation`).
- `TrainingView.tsx`: 두 함수 정의 삭제 + import 교체. `LineModelDetail`/`BandModelDetail`의 호출 JSX는 동일 컴포넌트명(또는 묶음)으로 유지.
- **검증**: `npx tsc --noEmit` EXIT 0. **+ 수동 시각 비교**(검증 섹션 절차): line-1d/band-1d/band-1w 슬롯에서 "초기 계획 평가"·"초기 계획과 차이점" 영역 무변경. **commit**: `refactor(fe): extract ProductMetricsSection (CP232 step3)`.

### Step 4 — ComparisonTable 추출
- 새 파일 `frontend/src/components/training/ComparisonTable.tsx`. `ComparisonTable`(1200–1230)을 옮기고 `ComparisonRow` 타입은 Step1에서 만든 `cardTypes.ts`에 함께 두거나 `comparisonTypes.ts` 신설 후 양쪽 import. (`ComparisonRow`는 `buildComparisonRows`/`getMetricInterpretation` 등 `TrainingView` 내부 여러 함수가 참조하므로, 타입만 공유 모듈로 빼고 함수들은 본체에 남긴다.)
- `TrainingView.tsx`: `ComparisonTable` 정의 삭제 + import 교체. `ExperimentDetail`(1285줄 `<ComparisonTable rows=.../>`) 그대로 동작.
- **검증**: `npx tsc --noEmit` EXIT 0. (ComparisonTable은 실험 상세 진입 시에만 보임 — 시각 비교는 Step6에 합쳐도 됨.) **commit**: `refactor(fe): extract ComparisonTable (CP232 step4)`.

### Step 5 — SignificanceSection 추출 (시각 핵심)
- 새 파일 `frontend/src/components/training/SignificanceSection.tsx`. `SignificanceSection`(399–549)을 **글자 단위로 그대로** 옮긴다. 이 컴포넌트는 `verdictClass` 내부 함수·`getStaticSignificance` 의존·다수의 className(`significance__*`)·`role="table"` 구조·`<details>`/`<summary>`·footnote 조건부 렌더를 포함 — **어느 것도 수정 금지**.
  - import: `getStaticSignificance`(`@/lib/training/staticEvaluation`). `SignificanceBlock` 타입이 export 안 돼 있으면 `getStaticSignificance` 반환 타입 추론으로 충분(별도 type import 불필요). 필요 시 `staticEvaluation.ts`에서 타입 export 추가는 허용(behavior 무관).
- `TrainingView.tsx`: 정의 삭제 + import 교체. `LineModelDetail`(865줄)·`BandModelDetail`(967줄) `<SignificanceSection slotId={slot?.id} />` 그대로.
- **검증**: `npx tsc --noEmit` EXIT 0. **+ 수동 시각 비교 필수**: line-1d 슬롯 "통계 검정" 섹션 — headline·8셀 표·`<details>` 펼침(DM·Bootstrap CI·GW regime)·footnote(`*`/`†`)·GW interpretation 블록까지 픽셀 동일. band-1w 슬롯의 GW regime sub-table도 확인. **commit**: `refactor(fe): extract SignificanceSection (CP232 step5)`.

### Step 6 — 본체 정리 + 최종 검증
- `TrainingView.tsx` 상단 import 정리(미사용 import 제거 — `getPptMapping`/`getV1ExtraIndicators`/`getStaticSignificance`가 본체에서 더는 직접 안 쓰이면 import에서 제거). 단 `getStaticEvaluation`/`StaticGoalCard`는 `LineModelDetail`/`BandModelDetail`이 계속 쓰므로 유지.
- 최종 `wc -l TrainingView.tsx`로 목표(~900줄, ≤950 허용) 확인.
- **검증**: `npx tsc --noEmit` EXIT 0 + 전체 화면 수동 시각 회귀(line-1d/line-1w/band-1d/band-1w 4슬롯 + 실험 상세 1개 진입). **commit**: `refactor(fe): finalize TrainingView split (CP232 step6)`.

## 인터페이스 보존

- **API 응답 schema 불변**: `fetchAiRun`/`fetchAiRuns` 호출 인자·응답 사용 방식 변경 금지. 백엔드 무관.
- **컴포넌트 props 인터페이스 보존**:
  - 추출되는 섹션 컴포넌트의 props는 **현재와 동일**해야 한다: `SignificanceSection({ slotId: string|null|undefined })`, `PptMappingSection({ slotId })`, `V1ExtraIndicatorsSection({ slotId })`, `ComparisonTable({ rows: ComparisonRow[] })`. 호출부 JSX(`<SignificanceSection slotId={slot?.id} />` 등) 변경 금지.
  - 묶음(ProductMetricsSection)으로 갈 경우에도 **렌더 결과 DOM(태그·className·순서)** 이 현재와 동일해야 한다.
- **함수 signature 보존**: `buildLineExperimentCards(detail: AiRunDetail): GoalCardData[]` 등 빌더 시그니처 그대로. 인자/반환 타입 변경 금지.
- **타입 이동은 호출자 영향 분석 후**: `GoalCardData`/`GoalCardProps`/`ComparisonRow`를 lib로 옮길 때, 이들을 참조하는 모든 위치(본 지시서 [진단]의 사용처 목록 + `Grep` 재확인)가 새 import를 가리키게 한다. 만약 어떤 컴포넌트(예: `GoalCard`/`GoalCardGrid`/`StoredEvaluationSection`)가 같은 타입을 쓰는데 본체에 남는다면, 본체도 `cardTypes.ts`에서 import하도록 교체(타입 정의 2벌 금지). 분석 결과 호출자가 예상보다 광범위(예: 다른 View가 같은 타입 import)하면 **차단 보고**.

## 성공 기준 (측정 가능)

| 항목 | 시작값 | 목표 | 허용 |
|---|---|---|---|
| `TrainingView.tsx` 줄 수 | 1560 | ~900 | ≤ 950 |
| 신규 lib/component 파일 | 0 | `experimentBuilder.ts` + `cardTypes.ts` + `SignificanceSection.tsx` + `ProductMetricsSection.tsx` + `ComparisonTable.tsx` (필요 시 `metricAccess.ts`/`comparisonTypes.ts`) | — |
| `npx tsc --noEmit` | EXIT 0 | EXIT 0 (신규 에러 0) | 0 |
| 수동 시각 회귀(통계 검정/목표 대비 평가/초기 계획 평가/비교 표) | baseline | diff 없음(픽셀/텍스트 동일) | 폰트 렌더링 anti-alias 수준의 비결정성만 |
| 동작 변경(네트워크 호출·상태·분기) | — | 0 | 0 |
| 예상 시간 | — | 2~3시간 | — |

(테스트 통과 수·snapshot diff 행은 Vitest/스냅샷 부재로 **해당 없음 → 생략**. 회귀 검증은 위 시각 항목으로 대체.)

## 검증

각 Step 끝 + 최종. PowerShell 기준.

**1) typecheck (모든 Step 필수, 기준 명령)**
```powershell
cd C:\Users\user\lens\frontend
npx tsc --noEmit
# 기대: 출력 없음 + 종료코드 0. ($LASTEXITCODE -eq 0)
```
신규 에러가 1개라도 나오면 그 Step에서 멈추고 원인 수정(또는 차단 보고). 시작 baseline이 0이므로 새로 생긴 에러는 100% 이번 변경 탓.

**2) 줄 수 확인 (Step2·6)**
```powershell
(Get-Content C:\Users\user\lens\frontend\src\components\TrainingView.tsx | Measure-Object -Line).Lines
# 기대: 최종 ~900 (≤950)
```

**3) 빌드 sanity (최종, 선택)**
```powershell
cd C:\Users\user\lens\frontend
npm run build
# 기대: 컴파일 성공. (tsc가 이미 통과했으면 대개 OK. 실패 시 차단 보고.)
```

**4) 수동 시각 비교 (Step3·5·6 필수) — 자동 스크린샷 diff 도구 없음, 사람/preview 눈 비교**
```powershell
# dev server 이미 떠 있으면 재사용. 없으면:
powershell -ExecutionPolicy Bypass -File C:\Users\user\lens\scripts\start_demo.ps1
# 프론트: http://127.0.0.1:3000  (AI 모델 탭 = TrainingView)
```
- 브라우저(또는 preview MCP)로 `http://127.0.0.1:3000` 열고 "AI 모델" 화면 진입.
- **CP230 기준 스크린샷이 있으면** 동일 슬롯·동일 뷰포트로 캡처해 1:1 비교. 없으면 **분리 직전 commit 상태에서 먼저 baseline 캡처 후 분리 적용본과 비교**(같은 세션 내 before/after).
- 확인 슬롯/영역:
  - `line-1d`: "목표 대비 평가" 카드 그리드(빌더 출력) / "초기 계획과 차이점" / "초기 계획 평가" / **"통계 검정"** 섹션 전체(헤드라인·8셀·details 펼침·GW·footnote).
  - `band-1d`, `band-1w`: 같은 4영역 + band-1w의 GW regime sub-table.
  - 실험 상세 1개 진입(이전 실험 → 항목 선택): "비교 지표" 표(ComparisonTable) 렌더.
- **기대**: 텍스트·표 구조·className 기반 스타일·펼침 동작 모두 동일. 차이 발견 시 [차단 트리거].

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **"통계 검정"(SignificanceSection) 영역 시각 diff** — 헤드라인/표 8셀/`<details>` 펼침/GW regime/footnote(`*`,`†`)/GW interpretation 중 하나라도 달라지면 CP216.2 산출물 회귀. 멈춰라.
2. **"목표 대비 평가"(StoredEvaluationSection, 빌더 출력) 카드 내용·개수·tone 변화** — 빌더(`buildLine/BandExperimentCards`) 이전 중 로직이 미세 변형됐다는 신호. 멈춰라.
3. **"초기 계획 평가"/"초기 계획과 차이점"(PptMapping/V1Extra) 영역 diff** — 멈춰라.
4. **"비교 표"(ComparisonTable) 행·열·해석 텍스트 변화** — 멈춰라.
5. **`npx tsc --noEmit`에 신규 에러** 가 한 Step에서 발생하고 즉시 원인이 명확히 잡히지 않으면 — 추측 수정 말고 멈춰서 보고.
6. **타입 이동 영향이 TrainingView 밖으로 번짐**(예: `GoalCardData`/`ComparisonRow`를 다른 View/컴포넌트가 import 중이라 광범위 수정 필요) — 멈춰서 영향 범위 정리 후 보고.
7. **className 문자열을 바꿔야만 tsc/렌더가 통과**하는 상황 — 그건 동작/스타일 변경이다. 멈춰라.
8. **CP230 기준(시각 baseline)이 없거나 비교 불가** — 분리 시작 전이면 시작하지 말고 보고. (선행 의존 미충족.)
9. **dev server가 환경변수/포트 문제로 안 뜸**(예: `NEXT_PUBLIC_BACKEND_URL` 누락, 3000 점유 충돌) — 임의로 포트/env 바꾸지 말고 상황 보고.
10. **`getExperimentFailureReason` 등 dead code를 "정리하려고" 삭제하고 싶을 때** — 이 CP 범위 아님. 삭제 말고 이전만. (삭제는 후속 CP 보고로.)

## ADR

완료 후 `docs/adr/0022-trainingview-split.md` 작성(200~300단어). `docs/adr/` 디렉토리가 없으면 생성.
기록할 결정: **TrainingView.tsx 1560→~900 분리에서 공유 카드 타입(`GoalCardData`/`GoalCardProps`)을 `lib/training/cardTypes.ts`로 분리한 결정**(왜 `staticEvaluation.StaticGoalCard`와 합치지 않고 별도 모듈로 뒀는지), 그리고 **정적 섹션은 기존 `slotId`-accessor 패턴(UsageData/Reproducibility와 동일)을 따랐고, 순수 빌더는 React 비의존이라 lib로 우선 추출**한 추출 순서 원칙, **Vitest/스냅샷 부재로 회귀 검증을 tsc + 수동 시각 비교로 대체**한 선택과 그 한계(후속 CP에서 시각 회귀 자동화 필요)를 남긴다.

## 자가 점검 결과 양식

완료 보고 시 아래를 채운다.

- **[Plan v3 정합]**: PASS / WARN / FAIL — 사유: ______ (구조 분리만, 밴드 본체·fidelity·cost 로직 무관이면 PASS 예상)
- **[구조 결함]**: PASS / WARN / FAIL — 사유: ______ (타입 중복 정의·import 순환·dead code 처리 등)
- **[모델 영향]**: PASS / WARN / FAIL — 사유: ______ (학습/calibration/지표 산출 코드 미변경이면 PASS)

## 산출물

- **변경/신규 파일 목록**:
  - 신규: `frontend/src/lib/training/experimentBuilder.ts`, `frontend/src/lib/training/cardTypes.ts`, `frontend/src/components/training/SignificanceSection.tsx`, `frontend/src/components/training/ProductMetricsSection.tsx`, `frontend/src/components/training/ComparisonTable.tsx` (필요 시 `metricAccess.ts` / `comparisonTypes.ts`).
  - 수정: `frontend/src/components/TrainingView.tsx`(줄 수 감소 + import 교체).
  - 신규: `docs/adr/0022-trainingview-split.md`.
- **리포트**: `docs/cp232_report.md` — 요구 / 한 일(Step별 commit 해시) / 결정(타입 분리·묶음 vs named export 선택·Vitest 부재 대응) / 후속(`getExperimentFailureReason` dead code 삭제 후보, Reproducibility/UsageData orphan 렌더 여부 확인 필요, 시각 회귀 자동화). 필요한 만큼만, 간결히.
