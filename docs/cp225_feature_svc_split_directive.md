# CP225 feature_svc.py 모듈 분리 (Directive)

> 이 문서는 단독 실행 가능한 지시서다. 새 Claude Code 세션이 런북(`docs/cp221_237_refactoring_runbook.md`)에서 이 CP를 꺼내 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지. 막히면 멈추고 보고.

---

## 역할 고정
- **모드**: `code` (구현 + 자가 점검만 보고. 기획/설계 토론 아님)
- **권한**: 코드 수정, 로컬 검증(pytest / ruff / mypy / snapshot 재실행)만.
- **금지**:
  - 새 모델 학습 금지.
  - 새 calibration 금지.
  - DB write 금지 (Postgres / parquet 스냅샷 쓰기 금지).
  - Supabase 호출 금지.
  - 사용자가 직접 수정한 파일 revert 금지.
  - **계산 로직 변경 금지.** 이 CP는 순수 구조 이동(코드 옮기기)이다. 숫자가 바뀌면 실패다.
- **자가 점검**: 종료 시 [Plan v3 정합] / [구조 결함] / [모델 영향] 3축 PASS·WARN·FAIL + 사유 보고 (아래 양식).
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경
- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). 활성화: `.\.venv\Scripts\Activate.ps1`
- **백엔드 기동**(검증 중 import sanity 확인용, 필요 시): `scripts\start_demo.ps1` 또는 `uvicorn backend.app.main:app --port <빈포트>`
- **프론트**(이 CP에서는 불필요): `npm run dev`
- **포트 충돌 주의**: 검증용으로 서버를 띄울 일이 생기면 이미 점유된 포트(데모가 쓰는 8000 등)를 피해 임시 포트를 쓴다. 이 CP는 순수 라이브러리 분리라 서버 기동 없이 pytest로 충분하다.
- 이 모듈은 순수 파이썬 라이브러리다. **DB·네트워크·GPU 없이** 단위 테스트만으로 검증된다.

---

## 진단 (근거)
**대상 파일**: `backend/app/services/feature_svc.py` — **현재 591줄** (확인: `git`/`wc -l` 591). 한 파일에 서로 다른 책임 4종이 섞여 SRP를 위반한다. 아래는 실제 코드 인용(줄번호 기준).

1. **피처 정의 상수 (definition)** — `feature_svc.py:15-113`
   - `_BASE_FEATURE_COLUMNS`(15), `_REGIME_FEATURE_COLUMNS`(34), `_FUNDAMENTAL_FEATURE_COLUMNS`(39), `_MACRO_FEATURE_COLUMNS`(47), `_BREADTH_FEATURE_COLUMNS`(53), 플래그 컬럼명(57-59), `REQUIRED_FEATURE_COLUMNS`(60), `FEATURE_COLUMNS`(67), `SUPPORTED_TIMEFRAMES`(76), `_EPSILON`(77), `_CONTEXT_COLUMNS`(78), `_REGIME_COLUMNS`(86), `_FUNDAMENTAL_SOURCE_COLUMNS`(87), `_INDICATOR_ONLY_COLUMNS`(95), `_OUTPUT_COLUMNS`(96), `PRICE_DERIVED_FEATURE_COLUMNS`(97), `_ADJUSTED_OHLC_COLUMNS`(110), `_RATIO_SANITY_COLUMNS`(111), `_MAX_RATIO_ABS_LIMIT`(112), `_P99_RATIO_ABS_LIMIT`(113).

2. **OHLC / 비율 검증 (validators)** — `feature_svc.py:139-205`
   - `_validate_adjusted_ohlc_contract`(139), `_apply_adjusted_ohlc_contract`(161), `_validate_ratio_feature_sanity`(181). high/low/open/close 정합성과 비율 분포 sanity(P99·max 한도)를 강제.

3. **리샘플링 / 타임프레임 변환 (resampling)** — `feature_svc.py:116-313` (정의·검증과 일부 인접)
   - `normalize_timeframe`(116), `default_horizon_for_timeframe`(125), `_ensure_datetime`(130), `latest_complete_period_end`(208), `drop_incomplete_resampled_periods`(224), `_resample_single_ticker`(245), `resample_price_frame`(277), `_resample_context_frame`(299).

4. **지표 계산 (calculator)** — `feature_svc.py:316-572`
   - `_compute_rsi`(316), `_compute_features_for_single_ticker`(324, RSI/MACD/볼린저/ATR/MA비율 계산), `build_price_features`(375), `_apply_regime_columns`(400), `_apply_context_flags`(423), `_apply_fundamental_features`(439), `build_features`(506), `build_latest_feature_rows`(575).

**문제의 본질**: 정의·계산·리샘플·검증이 한 파일에 묶여 있어, 계산식 한 줄 고치려 해도 591줄을 다 봐야 하고, 변경 영향 범위가 불투명하다. Plan v3는 "fidelity 우선"이라 계산 동작이 단 1비트도 바뀌면 안 되는데, 현재 구조는 그 보장을 어렵게 한다.

**조사 출처**: `backend/app/services/feature_svc.py` 전체 Read(591줄). caller는 repo 전역 `Grep "feature_svc"` (`*.py`) 결과. 마스터플랜 `docs/refactoring_master_plan.md:61, 92, 164`("D.4 / CP-RF-3 = feature_svc 591 → definition/calculator/resampling/validators 4모듈, **단 characterization test 먼저**").

**caller 인벤토리(중요, 실제 Grep 결과)** — import 경로/심볼이 깨지면 안 되는 대상:
- **공개 심볼 사용** (정상):
  - `backend/app/services/api_service.py:14` → `drop_incomplete_resampled_periods`
  - `backend/tests/test_feature_svc.py:11` → `FEATURE_COLUMNS, build_features, resample_price_frame`
  - `backend/collector/pipelines/bootstrap_snapshot.py:15` → `FEATURE_COLUMNS, build_features, normalize_timeframe`
  - `backend/collector/jobs/compute_indicators.py:9` → `build_features`
  - `backend/db/scripts/sync_source_to_lens.py:27` → `build_features`
  - `ai/preprocessing.py:27` → `FEATURE_COLUMNS as SOURCE_FEATURE_COLUMNS, PRICE_DERIVED_FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, build_price_features, normalize_timeframe, resample_price_frame`
  - `ai/cp202_1_band_baseline_comparison.py:13` → `resample_price_frame`
  - `scripts/*` 다수 (`cp98/cp100/cp101/cp115/cp133/cp134/cp142/cp145/cp151/cp87`) → `build_features, FEATURE_COLUMNS, resample_price_frame, latest_complete_period_end, normalize_timeframe` 조합.
- **⚠ 사설(언더스코어) 심볼 사용** (분리 시 반드시 re-export 또는 동시 수정 대상):
  - `scripts/diagnostics/data_length_audit.py:17-28` → `REQUIRED_FEATURE_COLUMNS, _BASE_FEATURE_COLUMNS, _CONTEXT_COLUMNS, _REGIME_FEATURE_COLUMNS, _apply_fundamental_features, _apply_regime_columns, _compute_features_for_single_ticker, _resample_context_frame, normalize_timeframe, resample_price_frame`
  - `scripts/diagnostics/base_feature_nan_tracer.py:13-19` → `_BASE_FEATURE_COLUMNS, _CONTEXT_COLUMNS, _compute_features_for_single_ticker, _resample_context_frame, resample_price_frame`

> 결론: `feature_svc`는 **공개 심볼 + 일부 사설 심볼까지** 외부에서 import한다. 따라서 분리 후 `feature_svc`는 **이 모든 심볼(언더스코어 포함)을 새 모듈에서 re-export**해야 import가 깨지지 않는다.

---

## 선행 의존
- **CP223 (백엔드 characterization 스냅샷 / 마스터플랜의 CP-RF-2) 그린 필수.**
- CP223 스냅샷 테스트가 존재하고 통과(green) 상태가 아니면 **이 CP를 시작하지 않는다.** 계산 동작이 바뀌었는지 비교할 기준선이 없는 상태에서 구조를 옮기면 회귀를 감지할 수 없다 (Fowler/Feathers 원칙: 동작 박제 후 리팩토링).
- **확인 절차**: 시작 전 아래를 실행해 그린을 확인. 스냅샷 테스트 파일이 없거나 빨간색이면 **즉시 중단하고 "CP223 미충족"으로 보고**.
  ```powershell
  .\.venv\Scripts\Activate.ps1
  # CP223이 만든 스냅샷 테스트 식별 (이름은 CP223 산출물 기준; 보통 backend/tests/ 하위 snapshot/characterization)
  python -m pytest backend/tests -k "snapshot or characterization or golden" -q
  python -m pytest backend/tests/test_feature_svc.py -q
  ```
  - 두 명령 모두 **0 failed** 여야 진행. 스냅샷 테스트가 `0 collected`로 잡히지 않으면 CP223 산출물이 없는 것 → 중단·보고.

---

## 범위
**포함**
- `backend/app/services/feature_svc.py`(591줄)를 책임별 4개 신규 모듈로 분리:
  - `backend/app/services/feature_definition.py` — 상수/컬럼 정의 (`FEATURE_COLUMNS` 등).
  - `backend/app/services/feature_calculator.py` — RSI·MACD·볼린저·ATR·MA비율·펀더멘털/레짐/컨텍스트 적용 + `build_features` 류 오케스트레이션.
  - `backend/app/services/resampling.py` — 타임프레임 정규화·리샘플링·기간 컷오프.
  - `backend/app/services/validators.py` — OHLC 계약·비율 sanity 검증.
- 기존 `feature_svc.py`는 **얇은 re-export 모듈**로 남겨 모든 기존 import 경로/심볼(언더스코어 포함)을 보존.
- caller 중 **사설 심볼을 직접 import하는 diagnostics 2개**는 re-export 보존으로 무수정 유지 (경로 변경 없음).
- 각 신규 모듈 목표 **200줄 이내**.
- ADR 1장(`docs/adr/0015-feature-svc-module-split.md`) 작성.
- 산출 리포트 `docs/cp225_report.md`.

**제외**
- 계산식/로직 변경 (숫자·동작 변경 일절 금지).
- 공개 API signature 변경, 함수 이름 변경, 인자 순서 변경.
- Supabase 관련 일체 (보류).
- 사용자가 직접 수정한 파일의 임의 revert.
- caller들의 import 경로를 새 모듈로 강제 이전하는 작업 — **하지 않는다.** `feature_svc` re-export로 호환을 유지하는 것이 이 CP의 안전 전략이다. (caller를 새 경로로 옮기는 것은 별도 후속 CP. 단, **본 모듈 내부 caller**(아래 Step 정의)는 새 모듈을 직접 참조한다.)
- DB write / 학습 / calibration.

---

## Sub-step (Strangler Fig, 작은 단위)
> 원칙: **옛 코드 옆에 새 코드 공존 → 호출자 이전 → 옛 코드 제거**. 한 Step = 한 revert 단위 = 한 커밋. 추출 순서는 **순수 함수(부작용 없는 상수/계산) → I/O·검증 경계 → 상태 의존 오케스트레이션** 역순 의존을 고려해 **definition → validators → resampling → calculator** 순으로 한다 (definition은 무의존, validators는 definition만, resampling은 definition+validators, calculator는 셋 다 의존).
>
> 각 Step 종료 시 **반드시**: ① `feature_svc.py`가 해당 심볼을 새 모듈에서 re-export 하는지 확인 ② 아래 검증 블록 전체 통과 ③ 커밋.
>
> **공통 검증 블록 (매 Step 끝에 실행, 하나라도 빨강이면 그 Step 중단·보고)**:
> ```powershell
> .\.venv\Scripts\Activate.ps1
> ruff check backend/app/services/feature_svc.py backend/app/services/feature_definition.py backend/app/services/feature_calculator.py backend/app/services/resampling.py backend/app/services/validators.py
> mypy backend/app/services/feature_svc.py backend/app/services/feature_definition.py backend/app/services/feature_calculator.py backend/app/services/resampling.py backend/app/services/validators.py
> python -m pytest backend/tests/test_feature_svc.py -q
> python -m pytest backend/tests -k "snapshot or characterization or golden" -q   # CP223 스냅샷
> # import sanity: 모든 기존 import 경로/심볼이 살아있는지
> python -c "from backend.app.services.feature_svc import FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, PRICE_DERIVED_FEATURE_COLUMNS, build_features, build_price_features, build_latest_feature_rows, resample_price_frame, normalize_timeframe, default_horizon_for_timeframe, latest_complete_period_end, drop_incomplete_resampled_periods, _BASE_FEATURE_COLUMNS, _CONTEXT_COLUMNS, _REGIME_FEATURE_COLUMNS, _compute_features_for_single_ticker, _resample_context_frame, _apply_fundamental_features, _apply_regime_columns; print('import-ok')"
> python -c "import scripts.diagnostics.data_length_audit; import scripts.diagnostics.base_feature_nan_tracer; print('diag-import-ok')"
> ```
> (ruff/mypy 호출 형식이 repo 설정과 다르면 repo의 기존 lint 명령을 따른다. 핵심은 "신규 4모듈 + feature_svc 모두 통과".)

### Step 0 — 선행 확인 (커밋 없음)
- 위 **선행 의존** 블록 실행. CP223 스냅샷 + `test_feature_svc.py` 그린 확인.
- 기준선 줄 수 기록: `feature_svc.py` 591줄. 빨강이면 중단·보고.

### Step 1 — `feature_definition.py` 추출 (정의 상수, 무의존)
- 신규 `backend/app/services/feature_definition.py` 생성. **`feature_svc.py:15-113`의 모든 상수**를 그대로 이동 (`_BASE_FEATURE_COLUMNS`~`_P99_RATIO_ABS_LIMIT`, `FEATURE_COLUMNS`, `REQUIRED_FEATURE_COLUMNS`, `PRICE_DERIVED_FEATURE_COLUMNS`, `SUPPORTED_TIMEFRAMES`, `_EPSILON`, `_CONTEXT_COLUMNS`, `_REGIME_COLUMNS`, `_FUNDAMENTAL_SOURCE_COLUMNS`, `_INDICATOR_ONLY_COLUMNS`, `_OUTPUT_COLUMNS`, `_ADJUSTED_OHLC_COLUMNS`, `_RATIO_SANITY_COLUMNS`, `_MAX_RATIO_ABS_LIMIT`, `_P99_RATIO_ABS_LIMIT`). 값/순서 1바이트도 바꾸지 말 것 (`_OUTPUT_COLUMNS`는 `FEATURE_COLUMNS`·`_INDICATOR_ONLY_COLUMNS`에 의존하므로 정의 순서 유지).
- `feature_svc.py`에서 해당 상수 정의를 제거하고 최상단에 `from backend.app.services.feature_definition import *` 대신 **명시적 re-export**:
  `from backend.app.services.feature_definition import (FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, PRICE_DERIVED_FEATURE_COLUMNS, SUPPORTED_TIMEFRAMES, _EPSILON, _CONTEXT_COLUMNS, _REGIME_COLUMNS, _FUNDAMENTAL_SOURCE_COLUMNS, _INDICATOR_ONLY_COLUMNS, _OUTPUT_COLUMNS, _ADJUSTED_OHLC_COLUMNS, _RATIO_SANITY_COLUMNS, _MAX_RATIO_ABS_LIMIT, _P99_RATIO_ABS_LIMIT, _BASE_FEATURE_COLUMNS, _REGIME_FEATURE_COLUMNS, _FUNDAMENTAL_FEATURE_COLUMNS, _MACRO_FEATURE_COLUMNS, _BREADTH_FEATURE_COLUMNS, _FUNDAMENTAL_FLAG_COLUMN, _MACRO_FLAG_COLUMN, _BREADTH_FLAG_COLUMN)`
  (ruff가 미사용 import F401을 잡으면, re-export 의도이므로 모듈에 `__all__` 추가하거나 해당 import에 `# noqa: F401`. `import *` 금지 — 명시 유지.)
- `feature_svc.py` 내부 함수들은 계속 이 상수 이름을 참조 → 위 import로 해결.
- **공통 검증 블록** 실행 → 그린 → 커밋: `refactor(feature): extract feature_definition constants`.

### Step 2 — `validators.py` 추출 (검증, definition만 의존)
- 신규 `backend/app/services/validators.py` 생성. **`feature_svc.py:139-205`의 3함수** 이동: `_validate_adjusted_ohlc_contract`, `_apply_adjusted_ohlc_contract`, `_validate_ratio_feature_sanity`. 이 함수들이 쓰는 상수(`_ADJUSTED_OHLC_COLUMNS`, `_EPSILON`, `_RATIO_SANITY_COLUMNS`, `_P99_RATIO_ABS_LIMIT`, `_MAX_RATIO_ABS_LIMIT`)는 `feature_definition`에서 import. `numpy`, `pandas` import 추가.
- `feature_svc.py`에서 세 함수 정의 제거 + `from backend.app.services.validators import _validate_adjusted_ohlc_contract, _apply_adjusted_ohlc_contract, _validate_ratio_feature_sanity` 로 re-export. (resampling·calculator 함수가 `_apply_adjusted_ohlc_contract` 등을 호출하므로 이름 보존 필수.)
- **공통 검증 블록** → 그린 → 커밋: `refactor(feature): extract validators`.

### Step 3 — `resampling.py` 추출 (리샘플, definition+validators 의존)
- 신규 `backend/app/services/resampling.py` 생성. **이동 함수**: `normalize_timeframe`(116), `default_horizon_for_timeframe`(125), `_ensure_datetime`(130), `latest_complete_period_end`(208), `drop_incomplete_resampled_periods`(224), `_resample_single_ticker`(245), `resample_price_frame`(277), `_resample_context_frame`(299).
  - 의존: `SUPPORTED_TIMEFRAMES`, `_EPSILON` 등은 `feature_definition`에서, `_apply_adjusted_ohlc_contract`·`_validate_adjusted_ohlc_contract`는 `validators`에서 import.
  - `_resample_single_ticker`(245)는 `_apply_adjusted_ohlc_contract`·`_ensure_datetime`·`drop_incomplete_resampled_periods`·`_validate_adjusted_ohlc_contract`를 호출 → 모두 이 모듈 내부 또는 validators import로 해결.
- `feature_svc.py`에서 8함수 제거 + re-export: `from backend.app.services.resampling import normalize_timeframe, default_horizon_for_timeframe, _ensure_datetime, latest_complete_period_end, drop_incomplete_resampled_periods, _resample_single_ticker, resample_price_frame, _resample_context_frame`.
- ⚠ `api_service.py:14`가 `drop_incomplete_resampled_periods`를 `feature_svc`에서 import → re-export로 무수정 유지됨(검증 블록의 import sanity로 확인).
- **공통 검증 블록** → 그린 → 커밋: `refactor(feature): extract resampling`.

### Step 4 — `feature_calculator.py` 추출 (계산·오케스트레이션, 셋 다 의존)
- 신규 `backend/app/services/feature_calculator.py` 생성. **이동 함수**: `_compute_rsi`(316), `_compute_features_for_single_ticker`(324), `build_price_features`(375), `_apply_regime_columns`(400), `_apply_context_flags`(423), `_apply_fundamental_features`(439), `build_features`(506), `build_latest_feature_rows`(575).
  - 의존: `feature_definition`(상수 전부), `validators`(`_apply_adjusted_ohlc_contract`, `_validate_ratio_feature_sanity`), `resampling`(`normalize_timeframe`, `resample_price_frame`, `_resample_context_frame`)에서 import.
- 이 시점에서 `feature_svc.py`에는 자체 정의 함수가 사실상 남지 않는다.
- **공통 검증 블록** → 그린 → 커밋: `refactor(feature): extract feature_calculator`.

### Step 5 — `feature_svc.py` 얇은 re-export로 정리
- `feature_svc.py`를 **순수 re-export 파사드**로 정리. 모듈 docstring(현재 1-6줄)은 "호환 유지용 re-export. 실제 구현은 feature_definition / feature_calculator / resampling / validators 참조"로 갱신. `from __future__ import annotations` 유지.
- **모든 공개 + 사설 심볼**을 4모듈에서 명시 import 후 `__all__`로 노출. 최소 목록(검증 블록의 import sanity가 요구하는 전체):
  - definition: `FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, PRICE_DERIVED_FEATURE_COLUMNS, SUPPORTED_TIMEFRAMES, _BASE_FEATURE_COLUMNS, _REGIME_FEATURE_COLUMNS, _CONTEXT_COLUMNS, _EPSILON` (+ 나머지 상수)
  - resampling: `normalize_timeframe, default_horizon_for_timeframe, latest_complete_period_end, drop_incomplete_resampled_periods, resample_price_frame, _resample_context_frame`
  - calculator: `build_features, build_price_features, build_latest_feature_rows, _compute_features_for_single_ticker, _apply_fundamental_features, _apply_regime_columns`
  - validators: (필요 시) `_apply_adjusted_ohlc_contract` 등.
- `numpy`/`pandas` 직접 import는 제거 가능(re-export만 남으면 불필요). ruff 미사용 경고 정리.
- **공통 검증 블록** → 그린 → 커밋: `refactor(feature): reduce feature_svc to re-export facade`.

> Step 1~5는 각각 독립 revert 가능. 중간 Step에서 막히면 그 Step 커밋만 되돌리면 직전 그린 상태로 복귀.

---

## 인터페이스 보존
- 다음 **공개 심볼**의 import 경로(`backend.app.services.feature_svc.X`)와 signature를 **그대로 유지**한다:
  - 함수: `build_features(price_df, macro_df=None, breadth_df=None, fundamentals_df=None, timeframe="1D")`, `build_price_features(price_df, timeframe="1D")`, `build_latest_feature_rows(...)`, `resample_price_frame(price_df, timeframe)`, `normalize_timeframe(timeframe)`, `default_horizon_for_timeframe(timeframe)`, `latest_complete_period_end(latest_daily_date, timeframe)`, `drop_incomplete_resampled_periods(frame, timeframe, *, latest_daily_date=None)`.
  - 상수: `FEATURE_COLUMNS`, `REQUIRED_FEATURE_COLUMNS`, `PRICE_DERIVED_FEATURE_COLUMNS`, `SUPPORTED_TIMEFRAMES`.
- 추가로 **diagnostics가 import하는 사설 심볼**도 `feature_svc`에서 보존: `_BASE_FEATURE_COLUMNS`, `_CONTEXT_COLUMNS`, `_REGIME_FEATURE_COLUMNS`, `_compute_features_for_single_ticker`, `_resample_context_frame`, `_apply_fundamental_features`, `_apply_regime_columns`.
- **반환 schema 불변**: `build_features` 출력 컬럼 = `_OUTPUT_COLUMNS`(`["ticker","date","timeframe","regime_label", *FEATURE_COLUMNS, "atr_ratio"]`) 그대로. `build_price_features` 출력 = `["ticker","date","timeframe", *PRICE_DERIVED_FEATURE_COLUMNS]` 그대로.
- 만약 어떤 이유로든 signature/경로/schema를 바꿔야 하는 상황이 발생하면(예: 순환 import 회피 불가): **그냥 바꾸지 말고**, 영향받는 caller 전체(위 인벤토리)를 나열한 영향 분석을 작성하고 **차단 보고** 후 사용자 판단을 기다린다.

---

## 성공 기준 (측정 가능)
| 항목 | 시작 | 목표 |
|---|---|---|
| `feature_svc.py` 줄 수 | 591 | re-export 파사드 (≤ 80줄 권장) |
| `feature_definition.py` | (신규) | ≤ 200줄 |
| `feature_calculator.py` | (신규) | ≤ 200줄 |
| `resampling.py` | (신규) | ≤ 200줄 |
| `validators.py` | (신규) | ≤ 200줄 |
| `pytest backend/tests/test_feature_svc.py` | green | green (회귀 0) |
| CP223 snapshot 테스트 | green | green, **diff 0** |
| import sanity (공개+사설 심볼) | ok | ok (깨짐 0) |
| diagnostics 2개 import | ok | ok |
| mypy 신규 error | — | 0 추가 |
| ruff | clean | clean (신규 위반 0) |
| 예상 시간 | — | 2~3시간 |

> 참고: calculator는 오케스트레이션 함수가 많아 200줄을 초과할 수 있다. 초과 시 강제로 한 모듈에 욱여넣지 말고, `build_features` 류(오케스트레이션)와 순수 지표(`_compute_rsi`, `_compute_features_for_single_ticker`)를 분리하는 안을 **차단 보고**로 제안한 뒤 진행 여부를 받는다. (이 CP 범위 내 임의 추가 분리 금지.)

---

## 검증
**매 Step 끝 + 최종**, 위 "공통 검증 블록"을 실행. 기대 결과:
- `ruff check ...` → `All checks passed!` (또는 신규 위반 0).
- `mypy ...` → 신규 error 0 (기존 baseline 대비 증가 없음).
- `python -m pytest backend/tests/test_feature_svc.py -q` → `passed`, failed 0.
- `python -m pytest backend/tests -k "snapshot or characterization or golden" -q` → `passed`, failed 0. **diff 0** (스냅샷 mismatch 출력 없음).
- import sanity `python -c "..."` → `import-ok` 출력.
- diagnostics import → `diag-import-ok` 출력.
- 줄 수 확인:
  ```powershell
  Get-Content backend/app/services/feature_svc.py | Measure-Object -Line
  Get-Content backend/app/services/feature_definition.py | Measure-Object -Line
  Get-Content backend/app/services/feature_calculator.py | Measure-Object -Line
  Get-Content backend/app/services/resampling.py | Measure-Object -Line
  Get-Content backend/app/services/validators.py | Measure-Object -Line
  ```
- (선택) 전체 회귀: `python -m pytest backend/tests -q` → 기존 통과 수 유지, 신규 실패 0.

---

## 차단 트리거 (중요)
다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**
1. **CP223 스냅샷 테스트가 빨강이거나 존재하지 않음** (Step 0). → 안전망 없음. 시작 금지, 보고.
2. **어느 Step에서든 CP223 snapshot diff 발생.** = 계산/리샘플/검증 동작이 바뀐 것. 구조만 옮겼는데 숫자가 바뀌면 **이동 중 로직이 훼손된 것**이므로 그 Step 커밋 직전 상태로 멈추고 diff 내용과 함께 보고.
3. **`test_feature_svc.py` 기존 테스트가 하나라도 실패.** → 인터페이스/동작 깨짐. 중단·보고.
4. **import sanity 실패** (공개 또는 사설 심볼이 `feature_svc`에서 사라짐) 또는 **diagnostics 2개 import 실패.** → caller import 깨짐. 중단·보고.
5. **순환 import 발생** (예: definition↔calculator). → 분리 경계 설계 문제. 임의 우회 코드 넣지 말고 멈추고 경계 재설계안 보고.
6. **mypy 신규 error 증가** 또는 **ruff 신규 위반** 으로 정리가 안 됨. → 무리한 `# noqa` 남발 금지, 보고.
7. **목표 200줄을 calculator가 크게 초과**해 추가 분리가 필요. → 범위 밖. 분리안 제안하고 보고(임의 진행 금지).
8. **사용자가 직접 수정한 흔적이 있는 파일을 건드려야 함** (revert 위험). → 멈추고 확인.

---

## ADR
- 완료 후 `docs/adr/0015-feature-svc-module-split.md` 1장(200~300단어) 작성. (`docs/adr/` 디렉토리가 없으므로 생성한다.)
- 기록할 것: **분리 경계 기준** — 왜 definition/calculator/resampling/validators 4축인가(책임=정의·계산·시간변환·검증), 의존 방향(definition←validators←resampling←calculator, 단방향), `feature_svc`를 re-export 파사드로 남긴 이유(공개+사설 심볼을 외부 caller·diagnostics가 직접 import하므로 호환 유지), 대안(파일 그대로 두기 / caller 전부 새 경로로 이전)을 버린 이유, fidelity 보장 방법(CP223 스냅샷 diff 0). 형식은 표준 ADR(Context / Decision / Consequences).

---

## 자가 점검 결과 양식
종료 보고에 아래를 채운다.
- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: (계산 동작 불변·fidelity 우선 위반 여부, 밴드 파이프라인 입력 피처 영향)
- **[구조 결함]** PASS / WARN / FAIL — 사유: (SRP 개선 여부, 순환 import 유무, 의존 단방향성, 모듈 줄 수 목표 달성)
- **[모델 영향]** PASS / WARN / FAIL — 사유: (build_features/build_price_features 출력 schema·값 불변, 학습/추론 파이프라인 무영향)

---

## 산출물
- 변경/신규 파일:
  - 신규: `backend/app/services/feature_definition.py`, `backend/app/services/feature_calculator.py`, `backend/app/services/resampling.py`, `backend/app/services/validators.py`
  - 변경: `backend/app/services/feature_svc.py` (re-export 파사드)
  - 신규: `docs/adr/0015-feature-svc-module-split.md`
- 리포트: `docs/cp225_report.md` (요구 / 한 일 / 결정 / 후속 — 필요한 만큼만). 후속에는 "caller들을 새 모듈 경로로 점진 이전(선택)" 1줄 포함.
