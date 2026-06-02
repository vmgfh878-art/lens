# CP224b BE/FE Dead Code 검출 + 안전 제거 (Directive)

> 이 문서 하나만 읽고 실행한다. 런북(`docs/cp221_237_refactoring_runbook.md`)이 CP224b 차례에 이 파일을 꺼내 실행 → 검증 → 커밋 → 다음. 차단 트리거에 하나라도 걸리면 즉시 멈추고 사용자에게 보고한다. 진행보다 정직이 우선.
>
> **이 CP의 한 줄 정의**: dead code를 *자동 도구로 후보 목록만* 뽑고, **각 후보를 grep으로 호출 0 직접 확인한 것만** 제거한다. 도구 출력 = 제안일 뿐, 근거 아님. 광범위 일괄 삭제 절대 금지.

---

## 역할 고정

- **모드**: `code` (구현 모드).
- **권한**: 코드 수정(import/dead symbol 제거에 한정), 로컬 검증(ruff·vulture·ts-prune·pytest·tsc 실행), grep/Read 조사.
- **금지**:
  - 새 학습 / 새 calibration / DB write / 운영 parquet 덮어쓰기.
  - Supabase 호출, Supabase 관련 코드(`db.py`, `market_repo.py` 등) 제거·변경 — **범위 섹션 참조, 결과에 떠도 무시**.
  - 사용자가 직접 수정한 파일을 revert.
  - 인터페이스(함수 signature / API 응답 schema / props) 변경. dead 판정이 애매하면 제거 말고 목록 보고.
  - 후보를 grep으로 확인하지 않고 도구 출력만 믿고 제거.
- **자가 점검**: 완료 시 [Plan v3 정합] / [구조 결함] / [모델 영향] 3축 PASS·WARN·FAIL + 사유 1줄. (양식 섹션 참조)
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). Python 실행은 `.venv\Scripts\python.exe`.
- **백엔드 기동(필요 시만)**: `scripts\start_demo.ps1` 또는 `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8010`. 이 CP는 코드 정적 분석 + pytest가 전부이므로 **서버 기동은 원칙적으로 불필요**.
- **프론트 기동(필요 시만)**: `cd frontend; npm run dev`. 이 CP는 `tsc --noEmit` 정적 검사면 충분, dev 서버 불필요.
- **포트 충돌 회피**: 검증용 기동이 필요하면 8000/3000 대신 비는 포트(예: 8010/3010) 사용.
- **도구 설치 현황(실측, 이 directive 작성 시점)**:
  - `ruff` — 미설치였으나 **CP222(안전망 도구)에서 설치됨이 선행 전제**. `.venv\Scripts\python.exe -m ruff --version`로 존재 확인. 없으면 CP222 미완 → 차단 보고.
  - `vulture` — **미설치**. 이 CP에서 로컬 설치 (`pip install vulture`). dev 일회성 도구이므로 **requirements 핀 금지**(핀은 CP224a 소관, 그리고 vulture는 런타임 의존 아님).
  - `ts-prune` — **미설치**, `frontend/package.json` devDeps에 없음. `npx ts-prune`로 일회성 실행(설치 영구화·package.json 수정 금지). `typescript ^5` 존재 확인됨.
- **검증 명령은 PowerShell 기준**으로 적는다(이 환경 셸 = PowerShell).

---

## 진단 (근거)

조사 출처: `docs/refactoring_master_plan.md` (§1 검증된 진단 + §2 D.1~D.5 평가), 런북 CP224b 행("vulture 오탐 사람 확인"). 단 **아래 두 가설은 master_plan이 아니라 CP 발주 스펙의 *추정*이며, 이 directive 작성 중 grep으로 검증한 결과 둘 다 거짓(=사용 중)으로 확인되었다.** 이 사실 자체가 이 CP의 존재 이유다: **추정·도구 출력보다 grep이 우선이고, 확인 없는 제거는 회귀를 만든다.**

### 가설 A (기각): `api_service.py`의 `aggregate_prices` / `resolve_price_window` 미사용 → **거짓**

`backend/app/services/api_service.py` (현재 111줄):
- `aggregate_prices` 정의 = **L25**, 호출 = **같은 파일 L83** (`get_price_response_data` 내부) + 테스트 `backend/tests/test_services.py` L24~41.
- `resolve_price_window` 정의 = **L57**, 호출 = **같은 파일 L78** (`get_price_response_data` 내부) + 테스트 `backend/tests/test_services.py` L47.

```python
# backend/app/services/api_service.py  L77-83
normalized_timeframe = normalize_display_timeframe(timeframe)
resolved_start, resolved_end = resolve_price_window(start, end)   # ← L78 사용
rows = fetch_price_rows_local(ticker, start=resolved_start, end=resolved_end)
if not rows:
    raise ResourceNotFoundError(...)
aggregated = aggregate_prices(rows, normalized_timeframe)          # ← L83 사용
```

→ **둘 다 제거 금지.** vulture가 이들을 띄우면 오탐(파일-내부 호출은 vulture가 "export 안 됨"으로 오인하지 않지만, 테스트-온리 사용으로 보일 수 있음). 무시.

### 가설 B (기각): `schemas/common.py`의 `ApiResponse` / `ErrorResponse` 미사용 → **거짓**

`backend/app/schemas/common.py` (현재 28줄):
- `ErrorResponse` = **L21**, `ApiResponse` = **L26**.
- 실제 사용처(grep `ApiResponse|ErrorResponse`, `**/*.py`): `routers/v1/ai.py` (import L18, 사용 L197·198·225·226·241·242·257·258), `routers/v1/health.py` (import L7, 사용 L15·16·30·31), `routers/v1/stocks.py` (import L8, 사용 L36·37·52·53·75·76·92·93). **4개 라우터가 response_model로 직접 사용.**

→ **둘 다 제거 금지.** 이들은 FastAPI `response_model=`/`responses={...}`에 *문자열이 아닌 심볼*로 들어가므로 vulture가 정확히 잡지만, 만약 떠도 무시.

### F401(unused import) 실측

샘플 2개 파일은 import가 전부 실사용이다(작성 중 확인): `common.py`의 `Any`=L18, `Generic`/`TypeVar`=L26 사용; `api_service.py`의 `date`/`timedelta`=L58·59, `ResourceNotFoundError`=L81 사용. → **F401은 전수 스캔으로 발견되는 것만 처리**한다(특정 파일 가정 금지). 백엔드 `backend/app` 전체 25개 `.py`(3953줄, `__pycache__` 제외)가 대상.

### 동적 참조 오탐 위험원(실측)

`market_repo.py`에 provider alias 딕셔너리(`_PROVIDER_ALIASES`, L34~) 등 **문자열 키 기반 디스패치**가 존재. 코드베이스 전반에 `getattr`/문자열 기반 참조가 있을 수 있으므로, vulture/ts-prune가 "미사용"이라 해도 **`getattr(...)`·문자열 리터럴·`__all__`·FastAPI/pydantic 데코레이터 등록**을 grep으로 함께 확인하기 전에는 제거하지 않는다.

---

## 선행 의존

- **CP223 (백엔드 characterization 스냅샷) 그린** — 필수. dead code 제거는 read-path 동작 변경 위험이 0이어야 하므로, snapshot이 박제돼 있어야 "제거 후 동작 동일"을 증명할 수 있다. CP223 미완·red면 **시작 금지, 차단 보고**.
- **CP222 (안전망 도구: ruff+pytest 설치) 그린** — 필수. ruff·pytest가 `.venv`에 있어야 한다. 없으면 차단 보고.
- 프론트는 별도 안전망(CP230 Playwright/Vitest)이 선행이지만, **이 CP의 FE 작업은 "ts-prune 결과 기록 + unused export 제거"로 동작 변경이 거의 없는 leaf 정리**다. 다만 **export 제거가 런타임 동작에 영향 가능한 경우(아래 차단 트리거)에는 CP230 그린 전까지 보류**하고 목록만 남긴다.

---

## 범위

### 포함
- 백엔드 `backend/app/**/*.py`: ruff `F401`(unused import) 자동 fix.
- 백엔드 `backend/app/**/*.py`: vulture로 dead code 후보 산출 → 표로 문서화 → grep 확인된 확정분만 제거.
- 프론트 `frontend/src/**/*.{ts,tsx}`: ts-prune로 unused export 후보 산출 → 표로 문서화 → grep 확인된 확정분만 제거.
- 후보 검증 결과(확정/보류/오탐)를 `docs/cp224b_report.md`에 표로 기록.

### 제외 (제거·변경 절대 금지)
- **Supabase 관련 전부**: `backend/app/db.py`(65줄, `get_supabase`/`supabase_is_configured`/`check_supabase_ready`/`reset_supabase_client`), `backend/app/repositories/market_repo.py`(367줄), 그리고 그 소비처(`routers/v1/admin.py` L11·163, `routers/v1/health.py` L6·34). master_plan §2 D.5 = "⏸️ 보류(동의)". **vulture/ts-prune 결과에 뜨면 무시하고 "Supabase 보류"로 표에 기록.**
- 사용자가 직접 수정한 파일의 revert.
- 큰 파일 구조 분리(그건 CP225/226/231/232 소관). 이 CP는 *제거만*.
- `backend/collector/**`(이 CP 대상 아님 — `backend/app`만). 단 import 추적 시 collector 쪽 참조는 "사용 중" 증거로 인정.
- 테스트 코드 자체의 dead code(테스트는 박제 자산이므로 건드리지 않음).
- requirements/package.json 의존성 핀·추가(vulture·ts-prune는 일회성, 영구 설치 금지).

---

## Sub-step (Strangler Fig / 작은 단위, 한 Step = 한 revert 단위)

각 Step 끝에 명시된 검증을 통과해야 다음 Step. 제거 Step은 **"옛 코드 제거 → 즉시 pytest/tsc로 호출자 부재 증명"** 패턴(추출이 아닌 제거이므로 공존 단계는 없고, 대신 *제거 전 grep 확인*이 공존 대체물이다).

### Step 1 — ruff F401 자동 fix (안전, 백엔드)
1. ruff 존재 확인: `.venv\Scripts\python.exe -m ruff --version`. 없으면 차단 보고(CP222 미완).
2. **dry-run 먼저**: `.venv\Scripts\python.exe -m ruff check backend/app --select F401`. 출력(파일:줄:심볼)을 그대로 `docs/cp224b_report.md`에 붙인다.
3. 출력 목록을 눈으로 검토 — `__init__.py`의 re-export(`__all__`에 있거나 의도적 노출)는 F401이지만 **의도적**일 수 있다. `backend/app/**/__init__.py`(대부분 0~1줄, 실측: 모두 거의 빈 파일)에 re-export가 있으면 grep으로 `__all__` 확인 후 판단. 애매하면 그 항목만 `# noqa: F401` 부여 또는 보류.
4. 자동 fix: `.venv\Scripts\python.exe -m ruff check backend/app --select F401 --fix`.
5. **검증**: `.venv\Scripts\python.exe -m ruff check backend/app --select F401` → 0 violations. `.venv\Scripts\python.exe -m pytest backend/tests -q` → 회귀 0 (CP223 snapshot 포함). CP223 snapshot diff = 0.
6. **commit**: `CP224b: remove unused imports (ruff F401 --fix)`.

### Step 2 — vulture 실행 + 후보 표 기록 (백엔드, 제거 없음)
1. 설치: `.venv\Scripts\python.exe -m pip install vulture` (requirements 수정 금지).
2. 실행(min-confidence를 명시해 오탐 줄임):
   `.venv\Scripts\python.exe -m vulture backend/app --min-confidence 60 --sort-by-size`
   (60% 미만은 노이즈가 많다. whitelist 파일은 만들지 않는다 — 대신 grep으로 개별 확인.)
3. 출력 전체를 `docs/cp224b_report.md`에 표로 기록. 컬럼: `파일:줄 | 심볼 | 종류(unused function/variable/import/attribute) | confidence | 판정(미정)`.
4. 이 Step에서는 **제거하지 않는다.** commit: `CP224b: record vulture dead-code candidates (no removal)` (docs만 변경).

### Step 3 — ts-prune 실행 + 후보 표 기록 (프론트, 제거 없음)
1. tsconfig 존재 확인: `frontend/tsconfig.json` (실측 존재).
2. 실행: `cd frontend; npx ts-prune`. (`-p tsconfig.json` 기본 사용. 설치 영구화·package.json 수정 금지.)
   - `ts-prune`는 "in module" 표시(같은 파일 내에서만 쓰임)와 `(used in module)` 주석을 구분한다. **`(used in module)`은 export만 불필요할 뿐 코드가 dead가 아니므로 별도 분류**.
3. 출력 전체를 `docs/cp224b_report.md`에 표로 기록. 컬럼: `파일:줄 | export 심볼 | ts-prune 분류(완전 unused / used-in-module) | 판정(미정)`.
4. 제거하지 않는다. commit: `CP224b: record ts-prune unused-export candidates (no removal)` (docs만 변경).

### Step 4 — 후보별 grep 검증 (동적 호출 오탐 차단)
각 후보(vulture·ts-prune)에 대해 아래를 수행하고 표의 "판정" 컬럼을 채운다:
1. **직접 호출 grep**: 심볼명을 코드베이스 전체에서 검색.
   - 백엔드: `Grep pattern="\b<symbol>\b" glob="**/*.py"` (테스트·collector 포함). import 라인 1개만 잡히고 실호출 0이면 dead 후보 유력.
   - 프론트: `Grep pattern="\b<symbol>\b" glob="**/*.{ts,tsx}"`.
2. **동적 참조 grep**(오탐 차단, 필수): 같은 심볼을 **문자열 리터럴**로도 검색 — `Grep pattern="<symbol>" -i`로 `getattr(obj, "<symbol>")`, `"<symbol>"` 딕셔너리 키, `__all__` 등록, FastAPI 라우트 `name=`, pydantic `Field(alias=...)`, JSON 응답 키와 동명 여부를 확인.
3. **데코레이터/등록 확인**: 함수가 `@router.*`, `@app.*`, `@lru_cache`, pytest fixture, pydantic validator 등으로 *등록*되어 호출자가 코드상 안 보일 수 있다 → 등록 데코레이터가 있으면 "사용 중"으로 간주.
4. **판정 규칙**:
   - 직접 호출 0 **AND** 동적 참조 0 **AND** 등록 데코레이터 없음 → **확정 dead**.
   - 위 중 하나라도 걸리면 → **오탐(사용 중)** 또는 **보류**로 기록, 제거 금지.
   - Supabase 관련 심볼 → 무조건 **보류(Supabase)**.
5. commit 없음(다음 Step과 묶음). 단 표가 채워진 `docs/cp224b_report.md`는 Step 5 커밋에 포함.

### Step 5 — 확정 dead만 제거 + 애매분 보고
1. Step 4에서 **확정 dead**로 판정된 항목만 제거.
   - 백엔드 함수/변수 제거 시: 해당 심볼만 삭제, 주변 코드·signature 보존.
   - 프론트 unused export: `used-in-module`이면 **`export` 키워드만 제거**(코드는 유지). **완전 unused**이고 grep 0이면 심볼 자체 제거.
2. 제거할 게 없으면(전부 오탐/보류) — 그것도 정상 결과다. 코드 변경 0, 보고만.
3. **검증**:
   - 백엔드: `.venv\Scripts\python.exe -m pytest backend/tests -q` 회귀 0 + CP223 snapshot diff 0 + `.venv\Scripts\python.exe -m ruff check backend/app` 신규 error 0 + (CP222가 mypy 설정했으면) `.venv\Scripts\python.exe -m mypy backend/app` 신규 error 0.
   - 프론트: `cd frontend; npx tsc --noEmit` → error 0.
4. **commit**: `CP224b: remove confirmed dead code (grep-verified, 0 dynamic refs)`. 제거가 없으면 docs-only commit: `CP224b: dead-code audit results, no confirmed removals`.
5. **애매/보류 항목**은 `docs/cp224b_report.md` "후속" 절에 그대로 남기고, 최종 보고에 요약.

---

## 인터페이스 보존

- **함수 signature 불변**: dead 판정된 *함수 전체*는 제거하되, 살아있는 함수의 인자·반환 타입은 한 글자도 바꾸지 않는다.
- **API 응답 schema 불변**: `ApiResponse`/`ErrorResponse`/`*ResponseData`/`MetaResponse`(`schemas/`)는 라우터 `response_model`에 묶여 있으므로 **제거·필드 변경 금지**(가설 B 참조). 응답 JSON 구조 변화 = CP223 snapshot diff로 즉시 검출됨 → 차단.
- **props 인터페이스 불변**: 컴포넌트 props 타입에서 export만 떼는 경우, 그 타입을 import하는 다른 파일이 없을 때만(grep 0) 수행. 호출자가 있으면 export 유지.
- 인터페이스를 바꿔야 dead가 제거된다면 → 그건 dead가 아니다. **제거 말고 호출자 영향 분석 + 차단 보고.**

---

## 성공 기준 (측정 가능)

| 항목 | 시작값(실측) | 목표 |
|---|---|---|
| ruff F401 violations (`backend/app`) | 미측정(Step1 dry-run에서 확정) | 0 |
| vulture 후보 문서화 | 0 | 전체 후보 표로 기록(`min-confidence 60`) |
| ts-prune 후보 문서화 | 0 | 전체 후보 표로 기록(used-in-module 구분) |
| 확정 dead 제거 | — | grep 확인된 것만 제거(미확인 0건 제거) |
| 백엔드 pytest 회귀 | 9개 테스트 파일(`backend/tests`) green 가정 | 회귀 0 |
| CP223 snapshot diff | 0 (선행 그린) | 0 (제거 후에도 0) |
| 프론트 `tsc --noEmit` | error 0 가정 | error 0 (제거 후에도 0) |
| mypy 신규 error (CP222 설정 시) | — | 0 추가 |
| 예상 시간 | — | 1.5~2.5시간 |

> 핵심: **"많이 지웠다"가 성공이 아니다.** 제거가 0건이어도 "후보 전수 검증 + Supabase/오탐 정확히 분류 + 회귀 0"이면 성공이다.

---

## 검증 (구체 명령 + 기대 결과)

PowerShell 기준:

```powershell
# Step 1 ruff F401
.venv\Scripts\python.exe -m ruff --version                                  # 존재 확인(없으면 차단)
.venv\Scripts\python.exe -m ruff check backend/app --select F401            # dry-run, 목록 기록
.venv\Scripts\python.exe -m ruff check backend/app --select F401 --fix      # 자동 제거
.venv\Scripts\python.exe -m ruff check backend/app --select F401            # 기대: All checks passed (0)

# Step 2 vulture
.venv\Scripts\python.exe -m pip install vulture
.venv\Scripts\python.exe -m vulture backend/app --min-confidence 60 --sort-by-size   # 후보 목록 → 표

# Step 3 ts-prune
cd frontend; npx ts-prune                                                    # 후보 목록 → 표; cd ..

# Step 4 후보별 grep (예시 — 실제 심볼로 반복)
#   Grep tool 사용 권장. Bash grep 예시:
#   grep -rn "\bSYMBOL\b" backend --include=*.py
#   grep -rni "SYMBOL" backend --include=*.py     # 문자열/getattr 동적 참조

# Step 5 최종 회귀 검증
.venv\Scripts\python.exe -m pytest backend/tests -q                         # 기대: 회귀 0
.venv\Scripts\python.exe -m ruff check backend/app                          # 기대: 신규 error 0
cd frontend; npx tsc --noEmit; cd ..                                        # 기대: error 0
# CP223 snapshot 재실행(런북/ CP223 directive가 지정한 명령으로) → diff 0 확인
```

기대 결과 요약: ruff F401 = 0, vulture·ts-prune 후보가 `docs/cp224b_report.md`에 전부 표로, pytest 회귀 0, snapshot diff 0, tsc 0.

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 정리해서 보고한다. 그냥 넘어가기 절대 금지.**

1. **grep으로 호출 0을 증명하지 못한 후보를 제거하려는 순간** — 도구가 "unused"라 해도 직접/동적 참조 grep을 못 끝냈으면 제거 금지, 보고.
2. **vulture/ts-prune 후보가 동적 호출(`getattr`, 문자열 키, `__all__`, FastAPI/pydantic 데코레이터 등록)과 동명** — 오탐 가능. 제거 말고 "오탐 의심"으로 보고.
3. **Supabase 코드(`db.py`, `market_repo.py`, `get_supabase`/`supabase_is_configured`/`check_supabase_ready` 및 소비처)가 결과에 뜸** — 무시하고 "Supabase 보류"로 기록. 절대 제거 금지.
4. **제거 후 CP223 snapshot diff ≠ 0** — 동작이 바뀌었다는 뜻(응답 schema/값 변경). 제거가 dead가 아니었다 → 해당 커밋 revert하고 보고.
5. **제거 후 pytest 실패(특히 다수)** — 호출자가 있었다는 증거. revert + 보고.
6. **제거 후 `tsc --noEmit` error** — 프론트 타입 참조가 살아있었다. revert + 보고.
7. **ruff 또는 pytest가 `.venv`에 없음(CP222 미완) / CP223 snapshot이 red·부재** — 선행 안전망 부재. 시작하지 말고 보고.
8. **F401 fix 대상이 `__init__.py` 의도적 re-export(`__all__` 포함)** — 자동 fix가 공개 API를 깰 수 있음. 해당 항목 보류 + 보고.
9. **vulture min-confidence 60으로도 후보가 수십 개 이상 쏟아짐** — 일괄 처리 유혹 차단. 절대 일괄 제거하지 말고, 상위 일부만 grep 검증 후 나머지는 목록으로 보고.
10. **인터페이스(signature/schema/props)를 바꿔야 제거 가능** — 그건 dead가 아님. 제거 말고 호출자 영향 분석 + 보고.

---

## ADR

해당 없음. (이 CP는 "검출 도구 출력 + grep 확인 후 확정분만 제거"로, 설계 결정이 아니라 기계적 청소다. 결정 기록 대신 `docs/cp224b_report.md`의 후보 표 + 판정 근거가 감사 추적을 대신한다.)

---

## 자가 점검 결과 양식

완료 보고에 아래를 채운다:

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ______ (예: read-path만 손댐, 모델·calibration·EODHD·밴드 파이프라인 무관 → PASS 예상)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ______ (예: dead code 제거로 결합도 감소, 신규 결함 없음 / 오탐 보류 항목 N건 잔존)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ______ (예: 학습·추론·calibration 코드 미변경, snapshot diff 0 → 모델 출력 불변 → PASS 예상)

---

## 산출물

- **변경 파일 목록**: (실제 변경분만 — 예: `backend/app/...py`에서 F401 제거된 파일들, 확정 dead 제거된 파일들. 제거 0건이면 docs만.)
- **`docs/cp224b_report.md`**: 다음을 포함(필요한 만큼만):
  - **요구**: dead code 자동 검출 + grep 확인된 확정분만 제거, 광범위 삭제 금지.
  - **한 일**: ruff F401 결과, vulture 후보 표, ts-prune 후보 표, 각 후보의 grep 판정(확정/오탐/Supabase보류), 실제 제거 목록.
  - **결정**: 제거 기준(직접+동적 참조 0 AND 등록 데코레이터 없음), Supabase·테스트·collector 제외.
  - **후속**: 보류/애매 항목 목록(다음에 사람이 판단할 것), used-in-module export 정리 잔여 등.
