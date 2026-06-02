# CP229 BE 안정성 — async blocking 제거 / 캐시 안전성 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 CP 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다.
> **이 CP의 핵심 성격: "수리"가 아니라 "검증 후 필요 시 수리"다.** 아래 진단의 세 가지 의심(① async blocking ② global 누락 ③ lru_cache)은 **모두 현재 코드에서 재현되지 않았다**(작성 시점 실측). 따라서 실행자의 1차 임무는 각 의심을 grep으로 재확인하고, 결함이 없으면 "결함 없음"으로 기록하고 그 sub-step을 no-op으로 닫는 것이다. **결함이 없는데 "고치는" 행위(예: 멀쩡한 sync 라우트를 건드리거나, 정상 global을 재배치)는 금지다.** 그 자체가 회귀 위험이다.

---

## 역할 고정

- **모드:** `code` (구현 모드). 지시받은 코드 작업만 수행하고 같은 턴에 자가 점검만 보고한다.
- **권한:** 코드 수정, 로컬 검증(pytest / mypy / 로컬 uvicorn 기동)만.
- **금지:**
  - 새 모델 학습, 새 calibration 산출.
  - DB write(Supabase insert/update/delete), Supabase 호출을 새로 추가하는 변경.
  - 사용자가 직접 수정한 파일을 revert.
  - 결함이 확인되지 않은 코드를 "정리" 목적으로 변경.
- **자가 점검(보고 필수):** [Plan v3 정합] / [구조 결함] / [모델 영향] 각각 PASS·WARN·FAIL + 사유.
- **커밋 메시지:** 간결. 끝에 다음 한 줄.
  - `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## 환경

- **워킹 디렉토리:** `C:\Users\user\lens`
- **가상환경:** `.venv` (Python 3.10.0, torch 2.11.0+cu128). 활성화: `.\.venv\Scripts\Activate.ps1`
- **백엔드 기동(검증용):** ASGI 앱은 `app.main:app`. `scripts\start_demo.ps1` 은 이 저장소에 **없다**(실측). 실제 기동은 uvicorn 직접 실행으로 한다.
  ```powershell
  # 워킹 디렉토리: C:\Users\user\lens  (backend 가 import 루트)
  .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8123
  ```
  - **포트 충돌 회피:** 8000/8001 은 데모가 쓸 수 있으니 검증은 `--port 8123` 처럼 비표준 포트로 띄운다. 기동 확인만 하고 즉시 종료(Ctrl+C)한다. CP229는 실서버 상주가 목적이 아니다.
- **프론트(이 CP에서는 불필요):** `npm run dev`. CP229는 BE 전용이라 프론트 기동은 하지 않는다.
- **테스트 러너:** `pytest` 설정 파일(`pytest.ini`/`pyproject.toml`/`setup.cfg`)은 **없다**(실측). 테스트는 `backend/tests/` 아래 `unittest` 스타일이며 pytest 기본 디스커버리로 돈다. import 루트가 `backend` 이므로 반드시 아래 형태로 실행한다.
  ```powershell
  .\.venv\Scripts\python.exe -m pytest backend\tests -q
  ```

---

## 진단 (근거)

작성 시점에 실제 코드를 직접 Read/Grep으로 확인한 결과를 그대로 박는다. **세 의심 모두 현 코드에서 미재현.** 그래서 이 CP는 "검증 우선" 구조다.

### 진단 A — async def 라우트 안 blocking pandas (의심: 이벤트루프 정지)

- **사실:** `backend/app/routers/` 의 모든 라우트 핸들러는 `def`(동기)다. `async def` 라우트는 **0개**.
  - `backend/app/routers/v1/stocks.py` — `list_stocks`(L39), `get_prices`(L55), `get_indicators`(L78), `get_product_prediction_history`(L95) 전부 `def`. (파일 현재 116줄)
  - `backend/app/routers/prices.py` — `get_prices`(L21), `get_tickers`(L40) `def`. (현재 45줄)
  - `backend/app/routers/predict.py` — `predict`(L13) `def`. (현재 26줄)
  - `backend/app/routers/v1/health.py` — 확인 대상(헬스 라우트).
- **코드베이스 전체에서 `async def` 는 단 한 곳:** `backend/app/middleware/request_id.py:8`
  ```python
  async def request_id_middleware(request: Request, call_next):
      ...
      response = await call_next(request)   # L12 — pandas 없음, 순수 패스스루
  ```
  이 미들웨어는 pandas/무거운 연산을 하지 않는다.
- **무거운 pandas 위치:** 동기 서비스 계층에만 있다 — `backend/app/services/api_service.py`(`pd` import L5), 특히 `aggregate_prices`(L21~L44, `pd.DataFrame`/`resample`/`to_dict`), `resolve_price_window`(L47). 그리고 `backend/app/services/feature_svc.py`(`build_features` L343 등). 이들은 **동기 `def` 라우트에서만 호출**된다.
- **결론:** FastAPI는 `def` 라우트를 자동으로 threadpool(`run_in_threadpool`)에서 돌린다. 따라서 무거운 pandas가 이벤트 루프를 막지 않는다. **spec이 말한 "async def 라우트 + pandas로 이벤트루프 정지" 문제는 이 저장소에 존재하지 않는다.** → 수정 대상 없음(검증으로 확정 + 회귀 가드만).

### 진단 B — `local_market_svc.py:79` global 키워드 누락 (의심: 캐시 재할당이 로컬변수화)

- **사실:** `local_market_svc.py` 라는 파일은 **없다**(실측, repo 전체 검색 0건). spec의 파일명은 가설이다.
- **모듈 레벨 가변 전역을 재할당하는 실제 위치 3곳 — 전부 `global` 선언 정상:**
  - `backend/app/db.py` — `_client`(L11) 를 `get_supabase()`에서 재할당. **`global _client`(L16) 선언 있음.** `reset_supabase_client()`도 `global _client`(L29) 있음. ✅
  - `backend/collector/sources/yf_common.py` — `_INITIALIZED`(L8) 를 `prepare_yfinance()`에서 재할당. **`global _INITIALIZED`(L13) 선언 있음.** ✅
  - `backend/collector/sources/edgar.py` — `_TICKER_CIK_CACHE`(L12) 를 `ticker_to_cik()`에서 재할당. **`global _TICKER_CIK_CACHE`(L37) 선언 있음.** ✅
- **결론:** "global 누락으로 캐시 재할당이 로컬변수화" 결함은 **현 코드에 없다.** → 검증으로 "결함 없음" 확정. 정상 코드를 건드리지 않는다.

### 진단 C — `strategy_backtest_svc lru_cache(maxsize=1)` (의심: 전략 1개만 캐시, 미스 시 lock 밖 실행)

- **사실:** `strategy_backtest_svc.py` 파일은 **없다**. repo 전체에서 `lru_cache` 사용 **0건**, `threading.Lock`/`Lock()` 사용 **0건**(실측).
  - 현존 전략 파일은 `backend/app/services/strategy_svc.py`(현재 18줄) — `generate_signal()` 순수 함수 하나뿐. 캐시·lock·무거운 로딩 없음.
- **spec이 "보존하라"고 한 레퍼런스 `parquet_store.py` 의 `_LOCK` double-check locking 패턴도 이 저장소에 없다**(파일 없음, `Lock` 0건).
- **결론:** lru_cache 튜닝 / 명시적 Lock 적용 대상이 **현재 없다.** → 검증으로 "대상 없음" 확정. 캐시 인프라를 새로 만들지 않는다(그건 기능 추가이며 CP229 범위 밖).

### 진단 종합

CP229가 가정한 3개 결함은 현 코드에 모두 부재. 이 CP의 실질 산출물은 **(1) 부재 사실을 grep 증거와 함께 확정 기록**, **(2) async-blocking 회귀를 막는 경량 가드 1건**(라우트가 `async def`로 바뀌고 그 안에서 pandas를 직접 호출하는 미래 변경을 잡는 가벼운 테스트), **(3) ADR로 "왜 손대지 않았는가"를 남기는 것**이다. 결함이 실제로 발견되면(아래 Step에서 grep이 spec과 다른 결과를 내면) 그때만 Strangler 패턴으로 수리한다.

> 출처: 본 진단의 모든 파일/줄번호는 작성자가 `Read`/`Grep`으로 직접 확인. (`backend/app/routers/**`, `backend/app/services/api_service.py`, `feature_svc.py`, `strategy_svc.py`, `backend/app/db.py`, `backend/collector/sources/yf_common.py`, `edgar.py`, `backend/app/middleware/request_id.py`, `backend/tests/**`)

---

## 선행 의존

- **CP223(백엔드 characterization 스냅샷) 그린이어야 시작 가능.** CP229는 BE 안정성 변경이므로, 안전망인 CP223 characterization 출력이 그린(byte-identical 기준 확보)인 상태가 전제다. CP223이 아직이면 **이 CP를 시작하지 말고 런북에 그 사실을 보고**한다.
- 이 저장소에는 별도 snapshot 프레임워크(syrupy 등)가 없으므로, "snapshot"의 실체는 **CP223가 고정한 characterization 테스트/출력**이다. CP229의 "snapshot diff 0" 판정은 그 산출물 기준으로 한다.
- 그 외 선행: 없음.

---

## 범위

### 포함

- 진단 A/B/C 각각을 grep으로 재확인하고 **결함 유무를 증거와 함께 확정**.
- 진단 A에 대한 **회귀 가드 1건 추가**: "라우트 핸들러가 `async def`가 되었는데 그 본문이 무거운 동기 pandas를 직접 호출"하는 미래 변경을 잡는 경량 정적 검사 테스트(런타임 동작/스키마 불변).
- 결함이 **실제로 발견된 경우에 한해**, 해당 Step의 Strangler 수리(아래 각 Step의 조건부 절차).

### 제외

- **Supabase 관련 일체 보류.** `db.py`의 `_client` 캐싱은 정상이며, Supabase 호출 추가·변경·write 금지(전역 금지 규칙과 동일).
- 사용자가 직접 수정한 파일의 revert.
- 캐시 인프라 신설(lru_cache/Lock 새로 도입). 대상 코드가 없으므로 이는 기능 추가이며 범위 밖.
- 라우트의 `def` → `async def` 전환(또는 그 역) **동작·스키마 변경**. 진단 A 결함이 없으므로 전환 자체가 불필요하다.
- 프론트, 모델, 학습, calibration.

---

## Sub-step (Strangler Fig, 작은 단위)

각 Step은 독립 revert 단위다. **Step 1~3은 "검증 후 결함 없으면 no-op 커밋(테스트/문서만)"이 정상 경로다.** Step 4는 가드 추가. Step 5는 sanity. 결함이 발견되면 해당 Step의 "조건부 수리" 절차를 따른다(옛 코드 옆 새 코드 공존 → caller 이전 → 옛 제거).

### Step 1 — async 라우트 blocking pandas 식별 (검증 전용, 코드 변경 없음 기대)

1. 다음 grep으로 라우트 계층의 `async def` 존재 여부를 재확인.
   ```powershell
   .\.venv\Scripts\python.exe -m pytest --version   # 환경 sanity
   ```
   ```powershell
   # 라우트 핸들러 async 여부
   Select-String -Path backend\app\routers\*.py, backend\app\routers\v1\*.py -Pattern 'async def' -AllMatches
   # 코드베이스 전역 async def 위치
   Select-String -Path backend\app\**\*.py -Pattern 'async def'
   # 서비스 계층 pandas 사용처(호출이 동기 def 라우트인지 교차확인용)
   Select-String -Path backend\app\services\*.py -Pattern 'import pandas|pd\.'
   ```
2. **기대 결과:** 라우트 `async def` 0건. 전역 `async def`는 `middleware\request_id.py` 1건뿐(pandas 없음). → 진단 A 확정: **blocking 결함 없음**.
3. **분기:**
   - 위 기대와 일치 → **코드 변경 없음.** 결과를 `docs/cp229_report.md`(아래 산출물)에 grep 증거로 기록. Step 1은 문서 커밋에 포함.
   - 만약 라우트에 `async def`가 발견되고 그 본문이 무거운 동기 pandas(`aggregate_prices`/`build_features`/`resample`/`to_dict` 등)를 `await` 없이 직접 호출 → **차단 트리거**(아래) 대상. 즉시 멈추고 보고. 임의로 `def` 전환/`run_in_threadpool` 래핑하지 말 것(스키마·직렬화 변경 위험). 보고 후 사용자 승인 시에만, "기존 핸들러 옆에 `run_in_threadpool(sync_impl, ...)` 위임형 새 본문 추가 → 동일 응답 확인 → 옛 본문 제거" Strangler로 진행.
4. **커밋(있다면):** 문서/증거만. 예: `docs(cp229): record async-route blocking audit (no defect)`

### Step 2 — global 키워드 정합 재확인 (검증 전용, 코드 변경 없음 기대)

1. grep으로 모듈 레벨 전역 재할당 지점과 `global` 선언 짝을 재확인.
   ```powershell
   Select-String -Path backend\app\db.py, backend\collector\sources\yf_common.py, backend\collector\sources\edgar.py -Pattern '^\s*global\s+|^_[A-Z]'
   # 누락 의심 패턴: 함수 내부에서 모듈 전역에 '=' 재할당하는데 global 선언이 없는 경우를 눈으로 교차확인
   ```
2. **기대 결과:** `db.py`(`_client`/`global _client` L16·L29), `yf_common.py`(`_INITIALIZED`/`global` L13), `edgar.py`(`_TICKER_CIK_CACHE`/`global` L37) 모두 짝 정상. → 진단 B 확정: **누락 없음**.
3. **분기:**
   - 일치 → **코드 변경 없음.** report에 기록.
   - 만약 어떤 함수가 모듈 전역을 재할당하는데 `global` 선언이 빠진 곳을 새로 발견 → 그곳만 `global X` 한 줄 추가. **단, 추가 직후 캐시 동작이 달라지면(예: 기존엔 사실상 매번 재계산되던 게 캐시되기 시작) 그것은 동작 변경 → 차단 트리거.** 보고하고 멈춘다.
4. **커밋(있다면):** `docs(cp229): record module-global keyword audit (all paired)` 또는 결함 발견 시 `fix(cp229): add missing global for <name>`.

### Step 3 — 캐시(lru_cache/Lock) 적정성 재확인 (검증 전용, 코드 변경 없음 기대)

1. grep으로 lru_cache/Lock/double-check 패턴 존재 여부 재확인.
   ```powershell
   Select-String -Path backend\app\**\*.py -Pattern 'lru_cache|functools\.cache|Lock\(\)|RLock\(\)'
   Select-String -Path backend\**\*.py -Pattern 'parquet_store'
   ```
2. **기대 결과:** `lru_cache`/`Lock` 0건, `parquet_store` 0건. 현존 전략은 `strategy_svc.py`의 순수 `generate_signal` 뿐. → 진단 C 확정: **튜닝/Lock 대상 없음**.
3. **분기:**
   - 일치 → **코드 변경 없음.** report에 기록. (캐시 인프라 신설은 범위 밖이므로 절대 만들지 않는다.)
   - 만약 `lru_cache(maxsize=...)`나 무거운 로딩 함수 + 동시성 노출 지점이 새로 발견 → 차단 트리거로 보고. 승인 시에만 `parquet_store`류 double-check locking 레퍼런스(현재 repo에 없으므로, 발견된 기존 정상 패턴이 있으면 그걸 레퍼런스로)를 그대로 본떠 적용. signature·반환값 불변 유지.
4. **커밋(있다면):** `docs(cp229): record cache/lock audit (no lru_cache/Lock present)`.

### Step 4 — async-blocking 회귀 가드 추가 (코드 변경 = 테스트 1개)

> 목적: 미래에 누군가 라우트를 `async def`로 바꾸면서 그 안에서 무거운 동기 pandas를 직접 호출하면 CI/pytest가 잡도록 한다. 런타임 동작·응답 스키마는 건드리지 않는다.

1. `backend/tests/test_async_safety.py` 신규 작성(아래 인터페이스 보존 규칙 준수). AST로 라우트 모듈을 정적 분석한다(서버 기동·네트워크 불필요).
   - 대상 모듈: `app.routers.prices`, `app.routers.predict`, `app.routers.v1.stocks`, `app.routers.v1.health`.
   - 검사 규칙: 각 모듈에서 `@router.<method>` 데코레이터가 붙은 함수가 `async def`이면, 그 함수 본문에 "무거운 동기 호출명"(허용목록: `aggregate_prices`, `build_features`, `build_latest_feature_rows`, `resample_price_frame`, `get_price_response_data`, `get_indicator_response_data`)이 `await` 없이 직접 등장하지 않아야 한다. 위반 시 fail(파일·함수명 출력).
   - 현재는 라우트가 전부 `def`이므로 이 테스트는 **자명하게 통과**한다(가드로서 미래만 잡음).
2. 작은 단위 원칙: 이 테스트는 기존 테스트를 수정하지 않는다(추가만). 기존 동작에 영향 0.
3. **검증:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest backend\tests\test_async_safety.py -q
   .\.venv\Scripts\python.exe -m pytest backend\tests -q   # 전체 회귀 0 확인
   ```
4. **커밋:** `test(cp229): add static guard against async-route blocking pandas`

### Step 5 — 간단 부하 sanity (검증 전용)

> 동시 요청에서 응답 스키마가 흔들리지 않는지(=캐시/전역이 요청 간 오염을 만들지 않는지) 가볍게 확인. DB 미설정 환경에서도 가능한 범위로 한정.

1. `TestClient`로 헬스 + (가능하면) 모킹된 가격 경로를 **동시/반복 호출**해 응답 형태가 일정한지 확인하는 일회성 검증을 한다. 새 영구 테스트가 필요하면 `backend/tests/` 에 추가하되, Supabase 실호출은 하지 않는다(기존 `test_api.py`처럼 `unittest.mock.patch`로 repo 함수를 모킹).
   - 최소선: 아래로 헬스 라우트를 반복 호출해 200 + `data.status == "ok"` + `meta.request_id` 존재가 매회 동일한지 확인.
   ```powershell
   .\.venv\Scripts\python.exe -m pytest backend\tests\test_api.py -q
   ```
2. **기대 결과:** 기존 `test_api.py`/`test_services.py` 전부 통과(회귀 0). 동시 호출에서도 스키마 동일.
3. **차단:** 반복/동시 호출에서 응답 스키마나 `request_id` 처리(미들웨어)가 흔들리면 → 보고.
4. **커밋(있다면):** sanity가 영구 테스트로 가치 있으면 `test(cp229): add concurrent health response sanity`, 아니면 커밋 없음.

---

## 인터페이스 보존

- **API 응답 schema 무변경.** `ApiResponse[...]` 래핑, `success_response()` 출력, ETag/`Cache-Control` 헤더(특히 `stocks.py` `_build_price_etag`, `prices.py`), 304 분기 로직 등 모두 그대로.
- **함수 signature 무변경:** `get_price_response_data(...)`, `get_latest_prediction_data(...)`, `aggregate_prices(rows, timeframe)`, `get_stocks(*, search, limit)` 등 서비스 함수 시그니처·반환 타입 동일.
- **전역/캐시 의미 무변경:** `db.get_supabase()`/`reset_supabase_client()`, `yf_common.prepare_yfinance()`, `edgar.ticker_to_cik()` 의 캐싱 동작(첫 호출 1회 초기화 후 재사용)을 바꾸지 않는다.
- 만약 결함이 실제로 발견되어 수정이 불가피하고 그 수정이 위 어느 인터페이스라도 건드려야 한다면 → **호출자 영향 분석을 먼저 적고, 차단 보고**한다(자동 진행 금지).

---

## 성공 기준 (측정 가능)

| 항목 | 시작값(실측) | 목표 | 판정 |
|---|---|---|---|
| 라우트 `async def` blocking | 0건(전부 `def`) | 0건 유지 + 회귀 가드 존재 | grep 0건 & Step4 테스트 통과 |
| 모듈 전역 `global` 정합 | db/yf/edgar 3곳 모두 짝 정상 | 누락 0 | Step2 grep 일치 |
| lru_cache/Lock 부적정 | 0건(미사용) | 신규 도입 0, 대상 없음 확정 | Step3 grep 0건 |
| pytest 회귀 | 기존 `test_api.py`/`test_services.py`/`test_feature_svc.py`/`test_collector_jobs.py` 통과 | 신규 가드 포함 전부 통과, 회귀 0 | `pytest backend\tests -q` 그린 |
| 신규 가드 테스트 | 없음 | `test_async_safety.py` 1개 통과 | pytest 통과 |
| snapshot diff (CP223 characterization 기준) | — | 0 | CP223 산출물 byte-identical |
| mypy 신규 에러 | 기준선 | 0 추가 | 아래 검증 명령 |
| 예상 시간 | — | 1.0~1.5시간 | — |

> tsc 항목은 BE 전용 CP라 **해당 없음(생략)**.

---

## 검증

워킹 디렉토리 `C:\Users\user\lens` 에서:

```powershell
# 0) 환경
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -c "import sys; print(sys.version)"   # 3.10.0 기대

# 1) 진단 재확인 grep (Step 1~3)
Select-String -Path backend\app\routers\*.py, backend\app\routers\v1\*.py -Pattern 'async def'        # 기대: 매치 없음
Select-String -Path backend\app\**\*.py -Pattern 'async def'                                           # 기대: middleware\request_id.py 1건
Select-String -Path backend\app\db.py, backend\collector\sources\yf_common.py, backend\collector\sources\edgar.py -Pattern 'global '   # 기대: 각 파일 global 선언 존재
Select-String -Path backend\app\**\*.py -Pattern 'lru_cache|Lock\(\)'                                  # 기대: 매치 없음

# 2) 신규 가드 + 전체 회귀
.\.venv\Scripts\python.exe -m pytest backend\tests\test_async_safety.py -q   # 기대: 1 passed
.\.venv\Scripts\python.exe -m pytest backend\tests -q                        # 기대: all passed, 회귀 0

# 3) mypy (신규 에러 0 확인; 기준선과 비교)
.\.venv\Scripts\python.exe -m mypy backend\app --ignore-missing-imports      # 기대: 신규 에러 0 추가

# 4) 기동 sanity (포트 8123, 즉시 종료)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8123
#   다른 터미널: Invoke-WebRequest http://127.0.0.1:8123/api/v1/health/live  → 200 / data.status=ok
#   확인 후 Ctrl+C
```

**기대 결과 요약:** async grep 라우트 0건 / global 짝 정상 / lru_cache·Lock 0건 / pytest 전부 통과(회귀 0) / mypy 신규 에러 0 / 헬스 200.

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **라우트에서 `async def` + 무거운 동기 pandas 직접 호출이 실제로 발견됨** (진단 A가 틀림). → 임의로 `def` 전환/`run_in_threadpool` 래핑하지 말 것. `def` 전환이 응답 schema나 직렬화 결과를 바꿀 수 있으므로 보고 후 승인받아 Strangler로만 진행.
2. **`def` ↔ `async def` 전환이 응답 schema/직렬화/헤더(ETag·Cache-Control·304)에 조금이라도 변화를 만듦.** → 보고.
3. **global 선언 추가/수정 후 캐시 동작이 달라짐**(snapshot/characterization diff 발생, 또는 "매번 재계산 → 캐시됨"처럼 의미 변화). → 동작 변경이므로 보고.
4. **CP223 characterization(=snapshot) diff가 0이 아님** = 동작이 바뀐 것. → 즉시 중단·보고.
5. **기존 pytest가 다수 실패**(특히 `test_api.py`/`test_services.py`/`test_feature_svc.py`) → 회귀. 보고.
6. **로컬 기동이 환경변수 누락(SUPABASE_URL/KEY 등)으로 실패**해서 검증이 막힘 → 추측으로 우회하지 말고 상황 보고(헬스 live는 DB 불필요하므로 그것까지는 확인).
7. **lru_cache/Lock/캐시 인프라를 "개선" 명목으로 새로 만들고 싶어지는 경우** → 범위 밖. 보고하고 멈춤.
8. **수정이 어떤 공개 함수 signature/응답 schema라도 변경해야 하는 상황** → 호출자 영향 분석 적고 보고.

---

## ADR

- 완료 후 `docs/adr/0019-async-blocking-cache-safety.md` 1장(200~300단어) 작성.
- `docs/adr/` 디렉토리가 없으면 생성한다(실측: 현재 ADR 디렉토리/파일 없음).
- 기록할 결정: **"CP229에서 async-blocking·캐시 안전성 의심 3건을 실측 검증한 결과 현 코드에 결함이 부재함을 확인했고, 따라서 라우트 동시성 모델(동기 `def` → FastAPI threadpool)과 기존 전역 캐시(`_client`/`_INITIALIZED`/`_TICKER_CIK_CACHE`)를 의도적으로 변경하지 않기로 했다. 대신 미래 회귀를 막는 정적 가드 테스트만 추가했다."** 결정의 근거(동기 라우트는 threadpool에서 실행되어 이벤트루프를 막지 않음), 대안(불필요한 `async` 전환/캐시 신설)을 기각한 이유, 영향(동작·스키마 불변)을 함께 적는다.

---

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다.

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ____ (밴드 본체·fidelity 우선·EODHD 유지 등 Plan v3 원칙에 어긋난 변경이 없는지)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ____ (전역/캐시/동시성 측면에서 새 결함을 만들지 않았는지, 추출 순서 원칙 준수)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ____ (학습·calibration·예측 산출물에 영향 0인지)

---

## 산출물

- **변경 파일(예상):**
  - `backend/tests/test_async_safety.py` (신규, Step 4 가드)
  - (조건부) Step 1~3에서 결함이 실제 발견된 경우에만 해당 소스 파일 1곳
  - `docs/adr/0019-async-blocking-cache-safety.md` (신규)
- **리포트:** `docs/cp229_report.md` — 요구 / 한 일(각 Step의 grep 증거 + 결함 유무 결론) / 결정(왜 손대지 않았는가) / 후속. 필요한 만큼만, 간결하게.
