# CP228 BE 관측성 — structlog + request_id 전파 (Directive)

> 이 문서는 단독 실행 가능한 지시서다. 실행자(새 Claude Code 세션)는 이 문서만 읽고
> 코드를 고치고 검증하고 중단 판단을 내릴 수 있어야 한다. 추측 금지. 막히면 멈추고 보고.

---

## 역할 고정

- **모드**: `code` (구현 모드). 설계 토론 아님. 지시받은 범위만 구현하고 자가 점검만 보고.
- **권한**: 코드 수정 + 로컬 검증(lint / mypy / pytest / 백엔드 기동 후 curl)만.
- **금지**:
  - 새 모델 학습 금지.
  - 새 calibration 산출 금지.
  - DB write 금지 (Supabase insert/update/delete 일절 금지).
  - Supabase 네트워크 호출 금지 (검증은 `LENS_USE_LOCAL_SNAPSHOTS=1` 로컬 모드로만).
  - 사용자가 직접 수정한 파일 revert 금지 (덮어쓰기 전 `git log -1 <file>` 로 작성자 확인, 의심되면 멈추고 보고).
- **자가 점검 (완료 후 필수)**: [Plan v3 정합] / [구조 결함] / [모델 영향] 각각 PASS·WARN·FAIL + 사유 한 줄.
- **커밋 메시지**: 간결. 끝에 반드시 다음 한 줄.
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). 활성화: `.\.venv\Scripts\Activate.ps1`
  - venv python 직접 경로: `C:\Users\user\lens\.venv\Scripts\python.exe`
- **백엔드 requirements**: `backend/requirements.txt` (현재 14줄, 마지막 줄 `wandb==0.19.11`).
- **백엔드 기동(택1)**:
  - `scripts\start_demo.ps1` (백엔드+프론트 동시; 백엔드는 cmd.exe 자식 프로세스로 `uvicorn app.main:app --host 127.0.0.1 --port 8000`, `PYTHONPATH=backend`, `LENS_USE_LOCAL_SNAPSHOTS=1` 세팅됨).
  - 백엔드만 수동 기동:
    ```powershell
    $env:PYTHONPATH = "C:\Users\user\lens\backend"
    $env:LENS_USE_LOCAL_SNAPSHOTS = "1"
    $env:LENS_LOCAL_SNAPSHOT_DIR = "C:\Users\user\lens\data\parquet"
    $env:MARKET_DATA_PROVIDER = "yfinance"
    & "C:\Users\user\lens\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8001
    ```
    (검증용은 8001 등 비어있는 포트 사용 — start_demo 가 8000 점유 중일 수 있음. **포트 충돌 피하라**.)
- **프론트**: `cd frontend; npm run dev` — **이 CP에선 프론트 불필요**(BE 로깅만). 기동하지 말 것.
- **PowerShell 표기**: 표준출력 버림은 `2>$null`, 환경변수는 `$env:VAR`, 줄 연속은 백틱(`` ` ``).

---

## 진단 (근거)

조사 출처: 아래 파일을 직접 Read/Grep 으로 확인함 (이 worktree `C:\Users\user\lens`).

### 1. request_id 는 미들웨어에만 있고 service/repo 로그엔 안 박힘

`backend/app/middleware/request_id.py` (현재 **15줄**):

```python
8  async def request_id_middleware(request: Request, call_next):
9      request_id = request.headers.get("X-Request-Id") or str(uuid4())
10     request.state.request_id = request_id
11
12     response = await call_next(request)
13     response.headers["X-Request-Id"] = request_id
14     return response
```

- request_id 는 `request.state.request_id` 에만 산다. 로깅 시스템과 연결이 없다.
- 응답 헤더/`meta` 에는 들어가지만(`backend/app/core/http.py:8-13` `build_meta`), **로그 라인엔 request_id 가 없다**.
- 헤더 키가 `X-Request-Id` (마지막 d 소문자) 임에 유의. asgi-correlation-id 기본 헤더는 `X-Request-ID`. 호환을 위해 미들웨어 설정에서 헤더명을 명시한다(아래 Step 3).

### 2. 로깅이 거의 없음 — service/repo 가 로거조차 없음

backend/app 소스(.pyc 제외) 중 로거를 가진 파일은 **정확히 3개**:

| 파일 | 줄 | 로거 |
|---|---|---|
| `backend/app/main.py` | 16 | `logging.getLogger("lens.api")` |
| `backend/app/repositories/market_repo.py` | 14 | `logging.getLogger(__name__)` |
| `backend/app/services/parquet_store.py` | 18 | `logging.getLogger("lens.parquet_store")` |

backend/app 전체 소스 파일 수는 **34개**(`.pyc` 제외 기준은 위 3개에 로거 존재). 프로젝트 전역으로 넓히면 `ai/eval/significance/*` 4개가 추가로 표준 logging 을 쓴다(이 CP 범위 밖, 아래 범위 참조). 즉 "로깅 7/40" 의 7 = backend/app 3 + ai/eval 4.

- **`backend/app/services/local_market_svc.py` (현재 167줄) 는 로거가 아예 없다.** 폴백 경로 전체가 로그 침묵.
- `parquet_store.py` 는 `logger.info("loaded parquet ...")`(69행), `logger.warning("parquet missing ...")`(63행) 등을 찍지만 request_id 가 없어 어느 요청이 로드를 유발했는지 추적 불가.

### 3. structured logging 부재

- 현재 로깅은 전부 표준 `logging` + `%`-format 문자열. JSON 출력 아님. 로그 집계/검색 어려움.
- `grep -rn -i "structlog|asgi.correlation|correlation"` backend → **0건**. structlog 미설치.
- `grep -rn -i "sentry"` backend → **0건**. **Sentry 미설치**. → Sentry 연동은 "설치돼 있으면" 조건이므로 이 CP에선 **skip**(아래 범위 참조).

### 4. 왜 contextvars 인가 (ASGI 단일 이벤트루프)

- FastAPI/Starlette 는 단일 스레드 이벤트루프에서 여러 요청을 코루틴으로 동시 처리한다.
- `threading.local()` 은 **요청 간 격리를 보장하지 못한다**(같은 스레드에서 여러 요청이 인터리브됨 → request_id 누수/오염).
- 따라서 요청별 컨텍스트는 반드시 `contextvars.ContextVar` 기반이어야 한다. asgi-correlation-id(snok) 가 정확히 이 방식이고, structlog 의 `merge_contextvars` 프로세서가 같은 contextvars 를 로그에 머지한다.

---

## 선행 의존

- **CP223 (백엔드 characterization 스냅샷) 그린**: 필수. 안전망 우선 규칙상 CP223 의 응답 스냅샷이 그린이 아니면 이 CP를 **시작하지 마라**. 이 CP는 로깅만 추가하므로 응답 schema 무변경이어야 하고, 그 무변경을 증명할 스냅샷이 CP223 산출물이다.
  - 확인: `docs/cp223_*_report.md` 또는 런북상 CP223 상태가 그린인지. 스냅샷 테스트 위치(예: `backend/tests/test_cp223_*` 또는 characterization 스냅샷 파일)를 먼저 찾아 1회 실행해 그린인지 확인 후 진행.
  - CP223 스냅샷을 못 찾거나 레드면 **즉시 중단·보고** (차단 트리거 참조).

---

## 범위

### 포함
- `backend/requirements.txt` 에 `structlog`, `asgi-correlation-id` 추가.
- `backend/app/core/logging.py` 신규 — structlog configure 함수.
- `backend/app/main.py` 에서 앱 생성 직후 structlog configure 호출 + `CorrelationIdMiddleware` 등록(미들웨어 순서 주의).
- 기존 3개 로거(`main.py` / `market_repo.py` / `parquet_store.py`)를 structlog 로 전환.
- `local_market_svc.py` 에 structlog 로거 신설(폴백 경로 핵심 지점 1~2곳에만 최소 로그; 과한 로그 추가 금지).
- 자체 `request_id_middleware` 와 `CorrelationIdMiddleware` 의 관계 정리(중복 제거 또는 호환 유지 — Step 3 에서 결정).

### 제외 (건드리지 마라)
- **Supabase 연동 일절 보류**. DB 호출/write 금지.
- **Sentry 연동 skip** (미설치 확인됨). 코드에 `sentry_sdk` import 추가 금지. ADR 에 "Sentry 미설치라 자동 연동 skip, 설치 시 correlation id→transaction_id 는 asgi-correlation-id 의 sentry 연동으로 후속" 한 줄만 남긴다.
- **`ai/eval/significance/*` 의 4개 로거 전환 제외** (학습/평가 경로, 이 CP는 BE 런타임 관측성에 한정).
- **`backend/collector/*`, `backend/db/*` 의 로깅 제외** (별도 CP).
- 프론트 일절 제외.
- 응답 schema / API 동작 변경 일절 금지.

---

## Sub-step (Strangler Fig, 작은 단위)

> 원칙: 옛 코드 옆에 새 코드 공존 → caller 이전 → 옛 제거. 한 Step = 한 revert 단위.
> 각 Step 끝에 commit + 검증. **리팩토링 커밋과 동작 변경 커밋을 섞지 마라.**
> 이 CP는 전부 "관측성 추가"라 동작 변경이 없어야 한다(있으면 차단 트리거).

추출 순서 원칙(순수→I/O→상태) 적용: Step 2(순수 설정 함수) → Step 3(미들웨어=I/O 경계) → Step 4(상태 의존 로거 전환).

### Step 1 — 패키지 설치 + requirements 반영
1. 버전 선정(이 CP 작성 시점 안정 핀; 설치 후 실제 해석된 버전으로 고정):
   - `structlog` (24.x 계열)
   - `asgi-correlation-id` (4.x 계열)
2. 설치:
   ```powershell
   & "C:\Users\user\lens\.venv\Scripts\python.exe" -m pip install structlog asgi-correlation-id
   ```
3. 실제 설치된 버전 확인 후 정확히 핀:
   ```powershell
   & "C:\Users\user\lens\.venv\Scripts\python.exe" -m pip show structlog asgi-correlation-id | Select-String "Name|Version"
   ```
4. `backend/requirements.txt` 마지막(`wandb==0.19.11` 다음 줄)에 추가:
   ```
   structlog==<해석된버전>
   asgi-correlation-id==<해석된버전>
   ```
5. **검증**: `python -c "import structlog, asgi_correlation_id; print(structlog.__version__)"` 가 에러 없이 버전 출력.
6. **commit**: `chore(be): add structlog + asgi-correlation-id deps (CP228)`

### Step 2 — structlog config 모듈 신설 (순수 설정, caller 아직 없음)
1. 신규 파일 `backend/app/core/logging.py` 작성. 핵심 프로세서 체인:
   - `structlog.contextvars.merge_contextvars` (request_id 등 contextvars 머지 — **이게 전파의 핵심**)
   - `structlog.processors.add_log_level`
   - `structlog.processors.TimeStamper(fmt="iso")`
   - `structlog.processors.StackInfoRenderer()`
   - `structlog.processors.format_exc_info`
   - 렌더러: 환경 분기 — 프로덕션 `structlog.processors.JSONRenderer()`, 개발 `structlog.dev.ConsoleRenderer()`.
     - 분기 기준: `os.environ.get("LENS_ENV", "dev")` 또는 `LENS_LOG_JSON`(없으면 dev=Console). 기준 env 이름은 자유지만 ADR 에 기록.
   - `wrapper_class=structlog.make_filtering_bound_logger(<level>)`, level 은 `LENS_LOG_LEVEL`(기본 INFO).
   - `cache_logger_on_first_use=True`.
2. **표준 logging 브리지**: 기존 `logging.getLogger` 가 찍는 라인도 같은 포맷으로 나오도록 `logging.basicConfig` 또는 `ProcessorFormatter` 로 stdlib→structlog 브리지 구성(전환 전 기존 로그가 깨지지 않게). 최소한 root logger 가 stdout 으로 나가게 하고 level 세팅.
   - 권장: structlog 공식 "Rendering Using structlog-based Formatters for stdlib logging" 패턴(`ProcessorFormatter`) 사용. 과설계 피하고 동작만 보장.
3. `configure_logging()` 함수 1개 export. 멱등(여러 번 호출해도 안전)하게.
4. **이 시점엔 main.py 에서 호출하지 않는다**(공존). import 만 가능한 상태.
5. **검증**:
   ```powershell
   $env:PYTHONPATH = "C:\Users\user\lens\backend"
   & "C:\Users\user\lens\.venv\Scripts\python.exe" -c "from app.core.logging import configure_logging; configure_logging(); import structlog; structlog.get_logger('t').info('hello', k=1)"
   ```
   → 한 줄 로그가 콘솔에 (dev면 ConsoleRenderer 컬러/键값, JSON 모드면 `{\"event\": \"hello\", ...}`) 출력되면 OK.
6. **commit**: `feat(be): add structlog configure_logging module (CP228, not yet wired)`

### Step 3 — CorrelationIdMiddleware 등록 + configure 호출 (I/O 경계)
현재 `main.py` 미들웨어 등록 순서(파일 그대로):
```python
47  app.add_middleware(CORSMiddleware, ...)         # 가장 바깥
55  app.add_middleware(GZipMiddleware, minimum_size=512)
56  app.middleware("http")(request_id_middleware)   # 가장 안쪽
```
> Starlette 미들웨어는 **나중에 add 된 것이 더 안쪽(요청을 더 늦게 받고 응답을 더 먼저 반환)**. 라우트에 가장 가까운 것이 마지막 add.

1. 앱 생성 직후(`app = FastAPI(...)`, 현재 45행) **바로 다음 줄**에 `configure_logging()` 호출 추가.
2. `from asgi_correlation_id import CorrelationIdMiddleware` import 추가.
3. `CorrelationIdMiddleware` 를 **CORS 보다 안쪽, 그러나 GZip/라우트보다 바깥** 위치에 등록. 권장 순서(바깥→안쪽):
   - CORS (가장 바깥; preflight 가 correlation 보다 먼저 처리되어야 함)
   - **CorrelationIdMiddleware** (여기서 request_id 컨텍스트 set → 이후 모든 처리에서 보임)
   - GZip
   - 라우트
   설정:
   ```python
   app.add_middleware(
       CorrelationIdMiddleware,
       header_name="X-Request-Id",   # 기존 자체 미들웨어와 동일 헤더키(소문자 d)
       update_request_header=True,
   )
   ```
   add 순서로 위 바깥→안쪽을 만들려면: `add_middleware` 는 **역순으로 바깥**이 되므로, 코드상
   `CORS` 를 마지막에 add 해야 가장 바깥이 된다. **현 코드는 CORS 가 먼저 add 되어 가장 바깥**이다(이미 맞음). CorrelationId 는 CORS add **다음**, GZip add **이전**에 넣으면 CORS 안쪽·GZip 바깥쪽이 된다.
   → 즉 47행(CORS) 다음, 55행(GZip) 이전에 `app.add_middleware(CorrelationIdMiddleware, ...)` 삽입.
4. **자체 `request_id_middleware` 처리(공존→이전→제거)**:
   - 먼저 **공존**: CorrelationIdMiddleware 추가하되 56행 자체 미들웨어는 그대로 둔다. 단 이중 생성 방지를 위해, 자체 미들웨어가 `request.state.request_id` 를 **CorrelationId 가 만든 값으로 채우게** 한다.
   - asgi-correlation-id 는 `asgi_correlation_id.correlation_id` ContextVar 에 값을 넣는다. `http.py:build_meta` 와 예외 핸들러(`main.py:99,138`)는 `request.state.request_id` 를 읽으므로, **그 값이 비지 않도록** 자체 미들웨어를 다음으로 축소:
     ```python
     # request_id.py (전환 후)
     from asgi_correlation_id import correlation_id
     async def request_id_middleware(request, call_next):
         request.state.request_id = correlation_id.get() or "-"
         return await call_next(request)
     ```
     (응답 헤더 echo 는 CorrelationIdMiddleware 가 담당하므로 제거)
   - **caller 이전 확인**: `request.state.request_id` 를 읽는 모든 지점이 여전히 값을 받는지 점검:
     - `backend/app/core/http.py:10` `build_meta`
     - `backend/app/main.py:99` `handle_app_error`
     - `backend/app/main.py:138` `handle_unexpected_error`
     (grep: `Grep "request_id" backend/app -n` 로 빠짐없이 확인)
5. **검증**:
   - 백엔드 기동(8001 포트, 위 환경 블록 명령).
   - 헤더 없이 호출 → 응답 `meta.request_id` 가 빈 문자열이 아니어야 함:
     ```powershell
     (Invoke-WebRequest "http://127.0.0.1:8001/api/v1/health/live" -UseBasicParsing).Content
     # body.meta.request_id 존재 + 응답헤더 X-Request-Id 존재 확인
     (Invoke-WebRequest "http://127.0.0.1:8001/api/v1/health/live" -UseBasicParsing).Headers["X-Request-Id"]
     ```
   - 헤더 주고 호출 → 그 값이 echo 되는지:
     ```powershell
     (Invoke-WebRequest "http://127.0.0.1:8001/api/v1/health/live" -Headers @{ "X-Request-Id" = "test-corr-123" } -UseBasicParsing).Headers["X-Request-Id"]
     # → test-corr-123 그대로 나와야 함
     ```
   - **pytest 회귀**: `backend/tests/test_api.py` 가 `meta` 에 `request_id` 존재를 이미 단언(24,33행). 깨지면 안 됨.
     ```powershell
     $env:PYTHONPATH = "C:\Users\user\lens\backend"
     & "C:\Users\user\lens\.venv\Scripts\python.exe" -m pytest backend/tests/test_api.py -q
     ```
6. **commit**: `feat(be): wire CorrelationIdMiddleware + configure logging at startup (CP228)`

### Step 4 — 기존 로거 structlog 전환 (상태 의존)
파일별로 옛 로거 옆에 structlog 로거를 두고 호출을 옮긴 뒤 옛 줄 제거. **한 파일 = 작은 단위**. 메시지/이벤트 내용은 의미 보존(로그 텍스트 임의 변경 금지, kwargs 로 구조화만).

1. `backend/app/main.py`:
   - 16행 `logger = logging.getLogger("lens.api")` → `logger = structlog.get_logger("lens.api")`.
   - 호출부 `%`-format 을 structlog kwargs 로 변환(예: `logger.info("v1 predictions cache %s: %s", slot, info)` → `logger.info("v1_predictions_cache", slot=slot, info=info)`). **75,81,83,99,138행** 점검.
   - `logger.exception(...)`(138행)은 structlog 에서도 `logger.exception(...)` 동일 사용 가능(format_exc_info 로 트레이스 포함).
2. `backend/app/services/parquet_store.py`:
   - 12행 `import logging` 제거 가능 여부 확인, 18행 → `logger = structlog.get_logger("lens.parquet_store")`.
   - 63/69/104행 호출 구조화.
3. `backend/app/repositories/market_repo.py`:
   - 3행/14행 → structlog. 60행 `logger.warning(...)` 구조화.
4. `backend/app/services/local_market_svc.py` (현재 **로거 없음**):
   - 파일 상단에 `import structlog` + `logger = structlog.get_logger("lens.local_market")` 추가.
   - 폴백 진입/미스 핵심 1~2곳에만 `logger.warning`/`logger.info` 최소 추가(예: parquet 미존재로 빈 결과 반환하는 분기). **과한 로그 금지**.
5. **검증**(파일 단위로 옮길 때마다):
   - `python -c "import app.main"` 등 import 깨짐 없음.
   - 백엔드 기동 후 `parquet_store` 로드 유발 엔드포인트 1회 호출 → 로그 라인에 `request_id`(merge_contextvars 효과)가 박혀 나오는지 stdout 에서 확인.
   - 전체 pytest:
     ```powershell
     $env:PYTHONPATH = "C:\Users\user\lens\backend"
     & "C:\Users\user\lens\.venv\Scripts\python.exe" -m pytest backend/tests -q
     ```
6. **commit**: `refactor(be): migrate stdlib loggers to structlog (CP228)`

### Step 5 — request_id 전파 최종 확인 + 옛 코드 제거 정리
1. 자체 `request_id_middleware` 축소본이 여전히 필요한지 재검토:
   - `request.state.request_id` 를 읽는 지점이 남아있으면(http.py 등) **유지**(축소본).
   - 전부 contextvars 로 대체 가능하면 자체 미들웨어 **제거**하고 `http.py:build_meta` 가 `correlation_id.get()` 을 읽도록 변경 — **단 이는 인터페이스/응답 schema 무변경 범위 내에서만**. 변경이 응답 schema 에 영향 주면 하지 말고 축소본 유지.
2. **request_id 가 service/repo 로그에 박히는지 end-to-end 확인**:
   - 기동 후 `parquet_store` 또는 `local_market_svc` 경로를 타는 요청을 헤더 `X-Request-Id: e2e-check-1` 로 호출.
   - stdout 로그에서 해당 서비스 로거 라인의 `request_id` == `e2e-check-1` 인지 확인.
   - **누락된 경로가 하나라도 있으면 차단 트리거** (미들웨어 순서 의심).
3. **commit**(정리 있으면): `refactor(be): finalize request_id propagation, drop legacy bits (CP228)`

---

## 인터페이스 보존

- **API 응답 schema 무변경**: `data` / `meta` / `error` 구조 그대로. `meta.request_id` 키/값 의미 유지. 이 CP는 로깅만 추가.
- **함수 signature 무변경**: `fetch_price_rows`, `fetch_indicator_rows`, `fetch_stocks`(local_market_svc), `get_raw`/`require`/`stats`(parquet_store), `request_id_middleware` 의 호출 형태 유지. (자체 미들웨어 내부 구현은 축소 가능하나 시그니처/등록 방식은 동일하게.)
- **응답 헤더 `X-Request-Id` 유지**: 키 대소문자(`X-Request-Id`) 그대로. 클라이언트가 의존할 수 있음.
- 바꿔야 하는 상황이 생기면(예: header_name 을 `X-Request-ID` 로 바꿔야 라이브러리가 동작) → **호출자 영향 분석 후 차단 보고**. 임의 변경 금지.

---

## 성공 기준 (측정 가능)

| 항목 | 현재 | 목표 |
|---|---|---|
| structlog 로거 보유 backend/app 소스 파일 | 3 (stdlib) | 4 (structlog: main, parquet_store, market_repo, local_market) |
| 로그 출력 포맷 | plain `%`-string | JSON(prod) / Console(dev), 각 라인에 `timestamp`+`level`+`event` |
| 로그 라인 `request_id` 포함 | 없음 | service/repo 로그 포함 요청 컨텍스트 전 라인에 `request_id` 전파 |
| 응답 schema snapshot diff (CP223) | — | **0** |
| pytest (`backend/tests`) | 통과(N개) | **회귀 0** (N개 그대로 통과) |
| mypy 신규 error | — | **0 추가** |
| 예상 시간 | — | 3~4시간 |

> N(현재 통과 테스트 수)는 Step 0 에서 `pytest backend/tests -q` 1회 실행해 베이스라인으로 기록하고, 종료 시 동일해야 한다.
> tsc / screenshot diff 항목은 **해당 없음**(프론트 미변경).

---

## 검증

각 Step 검증은 위에 인라인. 종합 게이트(마지막에 1회):

```powershell
# 0) 베이스라인 (작업 시작 전 1회 — N 기록)
$env:PYTHONPATH = "C:\Users\user\lens\backend"
& "C:\Users\user\lens\.venv\Scripts\python.exe" -m pytest backend/tests -q

# 1) import 무결성
& "C:\Users\user\lens\.venv\Scripts\python.exe" -c "import app.main; from app.core.logging import configure_logging; print('import ok')"

# 2) 전체 테스트 회귀 0
& "C:\Users\user\lens\.venv\Scripts\python.exe" -m pytest backend/tests -q

# 3) mypy 신규 에러 0 (mypy 설치돼 있을 때만)
& "C:\Users\user\lens\.venv\Scripts\python.exe" -m mypy backend/app/core/logging.py backend/app/main.py backend/app/middleware/request_id.py 2>$null

# 4) 기동 + request_id end-to-end (8001)
#   - 헤더 없이: meta.request_id 비어있지 않음 + 응답헤더 X-Request-Id 존재
#   - 헤더 X-Request-Id: e2e-check-1 → 응답헤더 echo + stdout 로그 라인 request_id=e2e-check-1
(Invoke-WebRequest "http://127.0.0.1:8001/api/v1/health/live" -Headers @{ "X-Request-Id"="e2e-check-1" } -UseBasicParsing).Headers["X-Request-Id"]
```

기대 결과:
- (2) 베이스라인과 동일 개수 통과, 실패 0.
- (4) 헤더 echo 정상, **service/repo 로그 라인에 `request_id`** 가 보임.
- 응답 body 의 `data`/`meta`/`error` 구조가 CP223 스냅샷과 **byte-identical**(snapshot diff 0).

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **CP223 스냅샷이 레드이거나 못 찾음** → 안전망 없이 진행 불가. 시작 전 멈추고 보고.
2. **응답 schema snapshot diff 발생** → 로깅만 추가했는데 응답이 바뀜 = 동작 변경됨. 원인 규명 전 진행 금지. 보고.
3. **미들웨어 순서로 request_id 가 일부 경로에서 누락** → 예: 예외 핸들러 경로, GZip 압축 응답, CORS preflight, parquet 미스 폴백 경로 중 한 곳이라도 로그/응답에 request_id 빠지면 보고. (CorrelationId 가 CORS 보다 바깥이면 일부 응답에서 컨텍스트 set 전에 처리됨 → 누락 위험.)
4. **기존 pytest 가 로깅 변경으로 깨짐** → 특히 `test_api.py` 의 `request_id in meta` 단언(24,33행). 1개라도 새로 실패하면 보고.
5. **자체 `request_id_middleware` 와 CorrelationIdMiddleware 가 request_id 를 서로 덮어써 값이 흔들림**(요청마다 다른 두 값) → 보고.
6. **헤더 키 충돌**(`X-Request-Id` vs 라이브러리 기본 `X-Request-ID`)로 echo 가 깨지거나 중복 헤더가 나감 → 보고.
7. **stdlib 로그 브리지 실패로 기존 `logging` 라인이 사라지거나 이중 출력** → 보고.
8. **환경변수 누락으로 기동 실패**(`LENS_USE_LOCAL_SNAPSHOTS`/`PYTHONPATH` 등) → 검증 환경 문제. 보고.
9. **사용자 직접 수정 파일을 덮어써야 하는 상황** → revert/덮어쓰기 전 멈추고 보고.

---

## ADR

완료 후 **ADR 디렉토리가 없으면 생성**: `docs/adr/` (현재 미존재 확인됨 — 이 CP가 첫 ADR).

작성: `docs/adr/0018-structlog-asgi-correlation-id.md` (200~300단어).
기록할 것: **왜 contextvars 기반(asgi-correlation-id)인가** — ASGI 단일 이벤트루프에서 thread-local 이 요청 격리를 깨므로 ContextVar 필수, structlog `merge_contextvars` 로 request_id 가 service/repo 로그까지 자동 전파되는 구조. 헤더키 `X-Request-Id` 호환 결정, dev=Console/prod=JSON 분기 기준 env, 자체 `request_id_middleware` 축소/유지 결정, Sentry 미설치라 자동 연동 skip(설치 시 후속) 메모.

---

## 자가 점검 결과 양식 (완료 후 채울 것)

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ______
- **[구조 결함]** PASS / WARN / FAIL — 사유: ______
- **[모델 영향]** PASS / WARN / FAIL — 사유: ______ (관측성 변경이라 모델/학습 영향 없어야 함 → 기대 PASS)

---

## 산출물

- 변경 파일 목록:
  - `backend/requirements.txt` (deps 2줄 추가)
  - `backend/app/core/logging.py` (신규)
  - `backend/app/main.py` (configure 호출 + CorrelationIdMiddleware + 로거 전환)
  - `backend/app/middleware/request_id.py` (축소 또는 제거)
  - `backend/app/services/parquet_store.py` (structlog 전환)
  - `backend/app/repositories/market_repo.py` (structlog 전환)
  - `backend/app/services/local_market_svc.py` (structlog 로거 신설 + 최소 로그)
  - `docs/adr/0018-structlog-asgi-correlation-id.md` (신규)
- `docs/cp228_report.md` 작성(요구 / 한 일 / 결정 / 후속, 필요한 만큼만). 후속에는 ai/eval·collector·db 로거 전환, Sentry 설치 시 correlation→transaction 연동을 적는다.
