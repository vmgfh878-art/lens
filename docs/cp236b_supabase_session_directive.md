# CP236b SQLAlchemy 2.0 async 세션 안전 패턴 골격 (Directive)

> 실행자(새 Claude Code 세션)는 이 문서 하나만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지. 막히면 §차단 트리거 양식으로 즉시 보고.

---

## 역할 고정

- **모드**: `code` (구현 + 같은 턴 자가 점검).
- **권한**: 코드 수정 · 로컬 검증(import smoke / pytest / ruff / mypy)만.
- **금지**:
  - 새 학습 / 새 calibration / DB write / 운영 parquet 덮어쓰기.
  - **실제 Supabase(Postgres) 호출 · 실연결 검증** — 이 CP는 **준비 골격만**. 실검증은 사용자 Supabase Pro 결제 후.
  - 사용자가 직접 수정한 파일 revert.
  - 기존 Supabase 코드(`backend/app/db.py`, `market_repo.py`, `collector/**`) **제거·시그니처 변경**. 살아있는 read-path 건드리면 안 됨.
- **자가 점검**: Plan v3 정합 / 구조 결함 / 모델 영향 (양식은 맨 끝).
- **커밋 메시지**: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). 활성화: `.\.venv\Scripts\Activate.ps1`
- **백엔드 기동**(검증 시): `scripts\start_demo.ps1` 또는 직접
  `python -m uvicorn app.main:app --host 127.0.0.1 --port 8011`
  (반드시 `backend/` 가 `PYTHONPATH` 에 있어야 함. import 루트는 `app.*` — 예: `app.main:app`, `from app.db import ...`. `start_demo.ps1` 가 이 worktree 에 없으면 직접 uvicorn 명령 사용.)
- **프론트**(이 CP 무관): `npm run dev`.
- **포트 충돌 회피**: 8000 점유 가능 → 검증 서버는 **18011 같은 빈 포트** 사용. 기동했으면 검증 후 프로세스 반드시 종료(`Stop-Process`).
- **중요**: 이 CP 는 **백엔드를 띄울 필요가 사실상 없다**. import smoke + pytest 로 충분. 서버 기동은 "기존 라이브 경로가 안 깨졌나" 최종 확인용 옵션.

---

## 진단 (근거)

조사 출처: `docs/refactoring_master_plan.md` §3.6 (코드 전수조사 + 현업 베스트프랙티스), `docs/cp221_237_refactoring_runbook.md` §0·§2. 실제 코드 직접 확인: `backend/app/db.py`(전체 66줄 읽음), `backend/app/main.py`(전체 147줄 읽음), `backend/requirements.txt`.

**현재 상태 (사실):**

1. `backend/app/db.py` (현재 **66줄**) 는 **전부 Supabase REST 클라이언트** 기반이다. SQLAlchemy / asyncpg / 세션 개념이 **전혀 없다**.
   - L4: `from supabase import Client, create_client`
   - L14 `supabase_is_configured()`, L25 `get_supabase() -> Client`, L38 `reset_supabase_client()`, L44 `check_supabase_ready()`.
   - 즉 DB 접근이 PostgREST(HTTP) 경유라 **트랜잭션 경계 · 커넥션 풀 · 세션 수명** 개념이 부재. 직접 Postgres(SQLAlchemy async)로 붙일 때 필요한 안전 패턴이 아무 데도 박혀있지 않다.

2. `backend/requirements.txt` (현재 14줄) 에 **`sqlalchemy` / `asyncpg` 핀이 없다** (L1~14: fastapi/uvicorn/supabase/pandas/pyarrow/python-dotenv/numpy/scikit-learn/statsmodels/httpx/wandb). → 골격 모듈이 **top-level 에서 `import sqlalchemy` 하면 ImportError 로 서버가 죽는다.** 골격은 반드시 **지연 import(함수 안 또는 try/except ImportError 가드)** 여야 한다. 이게 이 CP의 핵심 안전 제약.

3. `backend/app/main.py` (현재 **147줄**) 는 DB 세션을 전혀 와이어링하지 않는다.
   - L66 `@app.on_event("startup")` 1개 (v1 predictions 캐시 lazy 로드 게이트)만 존재. **lifespan / engine dispose / get_db 의존성 등록 없음.**
   - 라우터 6개(`health, stocks, ai, admin, v1_predictions, strategies`)는 `Depends(get_db)` 를 **하나도 쓰지 않는다**(현재 DB 접근은 `get_supabase()` 직접 호출).

4. `app.db` / `backend.app.db` 를 import 하는 **살아있는 호출자 9곳** (Grep 확인):
   - `backend/app/repositories/market_repo.py:7` `from app.db import get_supabase`
   - `backend/app/routers/v1/admin.py:11` `from app.db import supabase_is_configured`
   - `backend/app/routers/v1/health.py:6` `from app.db import check_supabase_ready`
   - `backend/collector/repositories/base.py:10`, `backend/collector/pipelines/preflight.py:13` `from backend.app.db import get_supabase`
   - `backend/app/tests` 2곳 (`test_api.py:8`, `test_product_prediction_history_api.py:10`) `reset_supabase_client`
   - `backend/db/scripts/export_parquet.py:17`, `test_connection.py:17`
   - → **이 9곳 중 단 하나도 깨지면 안 된다.** 그래서 새 세션 골격은 **`db.py` 를 수정하지 않고 새 파일에 둔다.**

**왜 지금 박나 (§3.6 근거):**
- pooled 포트 6543(transaction mode) + asyncpg = `prepared statement already exists`. 우리 규모 권장 해결은 **직접연결 5432**(PgBouncer 우회). transaction mode 강제 시 `NullPool` + `statement_cache_size=0`.
- SQLAlchemy 2.0 async 필수 안전 규칙(이 CP가 골격으로 박을 것):
  - **세션 = 요청당 1개 = 트랜잭션 경계.** `get_db` 의존성에서 `async with SessionLocal() as session: yield session`.
  - **`expire_on_commit=False` 필수.** 안 하면 commit 후 ORM 속성 접근 시 async 컨텍스트에서 **동기 lazy refresh I/O → `MissingGreenlet` / DetachedInstance** 에러.
  - **명시적 rollback.** `try/except/rollback/raise` 없으면 부분쓰기 silent 불일치.
  - **`pool_pre_ping=True`**(죽은 연결 감지) + **`pool_recycle`**(서버 idle timeout 전 재활용).

요약: **실제 Postgres async 안전 패턴이 코드에 0줄.** 결제 후 붙일 때 즉흥 코딩하면 위 함정에 그대로 빠진다 → 미리 **주석 달린 골격**으로 박아 며칠 절약. 단 **로컬 서빙 경로(parquet/Supabase read)에는 0 영향**이어야 한다.

---

## 선행 의존

- **없음** (Supabase 준비 트랙 CP236a~d 는 안전망 트랙과 독립, 언제든 가능 — 런북 §2).
- 단 **권장 직렬 순서상 CP236a 다음**(런북 §2.70). CP236a 가 async **engine/`AsyncEngine` 골격 + DB URL 설정 골격**을 만들 예정.
  - **CP236a 산출물(`backend/app/session.py` 의 engine 골격)이 이미 있으면**: 이 CP는 그 파일에 `SessionLocal` + `get_db` 를 **이어서** 추가한다(중복 engine 생성 금지).
  - **CP236a 가 아직 없으면(이 worktree 기준 `backend/app/session.py` 부재 확인됨)**: 이 CP가 engine 골격까지 **포함**해서 새로 만든다(Step 1 참조). CP236a 가 나중에 와도 같은 파일을 확장하도록 한 군데에 모은다. **engine 을 두 번 만들지 않는다.**
  - 시작 전 `backend/app/session.py` 존재 여부를 먼저 확인하고, 위 둘 중 어느 경로인지 §자가 점검에 1줄 기록.

---

## 범위

**포함:**
- 새 파일 `backend/app/session.py` 에 SQLAlchemy 2.0 async **세션 안전 패턴 골격 + 상세 주석**:
  1. async engine 골격(`create_async_engine`, `pool_pre_ping=True`, `pool_recycle`, 직접연결 5432 / transaction mode 분기 주석).
  2. `SessionLocal` (`async_sessionmaker`, **`expire_on_commit=False`**).
  3. `get_db()` FastAPI 의존성(`async with ... yield`, **명시적 rollback**).
- 모든 SQLAlchemy import 는 **지연/가드**(미설치 환경에서도 `import app.session` 가 죽지 않음).
- `backend/requirements.txt` 에 `sqlalchemy` / `asyncpg` 핀을 **주석으로** 추가(설치는 결제 후). 실제 설치/활성 금지.

**제외 (절대 건드리지 않음):**
- `backend/app/db.py` 수정 — Supabase 코드 보류(런북 §0.7). **0줄 변경.**
- `main.py` 에 `get_db` / engine **와이어링**(라우터 `Depends(get_db)` 부착, lifespan 추가) — 골격만, 활성 금지.
- ORM 모델 정의 / 테이블 매핑 / Alembic(→ CP236d) / N+1 eager-load 가이드(→ CP236c).
- 실제 DB 연결 · 마이그레이션 · `SUPABASE_*` / `DATABASE_URL` 환경변수로 실접속.

---

## Sub-step (Strangler Fig, 작은 단위)

> 이 CP는 "골격 신설"이라 기존 코드를 옮기는 strangler 가 아니라 **additive**다. 그래도 한 Step = 한 revert 단위 = 한 커밋으로 작게 쪼갠다. 각 Step 끝에서 **`import app.session` 가 깨지지 않음**(미설치 환경 기준)을 반드시 확인한다.

### Step 1 — async engine 골격 (지연 import, 미설치 안전)

- **(선행)** `backend/app/session.py` 존재 확인. 없으면 신설. CP236a 가 만든 게 있으면 그 파일을 연다.
- 파일 상단에 모듈 docstring: "CP236b — SQLAlchemy 2.0 async 세션 안전 패턴 골격. **현재 비활성**(sqlalchemy/asyncpg 미설치, DATABASE_URL 미설정). 실연결 검증은 결제 후. 로컬 서빙 경로 무영향."
- **지연/가드 import 패턴**(핵심):
  ```python
  try:
      from sqlalchemy.ext.asyncio import (
          AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine,
      )
      _SQLALCHEMY_AVAILABLE = True
  except ImportError:  # sqlalchemy 미설치(결제 전 기본 상태) — 골격만 박혀 있어도 import 가 죽지 않게.
      _SQLALCHEMY_AVAILABLE = False
  ```
- `DATABASE_URL` 을 `os.environ.get("DATABASE_URL")` 로만 읽고(없으면 `None`), **모듈 로드 시 engine 을 즉시 만들지 않는다**(lazy). `_engine: AsyncEngine | None = None` + `get_engine()` 함수 안에서 최초 1회 생성.
- `get_engine()` 안 `create_async_engine(...)` 인자에 **주석으로** 안전 규칙 박기:
  - `pool_pre_ping=True`  # 죽은 커넥션 감지(Supabase idle drop 대비)
  - `pool_recycle=1800`   # 서버 idle timeout 전 재활용(초)
  - 직접연결 **5432** 사용 주석 + "transaction mode(6543) 강제 시 `poolclass=NullPool` + `connect_args={"statement_cache_size": 0}` (prepared statement 충돌 회피)" 주석.
- `_SQLALCHEMY_AVAILABLE` 또는 `DATABASE_URL` 이 없으면 `get_engine()` 은 `RuntimeError("DB 비활성: 결제 후 sqlalchemy/asyncpg 설치 + DATABASE_URL 설정")` raise(호출되지 않는 한 무해).
- **검증 + 커밋**:
  - `python -c "import app.session"` (backend cwd / PYTHONPATH=backend) → **에러 0**.
  - `ruff check backend/app/session.py` → 0.
  - 커밋: `CP236b: add async engine skeleton (lazy, import-safe)`

### Step 2 — `get_db` 의존성 골격

- `get_db()` async generator 의존성 추가(주석으로 "FastAPI `Depends(get_db)` 용. **아직 어느 라우터에도 부착 안 함**"):
  ```python
  async def get_db() -> "AsyncIterator[AsyncSession]":
      # 세션 = 요청당 1개 = 트랜잭션 경계.
      async with SessionLocal() as session:   # SessionLocal 은 Step 3 에서 정의
          yield session
  ```
  - 이 시점엔 `SessionLocal` 이 아직 없으니, Step 2 에서는 `get_db` 본문을 `raise RuntimeError("DB 비활성 — CP236b 골격")` 로 두고 `# TODO(CP236b Step3): SessionLocal 연결` 주석 + **위 active 패턴을 docstring/주석으로** 명시. (Step 3 에서 실제 `async with` 로 교체.)
  - 또는 Step 2·3 을 합치고 싶으면 합쳐도 됨(단 커밋은 분리 권장). **합칠 경우 §자가 점검에 1줄 기록.**
- `AsyncIterator` 는 `typing`/`collections.abc` 에서 import(타입만, 런타임 무해).
- **검증 + 커밋**:
  - `python -c "import app.session"` → 에러 0.
  - 커밋: `CP236b: add get_db dependency skeleton`

### Step 3 — `expire_on_commit=False` + pre_ping 확정 (`SessionLocal`)

- `SessionLocal` 정의(lazy factory):
  ```python
  def _make_session_factory():
      return async_sessionmaker(
          bind=get_engine(),
          expire_on_commit=False,   # 필수: commit 후 속성 접근 시 동기 lazy I/O(MissingGreenlet) 방지
          class_=AsyncSession,
      )
  ```
  - 모듈 레벨에서 즉시 호출하지 말 것(미설치 환경에서 `get_engine()` 이 raise). `SessionLocal` 도 lazy(`get_session_factory()` 안에서 캐시) 로 둔다.
  - `expire_on_commit=False` 옆에 위 주석 **반드시** 유지(진단 §3.6 근거).
- Step 2 의 `get_db` 본문을 실제 패턴으로 교체:
  ```python
  async with get_session_factory()() as session:
      yield session
  ```
- **검증 + 커밋**:
  - `python -c "import app.session"` → 에러 0. (engine 은 lazy 라 호출 안 되면 raise 안 함.)
  - `mypy backend/app/session.py` → **새 error 0**(미설치라 sqlalchemy 심볼은 `# type: ignore[...]` 최소 사용 가능, 사유 주석).
  - 커밋: `CP236b: SessionLocal expire_on_commit=False + lazy factory`

### Step 4 — 명시적 rollback 패턴 + requirements 주석 핀

- `get_db` 에 **명시적 rollback** 패턴 박기(주석 포함):
  ```python
  async with get_session_factory()() as session:
      try:
          yield session
      except Exception:
          await session.rollback()   # 명시 rollback 없으면 부분쓰기 silent 불일치
          raise
      # 정상 종료 시 commit 은 호출 라우터가 명시적으로(자동 commit 안 함).
  ```
  - "자동 commit 안 함 — 라우터가 `await session.commit()` 명시" 정책을 주석으로 박는다(read-only 조회는 commit 불필요).
- `backend/requirements.txt` 끝에 **주석 핀** 추가(설치 금지, 결제 후 주석 해제):
  ```
  # --- CP236b: Supabase 직접연결(async) 준비. 결제 후 주석 해제 + 설치 ---
  # sqlalchemy[asyncio]==2.0.32
  # asyncpg==0.29.0
  ```
  (정확한 버전은 결제 시점 최신 안정 핀으로. 지금은 주석이라 빌드 무영향.)
- **검증 + 커밋**:
  - `python -c "import app.session"` → 에러 0.
  - `ruff check backend/app/session.py` → 0.
  - `pytest backend/app/tests -q`(또는 기존 test 경로) → **기존 테스트 전부 통과(회귀 0)**. (session.py 는 아무도 import 안 하므로 영향 0이어야 정상.)
  - 커밋: `CP236b: explicit rollback pattern + commented sqlalchemy/asyncpg pins`

---

## 인터페이스 보존

- `backend/app/db.py` 의 함수 시그니처(`supabase_is_configured`, `get_supabase`, `reset_supabase_client`, `check_supabase_ready`) **불변**. 0줄 수정.
- 어떤 **API 응답 schema 도 변경 없음**(새 라우터/의존성 부착 안 함).
- `main.py` 라우터 등록 · `@app.on_event` **불변**.
- 새 모듈 `app.session` 의 공개 심볼(`get_engine`, `get_db`, `SessionLocal`/`get_session_factory`)은 **현재 호출자 0** — 인터페이스 신설이라 보존 대상 없음.
- **만약** CP236a 가 만든 `session.py` 의 기존 공개 시그니처를 바꿔야 하는 상황이 오면 → **즉시 중단**, 호출자 영향 분석 후 §차단 트리거 양식으로 보고.

---

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 |
|---|---|---|
| `backend/app/session.py` 골격 | 없음(파일 부재) | 존재. async engine + `get_db` + `SessionLocal` + rollback 패턴 박힘 |
| `expire_on_commit=False` 주석 근거 | 0 | 1 (진단 §3.6 인용 주석) |
| `import app.session` (미설치 환경) | — | 에러 0 |
| 기존 pytest(`backend/app/tests`) | 통과 N개 | 동일 N개 통과, **회귀 0** |
| `backend/app/db.py` 변경 | — | **0줄** |
| `main.py` 변경 | — | **0줄** |
| ruff (`session.py`) | — | 0 |
| mypy 새 error | — | **0 추가** |
| requirements 실제 설치 | supabase 등 | **변동 0**(핀은 주석) |
| 예상 시간 | — | 약 1.5시간 |

---

## 검증

각 Step 후 + CP 종료 시 실행(PowerShell, backend 를 PYTHONPATH 에).

```powershell
cd C:\Users\user\lens
$env:PYTHONPATH = "C:\Users\user\lens\backend"

# 1) import 안전(미설치 환경에서도 죽지 않아야 함) — 가장 중요
.\.venv\Scripts\python.exe -c "import app.session; print('session import OK')"

# 2) lint
.\.venv\Scripts\python.exe -m ruff check backend/app/session.py

# 3) 타입(새 error 0)
.\.venv\Scripts\python.exe -m mypy backend/app/session.py

# 4) 기존 테스트 회귀 0 (session.py 는 아무도 안 쓰므로 영향 0이어야 정상)
.\.venv\Scripts\python.exe -m pytest backend/app/tests -q

# 5) db.py / main.py 미변경 확인
git diff --stat backend/app/db.py backend/app/main.py   # 출력 비어 있어야 함
```

**기대 결과**: (1) `session import OK` 출력 + 에러 0. (2) ruff 0. (3) mypy 새 error 0. (4) 기존 테스트 통과 수 유지. (5) `db.py`/`main.py` diff 비어 있음.

(옵션) 라이브 경로 무영향 최종 확인 — 빈 포트로 띄워 health 만 확인 후 즉시 종료:
```powershell
$env:LENS_FORCE_LOCAL = "1"
$p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","18011" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 6
try { (Invoke-WebRequest "http://127.0.0.1:18011/" -UseBasicParsing -TimeoutSec 8).StatusCode } catch { "ERR: $($_.Exception.Message)" }
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
```
기대: `200`. (session 골격이 서버 기동을 깨지 않았는지 확인. 깨지면 = 지연 import 가드 실패 → 차단.)

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.** (런북 §5 양식.)

1. **`import app.session` 가 ImportError/그 외로 실패** → 지연 import 가드가 깨졌거나 top-level 에서 sqlalchemy 를 import 함. **이게 가장 위험**(서버 전체 기동 실패로 번짐). 즉시 멈춤.
2. **서버 기동 시 health 가 200 이 아님** → 골격이 라이브 서빙 경로에 영향. 멈춤.
3. **기존 pytest 가 1개라도 실패(회귀)** → 무관해야 할 골격이 무언가를 깼다. 멈춤.
4. **`backend/app/db.py` 또는 `main.py` 에 diff 가 생김** → 보류 대상/와이어링 금지 위반. 되돌리고 멈춤.
5. **`pip install` / `sqlalchemy`·`asyncpg` 실제 설치를 하게 됨** → 이 CP는 주석 핀까지만. 설치는 결제 후. 멈춤.
6. **실제 DB(Postgres/Supabase)에 연결을 시도하게 됨**(DATABASE_URL 실접속, engine 실제 생성·ping) → 실연결 검증은 결제 후. 멈춤.
7. **CP236a 산출물과 충돌**(engine 중복 정의, 기존 공개 시그니처 변경 요구) → 합치는 판단이 애매하면 멈추고 보고.
8. **골격을 활성화하려고 라우터에 `Depends(get_db)` 를 부착**하거나 lifespan 을 추가하게 됨 → 범위 밖. 멈춤.

> 실연결 / 풀 동작 / rollback 실제 검증은 **전부 결제 후**. 이 CP에서 "실제로 돌려서 확인" 유혹이 오면 그게 차단 신호다.

---

## ADR

완료 후 `docs/adr/0026_async_db_session_pattern.md` 1장(200~300단어) 작성.
(확인 결과 `docs/adr/` 디렉토리·`0013` 파일 **부재** → fold-in 불가, 신규 `0026` 으로 작성하고 디렉토리도 생성.)

**무엇을 기록**: SQLAlchemy 2.0 async 세션을 "요청당 1세션=트랜잭션 경계"로 잡은 결정 / `expire_on_commit=False` 를 **필수**로 둔 이유(async commit 후 동기 lazy I/O = MissingGreenlet) / 명시적 rollback·`pool_pre_ping`·`pool_recycle` 채택 / 직접연결 5432 vs transaction mode 6543(+NullPool+statement_cache_size=0) 트레이드오프 / **지금은 비활성 골격**이며 실검증은 Supabase Pro 결제 후라는 점.

---

## 자가 점검 결과 양식

작성 완료 시 아래를 채워 보고한다.

- **CP236a 경로**: ( engine 골격 이미 있음 → 확장 / 없음 → 이 CP가 신설 ) — _______
- **[Plan v3 정합]**: PASS / WARN / FAIL — 사유: ______ (read-path fidelity 무영향, EODHD/parquet 경로 불변인가)
- **[구조 결함]**: PASS / WARN / FAIL — 사유: ______ (지연 import 가드로 미설치 안전한가, engine 중복 없는가, lazy 생성인가)
- **[모델 영향]**: PASS / WARN / FAIL — 사유: ______ (추론/calibration/parquet 0 영향 — DB 골격은 모델 경로와 무관)

---

## 산출물

- **변경 파일**:
  - `backend/app/session.py` (신설 또는 확장)
  - `backend/requirements.txt` (주석 핀 추가)
  - `docs/adr/0026_async_db_session_pattern.md` (신설)
- **리포트**: `docs/cp236b_report.md` — 섹션: 요구 / 한 일 / 결정 / 후속(결제 후 활성화 체크리스트: 핀 주석 해제 → 설치 → DATABASE_URL 직접연결 5432 → 라우터 `Depends(get_db)` 부착 → 실연결·rollback 검증). 필요한 만큼만 간결히.
