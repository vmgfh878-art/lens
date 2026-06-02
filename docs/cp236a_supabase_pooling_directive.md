# CP236a DB (Supabase 재연결 준비 — 풀링 함정 회피 설계) (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 CP 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 내릴 수 있어야 한다.
> **이 CP는 "준비(설계/골격/문서)" 전용이다. 실제 Supabase 연결·실연결 테스트는 사용자 Supabase Pro 결제 후로 명시 보류한다.**

---

## 역할 고정

- **모드**: `code` (구현 모드). 코드를 직접 고치고 자가 점검만 보고한다.
- **권한**: 코드 수정 · 로컬 검증(lint/pytest/백엔드 기동/health probe)만.
- **금지**:
  - 새 학습(training) 실행 금지.
  - 새 calibration 실행 금지.
  - DB write 금지 (Supabase·Postgres 어디로도 INSERT/UPDATE/UPSERT 금지).
  - **Supabase 호출 금지** (REST `.execute()` / psycopg2 connect / SQLAlchemy `engine.connect()` 실행 금지). 이 CP는 코드/주석/설정 골격만 박는다.
  - 사용자가 직접 수정한 파일 revert 금지.
- **자가 점검(필수, 같은 턴에)**: [Plan v3 정합] · [구조 결함] · [모델 영향] 세 축으로 점검해 본문 끝 양식에 PASS/WARN/FAIL + 사유 기재.
- **커밋 메시지**: 간결. 끝에 한 줄:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python 3.10.0, torch 2.11.0+cu128). Python 호출은 `.venv\Scripts\python.exe`.
- **백엔드 기동**: `scripts\start_demo.ps1` 또는 직접
  `\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (cwd=`backend`, `PYTHONPATH=backend`).
  - `start_demo.ps1`은 `LENS_USE_LOCAL_SNAPSHOTS=1`, `LENS_LOCAL_SNAPSHOT_DIR=...\data\parquet`, `MARKET_DATA_PROVIDER=yfinance`로 띄운다 → **Supabase 없이 로컬 parquet 경로로 동작**. 이 CP 검증은 이 상태를 그대로 쓴다.
  - health: `GET http://127.0.0.1:8000/api/v1/health/live` → 200.
- **프론트**: `npm run dev` (cwd=`frontend`, 포트 3000). **이 CP는 프론트 무관 — 띄울 필요 없음.**
- **검증용 포트 충돌 피하기**: 이미 8000/3000을 점유한 dev 서버가 있으면 새로 띄우지 말고 기존 프로세스의 health를 그대로 probe. 새로 띄울 거면 8000이 비었는지 먼저 확인.

---

## 진단 (근거)

조사 출처: 아래 파일들을 본 CP 작성 시 직접 Read/Grep으로 확인. 라인 번호는 확인 시점 기준.

### (A) `backend/app/db.py` — **REST 클라이언트**. 현재 비활성. (현재 66줄)
- `from supabase import Client, create_client` (L4) — 이건 **supabase-py REST 클라이언트(PostgREST over HTTP)**다. asyncpg/PgBouncer prepared-statement 함정은 **여기 직접 적용 안 됨** (HTTP 경유라 Postgres 세션 풀링과 무관).
- `supabase_is_configured()` (L14-22): `LENS_FORCE_LOCAL`이 truthy면 무조건 `False` 반환(L20-21) → 강제 로컬 parquet. 그 외엔 `SUPABASE_URL`+`SUPABASE_KEY` 존재 여부.
- `get_supabase()` (L25-35): 전역 `_client` 캐시 + `create_client(url, key)` (L34).
- `check_supabase_ready()` (L44-65): `client.table("stock_info").select(...).execute()`로 실제 읽기 probe. **이 CP에서 실행 금지(Supabase 호출)**.
- **결론**: db.py 자체는 REST라 prepared-statement 함정과 무관. 다만 "Supabase 연결 전략"의 **단일 정본 위치**로 삼아 직접연결(asyncpg/psycopg2/SQLAlchemy) 경로의 포트·풀링 규약을 여기에 문서화·골격화한다.

### (B) `ai/preprocessing.py` — **진짜 함정이 사는 곳**. 직접 Postgres + SQLAlchemy. (해당 함수 영역)
- `from sqlalchemy import create_engine` (L18).
- `_postgres_dsn()` (L355-364): DSN 조립.
  - 포트: `port = os.environ.get("SUPABASE_DB_PORT") or os.environ.get("DB_PORT", "5432")` (**L362**) → 기본 5432(직접연결, 안전). 하지만 누군가 `SUPABASE_DB_PORT=6543`을 넣으면 **무방비로 transaction-mode pooled 포트로 붙는다.**
  - 반환: `postgresql://{user}:{password}@{host}:{port}/{db_name}?sslmode={sslmode}` (L364) → 드라이버 미지정(=psycopg2). future asyncpg 전환 시 함정 노출.
- `_postgres_engine()` (L367-373): `create_engine(dsn)` (**L372**). **풀 설정 전무** — `poolclass`/`connect_args`/`pool_pre_ping` 없음. `_ENGINE_CACHE`(L73)에 DSN 키로 캐시.
- 엔진 소비처: **L477, L533, L1521** (`engine = _postgres_engine()`). 즉 6543 pooled + (장차 asyncpg) 조합이면 이 3곳에서 `prepared statement "__asyncpg_..." already exists`가 터진다.
- **결론**: 포트 가드 + 풀링/connect_args 규약을 박을 1차 대상. 단 **현재 동작(psycopg2 + 5432 기본)을 바꾸지 않는 선에서** 골격·가드·문서만.

### (C) `backend/db/scripts/test_connection.py` — psycopg2 직접 연결. (현재 100줄)
- `psycopg2.connect(... port=int(os.environ.get("DB_PORT", "5432")) ...)` (**L76**) — 기본 5432(안전). 6543 주입 시 psycopg2는 prepared statement를 기본으로 안 만들어 당장은 덜 위험하나, 규약 일관성을 위해 동일 가드 적용 후보.

### (D) `render.yaml` — 배포 환경변수. (현재 47줄)
- L18-21: `SUPABASE_URL`/`SUPABASE_KEY` 만 `sync: false`로 선언. **직접연결용 `SUPABASE_DB_*` 키 부재** → 재연결 시 누락으로 조용히 로컬 폴백되거나 포트 미지정.
- 주석 cron 블록(L26-46)도 동일하게 `SUPABASE_DB_*` 없음.

### 함정 요약 (master plan §3.6, `docs/refactoring_master_plan.md` L136-142 와 일치)
- **pooled 6543(transaction mode, Supavisor/PgBouncer) + asyncpg = `prepared statement already exists`.** transaction mode는 prepared statement를 연결 간 유지 안 함.
- **우리 규모 권장 해결**: **직접연결 5432**(PgBouncer 우회, 즉시 해결).
- transaction mode를 꼭 써야 하면: SQLAlchemy `NullPool` + `connect_args`에 `statement_cache_size=0`(asyncpg) / `prepared_statement_cache_size=0`. PgBouncer 경유 시 앱 풀 인스턴스당 5~10.

---

## 선행 의존

- **없음** (이 CP는 db.py/preprocessing.py에 비활성 코드·주석·골격만 더하는 **additive** 작업이며, 백엔드 구조 분리가 아니다).
- 단, 본 CP의 "성공 기준" 중 `snapshot diff 0` 검증은 **CP223(백엔드 characterization 스냅샷)이 그린일 때만 의미가 있다.**
  - CP223 스냅샷 테스트가 이미 존재하면 그걸로 회귀를 본다.
  - **CP223이 아직 없으면**: 스냅샷 비교는 "해당 없음"으로 처리하고, 대신 §검증의 (V1)(V2)(V3) 폴백(import smoke + DSN 단위 + health probe)으로 무회귀를 입증하고 그 사실을 리포트에 명시한다. (없는 스냅샷을 새로 만들지 말 것 — CP223 범위 침범.)

---

## 범위

### 포함
- `backend/app/db.py`: **Supabase 연결 전략 정본 주석** + **직접연결 설정 골격 함수**(미사용, 실행 안 됨) 추가.
- `ai/preprocessing.py`: `_postgres_dsn()`/`_postgres_engine()`에 **포트 가드 + 풀링/connect_args 규약 골격**을 **현재 동작 보존하며** 주입(기본 5432 유지, 6543 감지 시 경고/안전옵션 경로만 준비).
- `render.yaml`: 직접연결용 `SUPABASE_DB_*` env 키를 **주석 + `sync: false` 선언**으로 골격화(값 주입 X).
- `docs/adr/0013-supabase-port-5432-not-6543.md` 신규 작성.
- `docs/cp236a_report.md` 작성.

### 제외 (절대 건드리지 않음)
- **실제 Supabase 연결 / 실연결 테스트** → 사용자 Supabase Pro 결제 후. 이 CP에서 금지.
- **asyncpg 도입 / SQLAlchemy 2.0 async 전환** → 본 CP는 동기 psycopg2 경로 유지. async 세션·`expire_on_commit`·Alembic은 CP-RF-11(별도) 영역. 여기선 **주석으로 미래 규약만 남기고 코드 전환은 안 함.**
- `LENS_FORCE_LOCAL` / `LENS_USE_LOCAL_SNAPSHOTS` 로컬 폴백 로직 변경.
- 사용자가 직접 수정한 파일의 revert.
- 새 env 값(`SUPABASE_DB_HOST` 등) 실제 채우기 — 골격(키 선언 + 주석)만.

---

## Sub-step (Strangler Fig, 작은 단위)

각 Step = 한 revert 단위. 각 Step 끝에 commit + 검증.
패턴: **옛 코드 옆에 새(미사용) 코드 공존 → (이 CP에선 caller 이전·옛 제거 없음, 실연결이 결제 후라서)**. 즉 본 CP는 Strangler의 "씨앗 심기" 단계만 수행한다.

### Step 1 — `db.py` 연결 전략 정본 문서화 + 직접연결 설정 골격(미사용)
1. `backend/app/db.py` 상단(모듈 docstring 또는 `_client` 선언 위)에 **연결 전략 정본 주석 블록**을 추가. 내용:
   - REST(supabase-py)와 직접연결(Postgres) 두 경로가 있고, **함정은 직접연결+pooled 6543에만** 적용됨을 명시.
   - **규약**: 직접연결은 포트 **5432(direct)**. 6543(transaction pooled)은 금지(불가피하면 NullPool + `statement_cache_size=0`).
   - 실연결은 **사용자 Supabase Pro 결제 후** 활성.
2. db.py에 **미사용·미실행** 헬퍼를 추가(실행 안 됨, 골격):
   ```python
   # --- 직접 Postgres 연결 설정 골격 (CP236a) ---
   # 주의: 실제 연결/엔진 생성은 Supabase Pro 결제 후 활성. 지금은 호출되지 않는다.
   DIRECT_DB_PORT_DEFAULT = "5432"          # direct connection (PgBouncer 우회)
   POOLED_TXN_DB_PORT = "6543"              # transaction mode (Supavisor/PgBouncer) — 직접 asyncpg와 충돌
   APP_POOL_SIZE_PER_INSTANCE = 5           # PgBouncer 경유 시 인스턴스당 5~10 권장 상한의 하한

   def recommended_connect_args(port: str) -> dict:
       """포트별 안전 connect_args. 6543이면 prepared-statement 캐시를 끈다.

       호출자 없음(골격). 결제 후 직접연결 도입 시 사용.
       """
       if str(port) == POOLED_TXN_DB_PORT:
           # asyncpg 기준 키. psycopg2면 무시되며 NullPool 병행 필요.
           return {"statement_cache_size": 0, "prepared_statement_cache_size": 0}
       return {}
   ```
   - 함수/상수 이름은 위와 동일하게. **기존 함수 signature 변경 금지.**
3. **검증**: `ruff check backend/app/db.py`(설정 있으면) + import smoke(§검증 V1). **백엔드 health probe(§검증 V3)로 로컬 경로 무영향 확인.**
4. **commit**: `chore(db): document supabase direct-connection strategy + connect_args skeleton (CP236a)`

### Step 2 — `preprocessing.py` 포트 가드 + 풀링 규약(현재 동작 보존)
1. `ai/preprocessing.py` `_postgres_dsn()`(L355-364) **반환 직전**에 포트 가드 추가. **동작 변경 없이** 6543 감지 시 1회 경고만:
   ```python
   if str(port) == "6543":
       import warnings
       warnings.warn(
           "SUPABASE_DB_PORT=6543 (transaction pooled). 직접연결 5432 권장. "
           "6543 유지 시 NullPool + statement_cache_size=0 필요 (CP236a / ADR-0013).",
           RuntimeWarning,
           stacklevel=2,
       )
   ```
   - **기본 5432는 그대로.** DSN 문자열·반환 타입 불변.
2. `_postgres_engine()`(L367-373)에 **주석 규약**을 박되 현재 `create_engine(dsn)`(L372) 동작은 보존:
   - 바로 위에 주석: `# CP236a: 6543(pooled) 전환 시 create_engine(dsn, poolclass=NullPool, connect_args=recommended_connect_args(port)). 현재 5432 direct라 기본 풀 유지.`
   - **여기서 NullPool을 실제로 적용하지 않는다** (현재 psycopg2 + 5432 동작/스냅샷 보존이 최우선). 적용은 결제 후 별 CP.
   - import는 추가하지 않음(미사용 import = lint 경고). `NullPool`은 주석 안에만 언급.
3. **검증**: `ruff check ai/preprocessing.py` + DSN 단위(§검증 V2: 5432→경고 없음, 6543→`RuntimeWarning` 1회, DSN 문자열 동일 형식). **백엔드 health probe.**
4. **commit**: `chore(ai): guard pooled 6543 port + pooling note in postgres dsn/engine (CP236a)`

### Step 3 — `render.yaml` 직접연결 env 키 골격 + ADR + 리포트
1. `render.yaml` `lens-backend` 서비스 `envVars`(L18-21 뒤)에 **주석 + `sync: false` 키**만 추가(값 X):
   ```yaml
      # 직접 Postgres 연결(결제 후 활성). 포트는 5432(direct) — 6543(pooled) 금지. ADR-0013.
      - key: SUPABASE_DB_HOST
        sync: false
      - key: SUPABASE_DB_PORT      # 5432 권장. 6543 금지(PgBouncer transaction mode).
        sync: false
      - key: SUPABASE_DB_USER
        sync: false
      - key: SUPABASE_DB_PASSWORD
        sync: false
      - key: SUPABASE_DB_NAME
        sync: false
   ```
   - **주석 cron 블록(L26-46)은 건드리지 않는다** (비활성 그대로).
2. `docs/adr/0013-supabase-port-5432-not-6543.md` 작성(§ADR).
   - `docs/adr/` 디렉토리는 **존재하지 않으므로 생성**한다.
3. `docs/cp236a_report.md` 작성(§산출물).
4. **검증**: YAML 파싱 확인(§검증 V4). `git diff --stat`로 의도 파일만 변경됐는지.
5. **commit**: `docs(db): render SUPABASE_DB_* skeleton keys + ADR-0013 direct-port (CP236a)`

---

## 인터페이스 보존

- **함수 signature 불변**: `supabase_is_configured()`, `get_supabase()`, `reset_supabase_client()`, `check_supabase_ready()`(db.py), `_postgres_dsn()`, `_postgres_engine()`(preprocessing.py) — 시그니처/반환 타입/반환 형식 전부 그대로.
  - 신규 추가 `recommended_connect_args(port)`는 **호출자 없는 골격**이라 기존 인터페이스에 영향 없음.
- **API 응답 schema 불변**: `admin.py`의 진단 응답에 들어가는 `supabase_is_configured()`(admin.py L163) 값·키 변경 금지.
- **DSN 문자열 형식 불변**: `_postgres_dsn()` 반환 문자열은 `postgresql://...?sslmode=...` 형식 그대로(가드는 경고만, 문자열 미변경).
- **props 인터페이스**: 해당 없음(프론트 무관).
- 만약 위 중 하나라도 바꿔야만 하는 상황이 생기면 → **호출자 영향 분석 후 §차단 트리거에 따라 멈추고 보고.**

---

## 성공 기준 (측정 가능)

| 항목 | 기준 |
|---|---|
| `backend/app/db.py` 줄 수 | 66 → 약 90 이내 (골격 상수+함수+주석 추가분; 목표 ≤ 95) |
| `ai/preprocessing.py` 변경 | 함수 신규 추가 0, 가드/주석만 (`_postgres_dsn`·`_postgres_engine` 영역 한정) |
| 기존 pytest | `backend/tests/` 전부 통과, 회귀 0 (실패 0 추가) |
| snapshot diff | 0 (CP223 스냅샷 존재 시). 미존재 시 "해당 없음" + V1~V3 폴백 통과로 대체 |
| mypy error 추가 | 0 (설정 존재 시. 미존재면 ruff 0 추가) |
| tsc | 해당 없음(프론트 무변경) |
| 로컬 health | `GET /api/v1/health/live` → 200 (변경 전후 동일) |
| 실연결 테스트 | **해당 없음 — Supabase Pro 결제 후로 보류(리포트에 명시)** |
| 예상 시간 | 약 1.5~2시간 |

---

## 검증

> 모든 명령은 cwd=`C:\Users\user\lens`, venv 파이썬 `\.venv\Scripts\python.exe` 기준. **Supabase/Postgres 실연결 호출 금지** — 아래는 전부 오프라인 검증.

**(V0) 변경 범위 확인**
```powershell
git diff --stat
# 기대: backend/app/db.py, ai/preprocessing.py, render.yaml, docs/adr/0013-*.md, docs/cp236a_report.md 만.
```

**(V1) import smoke (db.py 골격이 깨지지 않았는가, Supabase 호출 없이)**
```powershell
$env:PYTHONPATH = "$PWD"
.\.venv\Scripts\python.exe -c "import backend.app.db as d; print('db ok', d.DIRECT_DB_PORT_DEFAULT, d.recommended_connect_args('6543'), d.recommended_connect_args('5432'))"
# 기대: db ok 5432 {'statement_cache_size': 0, 'prepared_statement_cache_size': 0} {}
```

**(V2) DSN 포트 가드 단위 검증 (실연결 없음, 환경변수만 주입)**
```powershell
$env:PYTHONPATH = "$PWD"
$env:SUPABASE_DB_HOST="h"; $env:SUPABASE_DB_USER="u"; $env:SUPABASE_DB_PASSWORD="p"; $env:SUPABASE_DB_NAME="db"
# 5432: 경고 없어야 함
$env:SUPABASE_DB_PORT="5432"
.\.venv\Scripts\python.exe -W error::RuntimeWarning -c "from ai.preprocessing import _postgres_dsn; print(_postgres_dsn())"
# 기대: postgresql://u:p@h:5432/db?sslmode=require  (경고로 인한 비정상 종료 없음)
# 6543: RuntimeWarning 1회 발생해야 함 (-W error로 SystemExit 유도해 발생 확인)
$env:SUPABASE_DB_PORT="6543"
.\.venv\Scripts\python.exe -W error::RuntimeWarning -c "from ai.preprocessing import _postgres_dsn; print(_postgres_dsn())"
# 기대: RuntimeWarning 발생(비정상 종료=가드 동작 확인). 종료 후 env 정리.
Remove-Item Env:SUPABASE_DB_HOST,Env:SUPABASE_DB_USER,Env:SUPABASE_DB_PASSWORD,Env:SUPABASE_DB_NAME,Env:SUPABASE_DB_PORT
```
- 주의: `_postgres_engine()`은 **호출하지 말 것**(실연결 시도). `_postgres_dsn()`만.

**(V3) 로컬 서빙 경로 무영향 (Supabase 없이 기존 모드)**
```powershell
# 이미 8000에 dev 서버가 떠 있으면 그걸 probe. 없으면 start_demo.ps1로 띄움.
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health/live" -UseBasicParsing -TimeoutSec 5 | Select-Object -ExpandProperty StatusCode
# 기대: 200
# (선택) 진단 라우트로 로컬 parquet probe가 그대로인지:
# Invoke-WebRequest .../api/v1/admin/diagnostics ... → market_probes loaded 유지, supabase_is_configured=false
```

**(V4) YAML 파싱**
```powershell
.\.venv\Scripts\python.exe -c "import yaml,sys; d=yaml.safe_load(open('render.yaml',encoding='utf-8')); print('yaml ok', d['services'][0]['name'])"
# 기대: yaml ok lens-backend
```

**(V5) 기존 테스트 회귀**
```powershell
$env:PYTHONPATH = "$PWD\backend"
.\.venv\Scripts\python.exe -m pytest backend/tests -q
# 기대: 기존과 동일 pass, 실패 0 추가. (수집 단계에서 실연결 시도하는 테스트 없음.)
```

**(V6) lint (설정 존재 시)**
```powershell
.\.venv\Scripts\python.exe -m ruff check backend/app/db.py ai/preprocessing.py
# 기대: 신규 에러 0. 미사용 import 경고 0 (NullPool은 주석에만).
```

---

## 차단 트리거 (중요)

> **다음 상황이면 즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **(V3) 로컬 health가 200이 아니거나, 백엔드가 변경 후 기동 실패** = 로컬 서빙 경로 깨짐. → 즉시 멈춤. 이 CP는 로컬 무영향이 대전제다.
2. **(V5) 기존 pytest가 변경 전 대비 1개라도 추가 실패**, 특히 수집/import 단계에서 **DB 연결을 실제로 시도**하는 정황 = 골격이 실행 경로로 새어 들어감. → 멈춤.
3. **snapshot diff 발생**(CP223 스냅샷 존재 시 어느 항목이든 diff ≠ 0) = 동작이 바뀜. 골격/주석이 동작을 바꾼 것. → 멈춤.
4. **(V2)에서 5432인데 RuntimeWarning이 뜸**, 또는 **DSN 문자열 형식이 기존과 달라짐**(드라이버 prefix/sslmode 등) = 인터페이스 위반. → 멈춤.
5. 작업 중 **`_postgres_engine()`/`create_client()`/`psycopg2.connect()`/`.execute()`가 실제로 호출되어 네트워크로 나가려 함**(연결 시도 로그/타임아웃) = 범위 위반(실연결 금지). → 즉시 멈춤.
6. db.py/preprocessing.py의 **기존 함수 signature를 바꿔야만** 골격이 들어가는 상황 = 인터페이스 보존 위반. → 멈추고 호출자 영향 분석 보고.
7. **실제 DB 없이 검증이 불가능한 항목**(실연결 round-trip, 6543 실제 prepared-statement 에러 재현 등)에 도달 = **그 항목은 검증하지 말고**, "사용자 Supabase Pro 결제 후"로 명시해 리포트에 남기고 나머지만 진행. (이건 멈춤이 아니라 **명시 보류**다 — 단, 임의로 결제 전 실연결을 시도하면 #5에 걸려 멈춤.)
8. `render.yaml`에 **실제 시크릿 값을 박게 되는** 상황(키만 선언해야 함) = 멈춤.

---

## ADR

완료 후 **`docs/adr/0013-supabase-port-5432-not-6543.md`** 1장(200~300단어) 작성.
- 기록 내용: **왜 직접연결 5432를 기본으로 고정하고 transaction-pooled 6543을 금지하는가.** 컨텍스트(supabase-py REST는 무관 / 직접연결만 함정), 결정(5432 direct = PgBouncer 우회로 `prepared statement already exists` 원천 차단), 대안(6543 + NullPool + `statement_cache_size=0`/`prepared_statement_cache_size=0` + 인스턴스당 풀 5~10 — 우리 규모엔 과함), 결과(우리 트래픽 규모에서 직접연결 충분, 풀 고갈 시 재검토), 상태(Proposed — 실연결은 Supabase Pro 결제 후 Accepted 전환).
- `docs/adr/` 디렉토리가 없으면 생성. (다른 ADR 번호와 충돌하지 않도록 0013 확정 전 `docs/adr/` 목록 확인 — 없으면 0013로 시작.)

---

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다.

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ___
  (체크 포인트: 밴드 본체·fidelity 우선·EODHD 유지·backtest cost 등 Plan v3 핵심을 이 변경이 건드리지 않는가. 본 CP는 DB 연결 골격이라 모델 수치/데이터 소스 무관해야 PASS.)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ___
  (체크 포인트: 골격이 실행 경로로 새지 않는가, 미사용 import/죽은 코드가 lint를 깨지 않는가, 연결 전략 정본이 db.py 한 곳에 모였는가.)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ___
  (체크 포인트: 학습/추론/calibration 수치에 0 영향. preprocessing DSN 동작 불변 → 모델 입력 불변이어야 PASS.)

---

## 산출물

1. **변경 파일 목록**:
   - `backend/app/db.py` (연결 전략 정본 주석 + 직접연결 설정 골격)
   - `ai/preprocessing.py` (`_postgres_dsn` 포트 가드 + `_postgres_engine` 풀링 주석)
   - `render.yaml` (`SUPABASE_DB_*` 골격 키 + 주석)
   - `docs/adr/0013-supabase-port-5432-not-6543.md` (신규)
2. **`docs/cp236a_report.md`** — 요구 / 한일 / 결정 / 후속(필요한 만큼만).
   - **후속에 반드시 명시**: "실연결·6543 에러 재현·NullPool 실적용·async 전환은 Supabase Pro 결제 후 별 CP(CP-RF-11 계열)에서 수행. 본 CP는 골격/가드/문서까지."
