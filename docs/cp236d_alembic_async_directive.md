# CP236d DB(준비) — Alembic async 마이그레이션 골격 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지, 확인 우선.
> **이 CP는 "준비(scaffolding)" CP다. 실제 마이그레이션은 작성하지 않는다.** 골격 + 컨벤션 문서만 만든다.

---

## 역할 고정

- **모드**: `code` (구현 모드). 설계 토론이 아니라 지시받은 골격을 직접 만든다.
- **권한**: 코드 수정 · 로컬 검증(파일 생성, `alembic` CLI dry 실행, lint, import 확인)만.
- **금지** (하나라도 건드리면 즉시 중단·보고):
  - 새 모델 학습 / 새 calibration 실행 금지.
  - **DB write 금지** — 실제 DB에 `alembic upgrade`/`downgrade`/DDL 실행 금지. 이 CP는 골격만이며 실 스키마를 건드리지 않는다.
  - **Supabase 호출 금지** (`backend/app/db.py`의 `get_supabase()` 등 호출 금지).
  - **사용자가 직접 수정한 파일 revert 금지.** 특히 `backend/db/scripts/ensure_runtime_schema.py`는 현행 운영 스키마 러너다. **삭제·이관·내용 변경 금지.** 이 CP는 그 옆에 Alembic 골격을 "공존"시키기만 한다.
  - 첫 마이그레이션 파일 작성 금지(결제 후 스키마 확정 시까지 보류 — 아래 진단·차단 참조).
- **자가 점검**: Plan v3 정합 / 구조 결함 / 모델 영향 3축으로 끝에 보고(양식 하단).
- **커밋 메시지**: 간결. 구조/준비 커밋과 기능 커밋 분리. 끝에:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## 환경

- **워킹 디렉토리**: `C:\Users\user\lens`
- **venv**: `.venv` (Python **3.10.0**, torch 2.11.0+cu128). 활성화: `.\.venv\Scripts\Activate.ps1`
- **이미 설치되어 있는 것**(확인 완료, `.venv/Lib/site-packages`):
  - `alembic==1.18.4` (★ venv엔 있으나 repo엔 초기화 안 됨 — 그래서 이 CP가 필요)
  - `SQLAlchemy==2.0.29`
  - `greenlet==3.4.0` (← `connection.run_sync()`가 내부적으로 요구. 있음. OK)
  - `psycopg2-binary==2.9.9` (sync 드라이버, `backend/db/requirements-crawler.txt`)
- **설치 안 되어 있는 것**(확인 완료): `asyncpg` 없음. → async env.py가 실제로 돌려면 async 드라이버 필요. **Step 1에서 requirements에 핀만 추가하고, 설치/실행은 차단 트리거 참조**(실 DB 접속 불가 환경이면 import 골격만 검증).
- **백엔드 기동**(검증에 필요할 때만): `scripts\start_demo.ps1`은 **이 repo에 존재하지 않는다.** 실제 기동은 README 기준:
  ```powershell
  cd backend
  uvicorn app.main:app --reload --port 8000
  ```
  (`backend/app/main.py:14`에서 `app.routers.v1` import. 이 CP는 백엔드 기동이 필수가 아니다 — 골격은 런타임에 import되지 않는다.)
- **프론트**(이 CP 무관): `cd frontend; npm run dev`
- **포트 충돌 회피**: 8000번이 떠 있으면 새로 띄우지 말 것. 이 CP는 서버 기동 없이 완료 가능.
- **DB 접속 env**(참고, 직접 호출 금지): sync 직결은 `DB_HOST` / `DB_PORT`(기본 5432) / `DB_USER` / `DB_PASSWORD` / `DB_NAME` / `DB_SSLMODE`(기본 `require`) — `backend/db/scripts/test_connection.py:74-82` 참조. Supabase는 `SUPABASE_URL` / `SUPABASE_KEY`.

---

## 진단 (근거)

**문제: 스키마 변경 이력이 코드로 버전관리되지 않는다. 현행 방식은 손으로 쌓는 idempotent DDL 스크립트 하나뿐이다.**

조사 출처(직접 Read/Grep 확인):

1. **Alembic 미초기화.**
   - `Glob backend/**/alembic*` → 매치 0건. `Glob backend/**/*.ini` → 0건.
   - repo 어디에도 `alembic.ini` / `env.py` / `migrations/` 없음. 단, venv에는 `alembic 1.18.4`가 이미 깔려 있다(`.venv/Lib/site-packages/alembic-1.18.4.dist-info`). 즉 **패키지는 있는데 프로젝트 초기화만 안 된 상태** → 이 CP의 작업은 "init + async 개조"다.

2. **현행 스키마 러너 = 손으로 쌓는 단일 스크립트.**
   - `backend/db/scripts/ensure_runtime_schema.py` (총 **414줄**).
   - 핵심은 `RUNTIME_SCHEMA_STATEMENTS` 리스트(**18–308번째 줄**) — 50여 개 raw SQL 문자열을 순서대로 `cursor.execute()`로 실행(`main()`, 333–410줄). 전부 `ADD COLUMN IF NOT EXISTS` / `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` / `DO $$ ... pg_constraint 존재검사 ... $$` 패턴의 **수동 idempotent** DDL이다.
   - 연결은 raw `psycopg2`(`_connect()`, 311–330줄), **sync**. SQLAlchemy도 async도 쓰지 않는다.
   - 문제점: (a) up만 있고 **down(되돌리기)이 없다**, (b) 변경 순서/버전 식별자가 없어 "지금 DB가 어느 리비전인가"를 알 수 없다, (c) 신규 컬럼이 생길 때마다 리스트 끝에 `ALTER ... IF NOT EXISTS`를 덧붙이는 방식이라 누적 부채가 커진다. → 정식 마이그레이션 도구(Alembic)의 부재가 근본 원인.

3. **현재 스키마엔 네이티브 PG enum이 없다(중요 — Step 3 근거).**
   - `ensure_runtime_schema.py`에서 상태/구분 값은 전부 **`CHECK (col IN (...))`** 로 표현한다. 예:
     - `timeframe VARCHAR(4) NOT NULL CHECK (timeframe IN ('1D', '1W', '1M'))` (197, 226, 292줄)
     - `status VARCHAR(20) ... CHECK (status IN ('completed', 'failed_nan', 'failed_quality_gate'))` (216–217, 268, 284–285줄)
     - `band_mode VARCHAR(20) ... DEFAULT 'direct'` (202, 272줄)
   - 즉 **`CREATE TYPE ... AS ENUM`은 아직 0개.** 결제/정산 스키마(주문 상태 등)가 들어올 때 enum을 새로 도입할 가능성이 높고, 그때 `ALTER TYPE ... ADD VALUE`가 Alembic 기본 트랜잭션과 충돌한다(PostgreSQL은 `ADD VALUE`를 트랜잭션 블록 안에서 금지). → 그래서 **지금 컨벤션 문서에 enum 주의를 미리 못박는다.**

4. **async 드라이버 부재.**
   - `Grep asyncpg backend/` → 0건. 현재 모든 DB 접근은 sync(`psycopg2`) 또는 Supabase REST. async env.py 패턴(`async_engine_from_config` + `run_sync`)을 쓰려면 `asyncpg`가 필요한데 미설치다. → Step 1에서 requirements 핀 추가.

**결론**: Alembic이 아예 없어서 스키마가 코드로 추적되지 않는다. 단, **지금은 결제 후 스키마가 확정되지 않았다.** 그래서 "골격 + 컨벤션"만 깔고 첫 마이그레이션은 의도적으로 비워 둔다(빈 `upgrade()/downgrade()` 1장만 placeholder로). 이렇게 하면 결제 스키마가 확정되는 순간 `alembic revision`만 돌리면 바로 굴러간다.

---

## 선행 의존

**없음.**

- 근거: 이 CP는 신규 디렉토리(`backend/migrations/`)와 신규 파일만 추가하고, 기존 코드의 함수 signature / API schema / props를 **건드리지 않는다.** 런타임 import 경로(`backend/app/main.py`, 라우터)에 연결되지 않으므로 백엔드 characterization 스냅샷(CP223)·프론트 characterization(CP230)의 그린 여부와 독립이다.
- 단, **`ensure_runtime_schema.py`를 절대 손대지 않는다는 전제** 하에서만 독립이다. 그 파일을 이관·삭제하려는 충동이 생기면 그건 별도 CP(결제 후)이며 이 CP 범위 밖이다 → 차단.

---

## 범위

### 포함
- `backend/migrations/` 디렉토리에 Alembic 골격 생성(`env.py`, `script.py.mako`, `versions/` 빈 폴더).
- `backend/alembic.ini` 생성 및 `script_location` / `file_template`(날짜-slug) 설정.
- `env.py`를 **async 패턴**으로 개조(`async_engine_from_config` + `connection.run_sync(do_run_migrations)`).
- DB URL을 **환경변수에서 조립**(하드코딩 금지). `asyncpg` 핀을 requirements에 추가.
- 컨벤션 문서 `docs/db_migration_conventions.md` 작성(파일명 규칙, reversible 원칙, 정적 데이터 원칙, enum `ADD VALUE` 주의).
- 첫 리비전은 **빈 placeholder 1장**만(스키마 확정 전이므로 `upgrade()/downgrade()` 본문 `pass`). 이건 "골격이 실제로 revision을 만들 수 있다"는 것만 증명하는 용도.
- ADR 1장(`docs/adr/0027-alembic-async-migrations.md`).

### 제외 (명시)
- **실 스키마 마이그레이션 작성 금지.** 결제/정산 테이블, 기존 테이블의 Alembic 이관 전부 보류(결제 후 별도 CP).
- **`backend/db/scripts/ensure_runtime_schema.py` 수정·삭제·이관 금지.** 현행 운영 러너로 그대로 둔다(Strangler: 옛것 보존, 새것 공존).
- **실제 DB 접속/`alembic upgrade`/`downgrade` 실행 금지** (DB write 금지 규칙).
- **Supabase 관련 일체 보류.** Supabase는 마이그레이션 대상이 아니며 이 CP에서 호출하지 않는다.
- Alembic의 autogenerate(모델 메타데이터 비교) 활성화 금지 — 우리는 ORM `MetaData`가 없으므로 `target_metadata = None`으로 둔다(아래 인터페이스 참조). 모델 기반 autogen은 결제 후 ORM 도입 시 검토.

---

## Sub-step (Strangler Fig, 작은 단위)

> 전제: 이 CP는 "새 코드를 옛 코드(`ensure_runtime_schema.py`) **옆에** 추가"하는 것 자체가 Strangler의 1단계다. caller 이전·옛 제거는 **이 CP에서 하지 않는다**(결제 후). 따라서 각 Step은 "추가 → 검증 → 커밋"이며, 한 Step = 한 revert 단위.

### Step 1 — Alembic 골격 초기화 (async env.py)

1. requirements에 핀 추가(설치 도구 가용 여부와 무관하게 **선언**부터):
   - `backend/db/requirements-crawler.txt`에 다음 두 줄 추가(파일 끝):
     ```
     alembic==1.18.4
     asyncpg==0.30.0
     ```
     - `alembic`은 venv에 1.18.4가 이미 있으니 그 버전으로 핀. `asyncpg`는 미설치 → 0.30.0 핀(SQLAlchemy 2.0.29 + Python 3.10 호환). 핀만 추가하고 **이 CP에서 굳이 `pip install` 강행하지 않는다**(네트워크/실행 불가 시 차단 트리거 참조).
2. 골격 디렉토리/파일을 **직접 작성**(=`alembic init`이 만드는 것과 동형이되 async로). `alembic init`을 돌려도 되지만, 돌릴 경우 생성된 `env.py`를 아래 async 형태로 **반드시 교체**한다. 최종 산출:
   - `backend/alembic.ini`
   - `backend/migrations/env.py`
   - `backend/migrations/script.py.mako`
   - `backend/migrations/versions/` (빈 폴더 + `.gitkeep`)
3. `backend/alembic.ini` 핵심 설정(아래 최소 필드만, 나머지 로깅 섹션은 alembic 기본 템플릿 유지):
   - `[alembic]` 섹션:
     - `script_location = migrations`
     - `prepend_sys_path = ..`  (← `backend/`의 부모를 sys.path에 → 필요 시 `backend.*` import 가능)
     - `file_template = %%(year)d-%%(month).2d-%%(day).2d_%%(slug)s`  (← **YYYY-MM-DD_slug** 파일명 규칙. ini라 `%`는 `%%`로 이스케이프)
     - `sqlalchemy.url =`  ← **비워 둔다.** URL은 env.py가 환경변수로 조립(ini에 비밀번호 하드코딩 금지).
4. `backend/migrations/env.py`를 **async 패턴**으로 작성. 골자(실행자는 아래 구조를 그대로 구현):
   ```python
   import asyncio
   import os
   from logging.config import fileConfig

   from alembic import context
   from sqlalchemy import pool
   from sqlalchemy.engine import Connection
   from sqlalchemy.ext.asyncio import async_engine_from_config

   config = context.config

   if config.config_file_name is not None:
       fileConfig(config.config_file_name)

   # ORM 메타데이터 없음 → autogenerate 비활성. 결제 후 ORM 도입 시 교체.
   target_metadata = None


   def _build_async_url() -> str:
       """환경변수에서 async DB URL을 조립한다. 비밀번호 하드코딩 금지."""
       url = os.environ.get("ALEMBIC_DATABASE_URL")
       if url:
           return url
       host = os.environ.get("DB_HOST")
       name = os.environ.get("DB_NAME")
       user = os.environ.get("DB_USER")
       password = os.environ.get("DB_PASSWORD")
       port = os.environ.get("DB_PORT", "5432")
       if not all([host, name, user, password]):
           raise RuntimeError(
               "Alembic DB URL을 만들 수 없습니다. "
               "ALEMBIC_DATABASE_URL 또는 DB_HOST/DB_NAME/DB_USER/DB_PASSWORD를 설정하세요."
           )
       sslmode = os.environ.get("DB_SSLMODE", "require")
       return (
           f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
           f"?ssl={'require' if sslmode == 'require' else 'prefer'}"
       )


   def run_migrations_offline() -> None:
       """--sql 모드: 실제 접속 없이 SQL만 출력."""
       context.configure(
           url=_build_async_url(),
           target_metadata=target_metadata,
           literal_binds=True,
           dialect_opts={"paramstyle": "named"},
       )
       with context.begin_transaction():
           context.run_migrations()


   def do_run_migrations(connection: Connection) -> None:
       context.configure(connection=connection, target_metadata=target_metadata)
       with context.begin_transaction():
           context.run_migrations()


   async def run_async_migrations() -> None:
       configuration = config.get_section(config.config_ini_section, {})
       configuration["sqlalchemy.url"] = _build_async_url()
       connectable = async_engine_from_config(
           configuration,
           prefix="sqlalchemy.",
           poolclass=pool.NullPool,
       )
       async with connectable.connect() as connection:
           # sync 마이그레이션 러너를 async connection 위에서 run_sync로 래핑
           await connection.run_sync(do_run_migrations)
       await connectable.dispose()


   def run_migrations_online() -> None:
       asyncio.run(run_async_migrations())


   if context.is_offline_mode():
       run_migrations_offline()
   else:
       run_migrations_online()
   ```
   - 핵심: `async_engine_from_config(...)` + `await connection.run_sync(do_run_migrations)` — **sync 마이그레이션 러너(`do_run_migrations`)를 async connection으로 감싼다.** 이게 스펙이 요구한 패턴이다.
5. `backend/migrations/script.py.mako`: alembic 기본 mako를 그대로 두되, 상단 docstring 주석에 "**reversible 필수: downgrade를 반드시 채운다. 동적/시점 데이터에 의존하는 마이그레이션 금지**" 한 줄을 넣는다.
6. **검증 후 커밋**:
   - 검증(아래 "검증" 섹션의 V1):
     - `.\.venv\Scripts\python.exe -c "import alembic; import sqlalchemy.ext.asyncio; import greenlet; print('imports ok')"`
     - `python -c "import ast; ast.parse(open('backend/migrations/env.py', encoding='utf-8').read()); print('env.py parses')"`
   - 커밋: `chore(db): Alembic async 마이그레이션 골격 추가 (env.py run_sync 래핑, 빈 versions)` + Co-Authored-By.
   - **revert 단위**: 이 Step 전체(골격 파일들 + requirements 2줄).

### Step 2 — 마이그레이션 컨벤션 문서

1. `docs/db_migration_conventions.md`(신규) 작성. 반드시 포함할 항목:
   - **파일명 규칙**: `YYYY-MM-DD_slug` (예: `2026-07-15_create_orders_table`). `alembic.ini`의 `file_template`이 이 규칙을 강제함을 명시.
   - **reversible 원칙**: 모든 마이그레이션은 `upgrade()`와 `downgrade()`를 둘 다 채운다. down이 불가능한 변경(파괴적 DROP 등)은 PR에서 사유를 적고 사용자 승인 필요.
   - **정적(static) 원칙**: 마이그레이션은 **시점/동적 데이터에 의존 금지**. `datetime.now()`로 분기, 실데이터 행 수에 따른 조건 분기, 외부 API 호출 등 금지. 같은 리비전은 언제 돌려도 같은 스키마를 만들어야 한다(idempotent하진 않아도 deterministic).
   - **현행 `ensure_runtime_schema.py`와의 관계**: 결제 스키마 확정 전까지 운영 스키마는 여전히 그 스크립트가 책임진다. Alembic은 **신규 스키마(결제/정산 등)부터** 적용한다. 기존 테이블의 Alembic 이관은 별도 CP. (Strangler — 옛것 보존, 새것이 신규 영역부터 잠식.)
   - **실행 명령 예시**(참고용, 이 CP에선 실행하지 않음):
     ```powershell
     cd backend
     # 리비전 생성 (빈 골격)
     $env:ALEMBIC_DATABASE_URL="postgresql+asyncpg://..." ; alembic revision -m "create orders table"
     # SQL만 미리보기 (offline, DB 접속 없음)
     alembic upgrade head --sql
     # 실제 적용 (운영자만, 승인 후)
     alembic upgrade head
     ```
   - **enum 주의로 가는 링크 한 줄**(Step 3 내용을 같은 문서 하단 섹션으로).
2. **검증 후 커밋**:
   - 검증: 문서가 위 5개 항목을 모두 포함하는지 `Grep`으로 확인(`파일명`, `reversible`, `정적`, `ensure_runtime_schema`, `--sql`).
   - 커밋: `docs(db): 마이그레이션 컨벤션 문서 추가` + Co-Authored-By.

### Step 3 — PostgreSQL enum `ADD VALUE` 주의 (문서 + placeholder 리비전)

1. `docs/db_migration_conventions.md` 하단에 **"## PostgreSQL enum 주의"** 섹션 추가:
   - 현재 스키마엔 네이티브 enum이 없고 `CHECK (col IN (...))`로 상태값을 표현한다는 사실 명시(근거: `ensure_runtime_schema.py:197,216-217,292`).
   - 결제 스키마에서 enum을 새로 도입할 경우: **`ALTER TYPE ... ADD VALUE`는 트랜잭션 블록 안에서 실행 불가**(PostgreSQL 제약). Alembic은 기본적으로 마이그레이션을 트랜잭션으로 감싸므로(우리 `env.py`의 `context.begin_transaction()`) 충돌한다.
   - 해결 패턴 두 가지를 문서에 적시:
     - (a) 해당 리비전 상단에 **`transactional_ddl`을 끄거나** `op.execute("COMMIT")` 후 `ALTER TYPE ... ADD VALUE` 실행(권장하지 않음, fragile).
     - (b) **권장**: enum 값 추가는 `op.execute(sa.text("ALTER TYPE status ADD VALUE IF NOT EXISTS 'refunded'"))`를 **별도 비-트랜잭션 리비전**으로 분리하고, 그 리비전에 `def upgrade()` 안에서 autocommit 블록을 명시:
       ```python
       with op.get_context().autocommit_block():
           op.execute("ALTER TYPE order_status ADD VALUE IF NOT EXISTS 'refunded'")
       ```
       `autocommit_block()`이 트랜잭션 밖에서 실행해 준다. **이 패턴을 표준으로 못박는다.**
     - `downgrade`에서 enum 값 제거는 PostgreSQL이 지원하지 않음 → enum 추가 리비전의 down은 "no-op + 주석으로 수동 절차 안내"로 둔다(reversible 원칙의 명시적 예외).
2. **placeholder 빈 리비전 1장 생성**(골격이 실제로 동작함을 증명, 단 본문 없음):
   - 파일: `backend/migrations/versions/2026-06-02_initial_placeholder.py` (날짜는 작성 시점으로; file_template 규칙 준수).
   - 내용:
     - `revision`/`down_revision = None`/`branch_labels = None`/`depends_on = None` 헤더.
     - `def upgrade(): pass` + 주석 "# 결제 후 스키마 확정 시 첫 실 마이그레이션으로 교체. 현재는 골격 검증용 빈 리비전."
     - `def downgrade(): pass` + 동일 취지 주석.
   - **이 파일은 빈 placeholder다. 실제 DDL을 넣지 않는다.** (스키마 미확정 → 차단 트리거)
3. **검증 후 커밋**:
   - 검증(아래 V2):
     - `python -c "import ast; ast.parse(open('backend/migrations/versions/2026-06-02_initial_placeholder.py', encoding='utf-8').read()); print('placeholder parses')"`
     - 가능하면(asyncpg/DB 불요) `cd backend; alembic history` 가 placeholder 1건을 인식하는지. **DB 접속이 필요한 `upgrade`/`downgrade`/`current`는 실행 금지.** `history`/`heads`/`--sql` offline만 허용.
   - 커밋: `docs(db): enum ADD VALUE 비-트랜잭션 주의 + 빈 placeholder 리비전` + Co-Authored-By.

---

## 인터페이스 보존

- **기존 함수 signature 변경 0건.** 이 CP는 `backend/db/scripts/ensure_runtime_schema.py`의 `main()`/`_connect()`/`RUNTIME_SCHEMA_STATEMENTS`, `backend/app/db.py`의 `get_supabase()`/`supabase_is_configured()`/`check_supabase_ready()` 등 **어떤 기존 함수도 호출하거나 수정하지 않는다.**
- **API 응답 schema 변경 0건.** 라우터(`backend/app/routers/v1/*`)·`backend/app/main.py`를 건드리지 않는다. 마이그레이션 골격은 런타임 import 그래프에 들어가지 않는다(FastAPI 앱이 `migrations/`를 import하지 않음).
- **props 인터페이스 변경 0건.** 프론트 무관.
- 만약 작업 중 "기존 코드를 고쳐야 굴러간다"는 판단이 서면(예: env.py가 `backend.app.db`를 import해야 한다는 식) → **그것은 인터페이스 침범이다. 즉시 멈추고 호출자 영향 분석을 적어 보고**(차단 트리거). env.py는 자기완결적으로 환경변수만 읽어 URL을 만든다(위 `_build_async_url`).

---

## 성공 기준 (측정 가능)

| 항목 | 시작 | 목표 | 측정 방법 |
| --- | --- | --- | --- |
| Alembic 골격 파일 | 0개 | 4개 존재(`alembic.ini`, `env.py`, `script.py.mako`, `versions/.gitkeep`) | `Glob backend/alembic.ini`, `backend/migrations/**` |
| env.py async 패턴 | 없음 | `async_engine_from_config` + `run_sync` 둘 다 포함 | `Grep "async_engine_from_config"` 1건, `Grep "run_sync"` 1건 |
| 파일명 규칙 | 없음 | `alembic.ini`에 `file_template = ...year...month...day...slug` | `Grep "file_template"` |
| 컨벤션 문서 | 없음 | `docs/db_migration_conventions.md` 존재 + 5개 필수 항목 + enum 섹션 | `Grep` 6개 키워드 |
| placeholder 리비전 | 없음 | `versions/`에 빈 리비전 1장(`upgrade`/`downgrade` `pass`) | `python -c "ast.parse(...)"` |
| import 무결성 | — | `import alembic, sqlalchemy.ext.asyncio, greenlet` 성공 | V1 명령 exit 0 |
| 기존 코드 변경 | — | `ensure_runtime_schema.py` / `app/db.py` / 라우터 diff 0줄 | `git diff --stat`에 해당 파일 없음 |
| pytest 회귀 | 기존 N개 통과 | 회귀 0 (이 CP는 신규 파일만 → 영향 0이어야 함) | `pytest -q` (DB 불요 테스트만; 아래 검증) |
| ADR | 없음 | `docs/adr/0027-alembic-async-migrations.md` 1장 | 파일 존재 |
| 예상 시간 | — | **약 1.5시간** | — |

> mypy/tsc 항목은 **해당 없음**(이 CP는 타입드 런타임 코드 변경이 없고 프론트 무관) → 생략.
> snapshot/screenshot diff는 **해당 없음**(동작·UI 변경 없음). 단 "혹시 발생하면" = 동작 침범 신호 → 차단(아래).

---

## 검증

각 명령은 `C:\Users\user\lens`에서 venv 활성화 후 실행. **DB 접속이 필요한 alembic 명령(`upgrade`/`downgrade`/`current`)은 절대 실행하지 않는다.**

**V1 — 골격 import/parse (Step 1):**
```powershell
.\.venv\Scripts\Activate.ps1
python -c "import alembic, sqlalchemy.ext.asyncio, greenlet; print('imports ok', alembic.__version__)"
# 기대: imports ok 1.18.4
python -c "import ast; ast.parse(open('backend/migrations/env.py', encoding='utf-8').read()); print('env.py parses')"
# 기대: env.py parses
```

**V2 — placeholder 인식 (Step 3, offline only):**
```powershell
python -c "import ast; ast.parse(open('backend/migrations/versions/2026-06-02_initial_placeholder.py', encoding='utf-8').read()); print('placeholder parses')"
# 기대: placeholder parses
# (선택, asyncpg 설치돼 있고 import만 필요한 경우) 골격 인식:
cd backend
alembic history
# 기대: placeholder 1건이 base로 표시. 에러로 죽지 않을 것.
# 주의: alembic history 가 DB 접속을 시도하지 않는다(스크립트 디렉토리만 읽음). upgrade/current 는 금지.
```

**V3 — 컨벤션 문서 필수 항목 (Step 2,3):**
```powershell
# 6개 키워드가 모두 잡혀야 함 (Grep 도구 사용 권장)
#   "YYYY-MM-DD" / "reversible" / "정적" / "ensure_runtime_schema" / "--sql" / "ADD VALUE"
```

**V4 — 기존 코드 불변 + 회귀:**
```powershell
git diff --stat
# 기대: backend/db/scripts/ensure_runtime_schema.py, backend/app/db.py, backend/app/main.py, backend/app/routers/** 가 목록에 없을 것.
pytest -q
# 기대: 기존 통과 테스트 회귀 0. (DB/네트워크 요구 테스트가 원래 skip/xfail이면 그 상태 유지.)
# 신규 골격은 테스트 수집 대상이 아니므로 통과 수가 줄지 않아야 함.
```

---

## 차단 트리거 (중요)

> 아래 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.** 임의 판단으로 우회·강행 금지.

1. **실 스키마가 없어 첫 마이그레이션을 채워야 할 것 같을 때** → 채우지 마라. 골격만 만들고 "결제 후 스키마 확정 시 첫 실 마이그레이션 작성"이라고 placeholder 주석 + 보고서에 명시하고 멈춘다. (이 CP의 의도된 한계다.)
2. **placeholder가 아니라 진짜 DDL을 쓰고 싶어질 때**(예: orders 테이블을 미리 만들려는 충동) → 중단. 스키마 미확정 상태에서 추측 DDL은 부채다.
3. **`ensure_runtime_schema.py`를 고치거나 Alembic으로 이관해야 굴러간다고 판단될 때** → 중단·보고. 그 이관은 별도 CP(결제 후). 이 CP는 공존만.
4. **`git diff --stat`에 기존 런타임 파일이 잡힐 때**(`app/db.py`, `app/main.py`, 라우터, `ensure_runtime_schema.py`) → 인터페이스 침범. 즉시 멈추고 무엇을 왜 건드렸는지 보고.
5. **`asyncpg` 미설치로 `alembic` 명령이 깨질 때** → `pip install`을 무단 강행하지 마라. requirements 핀만 남기고, "설치는 운영자 승인 필요"로 보고. (offline parse 검증 V1/V2로 골격 무결성은 증명 가능하다.)
6. **`import greenlet` 실패** → `run_sync`가 동작 못 함. 중단·보고(현재는 3.4.0 설치 확인됨 → 발생하면 환경 변형 신호).
7. **DB 접속을 시도하는 alembic 명령을 돌리게 됐을 때**(`upgrade`/`downgrade`/`current`/`stamp`) → DB write 위험. 즉시 멈춘다. offline(`--sql`)·`history`/`heads`만 허용.
8. **pytest 통과 수가 줄거나 새 실패가 생길 때** → 신규 파일만 추가했는데 회귀가 났다면 import 부작용 의심. 멈추고 원인 보고.
9. **snapshot/screenshot diff가 발생할 때** → 이 CP는 동작/UI를 바꾸지 않는다. diff = 예상치 못한 동작 변경 = 즉시 중단·보고.
10. **환경변수 누락으로 무언가 기동 실패**(예: env.py가 import 시점에 DB env를 요구) → env.py는 import 시점엔 절대 DB env를 읽지 말아야 한다(읽기는 `run_migrations_online` 실행 시에만). import만으로 실패하면 설계 결함 → 멈추고 수정 후 보고.

---

## ADR

완료 후 **`docs/adr/0027-alembic-async-migrations.md`** 1장(200~300단어) 작성.
- 디렉토리 `docs/adr/`는 현재 없음 → 생성한다.
- 기록할 결정: **"스키마 변경 도구로 Alembic(async env.py, `run_sync` 래핑)을 채택. 다만 결제 스키마 확정 전까지는 골격만 두고, 운영 스키마는 기존 `ensure_runtime_schema.py`가 계속 책임진다(Strangler). 파일명 `YYYY-MM-DD_slug`, reversible·정적 원칙, enum `ADD VALUE`는 `autocommit_block`으로 트랜잭션 밖 처리."** Context(왜 필요했나: up-only 수동 DDL의 한계)·Decision·Consequences(첫 마이그레이션 보류, asyncpg 도입 필요)를 담는다.

---

## 자가 점검 결과 양식

작업 종료 시 아래를 채워 보고한다(빈칸 금지, 사유 1–2줄).

- **[Plan v3 정합]** PASS / WARN / FAIL — 사유: ___
  (체크 관점: fidelity 우선·EODHD 유지 등 Plan v3 본체와 충돌하는가. 이 CP는 스키마 도구 준비이므로 모델/밴드 본체엔 영향 없어야 정상.)
- **[구조 결함]** PASS / WARN / FAIL — 사유: ___
  (체크 관점: env.py가 import 시점에 부작용을 갖는가, 기존 러너와의 책임 경계가 모호한가, 하드코딩 비밀번호 유무.)
- **[모델 영향]** PASS / WARN / FAIL — 사유: ___
  (체크 관점: 학습/추론/calibration 경로에 닿는가. 닿지 않아야 정상 → 닿으면 FAIL.)

---

## 산출물

- **변경/신규 파일 목록**(예상):
  - `backend/alembic.ini` (신규)
  - `backend/migrations/env.py` (신규, async)
  - `backend/migrations/script.py.mako` (신규)
  - `backend/migrations/versions/.gitkeep` (신규)
  - `backend/migrations/versions/2026-06-02_initial_placeholder.py` (신규, 빈 리비전)
  - `backend/db/requirements-crawler.txt` (`alembic`, `asyncpg` 2줄 추가)
  - `docs/db_migration_conventions.md` (신규)
  - `docs/adr/0027-alembic-async-migrations.md` (신규)
- **보고서** `docs/cp236d_report.md`: 요구 / 한 일 / 결정(채택 패턴·보류 사유) / 후속(결제 스키마 확정 시 첫 마이그레이션 + asyncpg 설치 + `ensure_runtime_schema.py` 이관 검토)을 필요한 만큼만.
