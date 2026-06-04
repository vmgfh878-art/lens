# ADR-0013: Supabase 직접 Postgres 연결은 포트 5432 (PgBouncer 우회)

Status: Accepted (skeleton only — 실연결은 Supabase Pro 결제 후 활성)
Date: 2026-06-04
CP: CP236a

## 결정

Supabase 직접 Postgres 연결 시 **포트 5432 (direct connection)** 사용. PgBouncer/Supavisor의 pooled 6543 (transaction mode) 포트는 **금지**.

## 함정

pooled 6543 (transaction mode) + asyncpg / asyncpg-style prepared statement 캐시 = `prepared statement "__asyncpg_..." already exists` 에러. transaction mode는 prepared statement를 연결 간 유지하지 않기 때문.

## 우리 규모 권장

직접 5432로 붙으면 PgBouncer를 우회해 함정 자체가 사라진다. Supabase free tier도 5432 직접 접속 허용. 우리 부하 규모(분당 수십 요청 이하)에서 PgBouncer 풀링은 불필요.

## 6543 강제 시 (불가피한 케이스)

- SQLAlchemy `poolclass=NullPool` (앱 풀 비활성화 → PgBouncer가 모든 풀링 담당)
- `connect_args={"statement_cache_size": 0, "prepared_statement_cache_size": 0}` (asyncpg 기준)
- psycopg2면 `statement_cache_size` 키는 무시되지만 `NullPool`은 필요
- 앱 풀이 필요하면 인스턴스당 5~10 (PgBouncer pool 한도 분배)

## 골격 위치

- `backend/app/db.py` — `DIRECT_DB_PORT_DEFAULT` / `POOLED_TXN_DB_PORT` / `APP_POOL_SIZE_PER_INSTANCE` 상수 + `recommended_connect_args(port)` 헬퍼. 미사용·미실행.
- `ai/preprocessing.py` — `_postgres_dsn()` 안에 6543 감지 시 `RuntimeWarning` 1회 + `_postgres_engine()` 주석으로 6543 전환 시 풀 변경 규약 명시. 현재 동작(psycopg2 + 5432 direct)은 무변경.
- `render.yaml` — `SUPABASE_DB_HOST/PORT/USER/PASSWORD/NAME` `sync: false` 키 5개 추가. 값 주입은 결제 후.

## 실행 안 함

실연결 / asyncpg 도입 / SQLAlchemy 2.0 async 전환은 본 CP 범위 밖. CP236b/CP-RF-11에서 별도 처리. 본 ADR는 결제 후 활성 시점에 "왜 5432인가"를 코드/문서에서 즉시 찾을 수 있도록 박제하는 것이 목적.
