# ADR-0027: Alembic + async 마이그레이션 골격

Status: Accepted (skeleton — 실 마이그레이션은 Supabase Pro 결제 후)
Date: 2026-06-04
CP: CP236d

## 결정

`backend/migrations/`에 Alembic 골격 (`env.py`, `script.py.mako`, `versions/`)을 작성하고 `backend/alembic.ini`에 설정을 박는다. 기존 `backend/db/scripts/ensure_runtime_schema.py` (현 운영 러너)는 손대지 않는다 — Strangler: 옛것 보존, 새것 공존. 실 마이그레이션은 결제 후 별도 CP에서 시작.

## 골격 구조

- `backend/alembic.ini`: `script_location=migrations` / `prepend_sys_path=..` / `file_template=YYYY-MM-DD_slug` / `sqlalchemy.url` 비움 (env.py 조립).
- `backend/migrations/env.py`: async 패턴. `async_engine_from_config(... poolclass=NullPool)` + `connection.run_sync(do_run_migrations)`. URL은 `ALEMBIC_DATABASE_URL` 또는 `DB_HOST/...`에서 조립.
- `backend/migrations/script.py.mako`: 기본 alembic 템플릿.
- `backend/migrations/versions/2026-06-04_initial_placeholder.py`: revision 1장 (upgrade/downgrade `pass`). 골격이 실제로 revision을 만들 수 있다는 증명용.
- `backend/db/requirements-crawler.txt`: `alembic==1.18.4` + `asyncpg==0.30.0` 핀 추가 (결제 후 활성).

## autogenerate 비활성

`target_metadata = None`. ORM `MetaData`가 없으므로 autogenerate 동작 불가. 결제 후 ORM 도입 시 `from app.models import Base; target_metadata = Base.metadata`로 교체.

## URL 환경변수 조립

`alembic.ini`의 `sqlalchemy.url`은 비워둠 — 비밀번호 하드코딩 금지. `env.py._build_async_url()`이 `ALEMBIC_DATABASE_URL` (full DSN) → `DB_HOST/...` 조립 → `RuntimeError` 순으로 처리.

## 옛 코드 보존

`backend/db/scripts/ensure_runtime_schema.py`는 현 운영 러너로 그대로. 결제 후 별도 CP에서 caller 이전 + 옛 제거. 이 CP에서 손대지 않음 (Strangler 1단계).

## 실 마이그레이션 작성 금지

이 CP는 골격만. 결제/정산 테이블, 기존 테이블 Alembic 이관 전부 보류. `versions/2026-06-04_initial_placeholder.py`도 `pass` 본문. 결제 후 실 스키마 변경 시 본 ADR + `docs/db_migration_conventions.md` 규약 적용.

## 관련

- ADR-0013: 직접연결 5432 (CP236a).
- ADR-0026: async 세션 골격 (CP236b).
- `docs/db_repository_guide.md`: N+1 / from_attributes (CP236c).
- `docs/db_migration_conventions.md`: 파일명/reversible/정적 데이터/enum/autogenerate 규약.
