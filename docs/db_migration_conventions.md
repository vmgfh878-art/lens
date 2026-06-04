# Lens DB Migration Conventions (CP236d)

**상태**: 골격. 실 스키마 마이그레이션은 Supabase Pro 결제 후 별도 CP. 현재 `backend/db/scripts/ensure_runtime_schema.py`가 옛 운영 러너로 그대로 살아있다 (Strangler — 옛것 보존, 새것 공존).

## 파일명 규칙

`backend/migrations/versions/YYYY-MM-DD_slug.py`

- `alembic.ini`의 `file_template = %(year)d-%(month).2d-%(day).2d_%(slug)s` 자동 적용.
- 같은 날 여러 마이그레이션이면 `slug`에 순서/주제 명시 (예: `2026-06-04_add_orders_table` / `2026-06-04_backfill_users_email`).

## Reversible 원칙

- `upgrade()`와 `downgrade()` 모두 작성. downgrade 불가능한 변경 (실 데이터 삭제 등)도 가능한 한 reversible plan (백업 + WHERE 절 기록).
- ALTER TABLE은 NOT NULL + DEFAULT 두 단계로 (1. 컬럼 추가 + DEFAULT 채움 → 2. 별도 마이그레이션에서 NOT NULL).
- 큰 테이블은 backfill 별도 마이그레이션 + 작은 batch로.

## 정적 데이터 원칙

- 시드 / 정적 마스터 데이터는 Alembic이 아닌 별도 seed 스크립트로. Alembic은 스키마 변경만.
- enum 값 확장은 ADD VALUE만 reversible (REMOVE는 reversible 아님) — 추가만 권장, 제거는 별도 분기.

## enum ADD VALUE 주의

Postgres `ALTER TYPE ... ADD VALUE`는 트랜잭션 안에서 실행할 수 없다 (커밋된 enum만 다른 트랜잭션에서 사용 가능). Alembic은 기본 begin/commit으로 감싸므로 enum ADD는 별도 마이그레이션 + `op.execute("COMMIT")` 트릭 또는 `transaction_per_migration=False` 설정 필요.

## autogenerate

현재 `target_metadata = None` → autogenerate 비활성. 결제 후 ORM 도입 시 `from app.models import Base; target_metadata = Base.metadata`로 교체. 다만 autogenerate는 schema rename / type narrowing 등 감지 못함 → 사람 검토 필수.

## async 패턴

`env.py`가 `async_engine_from_config` + `connection.run_sync(do_run_migrations)`로 sync 마이그레이션 러너를 async connection 위에서 실행. 직접 `engine.connect()` 안 됨 (asyncpg는 sync 인터페이스 미지원). 자세한 골자: `backend/migrations/env.py` 본문.

## DB URL 조립

`env.py`의 `_build_async_url()`:
1. `ALEMBIC_DATABASE_URL` (full DSN) 있으면 그대로.
2. 없으면 `DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/DB_PORT/DB_SSLMODE`로 조립.
3. 둘 다 없으면 `RuntimeError`.

`alembic.ini`의 `sqlalchemy.url`은 **비워둠** (비밀번호 하드코딩 금지).

## 관련

- ADR-0013: 직접연결 5432 (CP236a).
- ADR-0026: async 세션 골격 (CP236b).
- ADR-0027: Alembic + async (CP236d, 본 문서와 짝).
- docs/db_repository_guide.md: N+1 / from_attributes 가이드 (CP236c).
