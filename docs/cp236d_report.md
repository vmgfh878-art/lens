# CP236d 보고서 — Alembic + async 마이그레이션 골격

**완료일**: 2026-06-04
**선행 의존**: CP236a/b/c 그린.
**커밋**: 본 commit (골격, additive)

## 요구

Alembic 골격 (`env.py` async / `script.py.mako` / `versions/` 빈 폴더 + placeholder 1장) + `backend/alembic.ini` + 컨벤션 문서 + asyncpg 핀. 실 마이그레이션 작성 0. 기존 `ensure_runtime_schema.py` 손대지 않음.

## 한 일

| 파일 | 변경 |
|---|---|
| `backend/alembic.ini` (신규) | `script_location=migrations` / `prepend_sys_path=..` / `file_template=YYYY-MM-DD_slug` / `sqlalchemy.url` 비움 |
| `backend/migrations/env.py` (신규) | async 패턴 (`async_engine_from_config + NullPool + connection.run_sync`). URL은 `ALEMBIC_DATABASE_URL` 또는 `DB_HOST/...` 조립. `target_metadata=None` (autogenerate 비활성) |
| `backend/migrations/script.py.mako` (신규) | 기본 alembic 템플릿 |
| `backend/migrations/versions/.gitkeep` (신규) | 빈 폴더 git 보존 |
| `backend/migrations/versions/2026-06-04_initial_placeholder.py` (신규) | revision 1장, upgrade/downgrade `pass` |
| `backend/db/requirements-crawler.txt` | `alembic==1.18.4` + `asyncpg==0.30.0` 핀 추가 |
| `docs/db_migration_conventions.md` (신규) | 파일명/reversible/정적 데이터/enum ADD VALUE/autogenerate/async/URL 조립 규약 |
| `docs/adr/0027-alembic-async-migrations.md` (신규) | 골격 구조 + 옛 코드 보존 + 실 마이그레이션 금지 결정 |

## 보존 체크리스트

| 항목 | 확인 |
|---|---|
| `backend/db/scripts/ensure_runtime_schema.py` 0줄 수정 | OK |
| 기존 `backend/db/*.py` (export_parquet, test_connection 등) 0줄 수정 | OK |
| `backend/requirements.txt` 0줄 수정 (alembic은 crawler 도메인) | OK |
| `backend/app/*` 0줄 수정 | OK |
| API 응답 schema 0 변경 | OK |
| 실제 Alembic 명령 0 실행 | OK (`alembic upgrade` / `downgrade` 금지) |
| `target_metadata=None` (autogenerate 비활성) | OK |
| `sqlalchemy.url` 비움 (비밀번호 하드코딩 금지) | OK |

## 자가 점검

- **[Plan v3 정합]** PASS — 사유: 골격/문서만. 밴드/fidelity/cost 무관.
- **[구조 결함]** PASS — 사유: 새 디렉토리 (`migrations/`) + 새 ini + 신규 파일 5개. 기존 운영 러너 `ensure_runtime_schema.py` 그대로. caller 이전·옛 제거 0 (결제 후).
- **[모델 영향]** PASS (N/A) — 사유: 학습/calibration/feature 무관. asyncpg 핀은 실 설치 안 함.

## 후속

- 결제 후 CP236d-cont: 실 스키마 마이그레이션 (orders / 정산 테이블 등) + autogenerate 활성 + `ensure_runtime_schema.py` caller 이전.
- 결제 후: `pip install -r backend/db/requirements-crawler.txt`로 alembic+asyncpg 실설치 + `alembic upgrade head` 실 실행.
