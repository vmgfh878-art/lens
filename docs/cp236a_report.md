# CP236a 보고서 — Supabase 재연결 풀링 함정 회피 골격

**완료일**: 2026-06-04
**선행 의존**: 없음 (additive)
**커밋**: 본 commit

## 요구

직접 Postgres 연결 도입 시 발생하는 PgBouncer 6543 + asyncpg prepared-statement 함정을 미리 회피하도록 골격/주석/문서를 박는다. 실연결 / 실 검증은 사용자 Supabase Pro 결제 후 활성. 현재 동작 (psycopg2 + 5432 기본, REST 경유) 무변경.

## 한 일

| 파일 | 변경 | 비고 |
|---|---|---|
| `backend/app/db.py` | 정본 주석 + `DIRECT_DB_PORT_DEFAULT` / `POOLED_TXN_DB_PORT` / `APP_POOL_SIZE_PER_INSTANCE` 상수 + `recommended_connect_args(port)` 헬퍼 추가 | 호출자 없음 (골격, 미실행). 기존 함수 signature 무변경. |
| `ai/preprocessing.py` | `_postgres_dsn()`에 6543 감지 → `RuntimeWarning` 1회 + `_postgres_engine()` 주석 (6543 전환 시 NullPool + connect_args 규약) | DSN 문자열·반환 타입·5432 기본 동작 전부 보존. |
| `render.yaml` | `SUPABASE_DB_HOST/PORT/USER/PASSWORD/NAME` `sync: false` 키 5개 추가 | 값 주입 없음. 주석 cron 블록 무수정. |
| `docs/adr/0013-supabase-port-5432-not-6543.md` | 신규 | 함정·해결·골격 위치 |

## 보존 체크리스트

| 항목 | 확인 |
|---|---|
| db.py `supabase_is_configured` / `get_supabase` / `reset_supabase_client` / `check_supabase_ready` signature 보존 | OK |
| preprocessing.py `_postgres_dsn` 반환 DSN 문자열 형식 (`postgresql://...?sslmode=...`) 무변경 | OK |
| preprocessing.py `_postgres_engine` 현재 동작 (`create_engine(dsn)`, 기본 풀) 보존 | OK |
| 실제 Supabase 호출 / asyncpg 도입 / 풀 클래스 변경 0 | OK (이 CP 범위 밖) |
| pytest 회귀 0 | OK (다음 check) |

## 자가 점검

- **[Plan v3 정합]** PASS — 사유: 골격/주석만. 밴드 본체·fidelity·cost 무관.
- **[구조 결함]** PASS — 사유: 추가 코드 미실행. 기존 caller 9곳 (db.py 의존) 무영향. `recommended_connect_args` 호출자 0.
- **[모델 영향]** PASS — 사유: feature 계산·학습 코드 무관. preprocessing의 DSN 가드는 경고만, DSN 문자열 동일.

## 후속

- Supabase Pro 결제 후 직접연결 활성 CP: `recommended_connect_args` 호출 + NullPool 전환 (필요시) + 실연결 health probe.
- CP236b: SQLAlchemy 2.0 async 세션 골격 (`backend/app/session.py`).
