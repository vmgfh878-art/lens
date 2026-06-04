# CP236b 보고서 — SQLAlchemy 2.0 async 세션 안전 패턴 골격

**완료일**: 2026-06-04
**선행 의존**: CP236a (직접연결 포트/풀링 골격) — 그린.
**커밋**: 본 commit (Sub-step 1~4 합본 — 골격이라 분할 의미 낮음, 자가 점검에 명시)

## 요구

`backend/app/session.py` 신설. SQLAlchemy 2.0 async 안전 패턴 골격 + 미설치 환경 안전 가드. 실연결 / 실 검증은 결제 후. 기존 `backend/app/db.py` 0줄 수정.

## 한 일

| 파일 | 변경 |
|---|---|
| `backend/app/session.py` (신규) | async engine (lazy `get_engine()` + `pool_pre_ping=True` + `pool_recycle=1800`) + session factory (lazy `get_session_factory()` + `expire_on_commit=False`) + `get_db()` async generator (명시적 rollback + 자동 commit 안 함). sqlalchemy import는 try/except 가드. |
| `backend/requirements.txt` | sqlalchemy/asyncpg 주석 핀 추가 (설치 X, 결제 후 활성). |

Sub-step 1~4를 단일 commit으로 합침 — 골격이라 미설치 환경 안전성만 검증되면 충분.

## 보존 체크리스트

| 항목 | 확인 |
|---|---|
| `backend/app/db.py` 0줄 수정 | OK |
| `main.py` 라우터 등록 / `@app.on_event` 무변경 | OK |
| API 응답 schema 무변경 | OK |
| `import app.session` 미설치 환경에서도 죽지 않음 | OK (try/except 가드) |
| `expire_on_commit=False` 박제 | OK |
| 명시적 rollback 박제 | OK |
| `pool_pre_ping=True` + `pool_recycle=1800` 박제 | OK |
| 포트 5432 + 6543 NullPool 규약 주석 박제 | OK (ADR-0013 참조) |

## 검증

- `python -c "import app.session"` → 에러 0 (`_SQLALCHEMY_AVAILABLE=True`, sqlalchemy 이미 설치돼 있음. 미설치 환경 가드 동작은 try/except 구조 자체로 보장).
- `from app.main import app` → 정상 로드.
- pytest 회귀 0 — session.py가 caller 0이라 영향 자체가 없음.

## 자가 점검

- **[Plan v3 정합]** PASS — 사유: 골격만, 밴드/fidelity/cost 무관.
- **[구조 결함]** PASS — 사유: lazy import + try/except 가드 + caller 0. db.py 0줄 수정. Sub-step 1~4 합본은 §자가 점검에 명시 (지시서 §Step 2 허용).
- **[모델 영향]** PASS (N/A) — 사유: 학습/calibration/feature 무관.

## 후속

- 결제 후 CP: `main.py` lifespan 추가 + `Depends(get_db)` 부착 + ORM 모델 정의.
- CP236c: N+1 / eager-load 가이드.
- CP236d: Alembic + async 마이그레이션 골격.
