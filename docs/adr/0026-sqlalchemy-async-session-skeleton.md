# ADR-0026: SQLAlchemy 2.0 async 세션 안전 패턴 — 골격 (비활성)

Status: Accepted (skeleton — 실연결은 Supabase Pro 결제 후 활성)
Date: 2026-06-04
CP: CP236b

## 결정

`backend/app/session.py`에 async DB 세션 패턴을 박제한다. 호출자 0, 실연결 0. 결제 후 활성 시점에 바로 쓸 수 있도록 다음 규칙을 코드와 주석으로 박는다.

## 박제 규칙

1. **세션 = 요청당 1개 = 트랜잭션 경계**: `async with SessionLocal() as session: yield`.
2. **`expire_on_commit=False` 필수**: 안 하면 commit 후 ORM 속성 접근 시 동기 lazy refresh I/O → `MissingGreenlet` / `DetachedInstance`.
3. **명시적 rollback**: `try/except/rollback/raise`. 없으면 부분쓰기 silent 불일치.
4. **`pool_pre_ping=True`**: Supabase idle drop 대비 죽은 connection 감지.
5. **`pool_recycle=1800`**: 서버 idle timeout 전 재활용.
6. **포트 5432 direct**: ADR-0013 (PgBouncer 우회). 6543 강제 시 `NullPool` + `connect_args={"statement_cache_size": 0}`.
7. **commit은 라우터 명시**: `get_db()`는 자동 commit 안 함 (read-only는 commit 불필요).

## 미설치 안전

`from sqlalchemy.ext.asyncio import ...`는 try/except 가드. engine과 session factory는 lazy (모듈 로드 시 즉시 생성 X). `import app.session`만으로는 sqlalchemy 미설치 환경에서도 안 깨짐. `get_engine()` / `get_session_factory()`는 호출 시점에 미설치 / DATABASE_URL 미설정이면 명시적 `RuntimeError`.

## requirements 핀 (주석)

```
# --- CP236b: Supabase 직접연결 async 준비. 결제 후 주석 해제 + 설치 ---
# sqlalchemy[asyncio]==2.0.32
# asyncpg==0.29.0
```

결제 후 주석 해제 + `pip install`. 현재는 빌드 무영향.

## 미수행 (이 CP 범위 밖)

- `main.py` 와이어링 (`@app.on_event("startup")` engine init, lifespan, `Depends(get_db)` 부착): 결제 후 활성 시점.
- ORM 모델 정의 / 테이블 매핑: 결제 후.
- Alembic 마이그레이션: CP236d.
- N+1 / eager-load 가이드: CP236c.

## 결과

`backend/app/session.py` 신규 (134줄, 호출자 0). `backend/requirements.txt`에 sqlalchemy/asyncpg 주석 핀. `backend/app/db.py` 0줄 수정. 기존 라우터/API 응답 schema 무변경. import smoke 통과.
