"""CP236b — SQLAlchemy 2.0 async 세션 안전 패턴 골격.

**현재 비활성** (sqlalchemy/asyncpg 미설치, DATABASE_URL 미설정).
실연결 검증은 사용자 Supabase Pro 결제 후. 로컬 서빙 경로 (REST + parquet)
에는 0 영향. 본 모듈을 호출하는 caller도 현재 0건이라 import만으로는 부작용 없음.

박제 안전 패턴 (master plan §3.6):
1. **세션 = 요청당 1개 = 트랜잭션 경계**: `async with SessionLocal() as session: yield`.
2. **`expire_on_commit=False` 필수**: 안 하면 commit 후 ORM 속성 접근 시 동기
   lazy refresh I/O → `MissingGreenlet` / `DetachedInstance` 에러.
3. **명시적 rollback**: `try/except/rollback/raise`. 없으면 부분쓰기 silent 불일치.
4. **`pool_pre_ping=True`**: 죽은 connection 감지 (Supabase idle drop 대비).
5. **`pool_recycle=1800`**: 서버 idle timeout 전 재활용.
6. **포트 5432 direct 권장** (PgBouncer 우회). 6543 강제 시 `NullPool` +
   `connect_args={"statement_cache_size": 0}` (ADR-0013).

미설치 환경 안전 규칙: 모든 sqlalchemy import는 try/except 가드. engine /
session factory는 lazy (모듈 로드 시 즉시 생성 X). 호출자가 결제 후 활성화
시점에만 실제 연결을 트리거한다.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, AsyncIterator

# CP236b — sqlalchemy 미설치 환경에서도 `import app.session`이 죽지 않도록 가드.
try:
    from sqlalchemy.ext.asyncio import (  # type: ignore[import-not-found]
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False
    # 타입 힌트만 보여주기 위한 placeholder. 런타임에는 사용되지 않음.
    if TYPE_CHECKING:
        from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession  # type: ignore[import-not-found]


_engine: "AsyncEngine | None" = None
_session_factory = None


def _database_url() -> str | None:
    """결제 후 환경변수로 설정. 미설정이면 None."""
    return os.environ.get("DATABASE_URL")


def get_engine() -> "AsyncEngine":
    """Lazy async engine. 호출 시점에 최초 1회 생성.

    미설치 / DATABASE_URL 미설정 환경에서는 RuntimeError. 호출되지 않는 한 무해.
    """
    global _engine
    if _engine is not None:
        return _engine

    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "DB 비활성: sqlalchemy/asyncpg 미설치. 결제 후 backend/requirements.txt"
            "의 주석 핀을 해제하고 설치하세요 (CP236b)."
        )

    url = _database_url()
    if not url:
        raise RuntimeError(
            "DB 비활성: DATABASE_URL 미설정. 결제 후 환경변수에 직접연결 5432 DSN 설정 (ADR-0013)."
        )

    # CP236a / ADR-0013: 직접연결 5432 권장. transaction mode (6543) 강제 시
    # `poolclass=NullPool` + `connect_args={"statement_cache_size": 0}`.
    _engine = create_async_engine(
        url,
        pool_pre_ping=True,  # Supabase idle drop 대비 죽은 connection 감지
        pool_recycle=1800,  # 서버 idle timeout 전 재활용 (초)
    )
    return _engine


def get_session_factory():
    """Lazy session factory. engine과 같이 호출 시점에 최초 1회 생성.

    `expire_on_commit=False` 필수 — commit 후 속성 접근 시 동기 lazy I/O
    (MissingGreenlet) 방지.
    """
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError("DB 비활성: sqlalchemy 미설치 (CP236b 골격).")

    _session_factory = async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,  # 필수 (CP236b §3.6)
        class_=AsyncSession,
    )
    return _session_factory


# 호환: 외부에서 `from app.session import SessionLocal`로도 접근 가능하게 별칭.
# 단 lazy 초기화라 import 시점에는 None — 호출 직전에 get_session_factory() 사용 권장.
SessionLocal = None  # type: ignore[assignment]  # CP236b 골격 — 결제 후 get_session_factory() 사용.


async def get_db() -> "AsyncIterator[AsyncSession]":
    """FastAPI Depends(get_db) 용. **아직 어느 라우터에도 부착 안 함**.

    안전 패턴:
    - 세션 = 요청당 1개 = 트랜잭션 경계.
    - 예외 시 명시적 rollback.
    - **자동 commit 안 함** — 라우터가 `await session.commit()` 명시 (read-only는 불필요).
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()  # 명시 rollback 없으면 부분쓰기 silent 불일치
            raise


__all__ = [
    "get_engine",
    "get_session_factory",
    "get_db",
    "SessionLocal",
    "_SQLALCHEMY_AVAILABLE",
]
