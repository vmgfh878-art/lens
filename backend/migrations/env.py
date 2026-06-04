"""Alembic env — async 패턴 (CP236d 골격).

실연결은 사용자 Supabase Pro 결제 후 활성. 현재 호출자 0.
"""

from __future__ import annotations

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
