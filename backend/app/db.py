from dotenv import load_dotenv
from supabase import Client, create_client

from app.config import get_database_config
from .core.exceptions import ConfigError, UpstreamUnavailableError
from backend.collector.utils.network import sanitize_proxy_env

# ──────────────────────────────────────────────────────────────────────
# Supabase 연결 전략 정본 (CP236a, ADR-0013)
#
# 두 경로:
# 1) REST (supabase-py, 본 모듈) — PostgREST over HTTP. 트랜잭션/풀링 함정 없음.
# 2) 직접 Postgres (ai/preprocessing.py 의 SQLAlchemy psycopg2) — 함정 영역.
#
# 함정: pooled 6543 (Supavisor/PgBouncer transaction mode) + asyncpg/asyncpg-style
# prepared statement 캐시 → `prepared statement "__asyncpg_..." already exists`.
# 우리 규모 권장 해결: 직접연결 5432 (PgBouncer 우회). 불가피하면
# NullPool + statement_cache_size=0 + prepared_statement_cache_size=0.
#
# 실연결/실연결 테스트는 사용자 Supabase Pro 결제 후 활성. 본 모듈은 그때까지
# REST 전용이고, 아래 직접연결 골격 함수/상수는 호출자 없음 (미실행).
# ──────────────────────────────────────────────────────────────────────

# CP236a — 직접 Postgres 연결 설정 골격 (미사용·미실행, 결제 후 활성).
DIRECT_DB_PORT_DEFAULT = "5432"  # direct connection (PgBouncer 우회 권장)
POOLED_TXN_DB_PORT = "6543"  # transaction mode (Supavisor/PgBouncer) — asyncpg와 충돌
APP_POOL_SIZE_PER_INSTANCE = 5  # PgBouncer 경유 시 인스턴스당 5~10 권장의 하한


def recommended_connect_args(port: str) -> dict:
    """포트별 안전 connect_args. 6543이면 prepared-statement 캐시를 끈다.

    호출자 없음 (CP236a 골격). 결제 후 직접연결 도입 시 사용.
    asyncpg 키 기준. psycopg2 환경에서는 무시되며 NullPool 병행 필요.
    """
    if str(port) == POOLED_TXN_DB_PORT:
        return {"statement_cache_size": 0, "prepared_statement_cache_size": 0}
    return {}


load_dotenv()

_client: Client | None = None


def supabase_is_configured() -> bool:
    """Supabase 환경변수 두 개가 설정됐는가.

    LENS_FORCE_LOCAL=1 이면 SUPABASE_URL/KEY 가 있어도 강제로 local parquet 경로를 탄다.
    Supabase quota 소진 / Render redeploy 시 보험.
    """
    cfg = get_database_config()
    if cfg.force_local:
        return False
    return bool(cfg.supabase_url) and bool(cfg.supabase_key)


def get_supabase() -> Client:
    """공유 Supabase 클라이언트를 반환한다."""
    global _client
    if _client is None:
        sanitize_proxy_env()
        cfg = get_database_config()
        if not cfg.supabase_url or not cfg.supabase_key:
            raise ConfigError("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")
        _client = create_client(cfg.supabase_url, cfg.supabase_key)
    return _client


def reset_supabase_client() -> None:
    """테스트에서 전역 클라이언트 캐시를 초기화한다."""
    global _client
    _client = None


def check_supabase_ready() -> dict[str, bool]:
    """환경변수와 Supabase 읽기 가능 여부를 확인한다."""
    cfg = get_database_config()
    url = cfg.supabase_url
    key = cfg.supabase_key
    checks = {
        "supabase_url": bool(url),
        "supabase_key": bool(key),
        "database": False,
    }

    if not url or not key:
        raise ConfigError("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.", details=checks)

    try:
        client = get_supabase()
        client.table("stock_info").select("ticker").limit(1).execute()
        checks["database"] = True
        return checks
    except ConfigError:
        raise
    except Exception as exc:
        raise UpstreamUnavailableError(
            "Supabase 준비 상태를 확인할 수 없습니다.", details=checks
        ) from exc
