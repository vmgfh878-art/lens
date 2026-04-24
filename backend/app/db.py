import os

from dotenv import load_dotenv
from supabase import Client, create_client

from .core.exceptions import ConfigError, UpstreamUnavailableError
from backend.collector.utils.network import sanitize_proxy_env

load_dotenv()

_client: Client | None = None


def get_supabase() -> Client:
    """공유 Supabase 클라이언트를 반환한다."""
    global _client
    if _client is None:
        sanitize_proxy_env()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ConfigError("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")
        _client = create_client(url, key)
    return _client


def reset_supabase_client() -> None:
    """테스트에서 전역 클라이언트 캐시를 초기화한다."""
    global _client
    _client = None


def check_supabase_ready() -> dict[str, bool]:
    """환경변수와 Supabase 읽기 가능 여부를 확인한다."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
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
        raise UpstreamUnavailableError("Supabase 준비 상태를 확인할 수 없습니다.", details=checks) from exc
