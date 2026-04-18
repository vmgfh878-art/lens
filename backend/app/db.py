import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_client: Client | None = None


def get_supabase() -> Client:
    """공유 Supabase 클라이언트를 반환한다."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")
        _client = create_client(url, key)
    return _client
