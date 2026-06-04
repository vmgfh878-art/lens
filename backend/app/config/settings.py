"""CP235 — 도메인별 BaseSettings.

규약:
- 단일 거대 Settings 금지 — 도메인별 (Database/Market/Cors/Admin/Cache).
- 접근자 get_*_config()는 매 호출 새 인스턴스를 만들어 env 재평가
  (test_api.py:26 patch.dict(clear=True) 계약 + collector get_settings() 정합).
- truthy 파싱 / strip / CSV split은 코드 원본과 동일 (값 변경 0).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_TRUTHY = {"1", "true", "yes"}


def _truthy(value: str | None) -> bool:
    """원본 코드의 `.strip().lower() in {"1","true","yes"}` 보존."""
    return (value or "").strip().lower() in _TRUTHY


_BASE_CONFIG = SettingsConfigDict(
    env_file=None,  # .env는 app/db.py의 load_dotenv()가 process env에 이미 로드
    extra="ignore",
    case_sensitive=False,
)


# main.py:25-32 default. 코드 1글자 변경 0.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "https://lens-kimjihyeong-s-projects.vercel.app,"
    "https://lens-ten-delta.vercel.app"
)
# main.py:43-46 default.
_DEFAULT_CORS_ORIGIN_REGEX = r"^https://lens(?:-[a-z0-9-]+)?\.vercel\.app$"


class DatabaseConfig(BaseSettings):
    """SUPABASE_URL / SUPABASE_KEY / LENS_FORCE_LOCAL."""

    model_config = _BASE_CONFIG

    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_key: str | None = Field(default=None, alias="SUPABASE_KEY")
    force_local_raw: str = Field(default="0", alias="LENS_FORCE_LOCAL")

    @property
    def force_local(self) -> bool:
        return _truthy(self.force_local_raw)


class MarketConfig(BaseSettings):
    """MARKET_DATA_PROVIDER / LENS_LOCAL_SNAPSHOT_DIR."""

    model_config = _BASE_CONFIG

    market_data_provider: str = Field(default="yfinance", alias="MARKET_DATA_PROVIDER")
    local_snapshot_dir: str | None = Field(default=None, alias="LENS_LOCAL_SNAPSHOT_DIR")


class CorsConfig(BaseSettings):
    """BACKEND_CORS_ORIGINS / BACKEND_CORS_ORIGIN_REGEX."""

    model_config = _BASE_CONFIG

    raw_origins: str = Field(default=_DEFAULT_CORS_ORIGINS, alias="BACKEND_CORS_ORIGINS")
    origin_regex: str = Field(default=_DEFAULT_CORS_ORIGIN_REGEX, alias="BACKEND_CORS_ORIGIN_REGEX")

    @property
    def origins(self) -> list[str]:
        # main.py:34 _parse_cors_origins() 동일 규칙: split + strip + 빈 값 제거.
        return [o.strip() for o in self.raw_origins.split(",") if o.strip()]


class AdminConfig(BaseSettings):
    """LENS_ADMIN_RELOAD_TOKEN / LENS_ALLOW_LOCAL_ADMIN_RELOAD."""

    model_config = _BASE_CONFIG

    reload_token_raw: str = Field(default="", alias="LENS_ADMIN_RELOAD_TOKEN")
    allow_local_reload_raw: str = Field(default="0", alias="LENS_ALLOW_LOCAL_ADMIN_RELOAD")

    @property
    def reload_token(self) -> str:
        # admin.py:29 원본 .strip() 동일.
        return self.reload_token_raw.strip()

    @property
    def allow_local_reload(self) -> bool:
        return _truthy(self.allow_local_reload_raw)


class CacheConfig(BaseSettings):
    """LENS_EAGER_V1_CACHE + GZip minimum size."""

    model_config = _BASE_CONFIG

    eager_v1_cache_raw: str = Field(default="0", alias="LENS_EAGER_V1_CACHE")
    # GZip minimum_size는 환경변수 없음 (main.py:59 hardcoded). 코드 상수 그대로 유지.
    gzip_minimum_size: int = Field(default=512)

    @property
    def eager_v1_cache(self) -> bool:
        # main.py:78 정확히 `!= "1"` 게이트 → 활성=`== "1"` (반전).
        return self.eager_v1_cache_raw == "1"


# 접근자 — 매 호출 새 인스턴스를 만들어 env 재평가.
# 1) test_api.py:26 patch.dict(clear=True) 계약: 요청 시점에 env가 비어 있어야 한다.
# 2) collector get_settings() 선례와 정합 (해당 함수도 매 호출 새 dataclass 빌드).
# 3) import-time 싱글톤 캐시는 위 1)을 깬다 → 금지.


def get_database_config() -> DatabaseConfig:
    return DatabaseConfig()


def get_market_config() -> MarketConfig:
    return MarketConfig()


def get_cors_config() -> CorsConfig:
    return CorsConfig()


def get_admin_config() -> AdminConfig:
    return AdminConfig()


def get_cache_config() -> CacheConfig:
    return CacheConfig()
