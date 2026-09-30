"""CP235 — 도메인별 BaseSettings.

규약:
- 단일 거대 Settings 금지 — 도메인별 (Database/Market/ServingData/R2/Cors/Admin/Cache).
- 접근자 get_*_config()는 매 호출 새 인스턴스를 만들어 env 재평가
  (test_api.py:26 patch.dict(clear=True) 계약 + collector get_settings() 정합).
- truthy 파싱 / strip / CSV split은 코드 원본과 동일 (값 변경 0).
"""

from __future__ import annotations

import re

from pydantic import Field, field_validator
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

    @field_validator("supabase_url", "supabase_key", mode="before")
    @classmethod
    def _clean_secret(cls, v: str | None) -> str | None:
        # 붙여넣기 때 딸려온 공백·줄바꿈·비인쇄 제어문자 제거 (httpx InvalidURL / Invalid API key 방지).
        if v is None:
            return v
        cleaned = "".join(ch for ch in str(v).strip() if ch.isprintable())
        return cleaned or None

    @property
    def force_local(self) -> bool:
        return _truthy(self.force_local_raw)


class MarketConfig(BaseSettings):
    """MARKET_DATA_PROVIDER / LENS_LOCAL_SNAPSHOT_DIR."""

    model_config = _BASE_CONFIG

    market_data_provider: str = Field(default="yfinance", alias="MARKET_DATA_PROVIDER")
    local_snapshot_dir: str | None = Field(default=None, alias="LENS_LOCAL_SNAPSHOT_DIR")


class ServingDataConfig(BaseSettings):
    """서빙 스냅샷의 활성 경로와 번들 폴백 경로."""

    model_config = _BASE_CONFIG

    data_dir: str | None = Field(default=None, alias="LENS_SERVING_DATA_DIR")
    bootstrap_dir: str | None = Field(default=None, alias="LENS_SERVING_BOOTSTRAP_DIR")
    cache_dir: str | None = Field(default=None, alias="LENS_SERVING_CACHE_DIR")

    @field_validator("data_dir", "bootstrap_dir", "cache_dir", mode="before")
    @classmethod
    def _clean_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


class RemoteSyncConfig(BaseSettings):
    """실행 중인 서버의 원격 스냅샷 재확인 간격."""

    model_config = _BASE_CONFIG

    interval_seconds: int = Field(
        default=0, ge=0, le=3600, alias="LENS_REMOTE_SYNC_INTERVAL_SECONDS"
    )

    @field_validator("interval_seconds")
    @classmethod
    def _validate_interval(cls, value: int) -> int:
        if 0 < value < 60:
            raise ValueError("원격 재확인 간격은 0(비활성) 또는 60~3600초여야 합니다.")
        return value


class R2Config(BaseSettings):
    """비공개 Cloudflare R2 S3 API 연결 설정."""

    model_config = _BASE_CONFIG

    account_id: str | None = Field(default=None, alias="LENS_R2_ACCOUNT_ID")
    bucket: str | None = Field(default=None, alias="LENS_R2_BUCKET")
    access_key_id: str | None = Field(default=None, alias="LENS_R2_ACCESS_KEY_ID", repr=False)
    secret_access_key: str | None = Field(
        default=None,
        alias="LENS_R2_SECRET_ACCESS_KEY",
        repr=False,
    )
    session_token: str | None = Field(default=None, alias="LENS_R2_SESSION_TOKEN", repr=False)
    endpoint_url_override: str | None = Field(default=None, alias="LENS_R2_ENDPOINT_URL")
    object_prefix: str = Field(default="serving/v1", alias="LENS_R2_PREFIX")
    region: str = Field(default="auto", alias="LENS_R2_REGION")

    @field_validator(
        "account_id",
        "bucket",
        "access_key_id",
        "secret_access_key",
        "session_token",
        "endpoint_url_override",
        mode="before",
    )
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("object_prefix", mode="before")
    @classmethod
    def _clean_prefix(cls, value: str | None) -> str:
        cleaned = str(value or "serving/v1").strip().strip("/")
        if not cleaned or any(part in {".", ".."} for part in cleaned.split("/")):
            raise ValueError("LENS_R2_PREFIX가 올바르지 않습니다.")
        return cleaned

    @field_validator("region", mode="before")
    @classmethod
    def _clean_region(cls, value: str | None) -> str:
        return str(value or "auto").strip() or "auto"

    @property
    def endpoint_url(self) -> str | None:
        if self.endpoint_url_override:
            return self.endpoint_url_override.rstrip("/")
        if self.account_id:
            return f"https://{self.account_id}.r2.cloudflarestorage.com"
        return None

    @property
    def missing_required_fields(self) -> list[str]:
        missing = []
        if not self.endpoint_url:
            missing.append("LENS_R2_ACCOUNT_ID 또는 LENS_R2_ENDPOINT_URL")
        for env_name, value in (
            ("LENS_R2_BUCKET", self.bucket),
            ("LENS_R2_ACCESS_KEY_ID", self.access_key_id),
            ("LENS_R2_SECRET_ACCESS_KEY", self.secret_access_key),
        ):
            if not value:
                missing.append(env_name)
        return missing

    @property
    def configured(self) -> bool:
        return not self.missing_required_fields


class GitHubReleaseConfig(BaseSettings):
    """비공개 릴리스의 고정 채널과 접근 토큰 설정."""

    model_config = _BASE_CONFIG

    repository: str = Field(default="vmgfh878-art/lens-serving", alias="LENS_GITHUB_REPOSITORY")
    token: str | None = Field(default=None, alias="LENS_GITHUB_TOKEN", repr=False)
    release_tag: str = Field(default="serving-snapshots", alias="LENS_GITHUB_RELEASE_TAG")
    object_prefix: str = "serving/v1"

    @field_validator("repository", "release_tag", mode="before")
    @classmethod
    def _clean_identifier(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("repository")
    @classmethod
    def _validate_repository(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
            raise ValueError("LENS_GITHUB_REPOSITORY는 소유자/저장소 형식이어야 합니다.")
        if any(part in {".", ".."} for part in value.split("/")):
            raise ValueError("GitHub 저장소 식별자가 올바르지 않습니다.")
        return value

    @field_validator("release_tag")
    @classmethod
    def _validate_tag(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ValueError("고정 릴리스 태그가 올바르지 않습니다.")
        return value

    @field_validator("token", mode="before")
    @classmethod
    def _clean_token(cls, value: str | None) -> str | None:
        return str(value).strip() or None if value is not None else None

    @property
    def bucket(self) -> str:
        # 기존 파일 무결성 발행기의 저장 위치 계약을 재사용한다.
        return self.repository

    @property
    def missing_required_fields(self) -> list[str]:
        return [] if self.token else ["LENS_GITHUB_TOKEN"]

    @property
    def configured(self) -> bool:
        return not self.missing_required_fields


class CorsConfig(BaseSettings):
    """BACKEND_CORS_ORIGINS / BACKEND_CORS_ORIGIN_REGEX.

    CP242 — env 의 `*` 차단 validator. production / dev 무관 단일 guard
    (env 에 `*` 박는 실수 / 외부 침입 시 즉시 fail-loud).
    """

    model_config = _BASE_CONFIG

    raw_origins: str = Field(default=_DEFAULT_CORS_ORIGINS, alias="BACKEND_CORS_ORIGINS")
    origin_regex: str = Field(default=_DEFAULT_CORS_ORIGIN_REGEX, alias="BACKEND_CORS_ORIGIN_REGEX")

    @field_validator("raw_origins")
    @classmethod
    def _forbid_wildcard(cls, v: str) -> str:
        """env 의 BACKEND_CORS_ORIGINS 에 `*` 박힌 경우 즉시 fail.

        명시 origin 만 허용. `*` 는 모든 출처 허용이라 보안 트랙 종료 시점에
        근본 차단. 사용자가 Render dashboard 에서 `*` 박지 않게 강제.
        """
        items = [o.strip() for o in v.split(",") if o.strip()]
        if "*" in items:
            raise ValueError(
                "BACKEND_CORS_ORIGINS 에 '*' 박을 수 없습니다 (CP242). "
                "명시 origin (https://...) 만 허용."
            )
        return v

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


def get_serving_data_config() -> ServingDataConfig:
    return ServingDataConfig()


def get_remote_sync_config() -> RemoteSyncConfig:
    return RemoteSyncConfig()


def get_r2_config() -> R2Config:
    return R2Config()


def get_github_release_config() -> GitHubReleaseConfig:
    return GitHubReleaseConfig()


def get_cors_config() -> CorsConfig:
    return CorsConfig()


def get_admin_config() -> AdminConfig:
    return AdminConfig()


def get_cache_config() -> CacheConfig:
    return CacheConfig()
