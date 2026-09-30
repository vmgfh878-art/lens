"""서빙 스냅샷 경로를 한곳에서 결정한다."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from app.config import get_serving_data_config

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _BACKEND_DIR / "data" / "v1"
_DEFAULT_BOOTSTRAP_DIR = _BACKEND_DIR / "data" / "bootstrap" / "v1"
_DEFAULT_CACHE_DIR = _BACKEND_DIR / "data" / "runtime" / "v1"

_ACTIVE_DATA_DIR: Path | None = None
_ACTIVE_DATA_DIR_LOCK = RLock()


def _resolve_configured_path(raw_path: str | None, default: Path) -> Path:
    if not raw_path:
        return default.resolve()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _REPOSITORY_ROOT / path
    return path.resolve()


def configured_serving_data_dir() -> Path:
    """환경변수 기준 서빙 경로를 반환한다. 기본값은 기존 backend/data/v1이다."""
    return _resolve_configured_path(get_serving_data_config().data_dir, _DEFAULT_DATA_DIR)


def get_serving_data_dir() -> Path:
    """런타임 활성 경로가 있으면 우선하고, 없으면 설정 경로를 반환한다."""
    with _ACTIVE_DATA_DIR_LOCK:
        active = _ACTIVE_DATA_DIR
    return active if active is not None else configured_serving_data_dir()


def get_serving_bootstrap_dir() -> Path:
    """원격 동기화 실패 시 사용할 번들 폴백 경로를 반환한다."""
    return _resolve_configured_path(
        get_serving_data_config().bootstrap_dir,
        _DEFAULT_BOOTSTRAP_DIR,
    )


def get_serving_cache_dir() -> Path:
    """검증된 원격 스냅샷을 저장할 휘발성 런타임 캐시 루트를 반환한다."""
    return _resolve_configured_path(
        get_serving_data_config().cache_dir,
        _DEFAULT_CACHE_DIR,
    )


def activate_serving_data_dir(path: str | Path) -> Path:
    """검증이 끝난 스냅샷 경로를 현재 프로세스의 활성 경로로 지정한다."""
    global _ACTIVE_DATA_DIR
    resolved = _resolve_configured_path(str(path), _DEFAULT_DATA_DIR)
    with _ACTIVE_DATA_DIR_LOCK:
        _ACTIVE_DATA_DIR = resolved
    return resolved


def reset_active_serving_data_dir() -> None:
    """런타임 활성 경로를 해제하고 환경변수 기준 경로로 되돌린다."""
    global _ACTIVE_DATA_DIR
    with _ACTIVE_DATA_DIR_LOCK:
        _ACTIVE_DATA_DIR = None


def serving_file(name: str) -> Path:
    """현재 활성 서빙 디렉터리 아래 파일 경로를 반환한다."""
    return get_serving_data_dir() / name


__all__ = [
    "activate_serving_data_dir",
    "configured_serving_data_dir",
    "get_serving_bootstrap_dir",
    "get_serving_cache_dir",
    "get_serving_data_dir",
    "reset_active_serving_data_dir",
    "serving_file",
]
