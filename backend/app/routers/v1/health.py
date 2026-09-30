from __future__ import annotations

from app.config import R2Config
from app.core.exceptions import UpstreamUnavailableError
from app.core.http import success_response
from app.db import check_supabase_ready
from app.schemas.common import ApiResponse, ErrorResponse
from app.schemas.health import LiveHealthData, ReadyHealthData
from app.services import data_backend
from app.services.serving_paths import get_serving_data_dir
from app.services.serving_snapshot import REQUIRED_SERVING_FILES
from app.services.serving_storage import get_remote_serving_config
from app.services.serving_sync import get_serving_sync_state
from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=ApiResponse[LiveHealthData],
    responses={500: {"model": ErrorResponse}},
)
def live(request: Request):
    return success_response(
        request,
        {
            "status": "ok",
            "service": "lens-backend",
        },
    )


@router.get(
    "/ready",
    response_model=ApiResponse[ReadyHealthData],
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ready(request: Request):
    if data_backend.use_supabase():
        checks = check_supabase_ready()
        status = "ok" if checks.get("database") else "degraded"
        source = "supabase"
        snapshot_id = None
        sync_status = None
        last_r2_success_at = None
        last_remote_success_at = None
    else:
        active_dir = get_serving_data_dir()
        directory_ready = active_dir.is_dir()
        files_ready = directory_ready and all(
            (active_dir / name).is_file() and (active_dir / name).stat().st_size > 0
            for name in REQUIRED_SERVING_FILES
        )
        remote_config = get_remote_serving_config()
        sync_state = get_serving_sync_state()
        sync_ready = not remote_config.configured or sync_state["status"] == "ready"
        bootstrap_active = sync_state["source"] == "bootstrap"
        bootstrap_verified = not bootstrap_active or bool(sync_state["fallback_ready"])
        checks = {
            "local_backend": True,
            "serving_directory": directory_ready,
            "required_files": files_ready,
            "bootstrap_verified": bootstrap_verified,
            "r2_configured": isinstance(remote_config, R2Config) and remote_config.configured,
            "r2_sync_ready": sync_ready,
            "remote_storage_configured": remote_config.configured,
            "remote_sync_ready": sync_ready,
        }
        if not files_ready or not bootstrap_verified:
            raise UpstreamUnavailableError(
                "활성 서빙 스냅샷이 준비되지 않았습니다.",
                details={**checks, "active_dir": str(active_dir)},
            )
        status = "ok" if sync_ready else "degraded"
        source = str(sync_state["source"])
        snapshot_id = sync_state["snapshot_id"]
        sync_status = sync_state["status"]
        last_remote_success_at = sync_state["last_success_at"]
        last_r2_success_at = last_remote_success_at if source == "r2" else None
    return success_response(
        request,
        {
            "status": status,
            "service": "lens-backend",
            "checks": checks,
            "source": source,
            "snapshot_id": snapshot_id,
            "remote_sync_status": sync_status,
            "last_remote_success_at": last_remote_success_at,
            "r2_sync_status": sync_status,
            "last_r2_success_at": last_r2_success_at,
        },
    )
