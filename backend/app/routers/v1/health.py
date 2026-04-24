from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.http import success_response
from app.db import check_supabase_ready
from app.schemas.common import ApiResponse, ErrorResponse
from app.schemas.health import LiveHealthData, ReadyHealthData

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
    checks = check_supabase_ready()
    return success_response(
        request,
        {
            "status": "ok",
            "service": "lens-backend",
            "checks": checks,
        },
    )
