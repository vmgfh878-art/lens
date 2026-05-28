import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from app.core.exceptions import AppError
from app.core.http import error_response, success_response
from app.middleware.request_id import request_id_middleware
from app.routers.v1 import admin, ai, health, stocks, predictions as v1_predictions, strategies

logger = logging.getLogger("lens.api")


def _parse_cors_origins() -> list[str]:
    # 기본: 로컬 dev + Vercel production 도메인.
    # 새 vercel alias 가 추가될 때마다 손대지 않도록 regex 도 같이 사용.
    # Production 배포 시 BACKEND_CORS_ORIGINS 환경변수로 명시 override 가능.
    default_origins = ",".join([
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://lens-kimjihyeong-s-projects.vercel.app",
        "https://lens-ten-delta.vercel.app",
    ])
    raw = os.environ.get("BACKEND_CORS_ORIGINS", default_origins)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Vercel 의 모든 lens-* deployment URL 을 자동 허용한다.
# - lens-3hurhvz04-kimjihyeong-s-projects.vercel.app (preview hash + team)
# - lens-ten-delta.vercel.app (production alias)
# - lens-<branch>-<team>.vercel.app
# - lens-<hash>.vercel.app
# BACKEND_CORS_ORIGIN_REGEX 환경변수로 override 가능.
_CORS_ORIGIN_REGEX = os.environ.get(
    "BACKEND_CORS_ORIGIN_REGEX",
    r"^https://lens(?:-[a-z0-9-]+)?\.vercel\.app$",
)


app = FastAPI(title="Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.add_middleware(GZipMiddleware, minimum_size=512)
app.middleware("http")(request_id_middleware)

app.include_router(health.router, prefix="/api/v1")
app.include_router(stocks.router, prefix="/api/v1")
app.include_router(ai.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(v1_predictions.router, prefix="/api/v1")
app.include_router(strategies.router, prefix="/api/v1")


@app.on_event("startup")
def _load_v1_predictions_cache() -> None:
    """Startup 시 v1 predictions parquet 메모리 로드.
    Render free tier (512MB) 메모리 제약 때문에 startup 일괄 로드 비활성.
    각 endpoint 가 첫 호출 시 lazy load 하도록 두면 됨 (현재 frontend 가
    /api/v1/predictions/* 미사용이라 사실상 로드 안 됨).
    필요 시 LENS_EAGER_V1_CACHE=1 로 강제 활성.
    """
    if os.environ.get("LENS_EAGER_V1_CACHE", "0") != "1":
        logger.info("v1 predictions cache eager load disabled (set LENS_EAGER_V1_CACHE=1 to enable)")
        return
    base = Path(__file__).resolve().parent.parent / "data" / "v1"
    try:
        summary = v1_predictions.load_caches(base)
        for slot, info in summary.items():
            logger.info("v1 predictions cache %s: %s", slot, info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("v1 predictions cache load failed: %s", exc)


@app.get("/")
def health_check(request: Request):
    return success_response(
        request,
        {
            "status": "ok",
            "service": "lens-backend",
        },
    )


@app.exception_handler(AppError)
def handle_app_error(request: Request, exc: AppError):
    logger.warning("[%s] %s", getattr(request.state, "request_id", "-"), exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            request,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ),
    )


@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_response(
            request,
            code="VALIDATION_ERROR",
            message="요청 값이 올바르지 않습니다.",
            details=exc.errors(),
        ),
    )


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content=error_response(
            request,
            code="VALIDATION_ERROR",
            message=str(exc),
        ),
    )


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("[%s] 처리되지 않은 예외", getattr(request.state, "request_id", "-"))
    return JSONResponse(
        status_code=500,
        content=error_response(
            request,
            code="INTERNAL_ERROR",
            message="서버 내부 오류가 발생했습니다.",
        ),
    )
