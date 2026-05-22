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
from app.routers import predict, prices
from app.routers.v1 import ai, health, stocks, predictions as v1_predictions

logger = logging.getLogger("lens.api")


def _parse_cors_origins() -> list[str]:
    raw = os.environ.get("BACKEND_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)
app.middleware("http")(request_id_middleware)

app.include_router(health.router, prefix="/api/v1")
app.include_router(stocks.router, prefix="/api/v1")
app.include_router(ai.router, prefix="/api/v1")
app.include_router(v1_predictions.router, prefix="/api/v1")
app.include_router(prices.router, prefix="/prices", tags=["prices"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])


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
