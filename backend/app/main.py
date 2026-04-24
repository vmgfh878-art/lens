import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from app.core.exceptions import AppError
from app.core.http import error_response, success_response
from app.middleware.request_id import request_id_middleware
from app.routers import predict, prices
from app.routers.v1 import health, stocks

logger = logging.getLogger("lens.api")


def _parse_cors_origins() -> list[str]:
    raw = os.environ.get("BACKEND_CORS_ORIGINS", "http://localhost:3000")
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
app.include_router(prices.router, prefix="/prices", tags=["prices"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])


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
