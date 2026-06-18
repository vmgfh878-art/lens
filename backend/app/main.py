from pathlib import Path

import structlog
from app.config import get_cache_config, get_cors_config
from app.core.exceptions import AppError
from app.core.http import error_response, success_response
from app.core.logging import configure_logging
from app.core.security_headers import SecurityHeadersMiddleware
from app.middleware.request_id import request_id_middleware
from app.routers.v1 import admin, ai, health, stocks, strategies
from app.routers.v1 import predictions as v1_predictions
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

configure_logging()
logger = structlog.get_logger("lens.api")


def _parse_cors_origins() -> list[str]:
    # CP235 — 도메인별 CorsConfig가 BACKEND_CORS_ORIGINS env + 기본값을
    # 함께 들고 있다. 본 함수는 시그니처 보존 위해 얇은 위임으로 둔다.
    return get_cors_config().origins


app = FastAPI(title="Lens API", version="0.1.0")

_cors = get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors.origins,
    allow_origin_regex=_cors.origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.add_middleware(GZipMiddleware, minimum_size=get_cache_config().gzip_minimum_size)
# CP240 — 보안 헤더 미들웨어. last add = outermost. 응답 후처리 마지막에
# 5 헤더 (HSTS / X-Content-Type-Options / X-Frame-Options / Referrer-Policy /
# Content-Security-Policy) 박음. setdefault 라 기존 헤더 보존.
app.add_middleware(SecurityHeadersMiddleware)
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
    if not get_cache_config().eager_v1_cache:
        logger.info(
            "v1 predictions cache eager load disabled (set LENS_EAGER_V1_CACHE=1 to enable)"
        )
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
    # CP241 — details 를 loc + type 으로 minimal. 공격자에게 "이 필드는 이런
    # 형식" 같은 raw pydantic 메시지 (ctx.pattern 등) 노출 차단.
    minimal_details = [{"loc": err.get("loc"), "type": err.get("type")} for err in exc.errors()]
    return JSONResponse(
        status_code=422,
        content=error_response(
            request,
            code="VALIDATION_ERROR",
            message="요청 값이 올바르지 않습니다.",
            details=minimal_details,
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
