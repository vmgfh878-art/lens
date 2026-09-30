from threading import Event, Thread

import structlog
from app.config import get_cache_config, get_cors_config, get_remote_sync_config
from app.core.exceptions import AppError
from app.core.http import error_response, success_response
from app.core.logging import configure_logging
from app.core.security_headers import SecurityHeadersMiddleware
from app.middleware.request_id import request_id_middleware
from app.routers.v1 import admin, ai, health, stocks, strategies
from app.routers.v1 import predictions as v1_predictions
from app.services.serving_paths import (
    configured_serving_data_dir,
    get_serving_bootstrap_dir,
    get_serving_data_dir,
)
from app.services.serving_storage import get_remote_serving_config
from app.services.serving_sync import (
    activate_bootstrap_serving_snapshot,
    sync_latest_serving_snapshot,
)
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
def _activate_packaged_serving_fallback() -> None:
    """배포 기본 경로가 고정 폴백이면 해시 검증 후 먼저 활성화한다."""
    if configured_serving_data_dir() != get_serving_bootstrap_dir():
        return
    try:
        result = activate_bootstrap_serving_snapshot()
        logger.info(
            "bundled serving fallback ready: snapshot_id=%s",
            result["snapshot_id"],
        )
    except Exception as exc:  # noqa: BLE001 — R2 성공 가능성이 있어 다음 시작 단계를 계속한다.
        logger.error("bundled serving fallback validation failed: %s", exc)


@app.on_event("startup")
def _sync_remote_serving_snapshot() -> None:
    """원격 저장소가 설정되면 캐시 로드 전에 최신 완성 버전을 활성화한다."""
    config = get_remote_serving_config()
    if not config.configured:
        logger.info("원격 서빙 동기화 비활성")
        return
    try:
        result = sync_latest_serving_snapshot(config=config)
        logger.info(
            "원격 서빙 스냅샷 준비 완료: snapshot_id=%s changed=%s",
            result["snapshot_id"],
            result["changed"],
        )
    except Exception as exc:  # noqa: BLE001 — 검증된 고정 폴백으로 서버는 계속 기동한다.
        logger.warning("원격 서빙 동기화 실패. 검증된 폴백을 유지합니다: %s", exc)


def _periodic_remote_sync(stop_event: Event, interval_seconds: int) -> None:
    """서버가 켜진 동안만 원격 최신 버전을 재확인한다."""
    while not stop_event.wait(interval_seconds):
        try:
            result = sync_latest_serving_snapshot()
            if result["changed"]:
                logger.info("새 서빙 스냅샷 활성화: snapshot_id=%s", result["snapshot_id"])
        except Exception as exc:  # noqa: BLE001 — 현재 활성 버전을 유지하고 다음 간격에 재시도한다.
            logger.warning("주기적 원격 서빙 동기화 실패. 기존 버전을 유지합니다: %s", exc)


@app.on_event("startup")
def _start_periodic_remote_sync() -> None:
    """관리 토큰 없이도 켜져 있는 Render가 새 릴리스를 받도록 선택적으로 켠다."""
    interval_seconds = get_remote_sync_config().interval_seconds
    if interval_seconds == 0 or not get_remote_serving_config().configured:
        return
    stop_event = Event()
    thread = Thread(
        target=_periodic_remote_sync,
        args=(stop_event, interval_seconds),
        name="lens-remote-serving-sync",
        daemon=True,
    )
    app.state.remote_sync_stop_event = stop_event
    app.state.remote_sync_thread = thread
    thread.start()
    logger.info("주기적 원격 서빙 동기화 시작: 간격=%s초", interval_seconds)


@app.on_event("shutdown")
def _stop_periodic_remote_sync() -> None:
    stop_event = getattr(app.state, "remote_sync_stop_event", None)
    thread = getattr(app.state, "remote_sync_thread", None)
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout=1)
    app.state.remote_sync_stop_event = None
    app.state.remote_sync_thread = None


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
    base = get_serving_data_dir()
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
