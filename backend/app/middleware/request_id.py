"""
CP228 — 자체 request_id 미들웨어 (structlog bind 추가).

기존 동작 보존:
  - 헤더 X-Request-Id 가 있으면 그대로, 없으면 uuid4().
  - request.state.request_id 에 set, 응답 헤더에 echo.

CP228 추가:
  - structlog.contextvars.bind_contextvars(request_id=...) 호출로 service / repo
    의 structlog 로거가 라인에 자동 머지. ContextVar 기반이라 ASGI 동시 요청
    격리 안전.
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import Request


async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid4())
    request.state.request_id = request_id

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response
