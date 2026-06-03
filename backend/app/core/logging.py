"""
structlog + stdlib bridge 설정.

CP228 신설. configure_logging() 호출 후:
  - structlog.get_logger(...) 로 만든 로거가 dev=Console / prod=JSON 으로 출력.
  - stdlib logging.getLogger(...) 라인도 ProcessorFormatter 로 같은 포맷 통과.
  - structlog.contextvars.merge_contextvars 가 request_id (asgi-correlation-id) 를
    각 로그 라인에 자동 머지.

환경 변수:
  - LENS_LOG_JSON: "1"/"true"/"yes" 면 JSON 출력 (prod). 미설정/기타 = Console (dev).
    LENS_ENV=production 이어도 같은 효과.
  - LENS_LOG_LEVEL: 기본 "INFO". DEBUG / INFO / WARNING / ERROR / CRITICAL.

멱등 (여러 번 호출해도 안전).
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


def _wants_json() -> bool:
    if os.environ.get("LENS_ENV", "").strip().lower() == "production":
        return True
    return os.environ.get("LENS_LOG_JSON", "").strip().lower() in {"1", "true", "yes"}


def _resolve_level() -> int:
    raw = os.environ.get("LENS_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = _resolve_level()
    use_json = _wants_json()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    # structlog 자체 로거 설정.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # stdlib logging → structlog ProcessorFormatter 브리지.
    bridge_processors: list = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        renderer,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=bridge_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # 기존 핸들러 제거 (멱등, 재구성 시 중복 출력 방지).
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    _CONFIGURED = True


__all__ = ["configure_logging"]
