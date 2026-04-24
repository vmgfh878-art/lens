from __future__ import annotations

from typing import Any

from fastapi import Request


def build_meta(request: Request, **extra: Any) -> dict[str, Any]:
    meta = {
        "request_id": getattr(request.state, "request_id", ""),
    }
    meta.update({key: value for key, value in extra.items() if value is not None})
    return meta


def success_response(request: Request, data: Any, **meta: Any) -> dict[str, Any]:
    return {
        "data": data,
        "meta": build_meta(request, **meta),
    }


def error_response(
    request: Request,
    *,
    code: str,
    message: str,
    details: Any = None,
    **meta: Any,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
        "meta": build_meta(request, **meta),
    }
