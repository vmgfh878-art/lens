from __future__ import annotations

import os


def sanitize_proxy_env() -> None:
    """잘못 잡힌 로컬 프록시가 있으면 제거한다."""
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    blocked_prefixes = (
        "http://127.0.0.1:9",
        "https://127.0.0.1:9",
        "127.0.0.1:9",
    )

    for key in proxy_keys:
        value = os.environ.get(key)
        if not value:
            continue
        normalized = value.strip().lower()
        if any(normalized.startswith(prefix) for prefix in blocked_prefixes):
            os.environ.pop(key, None)
