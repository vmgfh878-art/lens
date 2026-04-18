from __future__ import annotations

from pathlib import Path

import yfinance as yf


_INITIALIZED = False


def prepare_yfinance() -> None:
    """샌드박스에서도 동작하도록 yfinance 캐시 위치를 워크스페이스로 고정한다."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    _INITIALIZED = True
