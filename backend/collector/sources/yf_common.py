from __future__ import annotations

import logging
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

    # CP256 — 일시적 티커 실패("Failed to get ticker ...", "1 Failed download")는
    # yfinance 가 stderr 로 로깅한다. 실패 티커는 이미 failed_tickers CSV 에
    # EMPTY/CONTRACT_FAIL 로 기록되므로 yfinance 자체 stderr 로그는 중복 노이즈다.
    # 또한 이 stderr 출력이 PowerShell 5.1(ErrorActionPreference=Stop) 에서
    # 오탐 FAIL 을 유발했으므로(일일 refresh 4일 중단), CRITICAL 로 억제한다.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    _INITIALIZED = True
