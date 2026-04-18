from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.db import get_supabase
from collector.config import get_settings
from collector.repositories.base import fetch_frame
from collector.utils.logging import log
from collector.utils.network import sanitize_proxy_env


REQUIRED_TABLES = (
    "stock_info",
    "price_data",
    "macroeconomic_indicators",
    "company_fundamentals",
    "sector_returns",
    "market_breadth",
    "indicators",
    "sync_state",
)


@dataclass
class PreflightResult:
    ok: bool
    message: str
    details: dict[str, object]


def _check_supabase_tables() -> tuple[bool, list[str]]:
    client = get_supabase()
    missing_or_failed: list[str] = []
    for table in REQUIRED_TABLES:
        try:
            client.table(table).select("*").limit(1).execute()
        except Exception:
            missing_or_failed.append(table)
    return (len(missing_or_failed) == 0, missing_or_failed)


def run_preflight() -> PreflightResult:
    """실행 전 환경과 DB 접근 가능 여부를 점검한다."""
    load_dotenv()
    sanitize_proxy_env()
    settings = get_settings()

    details: dict[str, object] = {
        "fred_api_key": bool(settings.fred_api_key),
        "fmp_api_key": bool(settings.fmp_api_key),
    }

    try:
        table_ok, failed_tables = _check_supabase_tables()
    except Exception as exc:
        return PreflightResult(
            ok=False,
            message=f"Supabase 연결 확인 실패: {exc}",
            details=details,
        )

    details["required_tables_ok"] = table_ok
    details["failed_tables"] = failed_tables

    try:
        known_tickers = fetch_frame("stock_info", columns="ticker", limit=5)
        details["sample_ticker_count"] = len(known_tickers)
        details["sample_tickers"] = known_tickers["ticker"].tolist() if not known_tickers.empty else []
    except Exception as exc:
        return PreflightResult(
            ok=False,
            message=f"stock_info 조회 확인 실패: {exc}",
            details=details,
        )

    if not table_ok:
        return PreflightResult(
            ok=False,
            message=f"필수 테이블 접근 실패: {', '.join(failed_tables)}",
            details=details,
        )

    if not settings.fred_api_key:
        log("[preflight] FRED_API_KEY가 없어 거시 데이터는 Yahoo 보조 지표 위주로 채워집니다.")
    if not settings.fmp_api_key:
        log("[preflight] FMP_API_KEY가 없어 재무 데이터는 Yahoo baseline까지만 채워집니다.")

    return PreflightResult(
        ok=True,
        message="사전 점검 통과",
        details=details,
    )


if __name__ == "__main__":
    result = run_preflight()
    log(result.message)
    log(str(result.details))
    if not result.ok:
        raise SystemExit(1)
