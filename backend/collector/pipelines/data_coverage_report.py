from __future__ import annotations

import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.collector.config import get_settings, resolve_target_tickers
from backend.collector.readiness import get_indicator_readiness, get_macro_readiness
from backend.collector.repositories.base import fetch_frame
from backend.collector.repositories.sync_state_repo import get_job_state_map
from backend.collector.universe import load_tickers_from_csv


def _ticker_set(table: str, target_tickers: list[str]) -> set[str]:
    frame = fetch_frame(table, columns="ticker", filters=[("in", "ticker", target_tickers)], order_by="ticker")
    return set(frame["ticker"].tolist()) if not frame.empty else set()


def build_report(target_tickers: list[str]) -> dict:
    stock_set = _ticker_set("stock_info", target_tickers)
    fund_set = _ticker_set("company_fundamentals", target_tickers)
    price_state = get_job_state_map("sync_prices")
    price_set = {ticker for ticker in target_tickers if price_state.get(ticker, {}).get("status") == "success"}
    indicator_readiness = get_indicator_readiness(target_tickers)
    macro_readiness = get_macro_readiness()
    fundamentals_state = get_job_state_map("sync_fundamentals")
    fundamentals_fmp_state = get_job_state_map("sync_fundamentals_fmp")

    return {
        "target_count": len(target_tickers),
        "stock_info": {
            "covered": len(stock_set),
            "missing": sorted(set(target_tickers) - stock_set)[:50],
            "missing_fields_hint": ["sector", "industry", "market_cap"],
        },
        "company_fundamentals": {
            "covered": len(fund_set),
            "missing": sorted(set(target_tickers) - fund_set)[:50],
            "missing_fields_hint": [
                "revenue",
                "net_income",
                "total_assets",
                "total_liabilities",
                "equity",
                "shares_issued",
                "eps",
                "roe",
                "debt_ratio",
                "interest_coverage",
                "operating_cash_flow",
            ],
            "quota_hit": any(row.get("status") == "failed" and "limit" in str(row.get("message", "")).lower() for row in fundamentals_fmp_state.values())
            or any("quota_hit=True" in str(row.get("message", "")) for row in fundamentals_state.values()),
        },
        "price_data": {
            "covered": len(price_set),
            "missing": sorted(set(target_tickers) - price_set)[:50],
            "missing_fields_hint": [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "amount",
                "per",
                "pbr",
            ],
        },
        "macro": macro_readiness,
        "indicators": {
            "ready_by_timeframe": indicator_readiness["ready_by_timeframe"],
            "pending_count": len(indicator_readiness["pending_tickers"]),
            "pending_tickers": indicator_readiness["pending_tickers"][:50],
        },
        "overall_complete": (
            len(stock_set) == len(target_tickers)
            and len(fund_set) == len(target_tickers)
            and len(price_set) == len(target_tickers)
            and len(indicator_readiness["pending_tickers"]) == 0
        ),
    }


def main() -> None:
    load_dotenv()
    settings = get_settings()
    target_tickers = resolve_target_tickers(None, load_tickers_from_csv(settings.universe_file))
    print(json.dumps(build_report(target_tickers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
