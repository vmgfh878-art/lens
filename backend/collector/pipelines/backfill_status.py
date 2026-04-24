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
from backend.collector.repositories.base import fetch_frame, get_latest_date, list_known_tickers
from backend.collector.repositories.sync_state_repo import get_job_state_map
from backend.collector.universe import load_tickers_from_csv


def build_status(target_tickers: list[str]) -> dict:
    price_state = get_job_state_map("sync_prices")
    latest_price_date = get_latest_date("price_data")
    earliest_price_frame = fetch_frame("price_data", columns="date", order_by="date", limit=1)
    breadth_frame = fetch_frame("market_breadth", columns="date", order_by="date")
    indicator_readiness = get_indicator_readiness(target_tickers)
    macro_readiness = get_macro_readiness()
    indicator_state = {
        job_name: len(get_job_state_map(job_name))
        for job_name in ("compute_indicators:1D", "compute_indicators:1W", "compute_indicators:1M")
    }

    return {
        "target_count": len(target_tickers),
        "price": {
            "tickers": sum(1 for row in price_state.values() if row.get("status") == "success"),
            "min_date": None if earliest_price_frame.empty else str(earliest_price_frame["date"].iloc[0]),
            "max_date": None if latest_price_date is None else latest_price_date.isoformat(),
        },
        "macro": macro_readiness,
        "breadth": {
            "rows": int(len(breadth_frame)),
            "min_date": None if breadth_frame.empty else str(breadth_frame["date"].min()),
            "max_date": None if breadth_frame.empty else str(breadth_frame["date"].max()),
        },
        "indicators": {
            "ready_by_timeframe": indicator_readiness["ready_by_timeframe"],
            "pending_count": len(indicator_readiness["pending_tickers"]),
            "pending_tickers_sample": indicator_readiness["pending_tickers"][:20],
            "state_counts": indicator_state,
        },
    }


def main() -> None:
    load_dotenv()
    settings = get_settings()
    target_tickers = resolve_target_tickers(
        None,
        load_tickers_from_csv(settings.universe_file) or list_known_tickers(),
    )
    print(json.dumps(build_status(target_tickers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
