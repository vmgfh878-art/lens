from __future__ import annotations

from pathlib import Path
from pprint import pformat
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from collector.config import get_settings
from collector.jobs.compute_indicators import run as run_indicators
from collector.jobs.compute_market_breadth import run as run_market_breadth
from collector.jobs.sync_fundamentals import run as run_fundamentals
from collector.jobs.sync_macro import run as run_macro
from collector.jobs.sync_prices import run as run_prices
from collector.jobs.sync_sector_returns import run as run_sector_returns
from collector.jobs.sync_stock_info import run as run_stock_info
from collector.pipelines.preflight import run_preflight
from collector.repositories.base import fetch_frame
from collector.repositories.sync_state_repo import upsert_job_state
from collector.universe import load_tickers_from_csv
from collector.utils.logging import log
from collector.utils.network import sanitize_proxy_env


def _summary(target_tickers: list[str]) -> dict:
    target_set = set(target_tickers)

    stock_info = fetch_frame("stock_info", columns="ticker", order_by="ticker")
    fundamentals = fetch_frame("company_fundamentals", columns="ticker", order_by="ticker")
    prices = fetch_frame("price_data", columns="ticker", order_by="ticker")

    stock_set = set(stock_info["ticker"].tolist()) if not stock_info.empty else set()
    fund_set = set(fundamentals["ticker"].tolist()) if not fundamentals.empty else set()
    price_set = set(prices["ticker"].tolist()) if not prices.empty else set()

    missing_stock = sorted(target_set - stock_set)
    missing_fund = sorted(target_set - fund_set)
    missing_price = sorted(target_set - price_set)

    return {
        "target_count": len(target_set),
        "stock_info_count": len(stock_set & target_set),
        "fundamentals_count": len(fund_set & target_set),
        "price_count": len(price_set & target_set),
        "missing_stock_info": missing_stock[:20],
        "missing_fundamentals": missing_fund[:20],
        "missing_prices": missing_price[:20],
        "is_complete": not missing_stock and not missing_fund and not missing_price,
    }


def main() -> None:
    load_dotenv()
    sanitize_proxy_env()
    settings = get_settings()

    preflight = run_preflight()
    if not preflight.ok:
        raise SystemExit(f"[preflight] {preflight.message}")
    log(f"[preflight] {preflight.message}")

    target_tickers = load_tickers_from_csv(settings.universe_file)
    if not target_tickers:
        raise SystemExit("유니버스 파일에서 티커를 읽지 못했습니다.")

    before = _summary(target_tickers)
    log("=" * 60)
    log("Lens 초기 백필 시작")
    log(f"대상 티커 수: {before['target_count']}개")
    log(
        "현재 진척: "
        f"stock_info={before['stock_info_count']}, "
        f"fundamentals={before['fundamentals_count']}, "
        f"price_data={before['price_count']}"
    )
    log("=" * 60)

    results: dict[str, dict] = {}
    quota_hit = False

    results["macro"] = run_macro(
        settings.macro_lookback_days,
        settings.fred_api_key,
        settings.fmp_api_key,
    )

    if not before["is_complete"]:
        results["stock_info"] = run_stock_info(
            target_tickers,
            settings.stock_info_sleep_seconds,
            settings.fmp_api_key,
            settings.stock_info_batch_limit,
            settings.allow_yahoo_fallback,
        )
        quota_hit = quota_hit or results["stock_info"].get("quota_hit", False)

    middle = _summary(target_tickers)
    if not quota_hit and middle["fundamentals_count"] < middle["target_count"]:
        results["fundamentals"] = run_fundamentals(
            target_tickers,
            settings.fmp_api_key,
            settings.fmp_daily_limit,
            settings.fundamentals_sleep_seconds,
            settings.use_yahoo_fundamentals_baseline,
        )
        quota_hit = quota_hit or results["fundamentals"].get("quota_hit", False)

    middle = _summary(target_tickers)
    if not quota_hit and middle["price_count"] < middle["target_count"]:
        results["prices"] = run_prices(
            target_tickers,
            default_start=settings.default_price_start,
            lookback_days=settings.price_lookback_days,
            repair_mode=True,
            fmp_api_key=settings.fmp_api_key,
            batch_limit=settings.price_batch_limit,
            sleep_seconds=settings.price_sleep_seconds,
            allow_yahoo_fallback=settings.allow_yahoo_fallback,
            require_fundamentals=True,
        )
        quota_hit = quota_hit or results["prices"].get("quota_hit", False)

    # 파생 테이블은 현재까지 쌓인 raw 기준으로 매번 최신화한다.
    results["sector_returns"] = run_sector_returns(settings.sector_lookback_days)
    results["market_breadth"] = run_market_breadth(
        repair_mode=True,
        min_ticker_count=settings.breadth_min_tickers,
    )
    results["indicators"] = run_indicators(settings.indicator_lookback_days, target_tickers)

    after = _summary(target_tickers)
    upsert_job_state(
        job_name="bootstrap_backfill",
        status="success",
        message=(
            f"stock_info={after['stock_info_count']}/{after['target_count']}, "
            f"fundamentals={after['fundamentals_count']}/{after['target_count']}, "
            f"prices={after['price_count']}/{after['target_count']}, "
            f"quota_hit={quota_hit}, complete={after['is_complete']}"
        ),
        meta={"results": results, "summary": after},
    )

    log("=" * 60)
    log("Lens 초기 백필 완료")
    log(pformat(results, sort_dicts=False))
    log(pformat(after, sort_dicts=False))
    log("=" * 60)


if __name__ == "__main__":
    main()
