from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.collector.config import get_settings, resolve_target_tickers
from backend.collector.jobs.compute_indicators import run as run_indicators
from backend.collector.jobs.compute_market_breadth import run as run_market_breadth
from backend.collector.jobs.sync_fundamentals import run as run_fundamentals
from backend.collector.jobs.sync_macro import run as run_macro
from backend.collector.jobs.sync_prices import run as run_prices
from backend.collector.jobs.sync_sector_returns import run as run_sector_returns
from backend.collector.jobs.sync_stock_info import run as run_stock_info
from backend.collector.pipelines.daily_market_sync import run_pipeline as run_daily_market_pipeline
from backend.collector.pipelines.preflight import run_preflight
from backend.collector.repositories.base import list_known_tickers
from backend.collector.repositories.sync_state_repo import upsert_job_state
from backend.collector.universe import load_tickers_from_csv
from backend.collector.utils.logging import log
from backend.collector.utils.network import sanitize_proxy_env
from backend.collector.utils.observability import tracked_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 통합 일일 수집 파이프라인")
    parser.add_argument("--tickers", nargs="*", help="동기화할 티커 목록")
    parser.add_argument("--repair", action="store_true", help="가격과 breadth를 전체 범위로 다시 계산한다")
    parser.add_argument("--skip-stock-info", action="store_true", help="stock_info 동기화를 건너뛴다")
    parser.add_argument("--skip-macro", action="store_true", help="거시 지표 동기화를 건너뛴다")
    parser.add_argument("--skip-fundamentals", action="store_true", help="재무 지표 동기화를 건너뛴다")
    parser.add_argument("--skip-prices", action="store_true", help="가격 동기화를 건너뛴다")
    parser.add_argument("--skip-sector", action="store_true", help="sector_returns 동기화를 건너뛴다")
    parser.add_argument("--skip-breadth", action="store_true", help="market_breadth 계산을 건너뛴다")
    parser.add_argument("--skip-indicators", action="store_true", help="indicators 계산을 건너뛴다")
    parser.add_argument("--skip-preflight", action="store_true", help="실행 전 사전 점검을 건너뛴다")
    parser.add_argument("--market-only", action="store_true", help="가격 중심 일일 시장 파이프라인만 실행한다")
    parser.add_argument("--min-coverage-pct", type=float, default=0.9, help="market-only 최소 커버리지 비율")
    parser.add_argument("--min-covered-tickers", type=int, default=None, help="market-only 최소 커버 종목 수")
    parser.add_argument("--price-batch-limit", type=int, default=None, help="market-only 가격 배치 크기 override")
    parser.add_argument("--price-lookback-days", type=int, default=None, help="market-only 가격 버퍼 일수 override")
    parser.add_argument("--breadth-min-tickers", type=int, default=None, help="market-only breadth 최소 종목 수 override")
    parser.add_argument("--sector-lookback-days", type=int, default=None, help="market-only sector lookback override")
    parser.add_argument("--indicator-lookback-days", type=int, default=14, help="market-only 지표 재계산 오버랩 일수")
    parser.add_argument("--allow-partial", action="store_true", help="market-only에서 커버리지가 부족해도 계속 진행한다")
    return parser.parse_args()


def _run_step(step_name: str, runner, *args, **kwargs) -> dict:
    with tracked_job(step_name, meta={"kind": "collector_step"}) as job:
        result = runner(*args, **kwargs)
        job["meta"].update({"result": result})
        return result


def main() -> None:
    load_dotenv()
    sanitize_proxy_env()
    args = parse_args()
    settings = get_settings()

    with tracked_job("daily_sync", meta={"repair_mode": args.repair, "market_only": args.market_only}) as job:
        if args.market_only:
            results = run_daily_market_pipeline(args)
            job["meta"].update({"results": results})
            return

        if not args.skip_preflight:
            preflight = run_preflight()
            if not preflight.ok:
                raise SystemExit(f"[preflight] {preflight.message}")
            log("사전 점검 완료", event="preflight_ok", job="daily_sync", message_text=preflight.message)

        universe_tickers = load_tickers_from_csv(settings.universe_file)
        known_tickers = list_known_tickers()
        target_tickers = resolve_target_tickers(args.tickers, universe_tickers or known_tickers)

        log(
            "통합 일일 수집 시작",
            event="daily_sync_started",
            job="daily_sync",
            ticker_count=len(target_tickers),
        )

        results: dict[str, dict] = {}

        if not args.skip_macro:
            results["macro"] = _run_step(
                "sync_macro",
                run_macro,
                settings.macro_lookback_days,
                settings.fred_api_key,
                settings.fmp_api_key,
            )
        if not args.skip_stock_info:
            results["stock_info"] = _run_step(
                "sync_stock_info",
                run_stock_info,
                target_tickers,
                settings.stock_info_sleep_seconds,
                settings.fmp_api_key,
                settings.stock_info_batch_limit,
                settings.allow_yahoo_fallback,
            )
        if not args.skip_fundamentals:
            results["fundamentals"] = _run_step(
                "sync_fundamentals",
                run_fundamentals,
                target_tickers,
                settings.fmp_api_key,
                settings.fmp_daily_limit,
                settings.fundamentals_sleep_seconds,
                settings.use_yahoo_fundamentals_baseline,
            )
        if not args.skip_prices:
            if not settings.eodhd_api_key:
                raise SystemExit("EODHD_API_KEY가 없어 가격 동기화를 시작할 수 없습니다.")
            results["prices"] = _run_step(
                "sync_prices",
                run_prices,
                target_tickers,
                default_start=settings.default_price_start,
                lookback_days=settings.price_lookback_days,
                repair_mode=args.repair,
                eodhd_api_key=settings.eodhd_api_key,
                batch_limit=settings.price_batch_limit,
                sleep_seconds=settings.price_sleep_seconds,
                allow_yahoo_fallback=settings.allow_yahoo_fallback,
            )
        if not args.skip_sector:
            results["sector_returns"] = _run_step(
                "sync_sector_returns",
                run_sector_returns,
                settings.sector_lookback_days,
            )
        if not args.skip_breadth:
            results["market_breadth"] = _run_step(
                "compute_market_breadth",
                run_market_breadth,
                repair_mode=args.repair,
                min_ticker_count=settings.breadth_min_tickers,
            )
        if not args.skip_indicators:
            results["indicators"] = _run_step(
                "compute_indicators",
                run_indicators,
                settings.indicator_lookback_days,
                target_tickers,
            )

        upsert_job_state(
            job_name="daily_sync",
            status="success",
            message=f"completed_steps={','.join(results.keys())}",
            meta=results,
        )
        job["meta"].update({"results": results})
        log("통합 일일 수집 완료", event="daily_sync_finished", job="daily_sync")


if __name__ == "__main__":
    main()
