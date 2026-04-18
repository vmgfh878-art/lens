from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from collector.config import get_settings, resolve_target_tickers
from collector.jobs.compute_indicators import run as run_indicators
from collector.jobs.compute_market_breadth import run as run_market_breadth
from collector.jobs.sync_fundamentals import run as run_fundamentals
from collector.jobs.sync_macro import run as run_macro
from collector.jobs.sync_prices import run as run_prices
from collector.jobs.sync_sector_returns import run as run_sector_returns
from collector.jobs.sync_stock_info import run as run_stock_info
from collector.pipelines.daily_market_sync import run_pipeline as run_daily_market_pipeline
from collector.pipelines.preflight import run_preflight
from collector.repositories.base import list_known_tickers
from collector.repositories.sync_state_repo import upsert_job_state
from collector.universe import load_tickers_from_csv
from collector.utils.logging import log
from collector.utils.network import sanitize_proxy_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 통합 일일 수집 파이프라인")
    parser.add_argument("--tickers", nargs="*", help="동기화할 티커 목록")
    parser.add_argument("--repair", action="store_true", help="가격과 breadth를 전체 범위로 다시 계산")
    parser.add_argument("--skip-stock-info", action="store_true", help="stock_info 동기화를 건너뛴다")
    parser.add_argument("--skip-macro", action="store_true", help="거시 지표 동기화를 건너뛴다")
    parser.add_argument("--skip-fundamentals", action="store_true", help="재무 지표 동기화를 건너뛴다")
    parser.add_argument("--skip-prices", action="store_true", help="가격 동기화를 건너뛴다")
    parser.add_argument("--skip-sector", action="store_true", help="sector_returns 동기화를 건너뛴다")
    parser.add_argument("--skip-breadth", action="store_true", help="market_breadth 계산을 건너뛴다")
    parser.add_argument("--skip-indicators", action="store_true", help="indicators 계산을 건너뛴다")
    parser.add_argument("--skip-preflight", action="store_true", help="실행 전 사전 점검을 건너뛴다")
    parser.add_argument("--market-only", action="store_true", help="EODHD 가격 중심 일일 시장 파이프라인만 실행")
    parser.add_argument("--min-coverage-pct", type=float, default=0.9, help="market-only 모드 커버리지 최소 비율")
    parser.add_argument("--min-covered-tickers", type=int, default=None, help="market-only 모드 최소 커버 종목 수")
    parser.add_argument("--price-batch-limit", type=int, default=None, help="market-only 모드 가격 배치 수 override")
    parser.add_argument("--price-lookback-days", type=int, default=None, help="market-only 모드 가격 lookback 버퍼 override")
    parser.add_argument("--breadth-min-tickers", type=int, default=None, help="market-only 모드 breadth 최소 종목 수 override")
    parser.add_argument("--sector-lookback-days", type=int, default=None, help="market-only 모드 sector lookback override")
    parser.add_argument("--indicator-lookback-days", type=int, default=400, help="market-only 모드 indicators lookback")
    parser.add_argument("--allow-partial", action="store_true", help="market-only 모드에서 커버리지가 낮아도 후속 계산 진행")
    return parser.parse_args()


def _run_step(step_name: str, runner, *args, **kwargs) -> dict:
    log(f"[step] {step_name} 시작")
    try:
        result = runner(*args, **kwargs)
        log(f"[step] {step_name} 완료")
        return result
    except Exception as exc:
        upsert_job_state(
            job_name=step_name,
            status="failed",
            message=str(exc),
        )
        raise


def main() -> None:
    load_dotenv()
    sanitize_proxy_env()
    args = parse_args()
    settings = get_settings()

    if args.market_only:
        run_daily_market_pipeline(args)
        return

    if not args.skip_preflight:
        preflight = run_preflight()
        if not preflight.ok:
            raise SystemExit(f"[preflight] {preflight.message}")
        log(f"[preflight] {preflight.message}")

    universe_tickers = load_tickers_from_csv(settings.universe_file)
    known_tickers = list_known_tickers()
    target_tickers = resolve_target_tickers(args.tickers, universe_tickers or known_tickers)

    log("=" * 60)
    log("Lens 통합 일일 수집 시작")
    log(f"대상 티커 수: {len(target_tickers)}개")
    if args.tickers:
        log("대상 소스: CLI 인자")
    elif universe_tickers:
        log(f"대상 소스: 고정 유니버스 파일 ({settings.universe_file})")
    elif known_tickers:
        log("대상 소스: stock_info 기존 티커")
    else:
        log("대상 소스: 기본 Big Tech 목록")
    log("=" * 60)

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

    log("=" * 60)
    log("Lens 통합 일일 수집 완료")
    log(pformat(results, sort_dicts=False))
    log("=" * 60)


if __name__ == "__main__":
    main()
