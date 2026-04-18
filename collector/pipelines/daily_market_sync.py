from __future__ import annotations

import argparse
import math
from pathlib import Path
from pprint import pformat
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from collector.config import get_settings, resolve_target_tickers
from collector.jobs.compute_indicators import run as run_indicators
from collector.jobs.compute_market_breadth import run as run_market_breadth
from collector.jobs.sync_macro import run as run_macro
from collector.jobs.sync_prices import run as run_prices
from collector.jobs.sync_sector_returns import run as run_sector_returns
from collector.pipelines.preflight import run_preflight
from collector.repositories.base import fetch_frame, get_latest_date, list_known_tickers
from collector.repositories.sync_state_repo import upsert_job_state
from collector.universe import load_tickers_from_csv
from collector.utils.logging import log
from collector.utils.network import sanitize_proxy_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 일일 시장 데이터 증분 파이프라인")
    parser.add_argument("--tickers", nargs="*", help="동기화할 티커 목록")
    parser.add_argument("--skip-preflight", action="store_true", help="실행 전 점검을 건너뜀")
    parser.add_argument("--skip-macro", action="store_true", help="거시 데이터 동기화를 건너뜀")
    parser.add_argument("--skip-derived", action="store_true", help="sector/breadth/indicators 계산을 건너뜀")
    parser.add_argument(
        "--min-coverage-pct",
        type=float,
        default=0.9,
        help="최신 가격 일자 커버리지 최소 비율. 기본값 0.9",
    )
    parser.add_argument(
        "--min-covered-tickers",
        type=int,
        default=None,
        help="최신 가격 일자 최소 커버 종목 수. 미지정 시 비율 기준으로 계산",
    )
    parser.add_argument(
        "--breadth-min-tickers",
        type=int,
        default=None,
        help="market_breadth 계산 최소 종목 수 override",
    )
    parser.add_argument(
        "--price-lookback-days",
        type=int,
        default=None,
        help="가격 증분 수집 시 티커별 재조회 버퍼 일수 override",
    )
    parser.add_argument(
        "--price-batch-limit",
        type=int,
        default=None,
        help="가격 동기화 배치 수 override. 기본은 전체 대상 티커 수",
    )
    parser.add_argument(
        "--sector-lookback-days",
        type=int,
        default=None,
        help="sector_returns 재계산 기간 override",
    )
    parser.add_argument(
        "--indicator-lookback-days",
        type=int,
        default=400,
        help="indicators 재계산 기간. 기본값 400일",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="커버리지가 낮아도 파생 계산까지 계속 진행",
    )
    return parser.parse_args()


def _build_price_coverage(target_tickers: list[str]) -> dict:
    latest_date = get_latest_date("price_data")
    if latest_date is None:
        return {
            "latest_date": None,
            "covered": 0,
            "target_count": len(target_tickers),
            "coverage_pct": 0.0,
            "covered_tickers": [],
            "missing_tickers": target_tickers[:20],
        }

    frame = fetch_frame(
        "price_data",
        columns="ticker,date",
        filters=[("eq", "date", latest_date.isoformat()), ("in", "ticker", target_tickers)],
        order_by="ticker",
    )
    covered_tickers = sorted(frame["ticker"].dropna().astype(str).unique().tolist()) if not frame.empty else []
    covered_set = set(covered_tickers)
    target_set = set(target_tickers)
    coverage_pct = (len(covered_set) / len(target_tickers)) if target_tickers else 0.0
    return {
        "latest_date": latest_date.isoformat(),
        "covered": len(covered_set),
        "target_count": len(target_tickers),
        "coverage_pct": coverage_pct,
        "covered_tickers": covered_tickers[:20],
        "missing_tickers": sorted(target_set - covered_set)[:20],
    }


def _run_step(step_name: str, runner, *args, **kwargs) -> dict:
    log(f"[step] {step_name} 시작")
    result = runner(*args, **kwargs)
    log(f"[step] {step_name} 완료")
    return result


def run_pipeline(args: argparse.Namespace) -> dict[str, dict]:
    settings = get_settings()

    if not getattr(args, "skip_preflight", False):
        preflight = run_preflight()
        if not preflight.ok:
            raise SystemExit(f"[preflight] {preflight.message}")
        log(f"[preflight] {preflight.message}")

    universe_tickers = load_tickers_from_csv(settings.universe_file)
    known_tickers = list_known_tickers()
    target_tickers = resolve_target_tickers(args.tickers, universe_tickers or known_tickers)
    price_lookback_days = args.price_lookback_days or settings.price_lookback_days
    price_batch_limit = args.price_batch_limit or len(target_tickers)
    sector_lookback_days = args.sector_lookback_days or settings.sector_lookback_days
    breadth_min_tickers = args.breadth_min_tickers or settings.breadth_min_tickers

    log("=" * 60)
    log("Lens 일일 시장 증분 동기화 시작")
    log(f"대상 티커 수: {len(target_tickers)}")
    log(f"가격 소스: Yahoo 증분")
    log(f"가격 lookback 버퍼: {price_lookback_days}일")
    log("=" * 60)

    results: dict[str, dict] = {}

    if not getattr(args, "skip_macro", False):
        results["macro"] = _run_step(
            "sync_macro",
            run_macro,
            settings.macro_lookback_days,
            settings.fred_api_key,
            settings.fmp_api_key,
        )

    results["prices"] = _run_step(
        "sync_prices",
        run_prices,
        target_tickers,
        default_start=settings.default_price_start,
        lookback_days=price_lookback_days,
        repair_mode=False,
        fmp_api_key=None,
        batch_limit=price_batch_limit,
        sleep_seconds=settings.price_sleep_seconds,
        allow_yahoo_fallback=True,
        require_fundamentals=False,
    )

    coverage = _build_price_coverage(target_tickers)
    required_count = args.min_covered_tickers or math.ceil(len(target_tickers) * args.min_coverage_pct)
    coverage_ok = coverage["covered"] >= required_count
    results["coverage"] = {
        **coverage,
        "required_count": required_count,
        "coverage_ok": coverage_ok,
    }

    if not coverage_ok and not getattr(args, "allow_partial", False):
        message = (
            f"latest_date={coverage['latest_date']}, covered={coverage['covered']}/{coverage['target_count']}, "
            f"required={required_count}, coverage_pct={coverage['coverage_pct']:.3f}"
        )
        upsert_job_state(
            job_name="daily_market_sync",
            status="failed",
            last_cursor_date=pd.to_datetime(coverage["latest_date"]).date() if coverage["latest_date"] else None,
            message=message,
            meta=results,
        )
        log("=" * 60)
        log("Lens 일일 시장 증분 동기화 중단")
        log(message)
        log("커버리지가 기준 미만이라 후속 계산을 중단했습니다.")
        log("=" * 60)
        raise SystemExit(message)

    if not getattr(args, "skip_sector", False):
        results["sector_returns"] = _run_step("sync_sector_returns", run_sector_returns, sector_lookback_days)
    if not getattr(args, "skip_breadth", False):
        results["market_breadth"] = _run_step(
            "compute_market_breadth",
            run_market_breadth,
            repair_mode=False,
            min_ticker_count=breadth_min_tickers,
        )
    if not getattr(args, "skip_indicators", False):
        results["indicators"] = _run_step("compute_indicators", run_indicators, args.indicator_lookback_days, target_tickers)

    upsert_job_state(
        job_name="daily_market_sync",
        status="success",
        last_cursor_date=pd.to_datetime(coverage["latest_date"]).date() if coverage["latest_date"] else None,
        message=(
            f"latest_date={coverage['latest_date']}, covered={coverage['covered']}/{coverage['target_count']}, "
            f"required={required_count}, coverage_pct={coverage['coverage_pct']:.3f}"
        ),
        meta=results,
    )

    log("=" * 60)
    log("Lens 일일 시장 증분 동기화 완료")
    log(pformat(results, sort_dicts=False))
    log("=" * 60)
    return results


def main() -> None:
    load_dotenv()
    sanitize_proxy_env()
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
