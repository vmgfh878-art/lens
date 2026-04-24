from __future__ import annotations

import argparse
from pathlib import Path
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
from backend.collector.pipelines.preflight import run_preflight
from backend.collector.readiness import get_indicator_readiness, get_macro_readiness
from backend.collector.repositories.base import fetch_frame
from backend.collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from backend.collector.universe import load_tickers_from_csv
from backend.collector.utils.logging import log
from backend.collector.utils.network import sanitize_proxy_env
from backend.collector.utils.observability import tracked_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 초기 역사 백필 파이프라인")
    parser.add_argument("--tickers", nargs="*", help="백필할 티커 목록")
    parser.add_argument("--skip-preflight", action="store_true", help="사전 점검을 건너뛴다")
    parser.add_argument("--skip-fundamentals", action="store_true", help="재무 데이터 백필을 건너뛴다")
    parser.add_argument("--skip-breadth", action="store_true", help="market_breadth 백필을 건너뛴다")
    parser.add_argument("--skip-indicators", action="store_true", help="indicators 백필을 건너뛴다")
    parser.add_argument("--indicator-batch-size", type=int, default=None, help="indicators 백필 티커 배치 크기")
    parser.add_argument("--max-indicator-batches", type=int, default=None, help="이번 실행에서 처리할 indicators 배치 수")
    parser.add_argument("--breadth-window-days", type=int, default=365, help="breadth 백필 출력 구간 길이")
    return parser.parse_args()


def _chunked(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _summary(target_tickers: list[str]) -> dict:
    target_set = set(target_tickers)

    stock_info = fetch_frame("stock_info", columns="ticker", order_by="ticker")
    fundamentals = fetch_frame("company_fundamentals", columns="ticker", order_by="ticker")
    price_state = get_job_state_map("sync_prices")
    indicator_readiness = get_indicator_readiness(target_tickers)
    macro_readiness = get_macro_readiness()

    stock_set = set(stock_info["ticker"].tolist()) if not stock_info.empty else set()
    fund_set = set(fundamentals["ticker"].tolist()) if not fundamentals.empty else set()
    price_set = {
        ticker
        for ticker in target_set
        if price_state.get(ticker, {}).get("status") == "success"
    }

    missing_stock = sorted(target_set - stock_set)
    missing_fund = sorted(target_set - fund_set)
    missing_price = sorted(target_set - price_set)

    return {
        "target_count": len(target_set),
        "stock_info_count": len(stock_set & target_set),
        "fundamentals_count": len(fund_set & target_set),
        "price_count": len(price_set & target_set),
        "indicator_ready_1d": indicator_readiness["ready_by_timeframe"]["1D"],
        "indicator_ready_1w": indicator_readiness["ready_by_timeframe"]["1W"],
        "indicator_ready_1m": indicator_readiness["ready_by_timeframe"]["1M"],
        "indicator_pending_count": len(indicator_readiness["pending_tickers"]),
        "macro_readiness": macro_readiness,
        "missing_stock_info": missing_stock[:20],
        "missing_fundamentals": missing_fund[:20],
        "missing_prices": missing_price[:20],
        "pending_indicator_tickers": indicator_readiness["pending_tickers"][:20],
        "is_complete": (
            not missing_stock
            and not missing_fund
            and not missing_price
            and len(indicator_readiness["pending_tickers"]) == 0
        ),
    }


def _run_step(step_name: str, func, *args, **kwargs) -> dict:
    with tracked_job(step_name, meta={"kind": "collector_step"}) as job:
        try:
            result = func(*args, **kwargs)
            job["meta"].update({"result": result})
            return result
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            log(
                f"{step_name} 실패",
                level="ERROR",
                event="bootstrap_step_failed",
                job="bootstrap_backfill",
                step=step_name,
                error=message,
            )
            upsert_job_state(
                job_name="bootstrap_backfill_step",
                target_key=step_name,
                status="failed",
                message=message,
            )
            return {"error": message}


def _run_indicator_backfill(
    target_tickers: list[str],
    batch_size: int,
    max_batches: int | None,
    settings,
) -> dict:
    readiness = get_indicator_readiness(target_tickers)
    pending_tickers = readiness["pending_tickers"]
    batches = _chunked(pending_tickers, batch_size)
    if max_batches is not None:
        batches = batches[:max_batches]

    total_stored = 0
    processed_batches = 0
    processed_tickers = 0

    for index, batch in enumerate(batches, start=1):
        with tracked_job(
            "indicator_backfill_batch",
            scope_key=f"batch-{index}",
            meta={"batch_index": index, "ticker_count": len(batch)},
        ) as job:
            result = run_indicators(
                settings.indicator_lookback_days,
                batch,
                force_full_backfill=True,
                full_start_date=settings.default_price_start,
            )
            processed_batches += 1
            processed_tickers += len(batch)
            total_stored += result["stored"]
            job["meta"].update({"result": result, "tickers": batch})
            upsert_job_state(
                job_name="indicator_backfill_progress",
                target_key="__all__",
                status="running",
                message=(
                    f"processed_batches={processed_batches}, processed_tickers={processed_tickers}, "
                    f"pending_before={len(pending_tickers)}"
                ),
                meta={
                    "batch_index": index,
                    "processed_batches": processed_batches,
                    "processed_tickers": processed_tickers,
                    "pending_before": len(pending_tickers),
                    "last_batch_tickers": batch,
                },
            )

    latest_readiness = get_indicator_readiness(target_tickers)
    upsert_job_state(
        job_name="indicator_backfill_progress",
        target_key="__all__",
        status="success" if not latest_readiness["pending_tickers"] else "partial",
        message=(
            f"processed_batches={processed_batches}, processed_tickers={processed_tickers}, "
            f"pending_after={len(latest_readiness['pending_tickers'])}"
        ),
        meta={
            "processed_batches": processed_batches,
            "processed_tickers": processed_tickers,
            "pending_after": len(latest_readiness["pending_tickers"]),
            "ready_by_timeframe": latest_readiness["ready_by_timeframe"],
        },
    )
    return {
        "stored": total_stored,
        "processed_batches": processed_batches,
        "processed_tickers": processed_tickers,
        "pending_after": len(latest_readiness["pending_tickers"]),
        "ready_by_timeframe": latest_readiness["ready_by_timeframe"],
    }


def main() -> None:
    load_dotenv()
    sanitize_proxy_env()
    args = parse_args()
    settings = get_settings()

    with tracked_job("bootstrap_backfill") as job:
        if not args.skip_preflight:
            preflight = run_preflight()
            if not preflight.ok:
                raise SystemExit(f"[preflight] {preflight.message}")
            log("사전 점검 완료", event="preflight_ok", job="bootstrap_backfill", message_text=preflight.message)

        if not settings.eodhd_api_key:
            raise SystemExit("EODHD_API_KEY가 없어 초기 가격 백필을 시작할 수 없습니다.")

        target_tickers = resolve_target_tickers(
            args.tickers,
            load_tickers_from_csv(settings.universe_file),
        )
        if not target_tickers:
            raise SystemExit("유니버스 파일에서 티커를 읽지 못했습니다.")

        before = _summary(target_tickers)
        log("초기 백필 시작", event="bootstrap_backfill_started", job="bootstrap_backfill", summary=before)

        results: dict[str, dict] = {}
        quota_hit = False

        results["macro"] = _run_step(
            "sync_macro",
            run_macro,
            settings.macro_lookback_days,
            settings.fred_api_key,
            settings.fmp_api_key,
            repair_mode=True,
            full_start_date=settings.macro_backfill_start,
        )

        if before["stock_info_count"] < before["target_count"]:
            results["stock_info"] = _run_step(
                "sync_stock_info",
                run_stock_info,
                target_tickers,
                settings.stock_info_sleep_seconds,
                settings.fmp_api_key,
                settings.stock_info_batch_limit,
                settings.allow_yahoo_fallback,
            )
            quota_hit = quota_hit or results["stock_info"].get("quota_hit", False)

        middle = _summary(target_tickers)
        if not args.skip_fundamentals and not quota_hit and middle["fundamentals_count"] < middle["target_count"]:
            results["fundamentals"] = _run_step(
                "sync_fundamentals",
                run_fundamentals,
                target_tickers,
                settings.fmp_api_key,
                settings.fmp_daily_limit,
                settings.fundamentals_sleep_seconds,
                settings.use_yahoo_fundamentals_baseline,
            )
            quota_hit = quota_hit or results["fundamentals"].get("quota_hit", False)

        middle = _summary(target_tickers)
        if middle["price_count"] < middle["target_count"]:
            results["prices"] = _run_step(
                "sync_prices",
                run_prices,
                target_tickers,
                default_start=settings.default_price_start,
                lookback_days=settings.price_lookback_days,
                repair_mode=True,
                eodhd_api_key=settings.eodhd_api_key,
                batch_limit=len(target_tickers),
                sleep_seconds=settings.price_sleep_seconds,
                allow_yahoo_fallback=settings.allow_yahoo_fallback,
                require_fundamentals=False,
            )
            quota_hit = quota_hit or results["prices"].get("quota_hit", False)

        results["sector_returns"] = _run_step(
            "sync_sector_returns",
            run_sector_returns,
            settings.sector_lookback_days,
        )
        if not args.skip_breadth:
            results["market_breadth"] = _run_step(
                "compute_market_breadth",
                run_market_breadth,
                repair_mode=True,
                min_ticker_count=settings.breadth_min_tickers,
                repair_window_days=args.breadth_window_days,
            )
        if not args.skip_indicators:
            results["indicators"] = _run_indicator_backfill(
                target_tickers=target_tickers,
                batch_size=args.indicator_batch_size or settings.indicator_backfill_batch_size,
                max_batches=args.max_indicator_batches,
                settings=settings,
            )

        after = _summary(target_tickers)
        has_errors = any("error" in result for result in results.values())
        overall_status = "success" if after["is_complete"] and not has_errors else "partial"

        upsert_job_state(
            job_name="bootstrap_backfill",
            status=overall_status,
            message=(
                f"stock_info={after['stock_info_count']}/{after['target_count']}, "
                f"fundamentals={after['fundamentals_count']}/{after['target_count']}, "
                f"prices={after['price_count']}/{after['target_count']}, "
                f"indicator_pending={after['indicator_pending_count']}, "
                f"quota_hit={quota_hit}, complete={after['is_complete']}, errors={has_errors}"
            ),
            meta={"results": results, "summary": after},
        )

        job["meta"].update({"results": results, "summary": after})
        log("초기 백필 종료", event="bootstrap_backfill_finished", job="bootstrap_backfill", summary=after)


if __name__ == "__main__":
    main()
