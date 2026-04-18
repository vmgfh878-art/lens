from __future__ import annotations

import time

from collector.errors import SourceLimitReachedError
from collector.repositories.base import fetch_all_rows, upsert_records
from collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from collector.sources.fundamentals import fetch_fmp_fundamentals, fetch_yahoo_fundamentals
from collector.utils.logging import log


def run(
    tickers: list[str],
    fmp_api_key: str | None,
    fmp_daily_limit: int,
    sleep_seconds: float = 0.5,
    use_yahoo_baseline: bool = False,
) -> dict:
    """재무 데이터를 배치 단위로 보강한다."""
    stock_rows = fetch_all_rows("stock_info", columns="ticker")
    available_tickers = {row["ticker"] for row in stock_rows if row.get("ticker")}
    target_tickers = [ticker for ticker in tickers if ticker in available_tickers]

    existing_rows = fetch_all_rows("company_fundamentals", columns="ticker")
    existing_tickers = {row["ticker"] for row in existing_rows if row.get("ticker")}

    baseline_targets = [ticker for ticker in target_tickers if ticker not in existing_tickers]
    baseline_records: list[dict] = []
    if use_yahoo_baseline and not fmp_api_key:
        log(f"[sync_fundamentals] YF baseline 대상 {len(baseline_targets)}개")
        for ticker in baseline_targets:
            baseline_records.extend(fetch_yahoo_fundamentals(ticker))
            time.sleep(sleep_seconds)
        upsert_records("company_fundamentals", baseline_records, on_conflict="ticker,date")

    existing_rows = fetch_all_rows("company_fundamentals", columns="ticker")
    existing_tickers = {row["ticker"] for row in existing_rows if row.get("ticker")}

    upgraded_targets: list[str] = []
    fmp_records: list[dict] = []
    quota_hit = False
    if fmp_api_key:
        state_map = get_job_state_map("sync_fundamentals_fmp")
        pending_targets = [
            ticker
            for ticker in target_tickers
            if state_map.get(ticker, {}).get("status") != "success" or ticker not in existing_tickers
        ][:fmp_daily_limit]
        log(f"[sync_fundamentals] FMP 업그레이드 대상 {len(pending_targets)}개")

        for ticker in pending_targets:
            try:
                rows = fetch_fmp_fundamentals(ticker, fmp_api_key, limit=5)
            except SourceLimitReachedError:
                quota_hit = True
                break

            if rows:
                fmp_records.extend(rows)
                upgraded_targets.append(ticker)
                upsert_job_state(
                    job_name="sync_fundamentals_fmp",
                    target_key=ticker,
                    status="success",
                    message=f"stored={len(rows)}",
                )
            else:
                upsert_job_state(
                    job_name="sync_fundamentals_fmp",
                    target_key=ticker,
                    status="failed",
                    message="stored=0",
                )
            time.sleep(sleep_seconds)

        upsert_records("company_fundamentals", fmp_records, on_conflict="ticker,date")

    upsert_job_state(
        job_name="sync_fundamentals",
        status="success",
        message=(
            f"baseline_targets={len(baseline_targets)}, baseline_rows={len(baseline_records)}, "
            f"fmp_targets={len(upgraded_targets)}, fmp_rows={len(fmp_records)}, quota_hit={quota_hit}"
        ),
    )
    return {
        "baseline_targets": len(baseline_targets),
        "baseline_rows": len(baseline_records),
        "fmp_targets": len(upgraded_targets),
        "fmp_rows": len(fmp_records),
        "quota_hit": quota_hit,
        "eligible_tickers": len(target_tickers),
    }
