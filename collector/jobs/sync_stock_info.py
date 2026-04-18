from __future__ import annotations

import time

from collector.errors import SourceLimitReachedError
from collector.repositories.base import fetch_all_rows, upsert_records
from collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from collector.sources.yahoo_stock_info import fetch_stock_info
from collector.utils.logging import log


def run(
    tickers: list[str],
    sleep_seconds: float = 0.2,
    fmp_api_key: str | None = None,
    batch_limit: int = 80,
    allow_yahoo_fallback: bool = True,
) -> dict:
    """종목 메타데이터를 배치 단위로 동기화한다."""
    existing_rows = fetch_all_rows("stock_info", columns="ticker")
    existing_tickers = {row["ticker"] for row in existing_rows if row.get("ticker")}
    state_map = get_job_state_map("sync_stock_info_item")

    pending_tickers = [
        ticker
        for ticker in tickers
        if ticker not in existing_tickers or state_map.get(ticker, {}).get("status") != "success"
    ][:batch_limit]

    records: list[dict] = []
    failed: list[str] = []
    processed = 0
    quota_hit = False

    log(f"[sync_stock_info] 대상 {len(pending_tickers)}개 종목 배치 시작")
    for ticker in pending_tickers:
        try:
            info = fetch_stock_info(
                ticker,
                fmp_api_key=fmp_api_key,
                allow_yahoo_fallback=allow_yahoo_fallback,
            )
        except SourceLimitReachedError:
            quota_hit = True
            break

        if info is None:
            failed.append(ticker)
            upsert_job_state(
                job_name="sync_stock_info_item",
                target_key=ticker,
                status="failed",
                message="stored=0",
            )
            continue

        records.append(info)
        processed += 1
        upsert_job_state(
            job_name="sync_stock_info_item",
            target_key=ticker,
            status="success",
            message="stored=1",
        )
        time.sleep(sleep_seconds)

    upsert_records("stock_info", records, on_conflict="ticker")
    upsert_job_state(
        job_name="sync_stock_info",
        status="success",
        message=f"stored={len(records)}, failed={len(failed)}, processed={processed}, quota_hit={quota_hit}",
        meta={"failed": failed[:20]},
    )
    return {
        "stored": len(records),
        "failed": failed,
        "processed": processed,
        "quota_hit": quota_hit,
        "pending_batch": len(pending_tickers),
    }
