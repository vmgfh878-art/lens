from __future__ import annotations

import time

from backend.collector.errors import SourceLimitReachedError
from backend.collector.repositories.base import fetch_all_rows, upsert_records
from backend.collector.repositories.sync_state_repo import get_job_state_map, upsert_job_state
from backend.collector.sources.edgar import fetch_edgar_fundamentals
from backend.collector.sources.fundamentals import fetch_fmp_fundamentals, fetch_yahoo_fundamentals
from backend.collector.utils.logging import log


def run(
    tickers: list[str],
    fmp_api_key: str | None,
    fmp_daily_limit: int,
    sleep_seconds: float = 0.5,
    use_yahoo_baseline: bool = False,
) -> dict:
    """종목별 재무 데이터를 채운다.

    FMP가 쿼터에 막히더라도 Yahoo 기준값으로 최소 백필을 이어가서
    초기 백필이 멈추지 않도록 만든다.
    """
    stock_rows = fetch_all_rows("stock_info", columns="ticker")
    available_tickers = {row["ticker"] for row in stock_rows if row.get("ticker")}
    target_tickers = [ticker for ticker in tickers if ticker in available_tickers]

    existing_rows = fetch_all_rows("company_fundamentals", columns="ticker")
    existing_tickers = {row["ticker"] for row in existing_rows if row.get("ticker")}

    # Yahoo baseline 사전 패스는 비용 대비 수확이 거의 없어 제거함.
    # FMP → Yahoo 보강 → EDGAR 순서로 진행.
    baseline_targets = [ticker for ticker in target_tickers if ticker not in existing_tickers]
    baseline_records: list[dict] = []

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
                rows, reason = fetch_fmp_fundamentals(ticker, fmp_api_key, limit=5)
            except SourceLimitReachedError as exc:
                quota_hit = True
                log(
                    f"[sync_fundamentals] FMP {exc.reason} 감지, Yahoo 보강으로 전환",
                    level="WARNING",
                )
                upsert_job_state(
                    job_name="sync_fundamentals_fmp",
                    target_key=ticker,
                    status="failed",
                    message=f"{exc.reason}: {str(exc)[:120]}",
                    meta={"source": "fmp", "reason": exc.reason},
                )
                break

            if rows:
                fmp_records.extend(rows)
                upgraded_targets.append(ticker)
                upsert_job_state(
                    job_name="sync_fundamentals_fmp",
                    target_key=ticker,
                    status="success",
                    message=f"stored={len(rows)}",
                    meta={"source": "fmp", "reason": "ok", "rows": len(rows)},
                )
            else:
                upsert_job_state(
                    job_name="sync_fundamentals_fmp",
                    target_key=ticker,
                    status="failed",
                    message=f"stored=0 ({reason})",
                    meta={"source": "fmp", "reason": reason},
                )
            time.sleep(sleep_seconds)

        upsert_records("company_fundamentals", fmp_records, on_conflict="ticker,date")

    # FMP가 막혀도 남은 종목은 Yahoo 기준값으로 채워 백필이 중단되지 않게 한다.
    if use_yahoo_baseline:
        existing_rows = fetch_all_rows("company_fundamentals", columns="ticker")
        existing_tickers = {row["ticker"] for row in existing_rows if row.get("ticker")}
        fallback_targets = [ticker for ticker in target_tickers if ticker not in existing_tickers]
        fallback_records: list[dict] = []
        yahoo_success_count = 0
        if fallback_targets:
            log(f"[sync_fundamentals] Yahoo 보강 대상 {len(fallback_targets)}개")
        for ticker in fallback_targets:
            try:
                rows = fetch_yahoo_fundamentals(ticker)
            except Exception as exc:
                upsert_job_state(
                    job_name="sync_fundamentals_yahoo",
                    target_key=ticker,
                    status="failed",
                    message=f"error: {str(exc)[:120]}",
                    meta={"source": "yahoo", "reason": "error"},
                )
                time.sleep(sleep_seconds)
                continue

            if rows:
                fallback_records.extend(rows)
                yahoo_success_count += 1
                upsert_job_state(
                    job_name="sync_fundamentals_yahoo",
                    target_key=ticker,
                    status="success",
                    message=f"stored={len(rows)}",
                    meta={"source": "yahoo", "reason": "ok", "rows": len(rows)},
                )
            else:
                upsert_job_state(
                    job_name="sync_fundamentals_yahoo",
                    target_key=ticker,
                    status="failed",
                    message="stored=0 (empty)",
                    meta={"source": "yahoo", "reason": "empty"},
                )
            time.sleep(sleep_seconds)
        if fallback_records:
            upsert_records("company_fundamentals", fallback_records, on_conflict="ticker,date")
            baseline_records.extend(fallback_records)

    existing_rows = fetch_all_rows("company_fundamentals", columns="ticker")
    existing_tickers = {row["ticker"] for row in existing_rows if row.get("ticker")}
    edgar_targets = [ticker for ticker in target_tickers if ticker not in existing_tickers]
    edgar_records: list[dict] = []
    edgar_success_count = 0
    if edgar_targets:
        log(f"[sync_fundamentals] EDGAR 보강 대상 {len(edgar_targets)}개")
        for ticker in edgar_targets:
            try:
                rows = fetch_edgar_fundamentals(ticker)
            except Exception as exc:  # EDGAR 레이트리밋·네트워크 등
                upsert_job_state(
                    job_name="sync_fundamentals_edgar",
                    target_key=ticker,
                    status="failed",
                    message=f"error: {str(exc)[:120]}",
                    meta={"source": "edgar", "reason": "error"},
                )
                time.sleep(sleep_seconds)
                continue

            if rows:
                edgar_records.extend(rows)
                edgar_success_count += 1
                upsert_job_state(
                    job_name="sync_fundamentals_edgar",
                    target_key=ticker,
                    status="success",
                    message=f"stored={len(rows)}",
                    meta={"source": "edgar", "reason": "ok", "rows": len(rows)},
                )
            else:
                upsert_job_state(
                    job_name="sync_fundamentals_edgar",
                    target_key=ticker,
                    status="failed",
                    message="stored=0 (empty)",
                    meta={"source": "edgar", "reason": "empty"},
                )
            time.sleep(sleep_seconds)
        if edgar_records:
            upsert_records("company_fundamentals", edgar_records, on_conflict="ticker,date")

    upsert_job_state(
        job_name="sync_fundamentals",
        status="success",
        message=(
            f"baseline_targets={len(baseline_targets)}, baseline_rows={len(baseline_records)}, "
            f"fmp_targets={len(upgraded_targets)}, fmp_rows={len(fmp_records)}, "
            f"edgar_targets={len(edgar_targets)}, edgar_rows={len(edgar_records)}, quota_hit={quota_hit}"
        ),
    )
    return {
        "baseline_targets": len(baseline_targets),
        "baseline_rows": len(baseline_records),
        "fmp_targets": len(upgraded_targets),
        "fmp_rows": len(fmp_records),
        "edgar_targets": len(edgar_targets),
        "edgar_rows": len(edgar_records),
        "edgar_success_tickers": edgar_success_count,
        "quota_hit": quota_hit,
        "eligible_tickers": len(target_tickers),
    }
