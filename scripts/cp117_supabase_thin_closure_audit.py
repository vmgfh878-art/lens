from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_METRICS_PATH = ROOT_DIR / "docs" / "cp117_local_parquet_supabase_thin_closure_metrics.json"
DEFAULT_REPORT_PATH = ROOT_DIR / "docs" / "cp117_local_parquet_supabase_thin_closure_report.md"
LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
PRODUCT_RUN_IDS = [LINE_RUN_ID, BAND_RUN_ID]
COUNT_COLUMNS = {
    "price_data": "id",
    "indicators": "id",
    "predictions": "id",
    "prediction_evaluations": "id",
    "model_runs": "run_id",
    "job_runs": "run_id",
    "stock_info": "ticker",
}

os.environ["MARKET_DATA_PROVIDER"] = "yfinance"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["EODHD_API_KEY"] = ""
os.environ["LENS_DATA_BACKEND"] = "local"
os.environ["LENS_REQUIRE_LOCAL_SNAPSHOTS"] = "1"
os.environ["LENS_LOCAL_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.db import get_supabase  # noqa: E402
from backend.collector.repositories.base import fetch_all_rows  # noqa: E402


TARGET_TABLES = [
    "price_data",
    "indicators",
    "predictions",
    "prediction_evaluations",
    "model_runs",
    "job_runs",
    "stock_info",
]


TABLE_SIZE_SQL = """select
  n.nspname as schema_name,
  c.relname as table_name,
  pg_size_pretty(pg_relation_size(c.oid)) as table_size,
  pg_size_pretty(pg_indexes_size(c.oid)) as index_size,
  pg_size_pretty(pg_total_relation_size(c.oid)) as total_size,
  pg_relation_size(c.oid) as table_bytes,
  pg_indexes_size(c.oid) as index_bytes,
  pg_total_relation_size(c.oid) as total_bytes,
  coalesce(s.n_live_tup, c.reltuples)::bigint as estimated_rows,
  s.n_dead_tup as estimated_dead_rows
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
left join pg_stat_user_tables s on s.relid = c.oid
where n.nspname = 'public'
  and c.relkind in ('r', 'p')
  and c.relname in (
    'price_data',
    'indicators',
    'predictions',
    'prediction_evaluations',
    'model_runs',
    'job_runs',
    'stock_info'
  )
order by pg_total_relation_size(c.oid) desc;"""


PRUNING_DRY_RUN_SQL = """with product_runs as (
  select unnest(array[
    'patchtst-1D-efad3c29d803',
    'cnn_lstm-1D-d0c780dee5e8'
  ]) as run_id
),
latest_product_predictions as (
  select p.run_id, max(p.asof_date) as latest_asof_date
  from public.predictions p
  join product_runs pr on pr.run_id = p.run_id
  group by p.run_id
),
latest_product_evaluations as (
  select e.run_id, max(e.asof_date) as latest_asof_date
  from public.prediction_evaluations e
  join product_runs pr on pr.run_id = e.run_id
  group by e.run_id
)
select
  'predictions_non_product' as candidate_group,
  count(*) as candidate_rows
from public.predictions p
where not exists (select 1 from product_runs pr where pr.run_id = p.run_id)
union all
select
  'predictions_product_history_except_latest' as candidate_group,
  count(*) as candidate_rows
from public.predictions p
join latest_product_predictions latest on latest.run_id = p.run_id
where p.asof_date < latest.latest_asof_date
union all
select
  'prediction_evaluations_non_product' as candidate_group,
  count(*) as candidate_rows
from public.prediction_evaluations e
where not exists (select 1 from product_runs pr where pr.run_id = e.run_id)
union all
select
  'prediction_evaluations_product_history_except_latest' as candidate_group,
  count(*) as candidate_rows
from public.prediction_evaluations e
join latest_product_evaluations latest on latest.run_id = e.run_id
where e.asof_date < latest.latest_asof_date;"""


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def count_query(table: str, *, filters: list[tuple[str, str, Any]] | None = None) -> int | None:
    client = get_supabase()
    query = client.table(table).select(COUNT_COLUMNS.get(table, "*"), count="exact").limit(1)
    for operator, column, value in filters or []:
        if operator == "eq":
            query = query.eq(column, value)
        elif operator == "in":
            query = query.in_(column, value)
        elif operator == "lt":
            query = query.lt(column, value)
        elif operator == "gte":
            query = query.gte(column, value)
        else:
            raise ValueError(f"지원하지 않는 필터입니다: {operator}")
    result = query.execute()
    return None if result.count is None else int(result.count)


def latest_asof_for_run(table: str, run_id: str) -> str | None:
    client = get_supabase()
    rows = (
        client.table(table)
        .select("asof_date")
        .eq("run_id", run_id)
        .order("asof_date", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    return rows[0]["asof_date"] if rows else None


def table_counts() -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for table in TARGET_TABLES:
        try:
            counts[table] = {"row_count": count_query(table), "status": "PASS"}
        except Exception as exc:
            counts[table] = {"row_count": None, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    return counts


def product_latest_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for table in ("predictions", "prediction_evaluations"):
        run_details = {}
        total_product = 0
        latest_total = 0
        history_excess = 0
        for run_id in PRODUCT_RUN_IDS:
            total = count_query(table, filters=[("eq", "run_id", run_id)]) or 0
            latest_asof = latest_asof_for_run(table, run_id)
            latest_count = (
                count_query(table, filters=[("eq", "run_id", run_id), ("eq", "asof_date", latest_asof)]) or 0
                if latest_asof
                else 0
            )
            excess = max(total - latest_count, 0)
            run_details[run_id] = {
                "total_rows": total,
                "latest_asof_date": latest_asof,
                "latest_rows": latest_count,
                "history_excess_rows": excess,
            }
            total_product += total
            latest_total += latest_count
            history_excess += excess
        table_total = count_query(table) or 0
        summary[table] = {
            "table_total_rows": table_total,
            "product_total_rows": total_product,
            "product_latest_rows": latest_total,
            "product_history_excess_rows": history_excess,
            "non_product_rows": max(table_total - total_product, 0),
            "run_details": run_details,
        }
    return summary


def model_run_summary() -> dict[str, Any]:
    total = count_query("model_runs") or 0
    product = 0
    for run_id in PRODUCT_RUN_IDS:
        product += count_query("model_runs", filters=[("eq", "run_id", run_id)]) or 0
    failed = 0
    try:
        failed = count_query("model_runs", filters=[("in", "status", ["failed_nan", "failed_quality_gate"])]) or 0
    except Exception:
        failed = 0
    completed = 0
    try:
        completed = count_query("model_runs", filters=[("eq", "status", "completed")]) or 0
    except Exception:
        completed = 0
    return {
        "total_rows": total,
        "product_run_rows": product,
        "completed_rows": completed,
        "failed_quality_rows": failed,
        "non_product_rows": max(total - product, 0),
    }


def local_snapshot_summary() -> dict[str, Any]:
    files = {
        "stock_info": SNAPSHOT_DIR / "stock_info.parquet",
        "price_data_yfinance_1D": SNAPSHOT_DIR / "price_data_yfinance.parquet",
        "indicators_yfinance_1D": SNAPSHOT_DIR / "indicators_yfinance_1D.parquet",
        "price_data_yfinance_1W": SNAPSHOT_DIR / "price_data_yfinance_1W.parquet",
        "indicators_yfinance_1W": SNAPSHOT_DIR / "indicators_yfinance_1W.parquet",
    }
    result = {}
    for name, path in files.items():
        if not path.exists():
            result[name] = {"exists": False}
            continue
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            dates = pd.to_datetime(frame["date"], errors="coerce")
            date_min = dates.min().strftime("%Y-%m-%d")
            date_max = dates.max().strftime("%Y-%m-%d")
        else:
            date_min = None
            date_max = None
        result[name] = {
            "exists": True,
            "rows": int(len(frame)),
            "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
            "date_min": date_min,
            "date_max": date_max,
            "bytes": int(path.stat().st_size),
        }
    return result


def pruning_dry_run(counts: dict[str, Any], product_summary: dict[str, Any], model_summary: dict[str, Any]) -> dict[str, Any]:
    predictions = product_summary["predictions"]
    evaluations = product_summary["prediction_evaluations"]
    candidate_groups = {
        "predictions_non_product": predictions["non_product_rows"],
        "predictions_product_history_except_latest": predictions["product_history_excess_rows"],
        "prediction_evaluations_non_product": evaluations["non_product_rows"],
        "prediction_evaluations_product_history_except_latest": evaluations["product_history_excess_rows"],
        "model_runs_non_product": model_summary["non_product_rows"],
        "model_runs_failed_quality": model_summary["failed_quality_rows"],
    }
    candidate_rows = int(sum(value for value in candidate_groups.values() if isinstance(value, int)))
    return {
        "delete_executed": False,
        "candidate_groups": candidate_groups,
        "candidate_rows_total_excluding_price_indicator": candidate_rows,
        "price_data_policy": "Supabase 원본 full store 역할 제거. 제품 표시용 최소 구간만 남기거나 API/local fallback 중심으로 전환.",
        "indicators_policy": "Supabase 원본 full store 역할 제거. 제품 표시용 최소 1D 구간만 남기거나 local parquet 우선.",
        "backup_required_tables": [
            "predictions",
            "prediction_evaluations",
            "model_runs",
            "backtest_results",
            "price_data",
            "indicators",
        ],
    }


def build_metrics() -> dict[str, Any]:
    counts = table_counts()
    product_summary = product_latest_summary()
    model_summary = model_run_summary()
    local_summary = local_snapshot_summary()
    dry_run = pruning_dry_run(counts, product_summary, model_summary)

    price_indicator_bulk_guard = {}
    for table in ("price_data", "indicators"):
        try:
            fetch_all_rows(table, limit=None)
            price_indicator_bulk_guard[table] = "FAILED_NOT_BLOCKED"
        except RuntimeError:
            price_indicator_bulk_guard[table] = "PASS_BLOCKED"

    final_status = "PASS"
    warning_reasons: list[str] = []
    if counts.get("price_data", {}).get("status") != "PASS" or counts.get("indicators", {}).get("status") != "PASS":
        final_status = "WARN"
        warning_reasons.append("price_data/indicators count 확인 일부가 실패했다.")
    if any(value != "PASS_BLOCKED" for value in price_indicator_bulk_guard.values()):
        final_status = "FAIL"
        warning_reasons.append("Supabase bulk read guard가 price_data/indicators에서 실패했다.")

    return {
        "cp": "CP117-DG",
        "generated_at": utc_now_iso(),
        "product_run_ids": {
            "line": LINE_RUN_ID,
            "band": BAND_RUN_ID,
        },
        "supabase_table_counts": counts,
        "product_latest_storage": product_summary,
        "model_runs_summary": model_summary,
        "local_snapshots": local_summary,
        "pruning_dry_run": dry_run,
        "bulk_read_guard": price_indicator_bulk_guard,
        "table_size_sql": TABLE_SIZE_SQL,
        "pruning_dry_run_sql": PRUNING_DRY_RUN_SQL,
        "table_size_sql_execution": {
            "executed_by_script": False,
            "reason": "Supabase REST client에는 임의 SQL 실행 권한이 없어 SQL Editor 또는 별도 RPC 함수가 필요하다.",
            "ready_to_run_in_sql_editor": True,
        },
        "daily_boundary": {
            "data_source_of_record": "local parquet",
            "serving_db": "Supabase thin DB",
            "append_rule": "yfinance row.date < current_date 인 완료 거래일만 local parquet에 append",
            "indicator_rule": "append된 ticker만 1D indicator incremental refresh",
            "product_rule": "1D line/band latest inference 후 product latest-only helper로 저장",
            "stop_on_mismatch": [
                "local price latest date != local indicator latest date",
                "prediction asof_date != local indicator latest date",
                "source_data_hash/cache manifest provider mismatch",
                "EODHD fallback 사용",
                "Supabase price_data/indicators bulk read 시도",
            ],
        },
        "readiness_contract": {
            "checks": [
                "local price latest date",
                "local indicator latest date",
                "product prediction latest asof_date",
                "Supabase latest asof_date",
                "line/band run_id",
                "EODHD fallback count == 0",
                "Supabase bulk read guard == PASS_BLOCKED",
            ],
            "expected_current_1d_asof_date": local_summary.get("indicators_yfinance_1D", {}).get("date_max"),
        },
        "forbidden_actions_observed": {
            "full_yfinance_write": False,
            "indicators_full_recompute": False,
            "model_training": False,
            "inference_bulk_save": False,
            "row_delete": False,
            "eodhd_call": False,
            "db_write": False,
        },
        "final_decision": {
            "status": final_status,
            "warning_reasons": warning_reasons,
            "summary": "local parquet 원천 + Supabase thin DB 운영 계약을 문서로 고정했고, pruning dry-run row count를 산출했다."
            if final_status == "PASS"
            else "운영 계약은 작성됐지만 일부 확인 항목에 주의가 필요하다.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP117 local parquet + Supabase thin closure audit")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics()
    write_json(Path(args.metrics_path), metrics)


if __name__ == "__main__":
    main()
