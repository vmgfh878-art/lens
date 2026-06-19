"""CP255 Phase 4a — Supabase 1년 rolling retention.

매일 증분 적재(import_v1_serving_to_supabase.py --since)와 짝을 이뤄, 윈도우 밖
오래된 행을 삭제해 무료 DB(500MB)를 정상상태로 유지한다.

설계:
- 테이블별 윈도우(일) = 프론트 화면 최대 요청 + 버퍼. 불변식: 윈도우 ≥ 화면 요청.
  · price/indicators/line/band_1d: 화면 370일 + 전략 365일 → 450일
  · band_1w: 화면 730일 요청 → 800일
  · stock_info/serving_stocks: 날짜 없음 — retention 대상 아님(전량 유지)
- 기준일은 "DB의 그 테이블 최신 날짜"(today 아님). 주말/공휴일/적재 지연에 안전.
- 삭제 기준 = 윈도우 시작일보다 **이전** 행만. REST(supabase-py)의 .lt() 사용.
- DELETE 는 PostgREST 가 WHERE 없는 전량 삭제를 막으므로 항상 날짜 조건을 건다.
- --dry-run: 삭제 대상 건수만 집계(실삭제 0).

윈도우 불변식이 깨지면(설정 윈도우 < 화면 요청) 즉시 예외 — 데모 차트에 빈
구간이 생기는 사고를 코드에서 차단.

Run:
    python backend/scripts/supabase_retention.py --dry-run
    python backend/scripts/supabase_retention.py            # 실삭제
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db.bootstrap import get_client  # noqa: E402

# (테이블, 날짜컬럼, 보관 윈도우 일수, 화면 최대 요청 일수[불변식 검증용])
RETENTION_SPECS: list[tuple[str, str, int, int]] = [
    ("price_data", "date", 450, 370),
    ("indicators", "date", 450, 370),
    ("predictions_line_1d", "asof_date", 450, 370),
    ("predictions_band_1d", "asof_date", 450, 370),
    ("predictions_band_1w", "asof_date", 800, 730),
    # product_prediction_history_1d 는 기본 미적재(프론트 미사용) — retention 대상 아님.
    # 적재돼 있으면 포함하려면 --include-product-history.
]
HISTORY_SPEC = ("product_prediction_history_1d", "asof_date", 450, 370)


def _validate_invariants(specs: list[tuple[str, str, int, int]]) -> None:
    for table, _col, window_days, screen_days in specs:
        if window_days < screen_days:
            raise ValueError(
                f"[{table}] retention 윈도우({window_days}d) < 화면 요청({screen_days}d) — "
                "데모 차트 빈 구간 위험. 윈도우를 늘려라."
            )


def _latest_date(client, table: str, date_column: str) -> str | None:
    rows = (
        client.table(table)
        .select(date_column)
        .order(date_column, desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows or not rows[0].get(date_column):
        return None
    return str(rows[0][date_column])[:10]


def _count_before(client, table: str, date_column: str, cutoff: str) -> int:
    resp = (
        client.table(table)
        .select(date_column, count="exact")
        .lt(date_column, cutoff)
        .limit(1)
        .execute()
    )
    return int(resp.count or 0)


def run_retention(
    *,
    dry_run: bool = False,
    include_product_history: bool = False,
) -> dict[str, dict]:
    specs = list(RETENTION_SPECS)
    if include_product_history:
        specs.append(HISTORY_SPEC)
    _validate_invariants(specs)

    client = get_client()
    summary: dict[str, dict] = {}

    print("=" * 64)
    print(" CP255 Phase 4a — Supabase 1년 rolling retention")
    print(f"  dry_run={dry_run}")
    print("=" * 64)

    for table, date_column, window_days, _screen in specs:
        latest = _latest_date(client, table, date_column)
        if latest is None:
            summary[table] = {"status": "empty"}
            print(f"  [{table}] 비어있음 — skip")
            continue
        cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=window_days)).date().isoformat()
        doomed = _count_before(client, table, date_column, cutoff)
        if dry_run:
            summary[table] = {
                "latest": latest,
                "cutoff": cutoff,
                "window_days": window_days,
                "would_delete": doomed,
            }
            print(f"  [{table}] latest={latest} cutoff={cutoff} would_delete={doomed:,}")
            continue
        if doomed > 0:
            # WHERE = date_column < cutoff (전량 삭제 방지: 항상 날짜 조건)
            client.table(table).delete().lt(date_column, cutoff).execute()
        summary[table] = {
            "latest": latest,
            "cutoff": cutoff,
            "window_days": window_days,
            "deleted": doomed,
        }
        print(f"  [{table}] latest={latest} cutoff={cutoff} deleted={doomed:,}")

    print("\n완료.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP255 Supabase 1년 rolling retention")
    parser.add_argument("--dry-run", action="store_true", help="삭제 대상 건수만 집계")
    parser.add_argument(
        "--include-product-history",
        action="store_true",
        help="product_history 도 retention (기본 미적재라 보통 불필요)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_retention(dry_run=args.dry_run, include_product_history=args.include_product_history)
