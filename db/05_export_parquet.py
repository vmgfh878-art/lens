"""Supabase DB 테이블을 parquet로 내려받는다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.db import get_supabase  # noqa: E402

TABLE_CONFIG = {
    "stock_info": ["ticker"],
    "price_data": ["ticker", "date", "id"],
    "macroeconomic_indicators": ["date"],
    "market_breadth": ["date"],
    "sector_returns": ["date", "sector", "id"],
    "company_fundamentals": ["ticker", "date", "id"],
    "indicators": ["ticker", "timeframe", "date", "id"],
    "predictions": ["ticker", "model_name", "timeframe", "asof_date", "id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supabase -> parquet export")
    parser.add_argument(
        "--tables",
        nargs="*",
        default=[
            "stock_info",
            "price_data",
            "macroeconomic_indicators",
            "market_breadth",
            "sector_returns",
            "company_fundamentals",
        ],
        help="내려받을 테이블 목록",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "parquet"),
        help="parquet 저장 폴더",
    )
    parser.add_argument("--page-size", type=int, default=1000, help="REST 페이지 크기")
    return parser.parse_args()


def fetch_page(table_name: str, start: int, end: int):
    query = get_supabase().table(table_name).select("*")
    for column in TABLE_CONFIG[table_name]:
        query = query.order(column)
    return query.range(start, end).execute().data or []


def export_table(table_name: str, output_dir: Path, page_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{table_name}.parquet"

    print(f"\n[Export] {table_name} -> {output_path}")
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    page_index = 0

    try:
        while True:
            start = page_index * page_size
            end = start + page_size - 1
            rows = fetch_page(table_name, start, end)
            if not rows:
                break

            frame = pd.DataFrame(rows)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

            total_rows += len(frame)
            page_index += 1
            print(f"  {total_rows:,} rows", end="\r")

        if writer is None:
            pd.DataFrame().to_parquet(output_path, index=False)
        print(f"  {total_rows:,} rows saved")
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for table_name in args.tables:
        if table_name not in TABLE_CONFIG:
            raise SystemExit(f"[Error] 지원하지 않는 테이블입니다: {table_name}")
        export_table(table_name, output_dir, args.page_size)

    print("\n모든 parquet export가 완료되었습니다.")


if __name__ == "__main__":
    main()
