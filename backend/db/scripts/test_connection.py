"""Supabase REST와 Postgres 직접 연결을 함께 점검한다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.app.db import get_supabase  # noqa: E402

TABLE_SPECS = [
    ("stock_info", "ticker"),
    ("price_data", "ticker"),
    ("macroeconomic_indicators", "date"),
    ("market_breadth", "date"),
    ("indicators", "ticker"),
    ("predictions", "ticker"),
]


def test_supabase_rest() -> None:
    print("=" * 60)
    print(" Supabase REST 연결 테스트")
    print("=" * 60)

    try:
        client = get_supabase()
    except Exception as exc:
        print(f"[Error] Supabase REST 연결 실패: {exc}")
        raise SystemExit(1) from exc

    print("[OK] Supabase 클라이언트 생성 성공")

    for table_name, sample_column in TABLE_SPECS:
        response = client.table(table_name).select(sample_column, count="exact").limit(1).execute()
        row_count = response.count if response.count is not None else 0
        print(f"[OK] {table_name:<28} count={row_count}")


def test_postgres_direct() -> None:
    print("\n" + "=" * 60)
    print(" Postgres 직접 연결 테스트")
    print("=" * 60)

    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print(f"[Skip] Postgres 연결 정보가 부족합니다: {', '.join(missing)}")
        return

    try:
        import psycopg2
    except Exception:
        print("[Skip] psycopg2가 설치되지 않았습니다. `pip install -r db/requirements-crawler.txt` 후 다시 시도하세요.")
        return

    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=os.environ["DB_NAME"],
        sslmode=os.environ.get("DB_SSLMODE", "require"),
        connect_timeout=5,
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            db_name, user_name = cursor.fetchone()
            print(f"[OK] DB={db_name} USER={user_name}")
    finally:
        conn.close()


def main() -> None:
    test_supabase_rest()
    test_postgres_direct()


if __name__ == "__main__":
    main()
