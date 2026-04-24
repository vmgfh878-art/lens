from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")


def connect():
    import psycopg2

    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"DB 연결 정보가 부족합니다: {', '.join(missing)}")

    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "5432")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=os.environ["DB_NAME"],
        sslmode=os.environ.get("DB_SSLMODE", "require"),
    )


def query_df(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def first_non_null_dates(frame: pd.DataFrame, columns: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    for column in columns:
        if column not in working.columns:
            result[column] = None
            continue
        dates = working.loc[working[column].notna(), "date"]
        result[column] = dates.min().date().isoformat() if not dates.empty else None
    return result


def main() -> None:
    conn = connect()
    try:
        macro = query_df(
            conn,
            """
            SELECT date, us10y, yield_spread, vix_close, credit_spread_hy
            FROM public.macroeconomic_indicators
            ORDER BY date
            """,
        )
        breadth = query_df(
            conn,
            """
            SELECT date, nh_nl_index, ma200_pct
            FROM public.market_breadth
            ORDER BY date
            """,
        )
        aapl_price = query_df(
            conn,
            """
            SELECT date, per, pbr
            FROM public.price_data
            WHERE ticker = 'AAPL'
            ORDER BY date
            """,
        )

        macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
        breadth["date"] = pd.to_datetime(breadth["date"], errors="coerce")
        aapl_price["date"] = pd.to_datetime(aapl_price["date"], errors="coerce")

        result = {
            "macroeconomic_indicators": {
                "min_date": macro["date"].min().date().isoformat() if not macro.empty else None,
                "max_date": macro["date"].max().date().isoformat() if not macro.empty else None,
                "row_count": int(len(macro)),
                "null_ratio": {
                    column: float(macro[column].isna().mean()) for column in ("us10y", "yield_spread", "vix_close", "credit_spread_hy")
                },
                "first_non_null": first_non_null_dates(macro, ["us10y", "yield_spread", "vix_close", "credit_spread_hy"]),
            },
            "market_breadth": {
                "min_date": breadth["date"].min().date().isoformat() if not breadth.empty else None,
                "max_date": breadth["date"].max().date().isoformat() if not breadth.empty else None,
                "row_count": int(len(breadth)),
                "null_ratio": {
                    column: float(breadth[column].isna().mean()) for column in ("nh_nl_index", "ma200_pct")
                },
                "first_non_null": first_non_null_dates(breadth, ["nh_nl_index", "ma200_pct"]),
            },
            "price_data_aapl": {
                "min_date": aapl_price["date"].min().date().isoformat() if not aapl_price.empty else None,
                "max_date": aapl_price["date"].max().date().isoformat() if not aapl_price.empty else None,
                "row_count": int(len(aapl_price)),
                "null_ratio": {
                    "per": float(aapl_price["per"].isna().mean()),
                    "pbr": float(aapl_price["pbr"].isna().mean()),
                },
                "first_non_null": first_non_null_dates(aapl_price, ["per", "pbr"]),
            },
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
