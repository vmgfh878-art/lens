"""원본 Postgres DB에서 Lens DB로 일일 동기화를 수행한다."""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import paramiko
import psycopg2
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.app.services.feature_svc import build_features  # noqa: E402
from backend.db.bootstrap import chunked_upsert, get_client  # noqa: E402

TABLE_ORDER = [
    "stock_info",
    "price_data",
    "macroeconomic_indicators",
    "market_breadth",
    "sector_returns",
    "company_fundamentals",
]

TARGET_TABLES = {
    "stock_info",
    "price_data",
    "macroeconomic_indicators",
    "market_breadth",
    "sector_returns",
    "company_fundamentals",
    "indicators",
}


@dataclass
class SourceDbConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    sslmode: str | None
    ssh_host: str | None
    ssh_port: int
    ssh_user: str | None
    ssh_private_key: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본 DB -> Lens 동기화")
    parser.add_argument(
        "--tables",
        nargs="*",
        default=TABLE_ORDER,
        help="동기화할 raw 테이블 목록",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="특정 티커만 동기화",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="watermark를 무시하고 전체 범위를 다시 적재",
    )
    parser.add_argument(
        "--price-lookback-days",
        type=int,
        default=45,
        help="price_data 증분 동기화 버퍼 일수",
    )
    parser.add_argument(
        "--macro-lookback-days",
        type=int,
        default=30,
        help="macro, breadth, sector_returns 증분 동기화 버퍼 일수",
    )
    parser.add_argument(
        "--fundamentals-lookback-days",
        type=int,
        default=180,
        help="company_fundamentals 증분 동기화 버퍼 일수",
    )
    parser.add_argument(
        "--skip-indicators",
        action="store_true",
        help="raw 테이블만 동기화하고 indicators 재계산은 건너뜀",
    )
    parser.add_argument(
        "--indicator-lookback-days",
        type=int,
        default=2400,
        help="indicators 재계산 시 가져올 과거 일수. 0이면 전체",
    )
    return parser.parse_args()


def load_source_config() -> SourceDbConfig:
    required = [
        "SOURCE_DB_HOST",
        "SOURCE_DB_PORT",
        "SOURCE_DB_USER",
        "SOURCE_DB_PASSWORD",
        "SOURCE_DB_NAME",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"[Error] 원본 DB 환경변수가 없습니다: {', '.join(missing)}")

    return SourceDbConfig(
        host=os.environ["SOURCE_DB_HOST"],
        port=int(os.environ["SOURCE_DB_PORT"]),
        user=os.environ["SOURCE_DB_USER"],
        password=os.environ["SOURCE_DB_PASSWORD"],
        dbname=os.environ["SOURCE_DB_NAME"],
        sslmode=os.environ.get("SOURCE_DB_SSLMODE") or None,
        ssh_host=os.environ.get("SOURCE_SSH_HOST"),
        ssh_port=int(os.environ.get("SOURCE_SSH_PORT") or "22"),
        ssh_user=os.environ.get("SOURCE_SSH_USER"),
        ssh_private_key=os.environ.get("SOURCE_SSH_PRIVATE_KEY"),
    )


def resolve_ssh_private_key(raw_value: str) -> paramiko.PKey:
    # 환경 변수에 키 본문을 직접 넣는 경우와 파일 경로를 넣는 경우를 모두 지원한다.
    key_source = raw_value
    if os.path.exists(raw_value):
        key_source = Path(raw_value).read_text(encoding="utf-8")

    for key_class in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return key_class.from_private_key(io.StringIO(key_source))
        except Exception:
            continue

    raise SystemExit("[Error] SOURCE_SSH_PRIVATE_KEY 형식을 해석하지 못했습니다.")


@contextmanager
def open_source_connection(config: SourceDbConfig) -> Iterator[psycopg2.extensions.connection]:
    tunnel: SSHTunnelForwarder | None = None
    connect_host = config.host
    connect_port = config.port

    if config.ssh_host and config.ssh_user and config.ssh_private_key:
        print("[Source] SSH 터널을 엽니다.")
        private_key = resolve_ssh_private_key(config.ssh_private_key)
        tunnel = SSHTunnelForwarder(
            (config.ssh_host, config.ssh_port),
            ssh_username=config.ssh_user,
            ssh_pkey=private_key,
            remote_bind_address=(config.host, config.port),
            local_bind_address=("127.0.0.1", 0),
        )
        tunnel.start()
        connect_host = "127.0.0.1"
        connect_port = tunnel.local_bind_port
        print(f"[Source] 로컬 포트 {connect_port}로 터널 연결 완료")
    else:
        print(f"[Source] 직접 연결 사용: {connect_host}:{connect_port}")

    conn = psycopg2.connect(
        host=connect_host,
        port=connect_port,
        user=config.user,
        password=config.password,
        dbname=config.dbname,
        sslmode=config.sslmode,
        connect_timeout=15,
    )

    try:
        yield conn
    finally:
        conn.close()
        if tunnel is not None:
            tunnel.stop()
            print("[Source] SSH 터널 종료")


def read_sql_frame(conn: psycopg2.extensions.connection, query: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def list_source_columns(conn: psycopg2.extensions.connection, table_name: str) -> set[str]:
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %(table_name)s
    """
    frame = read_sql_frame(conn, query, {"table_name": table_name})
    return set(frame["column_name"].tolist())


def source_table_exists(conn: psycopg2.extensions.connection, table_name: str) -> bool:
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %(table_name)s
        ) AS exists
    """
    frame = read_sql_frame(conn, query, {"table_name": table_name})
    return bool(frame.iloc[0]["exists"])


def get_target_max_date(table_name: str, column_name: str = "date") -> date | None:
    client = get_client()
    result = client.table(table_name).select(column_name).order(column_name, desc=True).limit(1).execute()
    rows = result.data or []
    if not rows or not rows[0].get(column_name):
        return None
    return pd.to_datetime(rows[0][column_name]).date()


def build_date_filter(
    target_max_date: date | None,
    lookback_days: int,
    full_refresh: bool,
) -> tuple[str, dict]:
    if full_refresh or target_max_date is None:
        return "", {}

    start_date = target_max_date - timedelta(days=lookback_days)
    return "date >= %(start_date)s", {"start_date": start_date}


def merge_conditions(base_conditions: list[str], extra_condition: str | None) -> list[str]:
    conditions = list(base_conditions)
    if extra_condition:
        conditions.append(extra_condition)
    return conditions


def add_ticker_condition(conditions: list[str], params: dict, tickers: list[str] | None) -> None:
    if tickers:
        params["tickers"] = [ticker.upper() for ticker in tickers]
        conditions.append("ticker = ANY(%(tickers)s)")


def build_where_clause(conditions: list[str]) -> str:
    if not conditions:
        return ""
    return " WHERE " + " AND ".join(conditions)


def sql_expr(columns: set[str], column_name: str, cast_type: str = "numeric", alias: str | None = None) -> str:
    alias_name = alias or column_name
    if column_name in columns:
        if alias_name == column_name:
            return column_name
        return f"{column_name} AS {alias_name}"
    return f"NULL::{cast_type} AS {alias_name}"


def fetch_stock_info(
    conn: psycopg2.extensions.connection,
    tickers: list[str] | None,
) -> pd.DataFrame:
    columns = list_source_columns(conn, "stock_info")
    params: dict = {}
    conditions: list[str] = []
    add_ticker_condition(conditions, params, tickers)
    query = f"""
        SELECT
            ticker,
            {sql_expr(columns, 'sector', 'text')},
            {sql_expr(columns, 'industry', 'text')},
            {sql_expr(columns, 'market_cap', 'numeric')}
        FROM public.stock_info
        {build_where_clause(conditions)}
        ORDER BY ticker
    """
    return read_sql_frame(conn, query, params)


def fetch_price_data(
    conn: psycopg2.extensions.connection,
    tickers: list[str] | None,
    full_refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    columns = list_source_columns(conn, "price_data")
    target_max_date = get_target_max_date("price_data")
    date_condition, params = build_date_filter(target_max_date, lookback_days, full_refresh)
    conditions = merge_conditions([], date_condition)
    add_ticker_condition(conditions, params, tickers)
    adjusted_expr = "adjusted_close" if "adjusted_close" in columns else "close AS adjusted_close"
    query = f"""
        SELECT
            ticker,
            date,
            open,
            high,
            low,
            close,
            {adjusted_expr},
            {sql_expr(columns, 'volume', 'bigint')},
            {sql_expr(columns, 'amount', 'numeric')},
            {sql_expr(columns, 'per', 'numeric')},
            {sql_expr(columns, 'pbr', 'numeric')}
        FROM public.price_data
        {build_where_clause(conditions)}
        ORDER BY date, ticker
    """
    return read_sql_frame(conn, query, params)


def fetch_macro(
    conn: psycopg2.extensions.connection,
    full_refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    columns = list_source_columns(conn, "macroeconomic_indicators")
    target_max_date = get_target_max_date("macroeconomic_indicators")
    date_condition, params = build_date_filter(target_max_date, lookback_days, full_refresh)
    query = f"""
        SELECT
            date,
            {sql_expr(columns, 'cpi')},
            {sql_expr(columns, 'gdp')},
            {sql_expr(columns, 'interest_rate')},
            {sql_expr(columns, 'unemployment_rate')},
            {sql_expr(columns, 'us10y')},
            {sql_expr(columns, 'us2y')},
            {sql_expr(columns, 'yield_spread')},
            {sql_expr(columns, 'vix_close')},
            {sql_expr(columns, 'dxy_close')},
            {sql_expr(columns, 'wti_price')},
            {sql_expr(columns, 'gold_price')},
            {sql_expr(columns, 'credit_spread_hy')},
            {sql_expr(columns, 'core_cpi')},
            {sql_expr(columns, 'pce')},
            {sql_expr(columns, 'core_pce')}
        FROM public.macroeconomic_indicators
        {build_where_clause(merge_conditions([], date_condition))}
        ORDER BY date
    """
    return read_sql_frame(conn, query, params)


def fetch_company_fundamentals(
    conn: psycopg2.extensions.connection,
    tickers: list[str] | None,
    full_refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    columns = list_source_columns(conn, "company_fundamentals")
    target_max_date = get_target_max_date("company_fundamentals")
    date_condition, params = build_date_filter(target_max_date, lookback_days, full_refresh)
    conditions = merge_conditions([], date_condition)
    add_ticker_condition(conditions, params, tickers)
    query = f"""
        SELECT
            ticker,
            date,
            {sql_expr(columns, 'revenue')},
            {sql_expr(columns, 'net_income')},
            {sql_expr(columns, 'total_assets')},
            {sql_expr(columns, 'total_liabilities')},
            {sql_expr(columns, 'equity')},
            {sql_expr(columns, 'shares_issued')},
            {sql_expr(columns, 'eps')},
            {sql_expr(columns, 'roe')},
            {sql_expr(columns, 'debt_ratio')},
            {sql_expr(columns, 'interest_coverage')},
            {sql_expr(columns, 'operating_cash_flow')}
        FROM public.company_fundamentals
        {build_where_clause(conditions)}
        ORDER BY date, ticker
    """
    return read_sql_frame(conn, query, params)


def fetch_market_breadth(
    conn: psycopg2.extensions.connection,
    full_refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    if not source_table_exists(conn, "market_breadth"):
        return build_market_breadth_from_price_source(conn, full_refresh)

    columns = list_source_columns(conn, "market_breadth")
    target_max_date = get_target_max_date("market_breadth")
    date_condition, params = build_date_filter(target_max_date, lookback_days, full_refresh)

    if "nh_nl_index" in columns:
        nh_expr = "nh_nl_index"
    elif {"new_highs", "new_lows"}.issubset(columns):
        nh_expr = "(COALESCE(new_highs, 0) - COALESCE(new_lows, 0))::numeric AS nh_nl_index"
    else:
        nh_expr = "NULL::numeric AS nh_nl_index"

    if "ma200_pct" in columns:
        ma_expr = "ma200_pct"
    elif "above_ma200_pct" in columns:
        ma_expr = "above_ma200_pct AS ma200_pct"
    else:
        ma_expr = "NULL::numeric AS ma200_pct"

    if nh_expr.startswith("NULL::") and ma_expr.startswith("NULL::"):
        return build_market_breadth_from_price_source(conn, full_refresh)

    query = f"""
        SELECT
            date,
            {nh_expr},
            {ma_expr}
        FROM public.market_breadth
        {build_where_clause(merge_conditions([], date_condition))}
        ORDER BY date
    """
    return read_sql_frame(conn, query, params)


def fetch_sector_returns(
    conn: psycopg2.extensions.connection,
    full_refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    if not source_table_exists(conn, "sector_returns"):
        return pd.DataFrame(columns=["date", "sector", "etf_ticker", "return", "close"])

    columns = list_source_columns(conn, "sector_returns")
    target_max_date = get_target_max_date("sector_returns")
    date_condition, params = build_date_filter(target_max_date, lookback_days, full_refresh)
    query = f"""
        SELECT
            date,
            {sql_expr(columns, 'sector', 'text')},
            {sql_expr(columns, 'etf_ticker', 'text')},
            {sql_expr(columns, 'return')},
            {sql_expr(columns, 'close')}
        FROM public.sector_returns
        {build_where_clause(merge_conditions([], date_condition))}
        ORDER BY date, sector
    """
    return read_sql_frame(conn, query, params)


def build_market_breadth_from_price_source(
    conn: psycopg2.extensions.connection,
    full_refresh: bool,
) -> pd.DataFrame:
    lookback_days = 3650 if full_refresh else 400
    print(f"[Sync] market_breadth 원본 계산값이 없어 price_data로 다시 계산합니다. lookback={lookback_days}일")
    price_frame = read_sql_frame(
        conn,
        """
            SELECT date, ticker, close
            FROM public.price_data
            WHERE date >= CURRENT_DATE - (%(lookback_days)s * INTERVAL '1 day')
            ORDER BY ticker, date
        """,
        {"lookback_days": lookback_days},
    )
    if price_frame.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"])

    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame = price_frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    price_frame["high_52w"] = price_frame.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=252, min_periods=200).max()
    )
    price_frame["low_52w"] = price_frame.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=252, min_periods=200).min()
    )
    price_frame["ma_200"] = price_frame.groupby("ticker")["close"].transform(
        lambda series: series.rolling(window=200, min_periods=150).mean()
    )
    price_frame = price_frame.dropna(subset=["high_52w", "low_52w", "ma_200"])
    if price_frame.empty:
        return pd.DataFrame(columns=["date", "nh_nl_index", "ma200_pct"])

    price_frame["is_nh"] = price_frame["close"] >= price_frame["high_52w"]
    price_frame["is_nl"] = price_frame["close"] <= price_frame["low_52w"]
    price_frame["is_above_ma200"] = price_frame["close"] > price_frame["ma_200"]

    stats = (
        price_frame.groupby("date")
        .agg(
            total_count=("ticker", "count"),
            nh_count=("is_nh", "sum"),
            nl_count=("is_nl", "sum"),
            above_ma200_count=("is_above_ma200", "sum"),
        )
        .reset_index()
    )
    stats["nh_nl_index"] = stats["nh_count"] - stats["nl_count"]
    stats["ma200_pct"] = (stats["above_ma200_count"] / stats["total_count"]) * 100.0
    stats = stats[stats["total_count"] > 50][["date", "nh_nl_index", "ma200_pct"]]
    return stats


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    if "ticker" in normalized.columns:
        normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    return normalized.where(pd.notnull(normalized), None)


def upsert_stock_info(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("[Sync] stock_info 신규 데이터 없음")
        return
    records = normalize_frame(frame).to_dict(orient="records")
    chunked_upsert(get_client(), "stock_info", records, on_conflict="ticker")
    print(f"[Sync] stock_info {len(records):,}건 적재")


def upsert_price_data(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("[Sync] price_data 신규 데이터 없음")
        return
    records = normalize_frame(frame).to_dict(orient="records")
    chunked_upsert(get_client(), "price_data", records, on_conflict="ticker,date")
    print(f"[Sync] price_data {len(records):,}건 적재")


def upsert_macro(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("[Sync] macroeconomic_indicators 신규 데이터 없음")
        return
    records = normalize_frame(frame).to_dict(orient="records")
    chunked_upsert(get_client(), "macroeconomic_indicators", records, on_conflict="date")
    print(f"[Sync] macroeconomic_indicators {len(records):,}건 적재")


def upsert_company_fundamentals(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("[Sync] company_fundamentals 신규 데이터 없음")
        return
    records = normalize_frame(frame).to_dict(orient="records")
    chunked_upsert(get_client(), "company_fundamentals", records, on_conflict="ticker,date")
    print(f"[Sync] company_fundamentals {len(records):,}건 적재")


def upsert_market_breadth(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("[Sync] market_breadth 신규 데이터 없음")
        return
    records = normalize_frame(frame).to_dict(orient="records")
    chunked_upsert(get_client(), "market_breadth", records, on_conflict="date")
    print(f"[Sync] market_breadth {len(records):,}건 적재")


def upsert_sector_returns(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("[Sync] sector_returns 신규 데이터 없음")
        return
    records = normalize_frame(frame).to_dict(orient="records")
    chunked_upsert(get_client(), "sector_returns", records, on_conflict="date,sector")
    print(f"[Sync] sector_returns {len(records):,}건 적재")


def fetch_indicator_source_frames(
    conn: psycopg2.extensions.connection,
    tickers: list[str] | None,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conditions: list[str] = []
    params: dict = {}

    if lookback_days > 0:
        latest_indicator_date = get_target_max_date("indicators")
        if latest_indicator_date is None:
            latest_indicator_date = get_target_max_date("price_data")
        if latest_indicator_date is not None:
            params["start_date"] = latest_indicator_date - timedelta(days=lookback_days)
            conditions.append("date >= %(start_date)s")

    add_ticker_condition(conditions, params, tickers)
    where_clause = build_where_clause(conditions)

    price_frame = read_sql_frame(
        conn,
        f"""
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                {sql_expr(list_source_columns(conn, 'price_data'), 'adjusted_close')},
                {sql_expr(list_source_columns(conn, 'price_data'), 'volume', 'bigint')},
                {sql_expr(list_source_columns(conn, 'price_data'), 'amount')},
                {sql_expr(list_source_columns(conn, 'price_data'), 'per')},
                {sql_expr(list_source_columns(conn, 'price_data'), 'pbr')}
            FROM public.price_data
            {where_clause}
            ORDER BY date, ticker
        """,
        params,
    )

    macro_columns = list_source_columns(conn, "macroeconomic_indicators")
    macro_frame = read_sql_frame(
        conn,
        f"""
            SELECT
                date,
                {sql_expr(macro_columns, 'us10y')},
                {sql_expr(macro_columns, 'yield_spread')},
                {sql_expr(macro_columns, 'vix_close')},
                {sql_expr(macro_columns, 'credit_spread_hy')}
            FROM public.macroeconomic_indicators
            ORDER BY date
        """,
    )

    breadth_frame = fetch_market_breadth(conn, full_refresh=True, lookback_days=0)
    return price_frame, macro_frame, breadth_frame


def upsert_indicators(
    conn: psycopg2.extensions.connection,
    tickers: list[str] | None,
    lookback_days: int,
) -> None:
    print("[Sync] indicators 재계산 시작")
    price_frame, macro_frame, breadth_frame = fetch_indicator_source_frames(conn, tickers, lookback_days)

    if price_frame.empty:
        print("[Sync] indicators 재계산용 price_data가 없습니다.")
        return

    price_frame["date"] = pd.to_datetime(price_frame["date"])
    if not macro_frame.empty:
        macro_frame["date"] = pd.to_datetime(macro_frame["date"])
    if not breadth_frame.empty:
        breadth_frame["date"] = pd.to_datetime(breadth_frame["date"])

    client = get_client()

    for timeframe in ("1D", "1W", "1M"):
        features = build_features(
            price_df=price_frame,
            macro_df=macro_frame,
            breadth_df=breadth_frame,
            timeframe=timeframe,
        )
        if features.empty:
            print(f"[Sync] indicators {timeframe} 신규 데이터 없음")
            continue

        frame = features.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        records = frame.where(pd.notnull(frame), None).to_dict(orient="records")
        chunked_upsert(client, "indicators", records, on_conflict="ticker,timeframe,date")
        print(f"[Sync] indicators {timeframe} {len(records):,}건 적재")


def main() -> None:
    args = parse_args()
    invalid_tables = sorted(set(args.tables) - TARGET_TABLES)
    if invalid_tables:
        raise SystemExit(f"[Error] 지원하지 않는 테이블입니다: {', '.join(invalid_tables)}")

    source_config = load_source_config()

    with open_source_connection(source_config) as conn:
        print("=" * 60)
        print(" Lens 일일 동기화 시작")
        print("=" * 60)

        if "stock_info" in args.tables:
            upsert_stock_info(fetch_stock_info(conn, args.tickers))
        if "price_data" in args.tables:
            upsert_price_data(fetch_price_data(conn, args.tickers, args.full_refresh, args.price_lookback_days))
        if "macroeconomic_indicators" in args.tables:
            upsert_macro(fetch_macro(conn, args.full_refresh, args.macro_lookback_days))
        if "market_breadth" in args.tables:
            upsert_market_breadth(fetch_market_breadth(conn, args.full_refresh, args.macro_lookback_days))
        if "sector_returns" in args.tables:
            upsert_sector_returns(fetch_sector_returns(conn, args.full_refresh, args.macro_lookback_days))
        if "company_fundamentals" in args.tables:
            upsert_company_fundamentals(
                fetch_company_fundamentals(
                    conn,
                    args.tickers,
                    args.full_refresh,
                    args.fundamentals_lookback_days,
                )
            )

        if not args.skip_indicators:
            upsert_indicators(conn, args.tickers, args.indicator_lookback_days)

        print("=" * 60)
        print(" Lens 일일 동기화 완료")
        print("=" * 60)


if __name__ == "__main__":
    main()
