"""CP254 — v1 서빙 데이터(backend/data/v1/*) → Supabase thin 적재.

대상 7테이블 (컬럼 = 서빙 화이트리스트, 전 컬럼 dump 금지):
  stock_info                      ← market_stock_info.parquet
  price_data                      ← market_prices_1d.parquet (source='eodhd')
  indicators                      ← market_indicators_1d.parquet (+ price volume join)
  predictions_line_1d             ← predictions_line_1d.parquet
  predictions_band_1d             ← predictions_band_1d.parquet
  predictions_band_1w             ← predictions_band_1w.parquet
  product_prediction_history_1d   ← product_prediction_history_1D.parquet

선행: schema.sql + migrations/v1_predictions_tables.sql + migrations/cp254_serving_tables.sql
적용된 DB. 업서트 on_conflict 키 기반 멱등(재실행 안전). prices_1w 는 미적재
— 서빙이 1D 에서 W-FRI 리샘플로 파생 (api_service.aggregate_prices).

Run:
    python backend/scripts/import_v1_serving_to_supabase.py                  # 전체
    python backend/scripts/import_v1_serving_to_supabase.py --tickers AAPL MSFT
    python backend/scripts/import_v1_serving_to_supabase.py --tables stock_info price_data
    python backend/scripts/import_v1_serving_to_supabase.py --since 2025-06-15   # thin retention
    python backend/scripts/import_v1_serving_to_supabase.py --dry-run       # 네트워크 0, 변환 검증
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# chunk 500 + returning=minimal + 지수백오프 재시도는 bootstrap 정본 재사용.
from backend.db.bootstrap import CHUNK_SIZE, _upsert_chunk_with_retry, get_client  # noqa: E402

V1_DIR = ROOT / "backend" / "data" / "v1"
SERVING_SOURCE = "eodhd"
# bootstrap.step_price_data 와 동일 정책 문자열 유지.
PROVIDER_ADJUSTMENT_POLICY = "eodhd_raw_ohlc_adjusted_close_factor_v3_adjusted_ohlc"
# 한 번에 records 로 변환하는 행 수 (782k 전체를 dict 로 들지 않기 위한 슬라이스).
FRAME_SLICE_ROWS = 20_000


def _date_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _plain_str(series: pd.Series) -> pd.Series:
    """category/object → str, 결측은 None 유지."""
    out = series.astype("object")
    return out.where(pd.notna(out), None).map(lambda v: None if v is None else str(v))


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    """numpy 스칼라 → 파이썬 native, NaN/NaT → None, 컬럼 화이트리스트 강제."""
    sub = frame[columns].copy()
    records: list[dict] = []
    for row in sub.to_dict(orient="records"):
        clean: dict = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                clean[key] = None
            elif pd.isna(value):
                clean[key] = None
            elif isinstance(value, np.floating):
                clean[key] = float(value)
            elif isinstance(value, np.integer):
                clean[key] = int(value)
            elif isinstance(value, np.bool_):
                clean[key] = bool(value)
            else:
                clean[key] = value
        records.append(clean)
    return records


def _read_v1(name: str) -> pd.DataFrame:
    path = V1_DIR / name
    if not path.exists():
        print(f"[Error] v1 서빙 parquet 이 없습니다: {path}")
        sys.exit(1)
    return pd.read_parquet(path)


def _filter_tickers(frame: pd.DataFrame, tickers: list[str] | None) -> pd.DataFrame:
    if not tickers or "ticker" not in frame.columns:
        return frame
    wanted = {t.upper() for t in tickers}
    return frame[frame["ticker"].astype(str).str.upper().isin(wanted)].copy()


def _filter_since(frame: pd.DataFrame, since: str | None, date_column: str) -> pd.DataFrame:
    if not since or date_column not in frame.columns:
        return frame
    cutoff = pd.Timestamp(since)
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    return frame[dates >= cutoff].copy()


def upsert_frame(
    client,
    table: str,
    frame: pd.DataFrame,
    columns: list[str],
    on_conflict: str,
    *,
    dry_run: bool = False,
) -> int:
    """frame 을 슬라이스 단위로 records 변환 후 chunk 업서트 (메모리 안전)."""
    total = len(frame)
    if total == 0:
        print(f"  [{table}] 0 rows — skip")
        return 0
    if dry_run:
        sample = _records(frame.head(2), columns)
        print(f"  [{table}] DRY-RUN rows={total:,} on_conflict=({on_conflict})")
        print(f"    sample: {sample[0] if sample else None}")
        return total
    done = 0
    for start in range(0, total, FRAME_SLICE_ROWS):
        records = _records(frame.iloc[start : start + FRAME_SLICE_ROWS], columns)
        for index in range(0, len(records), CHUNK_SIZE):
            chunk = records[index : index + CHUNK_SIZE]
            _upsert_chunk_with_retry(client, table, chunk, on_conflict)
            done += len(chunk)
            print(f"  [{table}] {done:,} / {total:,}", end="\r")
    print(f"  [{table}] {total:,} rows upserted          ")
    return total


# ---------------- 테이블별 frame 빌더 (변환 = 순수 함수, 테스트 대상) ----------------


def build_stock_info_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    if "industry" in frame.columns:
        frame["industry"] = _plain_str(frame["industry"]).fillna("Unknown")
    else:
        frame["industry"] = "Unknown"
    if "market_cap" not in frame.columns:
        frame["market_cap"] = None
    frame["sector"] = _plain_str(frame.get("sector", pd.Series(index=frame.index, dtype=object)))
    return frame, ["ticker", "sector", "industry", "market_cap"], "ticker"


def build_serving_stocks_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """serving_stocks 는 /stocks 응답 정본 — parquet 값을 그대로 보존 (fillna 금지).

    stock_info 의 industry 'Unknown' 채움은 bootstrap(소스 적재) 관례라 유지하지만,
    여기에 적용하면 로컬 모드(parquet null 그대로)와 /stocks 응답 parity 가 깨진다.
    """
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    for column in ("sector", "industry"):
        if column in frame.columns:
            frame[column] = _plain_str(frame[column])
        else:
            frame[column] = None
    if "market_cap" not in frame.columns:
        frame["market_cap"] = None
    return frame, ["ticker", "sector", "industry", "market_cap"], "ticker"


def build_price_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    frame["date"] = _date_str(frame["date"])
    frame = frame.dropna(subset=["ticker", "date", "close"])
    if "adjusted_close" not in frame.columns:
        frame["adjusted_close"] = None
    frame["source"] = SERVING_SOURCE
    frame["provider"] = SERVING_SOURCE
    frame["provider_adjustment_policy"] = PROVIDER_ADJUSTMENT_POLICY
    columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
        "provider",
        "provider_adjustment_policy",
    ]
    return frame, columns, "ticker,date,source"


def build_indicator_frame(
    raw: pd.DataFrame, price_raw: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str], str]:
    """IndicatorPoint 화이트리스트만. volume 은 price parquet join 으로 채운다."""
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    frame["date"] = _date_str(frame["date"])
    frame["timeframe"] = "1D"
    frame["source"] = SERVING_SOURCE
    frame["provider"] = SERVING_SOURCE
    frame["regime_label"] = _plain_str(
        frame.get("regime_label", pd.Series(index=frame.index, dtype=object))
    )
    if price_raw is not None and "volume" in price_raw.columns:
        volume = price_raw.copy()
        volume["ticker"] = _plain_str(volume["ticker"]).str.upper()
        volume["date"] = _date_str(volume["date"])
        volume = volume[["ticker", "date", "volume"]].drop_duplicates(
            ["ticker", "date"], keep="last"
        )
        frame = frame.merge(volume, on=["ticker", "date"], how="left")
    elif "volume" not in frame.columns:
        frame["volume"] = None
    columns = [
        "ticker",
        "timeframe",
        "date",
        "source",
        "provider",
        "rsi",
        "macd_ratio",
        "bb_position",
        "ma_5_ratio",
        "ma_20_ratio",
        "ma_60_ratio",
        "vol_change",
        "volume",
        "regime_label",
    ]
    return frame, columns, "ticker,timeframe,date,source"


def build_line_1d_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    frame["asof_date"] = _date_str(frame["asof_date"])
    frame["model_id"] = _plain_str(frame["model_id"])
    frame["source_cp"] = _plain_str(frame["source_cp"])
    columns = [
        "ticker",
        "asof_date",
        "line_score",
        "safe_line_score",
        "line_rank_by_date",
        "safe_line_rank_by_date",
        "line_top_decile_flag",
        "safe_line_top_decile_flag",
        "actual_h5_return",
        "model_id",
        "source_cp",
    ]
    return frame, columns, "ticker,asof_date"


def build_band_1d_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    frame["asof_date"] = _date_str(frame["asof_date"])
    frame["forecast_date"] = _date_str(frame["forecast_date"])
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="coerce").astype("int64")
    frame["model_id"] = _plain_str(frame["model_id"])
    frame["source_cp"] = _plain_str(frame["source_cp"])
    columns = [
        "ticker",
        "asof_date",
        "forecast_date",
        "horizon_step",
        "band_lower",
        "band_upper",
        "actual_return",
        "actual_return_available",
        "model_id",
        "source_cp",
    ]
    return frame, columns, "ticker,asof_date,horizon_step"


def build_band_1w_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    frame["asof_date"] = _date_str(frame["asof_date"])
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="coerce").astype("int64")
    frame["model_id"] = _plain_str(frame["model_id"])
    frame["source_cp"] = _plain_str(frame["source_cp"])
    columns = [
        "ticker",
        "asof_date",
        "horizon_step",
        "band_lower",
        "band_upper",
        "actual_return",
        "model_id",
        "source_cp",
    ]
    return frame, columns, "ticker,asof_date,horizon_step"


def build_product_history_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    frame = raw.copy()
    frame["ticker"] = _plain_str(frame["ticker"]).str.upper()
    frame["timeframe"] = _plain_str(frame["timeframe"]).str.upper()
    frame["role"] = _plain_str(frame["role"]).str.lower()
    frame["asof_date"] = _date_str(frame["asof_date"])
    frame["display_date"] = _date_str(frame["display_date"])
    frame["display_horizon"] = pd.to_numeric(frame["display_horizon"], errors="coerce").astype(
        "int64"
    )
    frame["run_id"] = _plain_str(frame["run_id"])
    # line/lower/upper 는 parquet float32 — float64 로 승격해 DB DOUBLE 과 동일 표현.
    for column in ("line_value", "lower_value", "upper_value"):
        frame[column] = frame[column].astype("float64")
    columns = [
        "ticker",
        "timeframe",
        "role",
        "asof_date",
        "display_date",
        "display_horizon",
        "run_id",
        "line_value",
        "lower_value",
        "upper_value",
    ]
    return frame, columns, "ticker,timeframe,role,asof_date,display_horizon,run_id"


# 적재 순서 = FK 의존 순서 (price/indicators 가 stock_info.ticker 참조).
TABLE_STEPS: list[tuple[str, str, str | None]] = [
    # (테이블, 소스 parquet, since 필터 기준 컬럼)
    # serving_stocks = /stocks 목록·sector map 전용 큐레이션 100종목.
    # stock_info 는 FK 유니버스(placeholder 포함)라 목록용으로 쓰면 오염된다.
    ("serving_stocks", "market_stock_info.parquet", None),
    ("stock_info", "market_stock_info.parquet", None),
    ("price_data", "market_prices_1d.parquet", "date"),
    ("indicators", "market_indicators_1d.parquet", "date"),
    ("predictions_line_1d", "predictions_line_1d.parquet", "asof_date"),
    ("predictions_band_1d", "predictions_band_1d.parquet", "asof_date"),
    ("predictions_band_1w", "predictions_band_1w.parquet", "asof_date"),
    ("product_prediction_history_1d", "product_prediction_history_1D.parquet", "asof_date"),
]

BUILDERS = {
    "serving_stocks": build_serving_stocks_frame,
    "stock_info": build_stock_info_frame,
    "price_data": build_price_frame,
    "predictions_line_1d": build_line_1d_frame,
    "predictions_band_1d": build_band_1d_frame,
    "predictions_band_1w": build_band_1w_frame,
    "product_prediction_history_1d": build_product_history_frame,
}

# CP255 Phase 4a — product_history(78만행, ~174MB)는 프론트 어느 화면도 호출하지 않는다
# (fetchProductPredictionHistory caller 0). 기본 적재에서 제외해 무료 DB 를 470→258MB 로
# 낮춘다. 필요 시 --include-product-history 로만 포함. 빌더/스키마는 보존(되살릴 때 대비).
HISTORY_TABLE = "product_prediction_history_1d"
DEFAULT_TABLES = [name for name, _, _ in TABLE_STEPS if name != HISTORY_TABLE]


def ensure_stock_info_covers(client, frame: pd.DataFrame, *, dry_run: bool) -> None:
    """price/indicators FK 보호 — stock_info 에 없는 ticker 를 placeholder 로 보충.

    market_stock_info.parquet(100) 보다 price parquet 티커(~500)가 많다.
    FK 위반으로 적재가 깨지지 않게 누락 ticker 를 placeholder 로 먼저 넣는다.
    반드시 ON CONFLICT DO NOTHING(ignore_duplicates) — 일반 upsert 면 이미 적재된
    실 sector/industry 값을 NULL 로 덮어쓴다.
    """
    import time

    from postgrest.types import ReturnMethod

    from backend.db.bootstrap import (
        UPSERT_MAX_ATTEMPTS,
        UPSERT_RETRY_BASE_SECONDS,
        _is_retryable_upsert_error,
    )

    tickers = sorted(set(frame["ticker"].dropna().astype(str).str.upper()))
    records = [
        {"ticker": ticker, "sector": None, "industry": "Unknown", "market_cap": None}
        for ticker in tickers
    ]
    if dry_run:
        print(f"  [stock_info] DRY-RUN placeholder(do-nothing) candidates={len(records):,}")
        return
    for index in range(0, len(records), CHUNK_SIZE):
        chunk = records[index : index + CHUNK_SIZE]
        for attempt in range(1, UPSERT_MAX_ATTEMPTS + 1):
            try:
                client.table("stock_info").upsert(
                    chunk,
                    on_conflict="ticker",
                    ignore_duplicates=True,  # DO NOTHING — 기존 행 절대 미변경
                    returning=ReturnMethod.minimal,
                ).execute()
                break
            except Exception as error:  # noqa: BLE001 — 재시도 판단 후 재전파
                if not _is_retryable_upsert_error(error) or attempt == UPSERT_MAX_ATTEMPTS:
                    raise
                time.sleep(UPSERT_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    print(f"  [stock_info] placeholder(do-nothing) ensured for {len(records):,} tickers")


def run_import(
    *,
    tables: list[str] | None = None,
    tickers: list[str] | None = None,
    since: str | None = None,
    dry_run: bool = False,
    include_product_history: bool = False,
) -> dict[str, int]:
    if tables:
        wanted = set(tables)
    else:
        wanted = set(DEFAULT_TABLES)
        if include_product_history:
            wanted.add(HISTORY_TABLE)
    selected = [step for step in TABLE_STEPS if step[0] in wanted]
    if not selected:
        print(f"[Error] 알 수 없는 --tables 값. 가능: {[s[0] for s in TABLE_STEPS]}")
        sys.exit(1)

    client = None if dry_run else get_client()
    counts: dict[str, int] = {}
    price_raw_for_volume: pd.DataFrame | None = None

    print("=" * 64)
    print(" CP254 v1 serving parquet → Supabase thin 적재")
    print(f"  dir={V1_DIR}")
    print(f"  tables={[s[0] for s in selected]} tickers={tickers or 'ALL'} since={since or '-'}")
    print(f"  dry_run={dry_run}")
    print("=" * 64)

    for table, filename, since_column in selected:
        raw = _read_v1(filename)
        raw = _filter_tickers(raw, tickers)
        if since_column:
            raw = _filter_since(raw, since, since_column)

        if table == "indicators":
            if price_raw_for_volume is None:
                price_raw_for_volume = _filter_tickers(
                    _read_v1("market_prices_1d.parquet"), tickers
                )
            frame, columns, on_conflict = build_indicator_frame(raw, price_raw_for_volume)
        else:
            frame, columns, on_conflict = BUILDERS[table](raw)

        if table == "price_data":
            price_raw_for_volume = raw
            # FK 보호: stock_info parquet(100) 에 없는 price 티커 placeholder 선적재.
            ensure_stock_info_covers(client, frame, dry_run=dry_run)
        if table == "indicators" and "price_data" not in [s[0] for s in selected]:
            ensure_stock_info_covers(client, frame, dry_run=dry_run)

        counts[table] = upsert_frame(client, table, frame, columns, on_conflict, dry_run=dry_run)

    print("\n적재 요약:")
    for table, count in counts.items():
        print(f"  {table:<32} {count:>9,} rows")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP254/255 v1 serving → Supabase thin 적재")
    parser.add_argument(
        "--tables", nargs="*", help="일부 테이블만 적재 (기본 = product_history 제외 7테이블)"
    )
    parser.add_argument("--tickers", nargs="*", help="샘플 적재 종목 (기본 전체)")
    parser.add_argument("--since", help="YYYY-MM-DD — 이 날짜 이후만 적재 (증분/윈도우)")
    parser.add_argument(
        "--include-product-history",
        action="store_true",
        help="product_history(프론트 미사용, 기본 제외)도 적재",
    )
    parser.add_argument("--dry-run", action="store_true", help="네트워크 없이 변환/행수만 출력")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_import(
        tables=args.tables,
        tickers=args.tickers,
        since=args.since,
        dry_run=args.dry_run,
        include_product_history=args.include_product_history,
    )
