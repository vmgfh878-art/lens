"""CP254 — 예측 서빙 슬롯 Supabase REST 조회.

predictions 라우터와 product_prediction_history_svc 의 REST 분기 전담.
모든 fetch_* 는 대응 parquet 슬롯과 **동일 컬럼 구성**의 DataFrame 을 돌려준다 —
응답 조립 코드를 parquet 모드와 공유해 직렬화 회귀를 0 으로 만들기 위함.
(date 계열은 datetime64 로 변환해 라우터의 Timestamp 연산/직렬화 경로 그대로.)

thin 규약(CP254 P4): ticker 스코프 + 컬럼 화이트리스트 + per-ticker 최신
asof 기준 날짜 윈도우를 서버사이드 필터로. 전량 dump 금지.
v1 컷오버는 read 전용 — 이 모듈에 write 경로 없음.
"""

from __future__ import annotations

import pandas as pd
import structlog

from app.core.exceptions import ConfigError, UpstreamUnavailableError
from app.db import get_supabase
from app.repositories.rest_paging import paged_select

logger = structlog.get_logger(__name__)

LINE_1D_COLUMNS = [
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
BAND_1D_COLUMNS = [
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
BAND_1W_COLUMNS = [
    "ticker",
    "asof_date",
    "horizon_step",
    "band_lower",
    "band_upper",
    "actual_return",
    "model_id",
    "source_cp",
]
HISTORY_COLUMNS = [
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

_SLOT_SPECS: dict[str, dict] = {
    "line_1d": {"table": "predictions_line_1d", "columns": LINE_1D_COLUMNS},
    "band_1d": {"table": "predictions_band_1d", "columns": BAND_1D_COLUMNS},
    "band_1w": {"table": "predictions_band_1w", "columns": BAND_1W_COLUMNS},
}
HISTORY_TABLE = "product_prediction_history_1d"


def _wrap(message: str, exc: Exception) -> UpstreamUnavailableError:
    logger.warning("prediction_repo REST 실패: %s", message, exc_info=exc)
    return UpstreamUnavailableError(message)


def _latest_asof(client, table: str, filters: list[tuple[str, str, object]]) -> str | None:
    """필터 스코프 안에서 최신 asof_date 1행 조회 (윈도우 cutoff 기준점)."""
    query = client.table(table).select("asof_date")
    for operator, column, value in filters:
        query = getattr(query, operator)(column, value)
    rows = query.order("asof_date", desc=True).limit(1).execute().data or []
    if not rows:
        return None
    return str(rows[0]["asof_date"])


def _to_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _fetch_slot_frame(
    slot: str,
    ticker: str,
    *,
    days: int,
    horizon: int | None = None,
) -> pd.DataFrame | None:
    """ticker 가 테이블에 없으면 None (라우터 404), horizon 필터로 비면 빈 frame (200).

    parquet 모드의 'ticker 부재 → 404 / 필터 결과 0행 → 200 빈 응답' 구분과 동일.
    """
    spec = _SLOT_SPECS[slot]
    table, columns = spec["table"], spec["columns"]
    selected = ", ".join(columns)
    try:
        client = get_supabase()
        latest = _latest_asof(client, table, [("eq", "ticker", ticker.upper())])
        if latest is None:
            return None
        cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=days)).date().isoformat()

        def build_query():
            query = (
                client.table(table)
                .select(selected)
                .eq("ticker", ticker.upper())
                .gte("asof_date", cutoff)
            )
            if horizon is not None:
                query = query.eq("horizon_step", horizon)
            query = query.order("asof_date")
            if "horizon_step" in columns:
                query = query.order("horizon_step")
            return query

        rows = paged_select(build_query)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 — 503 으로 수렴, 원인은 로그
        raise _wrap(f"{table} 조회 실패 (ticker={ticker})", exc) from exc

    frame = pd.DataFrame(rows, columns=columns)
    return _to_datetime(frame, ["asof_date", "forecast_date"])


def fetch_line_1d(ticker: str, *, days: int) -> pd.DataFrame | None:
    """1D Line — parquet 슬롯과 동일 컬럼, asof_date 는 datetime64. 부재 ticker 는 None."""
    return _fetch_slot_frame("line_1d", ticker, days=days)


def fetch_band_1d(ticker: str, *, days: int, horizon: int | None = None) -> pd.DataFrame | None:
    return _fetch_slot_frame("band_1d", ticker, days=days, horizon=horizon)


def fetch_band_1w(ticker: str, *, days: int, horizon: int | None = None) -> pd.DataFrame | None:
    return _fetch_slot_frame("band_1w", ticker, days=days, horizon=horizon)


def list_tickers(*, search: str | None = None, limit: int = 500) -> list[str]:
    """가용 ticker 목록 — REST 모드는 stock_info 기준 (서빙 유니버스 동일, 100~500행).

    parquet 모드의 "line_1d 캐시 기준"과의 동치성은 리허설 parity 로 검증한다.
    """
    try:
        client = get_supabase()
        query = client.table("stock_info").select("ticker")
        if search:
            query = query.ilike("ticker", f"{search.strip().upper()}%")
        rows = query.order("ticker").limit(limit).execute().data or []
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap("ticker 목록 조회 실패", exc) from exc
    seen: list[str] = []
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker and ticker not in seen:
            seen.append(ticker)
    return seen[:limit]


def slot_counts() -> dict[str, int | None]:
    """REST 모드 /predictions/health 용 — 슬롯 테이블 행수 (payload ~0)."""
    counts: dict[str, int | None] = {}
    try:
        client = get_supabase()
        for slot, spec in _SLOT_SPECS.items():
            response = (
                client.table(spec["table"]).select("ticker", count="exact").limit(1).execute()
            )
            counts[slot] = response.count
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap("슬롯 행수 조회 실패", exc) from exc
    return counts


def fetch_product_history(
    ticker: str,
    *,
    timeframe: str,
    roles: set[str],
    run_id: str | None = None,
    limit: int | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """제품 rolling history — 화이트리스트 10컬럼, 윈도우/limit 을 SQL 로 (전량 로드 금지).

    parquet 경로 _apply_window 와 동일 의미:
      lookback = (스코프 내 최신 asof) - lookback_days 이후만,
      limit = role 별 최근 N행 (asof desc N → asc 재정렬).
    """
    base_filters: list[tuple[str, str, object]] = [
        ("eq", "ticker", ticker.upper()),
        ("eq", "timeframe", timeframe.upper()),
        ("in_", "role", sorted(roles)),
    ]
    if run_id:
        base_filters.append(("eq", "run_id", run_id))
    selected = ", ".join(HISTORY_COLUMNS)
    try:
        client = get_supabase()
        latest = _latest_asof(client, HISTORY_TABLE, base_filters)
        if latest is None:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        cutoff: str | None = None
        if lookback_days is not None:
            cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=lookback_days)).date().isoformat()

        def apply_filters(query, role: str | None = None):
            for operator, column, value in base_filters:
                if role is not None and column == "role":
                    continue
                query = getattr(query, operator)(column, value)
            if role is not None:
                query = query.eq("role", role)
            if cutoff is not None:
                query = query.gte("asof_date", cutoff)
            return query

        if limit is None:

            def build_query():
                return (
                    apply_filters(client.table(HISTORY_TABLE).select(selected))
                    .order("asof_date")
                    .order("role")
                    .order("run_id")
                    .order("display_horizon")
                )

            rows = paged_select(build_query)
        else:
            # role 별 최근 limit 행 — parquet 경로의 groupby(role).tail(limit) 동치.
            rows = []
            for role in sorted(roles):
                role_rows = (
                    apply_filters(client.table(HISTORY_TABLE).select(selected), role=role)
                    .order("asof_date", desc=True)
                    .order("display_horizon", desc=True)
                    .limit(limit)
                    .execute()
                    .data
                    or []
                )
                rows.extend(reversed(role_rows))
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap(f"product history 조회 실패 (ticker={ticker})", exc) from exc

    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    frame = _to_datetime(frame, ["asof_date"])
    if not frame.empty:
        frame = frame.sort_values(["asof_date", "role", "run_id"], kind="stable").reset_index(
            drop=True
        )
    return frame


# ---------------- 전략 scan 용 cross-sectional 입력 (CP254 P3) ----------------
# thin per-ticker 가 불가능한 유일 경로. 컬럼 화이트리스트 + (band) horizon 서버
# 필터로 좁히고, 호출 빈도는 strategy_scan 의 lru_cache 가 억제한다.
# egress 실측은 P5 — 한도 위협이면 materialized 테이블이 v2 후속.

SCAN_LINE_COLUMNS = ["ticker", "asof_date", "line_score"]
SCAN_BAND_COLUMNS = ["ticker", "asof_date", "horizon_step", "band_lower", "band_upper"]


def fetch_line_frame_for_scan() -> pd.DataFrame:
    try:
        client = get_supabase()

        def build_query():
            return (
                client.table("predictions_line_1d")
                .select(", ".join(SCAN_LINE_COLUMNS))
                .order("ticker")
                .order("asof_date")
            )

        rows = paged_select(build_query)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap("scan line frame 조회 실패", exc) from exc
    return pd.DataFrame(rows, columns=SCAN_LINE_COLUMNS)


def fetch_band_frame_for_scan(*, horizon: int = 5) -> pd.DataFrame:
    try:
        client = get_supabase()

        def build_query():
            return (
                client.table("predictions_band_1d")
                .select(", ".join(SCAN_BAND_COLUMNS))
                .eq("horizon_step", horizon)
                .order("ticker")
                .order("asof_date")
            )

        rows = paged_select(build_query)
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap("scan band frame 조회 실패", exc) from exc
    return pd.DataFrame(rows, columns=SCAN_BAND_COLUMNS)
