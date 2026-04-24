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

from backend.app.services.feature_svc import (  # noqa: E402
    REQUIRED_FEATURE_COLUMNS,
    _BASE_FEATURE_COLUMNS,
    _CONTEXT_COLUMNS,
    _REGIME_FEATURE_COLUMNS,
    _apply_fundamental_features,
    _apply_regime_columns,
    _compute_features_for_single_ticker,
    _resample_context_frame,
    normalize_timeframe,
    resample_price_frame,
)


REPORT_PATH = ROOT_DIR / "docs" / "cp2.5_report.md"


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


def query_df(conn, sql: str, params: tuple | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def summarize_numeric(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0}
    return {
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def summarize_dates(series: pd.Series) -> dict[str, str | None]:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return {"p10": None, "p90": None}
    ints = values.astype("int64")
    return {
        "p10": pd.to_datetime(pd.Series(ints).quantile(0.10)).date().isoformat(),
        "p90": pd.to_datetime(pd.Series(ints).quantile(0.90)).date().isoformat(),
    }


def audit_price_data(conn) -> tuple[pd.DataFrame, dict]:
    frame = query_df(
        conn,
        """
        SELECT ticker,
               MIN(date) AS min_date,
               MAX(date) AS max_date,
               COUNT(*)::BIGINT AS row_count
        FROM public.price_data
        GROUP BY ticker
        ORDER BY ticker
        """,
    )
    frame["min_date"] = pd.to_datetime(frame["min_date"])
    frame["max_date"] = pd.to_datetime(frame["max_date"])
    summary = {
        "ticker_count": int(frame["ticker"].nunique()),
        "row_count": summarize_numeric(frame["row_count"]),
        "min_date": summarize_dates(frame["min_date"]),
        "before_2015": int((frame["min_date"] < pd.Timestamp("2015-01-01")).sum()),
        "after_2020": int((frame["min_date"] >= pd.Timestamp("2020-01-01")).sum()),
    }
    return frame, summary


def audit_indicators(conn, timeframe: str) -> tuple[pd.DataFrame, dict]:
    frame = query_df(
        conn,
        """
        SELECT ticker,
               MIN(date) AS min_date,
               MAX(date) AS max_date,
               COUNT(*)::BIGINT AS row_count,
               SUM(CASE WHEN has_fundamentals THEN 1 ELSE 0 END)::BIGINT AS with_fund_count
        FROM public.indicators
        WHERE timeframe = %s
        GROUP BY ticker
        ORDER BY ticker
        """,
        (timeframe,),
    )
    frame["min_date"] = pd.to_datetime(frame["min_date"])
    frame["max_date"] = pd.to_datetime(frame["max_date"])
    summary = {
        "ticker_count": int(frame["ticker"].nunique()),
        "row_count": summarize_numeric(frame["row_count"]),
        "min_date": summarize_dates(frame["min_date"]),
        "has_fund_ratio": float(frame["with_fund_count"].sum() / frame["row_count"].sum()) if frame["row_count"].sum() else 0.0,
    }
    return frame, summary


def audit_fundamentals(conn) -> tuple[pd.DataFrame, dict]:
    frame = query_df(
        conn,
        """
        SELECT ticker,
               MIN(date) AS min_date,
               MAX(date) AS max_date,
               COUNT(*)::BIGINT AS quarter_count
        FROM public.company_fundamentals
        GROUP BY ticker
        ORDER BY ticker
        """,
    )
    frame["min_date"] = pd.to_datetime(frame["min_date"])
    frame["max_date"] = pd.to_datetime(frame["max_date"])
    summary = {
        "ticker_count": int(frame["ticker"].nunique()),
        "quarter_count": summarize_numeric(frame["quarter_count"]),
        "eight_plus": int((frame["quarter_count"] >= 8).sum()),
    }
    return frame, summary


def audit_sync_state(conn) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for job_name in ("sync_prices:1D", "compute_indicators:1D"):
        frame = query_df(
            conn,
            """
            SELECT MIN(last_cursor_date) AS earliest_last_cursor,
                   MAX(last_cursor_date) AS latest_last_cursor
            FROM public.sync_state
            WHERE job_name = %s
            """,
            (job_name,),
        )
        earliest = frame.iloc[0]["earliest_last_cursor"]
        latest = frame.iloc[0]["latest_last_cursor"]
        result[job_name] = {
            "earliest": pd.to_datetime(earliest).date().isoformat() if pd.notna(earliest) else None,
            "latest": pd.to_datetime(latest).date().isoformat() if pd.notna(latest) else None,
        }
    return result


def load_aapl_frames(conn) -> dict[str, pd.DataFrame]:
    price = query_df(
        conn,
        """
        SELECT ticker, date, open, high, low, close, adjusted_close, volume
        FROM public.price_data
        WHERE ticker = 'AAPL'
        ORDER BY date
        """,
    )
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
    fundamentals = query_df(
        conn,
        """
        SELECT ticker, date, filing_date, revenue, net_income, total_liabilities, equity, eps
        FROM public.company_fundamentals
        WHERE ticker = 'AAPL'
        ORDER BY filing_date
        """,
    )
    for frame_name, frame in {
        "price": price,
        "macro": macro,
        "breadth": breadth,
        "fundamentals": fundamentals,
    }.items():
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        if frame_name == "fundamentals" and "filing_date" in frame.columns:
            frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    return {
        "price": price,
        "macro": macro,
        "breadth": breadth,
        "fundamentals": fundamentals,
    }


def aapl_direct_diagnosis(conn, frames: dict[str, pd.DataFrame]) -> dict:
    price = frames["price"]
    fundamentals = frames["fundamentals"]
    indicators_1d = query_df(
        conn,
        """
        SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*)::BIGINT AS row_count
        FROM public.indicators
        WHERE ticker = 'AAPL' AND timeframe = '1D'
        """,
    ).iloc[0].to_dict()
    indicators_1w = query_df(
        conn,
        """
        SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*)::BIGINT AS row_count
        FROM public.indicators
        WHERE ticker = 'AAPL' AND timeframe = '1W'
        """,
    ).iloc[0].to_dict()
    return {
        "price_data": {
            "min_date": price["date"].min().date().isoformat(),
            "max_date": price["date"].max().date().isoformat(),
            "row_count": int(len(price)),
        },
        "indicators_1D": {
            "min_date": pd.to_datetime(indicators_1d["min_date"]).date().isoformat(),
            "max_date": pd.to_datetime(indicators_1d["max_date"]).date().isoformat(),
            "row_count": int(indicators_1d["row_count"]),
        },
        "indicators_1W": {
            "min_date": pd.to_datetime(indicators_1w["min_date"]).date().isoformat(),
            "max_date": pd.to_datetime(indicators_1w["max_date"]).date().isoformat(),
            "row_count": int(indicators_1w["row_count"]),
        },
        "fundamentals": {
            "min_date": fundamentals["date"].min().date().isoformat(),
            "max_date": fundamentals["date"].max().date().isoformat(),
            "quarter_count": int(len(fundamentals)),
        },
    }


def aapl_pipeline_counts(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    timeframe = normalize_timeframe("1D")
    price_frame = resample_price_frame(frames["price"], timeframe)
    n0 = int(len(frames["price"]))
    n1 = int(len(price_frame))

    feature_frame = _compute_features_for_single_ticker(price_frame)
    macro_frame = _resample_context_frame(
        frames["macro"],
        timeframe,
        ("us10y", "yield_spread", "vix_close", "credit_spread_hy"),
    )
    breadth_frame = _resample_context_frame(
        frames["breadth"],
        timeframe,
        ("nh_nl_index", "ma200_pct"),
    )
    if not macro_frame.empty:
        feature_frame = feature_frame.merge(macro_frame, on="date", how="left")
    if not breadth_frame.empty:
        feature_frame = feature_frame.merge(breadth_frame, on="date", how="left")
    for column in _CONTEXT_COLUMNS:
        if column not in feature_frame.columns:
            feature_frame[column] = pd.NA
    fill_columns = [column for column in _CONTEXT_COLUMNS if column in feature_frame.columns]
    feature_frame[fill_columns] = feature_frame[fill_columns].ffill()

    n2 = int(len(feature_frame.dropna(subset=_BASE_FEATURE_COLUMNS)))
    regime_frame = _apply_regime_columns(feature_frame.copy())
    n3 = int(len(regime_frame.dropna(subset=[*_BASE_FEATURE_COLUMNS, *_REGIME_FEATURE_COLUMNS])))
    fundamental_frame = _apply_fundamental_features(regime_frame.copy(), frames["fundamentals"])
    n4 = int(len(fundamental_frame))
    n5 = int(len(fundamental_frame.dropna(subset=REQUIRED_FEATURE_COLUMNS)))
    return {"N0": n0, "N1": n1, "N2": n2, "N3": n3, "N4": n4, "N5": n5}


def decide_hypotheses(price_summary: dict, indicator_summaries: dict[str, dict], pipeline: dict[str, int]) -> dict[str, dict[str, str]]:
    price_p50 = price_summary["row_count"]["p50"]
    indicator_1d_p50 = indicator_summaries["1D"]["row_count"]["p50"]
    h1 = price_p50 < 2000
    h2 = price_p50 >= 2000 and indicator_1d_p50 < 2000
    h3 = pipeline["N1"] >= 2000 and pipeline["N5"] < (pipeline["N1"] * 0.5)

    if h3:
        root = "H3"
    elif h2:
        root = "H2"
    elif h1:
        root = "H1"
    else:
        root = "복합"

    return {
        "H1": {
            "status": "HOLDS" if h1 else "REJECTED",
            "reason": f"price_data row_count 중앙값={price_p50:.1f}",
        },
        "H2": {
            "status": "HOLDS" if h2 else "REJECTED",
            "reason": f"price_data p50={price_p50:.1f}, indicators 1D p50={indicator_1d_p50:.1f}",
        },
        "H3": {
            "status": "HOLDS" if h3 else "REJECTED",
            "reason": f"AAPL 파이프라인 N1={pipeline['N1']}, N5={pipeline['N5']}",
        },
        "root_cause": root,
    }


def build_report(
    *,
    price_summary: dict,
    indicator_summaries: dict[str, dict],
    fundamentals_summary: dict,
    aapl_summary: dict,
    pipeline: dict[str, int],
    sync_state: dict[str, dict[str, str | None]],
    hypotheses: dict[str, dict[str, str]],
) -> str:
    section_lines = [
        "[CP2.5] 완료",
        "",
        "## 1. price_data 분포",
        f"- 전체 ticker 수: {price_summary['ticker_count']}",
        (
            "- row_count 분위수: "
            f"p10={price_summary['row_count']['p10']:.1f}, "
            f"p25={price_summary['row_count']['p25']:.1f}, "
            f"p50={price_summary['row_count']['p50']:.1f}, "
            f"p75={price_summary['row_count']['p75']:.1f}, "
            f"p90={price_summary['row_count']['p90']:.1f}, "
            f"max={price_summary['row_count']['max']:.1f}"
        ),
        f"- min_date 분위수: 가장 이른 p10={price_summary['min_date']['p10']}, 가장 늦은 p90={price_summary['min_date']['p90']}",
        f"- 2015-01-01 이전부터 있는 ticker 수: {price_summary['before_2015']}",
        f"- 2020-01-01 이후 시작하는 ticker 수: {price_summary['after_2020']}",
        "",
        "## 2. indicators 분포 (1D / 1W / 1M)",
    ]
    for timeframe in ("1D", "1W", "1M"):
        summary = indicator_summaries[timeframe]
        section_lines.extend(
            [
                f"### {timeframe}",
                f"- 전체 ticker 수: {summary['ticker_count']}",
                (
                    "- row_count p10/p50/p90: "
                    f"{summary['row_count']['p10']:.1f} / {summary['row_count']['p50']:.1f} / {summary['row_count']['p90']:.1f}"
                ),
                f"- has_fundamentals=1 비율(time-weighted): {summary['has_fund_ratio']:.4f}",
                f"- min_date p10/p90: {summary['min_date']['p10']} / {summary['min_date']['p90']}",
                "",
            ]
        )

    section_lines.extend(
        [
            "## 3. company_fundamentals",
            f"- 전체 ticker 수: {fundamentals_summary['ticker_count']}",
            (
                "- 분기 수 분위수 p10/p50/p90: "
                f"{fundamentals_summary['quarter_count']['p10']:.1f} / "
                f"{fundamentals_summary['quarter_count']['p50']:.1f} / "
                f"{fundamentals_summary['quarter_count']['p90']:.1f}"
            ),
            f"- 8분기 이상 보유 ticker 수: {fundamentals_summary['eight_plus']}",
            f"- 확인된 기대치 477, 실제 {fundamentals_summary['eight_plus']}",
            "",
            "## 4. AAPL 직접 진단",
            (
                "- price_data: "
                f"min_date={aapl_summary['price_data']['min_date']}, "
                f"max_date={aapl_summary['price_data']['max_date']}, "
                f"row_count={aapl_summary['price_data']['row_count']}"
            ),
            (
                "- indicators 1D: "
                f"min_date={aapl_summary['indicators_1D']['min_date']}, "
                f"max_date={aapl_summary['indicators_1D']['max_date']}, "
                f"row_count={aapl_summary['indicators_1D']['row_count']}"
            ),
            (
                "- indicators 1W: "
                f"min_date={aapl_summary['indicators_1W']['min_date']}, "
                f"max_date={aapl_summary['indicators_1W']['max_date']}, "
                f"row_count={aapl_summary['indicators_1W']['row_count']}"
            ),
            (
                "- fundamentals: "
                f"min_date={aapl_summary['fundamentals']['min_date']}, "
                f"max_date={aapl_summary['fundamentals']['max_date']}, "
                f"quarter_count={aapl_summary['fundamentals']['quarter_count']}"
            ),
            "",
            "- 원시 파이프라인 row 감소 단계별",
            f"  N0 (price raw): {pipeline['N0']}",
            f"  N1 (resample): {pipeline['N1']}",
            f"  N2 (base dropna): {pipeline['N2']}",
            f"  N3 (regime merge): {pipeline['N3']}",
            f"  N4 (fundamentals merge + 8Q gate): {pipeline['N4']}",
            f"  N5 (final dropna): {pipeline['N5']}",
            "",
            "## 5. sync_state 힌트",
            f"- sync_prices:1D earliest last_cursor: {sync_state['sync_prices:1D']['earliest']}",
            f"- sync_prices:1D latest last_cursor: {sync_state['sync_prices:1D']['latest']}",
            f"- compute_indicators:1D earliest last_cursor: {sync_state['compute_indicators:1D']['earliest']}",
            f"- compute_indicators:1D latest last_cursor: {sync_state['compute_indicators:1D']['latest']}",
            "",
            "## 6. 가설 판정",
            f"- 가설 H1 (price_data 자체가 짧다): [{hypotheses['H1']['status']}]",
            f"  근거: {hypotheses['H1']['reason']}",
            f"- 가설 H2 (price OK, indicators 짧음): [{hypotheses['H2']['status']}]",
            f"  근거: {hypotheses['H2']['reason']}",
            f"- 가설 H3 (원천은 OK, 피처에서 줄어듦): [{hypotheses['H3']['status']}]",
            f"  근거: {hypotheses['H3']['reason']}",
            "",
            f"**원인 결정**: {hypotheses['root_cause']}",
            "",
            "## 7. CP2.6 제안 (다음 CP 참고)",
        ]
    )

    next_step = {
        "H1": "- 전체 유니버스 price_data 장기 backfill 또는 price source backfill 범위 점검",
        "H2": "- compute_indicators full-range recompute 또는 source_history 설정 점검",
        "H3": "- feature_svc 단계별 drop 규칙 재조정",
        "복합": "- price_data 장기성, indicators recompute 범위, feature drop 규칙을 함께 점검",
    }[hypotheses["root_cause"]]
    section_lines.append(next_step)
    return "\n".join(section_lines) + "\n"


def main() -> None:
    conn = connect()
    try:
        price_frame, price_summary = audit_price_data(conn)
        indicator_frames: dict[str, pd.DataFrame] = {}
        indicator_summaries: dict[str, dict] = {}
        for timeframe in ("1D", "1W", "1M"):
            frame, summary = audit_indicators(conn, timeframe)
            indicator_frames[timeframe] = frame
            indicator_summaries[timeframe] = summary
        fundamentals_frame, fundamentals_summary = audit_fundamentals(conn)
        sync_state = audit_sync_state(conn)
        aapl_frames = load_aapl_frames(conn)
        aapl_summary = aapl_direct_diagnosis(conn, aapl_frames)
        pipeline = aapl_pipeline_counts(aapl_frames)
        hypotheses = decide_hypotheses(price_summary, indicator_summaries, pipeline)

        report = build_report(
            price_summary=price_summary,
            indicator_summaries=indicator_summaries,
            fundamentals_summary=fundamentals_summary,
            aapl_summary=aapl_summary,
            pipeline=pipeline,
            sync_state=sync_state,
            hypotheses=hypotheses,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")

        stdout_summary = {
            "price_ticker_count": price_summary["ticker_count"],
            "price_p50": price_summary["row_count"]["p50"],
            "indicator_1d_p50": indicator_summaries["1D"]["row_count"]["p50"],
            "fundamentals_eight_plus": fundamentals_summary["eight_plus"],
            "aapl_price_rows": aapl_summary["price_data"]["row_count"],
            "aapl_indicator_1d_rows": aapl_summary["indicators_1D"]["row_count"],
            "root_cause": hypotheses["root_cause"],
            "report_path": str(REPORT_PATH),
        }
        print(json.dumps(stdout_summary, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
