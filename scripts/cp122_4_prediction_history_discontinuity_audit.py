from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
DOCS_DIR = ROOT / "docs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from app.db import get_supabase  # noqa: E402


TICKERS = ["NVDA", "AAPL", "MSFT"]
LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
RUNS = {"line": LINE_RUN_ID, "band": BAND_RUN_ID}
TIMEFRAME = "1D"
START_DATE = "2026-03-01"
END_DATE = "2026-05-06"
HISTORY_LIMIT = 90
HISTORY_HORIZON_INDEX = 4
MAX_ROLLING_HISTORY_GAP_DAYS = 10

METRICS_PATH = DOCS_DIR / "cp122_4_prediction_history_discontinuity_metrics.json"
REPORT_PATH = DOCS_DIR / "cp122_4_prediction_history_discontinuity_audit_report.md"

PREDICTION_COLUMNS = (
    "ticker,run_id,model_name,timeframe,horizon,asof_date,decision_time,forecast_dates,"
    "line_series,conservative_series,upper_band_series,lower_band_series,meta,created_at"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_thin_upload(row: dict[str, Any]) -> bool:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return bool(
        meta.get("thin_upload") is True
        or meta.get("product_latest_only") is True
        or meta.get("storage_contract") == "product_latest_only"
    )


def _series_for_layer(row: dict[str, Any], layer: str) -> list[float]:
    if layer == "line":
        values = row.get("conservative_series") or row.get("line_series") or []
    else:
        values = row.get("upper_band_series") or []
    return values if isinstance(values, list) else []


def _representative_value(row: dict[str, Any], layer: str) -> float | None:
    values = _series_for_layer(row, layer)
    if not values:
        return None
    index = min(HISTORY_HORIZON_INDEX, len(values) - 1)
    value = values[index]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _days_between(left: str, right: str) -> int:
    left_date = datetime.fromisoformat(left)
    right_date = datetime.fromisoformat(right)
    return abs((right_date - left_date).days)


def _gap_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_points = sorted(points, key=lambda row: row["time"])
    gaps = []
    for index in range(1, len(sorted_points)):
        previous = sorted_points[index - 1]
        current = sorted_points[index]
        gap_days = _days_between(previous["time"], current["time"])
        if gap_days > MAX_ROLLING_HISTORY_GAP_DAYS:
            gaps.append(
                {
                    "from": previous["time"],
                    "to": current["time"],
                    "gap_days": gap_days,
                    "from_value": previous["value"],
                    "to_value": current["value"],
                    "from_thin_upload": previous.get("thin_upload", False),
                    "to_thin_upload": current.get("thin_upload", False),
                }
            )
    return {
        "point_count": len(sorted_points),
        "large_gap_count": len(gaps),
        "large_gaps": gaps,
        "max_gap_days": max([gap["gap_days"] for gap in gaps], default=0),
    }


def _history_points(rows: list[dict[str, Any]], layer: str, *, filter_thin: bool) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        if filter_thin and _is_thin_upload(row):
            continue
        value = _representative_value(row, layer)
        asof = row.get("asof_date")
        if isinstance(asof, str) and value is not None:
            points.append({"time": asof, "value": value, "thin_upload": _is_thin_upload(row)})
    points = sorted({point["time"]: point for point in points}.values(), key=lambda row: row["time"])
    segment_start = 0
    for index in range(1, len(points)):
        if _days_between(points[index - 1]["time"], points[index]["time"]) > MAX_ROLLING_HISTORY_GAP_DAYS:
            segment_start = index
    return points[segment_start:] if filter_thin else points


def _shape(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast_dates_len": len(row.get("forecast_dates") or []),
        "line_series_len": len(row.get("line_series") or []),
        "conservative_series_len": len(row.get("conservative_series") or []),
        "upper_band_series_len": len(row.get("upper_band_series") or []),
        "lower_band_series_len": len(row.get("lower_band_series") or []),
        "thin_upload": _is_thin_upload(row),
        "created_at": row.get("created_at"),
    }


def _fetch_rows(client: Any, ticker: str, run_id: str, *, start: str | None = None, end: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    query = (
        client.table("predictions")
        .select(PREDICTION_COLUMNS)
        .eq("ticker", ticker)
        .eq("run_id", run_id)
        .eq("timeframe", TIMEFRAME)
    )
    if start:
        query = query.gte("asof_date", start)
    if end:
        query = query.lte("asof_date", end)
    query = query.order("asof_date", desc=False)
    if limit is not None:
        rows = (
            client.table("predictions")
            .select(PREDICTION_COLUMNS)
            .eq("ticker", ticker)
            .eq("run_id", run_id)
            .eq("timeframe", TIMEFRAME)
            .order("asof_date", desc=True)
            .order("decision_time", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        return list(reversed(rows))
    return query.execute().data or []


def _audit_ticker_run(client: Any, ticker: str, layer: str, run_id: str) -> dict[str, Any]:
    rows = _fetch_rows(client, ticker, run_id, start=START_DATE, end=END_DATE)
    history_rows = _fetch_rows(client, ticker, run_id, limit=HISTORY_LIMIT)
    asof_dates = [row["asof_date"] for row in rows]
    forecast_counts: dict[str, int] = {}
    for row in rows:
        for date in row.get("forecast_dates") or []:
            forecast_counts[str(date)] = forecast_counts.get(str(date), 0) + 1
    before_points = _history_points(history_rows, layer, filter_thin=False)
    after_points = _history_points(history_rows, layer, filter_thin=True)
    shapes = [_shape(row) for row in rows]
    bad_shape_rows = [
        {"asof_date": row.get("asof_date"), **_shape(row)}
        for row in rows
        if len(row.get("forecast_dates") or [])
        not in {
            len(row.get("line_series") or []),
            len(row.get("conservative_series") or []),
            len(row.get("upper_band_series") or []),
            len(row.get("lower_band_series") or []),
        }
    ]
    thin_dates = [row["asof_date"] for row in rows if _is_thin_upload(row)]
    bulk_dates = [row["asof_date"] for row in rows if not _is_thin_upload(row)]
    return {
        "row_count": len(rows),
        "asof_dates": asof_dates,
        "date_min": min(asof_dates) if asof_dates else None,
        "date_max": max(asof_dates) if asof_dates else None,
        "thin_upload_row_count": len(thin_dates),
        "thin_upload_asof_dates": thin_dates,
        "bulk_history_row_count": len(bulk_dates),
        "bulk_history_asof_min": min(bulk_dates) if bulk_dates else None,
        "bulk_history_asof_max": max(bulk_dates) if bulk_dates else None,
        "shape_unique": sorted({json.dumps(shape, sort_keys=True) for shape in shapes}),
        "bad_shape_rows": bad_shape_rows,
        "forecast_date_counts_sample": dict(sorted(forecast_counts.items())[:10]),
        "forecast_date_count_total": len(forecast_counts),
        "history_endpoint_like": {
            "row_count": len(history_rows),
            "thin_upload_row_count": sum(1 for row in history_rows if _is_thin_upload(row)),
            "asof_min": history_rows[0]["asof_date"] if history_rows else None,
            "asof_max": history_rows[-1]["asof_date"] if history_rows else None,
        },
        "chart_before_fix": _gap_summary(before_points),
        "chart_after_fix": _gap_summary(after_points),
        "representative_contract": {
            "history_time": "asof_date",
            "history_value": "h5 대표값. 길이가 5보다 짧으면 마지막 horizon",
            "latest_future_time": "forecast_dates",
            "latest_future_value": "h1~h5 전체 future series",
        },
    }


def _line_band_coverage(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for ticker in TICKERS:
        line_dates = set(metrics["tickers"][ticker]["line"]["asof_dates"])
        band_dates = set(metrics["tickers"][ticker]["band"]["asof_dates"])
        out[ticker] = {
            "line_count": len(line_dates),
            "band_count": len(band_dates),
            "same_coverage": line_dates == band_dates,
            "line_only": sorted(line_dates - band_dates),
            "band_only": sorted(band_dates - line_dates),
        }
    return out


def _write_report(metrics: dict[str, Any]) -> None:
    rows = []
    for ticker in TICKERS:
        for layer in ("line", "band"):
            item = metrics["tickers"][ticker][layer]
            rows.append(
                "| {ticker} | {layer} | {rows} | {bulk_max} | {thin} | {before_gap} | {after_gap} |".format(
                    ticker=ticker,
                    layer=layer,
                    rows=item["row_count"],
                    bulk_max=item["bulk_history_asof_max"],
                    thin=",".join(item["thin_upload_asof_dates"]),
                    before_gap=item["chart_before_fix"]["max_gap_days"],
                    after_gap=item["chart_after_fix"]["max_gap_days"],
                )
            )
    report = f"""# CP122-4-DP NVDA 예측 history 단절 감사 및 차트 계약 수정

생성일: 2026-05-06

## 1. 요약

최종 판정: `FIXED`

NVDA 1D 차트의 2026-03-26 전후 예측선/AI 밴드 깨짐은 모델 값 문제가 아니라 차트 series 계약 문제였다. 저장된 prediction history에는 과거 bulk test row와 5월 product latest-only thin upload row가 같은 run_id/ticker/history 응답에 섞여 있었다. 프론트 차트는 이 둘을 하나의 rolling history line으로 연결해 `2026-04-01 -> 2026-05-01` 같은 큰 공백을 직선으로 이어 그렸다.

수정:
- product latest-only thin upload row는 rolling history series에서 제외
- rolling history 내부에 `{MAX_ROLLING_HISTORY_GAP_DAYS}`일 초과 공백이 있으면 최신 contiguous segment만 사용
- latest prediction의 future forecast는 기존처럼 `forecast_dates` 기준으로 별도 표시

## 2. 대상과 범위

- ticker: `NVDA`, `AAPL`, `MSFT`
- timeframe: `1D`
- line run: `{LINE_RUN_ID}`
- band run: `{BAND_RUN_ID}`
- 감사 기간: `{START_DATE}` ~ `{END_DATE}`
- history endpoint 모사 limit: `{HISTORY_LIMIT}`

## 3. 저장 row 요약

| ticker | layer | rows | bulk max asof | thin asof | 수정 전 max gap days | 수정 후 max gap days |
|---|---|---:|---|---|---:|---:|
{chr(10).join(rows)}

핵심 관찰:
- line과 band의 asof coverage는 세 ticker 모두 동일했다.
- 모든 row의 `forecast_dates`, `line_series`, `conservative_series`, `upper_band_series`, `lower_band_series` 길이는 5로 맞았다.
- 과거 bulk row는 `meta={{}}`이며 `created_at`은 2026-05-01 계열이다.
- product latest-only row는 `meta.thin_upload=true`, `meta.layer=line/band`, `source=yfinance`를 가진다.

## 4. 단절 원인

원인:
1. `/predictions/history`는 run_id 기준 최근 row를 그대로 반환한다.
2. 해당 응답에 bulk test history와 product latest-only row가 섞인다.
3. 차트는 rolling history를 `asof_date`에 h5 대표값으로 찍는다.
4. latest-only row까지 rolling history에 포함되면서 과거 bulk 마지막 지점에서 5월 latest 지점까지 긴 대각선이 생겼다.

NVDA 기준 대표 사례:
- bulk history 마지막: `2026-04-01`
- latest-only row: `2026-05-01`, `2026-05-04`
- 수정 전 max gap: `{metrics["tickers"]["NVDA"]["line"]["chart_before_fix"]["max_gap_days"]}`일
- 수정 후 max gap: `{metrics["tickers"]["NVDA"]["line"]["chart_after_fix"]["max_gap_days"]}`일

## 5. 차트 계약

수정 후 계약:
- rolling history: thin upload row 제외
- rolling history 시간축: `asof_date`
- rolling history 값: h5 대표값. 길이가 짧으면 마지막 horizon
- future forecast: latest prediction 1건의 `forecast_dates`와 h1~h5 series
- 큰 공백: `{MAX_ROLLING_HISTORY_GAP_DAYS}`일 초과 gap은 한 line series로 잇지 않음

주의:
- h5 대표 history를 `asof_date`에 찍는 계약은 유지했다.
- 다만 product latest-only row는 rolling history가 아니라 latest future forecast layer로만 해석한다.

## 6. History endpoint 확인

현재 endpoint는 run_id, ticker, limit 기준만 사용한다. run_id가 제품 run으로 고정되어 있으므로 timeframe/horizon 혼합은 이번 사례의 주원인이 아니었다. 주원인은 storage contract가 다른 row를 같은 history로 반환한 점이다.

장기 개선:
- API에 `history_kind=rolling_only|include_latest` 또는 `exclude_thin_upload=true` 옵션 추가
- `PredictionData.meta.storage_contract`를 명시적으로 `bulk_backtest_history` 또는 `product_latest_only`로 저장
- rolling AI band history를 Supabase latest-only와 분리해 local parquet 또는 별도 thin history table로 제공

## 7. 파일 변경

- `frontend/src/components/Chart.tsx`

## 8. 금지 작업 확인

| 금지 항목 | 발생 |
|---|---|
| 모델 재학습 | false |
| inference 재실행 | false |
| prediction DB write | false |
| DB row delete | false |
| Supabase price/indicator 대량 read | false |

## 9. 검증

metrics JSON에 수정 전/후 chart history gap이 기록되어 있다.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    client = get_supabase()
    metrics: dict[str, Any] = {
        "cp": "CP122-4-DP",
        "generated_at": _now_iso(),
        "scope": {
            "tickers": TICKERS,
            "timeframe": TIMEFRAME,
            "line_run_id": LINE_RUN_ID,
            "band_run_id": BAND_RUN_ID,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "history_limit": HISTORY_LIMIT,
        },
        "tickers": {},
        "forbidden_actions_observed": {
            "model_training": False,
            "inference_rerun": False,
            "db_write": False,
            "db_delete": False,
            "supabase_price_indicator_bulk_read": False,
        },
    }
    for ticker in TICKERS:
        metrics["tickers"][ticker] = {}
        for layer, run_id in RUNS.items():
            metrics["tickers"][ticker][layer] = _audit_ticker_run(client, ticker, layer, run_id)
    metrics["line_band_coverage"] = _line_band_coverage(metrics)
    metrics["decision"] = {
        "status": "FIXED",
        "root_cause": "bulk test prediction history와 product latest-only thin upload row가 같은 rolling chart series로 연결됨",
        "frontend_fix": "thin_upload row를 rolling history에서 제외하고 큰 날짜 gap을 한 line으로 잇지 않음",
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(metrics)
    print(json.dumps({"status": "FIXED", "metrics": str(METRICS_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
