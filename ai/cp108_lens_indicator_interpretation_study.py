from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


LINE_RUN_ID = "patchtst-1D-efad3c29d803"
BAND_RUN_ID = "cnn_lstm-1D-d0c780dee5e8"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_DATA_DIR = Path("data/parquet")
DEFAULT_OUTPUT_JSON = Path("docs/cp108_lens_indicator_interpretation_study_metrics.json")
DEFAULT_OUTPUT_CSV = Path("docs/cp108_lens_indicator_interpretation_rules.csv")
EXPLORE_START = "2025-06-18"
HOLDOUT_END = "2026-05-01"


@dataclass
class TickerStudyData:
    ticker: str
    rows: pd.DataFrame
    line_prediction_count: int
    band_prediction_count: int
    min_asof: str
    max_asof: str


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_backend_url(value: str | None) -> str:
    raw = (value or DEFAULT_BACKEND_URL).strip().strip("\"'")
    if not raw:
        raw = DEFAULT_BACKEND_URL
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def normalize_rsi(value: Any) -> float | None:
    if not finite(value):
        return None
    number = float(value)
    return number * 100 if 0 <= number <= 1 else number


def read_json_url(url: str, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_prediction_history(backend_url: str, ticker: str, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = min(max(limit, 1), 200)
    params = urllib.parse.urlencode({"run_id": run_id, "limit": safe_limit})
    url = f"{backend_url}/api/v1/stocks/{urllib.parse.quote(ticker)}/predictions/history?{params}"
    try:
        payload = read_json_url(url)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404, 422}:
            return []
        raise
    return list(payload.get("data") or [])


def load_universe(data_dir: Path, limit: int) -> list[str]:
    stock_info = pd.read_parquet(data_dir / "stock_info.parquet")
    tickers = stock_info["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    return tickers[:limit]


def load_market_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_parquet(data_dir / "price_data_yfinance.parquet")
    close_column = "adjusted_close" if "adjusted_close" in price.columns else "close"
    price = price[["ticker", "date", close_column]].copy().rename(columns={close_column: "close"})
    price["ticker"] = price["ticker"].astype(str).str.upper()
    price["date"] = price["date"].astype(str)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])

    indicators = pd.read_parquet(data_dir / "indicators_yfinance_1D.parquet")
    indicator_columns = ["ticker", "date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"]
    indicators = indicators[[column for column in indicator_columns if column in indicators.columns]].copy()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["date"] = indicators["date"].astype(str)
    for column in ["rsi", "ma_60_ratio", "vol_change", "atr_ratio"]:
        if column in indicators.columns:
            indicators[column] = pd.to_numeric(indicators[column], errors="coerce")
    indicators = indicators.sort_values(["ticker", "date"])
    return price, indicators


def last_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    for value in reversed(values):
        if finite(value):
            return float(value)
    return None


def first_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    for value in values:
        if finite(value):
            return float(value)
    return None


def min_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    usable = [float(value) for value in values if finite(value)]
    return min(usable) if usable else None


def max_finite(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    usable = [float(value) for value in values if finite(value)]
    return max(usable) if usable else None


def line_end_value(prediction: dict[str, Any]) -> float | None:
    conservative = last_finite(prediction.get("conservative_series"))
    if conservative is not None:
        return conservative
    return last_finite(prediction.get("line_series"))


def line_start_value(prediction: dict[str, Any]) -> float | None:
    conservative = first_finite(prediction.get("conservative_series"))
    if conservative is not None:
        return conservative
    return first_finite(prediction.get("line_series"))


def map_by_asof(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        asof = str(row.get("asof_date") or "")
        if asof:
            mapped[asof] = row
    return mapped


def future_closes_for_forecast(price_by_date: dict[str, float], forecast_dates: list[str]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for date in forecast_dates:
        close = price_by_date.get(str(date))
        if close is not None and math.isfinite(close):
            values.append((str(date), float(close)))
    return values


def close_after_n_trading_days(price_dates: list[str], price_by_date: dict[str, float], asof_date: str, n_days: int) -> float | None:
    try:
        index = price_dates.index(asof_date)
    except ValueError:
        return None
    target_index = index + n_days
    if target_index >= len(price_dates):
        return None
    return price_by_date.get(price_dates[target_index])


def realized_volatility(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    returns = [values[index] / values[index - 1] - 1 for index in range(1, len(values)) if values[index - 1] > 0]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns)


def build_ticker_study_data(
    ticker: str,
    price_rows: pd.DataFrame,
    indicator_rows: pd.DataFrame,
    line_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
) -> TickerStudyData | None:
    if not line_rows or not band_rows:
        return None

    price = price_rows[price_rows["ticker"] == ticker].copy()
    if price.empty:
        return None
    price = price.sort_values("date").drop_duplicates("date", keep="last")
    price_by_date = dict(zip(price["date"].astype(str), price["close"].astype(float)))
    price_dates = price["date"].astype(str).tolist()

    indicators = indicator_rows[indicator_rows["ticker"] == ticker].copy()
    if indicators.empty:
        indicators = pd.DataFrame(columns=["date", "rsi", "ma_60_ratio", "vol_change", "atr_ratio"])
    indicators = indicators.sort_values("date").drop_duplicates("date", keep="last")
    indicator_by_date = indicators.set_index("date").to_dict("index") if not indicators.empty else {}

    line_by_asof = map_by_asof(line_rows)
    band_by_asof = map_by_asof(band_rows)
    common_dates = sorted(set(line_by_asof) & set(band_by_asof))
    records: list[dict[str, Any]] = []

    for asof_date in common_dates:
        close = price_by_date.get(asof_date)
        if close is None or close <= 0:
            continue
        line_prediction = line_by_asof[asof_date]
        band_prediction = band_by_asof[asof_date]

        line_end = line_end_value(line_prediction)
        line_start = line_start_value(line_prediction)
        lower = min_finite(band_prediction.get("lower_band_series"))
        upper = max_finite(band_prediction.get("upper_band_series"))
        if line_end is None or line_start is None or lower is None or upper is None or upper <= lower:
            continue

        forecast_dates = [str(date) for date in band_prediction.get("forecast_dates") or []]
        forecast_closes = future_closes_for_forecast(price_by_date, forecast_dates)
        if len(forecast_closes) < 3:
            continue
        future_values = [value for _, value in forecast_closes]
        final_future_close = future_values[-1]
        future_return_h5 = final_future_close / close - 1
        future_min_return_h5 = min(future_values) / close - 1
        future_max_return_h5 = max(future_values) / close - 1
        future_abs_return_h5 = abs(future_return_h5)
        realized_range_h5 = future_max_return_h5 - future_min_return_h5
        future_return_h10 = None
        h10_close = close_after_n_trading_days(price_dates, price_by_date, asof_date, 10)
        if h10_close is not None and close > 0:
            future_return_h10 = h10_close / close - 1

        lower_series = band_prediction.get("lower_band_series") or []
        upper_series = band_prediction.get("upper_band_series") or []
        lower_breach = False
        upper_breach = False
        for idx, (date, actual_close) in enumerate(forecast_closes):
            if idx < len(lower_series) and finite(lower_series[idx]) and actual_close < float(lower_series[idx]):
                lower_breach = True
            if idx < len(upper_series) and finite(upper_series[idx]) and actual_close > float(upper_series[idx]):
                upper_breach = True

        indicator = indicator_by_date.get(asof_date, {})
        rsi = normalize_rsi(indicator.get("rsi"))
        ma60_ratio = indicator.get("ma_60_ratio")
        vol_change = indicator.get("vol_change")
        atr_ratio = indicator.get("atr_ratio")

        width_return = (upper - lower) / close
        line_position = (line_end - lower) / (upper - lower)
        records.append(
            {
                "ticker": ticker,
                "date": asof_date,
                "close": close,
                "line_return_signal": line_end / close - 1,
                "line_slope_signal": (line_end - line_start) / close,
                "lower_band_risk": lower / close - 1,
                "upper_band_return": upper / close - 1,
                "band_width_return": width_return,
                "line_position_in_band": line_position,
                "lower_breach_event": lower_breach,
                "upper_breach_event": upper_breach,
                "future_return_h5": future_return_h5,
                "future_return_h10": future_return_h10,
                "future_min_return_h5": future_min_return_h5,
                "future_max_return_h5": future_max_return_h5,
                "future_abs_return_h5": future_abs_return_h5,
                "realized_range_h5": realized_range_h5,
                "realized_vol_h5": realized_volatility([close, *future_values]),
                "rsi": rsi,
                "ma60_ratio": ma60_ratio if finite(ma60_ratio) else None,
                "vol_change": vol_change if finite(vol_change) else None,
                "atr_ratio": atr_ratio if finite(atr_ratio) else None,
            }
        )

    if len(records) < 5:
        return None

    rows = pd.DataFrame(records).sort_values("date").drop_duplicates("date", keep="last")
    rows["band_width_ref"] = rows["band_width_return"].shift(1).rolling(20, min_periods=5).median()
    rows["band_width_expansion"] = rows["band_width_return"] / rows["band_width_ref"]
    rows["band_width_percentile"] = rows["band_width_return"].rank(pct=True)
    rows = rows.replace([math.inf, -math.inf], math.nan)
    rows = add_interpretation_columns(rows)

    return TickerStudyData(
        ticker=ticker,
        rows=rows,
        line_prediction_count=len(line_rows),
        band_prediction_count=len(band_rows),
        min_asof=str(rows["date"].min()),
        max_asof=str(rows["date"].max()),
    )


def line_return_bucket(value: float) -> str:
    if value < -0.01:
        return "약함"
    if value < 0:
        return "소폭 약함"
    if value < 0.01:
        return "소폭 양호"
    return "양호"


def line_slope_bucket(value: float) -> str:
    if value < -0.002:
        return "하향"
    if value > 0.002:
        return "상향"
    return "평탄"


def lower_risk_bucket(value: float) -> str:
    if value <= -0.08:
        return "깊음"
    if value <= -0.05:
        return "주의"
    return "보통"


def width_regime_bucket(value: float) -> str:
    if value >= 0.75:
        return "넓음"
    if value <= 0.25:
        return "좁음"
    return "보통"


def expansion_bucket(value: Any) -> str:
    if not finite(value):
        return "기준 부족"
    number = float(value)
    if number >= 1.25:
        return "확장"
    if number <= 0.90:
        return "축소"
    return "보통"


def position_bucket(value: float) -> str:
    if value < 0:
        return "밴드 하단 밖"
    if value <= 0.33:
        return "하단 근처"
    if value <= 0.66:
        return "중앙"
    if value <= 1:
        return "상단 근처"
    return "밴드 상단 밖"


def disagreement_bucket(row: pd.Series) -> str:
    line_good = float(row["line_return_signal"]) >= 0
    line_weak = float(row["line_return_signal"]) < 0
    lower_risky = float(row["lower_band_risk"]) <= -0.05
    width_expanded = finite(row.get("band_width_expansion")) and float(row["band_width_expansion"]) >= 1.25
    band_risky = lower_risky or width_expanded
    band_stable = not lower_risky and not width_expanded

    if line_good and band_stable:
        return "라인 양호 + 밴드 안정"
    if line_good and band_risky:
        return "라인 양호 + 밴드 불확실"
    if line_weak and band_stable:
        return "라인 약함 + 밴드 안정"
    return "라인 약함 + 밴드 위험"


def add_interpretation_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["line_return_bucket"] = rows["line_return_signal"].map(line_return_bucket)
    rows["line_slope_bucket"] = rows["line_slope_signal"].map(line_slope_bucket)
    rows["lower_risk_bucket"] = rows["lower_band_risk"].map(lower_risk_bucket)
    rows["band_width_regime"] = rows["band_width_percentile"].map(width_regime_bucket)
    rows["band_width_expansion_bucket"] = rows["band_width_expansion"].map(expansion_bucket)
    rows["line_position_bucket"] = rows["line_position_in_band"].map(position_bucket)
    rows["line_band_disagreement"] = rows.apply(disagreement_bucket, axis=1)
    rows["large_loss_event"] = rows["future_min_return_h5"] <= -0.03
    rows["future_positive_event"] = rows["future_return_h5"] > 0
    return rows


def mean(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(usable) if usable else None


def rate(values: list[Any]) -> float | None:
    usable = [bool(value) for value in values if value is not None]
    return sum(1 for value in usable if value) / len(usable) if usable else None


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0}
    return {
        "count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "avg_future_return_h5_pct": mean((frame["future_return_h5"] * 100).tolist()),
        "median_future_return_h5_pct": float(frame["future_return_h5"].median() * 100),
        "positive_rate": rate(frame["future_positive_event"].tolist()),
        "avg_future_min_return_h5_pct": mean((frame["future_min_return_h5"] * 100).tolist()),
        "avg_future_abs_return_h5_pct": mean((frame["future_abs_return_h5"] * 100).tolist()),
        "avg_realized_range_h5_pct": mean((frame["realized_range_h5"] * 100).tolist()),
        "avg_realized_vol_h5_pct": mean((frame["realized_vol_h5"] * 100).tolist()),
        "large_loss_rate": rate(frame["large_loss_event"].tolist()),
        "lower_breach_rate": rate(frame["lower_breach_event"].tolist()),
        "upper_breach_rate": rate(frame["upper_breach_event"].tolist()),
        "avg_future_return_h10_pct": mean((frame["future_return_h10"] * 100).tolist()),
    }


def group_summary(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value, group in frame.groupby(column, dropna=False):
        result[str(value)] = summarize_frame(group)
    return result


def comparison(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_summary = summarize_frame(left)
    right_summary = summarize_frame(right)
    return {
        "left": left_summary,
        "right": right_summary,
        "diff": {
            "avg_future_return_h5_pct": (left_summary.get("avg_future_return_h5_pct") or 0)
            - (right_summary.get("avg_future_return_h5_pct") or 0),
            "avg_future_abs_return_h5_pct": (left_summary.get("avg_future_abs_return_h5_pct") or 0)
            - (right_summary.get("avg_future_abs_return_h5_pct") or 0),
            "avg_realized_range_h5_pct": (left_summary.get("avg_realized_range_h5_pct") or 0)
            - (right_summary.get("avg_realized_range_h5_pct") or 0),
            "large_loss_rate": (left_summary.get("large_loss_rate") or 0) - (right_summary.get("large_loss_rate") or 0),
            "avg_future_return_h10_pct": (left_summary.get("avg_future_return_h10_pct") or 0)
            - (right_summary.get("avg_future_return_h10_pct") or 0),
        },
    }


def evaluate_questions(frame: pd.DataFrame) -> dict[str, Any]:
    expanded = frame[frame["band_width_expansion_bucket"] == "확장"]
    not_expanded = frame[frame["band_width_expansion_bucket"].isin(["보통", "축소"])]
    deep_lower = frame[frame["lower_risk_bucket"] == "깊음"]
    normal_lower = frame[frame["lower_risk_bucket"] == "보통"]
    weak_line = frame[frame["line_return_bucket"].isin(["약함", "소폭 약함"])]
    good_line = frame[frame["line_return_bucket"].isin(["소폭 양호", "양호"])]
    upper_breach = frame[frame["upper_breach_event"]]
    no_upper_breach = frame[~frame["upper_breach_event"]]
    both_risk = frame[frame["line_band_disagreement"] == "라인 약함 + 밴드 위험"]
    line_good_band_wide = frame[frame["line_band_disagreement"] == "라인 양호 + 밴드 불확실"]
    line_good_band_stable = frame[frame["line_band_disagreement"] == "라인 양호 + 밴드 안정"]

    return {
        "band_width_expansion_vs_not": comparison(expanded, not_expanded),
        "deep_lower_vs_normal_lower": comparison(deep_lower, normal_lower),
        "weak_line_vs_good_line": comparison(weak_line, good_line),
        "upper_breach_vs_no_upper_breach": comparison(upper_breach, no_upper_breach),
        "both_risk_summary": summarize_frame(both_risk),
        "line_good_band_wide_vs_stable": comparison(line_good_band_wide, line_good_band_stable),
    }


def interpretation_rules(question_metrics: dict[str, Any]) -> list[dict[str, str]]:
    width_diff = question_metrics["band_width_expansion_vs_not"]["diff"]
    lower_diff = question_metrics["deep_lower_vs_normal_lower"]["diff"]
    line_diff = question_metrics["weak_line_vs_good_line"]["diff"]
    upper_diff = question_metrics["upper_breach_vs_no_upper_breach"]["diff"]
    wide_line_diff = question_metrics["line_good_band_wide_vs_stable"]["diff"]

    def decision_from_diff(value: float, threshold: float, good_label: str, weak_label: str) -> str:
        return good_label if abs(value) >= threshold else weak_label

    return [
        {
            "indicator": "line_return_signal",
            "natural_interpretation": "보수적 예측선의 h5 예상 수익 방향이다. 0보다 높으면 방향 신호가 양호하다고 본다.",
            "observed_result": f"라인 약함 구간의 h5 평균 수익률은 양호 구간 대비 {line_diff['avg_future_return_h5_pct']:.2f}%p 낮았다.",
            "decision": decision_from_diff(line_diff["avg_future_return_h5_pct"], 0.3, "전략 후보에 사용", "단독 신호로는 약함"),
            "next_rule_candidate": "진입은 line_return_signal >= 0을 기본으로 두되, 추세가 강하면 약한 음수도 보류 조건으로 허용한다.",
        },
        {
            "indicator": "line_slope_signal",
            "natural_interpretation": "예측선 경로가 위로 기울면 예측이 개선되는 흐름, 아래로 기울면 약화 흐름으로 본다.",
            "observed_result": "기울기는 보조 해석 지표로만 사용한다. slope 단독으로 전략 최적화하지 않았다.",
            "decision": "보조 후보",
            "next_rule_candidate": "line_return이 애매할 때 slope 하향이면 청산 confirm을 강화한다.",
        },
        {
            "indicator": "lower_band_risk",
            "natural_interpretation": "AI 밴드 하단이 현재가보다 얼마나 깊은지 보는 하방 위험 여유 폭이다.",
            "observed_result": f"하단이 깊은 구간의 큰 손실률은 보통 구간 대비 {lower_diff['large_loss_rate']:.2%}p 차이를 보였다.",
            "decision": decision_from_diff(lower_diff["large_loss_rate"], 0.03, "risk veto 후보", "단독 매도 신호로는 약함"),
            "next_rule_candidate": "line이 약할 때 lower_band_risk가 깊으면 청산 confirm으로 사용한다.",
        },
        {
            "indicator": "band_width_regime",
            "natural_interpretation": "밴드 폭이 티커 자신의 평소 분포에서 넓은지 좁은지 보는 불확실성 지표다.",
            "observed_result": "밴드 폭은 위험 그 자체보다 변동 가능성의 폭으로 해석하는 편이 맞다.",
            "decision": "불확실성 후보",
            "next_rule_candidate": "넓은 밴드는 신규 진입 제한이나 포지션 유지 확인 조건으로만 사용한다.",
        },
        {
            "indicator": "band_width_expansion",
            "natural_interpretation": "최근 평균 대비 밴드 폭이 갑자기 넓어졌는지 보는 불확실성 확장 신호다.",
            "observed_result": f"확장 구간의 h5 실현 range는 비확장 구간 대비 {width_diff['avg_realized_range_h5_pct']:.2f}%p 차이를 보였다.",
            "decision": decision_from_diff(width_diff["avg_realized_range_h5_pct"], 0.3, "risk confirm 후보", "보조 후보"),
            "next_rule_candidate": "line 약화와 band_width_expansion이 동시에 발생할 때만 청산 후보로 둔다.",
        },
        {
            "indicator": "line_position_in_band",
            "natural_interpretation": "예측선이 AI 밴드 안에서 하단, 중앙, 상단 중 어디에 있는지 보는 위치 지표다.",
            "observed_result": "line과 band의 상대 위치를 보지만, 단독 방향 신호로 과장하지 않는다.",
            "decision": "해석 보조",
            "next_rule_candidate": "line이 밴드 하단 근처에 있을 때는 보수적 예측선 신뢰를 낮춘다.",
        },
        {
            "indicator": "lower_breach_event",
            "natural_interpretation": "실제 가격이 과거 예측 하단 밴드를 이탈했는지 보는 사후 검증 지표다.",
            "observed_result": "실시간 진입 신호가 아니라 밴드 보정 품질을 점검하는 calibration 지표다.",
            "decision": "전략 입력 제외",
            "next_rule_candidate": "전략 룰이 아니라 모델 평가/보정 리포트에 사용한다.",
        },
        {
            "indicator": "upper_breach_event",
            "natural_interpretation": "실제 가격이 과거 예측 상단 밴드를 돌파했는지 보는 사후 검증 지표다.",
            "observed_result": f"상단 돌파 구간의 h10 평균 수익률 차이는 {upper_diff['avg_future_return_h10_pct']:.2f}%p였다.",
            "decision": "무조건 매도 신호 아님",
            "next_rule_candidate": "상단 돌파는 line이 양호하면 추세 지속 후보, line이 약하면 과열 후보로 분리한다.",
        },
        {
            "indicator": "line_band_disagreement",
            "natural_interpretation": "라인과 밴드가 서로 다른 이야기를 하는 구간이다.",
            "observed_result": f"라인 양호+밴드 불확실 구간은 안정 구간 대비 h5 range가 {wide_line_diff['avg_realized_range_h5_pct']:.2f}%p 차이 났다.",
            "decision": "Balance 전략 핵심 후보",
            "next_rule_candidate": "line은 진입 방향, band는 신규 진입 제한과 청산 confirm으로 분리한다.",
        },
    ]


def write_rules_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["indicator", "natural_interpretation", "observed_result", "decision", "next_rule_candidate"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP108 Lens AI 지표 해석 검증")
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--universe-limit", type=int, default=100)
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()
    backend_url = normalize_backend_url(args.backend_url)
    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    print(f"backend={backend_url}")
    print(f"data_dir={data_dir}")
    print("CPU only: GPU 호출 없음")

    price, indicators = load_market_data(data_dir)
    universe = load_universe(data_dir, args.universe_limit)
    prepared: list[TickerStudyData] = []
    excluded: list[dict[str, str]] = []

    for index, ticker in enumerate(universe, start=1):
        line_rows = fetch_prediction_history(backend_url, ticker, LINE_RUN_ID, args.history_limit)
        band_rows = fetch_prediction_history(backend_url, ticker, BAND_RUN_ID, args.history_limit)
        item = build_ticker_study_data(ticker, price, indicators, line_rows, band_rows)
        if item is None:
            if not line_rows and not band_rows:
                reason = "line/band prediction history 없음"
            elif not line_rows:
                reason = "line prediction history 없음"
            elif not band_rows:
                reason = "band prediction history 없음"
            else:
                reason = "가격/forecast 날짜 매칭 부족"
            excluded.append({"ticker": ticker, "reason": reason})
        else:
            prepared.append(item)
        if index % 20 == 0:
            print(f"history fetch {index}/{len(universe)}")

    if not prepared:
        raise RuntimeError("분석 가능한 line/band prediction history가 없습니다.")

    frame = pd.concat([item.rows for item in prepared], ignore_index=True)
    frame = frame[(frame["date"] >= EXPLORE_START) & (frame["date"] <= HOLDOUT_END)].copy()
    frame = frame.dropna(subset=["future_return_h5", "future_min_return_h5", "band_width_return"])

    group_summaries = {
        "line_return_signal": group_summary(frame, "line_return_bucket"),
        "line_slope_signal": group_summary(frame, "line_slope_bucket"),
        "lower_band_risk": group_summary(frame, "lower_risk_bucket"),
        "band_width_regime": group_summary(frame, "band_width_regime"),
        "band_width_expansion": group_summary(frame, "band_width_expansion_bucket"),
        "line_position_in_band": group_summary(frame, "line_position_bucket"),
        "line_band_disagreement": group_summary(frame, "line_band_disagreement"),
        "lower_breach_event": group_summary(frame, "lower_breach_event"),
        "upper_breach_event": group_summary(frame, "upper_breach_event"),
    }
    question_metrics = evaluate_questions(frame)
    rule_rows = interpretation_rules(question_metrics)

    payload = {
        "cp": "CP108-P",
        "created_at_kst": time.strftime("%Y-%m-%d %H:%M:%S KST", time.localtime()),
        "line_run_id": LINE_RUN_ID,
        "band_run_id": BAND_RUN_ID,
        "backend_url": backend_url,
        "data_sources": {
            "price": str(data_dir / "price_data_yfinance.parquet"),
            "indicators": str(data_dir / "indicators_yfinance_1D.parquet"),
            "universe": str(data_dir / "stock_info.parquet"),
            "prediction_history_api": "/api/v1/stocks/{ticker}/predictions/history",
            "price_column_policy": "adjusted_close 우선, 없으면 close",
        },
        "forbidden_actions_confirmed": {
            "gpu_used": False,
            "model_training": False,
            "inference_execution": False,
            "db_write": False,
            "supabase_price_indicator_bulk_read": False,
            "frontend_modified": False,
            "strategy_optimization": False,
        },
        "universe": {
            "requested": len(universe),
            "usable": len(prepared),
            "excluded": len(excluded),
            "excluded_tickers": excluded,
        },
        "study_coverage": {
            "row_count": int(len(frame)),
            "ticker_count": int(frame["ticker"].nunique()),
            "min_asof": str(frame["date"].min()),
            "max_asof": str(frame["date"].max()),
            "line_prediction_rows_min": min(item.line_prediction_count for item in prepared),
            "line_prediction_rows_max": max(item.line_prediction_count for item in prepared),
            "band_prediction_rows_min": min(item.band_prediction_count for item in prepared),
            "band_prediction_rows_max": max(item.band_prediction_count for item in prepared),
        },
        "overall": summarize_frame(frame),
        "group_summaries": group_summaries,
        "question_metrics": question_metrics,
        "interpretation_rules": rule_rows,
        "line_trend_band_risk_context": {
            "line_trend_failure_summary": "CP107에서 line 단독은 수익률 방어를 일부 개선했지만 시장 참여율이 낮아 생존 실패",
            "band_risk_failure_summary": "CP106에서 band 단독은 손실 회피와 수익률 방어를 동시에 충족하지 못해 생존 실패",
        },
        "elapsed_seconds": time.time() - start_time,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rules_csv(output_csv, rule_rows)

    print(f"usable_tickers={len(prepared)} excluded={len(excluded)} rows={len(frame)}")
    print(f"json={output_json}")
    print(f"csv={output_csv}")
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
