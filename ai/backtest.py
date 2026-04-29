from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing import normalize_ai_timeframe
from ai.storage import fetch_run_evaluations, fetch_run_predictions, get_model_run, save_backtest_results
from backend.collector.repositories.base import fetch_frame

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저장된 예측 결과로 규칙 기반 백테스트를 실행한다")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--strategy-name", default="band_breakout_v1")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def build_backtest_frame(run_id: str, timeframe: str | None = None) -> pd.DataFrame:
    predictions = fetch_run_predictions(run_id, timeframe)
    evaluations = fetch_run_evaluations(run_id, timeframe)
    if predictions.empty or evaluations.empty:
        raise ValueError("백테스트 대상 예측 또는 평가 데이터가 없습니다. 먼저 추론 저장을 실행하세요.")

    frame = predictions.merge(
        evaluations,
        on=["run_id", "ticker", "timeframe", "asof_date"],
        suffixes=("_pred", "_eval"),
    )
    frame["asof_date"] = pd.to_datetime(frame["asof_date"]).dt.strftime("%Y-%m-%d")

    tickers = sorted(frame["ticker"].astype(str).str.upper().unique().tolist())
    min_date = frame["asof_date"].min()
    max_date = frame["asof_date"].max()
    price_frame = fetch_frame(
        "price_data",
        columns="ticker,date,close,adjusted_close",
        filters=[("in", "ticker", tickers), ("gte", "date", min_date), ("lte", "date", max_date)],
        order_by="date",
    )
    if price_frame.empty:
        raise ValueError("anchor close를 찾을 수 없습니다. price_data를 확인하세요.")

    price_frame["date"] = pd.to_datetime(price_frame["date"]).dt.strftime("%Y-%m-%d")
    if "adjusted_close" not in price_frame.columns:
        price_frame["adjusted_close"] = price_frame["close"]
    price_frame["anchor_close"] = price_frame["adjusted_close"].fillna(price_frame["close"])
    price_lookup = {
        (row["ticker"], row["date"]): float(row["anchor_close"])
        for _, row in price_frame.iterrows()
    }

    frame["anchor_close"] = frame.apply(
        lambda row: price_lookup.get((str(row["ticker"]).upper(), row["asof_date"])),
        axis=1,
    )
    frame = frame.dropna(subset=["anchor_close"]).copy()
    frame["line_last"] = frame["line_series"].apply(lambda values: float(values[-1]) if values else None)
    frame["actual_last"] = frame["actual_series"].apply(lambda values: float(values[-1]) if values else None)
    frame = frame.dropna(subset=["line_last", "actual_last"]).copy()

    frame["line_return"] = (frame["line_last"] / frame["anchor_close"]) - 1.0
    frame["realized_return"] = (frame["actual_last"] / frame["anchor_close"]) - 1.0
    return frame.sort_values("asof_date").reset_index(drop=True)


def run_rule_based_backtest(
    prediction_frame: pd.DataFrame,
    *,
    strategy_name: str = "band_breakout_v1",
    fee_bps: float = 10.0,
) -> dict[str, Any]:
    """규칙 기반 시그널의 방향성과 손익 특성을 빠르게 비교한다."""
    if prediction_frame.empty:
        raise ValueError("백테스트 입력 데이터가 비어 있습니다.")

    frame = prediction_frame.sort_values(["asof_date", "ticker"]).copy()
    signal_to_position = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
    fee_rate = float(fee_bps) / 10000.0

    date_returns: list[float] = []
    gross_returns: list[float] = []
    turnover_values: list[float] = []
    trade_counts: list[int] = []
    previous_weights: dict[str, float] = {}

    for asof_date, group in frame.groupby("asof_date", sort=True):
        del asof_date
        raw_positions = group.set_index("ticker")["signal"].map(signal_to_position).fillna(0.0).astype(float)
        exposure = float(raw_positions.abs().sum())
        weights = (
            {str(ticker): float(position / exposure) for ticker, position in raw_positions.items() if position != 0.0}
            if exposure > 0.0
            else {}
        )
        realized_returns = group.set_index("ticker")["realized_return"].astype(float).to_dict()
        gross_return = sum(weights.get(str(ticker), 0.0) * float(return_value) for ticker, return_value in realized_returns.items())
        all_tickers = set(weights) | set(previous_weights)
        turnover = sum(abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in all_tickers)

        gross_returns.append(gross_return)
        turnover_values.append(turnover)
        trade_counts.append(sum(1 for ticker in all_tickers if abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) > 0.0))
        date_returns.append(gross_return - (turnover * fee_rate))
        previous_weights = weights

    strategy_return = pd.Series(date_returns, dtype="float64")
    gross_strategy_return = pd.Series(gross_returns, dtype="float64")
    turnover_series = pd.Series(turnover_values, dtype="float64")
    cumulative = (1.0 + strategy_return).cumprod()
    gross_cumulative = (1.0 + gross_strategy_return).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0

    wins = strategy_return[strategy_return > 0]
    losses = strategy_return[strategy_return < 0].abs()
    profit_factor = float(wins.sum() / losses.sum()) if losses.sum() > 0 else None
    strategy_std = float(strategy_return.std(ddof=0))
    gross_std = float(gross_strategy_return.std(ddof=0))
    fee_adjusted_return_pct = float((cumulative.iloc[-1] - 1.0) * 100.0)
    gross_return_pct = float((gross_cumulative.iloc[-1] - 1.0) * 100.0)
    fee_adjusted_sharpe = float(strategy_return.mean() / strategy_std) if strategy_std else 0.0

    return {
        "strategy_name": strategy_name,
        "return_pct": fee_adjusted_return_pct,
        "mdd": float(drawdown.min()),
        "sharpe": fee_adjusted_sharpe,
        "win_rate": float((strategy_return > 0).mean()),
        "profit_factor": profit_factor,
        "num_trades": int(sum(trade_counts)),
        "fee_adjusted_return_pct": fee_adjusted_return_pct,
        "fee_adjusted_sharpe": fee_adjusted_sharpe,
        "avg_turnover": float(turnover_series.mean()) if not turnover_series.empty else 0.0,
        "meta": {
            "rows": len(frame),
            "portfolio_dates": int(len(strategy_return)),
            "position_contract": "date_equal_abs_exposure",
            "avg_realized_return": float(frame["realized_return"].mean()),
            "avg_line_return": float(frame["line_return"].mean()),
            "fee_bps": float(fee_bps),
            "gross_return_pct": gross_return_pct,
            "gross_sharpe": float(gross_strategy_return.mean() / gross_std) if gross_std else 0.0,
        },
    }


def persist_backtest_result(run_id: str, timeframe: str, result: dict[str, Any]) -> None:
    save_backtest_results(
        [
            {
                "run_id": run_id,
                "strategy_name": result["strategy_name"],
                "timeframe": timeframe,
                "return_pct": result["return_pct"],
                "mdd": result["mdd"],
                "sharpe": result["sharpe"],
                "win_rate": result["win_rate"],
                "profit_factor": result["profit_factor"],
                "num_trades": result["num_trades"],
                "meta": result["meta"],
            }
        ]
    )


def run_backtest(
    *,
    run_id: str,
    timeframe: str | None = None,
    strategy_name: str,
    fee_bps: float = 10.0,
    save: bool = False,
) -> dict[str, Any]:
    if timeframe is not None:
        normalize_ai_timeframe(timeframe)
    # CP12: NaN으로 실패한 run의 backtest 결과 저장을 차단한다.
    model_run = get_model_run(run_id)
    if model_run is None:
        raise ValueError(f"run_id={run_id}에 해당하는 model_runs 기록이 없습니다.")
    run_status = str(model_run.get("status") or "completed")
    if run_status != "completed":
        raise ValueError(
            f"run_id={run_id} status={run_status}: completed 상태의 run에서만 backtest를 실행할 수 있습니다."
        )
    frame = build_backtest_frame(run_id, timeframe)
    result = run_rule_based_backtest(frame, strategy_name=strategy_name, fee_bps=fee_bps)
    resolved_timeframe = timeframe or str(frame["timeframe"].iloc[0])
    normalize_ai_timeframe(resolved_timeframe)
    if save:
        persist_backtest_result(run_id, resolved_timeframe, result)
    return {
        "run_id": run_id,
        "timeframe": resolved_timeframe,
        **result,
    }


if __name__ == "__main__":
    args = parse_args()
    result = run_backtest(
        run_id=args.run_id,
        timeframe=args.timeframe,
        strategy_name=args.strategy_name,
        fee_bps=args.fee_bps,
        save=args.save,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
