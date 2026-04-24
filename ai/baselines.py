from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.loss import PinballLoss
from ai.preprocessing import default_horizon, normalize_ai_timeframe
from ai.storage import save_model_run
from backend.collector.repositories.base import fetch_frame


@dataclass
class BaselineResult:
    model_name: str
    ticker: str
    timeframe: str
    horizon: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens baseline 모델을 실행한다")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--timeframe", choices=["1D", "1W"], default="1D")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--save-run", action="store_true")
    return parser.parse_args()


def _resample_prices(price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = price_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    if timeframe == "1D":
        return frame.reset_index(drop=True)

    frame = frame.set_index("date")
    rule = "W-FRI" if timeframe == "1W" else "ME"
    aggregated = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "adjusted_close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["close"]).reset_index()
    aggregated["ticker"] = price_df["ticker"].iloc[0]
    return aggregated


def load_ticker_returns(ticker: str, timeframe: str) -> pd.DataFrame:
    frame = fetch_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume",
        filters=[("eq", "ticker", ticker.upper())],
        order_by="date",
    )
    if frame.empty:
        raise ValueError(f"{ticker.upper()} 가격 데이터가 없습니다.")
    resampled = _resample_prices(frame, timeframe)
    resampled["anchor_close"] = resampled["adjusted_close"].fillna(resampled["close"])
    resampled["return"] = resampled["anchor_close"].pct_change()
    return resampled.dropna(subset=["return"]).reset_index(drop=True)


def split_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise ValueError("baseline 분할에 필요한 시계열 길이가 부족합니다.")
    return (
        frame.iloc[:train_end].reset_index(drop=True),
        frame.iloc[train_end:val_end].reset_index(drop=True),
        frame.iloc[val_end:].reset_index(drop=True),
    )


def _build_windows(frame: pd.DataFrame, horizon: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    values = frame["return"].to_numpy(dtype="float32")
    dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    anchors = frame["anchor_close"].to_numpy(dtype="float32")
    for index in range(len(frame) - horizon):
        future_slice = slice(index + 1, index + 1 + horizon)
        if future_slice.stop > len(frame):
            break
        windows.append(
            {
                "asof_date": dates[index],
                "forecast_dates": dates[future_slice],
                "anchor_close": float(anchors[index]),
                "last_return": float(values[index]),
                "future_returns": values[future_slice],
            }
        )
    return windows


def _pinball_metric(prediction_quantiles: np.ndarray, targets: np.ndarray) -> float:
    criterion = PinballLoss((0.1, 0.5, 0.9), sort_quantiles=True)
    return float(
        criterion(
            prediction=torch.tensor(prediction_quantiles, dtype=torch.float32),
            target=torch.tensor(targets, dtype=torch.float32),
        ).item()
    )


def _evaluate_windows(windows: list[dict[str, Any]], predicted_line: np.ndarray, sigma: float) -> dict[str, float]:
    if not windows:
        return {"mae": 0.0, "mape": 0.0, "pinball_loss": 0.0}

    low_z = -1.2815515655446004
    high_z = 1.2815515655446004
    targets = np.stack([window["future_returns"] for window in windows], axis=0)
    lower = predicted_line + (low_z * sigma)
    upper = predicted_line + (high_z * sigma)
    quantiles = np.stack((lower, predicted_line, upper), axis=-1)
    mae = float(np.mean(np.abs(predicted_line - targets)))
    mape = float(np.mean(np.abs(predicted_line - targets) / np.clip(np.abs(targets), 1e-6, None)))
    return {
        "mae": mae,
        "mape": mape,
        "pinball_loss": _pinball_metric(quantiles, targets),
    }


def _naive_predictions(windows: list[dict[str, Any]], horizon: int) -> np.ndarray:
    return np.array([[window["last_return"]] * horizon for window in windows], dtype="float32")


def _drift_predictions(windows: list[dict[str, Any]], horizon: int, drift: float) -> np.ndarray:
    return np.array(
        [[window["last_return"] + (step * drift) for step in range(1, horizon + 1)] for window in windows],
        dtype="float32",
    )


def _arima_predictions(train_frame: pd.DataFrame, windows: list[dict[str, Any]], horizon: int) -> np.ndarray:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ARIMA baseline에는 statsmodels가 필요합니다.") from exc

    history = train_frame["return"].tolist()
    predictions: list[list[float]] = []
    for window in windows:
        model = ARIMA(history, order=(1, 0, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        predictions.append([float(value) for value in forecast])
        history.append(window["future_returns"][0])
    return np.array(predictions, dtype="float32")


def run_baselines(ticker: str, timeframe: str, horizon: int, *, save_run: bool = False) -> list[BaselineResult]:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    frame = load_ticker_returns(ticker, normalized_timeframe)
    train_frame, val_frame, test_frame = split_series(frame)
    val_windows = _build_windows(val_frame, horizon)
    test_windows = _build_windows(test_frame, horizon)
    train_sigma = float(train_frame["return"].std(ddof=0) or 0.0)
    drift = float(train_frame["return"].mean())

    baseline_specs = {
        "baseline_naive": lambda windows: _naive_predictions(windows, horizon),
        "baseline_drift": lambda windows: _drift_predictions(windows, horizon, drift),
        "baseline_arima": lambda windows: _arima_predictions(train_frame, windows, horizon),
    }

    results: list[BaselineResult] = []
    for model_name, builder in baseline_specs.items():
        val_prediction = builder(val_windows)
        test_prediction = builder(test_windows)
        val_metrics = _evaluate_windows(val_windows, val_prediction, train_sigma)
        test_metrics = _evaluate_windows(test_windows, test_prediction, train_sigma)
        result = BaselineResult(
            model_name=model_name,
            ticker=ticker.upper(),
            timeframe=normalized_timeframe,
            horizon=horizon,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        results.append(result)

        if save_run:
            save_model_run(
                {
                    "run_id": f"{model_name}-{uuid4().hex[:12]}",
                    "model_name": model_name,
                    "timeframe": normalized_timeframe,
                    "horizon": horizon,
                    "feature_version": "baseline_v1",
                    "band_quantile_low": 0.1,
                    "band_quantile_high": 0.9,
                    "alpha": 1.0,
                    "beta": 2.0,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "config": {
                        "ticker": ticker.upper(),
                        "model_type": model_name,
                        "sigma": train_sigma,
                        "drift": drift,
                    },
                    "checkpoint_path": None,
                }
            )

    return results


if __name__ == "__main__":
    args = parse_args()
    resolved_horizon = args.horizon or default_horizon(args.timeframe)
    results = run_baselines(
        ticker=args.ticker,
        timeframe=args.timeframe,
        horizon=resolved_horizon,
        save_run=args.save_run,
    )
    print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2))
