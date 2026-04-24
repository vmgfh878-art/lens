from __future__ import annotations

from dataclasses import dataclass
from math import floor

import pandas as pd

SPLIT_RATIO = (0.7, 0.15, 0.15)
MAX_HORIZON_BY_TIMEFRAME = {
    "1D": 20,
    "1W": 12,
}
FUNDAMENTAL_INSUFFICIENT_TICKERS = {
    "AMP",
    "APA",
    "BALL",
    "BK",
    "BXP",
    "CHTR",
    "CINF",
    "CPAY",
    "DUK",
    "EA",
    "EME",
    "EXE",
    "GLW",
    "HOOD",
    "INVH",
    "KEYS",
    "LMT",
    "MS",
    "PRU",
    "Q",
    "RF",
    "T",
    "TDG",
    "UBER",
    "VICI",
    "XYZ",
}


@dataclass(frozen=True)
class SplitSlice:
    start: int
    end: int

    @property
    def count(self) -> int:
        return max(self.end - self.start, 0)


@dataclass(frozen=True)
class TickerSplitSpec:
    ticker: str
    timeframe: str
    row_count: int
    sample_count: int
    train: SplitSlice
    val: SplitSlice
    test: SplitSlice
    gap: int


def _resolved_sample_count(row_count: int, seq_len: int, h_max: int) -> int:
    # anchor index 기준 마지막 예측 구간이 완전히 들어가도록 최대 horizon만큼 뒤를 남긴다.
    return max(row_count - seq_len - h_max + 1, 0)


def _effective_sample_count(sample_count: int, h_max: int) -> int:
    # train-val, val-test 경계에 각각 h_max 만큼 gap을 비워 두기 위해 사용 가능한 샘플 수를 줄인다.
    return max(sample_count - (2 * h_max), 0)


def make_splits(
    ticker_frame: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    h_max: int,
    min_fold_samples: int = 50,
) -> TickerSplitSpec:
    frame = ticker_frame.sort_values("date").reset_index(drop=True)
    row_count = len(frame)
    sample_count = _resolved_sample_count(row_count, seq_len, h_max)
    if sample_count < 1:
        raise ValueError("Gate A")

    effective_sample_count = _effective_sample_count(sample_count, h_max)
    if effective_sample_count < 1:
        raise ValueError("Gate A")

    train_count = floor(effective_sample_count * SPLIT_RATIO[0])
    val_count = floor(effective_sample_count * SPLIT_RATIO[1])
    test_count = effective_sample_count - train_count - val_count

    if train_count < min_fold_samples:
        raise ValueError("Gate B")
    if val_count < min_fold_samples or test_count < min_fold_samples:
        raise ValueError("Gate C")

    train = SplitSlice(0, train_count)
    val = SplitSlice(train.end + h_max, train.end + h_max + val_count)
    test = SplitSlice(val.end + h_max, val.end + h_max + test_count)
    if test.end > sample_count:
        raise ValueError("Gate C")

    return TickerSplitSpec(
        ticker=str(frame["ticker"].iloc[0]),
        timeframe=timeframe,
        row_count=row_count,
        sample_count=sample_count,
        train=train,
        val=val,
        test=test,
        gap=h_max,
    )


def eligible_tickers(
    feature_frame: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    h_max: int,
    min_fold_samples: int = 50,
) -> tuple[list[str], dict[str, str]]:
    filtered = feature_frame.copy()
    if "timeframe" in filtered.columns:
        filtered = filtered[filtered["timeframe"] == timeframe]

    eligible: list[str] = []
    excluded: dict[str, str] = {}

    for ticker, ticker_frame in filtered.groupby("ticker", sort=True):
        if ticker in FUNDAMENTAL_INSUFFICIENT_TICKERS:
            excluded[str(ticker)] = "Gate fundamentals"
            continue

        try:
            make_splits(
                ticker_frame,
                timeframe=timeframe,
                seq_len=seq_len,
                h_max=h_max,
                min_fold_samples=min_fold_samples,
            )
        except ValueError as exc:
            excluded[str(ticker)] = str(exc)
            continue
        eligible.append(str(ticker))

    return eligible, excluded


def build_split_specs(
    feature_frame: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    h_max: int,
    min_fold_samples: int = 50,
) -> tuple[dict[str, TickerSplitSpec], dict[str, str]]:
    filtered = feature_frame.copy()
    if "timeframe" in filtered.columns:
        filtered = filtered[filtered["timeframe"] == timeframe]

    specs: dict[str, TickerSplitSpec] = {}
    excluded: dict[str, str] = {}

    for ticker, ticker_frame in filtered.groupby("ticker", sort=True):
        if ticker in FUNDAMENTAL_INSUFFICIENT_TICKERS:
            excluded[str(ticker)] = "Gate fundamentals"
            continue

        try:
            specs[str(ticker)] = make_splits(
                ticker_frame,
                timeframe=timeframe,
                seq_len=seq_len,
                h_max=h_max,
                min_fold_samples=min_fold_samples,
            )
        except ValueError as exc:
            excluded[str(ticker)] = str(exc)

    return specs, excluded
