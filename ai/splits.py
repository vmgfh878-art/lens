from __future__ import annotations

from dataclasses import dataclass
from math import floor

import pandas as pd

SPLIT_RATIO = (0.7, 0.15, 0.15)
SPLIT_MODE_CALENDAR_ALIGNED = "calendar_aligned"
SPLIT_MODE_LEGACY_TICKER_INDEX = "legacy_ticker_index"
SPLIT_MODE_CHOICES = (SPLIT_MODE_CALENDAR_ALIGNED, SPLIT_MODE_LEGACY_TICKER_INDEX)
MAX_HORIZON_BY_TIMEFRAME = {
    "1D": 20,
    "1W": 12,
    "1M": 3,
}
TIMEFRAME_ABSOLUTE_MIN_ROWS = {
    "1D": 450,
    "1W": 78,
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


@dataclass(frozen=True)
class CalendarSplitDatePlan:
    split_mode: str
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    purge_gap_trading_days: int
    train_validation_gap_trading_days: int
    validation_test_gap_trading_days: int
    cross_split_date_overlap_count: int

    @property
    def train_start_date(self) -> str | None:
        return self.train_dates[0] if self.train_dates else None

    @property
    def train_end_date(self) -> str | None:
        return self.train_dates[-1] if self.train_dates else None

    @property
    def validation_start_date(self) -> str | None:
        return self.validation_dates[0] if self.validation_dates else None

    @property
    def validation_end_date(self) -> str | None:
        return self.validation_dates[-1] if self.validation_dates else None

    @property
    def test_start_date(self) -> str | None:
        return self.test_dates[0] if self.test_dates else None

    @property
    def test_end_date(self) -> str | None:
        return self.test_dates[-1] if self.test_dates else None

    def to_metadata(self) -> dict[str, int | str | None]:
        return {
            "split_mode": self.split_mode,
            "split_train_start_date": self.train_start_date,
            "split_train_end_date": self.train_end_date,
            "split_validation_start_date": self.validation_start_date,
            "split_validation_end_date": self.validation_end_date,
            "split_test_start_date": self.test_start_date,
            "split_test_end_date": self.test_end_date,
            "purge_gap_trading_days": self.purge_gap_trading_days,
            "split_train_validation_gap_trading_days": self.train_validation_gap_trading_days,
            "split_validation_test_gap_trading_days": self.validation_test_gap_trading_days,
            "split_unique_dates_train": len(self.train_dates),
            "split_unique_dates_validation": len(self.validation_dates),
            "split_unique_dates_test": len(self.test_dates),
            "cross_split_date_overlap_count": self.cross_split_date_overlap_count,
        }


def normalize_split_mode(split_mode: str | None) -> str:
    if split_mode is None:
        return SPLIT_MODE_CALENDAR_ALIGNED
    normalized = str(split_mode).strip().lower().replace("-", "_")
    if normalized in ("calendar", "calendar_aligned", "date_grouped"):
        return SPLIT_MODE_CALENDAR_ALIGNED
    if normalized in ("legacy", "ticker_index", "legacy_ticker_index"):
        return SPLIT_MODE_LEGACY_TICKER_INDEX
    allowed = ", ".join(SPLIT_MODE_CHOICES)
    raise ValueError(f"지원하지 않는 split_mode입니다: {split_mode}. 허용값: {allowed}")


def _metadata_date_strings(metadata: pd.DataFrame, *, date_column: str) -> pd.Series:
    if date_column not in metadata.columns:
        raise ValueError(f"calendar split에 필요한 {date_column} 컬럼이 없습니다.")
    dates = pd.to_datetime(metadata[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"calendar split {date_column} 컬럼에 파싱 불가능한 날짜가 있습니다.")
    return dates.dt.strftime("%Y-%m-%d")


def _split_overlap_count(
    train_dates: set[str],
    validation_dates: set[str],
    test_dates: set[str],
) -> int:
    overlap = (
        (train_dates & validation_dates)
        | (train_dates & test_dates)
        | (validation_dates & test_dates)
    )
    return len(overlap)


def make_calendar_split_date_plan(
    metadata: pd.DataFrame,
    *,
    purge_gap_trading_days: int,
    min_fold_samples: int = 50,
    ratios: tuple[float, float, float] = SPLIT_RATIO,
    date_column: str = "asof_date",
) -> CalendarSplitDatePlan:
    """전체 calendar date를 기준으로 모든 ticker가 같은 split에 들어가도록 날짜 bucket을 만든다."""

    if len(ratios) != 3 or not 0.99 <= sum(ratios) <= 1.01:
        raise ValueError("calendar split ratios는 합이 1인 3개 값이어야 합니다.")
    if metadata.empty:
        raise ValueError("calendar split을 만들 sample metadata가 비어 있습니다.")

    date_series = _metadata_date_strings(metadata, date_column=date_column)
    unique_dates = sorted(date_series.drop_duplicates().tolist())
    purge_gap = max(int(purge_gap_trading_days), 0)
    effective_date_count = len(unique_dates) - (2 * purge_gap)
    if effective_date_count < 3:
        raise ValueError(
            "calendar_split_insufficient_dates: "
            f"unique_dates={len(unique_dates)} purge_gap={purge_gap}"
        )

    train_count = floor(effective_date_count * ratios[0])
    validation_count = floor(effective_date_count * ratios[1])
    test_count = effective_date_count - train_count - validation_count
    if train_count < 1 or validation_count < 1 or test_count < 1:
        raise ValueError(
            "calendar_split_empty_date_bucket: "
            f"train_dates={train_count} validation_dates={validation_count} test_dates={test_count}"
        )

    train_start = 0
    train_end = train_start + train_count
    validation_start = train_end + purge_gap
    validation_end = validation_start + validation_count
    test_start = validation_end + purge_gap
    test_end = test_start + test_count
    if test_end > len(unique_dates):
        raise ValueError(
            "calendar_split_boundary_overflow: "
            f"test_end={test_end} unique_dates={len(unique_dates)}"
        )

    train_dates = tuple(unique_dates[train_start:train_end])
    validation_dates = tuple(unique_dates[validation_start:validation_end])
    test_dates = tuple(unique_dates[test_start:test_end])
    train_set = set(train_dates)
    validation_set = set(validation_dates)
    test_set = set(test_dates)

    train_samples = int(date_series.isin(train_set).sum())
    validation_samples = int(date_series.isin(validation_set).sum())
    test_samples = int(date_series.isin(test_set).sum())
    if (
        train_samples < min_fold_samples
        or validation_samples < min_fold_samples
        or test_samples < min_fold_samples
    ):
        raise ValueError(
            "calendar_split_insufficient_samples: "
            f"train={train_samples} validation={validation_samples} test={test_samples} "
            f"min_fold_samples={min_fold_samples}"
        )

    return CalendarSplitDatePlan(
        split_mode=SPLIT_MODE_CALENDAR_ALIGNED,
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
        purge_gap_trading_days=purge_gap,
        train_validation_gap_trading_days=validation_start - train_end,
        validation_test_gap_trading_days=test_start - validation_end,
        cross_split_date_overlap_count=_split_overlap_count(train_set, validation_set, test_set),
    )


def _resolved_sample_count(row_count: int, seq_len: int, h_max: int) -> int:
    # anchor index 기준 마지막 예측 구간이 완전히 들어가도록 최대 horizon만큼 뒤를 남긴다.
    return max(row_count - seq_len - h_max + 1, 0)


def absolute_min_rows_for_timeframe(timeframe: str) -> int | None:
    return TIMEFRAME_ABSOLUTE_MIN_ROWS.get(str(timeframe).upper())


def required_history_rows(timeframe: str, seq_len: int, h_max: int) -> int:
    absolute_min_rows = absolute_min_rows_for_timeframe(timeframe)
    base_required_rows = seq_len + h_max
    if absolute_min_rows is None:
        return base_required_rows
    return max(base_required_rows, absolute_min_rows)


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
    base_required_rows = seq_len + h_max
    if row_count < base_required_rows:
        raise ValueError("Gate A")

    absolute_min_rows = absolute_min_rows_for_timeframe(timeframe)
    if absolute_min_rows is not None and row_count < absolute_min_rows:
        normalized_timeframe = str(timeframe).upper()
        raise ValueError(f"insufficient_absolute_history_{normalized_timeframe}_min_{absolute_min_rows}")

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
