from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from ai.torch_bootstrap import bootstrap_torch

torch = bootstrap_torch()
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.app.services.feature_svc import FEATURE_COLUMNS as SOURCE_FEATURE_COLUMNS, PRICE_DERIVED_FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, build_price_features, normalize_timeframe, resample_price_frame  # noqa: E402
from backend.collector.repositories.base import fetch_frame  # noqa: E402
from backend.collector.repositories.local_snapshots import local_snapshots_required, read_snapshot_frame  # noqa: E402
from backend.collector.sources.market_data_providers import normalize_provider_name, provider_adjustment_policy  # noqa: E402
from ai.targets import build_target_array, normalize_target_type  # noqa: E402
from ai.splits import (  # noqa: E402
    CalendarSplitDatePlan,
    MAX_HORIZON_BY_TIMEFRAME,
    SPLIT_MODE_CALENDAR_ALIGNED,
    SPLIT_MODE_LEGACY_TICKER_INDEX,
    absolute_min_rows_for_timeframe,
    build_split_specs,
    make_calendar_split_date_plan,
    normalize_split_mode,
    required_history_rows,
)
from ai.ticker_registry import build_registry, lookup_id, registry_path_for_tickers, save_registry  # noqa: E402

SPLIT_RATIO = (0.7, 0.15, 0.15)
SUPPORTED_AI_TIMEFRAMES = ("1D", "1W", "1M")
CACHE_DIR = PROJECT_ROOT / "ai" / "cache"
CACHE_MANIFEST_SCHEMA_VERSION = "feature_cache_manifest_v1"
POSTGRES_TICKER_CHUNK_SIZE = int(os.environ.get("LENS_POSTGRES_TICKER_CHUNK_SIZE", "25"))
CALENDAR_FEATURE_COLUMNS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_month_end",
    "is_quarter_end",
    "is_opex_friday",
]
MODEL_FEATURE_COLUMNS = [*SOURCE_FEATURE_COLUMNS, *CALENDAR_FEATURE_COLUMNS]
FUTURE_CALENDAR_COLUMNS = list(CALENDAR_FEATURE_COLUMNS)
MODEL_N_FEATURES = len(MODEL_FEATURE_COLUMNS)
FUTURE_COVARIATE_DIM = len(FUTURE_CALENDAR_COLUMNS)
INDICATOR_CHECKSUM_COLUMNS = tuple(
    sorted(
        {
            *SOURCE_FEATURE_COLUMNS,
            "atr_ratio",
            "volume",
        }
    )
)
_PREPARED_SPLITS_CACHE: dict[str, Any] = {}
_ENGINE_CACHE: dict[str, Any] = {}
_STALE_CACHE_WARNED: set[str] = set()
_FEATURE_CONTRACT_VERSION = "v3_adjusted_ohlc"
FEATURE_CONTRACT_VERSION = _FEATURE_CONTRACT_VERSION
_FUNDAMENTAL_IMPUTE_COLUMNS = [
    "revenue",
    "net_income",
    "equity",
    "eps",
    "roe",
    "debt_ratio",
]


@dataclass
class SequenceDatasetBundle:
    """학습과 추론에 필요한 시계열 텐서 묶음이다."""

    features: torch.Tensor
    line_targets: torch.Tensor
    band_targets: torch.Tensor
    raw_future_returns: torch.Tensor
    anchor_closes: torch.Tensor
    ticker_ids: torch.Tensor
    future_covariates: torch.Tensor
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, indices: list[int]) -> "SequenceDatasetBundle":
        return SequenceDatasetBundle(
            features=self.features[indices],
            line_targets=self.line_targets[indices],
            band_targets=self.band_targets[indices],
            raw_future_returns=self.raw_future_returns[indices],
            anchor_closes=self.anchor_closes[indices],
            ticker_ids=self.ticker_ids[indices],
            future_covariates=self.future_covariates[indices],
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
        )


class SequenceDataset(Dataset):
    """티커별 원본 배열을 공유하고 샘플 슬라이스만 지연 계산하는 학습 데이터셋."""

    def __init__(
        self,
        *,
        ticker_arrays: dict[str, dict[str, Any]],
        sample_refs: list[tuple[str, int]],
        metadata: pd.DataFrame,
        seq_len: int,
        horizon: int,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        include_future_covariate: bool = True,
        line_target_type: str = "raw_future_return",
        band_target_type: str = "raw_future_return",
    ) -> None:
        self.ticker_arrays = ticker_arrays
        self.sample_refs = sample_refs
        self.metadata = metadata.reset_index(drop=True)
        self.seq_len = seq_len
        self.horizon = horizon
        self.mean = mean
        self.std = std
        self.include_future_covariate = include_future_covariate
        self.line_target_type = normalize_target_type(line_target_type)
        self.band_target_type = normalize_target_type(band_target_type)

    def __len__(self) -> int:
        return len(self.sample_refs)

    def subset(self, indices: list[int]) -> "SequenceDataset":
        return SequenceDataset(
            ticker_arrays=self.ticker_arrays,
            sample_refs=[self.sample_refs[index] for index in indices],
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            seq_len=self.seq_len,
            horizon=self.horizon,
            mean=self.mean,
            std=self.std,
            include_future_covariate=self.include_future_covariate,
            line_target_type=self.line_target_type,
            band_target_type=self.band_target_type,
        )

    def with_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> "SequenceDataset":
        return SequenceDataset(
            ticker_arrays=self.ticker_arrays,
            sample_refs=self.sample_refs,
            metadata=self.metadata.copy(),
            seq_len=self.seq_len,
            horizon=self.horizon,
            mean=mean,
            std=std,
            include_future_covariate=self.include_future_covariate,
            line_target_type=self.line_target_type,
            band_target_type=self.band_target_type,
        )

    def __getitem__(self, index: int):
        ticker, end_idx = self.sample_refs[index]
        ticker_array = self.ticker_arrays[ticker]
        start_idx = end_idx - self.seq_len + 1
        future_start = end_idx + 1
        future_end = future_start + self.horizon

        features = torch.from_numpy(
            ticker_array["features"][start_idx : end_idx + 1]
        ).to(dtype=torch.float32)
        if self.mean is not None and self.std is not None:
            features = (features - self.mean.view(1, -1)) / self.std.view(1, -1)

        anchor_close = float(ticker_array["closes"][end_idx])
        future_returns_np = (ticker_array["closes"][future_start:future_end] / anchor_close) - 1.0
        history_closes = ticker_array["closes"][start_idx : end_idx + 1]
        history_returns = np.diff(history_closes) / np.clip(history_closes[:-1], 1e-6, None)
        line_target = torch.from_numpy(
            build_target_array(
                future_returns_np,
                history_returns=history_returns,
                target_type=self.line_target_type,
            )
        ).to(dtype=torch.float32)
        band_target = torch.from_numpy(
            build_target_array(
                future_returns_np,
                history_returns=history_returns,
                target_type=self.band_target_type,
            )
        ).to(dtype=torch.float32)
        raw_future_returns = torch.from_numpy(
            np.asarray(future_returns_np, dtype="float32")
        ).to(dtype=torch.float32)

        if self.include_future_covariate:
            future_covariates = torch.from_numpy(
                ticker_array["calendar"][future_start:future_end]
            ).to(dtype=torch.float32)
        else:
            future_covariates = torch.empty((self.horizon, 0), dtype=torch.float32)

        return (
            features,
            line_target,
            band_target,
            raw_future_returns,
            torch.tensor(int(ticker_array["ticker_id"]), dtype=torch.long),
            future_covariates,
        )


@dataclass
class DatasetPlan:
    timeframe: str
    seq_len: int
    horizon: int
    h_max: int
    min_fold_samples: int
    absolute_min_rows: int | None
    required_history_rows: int
    provider: str
    source: str
    source_data_hash: str | None
    date_min: str | None
    date_max: str | None
    usable_row_count: int
    estimated_usable_sample_count: int
    input_ticker_count: int
    eligible_tickers: list[str]
    excluded_reasons: dict[str, str]
    split_specs: dict[str, Any]
    ticker_registry_path: str
    num_tickers: int
    split_mode: str = SPLIT_MODE_CALENDAR_ALIGNED
    split_train_start_date: str | None = None
    split_train_end_date: str | None = None
    split_validation_start_date: str | None = None
    split_validation_end_date: str | None = None
    split_test_start_date: str | None = None
    split_test_end_date: str | None = None
    purge_gap_trading_days: int | None = None
    split_train_validation_gap_trading_days: int | None = None
    split_validation_test_gap_trading_days: int | None = None
    split_unique_dates_train: int | None = None
    split_unique_dates_validation: int | None = None
    split_unique_dates_test: int | None = None
    cross_split_date_overlap_count: int | None = None


def _collect_nonfinite_feature_samples(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    max_samples: int = 10,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    counts: dict[str, int] = {}
    samples: list[dict[str, object]] = []
    if frame.empty:
        return counts, samples
    for column in columns:
        if column not in frame.columns:
            continue
        mask = ~np.isfinite(frame[column].to_numpy(dtype="float64", copy=False))
        count = int(mask.sum())
        if count <= 0:
            continue
        counts[column] = count
        if len(samples) >= max_samples:
            continue
        failing_rows = frame.loc[mask, ["ticker", "date", column]].head(max_samples - len(samples))
        for row in failing_rows.itertuples(index=False):
            samples.append(
                {
                    "ticker": str(row.ticker),
                    "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                    "column": column,
                    "value": None if pd.isna(getattr(row, column)) else float(getattr(row, column)),
                }
            )
    return counts, samples


def _enforce_feature_finite_contract(
    frame: pd.DataFrame,
    *,
    context_label: str,
    validate_columns: list[str] | None = None,
) -> pd.DataFrame:
    features = frame.copy()
    if "has_fundamentals" not in features.columns:
        features["has_fundamentals"] = False

    for column in _FUNDAMENTAL_IMPUTE_COLUMNS:
        if column not in features.columns:
            features[column] = np.nan

    nan_counts, samples = _collect_nonfinite_feature_samples(
        features,
        columns=_FUNDAMENTAL_IMPUTE_COLUMNS,
        max_samples=12,
    )
    if nan_counts:
        print(
            json.dumps(
                {
                    "feature_contract": context_label,
                    "stage": "before_impute",
                    "nan_counts": nan_counts,
                    "samples": samples,
                },
                ensure_ascii=False,
            )
        )

    features[_FUNDAMENTAL_IMPUTE_COLUMNS] = features[_FUNDAMENTAL_IMPUTE_COLUMNS].fillna(0.0)
    if "has_fundamentals" in features.columns:
        features["has_fundamentals"] = features["has_fundamentals"].fillna(False).astype(bool)

    if validate_columns is not None:
        remaining_counts, remaining_samples = _collect_nonfinite_feature_samples(
            features,
            columns=validate_columns,
            max_samples=12,
        )
        if remaining_counts:
            raise ValueError(
                json.dumps(
                    {
                        "feature_contract": context_label,
                        "stage": "after_impute",
                        "nonfinite_counts": remaining_counts,
                        "samples": remaining_samples,
                    },
                    ensure_ascii=False,
                )
            )
    return features


def _postgres_dsn() -> str | None:
    host = os.environ.get("SUPABASE_DB_HOST") or os.environ.get("DB_HOST")
    user = os.environ.get("SUPABASE_DB_USER") or os.environ.get("DB_USER")
    password = os.environ.get("SUPABASE_DB_PASSWORD") or os.environ.get("DB_PASSWORD")
    db_name = os.environ.get("SUPABASE_DB_NAME") or os.environ.get("DB_NAME")
    if not all((host, user, password, db_name)):
        return None
    port = os.environ.get("SUPABASE_DB_PORT") or os.environ.get("DB_PORT", "5432")
    sslmode = os.environ.get("DB_SSLMODE", "require")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}?sslmode={sslmode}"


def _postgres_engine():
    dsn = _postgres_dsn()
    if dsn is None:
        raise RuntimeError("Postgres 연결 정보가 없습니다.")
    if dsn not in _ENGINE_CACHE:
        _ENGINE_CACHE[dsn] = create_engine(dsn)
    return _ENGINE_CACHE[dsn]


def _sql_filter_clause(column: str, values: list[str]) -> str:
    escaped_values = []
    for value in values:
        sanitized = value.replace("'", "''")
        escaped_values.append(f"'{sanitized}'")
    escaped = ",".join(escaped_values)
    return f"{column} IN ({escaped})"


def _chunked(values: list[str], size: int) -> list[list[str]]:
    chunk_size = max(int(size), 1)
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _normalized_tickers(
    conn,
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> list[str]:
    if tickers:
        return [ticker.upper() for ticker in tickers]

    if limit_tickers is not None:
        ticker_df = pd.read_sql_query(
            f"SELECT ticker FROM stock_info ORDER BY ticker LIMIT {int(limit_tickers)}",
            conn,
        )
        return ticker_df["ticker"].astype(str).str.upper().tolist()

    indicator_columns = _indicator_columns_via_postgres(conn)
    provider = resolved_market_data_provider(market_data_provider)
    indicator_source_filter = _source_filter_clause(indicator_columns, provider)
    ticker_df = pd.read_sql_query(
        """
            SELECT DISTINCT ticker
            FROM indicators
            WHERE timeframe = %(timeframe)s
              AND {indicator_source_filter}
            ORDER BY ticker
        """.format(indicator_source_filter=indicator_source_filter),
        conn,
        params={"timeframe": timeframe},
    )
    return ticker_df["ticker"].astype(str).str.upper().tolist()


def build_calendar_feature_frame(dates: pd.Series | pd.Index | list[str] | list[pd.Timestamp]) -> pd.DataFrame:
    """날짜만으로 계산 가능한 결정론적 캘린더 피처를 만든다."""
    date_index = pd.to_datetime(pd.Index(dates))
    weekday = date_index.weekday.to_numpy()
    month = date_index.month.to_numpy()
    dow_angle = 2.0 * math.pi * weekday / 5.0
    month_angle = 2.0 * math.pi * month / 12.0

    month_end = pd.DatetimeIndex([timestamp + pd.offsets.BMonthEnd(0) for timestamp in date_index])
    quarter_end = pd.DatetimeIndex([timestamp + pd.offsets.BQuarterEnd(0) for timestamp in date_index])

    def _business_days_until(targets: pd.DatetimeIndex) -> np.ndarray:
        values = []
        for current, target in zip(date_index, targets, strict=False):
            current_day = current.normalize()
            target_day = target.normalize()
            values.append(len(pd.bdate_range(current_day, target_day)) - 1)
        return np.asarray(values, dtype=np.int64)

    month_end_offset = _business_days_until(month_end)
    quarter_end_offset = _business_days_until(quarter_end)

    return pd.DataFrame(
        {
            "day_of_week_sin": np.sin(dow_angle).astype("float32"),
            "day_of_week_cos": np.cos(dow_angle).astype("float32"),
            "month_sin": np.sin(month_angle).astype("float32"),
            "month_cos": np.cos(month_angle).astype("float32"),
            "is_month_end": (month_end_offset <= 4).astype("float32"),
            "is_quarter_end": (quarter_end_offset <= 4).astype("float32"),
            "is_opex_friday": ((weekday == 4) & (date_index.day.to_numpy() >= 15) & (date_index.day.to_numpy() <= 21)).astype("float32"),
        },
        index=date_index,
    )


def append_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    calendar = build_calendar_feature_frame(enriched["date"])
    calendar = calendar.reset_index(drop=True)
    for column in CALENDAR_FEATURE_COLUMNS:
        enriched[column] = calendar[column].to_numpy(dtype="float32")
    return enriched


def _fetch_training_frames_via_postgres(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = _postgres_engine()
    with engine.begin() as conn:
        price_columns = _price_data_columns_via_postgres(conn)
        indicator_columns = _indicator_columns_via_postgres(conn)
        provider = resolved_market_data_provider(market_data_provider)
        price_source_filter = _price_source_filter_clause(price_columns, provider)
        indicator_source_filter = _source_filter_clause(indicator_columns, provider)
        indicator_meta_columns = [
            column for column in ("source", "provider") if column in indicator_columns
        ]
        indicator_meta_select = f", {', '.join(indicator_meta_columns)}" if indicator_meta_columns else ""
        normalized_tickers = _normalized_tickers(
            conn,
            timeframe=timeframe,
            tickers=tickers,
            limit_tickers=limit_tickers,
            market_data_provider=provider,
        )
        feature_frames: list[pd.DataFrame] = []
        price_frames: list[pd.DataFrame] = []

        for ticker_chunk in _chunked(normalized_tickers, POSTGRES_TICKER_CHUNK_SIZE):
            ticker_filter = _sql_filter_clause("ticker", ticker_chunk)
            indicator_query = """
                SELECT ticker, timeframe, date, {feature_columns}{indicator_meta_select}
                FROM indicators
                WHERE timeframe = %(timeframe)s AND {ticker_filter} AND {indicator_source_filter}
                ORDER BY ticker, date
            """.format(
                feature_columns=", ".join(SOURCE_FEATURE_COLUMNS),
                indicator_meta_select=indicator_meta_select,
                ticker_filter=ticker_filter,
                indicator_source_filter=indicator_source_filter,
            )
            price_query = """
                SELECT ticker, date, open, high, low, close, adjusted_close, volume
                FROM price_data
                WHERE {ticker_filter} AND {price_source_filter}
                ORDER BY ticker, date
            """.format(ticker_filter=ticker_filter, price_source_filter=price_source_filter)

            feature_frames.append(pd.read_sql_query(indicator_query, conn, params={"timeframe": timeframe}))
            price_frames.append(pd.read_sql_query(price_query, conn))

    feature_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    price_df = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    return feature_df, price_df


def _fetch_feature_index_frame_via_postgres(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> pd.DataFrame:
    engine = _postgres_engine()
    with engine.begin() as conn:
        price_columns = _price_data_columns_via_postgres(conn)
        indicator_columns = _indicator_columns_via_postgres(conn)
        provider = resolved_market_data_provider(market_data_provider)
        price_source_filter = _price_source_filter_clause(price_columns, provider)
        indicator_source_filter = _source_filter_clause(indicator_columns, provider, column="i.source")
        normalized_tickers = _normalized_tickers(
            conn,
            timeframe=timeframe,
            tickers=tickers,
            limit_tickers=limit_tickers,
            market_data_provider=provider,
        )
        frames: list[pd.DataFrame] = []

        for ticker_chunk in _chunked(normalized_tickers, POSTGRES_TICKER_CHUNK_SIZE):
            query = """
                SELECT i.ticker, i.timeframe, i.date
                FROM indicators i
                INNER JOIN (
                    SELECT DISTINCT ticker, date
                    FROM price_data
                    WHERE {price_ticker_filter} AND {price_source_filter}
                ) p
                    ON p.ticker = i.ticker
                   AND p.date = i.date
                WHERE i.timeframe = %(timeframe)s AND {indicator_ticker_filter} AND {indicator_source_filter}
                ORDER BY i.ticker, i.date
            """.format(
                price_ticker_filter=_sql_filter_clause("ticker", ticker_chunk),
                price_source_filter=price_source_filter,
                indicator_ticker_filter=_sql_filter_clause("i.ticker", ticker_chunk),
                indicator_source_filter=indicator_source_filter,
            )
            frames.append(pd.read_sql_query(query, conn, params={"timeframe": timeframe}))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _filter_frame_by_provider_source(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "source" not in frame.columns:
        return frame.copy() if provider == "eodhd" else frame.iloc[0:0].copy()
    filtered = frame.copy()
    source = filtered["source"].astype("string")
    if provider == "eodhd":
        return filtered[source.isna() | (source.str.lower() == "eodhd")].copy()
    return filtered[source.str.lower() == provider].copy()


def _filter_price_frame_by_provider(price_df: pd.DataFrame, provider: str) -> pd.DataFrame:
    return _filter_frame_by_provider_source(price_df, provider)


def _filter_indicator_frame_by_provider(indicator_df: pd.DataFrame, provider: str) -> pd.DataFrame:
    return _filter_frame_by_provider_source(indicator_df, provider)


def _price_label_frame_from_price_data(price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    prices = price_df.copy()
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date"])
    if normalized_timeframe != "1D":
        if normalized_timeframe == "1W":
            prices["date"] = prices["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
        else:
            prices["date"] = prices["date"].dt.to_period("M").dt.end_time.dt.normalize()
    return (
        prices[["ticker", "date"]]
        .dropna(subset=["date"])
        .drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def audit_feature_price_label_alignment(
    indicator_index: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    timeframe: str,
    provider: str,
) -> dict[str, Any]:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    indicators = indicator_index.copy()
    if indicators.empty:
        return {
            "timeframe": normalized_timeframe,
            "provider": provider,
            "indicator_label_count": 0,
            "price_label_count": 0,
            "joined_label_count": 0,
            "row_loss_count": 0,
        }
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["timeframe"] = indicators["timeframe"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    indicators = indicators[indicators["timeframe"] == normalized_timeframe].dropna(subset=["date"])
    indicators = _filter_indicator_frame_by_provider(indicators, provider)
    price_labels = _price_label_frame_from_price_data(_filter_price_frame_by_provider(price_df, provider), normalized_timeframe)
    joined = indicators.merge(price_labels, on=["ticker", "date"], how="inner")
    return {
        "timeframe": normalized_timeframe,
        "provider": provider,
        "join_basis": "resampled_price_label_date" if normalized_timeframe != "1D" else "daily_price_date",
        "indicator_label_count": int(len(indicators)),
        "price_label_count": int(len(price_labels)),
        "joined_label_count": int(len(joined)),
        "row_loss_count": int(len(indicators) - len(joined)),
        "ticker_count": int(indicators["ticker"].nunique()) if not indicators.empty else 0,
    }


def _source_aware_feature_index_from_frames(
    indicator_index: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    timeframe: str,
    provider: str,
) -> pd.DataFrame:
    if indicator_index.empty or price_df.empty:
        return pd.DataFrame(columns=["ticker", "timeframe", "date"])
    indicators = indicator_index.copy()
    indicators["ticker"] = indicators["ticker"].astype(str).str.upper()
    indicators["timeframe"] = indicators["timeframe"].astype(str).str.upper()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    indicators = indicators[indicators["timeframe"] == normalize_ai_timeframe(timeframe)].dropna(subset=["date"])
    indicators = _filter_indicator_frame_by_provider(indicators, provider)

    prices = _filter_price_frame_by_provider(price_df, provider)
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "timeframe", "date"])
    price_labels = _price_label_frame_from_price_data(prices, normalize_ai_timeframe(timeframe))

    index_frame = indicators.merge(
        price_labels[["ticker", "date"]],
        on=["ticker", "date"],
        how="inner",
    )
    result = (
        index_frame[["ticker", "timeframe", "date"]]
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "timeframe", "date"])
        .reset_index(drop=True)
    )
    result.attrs["alignment_audit"] = audit_feature_price_label_alignment(
        indicator_index,
        price_df,
        timeframe=timeframe,
        provider=provider,
    )
    return result


def _local_snapshot_frame(
    table: str,
    *,
    columns: str = "*",
    filters: list[tuple[str, str, object]] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
    provider: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame | None:
    return read_snapshot_frame(
        table,
        columns=columns,
        filters=filters,
        order_by=order_by,
        limit=limit,
        provider=provider,
        timeframe=timeframe,
    )


def _local_limited_tickers(limit_tickers: int | None) -> list[str] | None:
    if limit_tickers is None:
        return None
    try:
        known = _local_snapshot_frame("stock_info", columns="ticker", order_by="ticker", limit=limit_tickers)
    except FileNotFoundError:
        known = None
    if known is not None and not known.empty and "ticker" in known.columns:
        return known["ticker"].astype(str).str.upper().tolist()
    try:
        price_tickers = _local_snapshot_frame("price_data", columns="ticker", order_by="ticker", limit=None)
    except FileNotFoundError:
        price_tickers = None
    if price_tickers is not None and not price_tickers.empty and "ticker" in price_tickers.columns:
        values = sorted({str(ticker).upper() for ticker in price_tickers["ticker"].tolist() if str(ticker)})
        return values[:limit_tickers]
    if local_snapshots_required():
        raise RuntimeError("로컬 snapshot 모드에서는 limit_tickers 해석을 위해 stock_info 또는 price_data parquet가 필요합니다.")
    return None


def _fetch_feature_index_frame_via_local(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    provider: str,
) -> pd.DataFrame | None:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    normalized_tickers = [ticker.upper() for ticker in tickers] if tickers else None
    if normalized_tickers is None and limit_tickers is not None:
        normalized_tickers = _local_limited_tickers(limit_tickers)

    indicator_filters: list[tuple[str, str, object]] = [("eq", "timeframe", normalized_timeframe)]
    price_filters: list[tuple[str, str, object]] = []
    if normalized_tickers:
        indicator_filters.append(("in", "ticker", normalized_tickers))
        price_filters.append(("in", "ticker", normalized_tickers))

    indicator_index = _local_snapshot_frame(
        "indicators",
        columns="ticker,timeframe,date,source,provider",
        filters=indicator_filters,
        order_by="date",
        provider=provider,
        timeframe=normalized_timeframe,
    )
    price_df = _local_snapshot_frame(
        "price_data",
        columns="ticker,date,source,provider",
        filters=price_filters,
        order_by="date",
        provider=provider,
        timeframe=normalized_timeframe,
    )
    if indicator_index is None or price_df is None:
        return None
    return _source_aware_feature_index_from_frames(
        indicator_index,
        price_df,
        timeframe=normalized_timeframe,
        provider=provider,
    )


def _fetch_training_frames_via_local(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    provider: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    normalized_tickers = [ticker.upper() for ticker in tickers] if tickers else None
    if normalized_tickers is None and limit_tickers is not None:
        normalized_tickers = _local_limited_tickers(limit_tickers)

    indicator_filters: list[tuple[str, str, object]] = [("eq", "timeframe", normalized_timeframe)]
    price_filters: list[tuple[str, str, object]] = []
    if normalized_tickers:
        indicator_filters.append(("in", "ticker", normalized_tickers))
        price_filters.append(("in", "ticker", normalized_tickers))

    indicator_columns = ",".join(["ticker", "timeframe", "date", *SOURCE_FEATURE_COLUMNS, "source", "provider"])
    feature_df = _local_snapshot_frame(
        "indicators",
        columns=indicator_columns,
        filters=indicator_filters,
        order_by="date",
        provider=provider,
        timeframe=normalized_timeframe,
    )
    price_df = _local_snapshot_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume,source,provider,provider_adjustment_policy,updated_at",
        filters=price_filters,
        order_by="date",
        provider=provider,
        timeframe=normalized_timeframe,
    )
    if feature_df is None or price_df is None:
        return None
    return _filter_indicator_frame_by_provider(feature_df, provider), _filter_price_frame_by_provider(price_df, provider)


def fetch_feature_index_frame(
    *,
    timeframe: str = "1D",
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> pd.DataFrame:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    provider = resolved_market_data_provider(market_data_provider)
    data_hash = resolve_data_fingerprint(
        normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    cache_path = resolve_feature_index_cache_path(
        timeframe=normalized_timeframe,
        data_hash=data_hash,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    expected_manifest = build_cache_manifest_payload(
        cache_kind="feature_index",
        timeframe=normalized_timeframe,
        source_data_hash=data_hash,
        feature_columns=["ticker", "timeframe", "date"],
        ticker_count=0,
        date_min=None,
        date_max=None,
        provider=provider,
    )
    _maybe_warn_stale_cache(f"feature_index_{normalized_timeframe}", cache_path)
    if is_cache_manifest_valid(cache_path, expected_manifest):
        return torch.load(cache_path, map_location="cpu", weights_only=False).copy()

    local_index_frame = _fetch_feature_index_frame_via_local(
        timeframe=normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        provider=provider,
    )
    if local_index_frame is not None:
        index_frame = local_index_frame
    else:
        if local_snapshots_required():
            raise RuntimeError("로컬 snapshot 모드에서는 feature index 생성을 위해 price_data와 indicators parquet가 필요합니다.")
        try:
            index_frame = _fetch_feature_index_frame_via_postgres(
                timeframe=normalized_timeframe,
                tickers=tickers,
                limit_tickers=limit_tickers,
                market_data_provider=provider,
            )
        except Exception:
            filters: list[tuple[str, str, object]] = [("eq", "timeframe", normalized_timeframe)]
            price_filters: list[tuple[str, str, object]] = []
            if tickers:
                normalized_tickers = [ticker.upper() for ticker in tickers]
                filters.append(("in", "ticker", normalized_tickers))
                price_filters.append(("in", "ticker", normalized_tickers))
            elif limit_tickers is not None:
                known = fetch_frame("stock_info", columns="ticker", order_by="ticker", limit=limit_tickers)
                normalized_tickers = known["ticker"].astype(str).str.upper().tolist()
                filters.append(("in", "ticker", normalized_tickers))
                price_filters.append(("in", "ticker", normalized_tickers))
            try:
                indicator_index = fetch_frame(
                    "indicators",
                    columns="ticker,timeframe,date,source,provider",
                    filters=filters,
                    order_by="date",
                )
            except Exception:
                indicator_index = fetch_frame(
                    "indicators",
                    columns="ticker,timeframe,date",
                    filters=filters,
                    order_by="date",
                )
            price_df = fetch_price_frame_via_rest(price_filters, provider=provider)
            index_frame = _source_aware_feature_index_from_frames(
                indicator_index,
                price_df,
                timeframe=normalized_timeframe,
                provider=provider,
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index_frame, cache_path)
    date_min, date_max = _frame_date_bounds(index_frame)
    write_cache_manifest(
        cache_path,
        build_cache_manifest_payload(
            cache_kind="feature_index",
            timeframe=normalized_timeframe,
            source_data_hash=data_hash,
            feature_columns=["ticker", "timeframe", "date"],
            ticker_count=int(index_frame["ticker"].nunique()) if not index_frame.empty and "ticker" in index_frame.columns else 0,
            date_min=date_min,
            date_max=date_max,
            provider=provider,
        ),
    )
    return index_frame.copy()


def _repair_price_feature_contract(feature_df: pd.DataFrame, price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if feature_df.empty or price_df.empty:
        return feature_df
    repaired_price_features = build_price_features(price_df=price_df, timeframe=timeframe)
    if repaired_price_features.empty:
        return feature_df

    join_columns = ["ticker", "timeframe", "date"]
    repaired = feature_df.copy()
    repaired["date"] = pd.to_datetime(repaired["date"])
    repaired_price_features = repaired_price_features.copy()
    repaired_price_features["date"] = pd.to_datetime(repaired_price_features["date"])
    repaired = repaired.drop(columns=[column for column in PRICE_DERIVED_FEATURE_COLUMNS if column in repaired.columns])
    repaired = repaired.merge(repaired_price_features[join_columns + PRICE_DERIVED_FEATURE_COLUMNS], on=join_columns, how="left")
    repaired = repaired.dropna(subset=PRICE_DERIVED_FEATURE_COLUMNS).reset_index(drop=True)
    return repaired


def fetch_price_frame_via_rest(
    price_filters: list[tuple[str, str, object]],
    *,
    provider: str,
) -> pd.DataFrame:
    base_columns = "ticker,date,open,high,low,close,adjusted_close,volume"
    local_price_df = _local_snapshot_frame(
        "price_data",
        columns=f"{base_columns},source,provider,provider_adjustment_policy,updated_at",
        filters=price_filters,
        order_by="date",
        provider=provider,
    )
    if local_price_df is not None:
        return _filter_price_frame_by_provider(local_price_df, provider)
    if local_snapshots_required():
        raise RuntimeError("로컬 snapshot 모드에서는 price_data Supabase REST 조회를 실행하지 않습니다.")
    try:
        price_df = fetch_frame(
            "price_data",
            columns=f"{base_columns},source,provider,provider_adjustment_policy,updated_at",
            filters=price_filters,
            order_by="date",
        )
    except Exception:
        price_df = fetch_frame(
            "price_data",
            columns=base_columns,
            filters=price_filters,
            order_by="date",
        )
    return _filter_price_frame_by_provider(price_df, provider)


def fetch_training_frames(
    *,
    timeframe: str = "1D",
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supabase에서 학습용 indicators와 price_data를 읽는다."""
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    provider = resolved_market_data_provider(market_data_provider)
    data_hash = resolve_data_fingerprint(
        normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    cache_path = resolve_feature_cache_path(
        timeframe=normalized_timeframe,
        data_hash=data_hash,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    expected_manifest = build_cache_manifest_payload(
        cache_kind="features",
        timeframe=normalized_timeframe,
        source_data_hash=data_hash,
        feature_columns=list(MODEL_FEATURE_COLUMNS),
        ticker_count=0,
        date_min=None,
        date_max=None,
        provider=provider,
    )
    _maybe_warn_stale_cache(f"features_{normalized_timeframe}", cache_path)
    if is_cache_manifest_valid(cache_path, expected_manifest):
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["feature_df"].copy(), cached["price_df"].copy()

    local_loaded = False
    local_frames = _fetch_training_frames_via_local(
        timeframe=normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        provider=provider,
    )
    if local_frames is not None:
        feature_df, price_df = local_frames
        local_loaded = True
    else:
        if local_snapshots_required():
            raise RuntimeError("로컬 snapshot 모드에서는 feature 생성을 위해 price_data와 indicators parquet가 필요합니다.")
        try:
            feature_df, price_df = _fetch_training_frames_via_postgres(
                timeframe=normalized_timeframe,
                tickers=tickers,
                limit_tickers=limit_tickers,
                market_data_provider=provider,
            )
        except Exception:
            feature_df, price_df = pd.DataFrame(), pd.DataFrame()

    ticker_filters: list[tuple[str, str, object]] = [("eq", "timeframe", normalized_timeframe)]
    price_filters: list[tuple[str, str, object]] = []

    if (feature_df.empty or price_df.empty) and not local_loaded:
        if tickers:
            normalized_tickers = [ticker.upper() for ticker in tickers]
            ticker_filters.append(("in", "ticker", normalized_tickers))
            price_filters.append(("in", "ticker", normalized_tickers))
        elif limit_tickers is not None:
            known = fetch_frame("stock_info", columns="ticker", order_by="ticker", limit=limit_tickers)
            selected = known["ticker"].astype(str).str.upper().tolist()
            ticker_filters.append(("in", "ticker", selected))
            price_filters.append(("in", "ticker", selected))

        indicator_columns = ",".join(["ticker", "timeframe", "date", *SOURCE_FEATURE_COLUMNS])
        try:
            feature_df = fetch_frame(
                "indicators",
                columns=f"{indicator_columns},source,provider",
                filters=ticker_filters,
                order_by="date",
            )
        except Exception:
            feature_df = fetch_frame(
                "indicators",
                columns=indicator_columns,
                filters=ticker_filters,
                order_by="date",
            )
        price_df = fetch_price_frame_via_rest(price_filters, provider=provider)

    feature_df = _filter_indicator_frame_by_provider(feature_df, provider)
    feature_df = _repair_price_feature_contract(feature_df, price_df, normalized_timeframe)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feature_df": feature_df, "price_df": price_df}, cache_path)
    date_min, date_max = _frame_date_bounds(feature_df, price_df)
    write_cache_manifest(
        cache_path,
        build_cache_manifest_payload(
            cache_kind="features",
            timeframe=normalized_timeframe,
            source_data_hash=data_hash,
            feature_columns=list(MODEL_FEATURE_COLUMNS),
            ticker_count=int(feature_df["ticker"].nunique()) if not feature_df.empty and "ticker" in feature_df.columns else 0,
            date_min=date_min,
            date_max=date_max,
            provider=provider,
        ),
    )
    return feature_df, price_df


def default_horizon(timeframe: str) -> int:
    normalized = normalize_ai_timeframe(timeframe)
    return {"1D": 5, "1W": 4, "1M": 3}[normalized]


def normalize_ai_timeframe(timeframe: str) -> str:
    normalized = normalize_timeframe(timeframe)
    if normalized not in SUPPORTED_AI_TIMEFRAMES:
        raise ValueError("월봉 AI 학습과 추론은 Phase 1에서 지원하지 않습니다.")
    return normalized


def active_market_data_provider() -> str:
    return normalize_provider_name(os.environ.get("MARKET_DATA_PROVIDER", "eodhd"))


def resolved_market_data_provider(market_data_provider: str | None = None) -> str:
    return normalize_provider_name(market_data_provider or active_market_data_provider())


def resolved_market_data_source(market_data_provider: str | None = None) -> str:
    return resolved_market_data_provider(market_data_provider)


def _resolved_fingerprint_tickers(tickers: list[str] | None, limit_tickers: int | None) -> list[str] | None:
    if tickers:
        return sorted({ticker.upper() for ticker in tickers})
    if limit_tickers is None:
        return None
    local_tickers = _local_limited_tickers(limit_tickers)
    if local_tickers is not None:
        return sorted({str(ticker).upper() for ticker in local_tickers if str(ticker)})
    try:
        known = fetch_frame("stock_info", columns="ticker", order_by="ticker", limit=limit_tickers)
    except Exception:
        return None
    if known.empty or "ticker" not in known.columns:
        return []
    return sorted({str(ticker).upper() for ticker in known["ticker"].tolist() if str(ticker)})


def _ticker_universe_fingerprint(tickers: list[str] | None) -> str:
    payload = ["__all__"] if tickers is None else sorted({ticker.upper() for ticker in tickers})
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _table_columns_via_postgres(conn, table_name: str) -> set[str]:
    columns = pd.read_sql_query(
        """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %(table_name)s
        """,
        conn,
        params={"table_name": table_name},
    )
    return {str(column) for column in columns["column_name"].tolist()}


def _price_data_columns_via_postgres(conn) -> set[str]:
    return _table_columns_via_postgres(conn, "price_data")


def _indicator_columns_via_postgres(conn) -> set[str]:
    return _table_columns_via_postgres(conn, "indicators")


def _source_filter_clause(columns: set[str], provider: str, *, column: str = "source") -> str:
    if "source" not in columns:
        return "TRUE"
    sanitized = provider.replace("'", "''")
    if provider == "eodhd":
        return f"({column} = '{sanitized}' OR {column} IS NULL)"
    return f"{column} = '{sanitized}'"


def _price_source_filter_clause(columns: set[str], provider: str) -> str:
    return _source_filter_clause(columns, provider)


def _price_meta_query(
    *,
    columns: set[str],
    provider: str,
    tickers: list[str] | None,
) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {"provider": provider}
    filters = ["TRUE"]
    filters.append(_price_source_filter_clause(columns, provider))
    if tickers:
        filters.append(_sql_filter_clause("ticker", tickers))

    updated_expr = "MAX(updated_at)" if "updated_at" in columns else "MAX(created_at)" if "created_at" in columns else "NULL"
    query = """
        SELECT
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(*) AS row_count,
            COUNT(DISTINCT ticker) AS ticker_count,
            {updated_expr} AS max_updated_at,
            MD5(
                CONCAT(
                    COUNT(*), '|',
                    COALESCE(MIN(date)::text, ''), '|',
                    COALESCE(MAX(date)::text, ''), '|',
                    COALESCE(SUM(ROUND((
                        COALESCE(open, 0) +
                        COALESCE(high, 0) +
                        COALESCE(low, 0) +
                        COALESCE(close, 0) +
                        COALESCE(adjusted_close, 0)
                    )::numeric * 100000)), 0), '|',
                    COALESCE(SUM(COALESCE(volume, 0)::numeric), 0)
                )
            ) AS data_checksum
        FROM price_data
        WHERE {filters}
    """.format(updated_expr=updated_expr, filters=" AND ".join(filters))
    return query, params


def _price_meta_from_frame(frame: pd.DataFrame, provider: str) -> dict[str, Any]:
    frame = _filter_price_frame_by_provider(frame, provider)
    if frame.empty:
        return {
            "min_date": None,
            "max_date": None,
            "row_count": 0,
            "ticker_count": 0,
            "max_updated_at": None,
            "data_checksum": None,
            "source_column_present": False,
            "updated_at_column_present": False,
        }
    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    checksum_payload = {
        "row_count": int(len(frame)),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "price_sum": float(numeric[["open", "high", "low", "close", "adjusted_close"]].sum().sum()),
        "volume_sum": float(numeric["volume"].sum()),
    }
    return {
        "min_date": str(frame["date"].min()),
        "max_date": str(frame["date"].max()),
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "max_updated_at": (
            str(frame["updated_at"].max())
            if "updated_at" in frame.columns
            else str(frame["created_at"].max()) if "created_at" in frame.columns else None
        ),
        "data_checksum": hashlib.sha256(
            json.dumps(checksum_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        "source_column_present": "source" in frame.columns,
        "updated_at_column_present": "updated_at" in frame.columns,
        "provider_filter": provider,
    }


def _price_meta_via_local(provider: str, tickers: list[str] | None, timeframe: str | None = None) -> dict[str, Any] | None:
    filters: list[tuple[str, str, object]] = []
    if tickers:
        filters.append(("in", "ticker", tickers))
    frame = _local_snapshot_frame(
        "price_data",
        columns="ticker,date,open,high,low,close,adjusted_close,volume,created_at,source,provider,provider_adjustment_policy,updated_at",
        filters=filters,
        order_by="date",
        provider=provider,
        timeframe=timeframe,
    )
    if frame is None:
        return None
    return _price_meta_from_frame(frame, provider)


def _price_meta_via_rest(provider: str, tickers: list[str] | None) -> dict[str, Any]:
    filters: list[tuple[str, str, object]] = []
    if tickers:
        filters.append(("in", "ticker", tickers))
    try:
        frame = fetch_frame(
            "price_data",
            columns="ticker,date,open,high,low,close,adjusted_close,volume,created_at,source,provider,provider_adjustment_policy,updated_at",
            filters=filters,
            order_by="date",
        )
    except Exception:
        frame = fetch_frame(
            "price_data",
            columns="ticker,date,open,high,low,close,adjusted_close,volume,created_at",
            filters=filters,
            order_by="date",
        )
    return _price_meta_from_frame(frame, provider)


def _indicator_checksum_columns(columns: set[str] | None = None) -> list[str]:
    if columns is None:
        return list(INDICATOR_CHECKSUM_COLUMNS)
    return [column for column in INDICATOR_CHECKSUM_COLUMNS if column in columns]


def _indicator_meta_query(
    *,
    columns: set[str],
    provider: str,
    timeframe: str,
    tickers: list[str] | None,
) -> tuple[str, dict[str, Any]]:
    filters = ["timeframe = %(timeframe)s", _source_filter_clause(columns, provider)]
    if tickers:
        filters.append(_sql_filter_clause("ticker", tickers))
    checksum_terms = [
        f"COALESCE({column}::text, '')"
        for column in _indicator_checksum_columns(columns)
    ]
    checksum_expr = ", ".join(["ticker", "date::text", *checksum_terms])
    query = """
        WITH row_hashes AS (
            SELECT
                ticker,
                date,
                MD5(CONCAT_WS('|', {checksum_expr})) AS row_hash
            FROM indicators
            WHERE {filters}
        ),
        ticker_hashes AS (
            SELECT
                ticker,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(*) AS row_count,
                MD5(COALESCE(STRING_AGG(row_hash, '' ORDER BY date), '')) AS ticker_checksum
            FROM row_hashes
            GROUP BY ticker
        )
        SELECT
            MIN(min_date) AS min_date,
            MAX(max_date) AS max_date,
            COALESCE(SUM(row_count), 0) AS row_count,
            COUNT(*) AS ticker_count,
            MD5(COALESCE(STRING_AGG(ticker || ':' || ticker_checksum, '|' ORDER BY ticker), '')) AS value_checksum
        FROM ticker_hashes
    """.format(checksum_expr=checksum_expr, filters=" AND ".join(filters))
    return query, {"timeframe": timeframe}


def _indicator_meta_from_frame(frame: pd.DataFrame, provider: str) -> dict[str, Any]:
    frame = _filter_indicator_frame_by_provider(frame, provider)
    if frame.empty:
        return {
            "min_date": None,
            "max_date": None,
            "row_count": 0,
            "ticker_count": 0,
            "value_checksum": None,
        }
    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sort_columns = [column for column in ("ticker", "date") if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns)
    checksum_columns = [column for column in _indicator_checksum_columns(set(working.columns)) if column in working.columns]
    row_payloads: list[str] = []
    for _, row in working.iterrows():
        payload = {
            "ticker": str(row.get("ticker", "")),
            "date": str(row.get("date", "")),
            "values": {
                column: None if pd.isna(row.get(column)) else str(row.get(column))
                for column in checksum_columns
            },
        }
        row_payloads.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    checksum_payload = {
        "provider": provider,
        "row_count": int(len(working)),
        "date_min": str(working["date"].min()) if "date" in working.columns else None,
        "date_max": str(working["date"].max()) if "date" in working.columns else None,
        "rows": row_payloads,
    }
    return {
        "min_date": str(working["date"].min()) if "date" in working.columns else None,
        "max_date": str(working["date"].max()) if "date" in working.columns else None,
        "row_count": int(len(working)),
        "ticker_count": int(working["ticker"].nunique()) if "ticker" in working.columns else None,
        "value_checksum": hashlib.sha256(
            json.dumps(checksum_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
    }


def _indicator_meta_via_local(provider: str, timeframe: str, tickers: list[str] | None) -> dict[str, Any] | None:
    filters: list[tuple[str, str, object]] = [("eq", "timeframe", timeframe)]
    if tickers:
        filters.append(("in", "ticker", tickers))
    columns = ",".join(["ticker", "date", "timeframe", "source", "provider", *INDICATOR_CHECKSUM_COLUMNS])
    frame = _local_snapshot_frame(
        "indicators",
        columns=columns,
        filters=filters,
        order_by="date",
        provider=provider,
        timeframe=timeframe,
    )
    if frame is None:
        return None
    return _indicator_meta_from_frame(frame, provider)


def _indicator_meta_via_rest(provider: str, timeframe: str, tickers: list[str] | None) -> dict[str, Any]:
    filters: list[tuple[str, str, object]] = [("eq", "timeframe", timeframe)]
    if tickers:
        filters.append(("in", "ticker", tickers))
    columns = ",".join(["ticker", "date", "timeframe", "source", "provider", *INDICATOR_CHECKSUM_COLUMNS])
    try:
        frame = fetch_frame(
            "indicators",
            columns=columns,
            filters=filters,
            order_by="date",
        )
    except Exception:
        try:
            frame = fetch_frame(
                "indicators",
                columns="ticker,date,timeframe,source,provider",
                filters=filters,
                order_by="date",
            )
        except Exception:
            frame = fetch_frame(
                "indicators",
                columns="date",
                filters=[("eq", "timeframe", timeframe)],
                order_by="date",
            )
    return _indicator_meta_from_frame(frame, provider)


def _meta_value(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)
    try:
        return meta[key]
    except Exception:
        return default


def _meta_int(meta: Any, key: str) -> int | None:
    value = _meta_value(meta, key)
    if value is None or pd.isna(value):
        return None
    return int(value)


def _apply_fingerprint_meta_payload(
    fingerprint_payload: dict[str, Any],
    *,
    price_meta: Any,
    indicator_meta: Any,
    source_column_present: bool | None = None,
    updated_at_column_present: bool | None = None,
) -> None:
    fingerprint_payload["date_range"]["price_min_date"] = None if pd.isna(_meta_value(price_meta, "min_date")) else str(_meta_value(price_meta, "min_date"))
    fingerprint_payload["date_range"]["price_max_date"] = None if pd.isna(_meta_value(price_meta, "max_date")) else str(_meta_value(price_meta, "max_date"))
    fingerprint_payload["date_range"]["indicator_min_date"] = None if pd.isna(_meta_value(indicator_meta, "min_date")) else str(_meta_value(indicator_meta, "min_date"))
    fingerprint_payload["date_range"]["indicator_max_date"] = None if pd.isna(_meta_value(indicator_meta, "max_date")) else str(_meta_value(indicator_meta, "max_date"))
    fingerprint_payload["price_row_count"] = _meta_int(price_meta, "row_count")
    fingerprint_payload["price_ticker_count"] = _meta_int(price_meta, "ticker_count")
    fingerprint_payload["price_max_updated_at"] = None if pd.isna(_meta_value(price_meta, "max_updated_at")) else str(_meta_value(price_meta, "max_updated_at"))
    fingerprint_payload["price_data_checksum"] = None if pd.isna(_meta_value(price_meta, "data_checksum")) else str(_meta_value(price_meta, "data_checksum"))
    fingerprint_payload["indicator_count"] = _meta_int(indicator_meta, "row_count")
    fingerprint_payload["indicator_ticker_count"] = _meta_int(indicator_meta, "ticker_count")
    fingerprint_payload["indicator_value_checksum"] = None if pd.isna(_meta_value(indicator_meta, "value_checksum")) else str(_meta_value(indicator_meta, "value_checksum"))
    fingerprint_payload["source_column_present"] = _meta_value(price_meta, "source_column_present", source_column_present)
    fingerprint_payload["updated_at_column_present"] = _meta_value(price_meta, "updated_at_column_present", updated_at_column_present)


def resolve_data_fingerprint(
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> str:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    provider = normalize_provider_name(market_data_provider or active_market_data_provider())
    fingerprint_tickers = _resolved_fingerprint_tickers(tickers, limit_tickers)
    fingerprint_payload: dict[str, Any] = {
        "timeframe": normalized_timeframe,
        "market_data_provider": provider,
        "provider_adjustment_policy": provider_adjustment_policy(provider),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "ticker_universe_fingerprint": _ticker_universe_fingerprint(fingerprint_tickers),
        "ticker_universe_count": None if fingerprint_tickers is None else len(fingerprint_tickers),
        "ticker_universe_scope": "all" if fingerprint_tickers is None else "explicit_or_limited",
        "date_range": {"price_min_date": None, "price_max_date": None, "indicator_min_date": None, "indicator_max_date": None},
        "price_row_count": None,
        "price_ticker_count": None,
        "price_max_updated_at": None,
        "price_data_checksum": None,
        "indicator_count": None,
        "indicator_ticker_count": None,
        "indicator_value_checksum": None,
        "source_column_present": None,
        "updated_at_column_present": None,
    }

    local_price_meta = _price_meta_via_local(provider, fingerprint_tickers, normalized_timeframe)
    local_indicator_meta = _indicator_meta_via_local(provider, normalized_timeframe, fingerprint_tickers)
    if local_price_meta is not None and local_indicator_meta is not None:
        _apply_fingerprint_meta_payload(
            fingerprint_payload,
            price_meta=local_price_meta,
            indicator_meta=local_indicator_meta,
        )
        return hashlib.sha256(
            json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
    if local_snapshots_required():
        raise RuntimeError("로컬 snapshot 모드에서는 source_data_hash 계산을 위해 price_data와 indicators parquet가 필요합니다.")

    try:
        engine = _postgres_engine()
        with engine.begin() as conn:
            price_columns = _price_data_columns_via_postgres(conn)
            indicator_columns = _indicator_columns_via_postgres(conn)
            indicator_source_filter = _source_filter_clause(indicator_columns, provider)
            price_query, price_params = _price_meta_query(
                columns=price_columns,
                provider=provider,
                tickers=fingerprint_tickers,
            )
            price_meta = pd.read_sql_query(
                price_query,
                conn,
                params=price_params,
            ).iloc[0]
            indicator_query, indicator_params = _indicator_meta_query(
                columns=indicator_columns,
                provider=provider,
                timeframe=normalized_timeframe,
                tickers=fingerprint_tickers,
            )
            indicator_meta = pd.read_sql_query(
                indicator_query,
                conn,
                params=indicator_params,
            ).iloc[0]
        _apply_fingerprint_meta_payload(
            fingerprint_payload,
            price_meta=price_meta,
            indicator_meta=indicator_meta,
            source_column_present="source" in price_columns,
            updated_at_column_present="updated_at" in price_columns,
        )
    except Exception:
        indicator_meta = _indicator_meta_via_rest(provider, normalized_timeframe, fingerprint_tickers)
        price_meta = _price_meta_via_rest(provider, fingerprint_tickers)
        _apply_fingerprint_meta_payload(
            fingerprint_payload,
            price_meta=price_meta,
            indicator_meta=indicator_meta,
        )

    return hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]


def resolve_cache_manifest_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.manifest.json")


def _frame_date_bounds(*frames: pd.DataFrame) -> tuple[str | None, str | None]:
    values: list[pd.Series] = []
    for frame in frames:
        if frame is not None and not frame.empty and "date" in frame.columns:
            values.append(pd.to_datetime(frame["date"], errors="coerce"))
    if not values:
        return None, None
    combined = pd.concat(values).dropna()
    if combined.empty:
        return None, None
    return str(combined.min().date()), str(combined.max().date())


def build_cache_manifest_payload(
    *,
    cache_kind: str,
    timeframe: str,
    source_data_hash: str,
    feature_columns: list[str],
    ticker_count: int,
    date_min: str | None,
    date_max: str | None,
    provider: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    resolved_provider = resolved_market_data_provider(provider)
    resolved_source = normalize_provider_name(source or resolved_provider)
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "cache_kind": cache_kind,
        "provider": resolved_provider,
        "source": resolved_source,
        "provider_adjustment_policy": provider_adjustment_policy(resolved_provider),
        "source_data_hash": source_data_hash,
        "feature_version": FEATURE_CONTRACT_VERSION,
        "feature_columns": list(feature_columns),
        "ticker_count": int(ticker_count),
        "timeframe": normalize_ai_timeframe(timeframe),
        "date_min": date_min,
        "date_max": date_max,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }


def write_cache_manifest(cache_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path = resolve_cache_manifest_path(cache_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def is_cache_manifest_valid(cache_path: Path, expected_manifest: dict[str, Any]) -> bool:
    manifest_path = resolve_cache_manifest_path(cache_path)
    if not cache_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    comparable_keys = [
        "schema_version",
        "cache_kind",
        "provider",
        "source",
        "provider_adjustment_policy",
        "source_data_hash",
        "feature_version",
        "feature_columns",
        "timeframe",
    ]
    return all(manifest.get(key) == expected_manifest.get(key) for key in comparable_keys)


def _maybe_warn_stale_cache(prefix: str, path: Path) -> None:
    if path.exists() or prefix in _STALE_CACHE_WARNED:
        return
    stale_matches = list(CACHE_DIR.glob(f"{prefix}_*.pt"))
    if stale_matches:
        print(f"기존 캐시 형식이 변경되어 {prefix} 캐시를 새로 생성합니다.")
        _STALE_CACHE_WARNED.add(prefix)


def resolve_feature_cache_path(
    *,
    timeframe: str,
    data_hash: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> Path:
    provider = resolved_market_data_provider(market_data_provider)
    payload = {
        "timeframe": timeframe,
        "source_feature_columns": SOURCE_FEATURE_COLUMNS,
        "calendar_feature_columns": CALENDAR_FEATURE_COLUMNS,
        "feature_contract_version": _FEATURE_CONTRACT_VERSION,
        "market_data_provider": provider,
        "market_data_source": resolved_market_data_source(provider),
        "provider_adjustment_policy": provider_adjustment_policy(provider),
        "tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"features_{timeframe}_{digest}_{data_hash}.pt"


def resolve_feature_index_cache_path(
    *,
    timeframe: str,
    data_hash: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    market_data_provider: str | None = None,
) -> Path:
    provider = resolved_market_data_provider(market_data_provider)
    payload = {
        "timeframe": timeframe,
        "tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
        "kind": "feature_index_v1",
        "feature_contract_version": _FEATURE_CONTRACT_VERSION,
        "market_data_provider": provider,
        "market_data_source": resolved_market_data_source(provider),
        "provider_adjustment_policy": provider_adjustment_policy(provider),
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"feature_index_{timeframe}_{digest}_{data_hash}.pt"


def resolve_prepared_splits_cache_key(
    *,
    timeframe: str,
    seq_len: int,
    horizon: int,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    min_fold_samples: int = 50,
    include_future_covariate: bool = True,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
    ticker_registry: dict[str, Any] | None = None,
    market_data_provider: str | None = None,
    split_mode: str = SPLIT_MODE_CALENDAR_ALIGNED,
) -> str:
    provider = resolved_market_data_provider(market_data_provider)
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    normalized_split_mode = normalize_split_mode(split_mode)
    data_hash = resolve_data_fingerprint(
        normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    registry_payload = None
    if ticker_registry is not None:
        registry_payload = {
            "timeframe": str(ticker_registry.get("timeframe", "")).upper(),
            "num_tickers": int(ticker_registry.get("num_tickers", 0)),
            "mapping": {
                str(ticker).upper(): int(value)
                for ticker, value in sorted((ticker_registry.get("mapping") or {}).items())
            },
        }
    payload = {
        "timeframe": normalized_timeframe,
        "data_hash": data_hash,
        "seq_len": seq_len,
        "horizon": horizon,
        "tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
        "min_fold_samples": min_fold_samples,
        "absolute_min_rows": absolute_min_rows_for_timeframe(normalized_timeframe),
        "required_history_rows": required_history_rows(normalized_timeframe, seq_len, MAX_HORIZON_BY_TIMEFRAME[normalized_timeframe]),
        "include_future_covariate": include_future_covariate,
        "line_target_type": normalize_target_type(line_target_type),
        "band_target_type": normalize_target_type(band_target_type),
        "split_mode": normalized_split_mode,
        "source_feature_columns": SOURCE_FEATURE_COLUMNS,
        "calendar_feature_columns": CALENDAR_FEATURE_COLUMNS,
        "feature_contract_version": _FEATURE_CONTRACT_VERSION,
        "market_data_provider": provider,
        "market_data_source": resolved_market_data_source(provider),
        "provider_adjustment_policy": provider_adjustment_policy(provider),
        "ticker_registry": registry_payload,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _build_target_frame(price_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = price_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "adjusted_close" not in frame.columns:
        frame["adjusted_close"] = frame["close"]
    frame["adjusted_close"] = frame["adjusted_close"].fillna(frame["close"])
    resampled = resample_price_frame(frame, timeframe)
    resampled["target_close"] = resampled["adjusted_close"].fillna(resampled["close"])
    return resampled[["ticker", "date", "target_close"]].copy()


def build_sequence_dataset(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    timeframe: str = "1D",
    seq_len: int = 60,
    horizon: int | None = None,
    ticker_registry: dict[str, Any] | None = None,
    include_future_covariate: bool = True,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
) -> SequenceDatasetBundle:
    """feature와 price를 묶어 multi-step 학습 샘플을 만든다."""

    normalized_timeframe = normalize_ai_timeframe(timeframe)
    resolved_horizon = horizon or default_horizon(normalized_timeframe)
    resolved_line_target_type = normalize_target_type(line_target_type)
    resolved_band_target_type = normalize_target_type(band_target_type)
    if feature_df.empty or price_df.empty:
        raise ValueError("학습 데이터가 비어 있습니다. indicators와 price_data를 먼저 확인하세요.")

    features = feature_df.copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["ticker", "date"])
    features = _enforce_feature_finite_contract(
        features,
        context_label=f"build_sequence_dataset:{normalized_timeframe}",
        validate_columns=None,
    )
    features = features.dropna(subset=REQUIRED_FEATURE_COLUMNS)
    features = append_calendar_features(features)
    features = _enforce_feature_finite_contract(
        features,
        context_label=f"build_sequence_dataset:{normalized_timeframe}:calendar",
        validate_columns=MODEL_FEATURE_COLUMNS,
    )

    targets = _build_target_frame(price_df, normalized_timeframe)
    merged = features.merge(targets, on=["ticker", "date"], how="inner")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged = merged.dropna(subset=["target_close"])

    ticker_frames: list[tuple[str, pd.DataFrame, int]] = []
    total_samples = 0
    metadata_rows: list[dict[str, object]] = []

    for ticker, ticker_frame in merged.groupby("ticker", sort=True):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        if len(ticker_frame) < (seq_len + resolved_horizon):
            continue

        closes = ticker_frame["target_close"].to_numpy(dtype="float32")
        valid_anchor_count = int(
            np.count_nonzero(closes[seq_len - 1 : len(ticker_frame) - resolved_horizon] != 0.0)
        )
        if valid_anchor_count <= 0:
            continue
        ticker_frames.append((str(ticker), ticker_frame, valid_anchor_count))
        total_samples += valid_anchor_count

    if total_samples <= 0:
        raise ValueError("지정한 조건에서 시퀀스 샘플을 만들지 못했습니다.")

    feature_array = np.empty((total_samples, seq_len, MODEL_N_FEATURES), dtype=np.float32)
    line_target_array = np.empty((total_samples, resolved_horizon), dtype=np.float32)
    band_target_array = np.empty((total_samples, resolved_horizon), dtype=np.float32)
    raw_future_return_array = np.empty((total_samples, resolved_horizon), dtype=np.float32)
    anchor_close_array = np.empty((total_samples,), dtype=np.float32)
    ticker_id_array = np.empty((total_samples,), dtype=np.int64)
    future_cov_dim = FUTURE_COVARIATE_DIM if include_future_covariate else 0
    future_covariate_array = np.empty((total_samples, resolved_horizon, future_cov_dim), dtype=np.float32)

    sample_cursor = 0
    for ticker, ticker_frame, _ in ticker_frames:
        feature_values = ticker_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")
        closes = ticker_frame["target_close"].to_numpy(dtype="float32")
        date_strings = pd.to_datetime(ticker_frame["date"]).dt.strftime("%Y-%m-%d").tolist()
        calendar_values = (
            ticker_frame[FUTURE_CALENDAR_COLUMNS].to_numpy(dtype="float32")
            if include_future_covariate
            else None
        )
        sample_index = 0

        for end_idx in range(seq_len - 1, len(ticker_frame) - resolved_horizon):
            anchor_close = float(closes[end_idx])
            if anchor_close == 0.0:
                continue

            start_idx = end_idx - seq_len + 1
            future_start = end_idx + 1
            future_end = future_start + resolved_horizon
            future_returns = (closes[future_start:future_end] / anchor_close) - 1.0
            history_closes = closes[start_idx : end_idx + 1]
            history_returns = np.diff(history_closes) / np.clip(history_closes[:-1], 1e-6, None)

            feature_array[sample_cursor] = feature_values[start_idx : end_idx + 1]
            line_target_array[sample_cursor] = build_target_array(
                future_returns,
                history_returns=history_returns,
                target_type=resolved_line_target_type,
            )
            band_target_array[sample_cursor] = build_target_array(
                future_returns,
                history_returns=history_returns,
                target_type=resolved_band_target_type,
            )
            raw_future_return_array[sample_cursor] = future_returns
            anchor_close_array[sample_cursor] = anchor_close
            ticker_id_array[sample_cursor] = lookup_id(ticker, ticker_registry) if ticker_registry is not None else 0
            if include_future_covariate and calendar_values is not None:
                future_covariate_array[sample_cursor] = calendar_values[future_start:future_end]
            metadata_rows.append(
                {
                    "ticker": ticker,
                    "timeframe": normalized_timeframe,
                    "asof_date": date_strings[end_idx],
                    "forecast_dates": date_strings[future_start:future_end],
                    "sample_index": sample_index,
                }
            )
            sample_index += 1
            sample_cursor += 1

    if sample_cursor != total_samples:
        feature_array = feature_array[:sample_cursor]
        line_target_array = line_target_array[:sample_cursor]
        band_target_array = band_target_array[:sample_cursor]
        raw_future_return_array = raw_future_return_array[:sample_cursor]
        anchor_close_array = anchor_close_array[:sample_cursor]
        ticker_id_array = ticker_id_array[:sample_cursor]
        future_covariate_array = future_covariate_array[:sample_cursor]

    return SequenceDatasetBundle(
        features=torch.from_numpy(feature_array),
        line_targets=torch.from_numpy(line_target_array),
        band_targets=torch.from_numpy(band_target_array),
        raw_future_returns=torch.from_numpy(raw_future_return_array),
        anchor_closes=torch.from_numpy(anchor_close_array),
        ticker_ids=torch.from_numpy(ticker_id_array),
        future_covariates=torch.from_numpy(future_covariate_array),
        metadata=pd.DataFrame(metadata_rows),
    )


def build_lazy_sequence_dataset(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    timeframe: str = "1D",
    seq_len: int = 60,
    horizon: int | None = None,
    ticker_registry: dict[str, Any] | None = None,
    include_future_covariate: bool = True,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
) -> SequenceDataset:
    """티커별 원본 배열을 공유하는 지연 로딩 시퀀스 데이터셋을 만든다."""

    normalized_timeframe = normalize_ai_timeframe(timeframe)
    resolved_horizon = horizon or default_horizon(normalized_timeframe)
    resolved_line_target_type = normalize_target_type(line_target_type)
    resolved_band_target_type = normalize_target_type(band_target_type)
    if feature_df.empty or price_df.empty:
        raise ValueError("학습 데이터가 비어 있습니다. indicators와 price_data를 먼저 확인하세요.")

    features = feature_df.copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["ticker", "date"])
    features = _enforce_feature_finite_contract(
        features,
        context_label=f"build_lazy_sequence_dataset:{normalized_timeframe}",
        validate_columns=None,
    )
    features = features.dropna(subset=REQUIRED_FEATURE_COLUMNS)
    features = append_calendar_features(features)
    features = _enforce_feature_finite_contract(
        features,
        context_label=f"build_lazy_sequence_dataset:{normalized_timeframe}:calendar",
        validate_columns=MODEL_FEATURE_COLUMNS,
    )

    targets = _build_target_frame(price_df, normalized_timeframe)
    merged = features.merge(targets, on=["ticker", "date"], how="inner")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged = merged.dropna(subset=["target_close"])

    ticker_arrays: dict[str, dict[str, Any]] = {}
    sample_refs: list[tuple[str, int]] = []
    metadata_rows: list[dict[str, object]] = []

    for ticker, ticker_frame in merged.groupby("ticker", sort=True):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        if len(ticker_frame) < (seq_len + resolved_horizon):
            continue

        ticker_key = str(ticker)
        feature_values = ticker_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")
        closes = ticker_frame["target_close"].to_numpy(dtype="float32")
        dates = pd.to_datetime(ticker_frame["date"]).to_numpy()
        date_strings = pd.to_datetime(ticker_frame["date"]).dt.strftime("%Y-%m-%d").tolist()
        ticker_arrays[ticker_key] = {
            "features": feature_values,
            "closes": closes,
            "dates": dates,
            "calendar": (
                ticker_frame[FUTURE_CALENDAR_COLUMNS].to_numpy(dtype="float32")
                if include_future_covariate
                else None
            ),
            "ticker_id": lookup_id(ticker_key, ticker_registry) if ticker_registry is not None else 0,
        }

        sample_index = 0
        for end_idx in range(seq_len - 1, len(ticker_frame) - resolved_horizon):
            if float(closes[end_idx]) == 0.0:
                continue
            future_start = end_idx + 1
            future_end = future_start + resolved_horizon
            sample_refs.append((ticker_key, end_idx))
            metadata_rows.append(
                {
                    "ticker": ticker_key,
                    "timeframe": normalized_timeframe,
                    "asof_date": date_strings[end_idx],
                    "forecast_dates": date_strings[future_start:future_end],
                    "sample_index": sample_index,
                }
            )
            sample_index += 1

    if not sample_refs:
        raise ValueError("지정한 조건에서 시계열 샘플을 만들지 못했습니다.")

    return SequenceDataset(
        ticker_arrays=ticker_arrays,
        sample_refs=sample_refs,
        metadata=pd.DataFrame(metadata_rows),
        seq_len=seq_len,
        horizon=resolved_horizon,
        include_future_covariate=include_future_covariate,
        line_target_type=resolved_line_target_type,
        band_target_type=resolved_band_target_type,
    )


def build_dataset_plan(
    feature_df: pd.DataFrame,
    *,
    timeframe: str,
    seq_len: int,
    horizon: int,
    min_fold_samples: int = 50,
    ticker_registry: dict[str, Any] | None = None,
    ticker_registry_path: str | None = None,
    market_data_provider: str | None = None,
    source_data_hash: str | None = None,
    split_mode: str = SPLIT_MODE_CALENDAR_ALIGNED,
) -> DatasetPlan:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    provider = resolved_market_data_provider(market_data_provider)
    normalized_split_mode = normalize_split_mode(split_mode)
    filtered = feature_df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered[filtered["timeframe"] == normalized_timeframe].sort_values(["ticker", "date"])
    h_max = MAX_HORIZON_BY_TIMEFRAME[normalized_timeframe]
    absolute_min_rows = absolute_min_rows_for_timeframe(normalized_timeframe)
    required_rows = required_history_rows(normalized_timeframe, seq_len, h_max)
    date_min, date_max = _frame_date_bounds(filtered)
    estimated_sample_count = 0
    for _, ticker_frame in filtered.groupby("ticker", sort=True):
        estimated_sample_count += max(len(ticker_frame) - seq_len - h_max + 1, 0)
    split_specs, excluded_reasons = build_split_specs(
        filtered,
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        h_max=h_max,
        min_fold_samples=min_fold_samples,
    )
    if ticker_registry is None:
        registry = build_registry(sorted(split_specs.keys()), normalized_timeframe)
        resolved_registry_path = str(
            save_registry(
                registry,
                normalized_timeframe,
                registry_path_for_tickers(normalized_timeframe, sorted(split_specs.keys())),
            )
        )
    else:
        registry = ticker_registry
        resolved_registry_path = str(ticker_registry_path or "")
    return DatasetPlan(
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        h_max=h_max,
        min_fold_samples=min_fold_samples,
        absolute_min_rows=absolute_min_rows,
        required_history_rows=required_rows,
        provider=provider,
        source=resolved_market_data_source(provider),
        source_data_hash=source_data_hash,
        date_min=date_min,
        date_max=date_max,
        usable_row_count=int(len(filtered)),
        estimated_usable_sample_count=int(estimated_sample_count),
        input_ticker_count=filtered["ticker"].nunique(),
        eligible_tickers=sorted(split_specs.keys()),
        excluded_reasons=excluded_reasons,
        split_specs=split_specs,
        ticker_registry_path=resolved_registry_path,
        num_tickers=registry["num_tickers"],
        split_mode=normalized_split_mode,
    )


def split_sequence_dataset(
    dataset: SequenceDatasetBundle | SequenceDataset,
    ratios: tuple[float, float, float] = SPLIT_RATIO,
) -> tuple[SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset]:
    frame = dataset.metadata.copy()
    frame["order_idx"] = range(len(frame))
    frame = frame.sort_values(["asof_date", "ticker"]).reset_index(drop=True)
    indices = frame["order_idx"].tolist()

    total = len(indices)
    train_end = int(total * ratios[0])
    val_end = int(total * (ratios[0] + ratios[1]))
    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise ValueError("시계열 split에 필요한 샘플 수가 부족합니다.")

    return (
        dataset.subset(indices[:train_end]),
        dataset.subset(indices[train_end:val_end]),
        dataset.subset(indices[val_end:]),
    )


def apply_calendar_split_metadata(plan: DatasetPlan, calendar_plan: CalendarSplitDatePlan) -> None:
    metadata = calendar_plan.to_metadata()
    plan.split_mode = str(metadata["split_mode"])
    plan.split_train_start_date = metadata["split_train_start_date"] if metadata["split_train_start_date"] else None
    plan.split_train_end_date = metadata["split_train_end_date"] if metadata["split_train_end_date"] else None
    plan.split_validation_start_date = metadata["split_validation_start_date"] if metadata["split_validation_start_date"] else None
    plan.split_validation_end_date = metadata["split_validation_end_date"] if metadata["split_validation_end_date"] else None
    plan.split_test_start_date = metadata["split_test_start_date"] if metadata["split_test_start_date"] else None
    plan.split_test_end_date = metadata["split_test_end_date"] if metadata["split_test_end_date"] else None
    plan.purge_gap_trading_days = int(metadata["purge_gap_trading_days"])
    plan.split_train_validation_gap_trading_days = int(metadata["split_train_validation_gap_trading_days"])
    plan.split_validation_test_gap_trading_days = int(metadata["split_validation_test_gap_trading_days"])
    plan.split_unique_dates_train = int(metadata["split_unique_dates_train"])
    plan.split_unique_dates_validation = int(metadata["split_unique_dates_validation"])
    plan.split_unique_dates_test = int(metadata["split_unique_dates_test"])
    plan.cross_split_date_overlap_count = int(metadata["cross_split_date_overlap_count"])


def dataset_plan_split_metadata(plan: DatasetPlan) -> dict[str, Any]:
    return {
        "split_mode": plan.split_mode,
        "split_train_start_date": plan.split_train_start_date,
        "split_train_end_date": plan.split_train_end_date,
        "split_validation_start_date": plan.split_validation_start_date,
        "split_validation_end_date": plan.split_validation_end_date,
        "split_test_start_date": plan.split_test_start_date,
        "split_test_end_date": plan.split_test_end_date,
        "purge_gap_trading_days": plan.purge_gap_trading_days,
        "split_train_validation_gap_trading_days": plan.split_train_validation_gap_trading_days,
        "split_validation_test_gap_trading_days": plan.split_validation_test_gap_trading_days,
        "split_unique_dates_train": plan.split_unique_dates_train,
        "split_unique_dates_validation": plan.split_unique_dates_validation,
        "split_unique_dates_test": plan.split_unique_dates_test,
        "cross_split_date_overlap_count": plan.cross_split_date_overlap_count,
    }


def _split_date_overlap_count(
    train_bundle: SequenceDatasetBundle | SequenceDataset,
    val_bundle: SequenceDatasetBundle | SequenceDataset,
    test_bundle: SequenceDatasetBundle | SequenceDataset,
) -> int:
    train_dates = set(pd.to_datetime(train_bundle.metadata["asof_date"]).dt.strftime("%Y-%m-%d").tolist())
    val_dates = set(pd.to_datetime(val_bundle.metadata["asof_date"]).dt.strftime("%Y-%m-%d").tolist())
    test_dates = set(pd.to_datetime(test_bundle.metadata["asof_date"]).dt.strftime("%Y-%m-%d").tolist())
    return len((train_dates & val_dates) | (train_dates & test_dates) | (val_dates & test_dates))


def apply_legacy_split_metadata(
    plan: DatasetPlan,
    train_bundle: SequenceDatasetBundle | SequenceDataset,
    val_bundle: SequenceDatasetBundle | SequenceDataset,
    test_bundle: SequenceDatasetBundle | SequenceDataset,
) -> None:
    def _date_min(bundle: SequenceDatasetBundle | SequenceDataset) -> str | None:
        if bundle.metadata.empty:
            return None
        return pd.to_datetime(bundle.metadata["asof_date"]).dt.strftime("%Y-%m-%d").min()

    def _date_max(bundle: SequenceDatasetBundle | SequenceDataset) -> str | None:
        if bundle.metadata.empty:
            return None
        return pd.to_datetime(bundle.metadata["asof_date"]).dt.strftime("%Y-%m-%d").max()

    def _unique_date_count(bundle: SequenceDatasetBundle | SequenceDataset) -> int:
        if bundle.metadata.empty:
            return 0
        return int(pd.to_datetime(bundle.metadata["asof_date"]).dt.strftime("%Y-%m-%d").nunique())

    plan.split_mode = SPLIT_MODE_LEGACY_TICKER_INDEX
    plan.split_train_start_date = _date_min(train_bundle)
    plan.split_train_end_date = _date_max(train_bundle)
    plan.split_validation_start_date = _date_min(val_bundle)
    plan.split_validation_end_date = _date_max(val_bundle)
    plan.split_test_start_date = _date_min(test_bundle)
    plan.split_test_end_date = _date_max(test_bundle)
    plan.purge_gap_trading_days = plan.h_max
    plan.split_train_validation_gap_trading_days = None
    plan.split_validation_test_gap_trading_days = None
    plan.split_unique_dates_train = _unique_date_count(train_bundle)
    plan.split_unique_dates_validation = _unique_date_count(val_bundle)
    plan.split_unique_dates_test = _unique_date_count(test_bundle)
    plan.cross_split_date_overlap_count = _split_date_overlap_count(train_bundle, val_bundle, test_bundle)


def split_sequence_dataset_calendar_aligned(
    dataset: SequenceDatasetBundle | SequenceDataset,
    *,
    purge_gap_trading_days: int,
    min_fold_samples: int = 50,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[
    SequenceDatasetBundle | SequenceDataset,
    SequenceDatasetBundle | SequenceDataset,
    SequenceDatasetBundle | SequenceDataset,
    CalendarSplitDatePlan,
]:
    calendar_plan = make_calendar_split_date_plan(
        dataset.metadata,
        purge_gap_trading_days=purge_gap_trading_days,
        min_fold_samples=min_fold_samples,
    )
    metadata = dataset.metadata.copy()
    metadata["order_idx"] = range(len(metadata))
    metadata["asof_date_key"] = pd.to_datetime(metadata["asof_date"]).dt.strftime("%Y-%m-%d")
    train_dates = set(calendar_plan.train_dates)
    validation_dates = set(calendar_plan.validation_dates)
    test_dates = set(calendar_plan.test_dates)
    train_indices = metadata[metadata["asof_date_key"].isin(train_dates)]["order_idx"].tolist()
    val_indices = metadata[metadata["asof_date_key"].isin(validation_dates)]["order_idx"].tolist()
    test_indices = metadata[metadata["asof_date_key"].isin(test_dates)]["order_idx"].tolist()

    if not train_indices or not val_indices or not test_indices:
        payload = dict(diagnostics or {})
        payload.update(
            {
                "error": "empty_calendar_split_result",
                "split_mode": calendar_plan.split_mode,
                "actual_train_samples": len(train_indices),
                "actual_val_samples": len(val_indices),
                "actual_test_samples": len(test_indices),
                "actual_usable_sample_count": int(len(metadata)),
                **calendar_plan.to_metadata(),
            }
        )
        raise ValueError(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return (
        dataset.subset(train_indices),
        dataset.subset(val_indices),
        dataset.subset(test_indices),
        calendar_plan,
    )


def split_sequence_dataset_by_plan(
    dataset: SequenceDatasetBundle | SequenceDataset,
    *,
    split_specs: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset]:
    metadata = dataset.metadata.copy()
    metadata["order_idx"] = range(len(metadata))

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for ticker, ticker_metadata in metadata.groupby("ticker", sort=True):
        spec = split_specs.get(str(ticker))
        if spec is None:
            continue
        ticker_metadata = ticker_metadata.sort_values("sample_index")
        train_indices.extend(
            ticker_metadata[
                (ticker_metadata["sample_index"] >= spec.train.start)
                & (ticker_metadata["sample_index"] < spec.train.end)
            ]["order_idx"].tolist()
        )
        val_indices.extend(
            ticker_metadata[
                (ticker_metadata["sample_index"] >= spec.val.start)
                & (ticker_metadata["sample_index"] < spec.val.end)
            ]["order_idx"].tolist()
        )
        test_indices.extend(
            ticker_metadata[
                (ticker_metadata["sample_index"] >= spec.test.start)
                & (ticker_metadata["sample_index"] < spec.test.end)
            ]["order_idx"].tolist()
        )

    if not train_indices or not val_indices or not test_indices:
        payload = dict(diagnostics or {})
        payload.update(
            {
                "error": "empty_split_result",
                "actual_train_samples": len(train_indices),
                "actual_val_samples": len(val_indices),
                "actual_test_samples": len(test_indices),
                "actual_usable_sample_count": int(len(metadata)),
            }
        )
        raise ValueError(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return dataset.subset(train_indices), dataset.subset(val_indices), dataset.subset(test_indices)


def _dataset_plan_diagnostics(
    *,
    plan: DatasetPlan,
    tickers: list[str] | None,
    limit_tickers: int | None,
    dataset: SequenceDatasetBundle | SequenceDataset | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": plan.provider,
        "source": plan.source,
        "timeframe": plan.timeframe,
        "ticker_count": int(plan.input_ticker_count),
        "eligible_ticker_count": len(plan.eligible_tickers),
        "date_min": plan.date_min,
        "date_max": plan.date_max,
        "seq_len": plan.seq_len,
        "horizon": plan.horizon,
        "gap": plan.h_max,
        "min_fold_samples": plan.min_fold_samples,
        "absolute_min_rows": plan.absolute_min_rows,
        "required_history_rows": plan.required_history_rows,
        "source_data_hash": plan.source_data_hash,
        "usable_row_count": plan.usable_row_count,
        "estimated_usable_sample_count": plan.estimated_usable_sample_count,
        "requested_tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
        "excluded_reasons": plan.excluded_reasons,
    }
    payload.update(dataset_plan_split_metadata(plan))
    if dataset is not None:
        metadata = dataset.metadata
        payload["actual_usable_sample_count"] = int(len(metadata))
        if not metadata.empty and "ticker" in metadata.columns:
            payload["actual_sample_count_by_ticker"] = {
                str(ticker): int(count)
                for ticker, count in metadata.groupby("ticker").size().sort_index().items()
            }
    return payload


def fit_feature_stats(train_data: torch.Tensor | SequenceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(train_data, torch.Tensor):
        mean = train_data.mean(dim=(0, 1))
        std = train_data.std(dim=(0, 1)).clamp_min(1e-6)
        return mean, std

    total_sum: torch.Tensor | None = None
    total_sq_sum: torch.Tensor | None = None
    total_count = 0
    ticker_to_end_indices: dict[str, list[int]] = {}
    for ticker, end_idx in train_data.sample_refs:
        ticker_to_end_indices.setdefault(ticker, []).append(end_idx)

    for ticker, end_indices in ticker_to_end_indices.items():
        feature_array = torch.from_numpy(train_data.ticker_arrays[ticker]["features"])
        windows = feature_array.unfold(0, train_data.seq_len, 1).permute(0, 2, 1)
        start_indices = torch.tensor(
            [end_idx - train_data.seq_len + 1 for end_idx in end_indices],
            dtype=torch.long,
        )
        selected = windows.index_select(0, start_indices).to(dtype=torch.float32)
        batch_sum = selected.sum(dim=(0, 1))
        batch_sq_sum = selected.square().sum(dim=(0, 1))
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_sq_sum = batch_sq_sum if total_sq_sum is None else total_sq_sum + batch_sq_sum
        total_count += selected.shape[0] * selected.shape[1]

    if total_sum is None or total_sq_sum is None or total_count == 0:
        raise ValueError("정규화 통계를 계산할 학습 샘플이 없습니다.")

    mean = total_sum / float(total_count)
    variance = (total_sq_sum / float(total_count)) - mean.square()
    std = variance.clamp_min(1e-6).sqrt()
    return mean, std


def apply_feature_stats(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def normalize_sequence_splits(
    train_bundle: SequenceDatasetBundle | SequenceDataset,
    val_bundle: SequenceDatasetBundle | SequenceDataset,
    test_bundle: SequenceDatasetBundle | SequenceDataset,
) -> tuple[SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, torch.Tensor, torch.Tensor]:
    mean, std = fit_feature_stats(train_bundle.features if isinstance(train_bundle, SequenceDatasetBundle) else train_bundle)

    def _apply(bundle: SequenceDatasetBundle | SequenceDataset) -> SequenceDatasetBundle | SequenceDataset:
        if isinstance(bundle, SequenceDataset):
            return bundle.with_normalization(mean, std)
        return SequenceDatasetBundle(
            features=apply_feature_stats(bundle.features, mean, std),
            line_targets=bundle.line_targets,
            band_targets=bundle.band_targets,
            raw_future_returns=bundle.raw_future_returns,
            anchor_closes=bundle.anchor_closes,
            ticker_ids=bundle.ticker_ids,
            future_covariates=bundle.future_covariates,
            metadata=bundle.metadata.copy(),
        )

    return _apply(train_bundle), _apply(val_bundle), _apply(test_bundle), mean, std


def prepare_dataset_splits(
    *,
    timeframe: str,
    seq_len: int,
    horizon: int,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    min_fold_samples: int = 50,
    include_future_covariate: bool = True,
    line_target_type: str = "raw_future_return",
    band_target_type: str = "raw_future_return",
    ticker_registry: dict[str, Any] | None = None,
    ticker_registry_path: str | None = None,
    market_data_provider: str | None = None,
    split_mode: str = SPLIT_MODE_CALENDAR_ALIGNED,
) -> tuple[SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, torch.Tensor, torch.Tensor, DatasetPlan]:
    """현재 DB 기준으로 학습용 시퀀스와 분할 계획을 한 번에 준비한다."""
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    provider = resolved_market_data_provider(market_data_provider)
    normalized_split_mode = normalize_split_mode(split_mode)
    cache_key = resolve_prepared_splits_cache_key(
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        tickers=tickers,
        limit_tickers=limit_tickers,
        min_fold_samples=min_fold_samples,
        include_future_covariate=include_future_covariate,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
        ticker_registry=ticker_registry,
        market_data_provider=provider,
        split_mode=normalized_split_mode,
    )
    if cache_key in _PREPARED_SPLITS_CACHE:
        return _PREPARED_SPLITS_CACHE[cache_key]

    index_frame = fetch_feature_index_frame(
        timeframe=normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    data_hash = resolve_data_fingerprint(
        normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
        market_data_provider=provider,
    )
    plan = build_dataset_plan(
        index_frame,
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        min_fold_samples=min_fold_samples,
        ticker_registry=ticker_registry,
        ticker_registry_path=ticker_registry_path,
        market_data_provider=provider,
        source_data_hash=data_hash,
        split_mode=normalized_split_mode,
    )
    if not plan.eligible_tickers:
        payload = _dataset_plan_diagnostics(
            plan=plan,
            tickers=tickers,
            limit_tickers=limit_tickers,
        )
        payload["error"] = "no_eligible_tickers_for_provider_source"
        raise ValueError(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    feature_df, price_df = fetch_training_frames(
        timeframe=normalized_timeframe,
        tickers=plan.eligible_tickers,
        limit_tickers=None,
        market_data_provider=provider,
    )
    feature_df = feature_df[feature_df["ticker"].isin(plan.eligible_tickers)].copy()
    price_df = price_df[price_df["ticker"].isin(plan.eligible_tickers)].copy()
    active_ticker_registry = ticker_registry or build_registry(plan.eligible_tickers, plan.timeframe)
    dataset = build_lazy_sequence_dataset(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=active_ticker_registry,
        include_future_covariate=include_future_covariate,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
    )
    diagnostics = _dataset_plan_diagnostics(
        plan=plan,
        tickers=tickers,
        limit_tickers=limit_tickers,
        dataset=dataset,
    )
    if normalized_split_mode == SPLIT_MODE_CALENDAR_ALIGNED:
        train_bundle, val_bundle, test_bundle, calendar_plan = split_sequence_dataset_calendar_aligned(
            dataset,
            purge_gap_trading_days=plan.h_max,
            min_fold_samples=plan.min_fold_samples,
            diagnostics=diagnostics,
        )
        apply_calendar_split_metadata(plan, calendar_plan)
    else:
        train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(
            dataset,
            split_specs=plan.split_specs,
            diagnostics=diagnostics,
        )
        apply_legacy_split_metadata(plan, train_bundle, val_bundle, test_bundle)
    normalized = normalize_sequence_splits(train_bundle, val_bundle, test_bundle)
    result = (*normalized, plan)
    _PREPARED_SPLITS_CACHE[cache_key] = result
    return result
