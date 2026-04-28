from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.app.services.feature_svc import FEATURE_COLUMNS as SOURCE_FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, normalize_timeframe, resample_price_frame  # noqa: E402
from backend.collector.repositories.base import fetch_frame  # noqa: E402
from ai.targets import build_target_array, normalize_target_type  # noqa: E402
from ai.splits import MAX_HORIZON_BY_TIMEFRAME, build_split_specs  # noqa: E402
from ai.ticker_registry import build_registry, lookup_id, save_registry  # noqa: E402

SPLIT_RATIO = (0.7, 0.15, 0.15)
SUPPORTED_AI_TIMEFRAMES = ("1D", "1W")
CACHE_DIR = PROJECT_ROOT / "ai" / "cache"
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
_PREPARED_SPLITS_CACHE: dict[str, Any] = {}
_ENGINE_CACHE: dict[str, Any] = {}
_STALE_CACHE_WARNED: set[str] = set()
_FEATURE_CONTRACT_VERSION = "v2_finite"
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
    input_ticker_count: int
    eligible_tickers: list[str]
    excluded_reasons: dict[str, str]
    split_specs: dict[str, Any]
    ticker_registry_path: str
    num_tickers: int


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
) -> list[str]:
    if tickers:
        return [ticker.upper() for ticker in tickers]

    if limit_tickers is not None:
        ticker_df = pd.read_sql_query(
            f"SELECT ticker FROM stock_info ORDER BY ticker LIMIT {int(limit_tickers)}",
            conn,
        )
        return ticker_df["ticker"].astype(str).str.upper().tolist()

    ticker_df = pd.read_sql_query(
        """
            SELECT DISTINCT ticker
            FROM indicators
            WHERE timeframe = %(timeframe)s
            ORDER BY ticker
        """,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = _postgres_engine()
    with engine.begin() as conn:
        normalized_tickers = _normalized_tickers(
            conn,
            timeframe=timeframe,
            tickers=tickers,
            limit_tickers=limit_tickers,
        )
        feature_frames: list[pd.DataFrame] = []
        price_frames: list[pd.DataFrame] = []

        for ticker_chunk in _chunked(normalized_tickers, POSTGRES_TICKER_CHUNK_SIZE):
            ticker_filter = _sql_filter_clause("ticker", ticker_chunk)
            indicator_query = """
                SELECT ticker, timeframe, date, {feature_columns}
                FROM indicators
                WHERE timeframe = %(timeframe)s AND {ticker_filter}
                ORDER BY ticker, date
            """.format(
                feature_columns=", ".join(SOURCE_FEATURE_COLUMNS),
                ticker_filter=ticker_filter,
            )
            price_query = """
                SELECT ticker, date, open, high, low, close, adjusted_close, volume
                FROM price_data
                WHERE {ticker_filter}
                ORDER BY ticker, date
            """.format(ticker_filter=ticker_filter)

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
) -> pd.DataFrame:
    engine = _postgres_engine()
    with engine.begin() as conn:
        normalized_tickers = _normalized_tickers(
            conn,
            timeframe=timeframe,
            tickers=tickers,
            limit_tickers=limit_tickers,
        )
        frames: list[pd.DataFrame] = []

        for ticker_chunk in _chunked(normalized_tickers, POSTGRES_TICKER_CHUNK_SIZE):
            query = """
                SELECT ticker, timeframe, date
                FROM indicators
                WHERE timeframe = %(timeframe)s AND {ticker_filter}
                ORDER BY ticker, date
            """.format(ticker_filter=_sql_filter_clause("ticker", ticker_chunk))
            frames.append(pd.read_sql_query(query, conn, params={"timeframe": timeframe}))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_feature_index_frame(
    *,
    timeframe: str = "1D",
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> pd.DataFrame:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    data_hash = resolve_data_fingerprint(normalized_timeframe)
    cache_path = resolve_feature_index_cache_path(
        timeframe=normalized_timeframe,
        data_hash=data_hash,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    _maybe_warn_stale_cache(f"feature_index_{normalized_timeframe}", cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False).copy()

    try:
        index_frame = _fetch_feature_index_frame_via_postgres(
            timeframe=normalized_timeframe,
            tickers=tickers,
            limit_tickers=limit_tickers,
        )
    except Exception:
        filters: list[tuple[str, str, object]] = [("eq", "timeframe", normalized_timeframe)]
        if tickers:
            filters.append(("in", "ticker", [ticker.upper() for ticker in tickers]))
        elif limit_tickers is not None:
            known = fetch_frame("stock_info", columns="ticker", order_by="ticker", limit=limit_tickers)
            filters.append(("in", "ticker", known["ticker"].astype(str).str.upper().tolist()))
        index_frame = fetch_frame(
            "indicators",
            columns="ticker,timeframe,date",
            filters=filters,
            order_by="date",
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index_frame, cache_path)
    return index_frame.copy()


def fetch_training_frames(
    *,
    timeframe: str = "1D",
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supabase에서 학습용 indicators와 price_data를 읽는다."""
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    data_hash = resolve_data_fingerprint(normalized_timeframe)
    cache_path = resolve_feature_cache_path(
        timeframe=normalized_timeframe,
        data_hash=data_hash,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    _maybe_warn_stale_cache(f"features_{normalized_timeframe}", cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["feature_df"].copy(), cached["price_df"].copy()

    try:
        feature_df, price_df = _fetch_training_frames_via_postgres(
            timeframe=normalized_timeframe,
            tickers=tickers,
            limit_tickers=limit_tickers,
        )
    except Exception:
        feature_df, price_df = pd.DataFrame(), pd.DataFrame()

    ticker_filters: list[tuple[str, str, object]] = [("eq", "timeframe", normalized_timeframe)]
    price_filters: list[tuple[str, str, object]] = []

    if feature_df.empty or price_df.empty:
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
        feature_df = fetch_frame(
            "indicators",
            columns=indicator_columns,
            filters=ticker_filters,
            order_by="date",
        )
        price_df = fetch_frame(
            "price_data",
            columns="ticker,date,open,high,low,close,adjusted_close,volume",
            filters=price_filters,
            order_by="date",
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feature_df": feature_df, "price_df": price_df}, cache_path)
    return feature_df, price_df


def default_horizon(timeframe: str) -> int:
    normalized = normalize_ai_timeframe(timeframe)
    return {"1D": 5, "1W": 4}[normalized]


def normalize_ai_timeframe(timeframe: str) -> str:
    normalized = normalize_timeframe(timeframe)
    if normalized not in SUPPORTED_AI_TIMEFRAMES:
        raise ValueError("월봉 AI 학습과 추론은 Phase 1에서 지원하지 않습니다.")
    return normalized


def resolve_data_fingerprint(timeframe: str) -> str:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    fingerprint_payload: dict[str, Any] = {
        "timeframe": normalized_timeframe,
        "price_max_date": None,
        "indicator_max_date": None,
        "indicator_count": None,
    }

    try:
        engine = _postgres_engine()
        with engine.begin() as conn:
            price_meta = pd.read_sql_query(
                "SELECT MAX(date) AS max_date FROM price_data",
                conn,
            ).iloc[0]
            indicator_meta = pd.read_sql_query(
                """
                    SELECT MAX(date) AS max_date, COUNT(*) AS row_count
                    FROM indicators
                    WHERE timeframe = %(timeframe)s
                """,
                conn,
                params={"timeframe": normalized_timeframe},
            ).iloc[0]
        fingerprint_payload["price_max_date"] = str(price_meta["max_date"])
        fingerprint_payload["indicator_max_date"] = str(indicator_meta["max_date"])
        fingerprint_payload["indicator_count"] = int(indicator_meta["row_count"])
    except Exception:
        indicator_frame = fetch_frame(
            "indicators",
            columns="date",
            filters=[("eq", "timeframe", normalized_timeframe)],
            order_by="date",
        )
        price_frame = fetch_frame("price_data", columns="date", order_by="date")
        fingerprint_payload["price_max_date"] = str(price_frame["date"].max()) if not price_frame.empty else None
        fingerprint_payload["indicator_max_date"] = str(indicator_frame["date"].max()) if not indicator_frame.empty else None
        fingerprint_payload["indicator_count"] = int(len(indicator_frame))

    return hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]


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
) -> Path:
    payload = {
        "timeframe": timeframe,
        "source_feature_columns": SOURCE_FEATURE_COLUMNS,
        "calendar_feature_columns": CALENDAR_FEATURE_COLUMNS,
        "feature_contract_version": _FEATURE_CONTRACT_VERSION,
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
) -> Path:
    payload = {
        "timeframe": timeframe,
        "tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
        "kind": "feature_index_v1",
        "feature_contract_version": _FEATURE_CONTRACT_VERSION,
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
) -> str:
    data_hash = resolve_data_fingerprint(timeframe)
    payload = {
        "timeframe": timeframe,
        "data_hash": data_hash,
        "seq_len": seq_len,
        "horizon": horizon,
        "tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
        "min_fold_samples": min_fold_samples,
        "include_future_covariate": include_future_covariate,
        "line_target_type": normalize_target_type(line_target_type),
        "band_target_type": normalize_target_type(band_target_type),
        "source_feature_columns": SOURCE_FEATURE_COLUMNS,
        "calendar_feature_columns": CALENDAR_FEATURE_COLUMNS,
        "feature_contract_version": _FEATURE_CONTRACT_VERSION,
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
) -> DatasetPlan:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    filtered = feature_df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered[filtered["timeframe"] == normalized_timeframe].sort_values(["ticker", "date"])
    h_max = MAX_HORIZON_BY_TIMEFRAME[normalized_timeframe]
    split_specs, excluded_reasons = build_split_specs(
        filtered,
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        h_max=h_max,
        min_fold_samples=min_fold_samples,
    )
    registry = build_registry(sorted(split_specs.keys()), normalized_timeframe)
    registry_path = save_registry(registry, normalized_timeframe)
    return DatasetPlan(
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        h_max=h_max,
        min_fold_samples=min_fold_samples,
        input_ticker_count=filtered["ticker"].nunique(),
        eligible_tickers=sorted(split_specs.keys()),
        excluded_reasons=excluded_reasons,
        split_specs=split_specs,
        ticker_registry_path=str(registry_path),
        num_tickers=registry["num_tickers"],
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


def split_sequence_dataset_by_plan(
    dataset: SequenceDatasetBundle | SequenceDataset,
    *,
    split_specs: dict[str, Any],
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
        raise ValueError("split 결과가 비어 있습니다.")

    return dataset.subset(train_indices), dataset.subset(val_indices), dataset.subset(test_indices)


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
) -> tuple[SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, SequenceDatasetBundle | SequenceDataset, torch.Tensor, torch.Tensor, DatasetPlan]:
    """현재 DB 기준으로 학습용 시퀀스와 분할 계획을 한 번에 준비한다."""
    normalized_timeframe = normalize_ai_timeframe(timeframe)
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
    )
    if cache_key in _PREPARED_SPLITS_CACHE:
        return _PREPARED_SPLITS_CACHE[cache_key]

    index_frame = fetch_feature_index_frame(
        timeframe=normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    plan = build_dataset_plan(
        index_frame,
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        min_fold_samples=min_fold_samples,
    )
    feature_df, price_df = fetch_training_frames(
        timeframe=normalized_timeframe,
        tickers=plan.eligible_tickers,
        limit_tickers=None,
    )
    feature_df = feature_df[feature_df["ticker"].isin(plan.eligible_tickers)].copy()
    price_df = price_df[price_df["ticker"].isin(plan.eligible_tickers)].copy()
    ticker_registry = build_registry(plan.eligible_tickers, plan.timeframe)
    dataset = build_lazy_sequence_dataset(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=normalized_timeframe,
        seq_len=seq_len,
        horizon=horizon,
        ticker_registry=ticker_registry,
        include_future_covariate=include_future_covariate,
        line_target_type=line_target_type,
        band_target_type=band_target_type,
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(
        dataset,
        split_specs=plan.split_specs,
    )
    normalized = normalize_sequence_splits(train_bundle, val_bundle, test_bundle)
    result = (*normalized, plan)
    _PREPARED_SPLITS_CACHE[cache_key] = result
    return result
