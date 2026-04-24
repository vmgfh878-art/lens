from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.app.services.feature_svc import FEATURE_COLUMNS, REQUIRED_FEATURE_COLUMNS, normalize_timeframe, resample_price_frame  # noqa: E402
from backend.collector.repositories.base import fetch_frame  # noqa: E402
from ai.splits import MAX_HORIZON_BY_TIMEFRAME, build_split_specs  # noqa: E402

SPLIT_RATIO = (0.7, 0.15, 0.15)
SUPPORTED_AI_TIMEFRAMES = ("1D", "1W")
CACHE_DIR = PROJECT_ROOT / "ai" / "cache"


@dataclass
class SequenceDatasetBundle:
    """학습과 추론에 필요한 시계열 텐서와 메타데이터 묶음이다."""

    features: torch.Tensor
    line_targets: torch.Tensor
    band_targets: torch.Tensor
    anchor_closes: torch.Tensor
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, indices: list[int]) -> "SequenceDatasetBundle":
        return SequenceDatasetBundle(
            features=self.features[indices],
            line_targets=self.line_targets[indices],
            band_targets=self.band_targets[indices],
            anchor_closes=self.anchor_closes[indices],
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
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


def _sql_filter_clause(column: str, values: list[str]) -> str:
    escaped_values = []
    for value in values:
        sanitized = value.replace("'", "''")
        escaped_values.append(f"'{sanitized}'")
    escaped = ",".join(escaped_values)
    return f"{column} IN ({escaped})"


def _fetch_training_frames_via_postgres(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dsn = _postgres_dsn()
    if dsn is None:
        raise RuntimeError("Postgres 접속 정보가 없습니다.")

    normalized_tickers = [ticker.upper() for ticker in tickers] if tickers else None
    with psycopg2.connect(dsn) as conn:
        if normalized_tickers is None and limit_tickers is not None:
            ticker_df = pd.read_sql_query(
                f"SELECT ticker FROM stock_info ORDER BY ticker LIMIT {int(limit_tickers)}",
                conn,
            )
            normalized_tickers = ticker_df["ticker"].astype(str).str.upper().tolist()

        indicator_where = [f"timeframe = '{timeframe}'"]
        price_where: list[str] = []
        if normalized_tickers:
            indicator_where.append(_sql_filter_clause("ticker", normalized_tickers))
            price_where.append(_sql_filter_clause("ticker", normalized_tickers))

        indicator_query = """
            SELECT ticker, timeframe, date, {feature_columns}
            FROM indicators
            WHERE {where_clause}
            ORDER BY ticker, date
        """.format(
            feature_columns=", ".join(FEATURE_COLUMNS),
            where_clause=" AND ".join(indicator_where),
        )
        price_query = """
            SELECT ticker, date, open, high, low, close, adjusted_close, volume
            FROM price_data
            {where_clause}
            ORDER BY ticker, date
        """.format(
            where_clause=f"WHERE {' AND '.join(price_where)}" if price_where else "",
        )

        feature_df = pd.read_sql_query(indicator_query, conn)
        price_df = pd.read_sql_query(price_query, conn)
    return feature_df, price_df


def _fetch_feature_index_frame_via_postgres(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> pd.DataFrame:
    dsn = _postgres_dsn()
    if dsn is None:
        raise RuntimeError("Postgres 접속 정보가 없습니다.")

    normalized_tickers = [ticker.upper() for ticker in tickers] if tickers else None
    with psycopg2.connect(dsn) as conn:
        if normalized_tickers is None and limit_tickers is not None:
            ticker_df = pd.read_sql_query(
                f"SELECT ticker FROM stock_info ORDER BY ticker LIMIT {int(limit_tickers)}",
                conn,
            )
            normalized_tickers = ticker_df["ticker"].astype(str).str.upper().tolist()

        where_clauses = [f"timeframe = '{timeframe}'"]
        if normalized_tickers:
            where_clauses.append(_sql_filter_clause("ticker", normalized_tickers))
        query = """
            SELECT ticker, timeframe, date
            FROM indicators
            WHERE {where_clause}
            ORDER BY ticker, date
        """.format(where_clause=" AND ".join(where_clauses))
        return pd.read_sql_query(query, conn)


def fetch_feature_index_frame(
    *,
    timeframe: str = "1D",
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> pd.DataFrame:
    normalized_timeframe = normalize_ai_timeframe(timeframe)
    try:
        return _fetch_feature_index_frame_via_postgres(
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
        return fetch_frame(
            "indicators",
            columns="ticker,timeframe,date",
            filters=filters,
            order_by="date",
        )


def fetch_training_frames(
    *,
    timeframe: str = "1D",
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supabase에서 학습용 indicators와 price_data를 읽는다."""

    normalized_timeframe = normalize_ai_timeframe(timeframe)
    cache_path = resolve_feature_cache_path(
        timeframe=normalized_timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
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

        indicator_columns = ",".join(["ticker", "timeframe", "date", *FEATURE_COLUMNS])
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


def resolve_feature_cache_path(
    *,
    timeframe: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> Path:
    payload = {
        "timeframe": timeframe,
        "feature_columns": FEATURE_COLUMNS,
        "tickers": [ticker.upper() for ticker in tickers] if tickers else None,
        "limit_tickers": limit_tickers,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"features_{timeframe}_{digest}.pt"


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
) -> SequenceDatasetBundle:
    """feature와 price를 묶어 multi-step 학습 샘플을 만든다."""

    normalized_timeframe = normalize_ai_timeframe(timeframe)
    resolved_horizon = horizon or default_horizon(normalized_timeframe)
    if feature_df.empty or price_df.empty:
        raise ValueError("학습 데이터가 비어 있습니다. indicators와 price_data를 먼저 확인하세요.")

    features = feature_df.copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["ticker", "date"]).dropna(subset=REQUIRED_FEATURE_COLUMNS)

    targets = _build_target_frame(price_df, normalized_timeframe)
    merged = features.merge(targets, on=["ticker", "date"], how="inner")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged = merged.dropna(subset=["target_close"])

    samples_x: list[list[list[float]]] = []
    samples_line: list[list[float]] = []
    samples_band: list[list[float]] = []
    anchor_closes: list[float] = []
    metadata_rows: list[dict[str, object]] = []

    for ticker, ticker_frame in merged.groupby("ticker", sort=True):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        if len(ticker_frame) < (seq_len + resolved_horizon):
            continue

        feature_values = ticker_frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
        closes = ticker_frame["target_close"].to_numpy(dtype="float32")
        dates = ticker_frame["date"].dt.strftime("%Y-%m-%d").tolist()
        sample_index = 0

        for end_idx in range(seq_len - 1, len(ticker_frame) - resolved_horizon):
            start_idx = end_idx - seq_len + 1
            future_slice = slice(end_idx + 1, end_idx + 1 + resolved_horizon)
            window_features = feature_values[start_idx : end_idx + 1]
            future_prices = closes[future_slice]
            anchor_close = float(closes[end_idx])
            if anchor_close == 0:
                continue

            future_returns = (future_prices / anchor_close) - 1.0
            samples_x.append(window_features.tolist())
            samples_line.append(future_returns.tolist())
            samples_band.append(future_returns.tolist())
            anchor_closes.append(anchor_close)
            metadata_rows.append(
                {
                    "ticker": ticker,
                    "timeframe": normalized_timeframe,
                    "asof_date": dates[end_idx],
                    "forecast_dates": dates[future_slice],
                    "sample_index": sample_index,
                }
            )
            sample_index += 1

    if not samples_x:
        raise ValueError("지정한 조건에서 시퀀스 샘플을 만들지 못했습니다.")

    return SequenceDatasetBundle(
        features=torch.tensor(samples_x, dtype=torch.float32),
        line_targets=torch.tensor(samples_line, dtype=torch.float32),
        band_targets=torch.tensor(samples_band, dtype=torch.float32),
        anchor_closes=torch.tensor(anchor_closes, dtype=torch.float32),
        metadata=pd.DataFrame(metadata_rows),
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
    )


def split_sequence_dataset(
    dataset: SequenceDatasetBundle,
    ratios: tuple[float, float, float] = SPLIT_RATIO,
) -> tuple[SequenceDatasetBundle, SequenceDatasetBundle, SequenceDatasetBundle]:
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
    dataset: SequenceDatasetBundle,
    *,
    split_specs: dict[str, Any],
) -> tuple[SequenceDatasetBundle, SequenceDatasetBundle, SequenceDatasetBundle]:
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


def fit_feature_stats(train_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=(0, 1))
    std = train_features.std(dim=(0, 1)).clamp_min(1e-6)
    return mean, std


def apply_feature_stats(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def normalize_sequence_splits(
    train_bundle: SequenceDatasetBundle,
    val_bundle: SequenceDatasetBundle,
    test_bundle: SequenceDatasetBundle,
) -> tuple[SequenceDatasetBundle, SequenceDatasetBundle, SequenceDatasetBundle, torch.Tensor, torch.Tensor]:
    mean, std = fit_feature_stats(train_bundle.features)

    def _apply(bundle: SequenceDatasetBundle) -> SequenceDatasetBundle:
        return SequenceDatasetBundle(
            features=apply_feature_stats(bundle.features, mean, std),
            line_targets=bundle.line_targets,
            band_targets=bundle.band_targets,
            anchor_closes=bundle.anchor_closes,
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
) -> tuple[SequenceDatasetBundle, SequenceDatasetBundle, SequenceDatasetBundle, torch.Tensor, torch.Tensor, DatasetPlan]:
    """현재 DB 기준으로 학습용 시퀀스와 정규화 통계를 한 번에 준비한다."""
    index_frame = fetch_feature_index_frame(
        timeframe=timeframe,
        tickers=tickers,
        limit_tickers=limit_tickers,
    )
    plan = build_dataset_plan(
        index_frame,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
        min_fold_samples=min_fold_samples,
    )
    feature_df, price_df = fetch_training_frames(
        timeframe=timeframe,
        tickers=plan.eligible_tickers,
        limit_tickers=None,
    )
    feature_df = feature_df[feature_df["ticker"].isin(plan.eligible_tickers)].copy()
    price_df = price_df[price_df["ticker"].isin(plan.eligible_tickers)].copy()
    dataset = build_sequence_dataset(
        feature_df=feature_df,
        price_df=price_df,
        timeframe=timeframe,
        seq_len=seq_len,
        horizon=horizon,
    )
    train_bundle, val_bundle, test_bundle = split_sequence_dataset_by_plan(
        dataset,
        split_specs=plan.split_specs,
    )
    normalized = normalize_sequence_splits(train_bundle, val_bundle, test_bundle)
    return (*normalized, plan)
