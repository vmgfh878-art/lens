"""
Lens 학습 / 배치 추론용 전처리 헬퍼.

규칙:
    - 17개 고정 피처 사용
    - 시간순 split만 허용
    - Z-score는 train에서만 fit
    - 타임프레임별 리샘플링은 feature_svc를 재사용
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.feature_svc import (  # noqa: E402
    FEATURE_COLUMNS,
    build_features,
    default_horizon_for_timeframe,
    normalize_timeframe,
)

SPLIT_RATIO = (0.7, 0.15, 0.15)


def build_feature_frame(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    timeframe: str = "1D",
) -> pd.DataFrame:
    return build_features(price_df=price_df, macro_df=macro_df, breadth_df=breadth_df, timeframe=timeframe)


def split_by_time(df: pd.DataFrame):
    frame = df.sort_values("date").reset_index(drop=True)
    total = len(frame)
    train_end = int(total * SPLIT_RATIO[0])
    val_end = int(total * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

    train = frame.iloc[:train_end].copy()
    val = frame.iloc[train_end:val_end].copy()
    test = frame.iloc[val_end:].copy()

    if not train.empty:
        print(f"Train: {train['date'].min()} ~ {train['date'].max()} ({len(train):,} rows)")
    if not val.empty:
        print(f"Val  : {val['date'].min()} ~ {val['date'].max()} ({len(val):,} rows)")
    if not test.empty:
        print(f"Test : {test['date'].min()} ~ {test['date'].max()} ({len(test):,} rows)")

    return train, val, test


def fit_zscore_stats(train_df: pd.DataFrame, feature_columns: list[str] | None = None):
    columns = feature_columns or FEATURE_COLUMNS
    active_columns = [column for column in columns if column in train_df.columns]
    mean = train_df[active_columns].mean()
    std = train_df[active_columns].std().replace(0, 1)
    return mean, std


def apply_zscore(
    df: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = feature_columns or FEATURE_COLUMNS
    active_columns = [column for column in columns if column in df.columns and column in mean.index]
    normalized = df.copy()
    normalized[active_columns] = (normalized[active_columns] - mean[active_columns]) / std[active_columns]
    return normalized


def zscore_normalize(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    mean, std = fit_zscore_stats(train)
    return apply_zscore(train, mean, std), apply_zscore(val, mean, std), apply_zscore(test, mean, std)


def default_horizon(timeframe: str) -> int:
    return default_horizon_for_timeframe(normalize_timeframe(timeframe))
