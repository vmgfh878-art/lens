"""
parquet I/O 경계 검증용 pandera DataFrameModel.

CP227 신설. CP214 사고 (datetime64 vs category merge 실패) 의 재발 방지가 목적.
parquet_store / local_market_svc 가 disk 에서 read 직후 / strftime 직전 에
validate(...) 를 호출해 dtype 계약 위반을 읽는 순간 잡는다.

★ 결정적 제약: disk 상 `asof_date` / `forecast_date` / `date` 는 object(str) 이고
parquet_store._compress_strings 가 의도적으로 그렇게 유지한다 (CP214 fix).
스키마는 이 컬럼들을 Series[str] + coerce=True 로 모델링한다. datetime 으로
강제하면 CP214 회귀를 부른다.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series


class LineDailyFrame(pa.DataFrameModel):
    ticker: Series[str] = pa.Field(coerce=True)
    asof_date: Series[str] = pa.Field(coerce=True)  # CP214 — datetime 아님
    line_score: Series[float] = pa.Field(coerce=True, nullable=True)
    safe_line_score: Series[float] = pa.Field(coerce=True, nullable=True)
    line_rank_by_date: Series[float] = pa.Field(coerce=True, nullable=True)
    safe_line_rank_by_date: Series[float] = pa.Field(coerce=True, nullable=True)
    model_id: Series[str] = pa.Field(coerce=True, nullable=True)
    source_cp: Series[str] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = False
        coerce = True


class Band1dFrame(pa.DataFrameModel):
    ticker: Series[str] = pa.Field(coerce=True)
    asof_date: Series[str] = pa.Field(coerce=True)  # CP214
    forecast_date: Series[str] = pa.Field(coerce=True)  # CP214
    horizon_step: Series[int] = pa.Field(coerce=True)
    band_lower: Series[float] = pa.Field(coerce=True, nullable=True)
    band_upper: Series[float] = pa.Field(coerce=True, nullable=True)
    model_id: Series[str] = pa.Field(coerce=True, nullable=True)
    source_cp: Series[str] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = False
        coerce = True


class Band1wFrame(pa.DataFrameModel):
    ticker: Series[str] = pa.Field(coerce=True)
    asof_date: Series[str] = pa.Field(coerce=True)  # CP214
    horizon_step: Series[int] = pa.Field(coerce=True)
    band_lower: Series[float] = pa.Field(coerce=True, nullable=True)
    band_upper: Series[float] = pa.Field(coerce=True, nullable=True)
    model_id: Series[str] = pa.Field(coerce=True, nullable=True)
    source_cp: Series[str] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = False
        coerce = True


class MarketPrices1d(pa.DataFrameModel):
    ticker: Series[str] = pa.Field(coerce=True)
    date: Series[str] = pa.Field(coerce=True)  # CP214
    open: Series[float] = pa.Field(coerce=True, nullable=True)
    high: Series[float] = pa.Field(coerce=True, nullable=True)
    low: Series[float] = pa.Field(coerce=True, nullable=True)
    close: Series[float] = pa.Field(coerce=True, nullable=True)
    volume: Series[int] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = False
        coerce = True


class MarketIndicators1d(pa.DataFrameModel):
    ticker: Series[str] = pa.Field(coerce=True)
    date: Series[str] = pa.Field(coerce=True)  # CP214

    class Config:
        strict = False
        coerce = True


def validate(model: type[pa.DataFrameModel], df, *, name: str):
    """lazy 검증 — 모든 위반을 한 번에 수집해 pandera.errors.SchemaErrors 로 전파.

    호출자 (parquet_store._load, local_market_svc._load 등) 는 실패 시
    예외를 그대로 올리고 main.py 의 글로벌 핸들러가 500 으로 매핑한다.
    name 은 슬롯 / 파일 라벨로 로그 추적용.
    """
    return model.validate(df, lazy=True)


__all__ = [
    "LineDailyFrame",
    "Band1dFrame",
    "Band1wFrame",
    "MarketPrices1d",
    "MarketIndicators1d",
    "validate",
]
