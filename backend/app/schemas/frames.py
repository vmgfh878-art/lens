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

import pandas as pd
import pandera as pa
from pandas.api import types as pdt
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

    ⚠ CP246 — read 핫패스에서는 이 함수 대신 `assert_contract` 를 쓴다.
    pandera 의 `coerce=True` 가 597 k 행 문자열 컬럼을 object 로 복제하면서
    cold load 마다 peak RSS 가 +236MB(band_1d) 튀어 512MB OOM 을 유발했다.
    위 DataFrameModel 들은 deep 계약 문서 / 테스트용으로 보존한다.
    """
    return model.validate(df, lazy=True)


# CP246 — read 경계 경량 dtype 계약 (할당 0).
# pandera coerce-validate 와 같은 의도(CP214/CP227 회귀 방지)지만 값/행을
# 만지지 않고 "컬럼 존재 + dtype family" 만 본다. date 컬럼은 단계에 따라
# str-like(disk 직후) 또는 datetime64(_compress_strings 후 / 재인코딩 파일)
# 둘 다 허용한다. 슬롯명(line_1d…)과 파일명(market_*.parquet) 둘 다 키로 둔다.
_CONTRACTS: dict[str, dict[str, str]] = {
    "line_1d": {
        "ticker": "str",
        "asof_date": "date",
        "line_score": "float",
        "model_id": "str",
        "source_cp": "str",
    },
    "band_1d": {
        "ticker": "str",
        "asof_date": "date",
        "forecast_date": "date",
        "horizon_step": "int",
        "band_lower": "float",
        "band_upper": "float",
        "model_id": "str",
        "source_cp": "str",
    },
    "band_1w": {
        "ticker": "str",
        "asof_date": "date",
        "horizon_step": "int",
        "band_lower": "float",
        "band_upper": "float",
        "model_id": "str",
        "source_cp": "str",
    },
    "market_prices_1d.parquet": {
        "ticker": "str",
        "date": "date",
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "num",
    },
    "market_indicators_1d.parquet": {
        "ticker": "str",
        "date": "date",
    },
}


def _family_ok(series: pd.Series, fam: str) -> bool:
    dt = series.dtype
    is_strlike = (
        pdt.is_object_dtype(dt) or isinstance(dt, pd.CategoricalDtype) or pdt.is_string_dtype(dt)
    )
    if fam == "str":
        return is_strlike
    if fam == "date":  # disk=str-like, 압축/재인코딩 후=datetime64 — 둘 다 OK
        return is_strlike or pdt.is_datetime64_any_dtype(dt)
    if fam == "float":
        return pdt.is_float_dtype(dt)
    if fam == "int":
        return pdt.is_integer_dtype(dt)
    if fam == "num":
        return pdt.is_numeric_dtype(dt)
    return True


def assert_contract(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """read 경계 경량 dtype 계약 검사 (할당 0). 위반 시 ValueError.

    매핑에 없는 이름은 통과(신규 슬롯 의도적 허용). 값/행이 아니라 컬럼
    존재 + dtype family 만 본다. 반환값은 입력 df 그대로(복사 안 함).
    """
    spec = _CONTRACTS.get(name)
    if spec is None:
        return df
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] read 계약 위반: 누락 컬럼 {missing}")
    bad = [
        f"{col}={df[col].dtype}(기대 {fam})"
        for col, fam in spec.items()
        if not _family_ok(df[col], fam)
    ]
    if bad:
        raise ValueError(f"[{name}] read 계약 위반: dtype {bad}")
    return df


__all__ = [
    "LineDailyFrame",
    "Band1dFrame",
    "Band1wFrame",
    "MarketPrices1d",
    "MarketIndicators1d",
    "validate",
    "assert_contract",
]
