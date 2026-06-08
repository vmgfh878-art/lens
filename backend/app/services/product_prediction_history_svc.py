from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from app.config import get_market_config
from app.core.exceptions import UpstreamUnavailableError
from pandas.api import types as pdt

# Render free tier (512MB) 에서 매 요청마다 11MB parquet 을 메모리에 로드하면
# 메모리 spike (~50-80MB) → OOM 으로 워커가 죽는 사례가 있었다. lru_cache 로
# 같은 경로의 파일은 한 번만 읽고 메모리 유지. parquet 갱신 시 admin/reload
# 가 cache 를 비워야 한다 (현재는 manual 재시작 필요).
_HISTORY_COLUMNS = [
    "ticker",
    "timeframe",
    "role",
    "asof_date",
    "display_date",
    "display_horizon",
    "run_id",
    "line_value",
    "lower_value",
    "upper_value",
]


ROOT_DIR = Path(__file__).resolve().parents[3]
# Render rootDir=backend 호환을 위해 backend/data/v1 우선, 기존 data/parquet 폴백 지원.
BACKEND_DIR = Path(__file__).resolve().parents[2]
V1_SNAPSHOT_DIR = BACKEND_DIR / "data" / "v1"
LEGACY_SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_SNAPSHOT_DIR = (
    V1_SNAPSHOT_DIR
    if (V1_SNAPSHOT_DIR / "product_prediction_history_1D.parquet").exists()
    else LEGACY_SNAPSHOT_DIR
)
PRODUCT_HISTORY_PARQUET_PATH = DEFAULT_SNAPSHOT_DIR / "product_prediction_history_1D.parquet"
PRODUCT_HISTORY_MANIFEST_PATH = DEFAULT_SNAPSHOT_DIR / "product_prediction_history_1D.manifest.json"
PRODUCT_HISTORY_RESPONSE_SOURCE = "product_rolling_replay"
SUPPORTED_TIMEFRAME = "1D"


def _snapshot_dir() -> Path:
    # product history는 raw parquet가 아니라 serving snapshot을 우선 사용해야 한다.
    if (V1_SNAPSHOT_DIR / "product_prediction_history_1D.parquet").exists():
        return V1_SNAPSHOT_DIR
    override = get_market_config().local_snapshot_dir
    if override:
        override_path = Path(override)
        if (override_path / "product_prediction_history_1D.parquet").exists():
            return override_path
    return LEGACY_SNAPSHOT_DIR


def _history_paths() -> tuple[Path, Path]:
    snapshot_dir = _snapshot_dir()
    parquet_path = PRODUCT_HISTORY_PARQUET_PATH
    manifest_path = PRODUCT_HISTORY_MANIFEST_PATH
    if snapshot_dir != DEFAULT_SNAPSHOT_DIR:
        parquet_path = snapshot_dir / "product_prediction_history_1D.parquet"
        manifest_path = snapshot_dir / "product_prediction_history_1D.manifest.json"
    return parquet_path, manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise UpstreamUnavailableError(
            "제품용 rolling prediction history manifest를 찾을 수 없습니다.",
            details={"path": str(path)},
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UpstreamUnavailableError(
            "제품용 rolling prediction history manifest JSON을 읽을 수 없습니다.",
            details={"path": str(path), "error": str(exc)},
        ) from exc


def _normalize_text(series: pd.Series, how: str) -> pd.Series:
    """ticker/timeframe/role 대소문자 정규화 — category 입력 시 object 미경유.

    CP246 — compact parquet 은 이 컬럼들을 category 로 저장한다. 기존
    `.astype(str).str.upper()` 는 category 를 597k+ object 로 되살려 cold load
    peak +270MB(product_history) → 512MB OOM 을 유발했다. category 면 **카테고리
    (수백 개)만** 변환해 category 를 유지한다. object 면 종전 경로 그대로.
    raw object 와 결과가 동일하다(데이터는 ticker 전부 upper / role 전부 lower /
    null 0 으로 검증됨).
    """
    upper = how == "upper"
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = series.cat.categories
        new_cats = cats.str.upper() if upper else cats.str.lower()
        if len(set(new_cats)) == len(new_cats):
            return series.cat.rename_categories(new_cats)
        # 변환 후 카테고리 충돌(드묾) — codes 매핑으로 처리(부분 object).
        mapping = dict(zip(cats, new_cats, strict=False))
        return series.map(mapping).astype("category")
    normalized = series.astype(str)
    return normalized.str.upper() if upper else normalized.str.lower()


@lru_cache(maxsize=2)
def _load_history_frame_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """parquet 을 한 번만 읽어 메모리 유지. mtime 이 바뀌면 자동 reload."""
    path = Path(path_str)
    try:
        frame = pd.read_parquet(path, columns=_HISTORY_COLUMNS)
    except ValueError:
        # 일부 schema 가 _HISTORY_COLUMNS 와 다를 수 있음 — fallback 전체 로드.
        frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["ticker"] = _normalize_text(frame["ticker"], "upper")
    frame["timeframe"] = _normalize_text(frame["timeframe"], "upper")
    frame["role"] = _normalize_text(frame["role"], "lower")
    if not pdt.is_datetime64_any_dtype(frame["asof_date"]):
        frame["asof_date"] = pd.to_datetime(frame["asof_date"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "timeframe", "role", "asof_date"])
    return _compress_history_dtypes(frame)


def _compress_history_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """CP238 — Render free tier(512MB) OOM 방지용 dtype 다이어트.

    product_history 780k행이 lru_cache 로 메모리에 상주하는데, object 문자열 +
    float64 라 ~330MB 를 점유해 OOM(자동 재시작 → 503)의 주원인이었다.
    category + float32 로 ~30MB (약 90% 절감) 까지 내려간다.

    회귀 안전성 (get_product_prediction_history_data / _apply_window / _line_history):
      - ticker / timeframe 은 `== str`, role 은 `.isin(set)` / `== str` 비교라
        categorical 에서도 동일하게 동작한다.
      - run_id 는 `.astype(str) == run_id` 로 비교하므로 categorical 무관.
      - display_date 는 categorical scalar(str) 로 보존되어 `pd.Timestamp(...).strftime`
        경로가 그대로다.
      - asof_date 는 이미 datetime64 라 정렬/cutoff 비교(Timedelta)와 충돌하지 않는다.
    """
    if frame.empty:
        return frame
    for col in ("ticker", "timeframe", "role", "run_id", "display_date"):
        if col in frame.columns:
            frame[col] = frame[col].astype("category")
    for col in ("line_value", "lower_value", "upper_value"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], downcast="float")
    if "display_horizon" in frame.columns:
        frame["display_horizon"] = pd.to_numeric(frame["display_horizon"], downcast="integer")
    return frame


def _load_history_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise UpstreamUnavailableError(
            "제품용 rolling prediction history parquet를 찾을 수 없습니다.",
            details={"path": str(path)},
        )
    try:
        return _load_history_frame_cached(str(path), path.stat().st_mtime)
    except Exception as exc:  # pragma: no cover - parquet 엔진 오류 메시지는 환경마다 다르다.
        raise UpstreamUnavailableError(
            "제품용 rolling prediction history parquet를 읽을 수 없습니다.",
            details={"path": str(path), "error": str(exc)},
        ) from exc


def clear_product_history_cache() -> None:
    """admin/reload 에서 호출해 lru_cache 비움."""
    _load_history_frame_cached.cache_clear()


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "line_run_id": manifest.get("line_run_id"),
        "band_run_id": manifest.get("band_run_id"),
        "date_range": {
            "start": manifest.get("asof_start"),
            "end": manifest.get("asof_end"),
        },
        "row_count": int(manifest.get("row_count") or 0),
    }


def _resolve_roles(raw_roles: str | None) -> set[str]:
    if raw_roles is None or raw_roles.strip().lower() in {"", "all"}:
        return {"line", "band"}
    roles = {role.strip().lower() for role in raw_roles.split(",") if role.strip()}
    invalid = roles - {"line", "band"}
    if invalid:
        raise ValueError("roles는 all, line, band 또는 line,band만 허용됩니다.")
    return roles or {"line", "band"}


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _empty_response(
    ticker: str, timeframe: str, manifest: dict[str, Any], *, reason: str
) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe.upper(),
        "latest_asof_date": None,
        "source": PRODUCT_HISTORY_RESPONSE_SOURCE,
        "line_history": [],
        "band_history": [],
        "manifest_summary": _manifest_summary(manifest),
        "empty_reason": reason,
    }


def _apply_window(
    frame: pd.DataFrame, *, limit: int | None, lookback_days: int | None
) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame
    if lookback_days is not None:
        latest = filtered["asof_date"].max()
        cutoff = latest - pd.Timedelta(days=lookback_days)
        filtered = filtered[filtered["asof_date"] >= cutoff]
    filtered = filtered.sort_values(["asof_date", "role", "run_id"])
    if limit is not None:
        # limit은 role별 최근 포인트 수로 적용해 line/band 길이가 서로 깨지지 않게 한다.
        filtered = filtered.groupby("role", group_keys=False).tail(limit)
    return filtered.sort_values(["asof_date", "role", "run_id"]).reset_index(drop=True)


def _line_history(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("asof_date").to_dict(orient="records"):
        value = _finite_or_none(row.get("line_value"))
        if value is None:
            continue
        forecast_date = row.get("display_date")
        rows.append(
            {
                "asof_date": pd.Timestamp(row["asof_date"]).strftime("%Y-%m-%d"),
                "forecast_date": pd.Timestamp(forecast_date).strftime("%Y-%m-%d")
                if forecast_date
                else None,
                "display_horizon": int(row["display_horizon"]),
                "value": value,
                "run_id": str(row["run_id"]),
            }
        )
    return rows


def _band_history(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("asof_date").to_dict(orient="records"):
        lower = _finite_or_none(row.get("lower_value"))
        upper = _finite_or_none(row.get("upper_value"))
        if lower is None or upper is None:
            continue
        forecast_date = row.get("display_date")
        rows.append(
            {
                "asof_date": pd.Timestamp(row["asof_date"]).strftime("%Y-%m-%d"),
                "forecast_date": pd.Timestamp(forecast_date).strftime("%Y-%m-%d")
                if forecast_date
                else None,
                "display_horizon": int(row["display_horizon"]),
                "lower": lower,
                "upper": upper,
                "run_id": str(row["run_id"]),
            }
        )
    return rows


def get_product_prediction_history_data(
    ticker: str,
    *,
    timeframe: str = SUPPORTED_TIMEFRAME,
    roles: str | None = "all",
    run_id: str | None = None,
    limit: int | None = None,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    parquet_path, manifest_path = _history_paths()
    manifest = _load_manifest(manifest_path)
    normalized_ticker = ticker.upper()
    normalized_timeframe = timeframe.upper()
    requested_roles = _resolve_roles(roles)

    if normalized_timeframe != SUPPORTED_TIMEFRAME:
        return _empty_response(
            normalized_ticker,
            normalized_timeframe,
            manifest,
            reason="unsupported_timeframe",
        )

    frame = _load_history_frame(parquet_path)
    frame = frame[
        (frame["ticker"] == normalized_ticker)
        & (frame["timeframe"] == normalized_timeframe)
        & (frame["role"].isin(requested_roles))
    ]
    if run_id:
        frame = frame[frame["run_id"].astype(str) == run_id]
    frame = _apply_window(frame, limit=limit, lookback_days=lookback_days)

    if frame.empty:
        return _empty_response(
            normalized_ticker,
            normalized_timeframe,
            manifest,
            reason="ticker_or_filter_has_no_product_history",
        )

    line_frame = frame[frame["role"] == "line"]
    band_frame = frame[frame["role"] == "band"]
    latest_asof_date = frame["asof_date"].max().strftime("%Y-%m-%d")
    return {
        "ticker": normalized_ticker,
        "timeframe": normalized_timeframe,
        "latest_asof_date": latest_asof_date,
        "source": PRODUCT_HISTORY_RESPONSE_SOURCE,
        "line_history": _line_history(line_frame),
        "band_history": _band_history(band_frame),
        "manifest_summary": _manifest_summary(manifest),
        "empty_reason": None,
    }
