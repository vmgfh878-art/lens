from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import UpstreamUnavailableError


ROOT_DIR = Path(__file__).resolve().parents[3]
# Render rootDir=backend 호환을 위해 backend/data/v1 우선, 기존 data/parquet 폴백 지원.
BACKEND_DIR = Path(__file__).resolve().parents[2]
V1_SNAPSHOT_DIR = BACKEND_DIR / "data" / "v1"
LEGACY_SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
DEFAULT_SNAPSHOT_DIR = V1_SNAPSHOT_DIR if (V1_SNAPSHOT_DIR / "product_prediction_history_1D.parquet").exists() else LEGACY_SNAPSHOT_DIR
PRODUCT_HISTORY_PARQUET_PATH = DEFAULT_SNAPSHOT_DIR / "product_prediction_history_1D.parquet"
PRODUCT_HISTORY_MANIFEST_PATH = DEFAULT_SNAPSHOT_DIR / "product_prediction_history_1D.manifest.json"
PRODUCT_HISTORY_RESPONSE_SOURCE = "product_rolling_replay"
SUPPORTED_TIMEFRAME = "1D"


def _snapshot_dir() -> Path:
    override = os.environ.get("LENS_LOCAL_SNAPSHOT_DIR")
    if override:
        return Path(override)
    # 우선 v1, 없으면 legacy
    if (V1_SNAPSHOT_DIR / "product_prediction_history_1D.parquet").exists():
        return V1_SNAPSHOT_DIR
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


def _load_history_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise UpstreamUnavailableError(
            "제품용 rolling prediction history parquet를 찾을 수 없습니다.",
            details={"path": str(path)},
        )
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - parquet 엔진 오류 메시지는 환경마다 다르다.
        raise UpstreamUnavailableError(
            "제품용 rolling prediction history parquet를 읽을 수 없습니다.",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["timeframe"] = frame["timeframe"].astype(str).str.upper()
    frame["role"] = frame["role"].astype(str).str.lower()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"], errors="coerce")
    return frame.dropna(subset=["ticker", "timeframe", "role", "asof_date"])


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


def _empty_response(ticker: str, timeframe: str, manifest: dict[str, Any], *, reason: str) -> dict[str, Any]:
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


def _apply_window(frame: pd.DataFrame, *, limit: int | None, lookback_days: int | None) -> pd.DataFrame:
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
        rows.append(
            {
                "asof_date": pd.Timestamp(row["asof_date"]).strftime("%Y-%m-%d"),
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
        rows.append(
            {
                "asof_date": pd.Timestamp(row["asof_date"]).strftime("%Y-%m-%d"),
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
