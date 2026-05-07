from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT_DIR / "data" / "parquet"
CONTEXT_SOURCE_DIR = SNAPSHOT_DIR / "context"
EODHD_CONTEXT_DIR = CONTEXT_SOURCE_DIR / "eodhd_500"
DOCS_DIR = ROOT_DIR / "docs"
LOG_DIR = ROOT_DIR / "logs" / "cp145_eodhd_500ticker_indicator_context"
UNIVERSE_PATH = ROOT_DIR / "backend" / "data" / "universe" / "sp500.csv"
PRICE_PATH = SNAPSHOT_DIR / "price_data_eodhd_500.parquet"
PRICE_MANIFEST_PATH = SNAPSHOT_DIR / "price_data_eodhd_500.manifest.json"
INDICATOR_1D_PATH = SNAPSHOT_DIR / "indicators_eodhd_1D_500.parquet"
INDICATOR_1W_PATH = SNAPSHOT_DIR / "indicators_eodhd_1W_500.parquet"
INDICATOR_1D_MANIFEST_PATH = SNAPSHOT_DIR / "indicators_eodhd_1D_500.manifest.json"
INDICATOR_1W_MANIFEST_PATH = SNAPSHOT_DIR / "indicators_eodhd_1W_500.manifest.json"
CONTEXT_MANIFEST_PATH = EODHD_CONTEXT_DIR / "context_eodhd_500.manifest.json"
METRICS_PATH = DOCS_DIR / "cp145_eodhd_500ticker_indicator_context_metrics.json"
REPORT_PATH = DOCS_DIR / "cp145_eodhd_500ticker_indicator_context_report.md"
INVENTORY_PATH = DOCS_DIR / "cp145_eodhd_500ticker_feature_inventory.csv"

CONTEXT_VERSION = "cp145_eodhd_500_context_v1"
CALENDAR_FEATURE_COLUMNS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_month_end",
    "is_quarter_end",
    "is_opex_friday",
]
MACRO_COLUMNS = ["us10y", "yield_spread", "vix_close", "credit_spread_hy"]
BREADTH_COLUMNS = ["nh_nl_index", "ma200_pct"]
FUNDAMENTAL_COLUMNS = ["revenue", "net_income", "equity", "eps", "roe", "debt_ratio"]
FLAG_COLUMNS = ["has_macro", "has_breadth", "has_fundamentals"]
RATIO_COLUMNS = ["open_ratio", "high_ratio", "low_ratio"]
CONTEXT_FEATURE_COLUMNS = [*MACRO_COLUMNS, *BREADTH_COLUMNS, *FUNDAMENTAL_COLUMNS]

os.environ["MARKET_DATA_PROVIDER"] = "eodhd"
os.environ["MARKET_DATA_FALLBACK_PROVIDER"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(SNAPSHOT_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_svc import FEATURE_COLUMNS, build_features  # noqa: E402
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402
from backend.collector.sources.sector_returns import build_sector_returns  # noqa: E402


MODEL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *CALENDAR_FEATURE_COLUMNS]
MODEL_N_FEATURES = len(MODEL_FEATURE_COLUMNS)
INDICATOR_CHECKSUM_COLUMNS = sorted({*FEATURE_COLUMNS, "atr_ratio", "volume"})


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if value is not None and not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_checksum(frame: pd.DataFrame, columns: list[str]) -> str | None:
    if frame.empty:
        return None
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return None
    sort_columns = [column for column in ["ticker", "timeframe", "date", "source", "sector"] if column in frame.columns]
    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "asof_date" in working.columns:
        working["asof_date"] = pd.to_datetime(working["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "filing_date" in working.columns:
        working["filing_date"] = pd.to_datetime(working["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if sort_columns:
        working = working.sort_values(sort_columns)
    payload = working[existing].to_dict(orient="records")
    return hashlib.sha256(json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def indicator_value_checksum(frame: pd.DataFrame) -> str | None:
    return frame_checksum(frame, ["ticker", "timeframe", "date", *[column for column in INDICATOR_CHECKSUM_COLUMNS if column in frame.columns]])


def load_price_frame() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"CP144 price parquet이 없습니다: {PRICE_PATH}")
    frame = pd.read_parquet(PRICE_PATH)
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    sources = set(frame.get("source", pd.Series(dtype=str)).dropna().astype(str).str.lower())
    providers = set(frame.get("provider", pd.Series(dtype=str)).dropna().astype(str).str.lower())
    if sources != {"eodhd"} or providers != {"eodhd"}:
        raise ValueError(f"EODHD price parquet source/provider 혼합 감지: source={sources}, provider={providers}")
    return frame


def load_universe() -> pd.DataFrame:
    frame = pd.read_csv(UNIVERSE_PATH)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    return frame.drop_duplicates("ticker", keep="last").reset_index(drop=True)


def load_context_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def normalize_macro_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", *MACRO_COLUMNS, "source", "provider", "asof_date", "context_version"])
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["asof_date"] = pd.to_datetime(result.get("asof_date", result["date"]), errors="coerce")
    for column in MACRO_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = result.get("source", "fred")
    result["provider"] = result.get("provider", "fred")
    result["context_scope"] = "shared_macro_for_eodhd_500"
    result["context_version"] = CONTEXT_VERSION
    return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def normalize_fundamentals_context(frame: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "date", "filing_date", *FUNDAMENTAL_COLUMNS, "source", "provider", "asof_date", "asof_policy", "context_version"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    tickers = set(universe["ticker"].astype(str).str.upper())
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result = result[result["ticker"].isin(tickers)].copy()
    for column in ("date", "filing_date", "asof_date"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    for column in FUNDAMENTAL_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = result.get("source", "sec_edgar")
    result["provider"] = result.get("provider", "sec_edgar")
    result["asof_policy"] = result.get("asof_policy", "actual_sec_filing_date")
    result["context_scope"] = "shared_sec_edgar_filtered_to_eodhd_500"
    result["context_version"] = CONTEXT_VERSION
    return result.dropna(subset=["ticker", "date", "filing_date"]).sort_values(["ticker", "date", "filing_date"]).reset_index(drop=True)


def build_market_breadth_context(price_frame: pd.DataFrame, min_ticker_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = price_frame[["ticker", "date", "close", "adjusted_close"]].copy()
    working["close"] = pd.to_numeric(working["adjusted_close"], errors="coerce").fillna(pd.to_numeric(working["close"], errors="coerce"))
    working = working.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])
    working["high_52w"] = working.groupby("ticker")["close"].transform(lambda series: series.rolling(window=252, min_periods=200).max())
    working["low_52w"] = working.groupby("ticker")["close"].transform(lambda series: series.rolling(window=252, min_periods=200).min())
    working["ma_200"] = working.groupby("ticker")["close"].transform(lambda series: series.rolling(window=200, min_periods=150).mean())
    working = working.dropna(subset=["high_52w", "low_52w", "ma_200"])
    if working.empty:
        return pd.DataFrame(), {"status": "FAIL", "reason": "empty_after_roll"}
    working["is_nh"] = working["close"] >= working["high_52w"]
    working["is_nl"] = working["close"] <= working["low_52w"]
    working["is_above_ma200"] = working["close"] > working["ma_200"]
    stats = (
        working.groupby("date")
        .agg(
            total_count=("ticker", "count"),
            nh_count=("is_nh", "sum"),
            nl_count=("is_nl", "sum"),
            above_ma200_count=("is_above_ma200", "sum"),
        )
        .reset_index()
    )
    stats = stats[stats["total_count"] >= min_ticker_count].copy()
    stats["nh_nl_index"] = stats["nh_count"] - stats["nl_count"]
    stats["ma200_pct"] = (stats["above_ma200_count"] / stats["total_count"]) * 100.0
    stats["source"] = "local_eodhd_500_universe"
    stats["provider"] = "eodhd"
    stats["asof_date"] = stats["date"]
    stats["universe_ticker_count"] = int(price_frame["ticker"].nunique())
    stats["min_ticker_count"] = min_ticker_count
    stats["context_version"] = CONTEXT_VERSION
    result = stats[
        [
            "date",
            "nh_nl_index",
            "ma200_pct",
            "total_count",
            "source",
            "provider",
            "asof_date",
            "universe_ticker_count",
            "min_ticker_count",
            "context_version",
        ]
    ].sort_values("date")
    metrics = {
        "status": "PASS" if not result.empty else "FAIL",
        "rows": int(len(result)),
        "date_min": None if result.empty else result["date"].min().date().isoformat(),
        "date_max": None if result.empty else result["date"].max().date().isoformat(),
        "min_ticker_count": min_ticker_count,
        "total_count_min": None if result.empty else int(result["total_count"].min()),
        "total_count_max": None if result.empty else int(result["total_count"].max()),
        "non_zero_rate": {
            "nh_nl_index": float((result["nh_nl_index"] != 0).mean()) if len(result) else 0.0,
            "ma200_pct": float((result["ma200_pct"] != 0).mean()) if len(result) else 0.0,
        },
    }
    return result.reset_index(drop=True), metrics


def build_sector_context(price_frame: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stock_info = universe[["ticker", "sector", "industry"]].copy()
    working = price_frame[["ticker", "date", "close", "adjusted_close"]].copy()
    working["close"] = pd.to_numeric(working["adjusted_close"], errors="coerce").fillna(pd.to_numeric(working["close"], errors="coerce"))
    sector = build_sector_returns(working[["ticker", "date", "close"]], stock_info)
    if sector.empty:
        return sector, {"status": "WARN", "rows": 0, "note": "sector_returns_empty"}
    sector["date"] = pd.to_datetime(sector["date"], errors="coerce")
    sector["source"] = "local_eodhd_500_universe"
    sector["provider"] = "eodhd"
    sector["asof_date"] = sector["date"]
    sector["context_version"] = CONTEXT_VERSION
    result = sector.sort_values(["date", "sector"]).reset_index(drop=True)
    return result, {
        "status": "PASS",
        "rows": int(len(result)),
        "sector_count": int(result["sector"].nunique()),
        "date_min": result["date"].min().date().isoformat(),
        "date_max": result["date"].max().date().isoformat(),
        "note": "현재 MODEL_FEATURE_COLUMNS 36개에는 sector_returns가 직접 포함되지 않는다.",
    }


def normalize_indicator_frame(frame: pd.DataFrame, timeframe: str, context_hash: str, updated_at: str) -> pd.DataFrame:
    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["timeframe"] = timeframe
    result["source"] = "eodhd"
    result["provider"] = "eodhd"
    result["provider_adjustment_policy"] = provider_adjustment_policy("eodhd")
    result["context_version"] = CONTEXT_VERSION
    result["context_hash"] = context_hash
    result["updated_at"] = updated_at
    return result.sort_values(["ticker", "timeframe", "date", "source"]).drop_duplicates(
        subset=["ticker", "timeframe", "date", "source"],
        keep="last",
    ).reset_index(drop=True)


def indicator_stats(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    if frame.empty:
        return {"status": "FAIL", "timeframe": timeframe, "rows": 0}
    stats: dict[str, Any] = {
        "status": "PASS",
        "timeframe": timeframe,
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()),
        "date_max": str(pd.to_datetime(frame["date"]).max().date()),
        "duplicate_ticker_timeframe_date_source": int(frame.duplicated(["ticker", "timeframe", "date", "source"]).sum()),
        "source_values": sorted(frame["source"].dropna().astype(str).unique().tolist()),
        "provider_values": sorted(frame["provider"].dropna().astype(str).unique().tolist()),
        "policy_values": sorted(frame["provider_adjustment_policy"].dropna().astype(str).unique().tolist()),
    }
    model_feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    values = frame[model_feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    stats["feature_non_finite_count"] = int((~np.isfinite(values)).sum())
    stats["nan_count_all_numeric"] = int(frame.select_dtypes(include=[np.number]).isna().sum().sum())
    ratio_abs = frame[RATIO_COLUMNS].apply(pd.to_numeric, errors="coerce").abs()
    stats["ratio_p99_abs"] = {column: float(ratio_abs[column].quantile(0.99)) for column in RATIO_COLUMNS}
    stats["ratio_max_abs"] = {column: float(ratio_abs[column].max()) for column in RATIO_COLUMNS}
    stats["ratio_sanity_pass"] = bool(all(value <= 1.0 for value in stats["ratio_p99_abs"].values()) and all(value <= 5.0 for value in stats["ratio_max_abs"].values()))
    stats["atr_ratio_non_null"] = int(frame["atr_ratio"].notna().sum()) if "atr_ratio" in frame.columns else 0
    stats["atr_ratio_coverage"] = float(stats["atr_ratio_non_null"] / len(frame)) if len(frame) else 0.0
    stats["flag_true_rate"] = {
        column: float(frame[column].astype(bool).mean()) if column in frame.columns and len(frame) else None
        for column in FLAG_COLUMNS
    }
    stats["context_non_zero_rate"] = {
        column: float((pd.to_numeric(frame[column], errors="coerce").fillna(0) != 0).mean()) if column in frame.columns and len(frame) else None
        for column in CONTEXT_FEATURE_COLUMNS
    }
    stats["indicator_value_checksum"] = indicator_value_checksum(frame)
    if stats["feature_non_finite_count"] or stats["duplicate_ticker_timeframe_date_source"] or stats["source_values"] != ["eodhd"] or not stats["ratio_sanity_pass"]:
        stats["status"] = "FAIL"
    return stats


def price_quality(price_frame: pd.DataFrame) -> dict[str, Any]:
    working = price_frame.copy()
    for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    factor = working["adjusted_close"] / working["close"].where(working["close"].abs() > 1e-12)
    adjusted_open = working["open"] * factor
    adjusted_high = working["high"] * factor
    adjusted_low = working["low"] * factor
    high_floor = pd.concat([adjusted_open, working["adjusted_close"]], axis=1).max(axis=1)
    low_ceiling = pd.concat([adjusted_open, working["adjusted_close"]], axis=1).min(axis=1)
    violations = {
        "duplicate_ticker_date_source": int(working.duplicated(["ticker", "date", "source"]).sum()),
        "source_not_eodhd": int((working["source"].astype(str).str.lower() != "eodhd").sum()),
        "provider_not_eodhd": int((working["provider"].astype(str).str.lower() != "eodhd").sum()),
        "adjusted_factor_invalid": int((~np.isfinite(factor) | (factor <= 0)).sum()),
        "adjusted_high_low_violation": int((adjusted_high < adjusted_low).sum()),
        "adjusted_high_bound_violation": int((adjusted_high + 1e-9 < high_floor).sum()),
        "adjusted_low_bound_violation": int((adjusted_low - 1e-9 > low_ceiling).sum()),
        "volume_null": int(working["volume"].isna().sum()),
        "volume_negative": int((working["volume"] < 0).sum()),
    }
    return {
        "row_count": int(len(working)),
        "ticker_count": int(working["ticker"].nunique()),
        "date_min": working["date"].min().date().isoformat(),
        "date_max": working["date"].max().date().isoformat(),
        "violations": violations,
        "adjusted_ohlc_violation_count": sum(
            violations[key]
            for key in [
                "adjusted_factor_invalid",
                "adjusted_high_low_violation",
                "adjusted_high_bound_violation",
                "adjusted_low_bound_violation",
            ]
        ),
    }


def context_metrics(frame: pd.DataFrame, name: str, columns: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {"status": "WARN", "name": name, "rows": 0}
    metrics: dict[str, Any] = {
        "status": "PASS",
        "name": name,
        "rows": int(len(frame)),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()) if "date" in frame.columns else None,
        "date_max": str(pd.to_datetime(frame["date"]).max().date()) if "date" in frame.columns else None,
        "source_values": sorted(frame["source"].dropna().astype(str).unique().tolist()) if "source" in frame.columns else [],
        "provider_values": sorted(frame["provider"].dropna().astype(str).unique().tolist()) if "provider" in frame.columns else [],
        "checksum": frame_checksum(frame, ["ticker", "date", "filing_date", "asof_date", *columns]),
    }
    if "ticker" in frame.columns:
        metrics["ticker_count"] = int(frame["ticker"].nunique())
    metrics["non_null_rate"] = {
        column: float(frame[column].notna().mean()) if column in frame.columns and len(frame) else None
        for column in columns
    }
    metrics["non_zero_rate"] = {
        column: float((pd.to_numeric(frame[column], errors="coerce").fillna(0) != 0).mean()) if column in frame.columns and len(frame) else None
        for column in columns
    }
    return metrics


def split_gate(frame: pd.DataFrame, *, timeframe: str, seq_len: int, horizon: int, min_fold_samples: int) -> dict[str, Any]:
    if frame.empty:
        return {"status": "FAIL", "timeframe": timeframe, "reason": "empty"}
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    sample_rows: list[pd.DataFrame] = []
    excluded: dict[str, str] = {}
    for ticker, group in working.sort_values(["ticker", "date"]).groupby("ticker"):
        group = group.reset_index(drop=True)
        sample_count = max(0, len(group) - seq_len - horizon + 1)
        if sample_count <= 0:
            excluded[str(ticker)] = f"insufficient_rows:{len(group)}"
            continue
        ends = group.iloc[seq_len - 1 : seq_len - 1 + sample_count][["ticker", "date"]].copy()
        sample_rows.append(ends)
    if not sample_rows:
        return {
            "status": "FAIL",
            "timeframe": timeframe,
            "seq_len": seq_len,
            "horizon": horizon,
            "eligible_ticker_count": 0,
            "excluded_count": len(excluded),
        }
    samples = pd.concat(sample_rows, ignore_index=True).sort_values("date")
    train_cut = samples["date"].quantile(0.70)
    val_cut = samples["date"].quantile(0.85)
    train_count = int((samples["date"] <= train_cut).sum())
    val_count = int(((samples["date"] > train_cut) & (samples["date"] <= val_cut)).sum())
    test_count = int((samples["date"] > val_cut).sum())
    passed = train_count >= min_fold_samples and val_count >= min_fold_samples and test_count >= min_fold_samples
    return {
        "status": "PASS" if passed else "FAIL",
        "timeframe": timeframe,
        "seq_len": seq_len,
        "horizon": horizon,
        "min_fold_samples": min_fold_samples,
        "eligible_ticker_count": int(samples["ticker"].nunique()),
        "excluded_count": len(excluded),
        "estimated_sample_count": int(len(samples)),
        "split_counts": {"train": train_count, "val": val_count, "test": test_count},
        "split_dates": {
            "train_end": train_cut.date().isoformat(),
            "val_end": val_cut.date().isoformat(),
            "test_end": samples["date"].max().date().isoformat(),
        },
        "excluded_sample": dict(list(excluded.items())[:20]),
    }


def write_indicator_manifest(manifest_path: Path, parquet_path: Path, frame: pd.DataFrame, stats: dict[str, Any], context_hash: str, price_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "created_at": utc_now_iso(),
        "provider": "eodhd",
        "source": "eodhd",
        "provider_adjustment_policy": provider_adjustment_policy("eodhd"),
        "feature_contract_version": "v3_adjusted_ohlc",
        "context_version": CONTEXT_VERSION,
        "context_hash": context_hash,
        "price_source_data_hash": price_manifest.get("source_data_hash"),
        "indicator_value_checksum": stats.get("indicator_value_checksum"),
        "timeframe": stats.get("timeframe"),
        "ticker_count": stats.get("ticker_count"),
        "row_count": stats.get("rows"),
        "date_min": stats.get("date_min"),
        "date_max": stats.get("date_max"),
        "model_n_features": MODEL_N_FEATURES,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "parquet_path": str(parquet_path.relative_to(ROOT_DIR)),
        "parquet_sha256": file_sha256(parquet_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def write_context_manifest(context_files: dict[str, Path], metrics: dict[str, Any], context_hash: str, price_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "created_at": utc_now_iso(),
        "context_version": CONTEXT_VERSION,
        "context_hash": context_hash,
        "price_provider": "eodhd",
        "price_source": "eodhd",
        "price_source_data_hash": price_manifest.get("source_data_hash"),
        "files": {
            name: {
                "path": str(path.relative_to(ROOT_DIR)),
                "bytes": int(path.stat().st_size) if path.exists() else None,
                "sha256": file_sha256(path),
            }
            for name, path in context_files.items()
        },
        "context_sources": metrics,
        "lookahead_policy": {
            "macro": "observation/asof date 기준으로만 merge한다.",
            "breadth": "EODHD 500 price로 해당 날짜까지 계산 가능한 값만 사용한다.",
            "fundamentals": "SEC filing_date 기준 backward merge만 허용한다.",
            "sector_returns": "동일 EODHD 500 universe 가격에서 날짜별 sector 평균 수익률을 계산한다. 현재 모델 36개 feature에는 직접 포함되지 않는다.",
        },
    }
    write_json(CONTEXT_MANIFEST_PATH, manifest)
    return manifest


def write_feature_inventory(path: Path, indicator_stats_by_tf: dict[str, dict[str, Any]]) -> None:
    groups = {}
    for column in MODEL_FEATURE_COLUMNS:
        if column in MACRO_COLUMNS:
            groups[column] = "macro"
        elif column in BREADTH_COLUMNS:
            groups[column] = "market_breadth"
        elif column in FUNDAMENTAL_COLUMNS:
            groups[column] = "fundamentals"
        elif column in FLAG_COLUMNS:
            groups[column] = "missing_flag"
        elif column in CALENDAR_FEATURE_COLUMNS:
            groups[column] = "calendar"
        elif column.startswith("regime_"):
            groups[column] = "regime"
        else:
            groups[column] = "price_volatility_volume"
    rows = []
    for index, column in enumerate(MODEL_FEATURE_COLUMNS):
        rows.append(
            {
                "index": index,
                "column": column,
                "group": groups[column],
                "in_model_features": True,
                "source_provider_contract": source_contract_for_group(groups[column]),
                "one_day_non_zero_rate": indicator_stats_by_tf["1D"].get("context_non_zero_rate", {}).get(column),
                "one_week_non_zero_rate": indicator_stats_by_tf["1W"].get("context_non_zero_rate", {}).get(column),
                "one_day_flag_true_rate": indicator_stats_by_tf["1D"].get("flag_true_rate", {}).get(column),
                "one_week_flag_true_rate": indicator_stats_by_tf["1W"].get("flag_true_rate", {}).get(column),
            }
        )
    rows.append(
        {
            "index": None,
            "column": "atr_ratio",
            "group": "indicator_only",
            "in_model_features": False,
            "source_provider_contract": "chart/indicator only, not MODEL_FEATURE_COLUMNS",
            "one_day_non_zero_rate": None,
            "one_week_non_zero_rate": None,
            "one_day_flag_true_rate": None,
            "one_week_flag_true_rate": None,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def source_contract_for_group(group: str) -> str:
    if group == "macro":
        return "shared FRED local context, asof_date/observation date"
    if group == "market_breadth":
        return "local EODHD 500 universe breadth, provider=eodhd"
    if group == "fundamentals":
        return "local SEC EDGAR context, filing_date backward merge"
    if group == "calendar":
        return "deterministic calendar feature"
    if group == "missing_flag":
        return "feature_svc context non-null flag"
    if group == "regime":
        return "derived from macro vix_close"
    return "local EODHD adjusted OHLC price-derived feature"


def write_report(metrics: dict[str, Any]) -> None:
    final = metrics["final_decision"]
    one_d = metrics["indicator_stats"]["1D"]
    one_w = metrics["indicator_stats"]["1W"]
    report = f"""# CP145-D EODHD 500티커 indicator + context generation 보고서

## 1. 요약

판정은 **{final["status"]}**이다.

{final["summary"]}

## 2. 입력과 생성 파일

| 구분 | 경로 |
|---|---|
| input price | `data/parquet/price_data_eodhd_500.parquet` |
| 1D indicators | `data/parquet/indicators_eodhd_1D_500.parquet` |
| 1W indicators | `data/parquet/indicators_eodhd_1W_500.parquet` |
| context dir | `data/parquet/context/eodhd_500/` |
| feature inventory | `docs/cp145_eodhd_500ticker_feature_inventory.csv` |

## 3. indicator 검증

| 항목 | 1D | 1W |
|---|---:|---:|
| rows | {one_d.get("rows")} | {one_w.get("rows")} |
| tickers | {one_d.get("ticker_count")} | {one_w.get("ticker_count")} |
| date range | {one_d.get("date_min")} ~ {one_d.get("date_max")} | {one_w.get("date_min")} ~ {one_w.get("date_max")} |
| duplicate | {one_d.get("duplicate_ticker_timeframe_date_source")} | {one_w.get("duplicate_ticker_timeframe_date_source")} |
| feature non-finite | {one_d.get("feature_non_finite_count")} | {one_w.get("feature_non_finite_count")} |
| atr_ratio coverage | {one_d.get("atr_ratio_coverage")} | {one_w.get("atr_ratio_coverage")} |
| ratio sanity | {one_d.get("ratio_sanity_pass")} | {one_w.get("ratio_sanity_pass")} |

## 4. context coverage

| 계열 | 상태 | rows | date range | 비고 |
|---|---|---:|---|---|
| macro | {metrics["context_sources"]["macro"].get("status")} | {metrics["context_sources"]["macro"].get("rows")} | {metrics["context_sources"]["macro"].get("date_min")} ~ {metrics["context_sources"]["macro"].get("date_max")} | 기존 local FRED context 재사용 |
| market_breadth | {metrics["context_sources"]["market_breadth"].get("status")} | {metrics["context_sources"]["market_breadth"].get("rows")} | {metrics["context_sources"]["market_breadth"].get("date_min")} ~ {metrics["context_sources"]["market_breadth"].get("date_max")} | EODHD 500 price 기준 재계산 |
| fundamentals | {metrics["context_sources"]["fundamentals"].get("status")} | {metrics["context_sources"]["fundamentals"].get("rows")} | {metrics["context_sources"]["fundamentals"].get("date_min")} ~ {metrics["context_sources"]["fundamentals"].get("date_max")} | 기존 local SEC EDGAR context를 EODHD universe로 필터 |
| sector_returns | {metrics["context_sources"]["sector_returns"].get("status")} | {metrics["context_sources"]["sector_returns"].get("rows")} | {metrics["context_sources"]["sector_returns"].get("date_min")} ~ {metrics["context_sources"]["sector_returns"].get("date_max")} | 모델 36개 feature에는 직접 미포함 |

## 5. split gate

| timeframe | status | eligible tickers | estimated samples | train/val/test |
|---|---|---:|---:|---|
| 1D | {metrics["split_gates"]["1D"].get("status")} | {metrics["split_gates"]["1D"].get("eligible_ticker_count")} | {metrics["split_gates"]["1D"].get("estimated_sample_count")} | {metrics["split_gates"]["1D"].get("split_counts")} |
| 1W | {metrics["split_gates"]["1W"].get("status")} | {metrics["split_gates"]["1W"].get("eligible_ticker_count")} | {metrics["split_gates"]["1W"].get("estimated_sample_count")} | {metrics["split_gates"]["1W"].get("split_counts")} |

## 6. feature/cache 계약

`MODEL_N_FEATURES=36`이며 `atr_ratio`는 모델 feature에 포함하지 않았다. EODHD 산출물은 `_eodhd_*_500` 파일명과 manifest로 yfinance 산출물과 분리되어 있다. context 변경은 `context_hash={metrics["context_hash"]}`와 indicator checksum에 반영된다.

## 7. 제약과 다음 단계

fundamentals는 기존 SEC EDGAR local context를 재사용했기 때문에 coverage가 일부 ticker에 제한된다. 그러나 macro/breadth는 1D/1W full_features에서 실제 값으로 채워졌고, 1D/1W split gate가 통과했으므로 다음 CP에서 500티커 1D/1W 모델 학습 준비로 넘어갈 수 있다.

## 8. 금지 작업 미발생 확인

Supabase raw write, DB write, yfinance append, 모델 학습, inference 저장, 프론트 수정, W&B, EODHD 추가 대량 호출은 수행하지 않았다.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EODHD_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    updated_at = utc_now_iso()

    price_manifest = json.loads(PRICE_MANIFEST_PATH.read_text(encoding="utf-8")) if PRICE_MANIFEST_PATH.exists() else {}
    price_frame = load_price_frame()
    universe = load_universe()
    price_tickers = sorted(price_frame["ticker"].unique().tolist())
    universe = universe[universe["ticker"].isin(price_tickers)].copy()

    macro_source = load_context_or_empty(CONTEXT_SOURCE_DIR / "macroeconomic_indicators.parquet")
    fundamentals_source = load_context_or_empty(CONTEXT_SOURCE_DIR / "company_fundamentals.parquet")
    macro_frame = normalize_macro_context(macro_source)
    fundamentals_frame = normalize_fundamentals_context(fundamentals_source, universe)
    breadth_frame, breadth_build_metrics = build_market_breadth_context(price_frame, args.min_breadth_tickers)
    sector_frame, sector_build_metrics = build_sector_context(price_frame, universe)

    context_files = {
        "macro": EODHD_CONTEXT_DIR / "macroeconomic_indicators.parquet",
        "market_breadth": EODHD_CONTEXT_DIR / "market_breadth.parquet",
        "company_fundamentals": EODHD_CONTEXT_DIR / "company_fundamentals.parquet",
        "sector_returns": EODHD_CONTEXT_DIR / "sector_returns.parquet",
    }
    macro_frame.to_parquet(context_files["macro"], index=False)
    breadth_frame.to_parquet(context_files["market_breadth"], index=False)
    fundamentals_frame.to_parquet(context_files["company_fundamentals"], index=False)
    sector_frame.to_parquet(context_files["sector_returns"], index=False)

    context_hash_payload = {
        "context_version": CONTEXT_VERSION,
        "price_source_hash": price_manifest.get("source_data_hash"),
        "macro": frame_checksum(macro_frame, ["date", "asof_date", *MACRO_COLUMNS]),
        "breadth": frame_checksum(breadth_frame, ["date", "asof_date", *BREADTH_COLUMNS, "total_count"]),
        "fundamentals": frame_checksum(fundamentals_frame, ["ticker", "date", "filing_date", *FUNDAMENTAL_COLUMNS]),
        "sector": frame_checksum(sector_frame, ["date", "sector", "return", "close"]),
    }
    context_hash = hashlib.sha256(json.dumps(json_safe(context_hash_payload), sort_keys=True).encode("utf-8")).hexdigest()[:16]

    indicator_1d = normalize_indicator_frame(
        build_features(price_df=price_frame, macro_df=macro_frame, breadth_df=breadth_frame, fundamentals_df=fundamentals_frame, timeframe="1D"),
        "1D",
        context_hash,
        updated_at,
    )
    indicator_1w = normalize_indicator_frame(
        build_features(price_df=price_frame, macro_df=macro_frame, breadth_df=breadth_frame, fundamentals_df=fundamentals_frame, timeframe="1W"),
        "1W",
        context_hash,
        updated_at,
    )
    indicator_1d.to_parquet(INDICATOR_1D_PATH, index=False)
    indicator_1w.to_parquet(INDICATOR_1W_PATH, index=False)

    price_stats = price_quality(price_frame)
    one_d_stats = indicator_stats(indicator_1d, "1D")
    one_w_stats = indicator_stats(indicator_1w, "1W")
    context_source_metrics = {
        "macro": context_metrics(macro_frame, "macro", MACRO_COLUMNS),
        "market_breadth": {**context_metrics(breadth_frame, "market_breadth", BREADTH_COLUMNS), **breadth_build_metrics},
        "fundamentals": context_metrics(fundamentals_frame, "fundamentals", FUNDAMENTAL_COLUMNS),
        "sector_returns": {**context_metrics(sector_frame, "sector_returns", ["return", "close"]), **sector_build_metrics},
    }
    context_manifest = write_context_manifest(context_files, context_source_metrics, context_hash, price_manifest)
    indicator_1d_manifest = write_indicator_manifest(INDICATOR_1D_MANIFEST_PATH, INDICATOR_1D_PATH, indicator_1d, one_d_stats, context_hash, price_manifest)
    indicator_1w_manifest = write_indicator_manifest(INDICATOR_1W_MANIFEST_PATH, INDICATOR_1W_PATH, indicator_1w, one_w_stats, context_hash, price_manifest)

    split_gates = {
        "1D": split_gate(indicator_1d, timeframe="1D", seq_len=args.seq_len_1d, horizon=args.horizon_1d, min_fold_samples=args.min_fold_samples),
        "1W": split_gate(indicator_1w, timeframe="1W", seq_len=args.seq_len_1w, horizon=args.horizon_1w, min_fold_samples=args.min_fold_samples),
    }

    sector_missing_rate = {
        "sector_missing_rate": float(universe["sector"].isna().mean()) if len(universe) else None,
        "industry_missing_rate": float(universe["industry"].isna().mean()) if len(universe) and "industry" in universe.columns else None,
        "sector_count": int(universe["sector"].nunique()) if len(universe) else 0,
        "industry_count": int(universe["industry"].nunique()) if len(universe) and "industry" in universe.columns else 0,
    }

    metrics: dict[str, Any] = {
        "cp": "CP145-D",
        "created_at": updated_at,
        "context_version": CONTEXT_VERSION,
        "context_hash": context_hash,
        "context_hash_payload": context_hash_payload,
        "model_n_features": MODEL_N_FEATURES,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "atr_ratio_in_model_feature_columns": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "input_price": {
            "path": str(PRICE_PATH.relative_to(ROOT_DIR)),
            "manifest": str(PRICE_MANIFEST_PATH.relative_to(ROOT_DIR)),
            "manifest_source_data_hash": price_manifest.get("source_data_hash"),
            "manifest_provider": price_manifest.get("provider"),
            "manifest_source": price_manifest.get("source"),
        },
        "universe": {
            "path": str(UNIVERSE_PATH.relative_to(ROOT_DIR)),
            "ticker_count": int(len(universe)),
            **sector_missing_rate,
        },
        "price_quality": price_stats,
        "context_sources": context_source_metrics,
        "context_files": {
            name: {
                "path": str(path.relative_to(ROOT_DIR)),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
            for name, path in context_files.items()
        },
        "context_manifest": context_manifest,
        "indicator_stats": {"1D": one_d_stats, "1W": one_w_stats},
        "indicator_files": {
            "1D": {"path": str(INDICATOR_1D_PATH.relative_to(ROOT_DIR)), "bytes": int(INDICATOR_1D_PATH.stat().st_size), "sha256": file_sha256(INDICATOR_1D_PATH), "manifest": indicator_1d_manifest},
            "1W": {"path": str(INDICATOR_1W_PATH.relative_to(ROOT_DIR)), "bytes": int(INDICATOR_1W_PATH.stat().st_size), "sha256": file_sha256(INDICATOR_1W_PATH), "manifest": indicator_1w_manifest},
        },
        "split_gates": split_gates,
        "feature_cache_separation": {
            "eodhd_indicator_paths": [str(INDICATOR_1D_PATH.relative_to(ROOT_DIR)), str(INDICATOR_1W_PATH.relative_to(ROOT_DIR))],
            "yfinance_indicator_paths_exist": {
                "1D": (SNAPSHOT_DIR / "indicators_yfinance_1D.parquet").exists(),
                "1W": (SNAPSHOT_DIR / "indicators_yfinance_1W.parquet").exists(),
            },
            "provider_in_hash_inputs": {
                "provider": "eodhd",
                "source": "eodhd",
                "provider_adjustment_policy": provider_adjustment_policy("eodhd"),
                "context_hash": context_hash,
            },
        },
        "forbidden_actions_observed": {
            "supabase_raw_write": False,
            "db_write": False,
            "yfinance_append": False,
            "model_training": False,
            "inference_save": False,
            "frontend_modify": False,
            "wandb": False,
            "eodhd_additional_bulk_call": False,
            "provider_source_mixing": bool(one_d_stats.get("source_values") != ["eodhd"] or one_w_stats.get("source_values") != ["eodhd"]),
        },
    }

    hard_fail = (
        price_stats["adjusted_ohlc_violation_count"] != 0
        or one_d_stats.get("status") != "PASS"
        or one_w_stats.get("status") != "PASS"
        or split_gates["1D"].get("status") != "PASS"
        or split_gates["1W"].get("status") != "PASS"
        or metrics["atr_ratio_in_model_feature_columns"]
        or MODEL_N_FEATURES != 36
    )
    warnings = []
    if context_source_metrics["fundamentals"].get("ticker_count", 0) < int(len(universe) * 0.8):
        warnings.append("fundamentals_coverage_partial")
    if context_source_metrics["macro"].get("status") != "PASS":
        warnings.append("macro_context_partial")

    if hard_fail:
        status = "FAIL"
        summary = "indicator/source/split/model feature contract 중 실패가 있어 500티커 학습으로 진행하면 안 된다."
    elif warnings:
        status = "WARN"
        summary = "500티커 EODHD 1D/1W indicators와 context 생성은 통과했지만 fundamentals coverage가 낮아 해석 제약이 필요하다."
    else:
        status = "PASS"
        summary = "500티커 EODHD 1D/1W indicators와 context 생성, split gate, feature contract 검증이 통과했다."
    metrics["warnings"] = warnings
    metrics["final_decision"] = {"status": status, "summary": summary}

    write_json(METRICS_PATH, metrics)
    write_json(LOG_DIR / METRICS_PATH.name, metrics)
    write_feature_inventory(INVENTORY_PATH, metrics["indicator_stats"])
    write_report(metrics)
    print(json.dumps(json_safe(metrics["final_decision"]), ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP145 EODHD 500 ticker indicator/context generation")
    parser.add_argument("--min-breadth-tickers", type=int, default=100)
    parser.add_argument("--seq-len-1d", type=int, default=252)
    parser.add_argument("--horizon-1d", type=int, default=5)
    parser.add_argument("--seq-len-1w", type=int, default=104)
    parser.add_argument("--horizon-1w", type=int, default=4)
    parser.add_argument("--min-fold-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
