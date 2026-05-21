from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch()  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.loss import ForecastCompositeLoss  # noqa: E402
from ai.models.cnn_lstm import CNNLSTM  # noqa: E402
from ai.models.patchtst import PatchTST  # noqa: E402
from ai.models.tcn_quantile import TCNQuantile  # noqa: E402
from ai.models.tide import TiDE  # noqa: E402
from ai.preprocessing import (  # noqa: E402
    FEATURE_CONTRACT_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_N_FEATURES,
    SOURCE_FEATURE_COLUMNS,
    append_calendar_features,
)
from ai.splits import MAX_HORIZON_BY_TIMEFRAME, build_split_specs  # noqa: E402
from backend.collector.sources.market_data_providers import provider_adjustment_policy  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "parquet"
CACHE_DIR = PROJECT_ROOT / "ai" / "cache"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "docs" / "cp148_0_model_lineup_preflight_metrics.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "cp148_0_model_lineup_preflight_report.md"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "docs" / "cp148_0_model_lineup_experiment_plan.md"
FEATURE_SET_PLAN_PATH = PROJECT_ROOT / "docs" / "cp63_bm_feature_set_plan.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def frame_summary(frame: pd.DataFrame, *, duplicate_subset: list[str]) -> dict[str, Any]:
    source_values = sorted(frame["source"].dropna().astype(str).str.lower().unique().tolist()) if "source" in frame.columns else []
    provider_values = sorted(frame["provider"].dropna().astype(str).str.lower().unique().tolist()) if "provider" in frame.columns else []
    policy_values = (
        sorted(frame["provider_adjustment_policy"].dropna().astype(str).unique().tolist())
        if "provider_adjustment_policy" in frame.columns
        else []
    )
    return {
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "date_min": frame["date"].min().strftime("%Y-%m-%d") if "date" in frame.columns and frame["date"].notna().any() else None,
        "date_max": frame["date"].max().strftime("%Y-%m-%d") if "date" in frame.columns and frame["date"].notna().any() else None,
        "source_values": source_values,
        "provider_values": provider_values,
        "provider_adjustment_policy_values": policy_values,
        "duplicate_ticker_date_source_count": int(frame.duplicated(subset=[column for column in duplicate_subset if column in frame.columns]).sum())
        if all(column in frame.columns for column in duplicate_subset)
        else None,
    }


def adjusted_ohlc_summary(price: pd.DataFrame) -> dict[str, Any]:
    required = ["open", "high", "low", "close", "adjusted_close"]
    if any(column not in price.columns for column in required):
        return {"available": False, "violation_count": None}
    numeric = price[required].apply(pd.to_numeric, errors="coerce")
    factor = numeric["adjusted_close"] / numeric["close"].replace(0, np.nan)
    adjusted_open = numeric["open"] * factor
    adjusted_high = numeric["high"] * factor
    adjusted_low = numeric["low"] * factor
    adjusted_close = numeric["adjusted_close"]
    finite_required = np.isfinite(np.column_stack([adjusted_open, adjusted_high, adjusted_low, adjusted_close, factor]))
    finite_all = finite_required.all(axis=1)
    high_floor = np.maximum.reduce([adjusted_open.to_numpy(), adjusted_low.to_numpy(), adjusted_close.to_numpy()])
    low_ceiling = np.minimum.reduce([adjusted_open.to_numpy(), adjusted_high.to_numpy(), adjusted_close.to_numpy()])
    violations = finite_all & ((adjusted_high.to_numpy() + 1e-8 < high_floor) | (adjusted_low.to_numpy() - 1e-8 > low_ceiling))
    return {
        "available": True,
        "finite_factor_count": int(np.isfinite(factor.to_numpy()).sum()),
        "nonfinite_factor_count": int((~np.isfinite(factor.to_numpy())).sum()),
        "adjusted_ohlc_violation_count": int(violations.sum()),
        "adjusted_factor_min": float(np.nanmin(factor.to_numpy())),
        "adjusted_factor_max": float(np.nanmax(factor.to_numpy())),
    }


def finite_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    present = [column for column in columns if column in frame.columns]
    missing = [column for column in columns if column not in frame.columns]
    if not present:
        return {"present_columns": 0, "missing_columns": missing, "nan_count": None, "inf_count": None}
    numeric = frame[present].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype="float64", copy=False)
    return {
        "present_columns": len(present),
        "missing_columns": missing,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "nonfinite_count": int((~np.isfinite(values)).sum()),
    }


def ratio_extremes(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ["open_ratio", "high_ratio", "low_ratio", "atr_ratio", "log_return", "vol_change"]:
        if column not in frame.columns:
            result[column] = {"exists": False}
            continue
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        abs_series = series.abs()
        result[column] = {
            "exists": True,
            "p99_abs": float(abs_series.quantile(0.99)) if not abs_series.empty else None,
            "max_abs": float(abs_series.max()) if not abs_series.empty else None,
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
        }
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        result["volume"] = {
            "exists": True,
            "p99": float(volume.quantile(0.99)) if not volume.empty else None,
            "max": float(volume.max()) if not volume.empty else None,
        }
    else:
        result["volume"] = {"exists": False}
    return result


def coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ["has_macro", "has_breadth", "has_fundamentals"]:
        if column not in frame.columns:
            result[column] = {"exists": False}
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        result[column] = {
            "exists": True,
            "mean": float(values.mean()),
            "true_rows": int((values > 0).sum()),
            "false_rows": int((values <= 0).sum()),
        }
    fundamental_columns = ["revenue", "net_income", "equity", "eps", "roe", "debt_ratio"]
    present = [column for column in fundamental_columns if column in frame.columns]
    if present:
        values = frame[present].apply(pd.to_numeric, errors="coerce")
        nonzero_any = (values.fillna(0).abs().sum(axis=1) > 0)
        result["fundamental_nonzero_any"] = {
            "present_columns": present,
            "ratio": float(nonzero_any.mean()),
            "rows": int(nonzero_any.sum()),
        }
    return result


def append_model_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    minimal = frame.copy()
    return append_calendar_features(minimal)


def _target_price_frame(price: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    close = price[["ticker", "date", "adjusted_close"]].copy()
    close["adjusted_close"] = pd.to_numeric(close["adjusted_close"], errors="coerce")
    if timeframe == "1D":
        return close
    if timeframe == "1W":
        weekly_frames: list[pd.DataFrame] = []
        for ticker, ticker_frame in close.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
            weekly = (
                ticker_frame.set_index("date")["adjusted_close"]
                .resample("W-FRI")
                .last()
                .dropna()
                .reset_index()
            )
            weekly["ticker"] = ticker
            weekly_frames.append(weekly[["ticker", "date", "adjusted_close"]])
        return pd.concat(weekly_frames, ignore_index=True) if weekly_frames else close.iloc[0:0].copy()
    return close


def target_finite_summary(indicators: pd.DataFrame, price: pd.DataFrame, *, timeframe: str, horizon: int) -> dict[str, Any]:
    close = _target_price_frame(price, timeframe)
    frame = indicators[["ticker", "date"]].merge(close, on=["ticker", "date"], how="left")
    target_values: list[np.ndarray] = []
    tail_missing = 0
    anchor_missing = int(frame["adjusted_close"].isna().sum())
    for _, ticker_frame in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        closes = ticker_frame["adjusted_close"].to_numpy(dtype="float64")
        if len(closes) <= horizon:
            tail_missing += len(closes)
            continue
        anchors = closes[:-horizon]
        future = np.column_stack([closes[step : step + len(anchors)] for step in range(1, horizon + 1)])
        returns = (future / anchors[:, None]) - 1.0
        target_values.append(returns.reshape(-1))
        tail_missing += horizon
    if not target_values:
        return {
            "horizon": horizon,
            "target_count": 0,
            "nonfinite_count": None,
            "tail_missing_rows": tail_missing,
            "anchor_missing_rows": anchor_missing,
        }
    values = np.concatenate(target_values)
    return {
        "horizon": horizon,
        "target_count": int(values.size),
        "nonfinite_count": int((~np.isfinite(values)).sum()),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "tail_missing_rows": int(tail_missing),
        "anchor_missing_rows": anchor_missing,
        "p99_abs": float(np.nanquantile(np.abs(values[np.isfinite(values)]), 0.99)) if np.isfinite(values).any() else None,
        "max_abs": float(np.nanmax(np.abs(values[np.isfinite(values)]))) if np.isfinite(values).any() else None,
    }


def split_summary(indicators: pd.DataFrame, *, timeframe: str, seq_len: int, horizon: int) -> dict[str, Any]:
    h_max = MAX_HORIZON_BY_TIMEFRAME[timeframe]
    frame = indicators[["ticker", "timeframe", "date"]].copy()
    specs, excluded = build_split_specs(
        frame,
        timeframe=timeframe,
        seq_len=seq_len,
        h_max=h_max,
        min_fold_samples=50,
    )
    overlap_count = 0
    purge_gap_ok = True
    min_fold_ok = True
    sample_count = 0
    train_samples = 0
    val_samples = 0
    test_samples = 0
    for spec in specs.values():
        train = set(range(spec.train.start, spec.train.end))
        val = set(range(spec.val.start, spec.val.end))
        test = set(range(spec.test.start, spec.test.end))
        overlap_count += len(train & val) + len(train & test) + len(val & test)
        purge_gap_ok = purge_gap_ok and (spec.val.start - spec.train.end == h_max) and (spec.test.start - spec.val.end == h_max)
        min_fold_ok = min_fold_ok and spec.train.count >= 50 and spec.val.count >= 50 and spec.test.count >= 50
        sample_count += spec.sample_count
        train_samples += spec.train.count
        val_samples += spec.val.count
        test_samples += spec.test.count
    return {
        "timeframe": timeframe,
        "seq_len": seq_len,
        "horizon": horizon,
        "h_max": h_max,
        "input_ticker_count": int(frame["ticker"].nunique()),
        "eligible_ticker_count": len(specs),
        "excluded_ticker_count": len(excluded),
        "excluded_reasons_sample": dict(list(sorted(excluded.items()))[:20]),
        "sample_count_sum": int(sample_count),
        "train_samples": int(train_samples),
        "val_samples": int(val_samples),
        "test_samples": int(test_samples),
        "train_val_test_overlap_count": int(overlap_count),
        "purge_gap_ok": bool(purge_gap_ok),
        "min_fold_samples_ok": bool(min_fold_ok),
        "pass": bool(specs and overlap_count == 0 and purge_gap_ok and min_fold_ok),
    }


def latest_manifest_summary(pattern: str) -> dict[str, Any] | None:
    manifests = sorted(CACHE_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    path = manifests[0]
    payload = load_json(path)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "schema_version": payload.get("schema_version"),
        "cache_kind": payload.get("cache_kind"),
        "timeframe": payload.get("timeframe"),
        "feature_version": payload.get("feature_version"),
        "provider": payload.get("provider") or payload.get("market_data_provider"),
        "ticker_count": payload.get("ticker_count"),
        "feature_columns_count": len(payload.get("feature_columns") or []),
    }


def ticker_registry_summary(timeframe: str) -> dict[str, Any] | None:
    paths = sorted(CACHE_DIR.glob(f"ticker_id_map_{timeframe.lower()}*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return None
    path = paths[0]
    payload = load_json(path)
    mapping = payload.get("mapping") or {}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "timeframe": payload.get("timeframe"),
        "num_tickers": payload.get("num_tickers", len(mapping)),
        "first_ticker": next(iter(mapping.keys()), None),
        "last_ticker": next(reversed(mapping.keys()), None) if mapping else None,
    }


def feature_set_summary() -> dict[str, Any]:
    plan = load_json(FEATURE_SET_PLAN_PATH)
    feature_sets = plan.get("feature_sets") or {}
    required = ["full_features", "no_fundamentals", "price_volatility_volume", "context_light"]
    result: dict[str, Any] = {}
    for name in required:
        spec = feature_sets.get(name)
        if spec is None:
            result[name] = {"exists": False, "status": "design_needed" if name == "context_light" else "missing"}
            continue
        result[name] = {
            "exists": True,
            "column_count": len(spec.get("columns") or []),
            "requires_contract_change": bool(spec.get("requires_contract_change", False)),
            "columns": spec.get("columns") or [],
        }
    result["available_feature_sets"] = sorted(feature_sets.keys())
    return result


def model_forward_summary() -> dict[str, Any]:
    models = {
        "patchtst": PatchTST(
            n_features=36,
            seq_len=60,
            patch_len=16,
            stride=8,
            d_model=32,
            n_heads=4,
            n_layers=1,
            horizon=5,
            num_tickers=8,
        ),
        "cnn_lstm": CNNLSTM(n_features=36, seq_len=60, horizon=5, num_tickers=8, fp32_modules="none"),
        "tide": TiDE(n_features=36, seq_len=60, horizon=5, num_tickers=8),
        "tcn_quantile": TCNQuantile(n_features=36, seq_len=60, horizon=5, num_tickers=8),
    }
    features = torch.randn(2, 60, 36)
    ticker_ids = torch.tensor([0, 1], dtype=torch.long)
    criterion = ForecastCompositeLoss(q_low=0.15, q_high=0.85, lambda_width=0.0)
    target = torch.randn(2, 5)
    result: dict[str, Any] = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(features, ticker_id=ticker_ids)
            losses = criterion(output, target, target, target)
        result[name] = {
            "import_ok": True,
            "line_shape": list(output.line.shape),
            "lower_shape": list(output.lower_band.shape),
            "upper_shape": list(output.upper_band.shape),
            "loss_finite": bool(torch.isfinite(losses.total).item()),
            "line_band_contract_ok": list(output.line.shape) == [2, 5]
            and list(output.lower_band.shape) == [2, 5]
            and list(output.upper_band.shape) == [2, 5],
        }
    return result


def load_scope() -> dict[str, pd.DataFrame]:
    return {
        "price": read_frame(DATA_DIR / "price_data_eodhd_500.parquet"),
        "indicators_1D": read_frame(DATA_DIR / "indicators_eodhd_1D_500.parquet"),
        "indicators_1W": read_frame(DATA_DIR / "indicators_eodhd_1W_500.parquet"),
    }


def context_summary() -> dict[str, Any]:
    context_dir = DATA_DIR / "context" / "eodhd_500"
    result: dict[str, Any] = {"path": str(context_dir.relative_to(PROJECT_ROOT)), "files": {}}
    for path in sorted(context_dir.glob("*.parquet")):
        frame = read_frame(path)
        result["files"][path.name] = frame_summary(frame, duplicate_subset=["ticker", "date", "source"])
    manifest = context_dir / "context_eodhd_500.manifest.json"
    result["manifest"] = load_json(manifest) if manifest.exists() else None
    return result


def build_metrics() -> dict[str, Any]:
    frames = load_scope()
    price = frames["price"]
    ind_1d = frames["indicators_1D"]
    ind_1w = frames["indicators_1W"]
    ind_1d_model = append_model_calendar(ind_1d)
    ind_1w_model = append_model_calendar(ind_1w)

    data = {
        "price_1D": {
            **frame_summary(price, duplicate_subset=["ticker", "date", "source"]),
            "manifest": load_json(DATA_DIR / "price_data_eodhd_500.manifest.json"),
            "adjusted_ohlc": adjusted_ohlc_summary(price),
        },
        "indicators_1D": {
            **frame_summary(ind_1d, duplicate_subset=["ticker", "date", "source"]),
            "manifest": load_json(DATA_DIR / "indicators_eodhd_1D_500.manifest.json"),
        },
        "indicators_1W": {
            **frame_summary(ind_1w, duplicate_subset=["ticker", "date", "source"]),
            "manifest": load_json(DATA_DIR / "indicators_eodhd_1W_500.manifest.json"),
        },
        "context": context_summary(),
    }

    feature_contract = {
        "FEATURE_CONTRACT_VERSION": FEATURE_CONTRACT_VERSION,
        "MODEL_N_FEATURES": MODEL_N_FEATURES,
        "source_feature_count": len(SOURCE_FEATURE_COLUMNS),
        "source_feature_columns": list(SOURCE_FEATURE_COLUMNS),
        "model_feature_count": len(MODEL_FEATURE_COLUMNS),
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "atr_ratio_exists_1D": "atr_ratio" in ind_1d.columns,
        "atr_ratio_exists_1W": "atr_ratio" in ind_1w.columns,
        "atr_ratio_in_model_features": "atr_ratio" in MODEL_FEATURE_COLUMNS,
        "provider_adjustment_policy_expected": provider_adjustment_policy("eodhd"),
        "feature_sets": feature_set_summary(),
    }

    sanity = {
        "1D": {
            "source_features": finite_summary(ind_1d, list(SOURCE_FEATURE_COLUMNS)),
            "model_features_after_calendar": finite_summary(ind_1d_model, list(MODEL_FEATURE_COLUMNS)),
            "target_h5": target_finite_summary(ind_1d, price, timeframe="1D", horizon=5),
            "extremes": ratio_extremes(ind_1d),
            "coverage": coverage_summary(ind_1d),
        },
        "1W": {
            "source_features": finite_summary(ind_1w, list(SOURCE_FEATURE_COLUMNS)),
            "model_features_after_calendar": finite_summary(ind_1w_model, list(MODEL_FEATURE_COLUMNS)),
            "target_h4": target_finite_summary(ind_1w, price, timeframe="1W", horizon=4),
            "target_h6": target_finite_summary(ind_1w, price, timeframe="1W", horizon=6),
            "extremes": ratio_extremes(ind_1w),
            "coverage": coverage_summary(ind_1w),
        },
    }

    splits = {
        "1D_line_h5": split_summary(ind_1d, timeframe="1D", seq_len=252, horizon=5),
        "1D_band_h5": split_summary(ind_1d, timeframe="1D", seq_len=252, horizon=5),
        "1W_line_h4": split_summary(ind_1w, timeframe="1W", seq_len=104, horizon=4),
        "1W_line_h6": split_summary(ind_1w, timeframe="1W", seq_len=104, horizon=6),
        "1W_band_h4": split_summary(ind_1w, timeframe="1W", seq_len=104, horizon=4),
    }

    cache = {
        "feature_cache_manifest_latest_1D": latest_manifest_summary("features_1D_*.manifest.json"),
        "feature_cache_manifest_latest_1W": latest_manifest_summary("features_1W_*.manifest.json"),
        "feature_index_manifest_latest_1D": latest_manifest_summary("feature_index_1D_*.manifest.json"),
        "feature_index_manifest_latest_1W": latest_manifest_summary("feature_index_1W_*.manifest.json"),
        "ticker_registry_1D": ticker_registry_summary("1d"),
        "ticker_registry_1W": ticker_registry_summary("1w"),
    }

    model_availability = model_forward_summary()
    torch_contract = {
        "torch_loaded": "torch" in sys.modules,
        "torch_version": torch.__version__,
        "bootstrap_contract": "torch imported before pandas/numpy in this runner",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    blockers: list[str] = []
    warnings: list[str] = []
    if data["price_1D"]["duplicate_ticker_date_source_count"] != 0:
        blockers.append("price duplicate ticker/date/source 존재")
    for key in ["indicators_1D", "indicators_1W"]:
        if data[key]["duplicate_ticker_date_source_count"] != 0:
            blockers.append(f"{key} duplicate ticker/date/source 존재")
    if data["price_1D"]["adjusted_ohlc"]["adjusted_ohlc_violation_count"] != 0:
        blockers.append("adjusted OHLC violation 존재")
    for timeframe, item in sanity.items():
        if item["source_features"]["nonfinite_count"] != 0:
            blockers.append(f"{timeframe} source feature NaN/Inf 존재")
        if item["model_features_after_calendar"]["nonfinite_count"] != 0:
            blockers.append(f"{timeframe} model feature NaN/Inf 존재")
        for key, value in item.items():
            if key.startswith("target_") and value.get("nonfinite_count") not in (0, None):
                blockers.append(f"{timeframe} {key} target NaN/Inf 존재: {value.get('nonfinite_count')}")
    for name, item in splits.items():
        if not item["pass"]:
            blockers.append(f"{name} split 실패")
    if not all(item["line_band_contract_ok"] and item["loss_finite"] for item in model_availability.values()):
        blockers.append("모델 forward/loss 계약 실패")
    if not feature_contract["feature_sets"]["context_light"]["exists"]:
        warnings.append("context_light feature_set 미정의: design_needed")
    registry_1d = cache.get("ticker_registry_1D") or {}
    registry_1w = cache.get("ticker_registry_1W") or {}
    one_w_cache_registry_refresh_required = False
    if registry_1d.get("num_tickers") != splits["1D_line_h5"]["eligible_ticker_count"]:
        warnings.append(
            f"1D ticker registry count mismatch: registry={registry_1d.get('num_tickers')} split={splits['1D_line_h5']['eligible_ticker_count']}"
        )
    if registry_1w.get("num_tickers") != splits["1W_line_h4"]["eligible_ticker_count"]:
        one_w_cache_registry_refresh_required = True
        warnings.append(
            f"1W ticker registry count mismatch: registry={registry_1w.get('num_tickers')} split={splits['1W_line_h4']['eligible_ticker_count']}"
        )
    feature_cache_1w = cache.get("feature_cache_manifest_latest_1W") or {}
    if feature_cache_1w.get("provider") != "eodhd":
        one_w_cache_registry_refresh_required = True
        warnings.append(f"1W latest feature cache provider is not eodhd: {feature_cache_1w.get('provider')}")
    for timeframe in ["1D", "1W"]:
        fund_ratio = sanity[timeframe]["coverage"].get("has_fundamentals", {}).get("mean")
        if fund_ratio is not None and fund_ratio < 0.9:
            warnings.append(f"{timeframe} has_fundamentals coverage 낮음: {fund_ratio:.4f}")

    judgement = "FAIL" if blockers else ("WARN" if warnings else "PASS")
    cp150_status = "blocked"
    if splits["1W_line_h4"]["pass"] and splits["1W_line_h6"]["pass"] and not blockers:
        cp150_status = "ready_with_cache_registry_refresh" if one_w_cache_registry_refresh_required else "ready"
    cp151_status = "blocked"
    if splits["1W_band_h4"]["pass"] and not blockers:
        cp151_status = "ready_with_cache_registry_refresh" if one_w_cache_registry_refresh_required else "ready"
    readiness = {
        "CP148_LM_1D": "ready" if splits["1D_line_h5"]["pass"] and not blockers else "blocked",
        "CP149_BM_1D": "ready" if splits["1D_band_h5"]["pass"] and not blockers else "blocked",
        "CP150_LM_1W": cp150_status,
        "CP151_BM_1W": cp151_status,
        "first_model_feature_horizon": {
            "CP148_LM_1D": "PatchTST / TCNQuantile, full_features 또는 no_fundamentals, h5",
            "CP149_BM_1D": "CNN-LSTM / TCNQuantile, price_volatility_volume, h5",
            "CP150_LM_1W": "PatchTST / TiDE / TCNQuantile, price_volatility_volume, h4 우선 h6 보조",
            "CP151_BM_1W": "CNN-LSTM / TiDE / TCNQuantile, price_volatility_volume, h4",
        },
    }

    return {
        "cp": "CP148-0-S",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": {
            "db_write": False,
            "supabase_raw_read_write": False,
            "inference_save": False,
            "full_training": False,
            "local_parquet_only": True,
        },
        "judgement": judgement,
        "blockers": blockers,
        "warnings": warnings,
        "data": data,
        "feature_contract": feature_contract,
        "feature_sanity": sanity,
        "splits": splits,
        "cache_and_registry": cache,
        "model_availability": model_availability,
        "torch_contract": torch_contract,
        "wandb_optuna": {
            "wandb_execution_policy": "실험 CP에서 사용자가 VSCode/로컬 터미널에서 직접 활성화한다.",
            "wandb_template": "$env:WANDB_MODE='online'; C:\\Users\\user\\lens\\.venv\\Scripts\\python.exe -m ai.train --model patchtst --timeframe 1D --epochs 3 --device cuda --no-compile --wandb --wandb-project lens-cp148",
            "optuna_path": "ai.sweep",
            "optuna_executed": False,
        },
        "readiness": readiness,
    }


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    data = metrics["data"]
    feature = metrics["feature_contract"]
    sanity = metrics["feature_sanity"]
    splits = metrics["splits"]
    models = metrics["model_availability"]
    readiness = metrics["readiness"]
    cache = metrics["cache_and_registry"]
    lines = [
        "# CP148-0-S 모델 라인업 preflight 보고서",
        "",
        f"판정: **{metrics['judgement']}**",
        "",
        "이번 CP는 500티커 1D/1W 모델 라인업 실험 전 데이터/피처/학습 환경/TCN quantile baseline 준비 상태를 확인하는 preflight다. DB write, Supabase raw read/write, inference 저장, product run 교체, full training은 수행하지 않았다.",
        "",
        "## 1. 데이터 최신성 및 coverage",
        "",
        "| 항목 | rows | tickers | date_min | date_max | duplicate | source | provider |",
        "|---|---:|---:|---|---|---:|---|---|",
    ]
    for key in ["price_1D", "indicators_1D", "indicators_1W"]:
        item = data[key]
        lines.append(
            f"| {key} | {item.get('rows')} | {item.get('ticker_count')} | {item.get('date_min')} | {item.get('date_max')} | "
            f"{item.get('duplicate_ticker_date_source_count')} | {','.join(item.get('source_values') or [])} | {','.join(item.get('provider_values') or [])} |"
        )
    lines.extend(
        [
            "",
            f"- adjusted OHLC violation: `{data['price_1D']['adjusted_ohlc']['adjusted_ohlc_violation_count']}`",
            f"- expected provider policy: `{feature['provider_adjustment_policy_expected']}`",
            "",
            "## 2. feature contract",
            "",
            f"- FEATURE_CONTRACT_VERSION: `{feature['FEATURE_CONTRACT_VERSION']}`",
            f"- MODEL_N_FEATURES: `{feature['MODEL_N_FEATURES']}`",
            f"- source feature 수: `{feature['source_feature_count']}`",
            f"- model feature 수: `{feature['model_feature_count']}`",
            f"- atr_ratio 존재: 1D `{feature['atr_ratio_exists_1D']}`, 1W `{feature['atr_ratio_exists_1W']}`",
            f"- atr_ratio 모델 feature 포함: `{feature['atr_ratio_in_model_features']}`",
            f"- context_light: `{feature['feature_sets']['context_light']['status']}`",
            "",
            "## 3. feature / target sanity",
            "",
            "| timeframe | source nonfinite | model nonfinite | target | target nonfinite | has_macro | has_breadth | has_fundamentals |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for timeframe in ["1D", "1W"]:
        item = sanity[timeframe]
        target_key = "target_h5" if timeframe == "1D" else "target_h4"
        lines.append(
            f"| {timeframe} | {item['source_features']['nonfinite_count']} | {item['model_features_after_calendar']['nonfinite_count']} | "
            f"{target_key} | {item[target_key]['nonfinite_count']} | "
            f"{item['coverage']['has_macro']['mean']:.4f} | {item['coverage']['has_breadth']['mean']:.4f} | {item['coverage']['has_fundamentals']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. split sanity",
            "",
            "| split | eligible | excluded | train | val | test | overlap | purge_gap | min_fold |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for name, item in splits.items():
        lines.append(
            f"| {name} | {item['eligible_ticker_count']} | {item['excluded_ticker_count']} | {item['train_samples']} | "
            f"{item['val_samples']} | {item['test_samples']} | {item['train_val_test_overlap_count']} | "
            f"{_status(item['purge_gap_ok'])} | {_status(item['min_fold_samples_ok'])} |"
        )
    lines.extend(
        [
            "",
            "## 5. 모델 forward / loss 계약",
            "",
            "| model | line | lower | upper | loss finite | contract |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, item in models.items():
        lines.append(
            f"| {name} | {item['line_shape']} | {item['lower_shape']} | {item['upper_shape']} | "
            f"{_status(item['loss_finite'])} | {_status(item['line_band_contract_ok'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. cache / ticker registry",
            "",
            f"- latest 1D feature cache manifest: `{(cache['feature_cache_manifest_latest_1D'] or {}).get('path')}`",
            f"- latest 1W feature cache manifest: `{(cache['feature_cache_manifest_latest_1W'] or {}).get('path')}`",
            f"- latest 1D ticker registry: `{(cache['ticker_registry_1D'] or {}).get('path')}`, count `{(cache['ticker_registry_1D'] or {}).get('num_tickers')}`",
            f"- latest 1W ticker registry: `{(cache['ticker_registry_1W'] or {}).get('path')}`, count `{(cache['ticker_registry_1W'] or {}).get('num_tickers')}`",
            "",
            "## 7. W&B / Optuna 준비",
            "",
            "- W&B는 큰 실험 CP에서 사용자가 VSCode 로컬 터미널에서 직접 켠다.",
            "- CP148-0-S에서는 W&B/Optuna sweep을 실행하지 않았다.",
            f"- 명령 템플릿: `{metrics['wandb_optuna']['wandb_template']}`",
            "",
            "## 8. 판정",
            "",
            f"- blockers: `{metrics['blockers']}`",
            f"- warnings: `{metrics['warnings']}`",
            f"- CP148-LM-1D 바로 실행 가능 여부: `{readiness['CP148_LM_1D']}`",
            f"- CP149-BM-1D 바로 실행 가능 여부: `{readiness['CP149_BM_1D']}`",
            f"- CP150-LM-1W 바로 실행 가능 여부: `{readiness['CP150_LM_1W']}`",
            f"- CP151-BM-1W 바로 실행 가능 여부: `{readiness['CP151_BM_1W']}`",
            "",
            "## 9. 우선 조합",
            "",
        ]
    )
    for cp_name, combo in readiness["first_model_feature_horizon"].items():
        lines.append(f"- {cp_name}: {combo}")
    lines.extend(
        [
            "",
            "## 10. 사람이 판단해야 할 항목",
            "",
            "- `context_light`는 현재 feature_set plan에 없으므로 이번 CP에서 만들지 않고 design_needed로 기록했다.",
            "- full_features는 fundamentals coverage 해석 WARN을 유지해야 한다.",
            "- TCNQuantile은 skeleton/tiny forward/loss만 준비됐고, 성능 후보 여부는 다음 smoke에서 판단해야 한다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_plan(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# CP148-0 모델 라인업 실험 계획 초안",
        "",
        "이 문서는 CP148-0-S preflight 이후 바로 실행할 수 있는 실험 축 초안이다. 실제 W&B online 실행은 사용자가 VSCode 로컬 터미널에서 진행한다.",
        "",
        "## 공통 금지/운영",
        "",
        "- product run 교체 금지",
        "- inference 저장 금지",
        "- full 473/500 장기 학습은 각 CP 승인 후 실행",
        "- W&B online 큰 실험은 사용자가 직접 실행",
        "",
        "## CP148-LM-1D",
        "",
        "- 우선 모델: PatchTST, TCNQuantile",
        "- horizon: h5",
        "- feature_set: full_features, no_fundamentals",
        "- selector: line_gate",
        "- search space 초안: lr, weight_decay, dropout, patch_len/stride 또는 TCN dilation/channel",
        "",
        "## CP149-BM-1D",
        "",
        "- 우선 모델: CNN-LSTM, TCNQuantile, TiDE 보조",
        "- horizon: h5",
        "- feature_set: price_volatility_volume, price_volatility",
        "- selector: band_gate",
        "- search space 초안: q_low/q_high, lambda_band, seq_len, band_mode, TCN channel/dilation",
        "",
        "## CP150-LM-1W",
        "",
        "- 우선 모델: PatchTST, TiDE, TCNQuantile",
        "- horizon: h4 우선, h6 보조",
        "- seq_len: 104",
        "- feature_set: price_volatility_volume, no_fundamentals",
        "- selector: line_gate",
        "",
        "## CP151-BM-1W",
        "",
        "- 우선 모델: CNN-LSTM, TiDE, TCNQuantile",
        "- horizon: h4",
        "- seq_len: 104",
        "- feature_set: price_volatility_volume",
        "- selector: band_gate",
        "",
        "## W&B 명령 템플릿",
        "",
        "```powershell",
        "cd C:\\Users\\user\\lens",
        ".\\.venv\\Scripts\\Activate.ps1",
        "$env:WANDB_MODE='online'",
        "C:\\Users\\user\\lens\\.venv\\Scripts\\python.exe -m ai.train --model patchtst --timeframe 1D --horizon 5 --seq-len 252 --epochs 3 --batch-size 256 --device cuda --no-compile --wandb --wandb-project lens-cp148 --feature-set full_features --checkpoint-selection line_gate",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP148-0-S 모델 라인업 preflight")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_metrics()
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    write_experiment_plan(Path(args.plan_path), metrics)
    print(json.dumps(_json_safe({"judgement": metrics["judgement"], "metrics_path": args.metrics_path}), ensure_ascii=False))


if __name__ == "__main__":
    main()
