from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
os.environ.setdefault("MARKET_DATA_FALLBACK_PROVIDER", "")
os.environ.setdefault("EODHD_API_KEY", "")
os.environ.setdefault("LENS_DATA_BACKEND", "local")
os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
os.environ.setdefault("LENS_LOCAL_SNAPSHOT_DIR", str(ROOT / "data" / "parquet"))
os.environ.setdefault("WANDB_MODE", "disabled")

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

torch = bootstrap_torch(cpu_only=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.calibration_artifacts import apply_product_band_calibration  # noqa: E402
from ai.inference import load_checkpoint  # noqa: E402
from ai.preprocessing import CALENDAR_FEATURE_COLUMNS, build_calendar_feature_frame  # noqa: E402

DATA_DIR = ROOT / "data" / "parquet"
V1_DIR = ROOT / "backend" / "data" / "v1"
DOCS_DIR = ROOT / "docs"
LOG_DIR = ROOT / "logs" / "cp210_band_refresh"

PRICE_500 = DATA_DIR / "price_data_yfinance_500.parquet"
INDICATORS_1D = DATA_DIR / "indicators_yfinance_1D_500.parquet"
INDICATORS_1W = DATA_DIR / "indicators_yfinance_1W_500.parquet"
BAND_1D_OUT = V1_DIR / "predictions_band_1d.parquet"
BAND_1W_OUT = V1_DIR / "predictions_band_1w.parquet"
HISTORY_OUT = V1_DIR / "product_prediction_history_1D.parquet"
HISTORY_MANIFEST_OUT = V1_DIR / "product_prediction_history_1D.manifest.json"
LINE_1D = V1_DIR / "predictions_line_1d.parquet"

CP153_META = DOCS_DIR / "cp153_bm_1d_band_primary_product_candidate_run_meta.json"
CP178_SUMMARY = DOCS_DIR / "cp178_bm_1w_band_500_stage5_true_walk_forward_summary.csv"


@dataclass
class RefreshSpec:
    slot: str
    timeframe: str
    horizon: int
    seq_len: int
    feature_columns: list[str]
    checkpoint_path: Path
    model_id: str
    source_cp: str
    calibration: dict[str, Any]
    ticker_registry_path: str | None
    use_future_covariate: bool
    future_cov_dim: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def calibration_payload(method: str | None, params: dict[str, Any]) -> dict[str, Any]:
    method_text = str(method or "raw")
    if method_text == "raw":
        return {"status": "raw_band_output_used", "applied": False, "method": "raw", "params": {}}
    return {
        "status": "calibration_applied",
        "applied": True,
        "method": method_text,
        "params": params,
        "source": "existing_product_candidate_calibration",
    }


def resolve_path(raw: str | Path | None) -> Path:
    if raw is None:
        raise ValueError("checkpoint path가 비어 있습니다.")
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def load_registry(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    registry_path = resolve_path(path)
    payload = read_json(registry_path)
    return {str(ticker).upper(): int(idx) for ticker, idx in (payload.get("mapping") or {}).items()}


def load_cp153_spec() -> RefreshSpec:
    meta = read_json(CP153_META)
    _, checkpoint = load_checkpoint(resolve_path(meta["checkpoint_path"]))
    config = checkpoint["config"]
    calibration = calibration_payload(
        str((meta.get("calibration") or {}).get("method") or "raw"),
        dict((meta.get("calibration") or {}).get("params") or {}),
    )
    return RefreshSpec(
        slot="band_1d",
        timeframe="1D",
        horizon=int(config["horizon"]),
        seq_len=int(config["seq_len"]),
        feature_columns=list(config.get("feature_columns") or meta["feature_columns"]),
        checkpoint_path=resolve_path(meta["checkpoint_path"]),
        model_id=str(meta.get("run_id") or "tide-1D-ea54dcae654d"),
        source_cp="CP153",
        calibration=calibration,
        ticker_registry_path=config.get("ticker_registry_path"),
        use_future_covariate=bool(config.get("use_future_covariate", False)),
        future_cov_dim=int(config.get("future_cov_dim") or 0),
    )


def load_cp178_specs() -> list[RefreshSpec]:
    summary = pd.read_csv(CP178_SUMMARY)
    mask = (
        (summary.get("candidate_id", pd.Series(dtype=object)).astype(str) == "tide_s104_q10q90_param")
        & (summary.get("model", pd.Series(dtype=object)).astype(str) == "tide")
        & summary.get("checkpoint_path", pd.Series(dtype=object)).notna()
    )
    rows = summary.loc[mask].drop_duplicates(subset=["checkpoint_path"]).copy()
    specs: list[RefreshSpec] = []
    for _, row in rows.iterrows():
        checkpoint_path = resolve_path(str(row["checkpoint_path"]))
        _, checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint["config"]
        specs.append(
            RefreshSpec(
                slot="band_1w",
                timeframe="1W",
                horizon=int(config["horizon"]),
                seq_len=int(config["seq_len"]),
                feature_columns=list(config.get("feature_columns") or []),
                checkpoint_path=checkpoint_path,
                model_id=str(row.get("candidate_id") or "tide_s104_q10q90_param"),
                source_cp="CP178-WFLOCK",
                calibration=calibration_payload(str(row.get("calibration_method") or "raw"), parse_params(row.get("calibration_params"))),
                ticker_registry_path=config.get("ticker_registry_path"),
                use_future_covariate=bool(config.get("use_future_covariate", False)),
                future_cov_dim=int(config.get("future_cov_dim") or 0),
            )
        )
    return specs


def read_input_frames() -> dict[str, pd.DataFrame]:
    prices = pd.read_parquet(PRICE_500)
    indicators_1d = pd.read_parquet(INDICATORS_1D)
    indicators_1w = pd.read_parquet(INDICATORS_1W)
    for frame in (prices, indicators_1d, indicators_1w):
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    prices = prices.dropna(subset=["ticker", "date"])
    indicators_1d = indicators_1d.dropna(subset=["ticker", "date"])
    indicators_1w = indicators_1w.dropna(subset=["ticker", "date"])
    return {
        "prices": prices,
        "indicators_1d": indicators_1d,
        "indicators_1w": indicators_1w,
    }


def weekly_close_frame(prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ticker, group in prices.sort_values("date").groupby("ticker", sort=False):
        weekly = (
            group.set_index("date")
            .resample("W-FRI")
            .agg({"close": "last"})
            .dropna(subset=["close"])
            .reset_index()
        )
        weekly["ticker"] = ticker
        rows.append(weekly[["ticker", "date", "close"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ticker", "date", "close"])


def frame_summary(frame: pd.DataFrame, date_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "ticker_count": 0, "date_min": None, "date_max": None}
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    return {
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else None,
        "date_min": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
        "date_max": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
    }


def cutoff_date(latest_date: pd.Timestamp, days: int) -> pd.Timestamp:
    return latest_date.normalize() - pd.Timedelta(days=days)


def future_dates(dates: list[pd.Timestamp], index: int, horizon: int, timeframe: str) -> list[pd.Timestamp]:
    resolved: list[pd.Timestamp] = []
    asof = dates[index]
    if timeframe == "1D":
        synthetic = list(pd.bdate_range(asof + pd.Timedelta(days=1), periods=horizon))
    else:
        synthetic = [asof + pd.Timedelta(days=7 * step) for step in range(1, horizon + 1)]
    for step in range(1, horizon + 1):
        target_index = index + step
        if target_index < len(dates):
            resolved.append(dates[target_index])
        else:
            resolved.append(synthetic[step - 1])
    return resolved


def build_samples(
    indicators: pd.DataFrame,
    closes: pd.DataFrame,
    spec: RefreshSpec,
    *,
    history_days: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[int]]:
    registry = load_registry(spec.ticker_registry_path)
    allowed_tickers = set(registry.keys()) if registry else set(indicators["ticker"].unique())
    missing_features = [column for column in spec.feature_columns if column not in indicators.columns]
    if missing_features:
        raise ValueError(f"{spec.slot} feature column 누락: {missing_features}")

    close_map = {
        (str(row.ticker).upper(), pd.Timestamp(row.date).normalize()): float(row.close)
        for row in closes[["ticker", "date", "close"]].itertuples(index=False)
        if pd.notna(row.close)
    }
    latest = pd.to_datetime(indicators["date"], errors="coerce").max().normalize()
    cutoff = cutoff_date(latest, history_days)

    metadata: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    ticker_ids: list[int] = []
    for ticker, group in indicators[indicators["ticker"].isin(allowed_tickers)].sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[spec.feature_columns].astype("float32").to_numpy()
        finite_mask = np.isfinite(values).all(axis=1)
        dates = [pd.Timestamp(value).normalize() for value in group["date"].tolist()]
        for index in range(spec.seq_len - 1, len(group)):
            asof = dates[index]
            if asof < cutoff:
                continue
            window_start = index - spec.seq_len + 1
            if not bool(finite_mask[window_start : index + 1].all()):
                continue
            anchor_close = close_map.get((ticker, asof))
            if anchor_close is None or not math.isfinite(anchor_close):
                continue
            fdates = future_dates(dates, index, spec.horizon, spec.timeframe)
            actual_returns: list[float] = []
            for future_date in fdates:
                future_close = close_map.get((ticker, future_date.normalize()))
                if future_close is None or not math.isfinite(future_close):
                    actual_returns.append(float("nan"))
                else:
                    actual_returns.append((future_close / anchor_close) - 1.0)
            feature_rows.append(values[window_start : index + 1])
            future_calendar = build_calendar_feature_frame(fdates)[CALENDAR_FEATURE_COLUMNS].astype("float32").to_numpy()
            future_rows.append(future_calendar)
            ticker_ids.append(int(registry.get(ticker, 0)))
            metadata.append(
                {
                    "ticker": ticker,
                    "asof_date": asof.strftime("%Y-%m-%d"),
                    "forecast_dates": [date.strftime("%Y-%m-%d") for date in fdates],
                    "anchor_close": float(anchor_close),
                    "actual_returns": actual_returns,
                }
            )
    return metadata, feature_rows, future_rows, ticker_ids


def run_spec_inference(
    spec: RefreshSpec,
    metadata: list[dict[str, Any]],
    feature_rows: list[np.ndarray],
    future_rows: list[np.ndarray],
    ticker_ids: list[int],
    *,
    batch_size: int,
) -> pd.DataFrame:
    model, _ = load_checkpoint(spec.checkpoint_path)
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(metadata), batch_size):
            end = min(start + batch_size, len(metadata))
            features = torch.tensor(np.stack(feature_rows[start:end]), dtype=torch.float32)
            ticker_tensor = torch.tensor(ticker_ids[start:end], dtype=torch.long)
            if spec.use_future_covariate and spec.future_cov_dim > 0:
                future_covariates = torch.tensor(np.stack(future_rows[start:end]), dtype=torch.float32)
                output = model(features, ticker_id=ticker_tensor, future_covariate=future_covariates)
            else:
                output = model(features, ticker_id=ticker_tensor)
            lower_returns, upper_returns = apply_product_band_calibration(
                lower_returns=output.lower_band.detach().cpu(),
                upper_returns=output.upper_band.detach().cpu(),
                calibration=spec.calibration,
            )
            lower_np = lower_returns.numpy()
            upper_np = upper_returns.numpy()
            for local_index, meta in enumerate(metadata[start:end]):
                for horizon_index in range(spec.horizon):
                    lower_value = float(lower_np[local_index, horizon_index])
                    upper_value = float(upper_np[local_index, horizon_index])
                    if spec.timeframe == "1D":
                        lower_value = float(meta["anchor_close"] * (1.0 + lower_value))
                        upper_value = float(meta["anchor_close"] * (1.0 + upper_value))
                    actual_return = float(meta["actual_returns"][horizon_index])
                    records.append(
                        {
                            "ticker": meta["ticker"],
                            "asof_date": meta["asof_date"],
                            "forecast_date": meta["forecast_dates"][horizon_index],
                            "horizon_step": int(horizon_index + 1),
                            "band_lower": min(lower_value, upper_value),
                            "band_upper": max(lower_value, upper_value),
                            "actual_return": actual_return if math.isfinite(actual_return) else np.nan,
                            "actual_return_available": bool(math.isfinite(actual_return)),
                            "model_id": spec.model_id,
                            "source_cp": spec.source_cp,
                        }
                    )
    return pd.DataFrame(records)


def atomic_write_parquet(frame: pd.DataFrame, path: Path, *, apply: bool) -> None:
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(path)


def backup_file(path: Path, run_stamp: str) -> str | None:
    if not path.exists():
        return None
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.stem}_before_cp210_{run_stamp}{path.suffix}"
    shutil.copy2(path, backup)
    return str(backup)


def validate_band_frame(frame: pd.DataFrame, *, timeframe: str) -> dict[str, Any]:
    duplicate_keys = ["ticker", "asof_date", "horizon_step"]
    if "forecast_date" in frame.columns:
        duplicate_keys.append("forecast_date")
    duplicate_count = int(frame.duplicated(duplicate_keys).sum())
    lower = pd.to_numeric(frame["band_lower"], errors="coerce")
    upper = pd.to_numeric(frame["band_upper"], errors="coerce")
    finite_count = int(np.isfinite(lower).sum() + np.isfinite(upper).sum())
    non_finite_count = int((~np.isfinite(lower)).sum() + (~np.isfinite(upper)).sum())
    inverted_count = int((lower > upper).sum())
    dates = pd.to_datetime(frame["asof_date"], errors="coerce")
    return {
        "timeframe": timeframe,
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if not frame.empty else 0,
        "asof_min": dates.min().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
        "asof_max": dates.max().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
        "duplicate_count": duplicate_count,
        "band_non_finite_count": non_finite_count,
        "band_finite_value_count": finite_count,
        "inverted_band_count": inverted_count,
    }


def rebuild_history_with_new_band(band_1d: pd.DataFrame, *, apply: bool, run_stamp: str) -> dict[str, Any]:
    now = utc_now()
    if HISTORY_OUT.exists():
        old_history = pd.read_parquet(HISTORY_OUT)
        line_rows = old_history[old_history["role"].astype(str).str.lower() == "line"].copy()
    else:
        line = pd.read_parquet(LINE_1D)
        line_rows = pd.DataFrame(
            {
                "ticker": line["ticker"].astype(str).str.upper(),
                "timeframe": "1D",
                "role": "line",
                "run_id": line["model_id"].astype(str),
                "asof_date": pd.to_datetime(line["asof_date"]).dt.strftime("%Y-%m-%d"),
                "display_horizon": 5,
                "display_date": pd.to_datetime(line["asof_date"]).dt.strftime("%Y-%m-%d"),
                "line_value": line["safe_line_score"].astype("float64"),
                "lower_value": np.nan,
                "upper_value": np.nan,
                "source": "v1_local_parquet",
                "model_feature_hash": line["source_cp"].astype(str),
                "created_at": now,
            }
        )
    band_rows = pd.DataFrame(
        {
            "ticker": band_1d["ticker"].astype(str).str.upper(),
            "timeframe": "1D",
            "role": "band",
            "run_id": band_1d["model_id"].astype(str),
            "asof_date": pd.to_datetime(band_1d["asof_date"]).dt.strftime("%Y-%m-%d"),
            "display_horizon": band_1d["horizon_step"].astype(int),
            "display_date": pd.to_datetime(band_1d["forecast_date"]).dt.strftime("%Y-%m-%d"),
            "line_value": np.nan,
            "lower_value": band_1d["band_lower"].astype("float64"),
            "upper_value": band_1d["band_upper"].astype("float64"),
            "source": "v1_local_parquet",
            "model_feature_hash": band_1d["source_cp"].astype(str),
            "created_at": now,
        }
    )
    unified = pd.concat([line_rows, band_rows], ignore_index=True)
    unified = unified.sort_values(["ticker", "role", "asof_date", "display_horizon"]).reset_index(drop=True)
    if apply:
        backup_file(HISTORY_OUT, run_stamp)
        atomic_write_parquet(unified, HISTORY_OUT, apply=True)
        manifest = {
            "line_run_id": str(line_rows["run_id"].iloc[0]) if len(line_rows) else None,
            "band_run_id": str(band_rows["run_id"].iloc[0]) if len(band_rows) else None,
            "asof_start": str(unified["asof_date"].min()) if len(unified) else None,
            "asof_end": str(unified["asof_date"].max()) if len(unified) else None,
            "row_count": int(len(unified)),
            "source": "v1_local_parquet",
            "built_at": now,
            "cp": "CP210",
            "line_policy": "existing_line_rows_preserved",
            "band_policy": "cp153_1d_band_rows_replaced",
        }
        HISTORY_MANIFEST_OUT.write_text(json.dumps(clean_json(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "rows": int(len(unified)),
        "line_rows": int(len(line_rows)),
        "band_rows": int(len(band_rows)),
        "asof_max": str(unified["asof_date"].max()) if len(unified) else None,
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# CP210 band forward refresh 실행 보고",
        "",
        f"- final_status: `{metrics['final_status']}`",
        f"- created_at: `{metrics['created_at']}`",
        f"- apply: `{metrics['apply']}`",
        "",
        "## 입력 최신일",
        "",
        f"- price yfinance 500: `{metrics['inputs']['price']['date_max']}`",
        f"- indicators 1D: `{metrics['inputs']['indicators_1d']['date_max']}`",
        f"- indicators 1W: `{metrics['inputs']['indicators_1w']['date_max']}`",
        "",
        "## serving parquet",
        "",
        f"- 1D band before/after: `{metrics['before']['band_1d'].get('date_max')}` -> `{metrics['after']['band_1d'].get('asof_max')}`",
        f"- 1W band before/after: `{metrics['before']['band_1w'].get('date_max')}` -> `{metrics['after']['band_1w'].get('asof_max')}`",
        f"- product history after: `{metrics['history'].get('asof_max')}`",
        "",
        "## 금지 작업 확인",
        "",
        "- line refresh: 실행 안 함",
        "- 새 학습: 실행 안 함",
        "- 새 calibration: 실행 안 함",
        "- Supabase/DB write: 실행 안 함",
        "- EODHD fallback: 실행 안 함",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames = read_input_frames()
    weekly_closes = weekly_close_frame(frames["prices"])
    before = {
        "band_1d": frame_summary(pd.read_parquet(BAND_1D_OUT), "asof_date") if BAND_1D_OUT.exists() else {},
        "band_1w": frame_summary(pd.read_parquet(BAND_1W_OUT), "asof_date") if BAND_1W_OUT.exists() else {},
        "history_1d": frame_summary(pd.read_parquet(HISTORY_OUT), "asof_date") if HISTORY_OUT.exists() else {},
    }
    inputs = {
        "price": frame_summary(frames["prices"], "date"),
        "indicators_1d": frame_summary(frames["indicators_1d"], "date"),
        "indicators_1w": frame_summary(frames["indicators_1w"], "date"),
    }

    cp153 = load_cp153_spec()
    cp178_specs = load_cp178_specs()
    if len(cp178_specs) != 9:
        raise RuntimeError(f"CP178 checkpoint 수가 9개가 아닙니다: {len(cp178_specs)}")

    if args.apply:
        backups = {
            "band_1d": backup_file(BAND_1D_OUT, run_stamp),
            "band_1w": backup_file(BAND_1W_OUT, run_stamp),
        }
    else:
        backups = {}

    metadata_1d, features_1d, future_1d, ids_1d = build_samples(
        frames["indicators_1d"],
        frames["prices"][["ticker", "date", "close"]],
        cp153,
        history_days=args.history_days_1d,
    )
    band_1d = run_spec_inference(cp153, metadata_1d, features_1d, future_1d, ids_1d, batch_size=args.batch_size)
    band_1d = band_1d.sort_values(["ticker", "asof_date", "horizon_step"]).reset_index(drop=True)
    atomic_write_parquet(band_1d, BAND_1D_OUT, apply=args.apply)

    metadata_1w, features_1w, future_1w, ids_1w = build_samples(
        frames["indicators_1w"],
        weekly_closes,
        cp178_specs[0],
        history_days=args.history_days_1w,
    )
    ensemble_parts = []
    for spec in cp178_specs:
        part = run_spec_inference(spec, metadata_1w, features_1w, future_1w, ids_1w, batch_size=args.batch_size)
        ensemble_parts.append(part)
    cp178_all = pd.concat(ensemble_parts, ignore_index=True)
    aggregate_columns = {
        "band_lower": "mean",
        "band_upper": "mean",
        "actual_return": "mean",
        "model_id": "first",
        "source_cp": "first",
    }
    band_1w = (
        cp178_all.groupby(["ticker", "asof_date", "horizon_step"], as_index=False)
        .agg(aggregate_columns)
        .sort_values(["ticker", "asof_date", "horizon_step"])
        .reset_index(drop=True)
    )
    band_1w["actual_return_available"] = band_1w["actual_return"].map(lambda value: bool(math.isfinite(float(value))) if pd.notna(value) else False)
    band_1w = band_1w[["ticker", "asof_date", "horizon_step", "band_lower", "band_upper", "actual_return", "model_id", "source_cp"]]
    atomic_write_parquet(band_1w, BAND_1W_OUT, apply=args.apply)

    history = rebuild_history_with_new_band(band_1d, apply=args.apply, run_stamp=run_stamp)
    after = {
        "band_1d": validate_band_frame(band_1d, timeframe="1D"),
        "band_1w": validate_band_frame(band_1w, timeframe="1W"),
    }
    aapl = {
        "band_1d": band_1d[band_1d["ticker"] == "AAPL"].sort_values(["asof_date", "horizon_step"]).tail(cp153.horizon).to_dict(orient="records"),
        "band_1w": band_1w[band_1w["ticker"] == "AAPL"].sort_values(["asof_date", "horizon_step"]).tail(cp178_specs[0].horizon).to_dict(orient="records"),
    }
    final_status = "PASS_BAND_FORWARD_REFRESH_DRY_RUN"
    if args.apply:
        final_status = "PASS_BAND_FORWARD_REFRESH_APPLIED"
    metrics = {
        "cp": "CP210-LG",
        "created_at": utc_now(),
        "apply": bool(args.apply),
        "final_status": final_status,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "inputs": inputs,
        "before": before,
        "after": after,
        "history": history,
        "samples": {
            "band_1d_sample_count": len(metadata_1d),
            "band_1w_sample_count": len(metadata_1w),
            "cp178_checkpoint_count": len(cp178_specs),
        },
        "backups": backups,
        "aapl_smoke": aapl,
        "forbidden_actions_observed": {
            "line_refresh": False,
            "new_training": False,
            "new_calibration": False,
            "checkpoint_reselection": False,
            "supabase_write": False,
            "db_write": False,
            "eodhd_fallback": False,
        },
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP210 1D/1W band serving parquet forward refresh")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="serving parquet을 실제 갱신합니다.")
    mode.add_argument("--dry-run", action="store_true", help="산출물 계산만 검증하고 parquet은 쓰지 않습니다.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--history-days-1d", type=int, default=365)
    parser.add_argument("--history-days-1w", type=int, default=730)
    parser.add_argument("--metrics-path", default=str(DOCS_DIR / "cp210_band_refresh_metrics.json"))
    parser.add_argument("--report-path", default=str(DOCS_DIR / "cp210_band_refresh_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V1_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = refresh(args)
    write_json(Path(args.metrics_path), metrics)
    write_report(Path(args.report_path), metrics)
    print(json.dumps({"status": metrics["final_status"], "elapsed_seconds": metrics["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
