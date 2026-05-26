from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.torch_bootstrap import bootstrap_torch  # noqa: E402

_TORCH = bootstrap_torch(cpu_only=True)

import pandas as pd  # noqa: E402
DOCS = ROOT / "docs"
LOG_DIR = ROOT / "logs" / "cp206_daily_refresh"
V1_DIR = ROOT / "backend" / "data" / "v1"
CP204_PACKAGE = ROOT / "data" / "artifacts" / "cp204_v1_import_package"

EXPECTED_SOURCE_HASH = "90666b44cbfb8e5c"
PROGRESS_PATH = LOG_DIR / "progress.log"
METRICS_PATH = DOCS / "cp206_daily_refresh_metrics.json"
TARGET_LOCK_REPORT = DOCS / "cp206_target_lock_report.md"
PREFLIGHT_REPORT = DOCS / "cp206_inference_rebuild_preflight.md"
LOCAL_CONTRACT_REPORT = DOCS / "cp206_local_refresh_contract.md"
RUNTIME_PROFILE_REPORT = DOCS / "cp206_1w_band_runtime_profile.md"
LATEST_REPORT = DOCS / "cp206_daily_refresh_latest.md"

PRICE_PARQUET = ROOT / "data" / "parquet" / "price_data_yfinance_500.parquet"
INDICATOR_1D_PARQUET = ROOT / "data" / "parquet" / "indicators_yfinance_1D_500.parquet"
INDICATOR_1W_PARQUET = ROOT / "data" / "parquet" / "indicators_yfinance_1W_500.parquet"

SERVING_FILES = {
    "line_1d": V1_DIR / "predictions_line_1d.parquet",
    "band_1d": V1_DIR / "predictions_band_1d.parquet",
    "band_1w": V1_DIR / "predictions_band_1w.parquet",
    "history_1d": V1_DIR / "product_prediction_history_1D.parquet",
}


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
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_progress(stage: str, **payload: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "time": utc_now(),
        "stage": stage,
        **payload,
    }
    with PROGRESS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean_json(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_parquet_summary(path: Path, date_column: str = "date") -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_parquet(path)
    summary: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
    }
    if "ticker" in df.columns:
        summary["ticker_count"] = int(df["ticker"].nunique())
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            summary["min_date"] = str(dates.min().date())
            summary["max_date"] = str(dates.max().date())
    return summary


def current_source_hash(timeframe: str) -> dict[str, Any]:
    try:
        os.environ.setdefault("MARKET_DATA_PROVIDER", "yfinance")
        os.environ.setdefault("LENS_USE_LOCAL_SNAPSHOTS", "1")
        os.environ.setdefault("LENS_REQUIRE_LOCAL_SNAPSHOTS", "1")
        from ai.preprocessing import resolve_data_fingerprint

        value = resolve_data_fingerprint(timeframe, market_data_provider="yfinance")
        return {
            "status": "OK",
            "timeframe": timeframe,
            "hash": value,
            "matches_locked_cp153_cp178_hash": value == EXPECTED_SOURCE_HASH,
            "locked_reference_hash": EXPECTED_SOURCE_HASH,
            "note": "일일 cron 이후 source hash는 변동될 수 있다. CP206은 실행 시점 local parquet를 진실 원천으로 둔다.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "FAIL",
            "timeframe": timeframe,
            "error": repr(exc),
            "locked_reference_hash": EXPECTED_SOURCE_HASH,
        }


def load_torch_cpu():
    return _TORCH


def load_checkpoint_config(path: Path) -> dict[str, Any]:
    torch = load_torch_cpu()
    checkpoint = torch.load(path, map_location="cpu")
    return dict(checkpoint.get("config") or {})


def load_checkpoint_model(path: Path):
    from ai.inference import load_checkpoint

    return load_checkpoint(path)


def cp153_checkpoint_path() -> Path | None:
    meta_path = DOCS / "cp153_bm_1d_band_primary_product_candidate_run_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_path = meta.get("checkpoint_path")
    return (ROOT / str(raw_path)) if raw_path else None


def cp178_checkpoint_paths() -> list[Path]:
    summary_path = DOCS / "cp178_bm_1w_band_500_stage5_true_walk_forward_summary.csv"
    if not summary_path.exists():
        return []
    df = pd.read_csv(summary_path)
    mask = (
        (df.get("candidate_id", pd.Series(dtype=object)).astype(str) == "tide_s104_q10q90_param")
        & (df.get("model", pd.Series(dtype=object)).astype(str) == "tide")
        & df.get("checkpoint_path", pd.Series(dtype=object)).notna()
    )
    paths = []
    for raw in df.loc[mask, "checkpoint_path"].astype(str).drop_duplicates().tolist():
        paths.append(ROOT / raw)
    return paths


def pick_latest_sequence(indicator_path: Path, feature_columns: list[str], seq_len: int, registry_path: str | None) -> dict[str, Any]:
    indicators = pd.read_parquet(indicator_path)
    missing = [col for col in feature_columns if col not in indicators.columns]
    if missing:
        return {"status": "SCHEMA_MISMATCH", "missing_features": missing}

    mapping: dict[str, int] = {}
    if registry_path:
        registry = json.loads((ROOT / registry_path).read_text(encoding="utf-8"))
        mapping = {str(k): int(v) for k, v in (registry.get("mapping") or {}).items()}
        indicators = indicators[indicators["ticker"].astype(str).isin(mapping.keys())].copy()

    indicators["date"] = pd.to_datetime(indicators["date"], errors="coerce")
    indicators = indicators.dropna(subset=["date"]).sort_values(["ticker", "date"])
    latest_ticker = None
    latest_frame = None
    latest_date = None
    for ticker, group in indicators.groupby("ticker", sort=True):
        group = group.dropna(subset=feature_columns)
        if len(group) < seq_len:
            continue
        candidate_date = group["date"].max()
        if latest_date is None or candidate_date > latest_date:
            latest_ticker = str(ticker)
            latest_frame = group.tail(seq_len)
            latest_date = candidate_date

    if latest_frame is None or latest_ticker is None or latest_date is None:
        return {"status": "INFERENCE_RUNTIME_FAIL", "reason": "feature finite sequence not found"}

    values = latest_frame[feature_columns].astype("float32").to_numpy()
    finite = bool(pd.DataFrame(values).map(math.isfinite).all().all())
    if not finite:
        return {"status": "INFERENCE_RUNTIME_FAIL", "reason": "non-finite feature sequence"}

    return {
        "status": "OK",
        "ticker": latest_ticker,
        "ticker_id": mapping.get(latest_ticker, 0),
        "date": str(latest_date.date()),
        "values": values,
    }


def dry_run_checkpoint(slot: str, checkpoint_path: Path | None, timeframe: str, indicator_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if checkpoint_path is None:
        return {
            "slot": slot,
            "status": "CHECKPOINT_MISSING",
            "elapsed_seconds": 0.0,
        }
    if not checkpoint_path.exists():
        return {
            "slot": slot,
            "status": "CHECKPOINT_MISSING",
            "checkpoint_path": str(checkpoint_path),
            "elapsed_seconds": 0.0,
        }

    try:
        config = load_checkpoint_config(checkpoint_path)
        feature_columns = list(config.get("feature_columns") or [])
        seq_len = int(config.get("seq_len") or 0)
        horizon = int(config.get("horizon") or 0)
        if not feature_columns or seq_len <= 0 or horizon <= 0:
            return {
                "slot": slot,
                "status": "SCHEMA_MISMATCH",
                "checkpoint_path": str(checkpoint_path),
                "reason": "checkpoint feature_columns/seq_len/horizon missing",
            }

        sequence = pick_latest_sequence(
            indicator_path,
            feature_columns=feature_columns,
            seq_len=seq_len,
            registry_path=config.get("ticker_registry_path"),
        )
        if sequence.get("status") != "OK":
            return {
                "slot": slot,
                "status": sequence.get("status", "INFERENCE_RUNTIME_FAIL"),
                "checkpoint_path": str(checkpoint_path),
                "reason": sequence,
            }

        torch = load_torch_cpu()
        model, checkpoint = load_checkpoint_model(checkpoint_path)
        model.eval()
        features = torch.tensor(sequence["values"], dtype=torch.float32).unsqueeze(0)
        ticker_ids = torch.tensor([int(sequence["ticker_id"])], dtype=torch.long)
        future_cov_dim = int(config.get("future_cov_dim") or 0) if bool(config.get("use_future_covariate", False)) else 0
        with torch.no_grad():
            if future_cov_dim > 0:
                future_covariate = torch.zeros((1, horizon, future_cov_dim), dtype=torch.float32)
                output = model(features, ticker_id=ticker_ids, future_covariate=future_covariate)
            else:
                output = model(features, ticker_id=ticker_ids)

        output_fields: dict[str, Any] = {}
        for name in ("line", "lower_band", "upper_band", "warning_logit"):
            if hasattr(output, name):
                tensor = getattr(output, name)
                output_fields[name] = list(tensor.shape)

        elapsed = time.perf_counter() - started
        return {
            "slot": slot,
            "status": "OK",
            "checkpoint_path": str(checkpoint_path),
            "model": config.get("model"),
            "model_role": config.get("model_role") or config.get("output_role") or config.get("role"),
            "timeframe": timeframe,
            "horizon": horizon,
            "seq_len": seq_len,
            "feature_count": len(feature_columns),
            "dry_run_ticker": sequence["ticker"],
            "dry_run_date": sequence["date"],
            "output_fields": output_fields,
            "elapsed_seconds": round(elapsed, 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "slot": slot,
            "status": "INFERENCE_RUNTIME_FAIL",
            "checkpoint_path": str(checkpoint_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def preflight() -> dict[str, Any]:
    append_progress("stage0_contract_lock_start")
    input_summary = {
        "price": read_parquet_summary(PRICE_PARQUET),
        "indicators_1d": read_parquet_summary(INDICATOR_1D_PARQUET),
        "indicators_1w": read_parquet_summary(INDICATOR_1W_PARQUET),
    }
    serving_summary = {slot: read_parquet_summary(path, date_column="asof_date") for slot, path in SERVING_FILES.items()}
    source_hashes = {
        "1D": current_source_hash("1D"),
        "1W": current_source_hash("1W"),
    }

    target_lock = {
        "created_at": utc_now(),
        "slots": {
            "1D Line": {"source_cp": "CP175", "status": "target_locked", "auto_rebuild_target": True},
            "1D Band": {"source_cp": "CP153", "status": "target_locked", "auto_rebuild_target": True},
            "1W Band": {"source_cp": "CP178", "status": "target_locked", "auto_rebuild_target": True},
            "1W Line": {"source_cp": "Deferred", "status": "deferred", "auto_rebuild_target": False},
        },
        "input_summary": input_summary,
        "serving_summary": serving_summary,
        "source_hashes": source_hashes,
        "forbidden_actions": {
            "new_training": False,
            "new_calibration": False,
            "checkpoint_reselection": False,
            "db_write": False,
            "supabase_read_write": False,
            "composite_legacy_path": False,
            "line_1w_generation": False,
        },
    }
    write_target_lock_report(target_lock)
    append_progress("stage0_contract_lock_done", current_candidate="target_lock")

    append_progress("stage0_5_preflight_start")
    cp153 = dry_run_checkpoint("CP153 1D Band", cp153_checkpoint_path(), "1D", INDICATOR_1D_PARQUET)
    cp178_paths = cp178_checkpoint_paths()
    cp178_results = [
        dry_run_checkpoint(f"CP178 1W Band checkpoint {index + 1}", path, "1W", INDICATOR_1W_PARQUET)
        for index, path in enumerate(cp178_paths)
    ]
    cp175 = dry_run_checkpoint("CP175 1D Line", None, "1D", INDICATOR_1D_PARQUET)
    cp178_ok = len(cp178_results) == 9 and all(row.get("status") == "OK" for row in cp178_results)
    cp153_ok = cp153.get("status") == "OK"
    cp175_ok = cp175.get("status") == "OK"

    if cp175_ok and cp153_ok and cp178_ok:
        overall = "PASS_CP206_INFERENCE_REBUILD_READY"
    elif (not cp175_ok) and cp153_ok and cp178_ok:
        overall = "WARN_CP206_LINE_REBUILD_BLOCKED_BAND_ONLY"
    else:
        overall = "BLOCKED_CP206_INFERENCE_PATH_BROKEN"

    preflight_payload = {
        "created_at": utc_now(),
        "overall_status": overall,
        "line_fallback": not cp175_ok,
        "cp175": cp175,
        "cp153": cp153,
        "cp178": {
            "checkpoint_count": len(cp178_paths),
            "ok_count": sum(1 for row in cp178_results if row.get("status") == "OK"),
            "results": cp178_results,
        },
        "runtime_profile": build_runtime_profile(cp178_results),
        "policy": {
            "cp175_failure_action": "기존 frozen parquet 유지 및 날짜 cutoff 이동 fallback",
            "cp153_or_cp178_failure_action": "전체 자동 갱신 경로 중단 후 보고",
            "actual_h5_return_recent_policy": "최근 5 거래일은 NaN 유지, 0 대입 금지",
            "partial_failure_policy": "성공 ticker 반영, 실패 ticker 이전 값 유지, 성공 ticker 비율 80% 미만이면 WARN 또는 ABORT",
        },
    }
    write_preflight_report(preflight_payload)
    write_runtime_profile(preflight_payload["runtime_profile"])
    write_local_contract_report(preflight_payload)
    write_latest_report(preflight_payload, target_lock)
    write_json(METRICS_PATH, {"target_lock": target_lock, "preflight": preflight_payload})
    append_progress("stage0_5_preflight_done", current_candidate=overall, last_metric_snapshot={"overall_status": overall})
    return preflight_payload


def build_runtime_profile(cp178_results: list[dict[str, Any]]) -> dict[str, Any]:
    times = [float(row.get("elapsed_seconds") or 0.0) for row in cp178_results if row.get("status") == "OK"]
    checkpoint_count = len(cp178_results)
    avg_time = sum(times) / len(times) if times else None
    return {
        "scope": "Stage 0.5 one-sample CPU dry-run",
        "checkpoint_count": checkpoint_count,
        "ok_checkpoint_count": len(times),
        "total_elapsed_seconds": round(sum(times), 4),
        "avg_checkpoint_elapsed_seconds": round(avg_time, 4) if avg_time is not None else None,
        "estimated_full_cycle_eta": "미측정. Stage 0.5는 1 ticker/latest date dry-run이라 전체 500 ticker ETA로 외삽하지 않는다.",
        "device": "CPU dry-run",
        "note": "CP178 본 inference는 9 checkpoint ensemble 구조다. 전체 cycle ETA는 실제 full forward 실행 시 별도 측정해야 한다.",
    }


def write_target_lock_report(payload: dict[str, Any]) -> None:
    lines = [
        "# CP206 Target Lock Report",
        "",
        "## 결론",
        "",
        "CP206 대상 슬롯은 1D Line=CP175, 1D Band=CP153, 1W Band=CP178, 1W Line=Deferred로 잠갔다.",
        "새 학습, 새 calibration, checkpoint 재선택, DB/Supabase read/write, composite legacy 경로는 금지한다.",
        "",
        "## 입력 parquet",
        "",
        "| 입력 | 상태 | rows | tickers | max_date | path |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, info in payload["input_summary"].items():
        lines.append(
            f"| {name} | {info.get('exists')} | {info.get('rows', '')} | {info.get('ticker_count', '')} | {info.get('max_date', '')} | `{info.get('path')}` |"
        )
    lines.extend([
        "",
        "## Serving parquet",
        "",
        "| slot | exists | rows | tickers | max_asof_date | path |",
        "|---|---:|---:|---:|---|---|",
    ])
    for name, info in payload["serving_summary"].items():
        lines.append(
            f"| {name} | {info.get('exists')} | {info.get('rows', '')} | {info.get('ticker_count', '')} | {info.get('max_date', '')} | `{info.get('path')}` |"
        )
    TARGET_LOCK_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preflight_report(payload: dict[str, Any]) -> None:
    cp178 = payload["cp178"]
    lines = [
        "# CP206 Frozen Inference Rebuild Preflight",
        "",
        f"- 최종 상태: `{payload['overall_status']}`",
        f"- CP175 line fallback: `{payload['line_fallback']}`",
        "",
        "## 슬롯별 결과",
        "",
        "| slot | status | checkpoint | dry_run_ticker | dry_run_date | output |",
        "|---|---|---|---|---|---|",
    ]
    for row in [payload["cp175"], payload["cp153"]]:
        lines.append(
            f"| {row.get('slot')} | {row.get('status')} | `{row.get('checkpoint_path', '')}` | {row.get('dry_run_ticker', '')} | {row.get('dry_run_date', '')} | `{row.get('output_fields', '')}` |"
        )
    for row in cp178["results"]:
        lines.append(
            f"| {row.get('slot')} | {row.get('status')} | `{row.get('checkpoint_path', '')}` | {row.get('dry_run_ticker', '')} | {row.get('dry_run_date', '')} | `{row.get('output_fields', '')}` |"
        )
    lines.extend([
        "",
        "## 판정",
        "",
        "- CP175는 현재 checkpoint artifact가 확인되지 않아 자동 재생성 경로를 열지 않는다.",
        "- CP153/CP178은 checkpoint 존재와 현재 local parquet feature schema 및 1 ticker 최신 날짜 forward dry-run을 통과해야 band 자동 갱신 대상으로 본다.",
        "- CP153 또는 CP178 중 하나라도 막히면 `BLOCKED_CP206_INFERENCE_PATH_BROKEN`으로 보고하고 Stage 2 full refresh를 진행하지 않는다.",
        "",
        "## actual return 정책",
        "",
        "- 최근 5 거래일의 `actual_h5_return`은 미래값이 닫히지 않았으므로 NaN으로 둔다.",
        "- 1W band의 actual return도 horizon 미래 구간이 닫히기 전에는 NaN으로 둔다.",
        "- 조용한 0 대입은 금지한다.",
    ])
    PREFLIGHT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_profile(payload: dict[str, Any]) -> None:
    lines = [
        "# CP206 1W Band Runtime Profile",
        "",
        f"- 측정 범위: {payload['scope']}",
        f"- checkpoint 수: {payload['checkpoint_count']}",
        f"- 성공 checkpoint 수: {payload['ok_checkpoint_count']}",
        f"- 총 dry-run 시간: {payload['total_elapsed_seconds']}초",
        f"- checkpoint당 평균 dry-run 시간: {payload['avg_checkpoint_elapsed_seconds']}초",
        f"- 장치: {payload['device']}",
        "",
        "## 해석",
        "",
        payload["note"],
        payload["estimated_full_cycle_eta"],
        "",
        "전체 cron 주기와 chunk 전략은 full forward 실행 시간 측정 후 다시 정해야 한다.",
    ]
    RUNTIME_PROFILE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local_contract_report(payload: dict[str, Any]) -> None:
    lines = [
        "# CP206 Local Refresh Contract",
        "",
        "## 실행 순서",
        "",
        "1. yfinance 500 daily append를 실행한다.",
        "2. append 결과를 확인한다.",
        "3. Stage 0.5 preflight 결과에 따라 frozen prediction refresh 또는 fallback을 수행한다.",
        "4. serving parquet를 재빌드한다.",
        "5. `product_prediction_history_1D.parquet`를 재빌드한다.",
        "6. local mode에서는 cache reload를 수행하고, render mode에서는 commit/push 후 redeploy로 반영한다.",
        "",
        "## cache reload",
        "",
        "- local mode: 파일 mtime 기반 lazy reload를 허용하고, 명시적 `POST /api/v1/admin/reload` endpoint를 제공한다.",
        "- render mode: redeploy가 restart이므로 별도 mtime watch를 강제하지 않는다.",
        "",
        "## 부분 실패 정책",
        "",
        "- 일부 ticker append 실패는 허용한다.",
        "- 성공 ticker는 갱신 반영하고 실패 ticker는 이전 값을 유지한다.",
        "- 성공 ticker 비율이 80% 미만이면 시스템 장애 가능성이 있으므로 WARN 또는 ABORT로 둔다.",
        "- append 실패 ticker 수, inference 실패 ticker 수, 이전 값 유지 ticker 수를 매 run 기록한다.",
        "",
        "## actual return 정책",
        "",
        "- 1D line의 최근 5 거래일 `actual_h5_return`은 NaN으로 둔다.",
        "- 1D/1W band도 horizon 미래 구간이 닫히지 않았으면 actual return을 NaN으로 둔다.",
        "- 조용한 0 대입은 금지한다.",
        "",
        "## 정적 payload 금지",
        "",
        "- `backend/scripts/build_v1_predictions_local.py`는 CP204 import package를 잘라 쓰는 fallback 도구다.",
        "- CP206 자동 갱신 본체는 최신 parquet 기준 frozen inference payload를 새로 만들어야 한다.",
        "- 그래서 orchestration 스크립트는 명시적 `-AllowStaticFallbackRebuild` 없이는 정적 payload rebuild를 실행하지 않는다.",
        "",
        "## line fallback",
        "",
        payload["policy"]["cp175_failure_action"],
    ]
    LOCAL_CONTRACT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latest_report(preflight_payload: dict[str, Any], target_lock: dict[str, Any]) -> None:
    serving = target_lock["serving_summary"]
    lines = [
        "# CP206 Daily Refresh Latest",
        "",
        f"- updated_at: {utc_now()}",
        f"- status: `{preflight_payload['overall_status']}`",
        f"- line fallback: `{preflight_payload['line_fallback']}`",
        "",
        "## latest asof",
        "",
        f"- line 1D latest asof_date: `{serving.get('line_1d', {}).get('max_date')}`",
        f"- band 1D latest asof_date: `{serving.get('band_1d', {}).get('max_date')}`",
        f"- band 1W latest asof_date: `{serving.get('band_1w', {}).get('max_date')}`",
        "",
        "## 이번 run에서 확인한 위험",
        "",
        "- CP175 checkpoint가 없어 line 1D 자동 재생성은 현재 막혀 있다.",
        "- CP153/CP178 band inference는 Stage 0.5 dry-run 통과 여부에 따라 자동 갱신 진입 가능성을 판단한다.",
        "- 현재 스크립트는 preflight와 계약 잠금까지 수행한다. full forward refresh는 preflight PASS 후 별도 실행으로 분리한다.",
    ]
    LATEST_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CP206 로컬 parquet 기반 v1 refresh preflight")
    parser.add_argument("--preflight-only", action="store_true", help="Stage 0/0.5와 보고서만 생성한다.")
    args = parser.parse_args()

    append_progress("started", mode="preflight_only" if args.preflight_only else "preflight")
    try:
        payload = preflight()
        append_progress("completed", current_candidate=payload["overall_status"])
    except Exception as exc:  # noqa: BLE001
        append_progress("failed", risk=repr(exc))
        raise


if __name__ == "__main__":
    main()
