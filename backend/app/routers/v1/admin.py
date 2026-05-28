from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.core.http import success_response
from app.db import supabase_is_configured
from app.repositories.ai_repo import _load_mock
from app.routers.v1 import predictions as v1_predictions
from app.services import local_market_svc, parquet_store
from app.services.product_prediction_history_svc import clear_product_history_cache
from app.services.strategy_backtest_svc import clear_strategy_cache


router = APIRouter(prefix="/admin", tags=["admin"])


def _is_local_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    return client_host in {"127.0.0.1", "::1", "localhost"}


def _require_reload_allowed(request: Request, token: str | None) -> None:
    expected = os.environ.get("LENS_ADMIN_RELOAD_TOKEN", "").strip()
    if expected:
        if token != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="admin reload token이 올바르지 않습니다.",
            )
        return

    allow_local = os.environ.get("LENS_ALLOW_LOCAL_ADMIN_RELOAD", "0").strip().lower() in {"1", "true", "yes"}
    if allow_local and _is_local_request(request):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="admin reload가 비활성화되어 있습니다. LENS_ADMIN_RELOAD_TOKEN을 설정하거나 로컬 개발 예외를 명시해야 합니다.",
    )


@router.post("/reload")
def reload_v1_predictions(request: Request, x_lens_admin_token: str | None = Header(default=None)):
    """로컬 v1 parquet cache를 명시적으로 다시 읽는다."""
    _require_reload_allowed(request, x_lens_admin_token)
    base_dir = Path(__file__).resolve().parents[3] / "data" / "v1"
    # Clear shared prediction parquet store first; derived caches follow.
    store_summary = parquet_store.clear_all()
    prediction_summary = v1_predictions.load_caches(base_dir)
    market_summary = local_market_svc.reload_caches()
    clear_product_history_cache()
    clear_strategy_cache()
    _load_mock.cache_clear()
    return success_response(
        request,
        {
            "reloaded": True,
            "base_dir": str(base_dir),
            "parquet_store": store_summary,
            "predictions": prediction_summary,
            "market": market_summary,
            "product_history_cache": "cleared",
            "strategy_cache": "cleared",
            "ai_runs_mock_cache": "cleared",
        },
    )


@router.get("/debug-state")
def debug_state(request: Request):
    """503 원인 진단용. 환경변수 KEY 존재 여부 / parquet 파일 / 메모리 / market 경로 dump.

    sensitive value 노출 X — KEY 존재 여부와 path / size 만 보고. 인증 없이 호출 가능.
    """
    base_dir = Path(__file__).resolve().parents[3] / "data" / "v1"

    # 1) parquet 파일 점검
    parquet_files: dict[str, dict] = {}
    if base_dir.exists():
        for p in sorted(base_dir.glob("*.parquet")):
            try:
                size = p.stat().st_size
                parquet_files[p.name] = {
                    "exists": True,
                    "size_mb": round(size / 1024 / 1024, 2),
                }
            except Exception as exc:  # noqa: BLE001
                parquet_files[p.name] = {"exists": False, "error": str(exc)}

    # 2) 관심 환경변수 (값 노출 X, 존재 여부만)
    interesting_keys = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "LENS_FORCE_LOCAL",
        "LENS_EAGER_V1_CACHE",
        "LENS_ADMIN_RELOAD_TOKEN",
        "LENS_ALLOW_LOCAL_ADMIN_RELOAD",
        "MARKET_DATA_PROVIDER",
        "BACKEND_CORS_ORIGINS",
        "PYTHONPATH",
        "PORT",
        "RENDER",
        "RENDER_SERVICE_NAME",
    ]
    interesting_env = {
        k: ("set" if os.environ.get(k) else "empty") for k in interesting_keys
    }

    # 3) 메모리 (Linux /proc/self/status 또는 resource fallback)
    memory: dict[str, str | float] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmPeak:", "VmSize:")):
                    parts = line.split(":")
                    memory[parts[0]] = parts[1].strip()
    except Exception:
        try:
            import resource  # type: ignore

            ru = resource.getrusage(resource.RUSAGE_SELF)
            memory["max_rss_mb"] = round(ru.ru_maxrss / 1024, 2)
        except Exception as exc:  # noqa: BLE001
            memory["error"] = str(exc)

    # 4) market 경로 probe — 실제 lazy-load 가 어떻게 행동하는지
    market_probes: dict[str, dict] = {}
    for slot, getter in (
        ("prices_1d", local_market_svc.get_prices_1d),
        ("prices_1w", local_market_svc.get_prices_1w),
        ("indicators_1d", local_market_svc.get_indicators_1d),
        ("stock_info", local_market_svc.get_stock_info),
    ):
        try:
            df = getter()
            if df is None:
                market_probes[slot] = {"status": "missing"}
            else:
                market_probes[slot] = {
                    "status": "loaded",
                    "rows": int(len(df)),
                    "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
                }
        except Exception as exc:  # noqa: BLE001
            market_probes[slot] = {
                "status": "error",
                "exc_type": type(exc).__name__,
                "exc_msg": str(exc)[:500],
                "traceback_tail": traceback.format_exc()[-1000:],
            }

    return success_response(
        request,
        {
            "base_dir": str(base_dir),
            "base_dir_exists": base_dir.exists(),
            "parquet_files": parquet_files,
            "supabase_is_configured": supabase_is_configured(),
            "interesting_env": interesting_env,
            "memory": memory,
            "market_probes": market_probes,
            "parquet_store": parquet_store.stats(),
            "sys_path_first5": sys.path[:5],
            "python_version": sys.version,
        },
    )
