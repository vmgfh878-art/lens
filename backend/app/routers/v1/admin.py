from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from app.config import get_admin_config
from app.core.http import success_response
from app.db import supabase_is_configured
from app.repositories.ai_repo import _load_mock
from app.routers.v1 import predictions as v1_predictions
from app.services import data_backend, local_market_svc, parquet_store
from app.services.product_prediction_history_svc import clear_product_history_cache
from app.services.strategy_backtest_svc import clear_strategy_cache
from fastapi import APIRouter, Header, HTTPException, Request, status

logger = logging.getLogger("lens.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


def _is_local_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    return client_host in {"127.0.0.1", "::1", "localhost"}


def _require_reload_allowed(request: Request, token: str | None) -> None:
    cfg = get_admin_config()
    expected = cfg.reload_token
    if expected:
        if token != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="admin reload token이 올바르지 않습니다.",
            )
        return

    if cfg.allow_local_reload and _is_local_request(request):
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


def _debug_state_parquet_files(request: Request, base_dir: Path) -> dict[str, dict]:
    parquet_files: dict[str, dict] = {}
    if not base_dir.exists():
        return parquet_files
    for p in sorted(base_dir.glob("*.parquet")):
        try:
            size = p.stat().st_size
            parquet_files[p.name] = {
                "exists": True,
                "size_mb": round(size / 1024 / 1024, 2),
            }
        except Exception as exc:  # noqa: BLE001 — debug 엔드포인트, 원인은 로그로만
            rid = getattr(request.state, "request_id", "-")
            logger.warning("[%s] debug-state parquet stat '%s' 실패", rid, p.name, exc_info=exc)
            parquet_files[p.name] = {"exists": False}
    return parquet_files


def _debug_state_env() -> dict[str, str]:
    # 관심 환경변수 (값 노출 X, 존재 여부만)
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
    return {k: ("set" if os.environ.get(k) else "empty") for k in interesting_keys}


def _debug_state_memory(request: Request) -> dict[str, str | float]:
    # Linux /proc/self/status 또는 resource fallback
    memory: dict[str, str | float] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmPeak:", "VmSize:")):
                    parts = line.split(":")
                    memory[parts[0]] = parts[1].strip()
        return memory
    except Exception:
        pass
    try:
        import resource  # type: ignore

        ru = resource.getrusage(resource.RUSAGE_SELF)
        memory["max_rss_mb"] = round(ru.ru_maxrss / 1024, 2)
    except Exception as exc:  # noqa: BLE001 — debug 엔드포인트, 원인은 로그로만
        rid = getattr(request.state, "request_id", "-")
        logger.warning("[%s] debug-state memory probe 실패", rid, exc_info=exc)
    return memory


def _debug_state_market_probes(request: Request) -> dict[str, dict]:
    # market 경로 probe — 실제 lazy-load 가 어떻게 행동하는지
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
        except Exception as exc:  # noqa: BLE001 — debug 엔드포인트, 원인은 로그로만
            rid = getattr(request.state, "request_id", "-")
            logger.warning("[%s] debug-state market probe '%s' 실패", rid, slot, exc_info=exc)
            market_probes[slot] = {"status": "error"}
    return market_probes


def _debug_state_supabase_probe() -> dict[str, object]:
    # CP256 컷오버 진단 — 어느 프로젝트(host)·키 role·실패 원인(예외)을 노출. 키 값은 안 보인다.
    supabase_probe: dict[str, object] = {}
    try:
        import base64 as _b64
        import json as _json
        from urllib.parse import urlparse as _urlparse

        from app.config import get_database_config as _get_db_cfg
        from app.db import get_supabase as _get_supabase

        _cfg = _get_db_cfg()
        supabase_probe["url_host"] = _urlparse(_cfg.supabase_url or "").netloc or None
        try:
            _seg = (_cfg.supabase_key or "").split(".")[1]
            _seg += "=" * (-len(_seg) % 4)
            supabase_probe["key_role"] = _json.loads(_b64.urlsafe_b64decode(_seg)).get("role")
        except Exception:  # noqa: BLE001
            supabase_probe["key_role"] = "unparseable"
        try:
            _get_supabase().table("stock_info").select("ticker").limit(1).execute()
            supabase_probe["probe"] = "ok"
        except Exception as _exc:  # noqa: BLE001
            supabase_probe["probe"] = "error"
            supabase_probe["error"] = f"{type(_exc).__name__}: {str(_exc)[:300]}"
    except Exception as _outer:  # noqa: BLE001
        supabase_probe["meta_error"] = str(_outer)[:200]
    return supabase_probe


@router.get("/debug-state")
def debug_state(request: Request):
    """503 원인 진단용. 환경변수 KEY 존재 여부 / parquet 파일 / 메모리 / market 경로 dump.

    sensitive value 노출 X — KEY 존재 여부와 path / size 만 보고. 인증 없이 호출 가능.
    """
    base_dir = Path(__file__).resolve().parents[3] / "data" / "v1"

    return success_response(
        request,
        {
            "base_dir": str(base_dir),
            "base_dir_exists": base_dir.exists(),
            "parquet_files": _debug_state_parquet_files(request, base_dir),
            "supabase_is_configured": supabase_is_configured(),
            # CP254/255 — 서빙 read 가 실제로 타는 백엔드 + 자동 폴백이 보는 도달성
            # (전환 상태 즉시 확인용). reachable=None 이면 미구성(=항상 로컬).
            "data_backend": "supabase" if data_backend.use_supabase() else "local",
            "supabase_reachable": data_backend.supabase_reachable_now(),
            "supabase_probe": _debug_state_supabase_probe(),
            "interesting_env": _debug_state_env(),
            "memory": _debug_state_memory(request),
            "market_probes": _debug_state_market_probes(request),
            "parquet_store": parquet_store.stats(),
            "sys_path_first5": sys.path[:5],
            "python_version": sys.version,
        },
    )
