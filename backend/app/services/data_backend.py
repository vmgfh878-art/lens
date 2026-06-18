"""CP254/255 — 서빙 read 경로 백엔드 선택 단일 정본.

"이 read 를 Supabase REST 로 보낼 것인가"는 전부 use_supabase() 하나로 판단한다.
read-site 가 env 를 직접 해석하는 것 금지 — 분기 조건이 흩어지면 전환이 한 곳에서 안 끝난다.

판정 순서 (위에서 걸리면 즉시 로컬 parquet):
1) LENS_DATA_BACKEND=local|parquet|snapshot 또는 LENS_REQUIRE_LOCAL_SNAPSHOTS truthy → 로컬.
2) LENS_USE_LOCAL_SNAPSHOTS truthy → 로컬 (로컬 데모 기본).
3) LENS_FORCE_LOCAL truthy → 로컬 (supabase_is_configured 가 False). **비상 강제 스위치.**
4) SUPABASE_URL+KEY 미구성 → 로컬.
5) (CP255 자동 폴백) LENS_SUPABASE_AUTOFALLBACK!=0 이고 Supabase 가 **현재 도달 불가**면 → 로컬.
6) 그 외 → Supabase REST.

★ CP255 자동 폴백 (사용자 요구: "복구되면 바로 Supabase, 죽으면 바로 윈도우, 자유롭게"):
   configured 상태에서 백엔드가 매번 env 를 안 바꿔도, Supabase REST 가 살아있으면 DB·죽으면
   parquet 으로 **자동** 전환된다. 도달성은 캐싱한다(성공 300s / 실패 20s TTL) — 매 요청 ping 으로
   인한 cold-start·egress 폭증을 막으면서, 죽으면 ≤20s 안에 parquet 으로, 복구되면 ≤20s 안에 DB 로
   돌아온다. LENS_SUPABASE_AUTOFALLBACK=0 으로 끄면 (구) "configured 면 무조건 REST" 동작.
   LENS_FORCE_LOCAL 은 자동 위의 수동 오버라이드로 항상 우선.
"""

from __future__ import annotations

import os
import time

from app.db import get_supabase, supabase_is_configured

try:
    from collector.repositories.local_snapshots import local_snapshots_required
except ModuleNotFoundError:  # pragma: no cover — 배포 rootDir(backend) 차이 흡수
    from backend.collector.repositories.local_snapshots import local_snapshots_required

# v1 서빙 데이터 출처 (EODHD 파이프라인, Plan v3 결정). REST source 필터 정본 —
# MARKET_DATA_PROVIDER 기본값(yfinance)에 서빙 경로가 휘둘리지 않게 명시 고정.
SERVING_SOURCE = "eodhd"

_TRUTHY = {"1", "true", "yes", "on"}  # collector local_snapshots 와 동일 집합

# 자동 폴백 도달성 캐시. 살아있을 땐 5분, 죽었을 땐 20초마다 재확인.
_REACHABLE_TTL_OK = 300.0
_REACHABLE_TTL_FAIL = 20.0
_reachable_cache: dict[str, float | bool] = {"ts": 0.0, "ok": False, "checked": False}


def _autofallback_enabled() -> bool:
    # 기본 ON. 명시적으로 0/false 면 (구) 무조건 REST 동작.
    return os.environ.get("LENS_SUPABASE_AUTOFALLBACK", "1").strip().lower() in _TRUTHY


def _supabase_reachable() -> bool:
    """Supabase REST 도달 가능? 결과를 TTL 캐싱해 매 요청 ping 을 피한다.

    가벼운 단일 행 SELECT 로 확인. DNS 실패/타임아웃/권한 등 어떤 예외든 '도달 불가'.
    """
    now = time.monotonic()
    cache = _reachable_cache
    ttl = _REACHABLE_TTL_OK if cache["ok"] else _REACHABLE_TTL_FAIL
    if cache["checked"] and (now - float(cache["ts"])) < ttl:
        return bool(cache["ok"])
    ok = False
    try:
        get_supabase().table("stock_info").select("ticker").limit(1).execute()
        ok = True
    except Exception:  # noqa: BLE001 — 도달성 판정이라 모든 실패는 False 로 수렴
        ok = False
    cache.update({"ts": now, "ok": ok, "checked": True})
    return ok


def reset_reachable_cache() -> None:
    """테스트/관리 reload 에서 도달성 캐시를 비운다 (다음 호출이 재확인)."""
    _reachable_cache.update({"ts": 0.0, "ok": False, "checked": False})


def supabase_reachable_now() -> bool | None:
    """진단용 — 자동 폴백이 보는 현재 도달성 (configured 아니면 None)."""
    if not supabase_is_configured():
        return None
    return _supabase_reachable()


def use_supabase() -> bool:
    """서빙 read 가 Supabase REST 를 타야 하면 True (아니면 로컬 parquet)."""
    if local_snapshots_required():
        return False
    if os.environ.get("LENS_USE_LOCAL_SNAPSHOTS", "").strip().lower() in _TRUTHY:
        return False
    if not supabase_is_configured():  # LENS_FORCE_LOCAL 포함
        return False
    if not _autofallback_enabled():
        return True
    return _supabase_reachable()
