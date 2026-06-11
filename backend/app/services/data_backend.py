"""CP254 — 서빙 read 경로 백엔드 토글 단일 정본.

"이 read 를 Supabase REST 로 보낼 것인가"는 전부 use_supabase() 하나로 판단한다.
read-site 가 env 를 직접 해석하는 것 금지 — 분기 조건이 흩어지면 컷오버/롤백이
한 줄로 안 끝난다.

판정 순서:
1) LENS_DATA_BACKEND=local|parquet|snapshot 또는 LENS_REQUIRE_LOCAL_SNAPSHOTS truthy
   → 로컬 강제 (collector 의 local_snapshots_required 와 동일 해석 재사용).
2) LENS_USE_LOCAL_SNAPSHOTS truthy → 로컬 강제. .env.example 의 문서화된 의미
   ("1 이면 SUPABASE 대신 local parquet, start_demo.ps1 이 기본 주입") 보존 —
   이게 없으면 .env 에 SUPABASE 키가 있는 채로 로컬 데모를 띄울 때 REST 로 샌다.
3) LENS_FORCE_LOCAL truthy → 로컬 (app.db.supabase_is_configured 가 False 반환).
4) SUPABASE_URL + SUPABASE_KEY 둘 다 구성 → Supabase REST.
5) 그 외 → 로컬 parquet.

컷오버(CP254 P6): Render 에 SUPABASE_URL/KEY 두 개만 넣으면 REST 로 돈다
(Render 에는 위 로컬 강제 플래그가 없음). 문제 시 LENS_FORCE_LOCAL=1 한 줄로
parquet 즉시 복귀 (탈출구 상시 유지).
"""

from __future__ import annotations

import os

from app.db import supabase_is_configured

try:
    from collector.repositories.local_snapshots import local_snapshots_required
except ModuleNotFoundError:  # pragma: no cover — 배포 rootDir(backend) 차이 흡수
    from backend.collector.repositories.local_snapshots import local_snapshots_required

# v1 서빙 데이터 출처 (EODHD 파이프라인, Plan v3 결정). REST source 필터 정본 —
# MARKET_DATA_PROVIDER 기본값(yfinance)에 서빙 경로가 휘둘리지 않게 명시 고정.
SERVING_SOURCE = "eodhd"

_TRUTHY = {"1", "true", "yes", "on"}  # collector local_snapshots 와 동일 집합


def use_supabase() -> bool:
    """서빙 read 가 Supabase REST 를 타야 하면 True."""
    if local_snapshots_required():
        return False
    if os.environ.get("LENS_USE_LOCAL_SNAPSHOTS", "").strip().lower() in _TRUTHY:
        return False
    return supabase_is_configured()
