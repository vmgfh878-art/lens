"""CP254 P5 — 단일 모드 RSS 실측 (모드별 별도 프로세스로 실행해야 공정 비교).

사용:
    # parquet 모드
    LENS_FORCE_LOCAL=1 python backend/scripts/cp254_rss_probe.py
    # REST 모드 (SUPABASE_URL/KEY 설정, FORCE_LOCAL 미설정)
    python backend/scripts/cp254_rss_probe.py

데모 1세션 시나리오(서빙 12케이스 + scan/backtest)를 1회 돌리고 peak RSS 출력.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.scripts.cp254_rehearsal_check import CASES, SCAN_CASES  # noqa: E402


def rss() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def main() -> None:
    from starlette.testclient import TestClient

    from app.main import app
    from app.services import data_backend

    mode = "supabase" if data_backend.use_supabase() else "local-parquet"
    client = TestClient(app)
    print(f"mode={mode} import 직후 RSS={rss():.0f}MB")
    for name, path, params in CASES + SCAN_CASES:
        resp = client.get(path, params=params)
        print(f"  {name:<26} {resp.status_code} RSS={rss():.0f}MB")
    print(f"mode={mode} 최종 RSS={rss():.0f}MB")


if __name__ == "__main__":
    main()
