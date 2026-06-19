"""CP254 P5 — 로컬 Supabase 리허설 검증 하니스.

parquet 모드와 REST 모드(로컬 PostgREST 스택)를 같은 프로세스에서 번갈아 띄워
(1) 엔드포인트 parity (값 동일), (2) 응답 bytes(클라이언트측 egress),
(3) REST 입력 egress(셔틀 /_shim/stats), (4) 폴백(LENS_FORCE_LOCAL=1),
(5) RSS 를 실측한다.

실행 전제: .lens_local_db 스택 기동 + import_v1_serving_to_supabase.py 적재 완료,
SUPABASE_URL=http://127.0.0.1:54321, SUPABASE_KEY=<로컬 anon> env 설정.

사용:
    python backend/scripts/cp254_rehearsal_check.py [--skip-scan]

모드 전환은 같은 프로세스에서 env 변경 + 캐시 클리어로 한다 (TestClient).
RSS 는 각 모드 직후 psutil 로 기록 — 단일 프로세스라 "REST 모드가 parquet 를
안 올렸을 때의 증분"으로 해석한다 (절대값은 별도 프로세스 측정이 정확).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

SHIM = "http://127.0.0.1:54321"

# (이름, 경로, params) — 데모 1세션 구성 (§5 egress 예산 시나리오 포함)
CASES: list[tuple[str, str, dict]] = [
    ("stocks_list", "/api/v1/stocks", {"limit": 50}),
    ("stocks_search", "/api/v1/stocks", {"search": "AA", "limit": 10}),
    ("prices_1d", "/api/v1/stocks/AAPL/prices", {"start": "2025-06-10", "end": "2026-06-10"}),
    (
        "prices_1w",
        "/api/v1/stocks/AAPL/prices",
        {"start": "2025-06-10", "end": "2026-06-10", "timeframe": "1W"},
    ),
    ("indicators", "/api/v1/stocks/AAPL/indicators", {"limit": 300}),
    ("product_history_default", "/api/v1/stocks/AAPL/predictions/product-history", {}),
    (
        "product_history_line100",
        "/api/v1/stocks/AAPL/predictions/product-history",
        {"roles": "line", "limit": 100},
    ),
    ("pred_line", "/api/v1/predictions/line/AAPL", {"days": 370}),
    ("pred_band_1d", "/api/v1/predictions/band/1d/AAPL", {"days": 370}),
    ("pred_band_1d_h5", "/api/v1/predictions/band/1d/AAPL", {"days": 370, "horizon": 5}),
    ("pred_band_1w", "/api/v1/predictions/band/1w/AAPL", {"days": 730}),
    ("pred_tickers", "/api/v1/predictions/tickers", {"limit": 500}),
]
SCAN_CASES: list[tuple[str, str, dict]] = [
    ("scan_indicator", "/api/v1/strategies/indicator_balance_v2/scan", {"limit": 500}),
    ("backtest_aapl", "/api/v1/strategies/indicator_balance_v2/backtest/AAPL", {}),
]

# 알려진 의도적 차이 (보고서에 기록):
#  - pred_tickers: 로컬=line_1d 유니버스 / REST=stock_info 유니버스 (프론트 미사용)
EXPECTED_DIFF_CASES = {"pred_tickers"}

# 필드 한정 허용 차이: indicators.volume — 로컬 parquet 엔 volume 컬럼이 없어
# 항상 None, REST 는 DB 컬럼을 제공 (의도된 개선, IndicatorPoint 계약 내).
import re  # noqa: E402

_ALLOWED_DIFF_PATTERNS: dict[str, re.Pattern] = {
    "indicators": re.compile(r"^\$\.data\[\d+\]\.volume: (type NoneType != int|None != \d+)$"),
}


def _clear_all_caches() -> None:
    from app.repositories.ai_repo import _load_mock
    from app.services import local_market_svc, parquet_store
    from app.services.product_prediction_history_svc import clear_product_history_cache
    from app.services.strategy_backtest_svc import clear_strategy_cache

    parquet_store.clear_all()
    local_market_svc.clear_caches()
    clear_product_history_cache()
    clear_strategy_cache()
    _load_mock.cache_clear()


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1024 / 1024


def _run_mode(client, cases) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, path, params in cases:
        resp = client.get(path, params=params, headers={"X-Request-Id": "parity-fixed"})
        body = resp.json() if resp.status_code == 200 else {"error": resp.status_code}
        data = body.get("data") if isinstance(body, dict) else body
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        out[name] = {
            "status": resp.status_code,
            "data": data,
            "bytes": len(raw.encode("utf-8")),
            "gzip_bytes": len(gzip.compress(raw.encode("utf-8"), 6)),
            "etag": resp.headers.get("etag"),
            "cache_control": resp.headers.get("cache-control"),
        }
    return out


def _diff(a: Any, b: Any, path="$") -> list[str]:
    diffs: list[str] = []
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                diffs.append(f"{path}.{k}: missing in LOCAL")
            elif k not in b:
                diffs.append(f"{path}.{k}: missing in REST")
            else:
                diffs.extend(_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_diff(x, y, f"{path}[{i}]"))
            if len(diffs) > 20:
                return diffs
    else:
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
                diffs.append(f"{path}: {a!r} != {b!r}")
        elif a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def _shim_stats(reset: bool = False) -> dict:
    try:
        if reset:
            httpx.post(f"{SHIM}/_shim/reset", timeout=5)
            return {}
        return httpx.get(f"{SHIM}/_shim/stats", timeout=5).json()
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scan", action="store_true", help="전종목 scan 케이스 생략")
    args = parser.parse_args()
    cases = CASES + ([] if args.skip_scan else SCAN_CASES)

    if not os.environ.get("SUPABASE_URL", "").startswith("http://127.0.0.1"):
        print("[ABORT] SUPABASE_URL 이 로컬 셔틀(127.0.0.1:54321)이 아니다 — 클라우드 보호.")
        return 2

    from starlette.testclient import TestClient

    from app.main import app
    from app.services import data_backend

    client = TestClient(app)

    # ---------- 1) parquet(local) 모드 ----------
    os.environ["LENS_FORCE_LOCAL"] = "1"
    assert not data_backend.use_supabase()
    _clear_all_caches()
    local = _run_mode(client, cases)
    rss_local = _rss_mb()

    # ---------- 2) REST 모드 ----------
    os.environ.pop("LENS_FORCE_LOCAL", None)
    os.environ.pop("LENS_USE_LOCAL_SNAPSHOTS", None)
    os.environ.pop("LENS_DATA_BACKEND", None)
    assert data_backend.use_supabase(), "REST 모드 전환 실패 (env 확인)"
    _clear_all_caches()
    _shim_stats(reset=True)
    rss_before_rest = _rss_mb()
    rest = _run_mode(client, cases)
    rss_rest = _rss_mb()
    shim = _shim_stats()

    # ---------- 3) 폴백 모드 (다시 parquet) ----------
    os.environ["LENS_FORCE_LOCAL"] = "1"
    _clear_all_caches()
    fallback = _run_mode(client, cases)
    fallback_ok = all(fallback[name]["status"] == local[name]["status"] for name, _, _ in cases)

    # ---------- 보고 ----------
    print("=" * 88)
    print(" CP254 P5 리허설 결과 (parquet vs 로컬 Supabase REST)")
    print("=" * 88)
    header = f"{'case':<26}{'st(L/R)':<10}{'bytes(L)':>10}{'bytes(R)':>10}{'gzip(R)':>9}  parity"
    print(header)
    print("-" * 88)
    failures: list[str] = []
    for name, _, _ in cases:
        l, r = local[name], rest[name]
        diffs = _diff(l["data"], r["data"])
        allowed = _ALLOWED_DIFF_PATTERNS.get(name)
        if allowed is not None:
            tolerated = [d for d in diffs if allowed.match(d)]
            diffs = [d for d in diffs if not allowed.match(d)]
        else:
            tolerated = []
        if name in EXPECTED_DIFF_CASES:
            verdict = "KNOWN-DIFF" if diffs else "OK"
        elif l["status"] != r["status"]:
            verdict = "STATUS-DIFF"
            failures.append(name)
        elif diffs:
            verdict = f"DIFF({len(diffs)})"
            failures.append(name)
        elif tolerated:
            verdict = f"OK(+{len(tolerated)} allowed: volume)"
        else:
            verdict = "OK"
        print(
            f"{name:<26}{str(l['status']) + '/' + str(r['status']):<10}"
            f"{l['bytes']:>10,}{r['bytes']:>10,}{r['gzip_bytes']:>9,}  {verdict}"
        )
        if diffs and name not in EXPECTED_DIFF_CASES:
            for d in diffs[:5]:
                print(f"    - {d}")

    print("-" * 88)
    total_l = sum(local[n]["bytes"] for n, _, _ in cases)
    total_r = sum(rest[n]["bytes"] for n, _, _ in cases)
    total_rz = sum(rest[n]["gzip_bytes"] for n, _, _ in cases)
    print(f"{'TOTAL(데모 1세션)':<26}{'':<10}{total_l:>10,}{total_r:>10,}{total_rz:>9,}")

    if shim:
        print("\nREST 입력 egress (셔틀 실측 — backend↔PostgREST 구간, scan 입력 포함):")
        for table, s in sorted(shim.get("per_table", {}).items()):
            print(f"  {table:<34} req={int(s['requests']):>5,}  bytes={int(s['bytes']):>13,}")
        t = shim.get("total", {})
        print(
            f"  {'TOTAL':<34} req={int(t.get('requests', 0)):>5,}  bytes={int(t.get('bytes', 0)):>13,}"
        )

    print(f"\n폴백(LENS_FORCE_LOCAL=1) 전 엔드포인트 정상: {fallback_ok}")
    print(
        f"RSS: parquet 모드 직후 {rss_local:.0f}MB · REST 직전 {rss_before_rest:.0f}MB · "
        f"REST 직후 {rss_rest:.0f}MB (단일 프로세스 증분 해석)"
    )
    print(f"\nparity 실패 케이스: {failures if failures else '없음 (0회귀)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
