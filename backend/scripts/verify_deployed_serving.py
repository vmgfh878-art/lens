"""CP256 — 배포된 프로덕션 서빙이 실제로 최신인지 검수.

git push / Supabase 발행이 끝나도, 배포 백엔드가 최신 예측을 서빙하지 않으면 웹 화면엔
안 뜬다(2026-07 예측 정체 사고). 이 스크립트는 배포된 백엔드를 직접 조회해서 프론트의
stale 가드(CP215)와 같은 기준으로 "화면에 h5(1D)/4주(1W) 최신 예측이 뜨는 상태인가"를 판정한다.

판정:
  - 1D: band_1d / line 의 최신 asof 가 가격 최신일과 5거래일 이내면 화면에 뜬다(stale 가드 기준).
  - 1W: band_1w 최신 asof 가 가격 최신일과 10거래일 이내면 뜬다(주봉이라 여유).

출력 마지막 줄: `VERIFY result=<VERIFIED|STALE|ERROR> ...` (ps1 이 파싱).
exit code: 0 VERIFIED / 2 STALE / 1 ERROR(도달 불가 등).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

BASE = os.environ.get("LENS_DEPLOYED_BACKEND_URL", "https://lens-backend-7stj.onrender.com").rstrip(
    "/"
)
TICKER = os.environ.get("LENS_VERIFY_TICKER", "AAPL")
GAP_1D_MAX = 5  # 거래일. 프론트 stale 가드 임계값과 동일.
GAP_1W_MAX = 10  # 거래일. 주봉은 여유.


def _get_json(url: str, attempts: int = 3, timeout: int = 30):
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lens-verify"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (i + 1))  # Render 콜드스타트 흡수
    raise RuntimeError(f"unreachable: {url} ({last})")


def _latest_asof(endpoint: str) -> str | None:
    rows = _get_json(f"{BASE}/api/v1/predictions/{endpoint}")["data"]["data"]
    asofs = sorted({r["asof_date"] for r in rows if r.get("asof_date")})
    return asofs[-1] if asofs else None


def _latest_price() -> str | None:
    start = (datetime.utcnow().date() - timedelta(days=60)).isoformat()
    end = (datetime.utcnow().date() + timedelta(days=2)).isoformat()
    rows = _get_json(f"{BASE}/api/v1/stocks/{TICKER}/prices?timeframe=1D&start={start}&end={end}")[
        "data"
    ]["data"]
    return rows[-1]["date"] if rows else None


def _business_days(d_from: str | None, d_to: str | None) -> int | None:
    if not d_from or not d_to:
        return None
    a = datetime.strptime(d_from[:10], "%Y-%m-%d").date()
    b = datetime.strptime(d_to[:10], "%Y-%m-%d").date()
    if a >= b:
        return 0
    n, c = 0, a
    while c < b:
        c += timedelta(days=1)
        if c.weekday() < 5:
            n += 1
    return n


def main() -> int:
    try:
        price = _latest_price()
        band_1d = _latest_asof(f"band/1d/{TICKER}?days=45")
        band_1w = _latest_asof(f"band/1w/{TICKER}?days=90")
        line = _latest_asof(f"line/{TICKER}?days=45")
    except Exception as exc:  # noqa: BLE001
        print(f'VERIFY result=ERROR reason="{exc}"')
        return 1

    gap_band_1d = _business_days(band_1d, price)
    gap_line = _business_days(line, price)
    gap_band_1w = _business_days(band_1w, price)

    ok_1d = (
        band_1d is not None
        and line is not None
        and (gap_band_1d is not None and gap_band_1d <= GAP_1D_MAX)
        and (gap_line is not None and gap_line <= GAP_1D_MAX)
    )
    ok_1w = band_1w is not None and (gap_band_1w is not None and gap_band_1w <= GAP_1W_MAX)

    verdict = "VERIFIED" if (ok_1d and ok_1w) else "STALE"
    detail = (
        f"price={price} band1d={band_1d}(gap{gap_band_1d}) "
        f"line={line}(gap{gap_line}) band1w={band_1w}(gap{gap_band_1w}) "
        f"1D_ok={int(bool(ok_1d))} 1W_ok={int(bool(ok_1w))}"
    )
    print(f"VERIFY result={verdict} {detail}")
    return 0 if verdict == "VERIFIED" else 2


if __name__ == "__main__":
    sys.exit(main())
