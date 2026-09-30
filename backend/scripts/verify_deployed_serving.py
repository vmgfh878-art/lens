"""배포 데이터의 출처·버전·가격 및 예측 최신성을 읽기 전용으로 검수한다.

정상 원격 버전, 고정 폴백, 이전 버전 유지, 오래된 데이터, 버전 불일치를 구분한다.
폴백이 정상 응답해도 새 데이터 발행 성공으로 처리하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = os.environ.get("LENS_DEPLOYED_BACKEND_URL", "https://lens-backend-7stj.onrender.com").rstrip(
    "/"
)
TICKER = os.environ.get("LENS_VERIFY_TICKER", "AAPL")
GAP_1D_MAX = 5
GAP_1W_MAX = 10
GAP_PRICE_MAX = 5
EXIT_CODES = {
    "VERIFIED": 0,
    "ERROR": 1,
    "STALE": 2,
    "FALLBACK": 3,
    "MISMATCH": 4,
    "R2_DEGRADED": 5,
    "REMOTE_DEGRADED": 5,
    "LEGACY": 6,
}


def _get_json(url: str, attempts: int = 3, timeout: int = 30) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "lens-verify"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts - 1:
                raise
        except Exception:
            if attempt == attempts - 1:
                raise
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("배포 상태를 조회하지 못했습니다.")


def _business_days(d_from: str | None, d_to: str | None) -> int | None:
    if not d_from or not d_to:
        return None
    start = date.fromisoformat(d_from[:10])
    end = date.fromisoformat(d_to[:10])
    count = 0
    while start < end:
        start += timedelta(days=1)
        count += int(start.weekday() < 5)
    return count


def verify_deployed_serving(
    *,
    base_url: str = BASE,
    ticker: str = TICKER,
    expected_snapshot_id: str | None = None,
    get_json: Any = _get_json,
    today: date | None = None,
) -> dict[str, Any]:
    """상태와 실제 가격·예측 응답을 함께 검사해 구조화된 결과를 반환한다."""
    current_date = today or datetime.now(timezone.utc).date()
    base_url = base_url.rstrip("/")
    result: dict[str, Any] = {
        "result": "ERROR",
        "source": None,
        "snapshot_id": None,
        "expected_snapshot_id": expected_snapshot_id,
        "expected_snapshot_match": None,
        "r2_sync_status": None,
        "last_r2_success_at": None,
        "remote_sync_status": None,
        "last_remote_success_at": None,
        "fallback_active": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        health = get_json(f"{base_url}/api/v1/health/ready")["data"]
        source = health.get("source")
        result.update(
            source=source,
            snapshot_id=health.get("snapshot_id"),
            health_status=health.get("status"),
            r2_sync_status=health.get("r2_sync_status"),
            last_r2_success_at=health.get("last_r2_success_at"),
            remote_sync_status=health.get("remote_sync_status") or health.get("r2_sync_status"),
            last_remote_success_at=health.get("last_remote_success_at")
            or health.get("last_r2_success_at"),
            fallback_active=source == "bootstrap",
        )
        if expected_snapshot_id:
            result["expected_snapshot_match"] = health.get("snapshot_id") == expected_snapshot_id
        checks = health.get("checks") or {}
        if source in {"r2", "github_release", "bootstrap"} and not checks.get("required_files"):
            result["reason"] = "필수 서빙 파일이 준비되지 않았습니다."
            return result
        if source == "bootstrap" and not checks.get("bootstrap_verified"):
            result["reason"] = "고정 폴백의 해시 검증이 완료되지 않았습니다."
            return result

        start = (current_date - timedelta(days=60)).isoformat()
        end = (current_date + timedelta(days=2)).isoformat()
        prices = get_json(
            f"{base_url}/api/v1/stocks/{ticker}/prices?timeframe=1D&start={start}&end={end}"
        )["data"]["data"]
        price = max((row["date"] for row in prices if row.get("date")), default=None)
        latest: dict[str, str | None] = {}
        for key, endpoint in (
            ("band_1d", f"band/1d/{ticker}?days=45"),
            ("band_1w", f"band/1w/{ticker}?days=90"),
            ("line", f"line/{ticker}?days=45"),
        ):
            rows = get_json(f"{base_url}/api/v1/predictions/{endpoint}")["data"]["data"]
            latest[key] = max(
                (row["asof_date"] for row in rows if row.get("asof_date")), default=None
            )
        gaps = {key: _business_days(value, price) for key, value in latest.items()}
        price_age = _business_days(price, current_date.isoformat())
        fresh = (
            price is not None
            and price_age is not None
            and price_age <= GAP_PRICE_MAX
            and all(value is not None for value in latest.values())
            and gaps["band_1d"] is not None
            and gaps["band_1d"] <= GAP_1D_MAX
            and gaps["line"] is not None
            and gaps["line"] <= GAP_1D_MAX
            and gaps["band_1w"] is not None
            and gaps["band_1w"] <= GAP_1W_MAX
        )
        result.update(
            price=price,
            price_age_business_days=price_age,
            predictions=latest,
            prediction_gaps=gaps,
            data_fresh=fresh,
        )
        if not fresh:
            result.update(
                result="STALE", reason="가격 또는 예측이 비어 있거나 최신성 기준을 넘었습니다."
            )
        elif source == "bootstrap":
            result.update(result="FALLBACK", reason="검증된 고정 폴백으로 응답하고 있습니다.")
        elif source not in {"r2", "github_release"}:
            result.update(
                result="LEGACY", reason="배포가 아직 원격 스냅샷 서빙 출처를 보고하지 않습니다."
            )
        elif not health.get("snapshot_id"):
            result.update(result="ERROR", reason="원격 서빙 스냅샷 ID가 없습니다.")
        elif expected_snapshot_id and not result["expected_snapshot_match"]:
            result.update(result="MISMATCH", reason="활성 버전이 발행한 스냅샷과 다릅니다.")
        elif health.get("status") != "ok":
            result.update(
                result="R2_DEGRADED" if source == "r2" else "REMOTE_DEGRADED",
                reason="동기화 실패 후 이전 원격 버전으로 응답하고 있습니다.",
            )
        else:
            result.update(result="VERIFIED", reason="원격 데이터 출처와 최신성을 확인했습니다.")
    except Exception as exc:
        # URL이나 자격증명 문자열을 알림에 흘리지 않고 오류 종류만 기록한다.
        result.update(
            result="ERROR",
            reason="배포 상태 또는 데이터 응답을 확인하지 못했습니다.",
            error_type=type(exc).__name__,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="배포 서빙 출처·버전·최신성 검수")
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--expected-snapshot-id")
    parser.add_argument("--output", help="구조화된 결과를 저장할 JSON 파일")
    args = parser.parse_args(argv)
    result = verify_deployed_serving(
        base_url=args.base_url, ticker=args.ticker, expected_snapshot_id=args.expected_snapshot_id
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(
        f"VERIFY result={result['result']} source={result['source']} snapshot_id={result['snapshot_id']} "
        f"remote_sync_status={result['remote_sync_status']} price={result.get('price')} "
        f"expected_match={result['expected_snapshot_match']}"
    )
    return EXIT_CODES[result["result"]]


if __name__ == "__main__":
    sys.exit(main())
