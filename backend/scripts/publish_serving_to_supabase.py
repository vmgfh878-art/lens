"""CP255 Phase 4a — 서빙 데이터 DB 발행 단일 진입점 (import 증분 + retention).

설계 의도 — **운영 윈도우 push 경로(run_v1_unified_refresh_local.ps1 + v1_refresh_push.ps1)를
건드리지 않는다.** 그 경로는 6/11~17 프로덕션 8일 정체 사고를 막 수습한 민감 영역이고 21일
제출이 거기 달렸다. DB 발행은 이 독립 스크립트로 완전히 분리해, 컷오버 시점에만 ps1 끝에
한 줄(또는 GitHub Actions가 호출)로 붙인다.

하는 일 (멱등, 순서 고정):
  1. import_v1_serving_to_supabase.run_import — 화면용 7테이블 증분 upsert
     (product_history 제외 = 기본). --full 이 아니면 최근 윈도우만(--since).
  2. supabase_retention.run_retention — 1년 rolling 밖 행 삭제.

전제: SUPABASE_URL/KEY 가 대상 DB(클라우드 또는 로컬 셔틀)를 가리켜야 한다.
Supabase 장애/미구성 시엔 호출하지 말 것(이 스크립트는 DB가 살아있을 때만 의미).

Run:
    # 매일 증분 (최근 10일 적재 + retention)
    python backend/scripts/publish_serving_to_supabase.py
    # 첫 적재 / 전체 재동기화 (1년치)
    python backend/scripts/publish_serving_to_supabase.py --full
    # 검증만 (네트워크 변경 0)
    python backend/scripts/publish_serving_to_supabase.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CP256 — 진행 로그에 em-dash(—) 등 비-ASCII 가 있어 Windows cp949 콘솔에서
# UnicodeEncodeError 로 죽던 문제 방지. 자동화(일일 refresh)에서도 안전하게 발행되도록
# stdout/stderr 를 utf-8 로 고정한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from backend.scripts.import_v1_serving_to_supabase import run_import  # noqa: E402
from backend.scripts.supabase_retention import run_retention  # noqa: E402

# 증분 적재 lookback (일). 매일 돌면 최근 며칠이면 충분하나, 주말/공휴일/적재 지연을
# 흡수하도록 넉넉히. retention 윈도우(450/800d)보다 훨씬 작아 무료 용량에 무해.
INCREMENTAL_LOOKBACK_DAYS = 10
# 전체 재동기화(--full) 시 적재 윈도우 = retention 최장(800d, band_1w) 이상.
FULL_WINDOW_DAYS = 800


def main() -> int:
    parser = argparse.ArgumentParser(description="CP255 Phase 4a 서빙 DB 발행 (import+retention)")
    parser.add_argument("--full", action="store_true", help="증분 대신 전체 윈도우(800d) 재동기화")
    parser.add_argument(
        "--since", help="YYYY-MM-DD 명시 (이 날짜 이후만 적재). --full/기본보다 우선"
    )
    parser.add_argument(
        "--include-product-history",
        action="store_true",
        help="product_history 도 발행 (기본 제외)",
    )
    parser.add_argument("--dry-run", action="store_true", help="변경 0 — 적재/삭제 대상만 출력")
    args = parser.parse_args()

    # date.today() 는 매일 자동화의 정상 입력. (워크플로 결정성 제약과 무관한 일반 CLI)
    if args.since:
        since = args.since
    elif args.full:
        since = (date.today() - timedelta(days=FULL_WINDOW_DAYS)).isoformat()
    else:
        since = (date.today() - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).isoformat()

    print("#" * 64)
    print(
        f"# CP255 Phase 4a 서빙 DB 발행  mode={'FULL' if args.full else 'INCREMENTAL'} "
        f"since={since} dry_run={args.dry_run}"
    )
    print("#" * 64)

    # 1) 증분/전체 import
    run_import(
        since=since,
        dry_run=args.dry_run,
        include_product_history=args.include_product_history,
    )

    # 2) retention (1년 rolling 유지)
    print()
    run_retention(
        dry_run=args.dry_run,
        include_product_history=args.include_product_history,
    )

    print("\n발행 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
