from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.collector.jobs.compute_indicators import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="indicators 재계산 CLI")
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--timeframes", nargs="*", default=["1D", "1W", "1M"])
    parser.add_argument("--provider", default=None)
    parser.add_argument("--force-full-backfill", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        lookback_days=args.lookback_days,
        tickers=args.tickers,
        timeframes=args.timeframes,
        force_full_backfill=args.force_full_backfill,
        provider=args.provider,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
