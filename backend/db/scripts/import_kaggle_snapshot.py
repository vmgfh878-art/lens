"""?몃? parquet ?ㅻ깄?룹쓣 Lens ?ㅽ궎留덈줈 ?곸옱?쒕떎."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from backend.db.bootstrap import run_bootstrap  # noqa: E402
from backend.collector.pipelines.bootstrap_snapshot import step_indicators  # noqa: E402

DEFAULT_SOURCE_DIR = Path(r"C:\Users\user\projects\sisc-web\AI\data\kaggle_data")
REQUIRED_FILES = [
    "stock_info.parquet",
    "price_data.parquet",
    "macroeconomic_indicators.parquet",
    "company_fundamentals.parquet",
]
OPTIONAL_FILES = [
    "market_breadth.parquet",
    "sector_returns.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="?몃? Kaggle parquet ?ㅻ깄?룹쓣 Lens DB濡??곸옱")
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="?먮낯 parquet ?붾젆?곕━ 寃쎈줈",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="?뱀젙 ?곗빱留??곸옱?????ъ슜",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=["1D", "1W", "1M"],
        help="indicators ?앹꽦 ??꾪봽?덉엫",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="?뚯씪 ?곹깭留??먭??섍퀬 ?곸옱???섏? ?딆쓬",
    )
    parser.add_argument(
        "--skip-indicators",
        action="store_true",
        help="raw ?곸옱留??섍퀬 indicators ?앹꽦? 嫄대꼫?",
    )
    return parser.parse_args()


def read_parquet_info(path: Path) -> dict:
    frame = pd.read_parquet(path)
    info: dict[str, object] = {
        "name": path.name,
        "rows": len(frame),
        "columns": list(frame.columns),
    }

    if "date" in frame.columns:
        series = pd.to_datetime(frame["date"], errors="coerce")
        info["date_min"] = series.min()
        info["date_max"] = series.max()

    if "ticker" in frame.columns:
        info["tickers"] = frame["ticker"].nunique(dropna=True)

    return info


def validate_source_dir(source_dir: Path) -> tuple[list[dict], list[str]]:
    missing: list[str] = []
    summaries: list[dict] = []

    for filename in REQUIRED_FILES + OPTIONAL_FILES:
        path = source_dir / filename
        if not path.exists():
            if filename in REQUIRED_FILES:
                missing.append(filename)
            continue
        summaries.append(read_parquet_info(path))

    return summaries, missing


def print_summary(summaries: list[dict], source_dir: Path) -> None:
    print("=" * 60)
    print(" ?몃? parquet ?ㅻ깄???먭?")
    print("=" * 60)
    print(f"source_dir : {source_dir}")

    price_max = None
    macro_max = None
    breadth_max = None

    for summary in summaries:
        print(f"\n[{summary['name']}]")
        print(f"rows      : {summary['rows']:,}")
        print(f"columns   : {', '.join(summary['columns'])}")
        if "tickers" in summary:
            print(f"tickers   : {summary['tickers']}")
        if "date_min" in summary:
            print(f"date_range: {summary['date_min']} ~ {summary['date_max']}")

        if summary["name"] == "price_data.parquet":
            price_max = summary.get("date_max")
        elif summary["name"] == "macroeconomic_indicators.parquet":
            macro_max = summary.get("date_max")
        elif summary["name"] == "market_breadth.parquet":
            breadth_max = summary.get("date_max")

    if price_max and macro_max and price_max < macro_max:
        gap_days = (macro_max - price_max).days
        print(f"\n[Warn] price_data媛 macro蹂대떎 {gap_days}???ㅻ옒?섏뿀?듬땲??")
    if price_max and breadth_max and price_max < breadth_max:
        gap_days = (breadth_max - price_max).days
        print(f"[Warn] price_data媛 market_breadth蹂대떎 {gap_days}???ㅻ옒?섏뿀?듬땲??")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)

    if not source_dir.exists():
        raise SystemExit(f"[Error] source_dir媛 議댁옱?섏? ?딆뒿?덈떎: {source_dir}")

    summaries, missing = validate_source_dir(source_dir)
    print_summary(summaries, source_dir)

    if missing:
        raise SystemExit(f"[Error] ?꾩닔 parquet ?뚯씪???놁뒿?덈떎: {', '.join(missing)}")

    if args.summary_only:
        return

    print("\n" + "=" * 60)
    print(" Lens raw ?곗씠???곸옱")
    print("=" * 60)
    run_bootstrap(source_dir, args.tickers)

    if args.skip_indicators:
        return

    print("\n" + "=" * 60)
    print(" Lens indicators ?앹꽦")
    print("=" * 60)
    step_indicators(source_dir, args.timeframes, args.tickers)


if __name__ == "__main__":
    main()

