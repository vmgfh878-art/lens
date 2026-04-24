"""스냅샷 parquet 묶음을 Lens 스키마로 적재한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.services.feature_svc import FEATURE_COLUMNS, build_features, normalize_timeframe
from backend.db.bootstrap import (  # noqa: E402
    DEFAULT_DATA_DIR,
    chunked_upsert,
    get_client,
    read_parquet,
    run_bootstrap,
)


def build_indicator_records(
    source_dir: Path, timeframes: list[str], sample_tickers: list[str] | None = None
) -> list[dict]:
    price_df = read_parquet("price_data.parquet", source_dir)
    macro_df = read_parquet("macroeconomic_indicators.parquet", source_dir)
    breadth_df = read_parquet("market_breadth.parquet", source_dir)

    if sample_tickers:
        price_df = price_df[price_df["ticker"].isin(sample_tickers)].copy()

    all_records: list[dict] = []
    for raw_timeframe in timeframes:
        timeframe = normalize_timeframe(raw_timeframe)
        features = build_features(price_df, macro_df, breadth_df, timeframe=timeframe)
        if sample_tickers:
            features = features[features["ticker"].isin(sample_tickers)].copy()
        if features.empty:
            print(f"[Warn] {timeframe} indicators 후보가 없습니다.")
            continue

        subset = features[["ticker", "date", "timeframe", *FEATURE_COLUMNS]].copy()
        subset["date"] = pd.to_datetime(subset["date"]).dt.strftime("%Y-%m-%d")
        all_records.extend(subset.where(pd.notnull(subset), None).to_dict(orient="records"))
        print(f"  - {timeframe}: {len(subset):,} rows 준비")

    return all_records


def step_indicators(source_dir: Path, timeframes: list[str], sample_tickers: list[str] | None = None) -> None:
    print("\n[7/7] indicators 생성 및 적재 중")
    records = build_indicator_records(source_dir, timeframes, sample_tickers)
    if not records:
        print("[Warn] 적재할 indicators 레코드가 없습니다.")
        return

    supabase = get_client()
    chunked_upsert(supabase, "indicators", records, on_conflict="ticker,date,timeframe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens 스냅샷 부트스트랩")
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_DATA_DIR),
        help="원본 parquet 폴더 경로",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="샘플 적재용 종목 목록. 지정하지 않으면 전체를 사용합니다.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=["1D", "1W", "1M"],
        help="indicators 생성 타임프레임 목록",
    )
    parser.add_argument("--skip-bootstrap", action="store_true", help="raw 적재 단계를 건너뜁니다.")
    parser.add_argument("--skip-indicators", action="store_true", help="indicators 생성을 건너뜁니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)

    print("=" * 60)
    print(" Lens 스냅샷 부트스트랩 시작")
    print("=" * 60)
    print(f"source_dir   : {source_dir}")
    print(f"sampleTicker : {args.tickers or 'ALL'}")
    print(f"timeframes   : {args.timeframes}")

    if not args.skip_bootstrap:
        run_bootstrap(source_dir, args.tickers)
    if not args.skip_indicators:
        step_indicators(source_dir, args.timeframes, args.tickers)

    print("\n" + "=" * 60)
    print(" Lens 스냅샷 부트스트랩 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
