from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path("ai/cache")
DEFAULT_PATHS = {
    "1D": CACHE_DIR / "ticker_id_map_1d.json",
    "1W": CACHE_DIR / "ticker_id_map_1w.json",
}


def registry_path_for_timeframe(timeframe: str) -> Path:
    normalized = timeframe.upper()
    if normalized not in DEFAULT_PATHS:
        raise ValueError(f"지원하지 않는 ticker registry timeframe 입니다: {timeframe}")
    return DEFAULT_PATHS[normalized]


def build_registry(eligible_tickers: list[str], timeframe: str) -> dict[str, Any]:
    normalized = timeframe.upper()
    mapping = {ticker: idx for idx, ticker in enumerate(sorted({ticker.upper() for ticker in eligible_tickers}))}
    return {
        "timeframe": normalized,
        "mapping": mapping,
        "num_tickers": len(mapping),
    }


def save_registry(registry: dict[str, Any], timeframe: str, path: Path | None = None) -> Path:
    target = path or registry_path_for_timeframe(timeframe)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return target


def load_registry(timeframe: str, path: Path | None = None) -> dict[str, Any]:
    target = path or registry_path_for_timeframe(timeframe)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lookup_id(ticker: str, registry: dict[str, Any]) -> int:
    normalized = str(ticker).upper()
    return int(registry["mapping"].get(normalized, registry["num_tickers"]))


def build_and_save_registry(timeframe: str) -> Path:
    from ai.preprocessing import build_dataset_plan, default_horizon, fetch_feature_index_frame

    normalized = timeframe.upper()
    seq_len = 252 if normalized == "1D" else 104
    horizon = default_horizon(normalized)
    index_frame = fetch_feature_index_frame(timeframe=normalized)
    plan = build_dataset_plan(index_frame, timeframe=normalized, seq_len=seq_len, horizon=horizon)
    registry = build_registry(plan.eligible_tickers, normalized)
    return save_registry(registry, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 학습용 ticker id 매핑을 생성합니다.")
    parser.add_argument("--build", action="store_true", help="1D와 1W ticker registry를 생성합니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.build:
        raise SystemExit("--build 옵션이 필요합니다.")
    for timeframe in ("1D", "1W"):
        path = build_and_save_registry(timeframe)
        registry = load_registry(timeframe, path)
        print(json.dumps({"timeframe": timeframe, "path": str(path), "num_tickers": registry["num_tickers"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
