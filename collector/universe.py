from __future__ import annotations

import csv
from pathlib import Path


DEFAULT_UNIVERSE_FILE = Path("data/universe/sp500.csv")


def get_default_universe_file() -> Path:
    """기본 유니버스 파일 경로를 반환한다."""
    return DEFAULT_UNIVERSE_FILE


def load_tickers_from_csv(file_path: str | Path | None) -> list[str]:
    """고정 CSV 파일에서 티커 목록을 읽는다."""
    if not file_path:
        return []

    path = Path(file_path)
    if not path.exists():
        return []

    tickers: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = (row.get("ticker") or row.get("symbol") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
    return tickers
