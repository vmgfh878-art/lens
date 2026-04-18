from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """타임스탬프 포함 로그를 출력한다."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")
