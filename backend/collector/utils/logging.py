from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def log(message: str, *, level: str = "INFO", **fields: Any) -> None:
    """구조화된 JSON 로그 한 줄을 출력한다."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
    }
    payload.update({key: value for key, value in fields.items() if value is not None})
    print(json.dumps(payload, ensure_ascii=False, default=str))
