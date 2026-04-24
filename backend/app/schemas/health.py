from __future__ import annotations

from pydantic import BaseModel


class LiveHealthData(BaseModel):
    status: str
    service: str


class ReadyHealthData(BaseModel):
    status: str
    service: str
    checks: dict[str, bool]
