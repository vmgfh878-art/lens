from __future__ import annotations

from pydantic import BaseModel


class LiveHealthData(BaseModel):
    status: str
    service: str


class ReadyHealthData(BaseModel):
    status: str
    service: str
    checks: dict[str, bool]
    source: str | None = None
    snapshot_id: str | None = None
    remote_sync_status: str | None = None
    last_remote_success_at: str | None = None
    r2_sync_status: str | None = None
    last_r2_success_at: str | None = None
