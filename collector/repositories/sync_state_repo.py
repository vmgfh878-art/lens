from __future__ import annotations

from datetime import date, datetime

from collector.repositories.base import fetch_all_rows, upsert_records


def get_job_state_map(job_name: str) -> dict[str, dict]:
    rows = fetch_all_rows("sync_state", columns="*", filters=[("eq", "job_name", job_name)], order_by="target_key")
    return {row["target_key"]: row for row in rows}


def upsert_job_state(
    job_name: str,
    target_key: str = "__all__",
    status: str = "success",
    last_cursor_date: date | None = None,
    message: str | None = None,
    meta: dict | None = None,
) -> None:
    payload = {
        "job_name": job_name,
        "target_key": target_key,
        "status": status,
        "last_cursor_date": last_cursor_date.isoformat() if last_cursor_date else None,
        "last_success_at": datetime.utcnow().isoformat() if status == "success" else None,
        "message": message,
        "meta": meta or {},
    }
    upsert_records("sync_state", [payload], on_conflict="job_name,target_key")
