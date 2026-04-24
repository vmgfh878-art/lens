from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from backend.collector.repositories.base import upsert_records


def start_job_run(job_name: str, *, scope_key: str = "__all__", meta: dict | None = None) -> str:
    """실행 단위 레코드를 만들고 run_id를 반환한다."""
    run_id = str(uuid4())
    upsert_records(
        "job_runs",
        [
            {
                "run_id": run_id,
                "job_name": job_name,
                "scope_key": scope_key,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "meta": meta or {},
            }
        ],
        on_conflict="run_id",
    )
    return run_id


def finish_job_run(
    run_id: str,
    *,
    job_name: str,
    status: str,
    error_text: str | None = None,
    meta: dict | None = None,
) -> None:
    """실행 단위 레코드의 종료 상태를 갱신한다."""
    upsert_records(
        "job_runs",
        [
            {
                "run_id": run_id,
                "job_name": job_name,
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_text": error_text,
                "meta": meta or {},
            }
        ],
        on_conflict="run_id",
    )
