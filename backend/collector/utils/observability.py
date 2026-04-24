from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from backend.collector.repositories.job_run_repo import finish_job_run, start_job_run
from backend.collector.utils.logging import log


@contextmanager
def tracked_job(job_name: str, *, scope_key: str = "__all__", meta: dict | None = None) -> Iterator[dict[str, Any]]:
    """실행 단위 이력과 구조화 로그를 함께 남기는 컨텍스트다."""
    base_meta = dict(meta or {})
    run_id = start_job_run(job_name, scope_key=scope_key, meta=base_meta)
    started_at = perf_counter()
    context: dict[str, Any] = {
        "run_id": run_id,
        "job_name": job_name,
        "scope_key": scope_key,
        "meta": {},
    }
    log(
        f"{job_name} 시작",
        event="job_started",
        job=job_name,
        scope_key=scope_key,
        run_id=run_id,
        **base_meta,
    )
    try:
        yield context
    except BaseException as exc:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        final_meta = {**base_meta, **context["meta"], "duration_ms": duration_ms}
        finish_job_run(run_id, job_name=job_name, status="failed", error_text=str(exc), meta=final_meta)
        log(
            f"{job_name} 실패",
            level="ERROR",
            event="job_failed",
            job=job_name,
            scope_key=scope_key,
            run_id=run_id,
            error=str(exc),
            duration_ms=duration_ms,
            **context["meta"],
        )
        raise
    else:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        final_meta = {**base_meta, **context["meta"], "duration_ms": duration_ms}
        finish_job_run(run_id, job_name=job_name, status="success", meta=final_meta)
        log(
            f"{job_name} 완료",
            event="job_finished",
            job=job_name,
            scope_key=scope_key,
            run_id=run_id,
            duration_ms=duration_ms,
            **context["meta"],
        )
