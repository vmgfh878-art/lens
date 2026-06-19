"""CP254 — PostgREST range 페이지네이션 헬퍼.

Supabase PostgREST 는 기본 max_rows=1000 으로 응답을 자른다. 전종목(cross-sectional)
프레임처럼 1000행 초과 SELECT 는 .range() 로 페이지를 이어 붙여야 한다.

전제(중요): page_size 는 서버 max_rows 와 같거나 작아야 한다. 서버 cap 이
page_size 보다 작으면 range 보폭 사이에 구멍이 생겨도 감지할 수 없다.
Supabase 기본 1000 / 로컬 리허설 PostgREST 도 db-max-rows=1000 으로 맞춘다
(P6 컷오버 체크리스트 항목).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PAGE_SIZE = 1000
# 폭주 가드. 서빙 최대 테이블(product_history 782k) << 2000 페이지 × 1000행.
MAX_PAGES = 2000


def paged_select(
    build_query: Callable[[], Any],
    *,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """build_query 가 만든 (select/filter/order 적용된) 쿼리를 range 로 전 페이지 수집.

    build_query 는 호출마다 **새 쿼리 빌더**를 만들어야 한다 — postgrest 빌더는
    mutable 이라 재사용하면 range 가 누적된다. order 는 결정적(예: ticker, date)
    이어야 페이지 간 중복/누락이 없다.
    """
    rows: list[dict] = []
    for page in range(max_pages):
        start = page * page_size
        batch = build_query().range(start, start + page_size - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
    raise RuntimeError(f"paged_select 페이지 한도 초과: {max_pages} pages × {page_size} rows")
