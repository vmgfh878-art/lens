# Lens DB Repository 가이드 — N+1 회피 / from_attributes 함정 (CP236c)

**상태**: 가이드 (준비). 실 ORM·실 N+1 측정은 Supabase Pro 결제 + CP236d (Alembic/async + ORM 골격) 이후. 현재는 supabase-py REST만 쓰며 ORM 없음.

이 가이드는 (a) 지금 REST 쿼리를 작성할 때, (b) 결제 후 ORM repository를 도입할 때 **둘 다** 적용되는 데이터 로딩 규약이다.

---

## §1. N+1이란

부모 N행을 받고 각 행마다 자식을 1쿼리씩 더 날리면 총 1+N쿼리가 발생한다. 부모 100행이면 101쿼리, 1000행이면 1001쿼리. Supabase free tier에서는 quota를 단숨에 소진하고, 직접연결로도 latency 폭발한다.

```python
# BAD (개념 예시 — 실제 도입 금지): 행마다 추가 쿼리
for row in indicator_rows:
    vol = client.table("price_data").select("volume").eq("date", row["date"]).execute().data
```

---

## §2. 회피 — REST (현재형)

**배치 IN**: 자식을 한 번에 가져와 dict로 병합. 모범 사례는 이미 코드에 있다 — `backend/app/repositories/market_repo.py` `_merge_indicator_volume()`:

- `in_("date", dates)` 로 1쿼리 (행별 쿼리 X)
- 결과를 `volume_by_date` dict로 묶고 indicator rows에 병합

**embedded resource**: PostgREST FK 관계가 정의돼 있으면 `select("parent_cols, child_table(child_cols)")` 로 한 번에. 우리 스키마 확정 후 적용.

---

## §3. 회피 — ORM (미래형, 결제 후 CP236d 이후)

```python
from sqlalchemy.orm import joinedload, selectinload

# selectinload: 관계당 별도 SELECT ... WHERE fk IN (...). 자식 다(多)·중복 적을 때 유리.
stmt = select(Parent).options(selectinload(Parent.children))

# joinedload: LEFT OUTER JOIN 한 방. 자식 1:1·소수일 때 유리. 1:N에 쓰면 부모 행 중복.
stmt = select(Parent).options(joinedload(Parent.child))
```

**선택 기준**:

| 관계 | 권장 옵션 | 이유 |
|---|---|---|
| 1:N many | `selectinload` | JOIN 행 중복 피하고 fk IN 1쿼리 |
| to-one / 소수 | `joinedload` | LEFT JOIN 한 방, round-trip 1번 |
| lazy (기본) | **금지** | async 세션에서 동기 I/O 폭발 |
| `subquery` | 피한다 | 대부분 selectinload보다 비효율 |

---

## §4. `from_attributes` (Pydantic) 함정

Pydantic `ConfigDict(from_attributes=True)` (구 `orm_mode`)는 ORM 객체를 직접 직렬화하게 해준다. 그러나 **관계 속성에 접근하는 순간 unloaded면 SQLAlchemy lazy load**. async 세션에서는 이게 금지된 동기 I/O라 `MissingGreenlet` 또는 `DetachedInstance`로 터진다.

**규칙**: `from_attributes=True` 모델을 쓰려면 쿼리에서 응답 스키마에 들어가는 **모든 관계를 eager load 한다**. 안 그러면 직렬화 단계에서 폭발.

현재 `backend/app/schemas/common.py`의 `MetaResponse` / `ApiResponse` / `ErrorResponse`는 `extra="allow"` + plain dict input 기반이라 함정 영역 밖. ORM 객체를 받는 모델을 새로 만들면 그 모델에 본 가이드 적용.

---

## §5. 골든 룰

> **응답 스키마에 들어가는 관계는 쿼리 시점에 전부 eager/embedded로 로딩한다. 직렬화 단계에서 추가 I/O가 일어나면 안 된다.**

위 §2 (REST) / §3 (ORM)으로 백링크.

## 관련 ADR / CP

- ADR-0013: 직접연결 5432 + PgBouncer 6543 함정 (CP236a).
- ADR-0026: SQLAlchemy 2.0 async 세션 골격 (CP236b).
- CP236d: Alembic + async 마이그레이션 골격 (예정).
