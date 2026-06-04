# CP236c 보고서 — N+1 회피 / from_attributes 함정 가이드

**완료일**: 2026-06-04
**선행 의존**: CP236a (ADR-0013 존재), CP236b (session.py 골격). 둘 다 그린.
**커밋**: 본 commit (가이드/주석/문서만, additive)

## 요구

ORM 도입 전에 데이터 로딩 규약 (N+1 회피, from_attributes lazy load 함정)을 가이드 + 주석으로 박제. 실 ORM / 실 코드 변경 0.

## 한 일

| 파일 | 변경 |
|---|---|
| `docs/db_repository_guide.md` (신규) | §1 N+1이란 / §2 REST 배치 IN 모범 (`_merge_indicator_volume` 인용) / §3 ORM `selectinload` vs `joinedload` 선택 기준 / §4 `from_attributes` 함정 / §5 골든 룰 |
| `backend/app/repositories/__init__.py` | 빈 파일 → 모듈 docstring 추가 ("ORM repository 추가 시 eager load 필수, N+1 금지, 가이드 링크") |
| `backend/app/schemas/common.py` | `MetaResponse` 위에 주석 블록 추가 ("from_attributes 함정 + 켤 때 규약"). `ConfigDict(extra="allow")` 그대로 — `from_attributes`는 켜지 않음. |
| `docs/adr/0013-supabase-port-5432-not-6543.md` | "## N+1 & from_attributes (CP236c)" 섹션 append |

## 보존 체크리스트

| 항목 | 확인 |
|---|---|
| `market_repo.py` / `db.py` / `base.py` 함수 signature 0 수정 | OK (인용만) |
| `common.py` `ConfigDict(extra="allow")` 그대로 | OK |
| `from_attributes=True` 실제로 켜지 않음 | OK (주석만) |
| `repositories/__init__.py` 공개 심볼 0 추가 (docstring만) | OK |
| 직렬화 입력 형식 (plain dict) 동일 → 응답 바이트 동일 | OK |

## 자가 점검

- **[Plan v3 정합]** PASS — 사유: 가이드/주석만. 밴드/fidelity/cost 무관.
- **[구조 결함]** PASS — 사유: 코드 동작 0 변경. import smoke OK. 미래 ORM repository 규약 박제.
- **[모델 영향]** PASS (N/A) — 사유: 학습/calibration/feature 무관.

## 후속

- CP236d: Alembic + async 마이그레이션 골격.
- 결제 후: 실 ORM repository 작성 시 본 가이드 적용.
