# ADR-0028.5: CP223 Snapshot Drift-Resilient Redesign

Status: Accepted
Date: 2026-06-07
CP: CP237.5 (CP238~242 보안 트랙 prerequisite)

## Context

CP223 의 9 endpoint characterization snapshot 이 daily refresh 의 새 row 추가
+ scalar 값 변동 + cumulative list 증가로 매일 깨짐. 코드 회귀 검출 baseline
으로 기능 못 함. 보안 트랙(CP238~242) 진입 전 prerequisite.

2026-06-06 시점 8/9 RED (stocks_list 만 GREEN — fundamental 응답이라 daily 변동
없음). 깨진 8 개는 모두 코드 회귀 0, 데이터 drift 만.

## Decision

응답 값 자체는 비교에서 제외하고 **구조 (재귀 shape) 만** snapshot 박는 정규화
헬퍼 `backend/tests/_snapshot_normalize.py::normalize_response` 도입. drift
완전 면역.

### 정규화 규칙 (재귀, depth 무제한)

- 응답 전체: `{status_code, shape}` 로 박음
- `shape` 는 `_shape(value)` 로 재귀 정규화:
  - dict → `{dict_keys: sorted_keys, dict_schema: {k: _shape(v)}}`
  - list → `{list_row_schema: _shape(first_row_or_empty)}` (len 안 박음)
  - scalar → dtype 이름 (`int`, `float`, `str`, `bool`, `null`, ...)
- JSON 응답은 finite tree (cycle 없음) → 무한 재귀 위험 없음

### 검출 가능 회귀

- top-level / nested dict 의 key 추가·제거 / value dtype 변경
- list[dict] 의 첫 row 의 key 추가·제거 / value dtype 변경
- list 의 row dtype 변경 (scalar list)
- 빈 list → row 채워짐 (schema 진화)
- HTTP status code 변경

### 검출 불가 (trade-off — 별도 fixture 테스트로 보강 권고)

- row 의 정밀 float 값 회귀
- list 길이 변경 (cumulative 증가 면역의 대가)
- heterogeneous list (row 마다 dtype 다른 경우; 첫 row 기준)
- 첫 row 의 nullable 필드가 dtype 바뀜 (오늘 str / 내일 None)

## 기각된 대안

| 대안 | 이유 |
|---|---|
| 지시서 §2 last_n=5 + drift 필드 제거 + scalar value 박음 | rolling window list 가 매일 row swap 시키는 endpoint 8 개에서 last 5 가 매일 다름. 또 backtest_aapl 의 top-level scalar (averageHoldingDays 등) 가 매일 변동 → 매일 RED |
| 1-level shallow dict + last_n=5 | 응답이 `{data: dict{data: list}}` 중첩이라 inner list 의 row schema 가 비교 대상에서 빠짐 → 보안 트랙 회귀 안전망 절반 무력 |
| first_n=5 (가장 오래된 5 행) | rolling 60 day endpoint (product_history) / limit cap endpoint (indicators) 는 first 도 swap 될 수 있고 EODHD historical close retroactive 보정 시 prices/band 도 fail |
| endpoint 별 다른 전략 (start/end 고정 + cap 기반 분기) | 복잡, 일관성 없음, 유지보수 부담 |

## 구현

- 신규: `backend/tests/_snapshot_normalize.py` (138 줄, `_shape` + `normalize_response`)
- 수정: `backend/tests/test_characterization_api.py::test_endpoint_snapshot`
  → `normalize_response(resp.status_code, payload)` 거쳐 snapshot 비교
- 갱신: `backend/tests/__snapshots__/test_characterization_api.ambr`
  (45056 줄 raw JSON → 337 줄 shape only → 재귀 변환 후 ~700 줄 shape)
- 신규: `backend/tests/test_snapshot_normalize.py` (drift simulation 11 케이스,
  파일명은 .gitignore 의 `test_cp*.py` 패턴 우회 + CP223 영구 안전망 정책 일관)

## Consequences

장점:
- daily refresh 영향 0 (drift 완전 면역, 매일 RED 사라짐)
- 보안 트랙(CP238~242) 진입 baseline 회복 — 9 endpoint 모두 GREEN
- 재귀 정규화로 nested schema 변동도 검출 (이전 1-level shallow 대비 강화)
- drift simulation 11 케이스로 정규화 자체 검증

단점 / Trade-off:
- 정밀한 row-level value 회귀 검출력 약화 — 별도 fixture 기반 테스트로 보강
  하는 게 v2 후속 (CP237.5 범위 밖)
- 첫 row 기준의 row_schema 라 heterogeneous list / nullable 첫 row 케이스
  검출 못 함 (현 응답에 영향 미미하나 명시)

## v2 후속

- daily refresh cron 직후 자동 snapshot diff 알람 (drift 외 변화 감시)
- row-level value 회귀 위한 fixture parquet + 응답 비교 테스트 별도 도입
- list_row_schema 를 첫 row 가 아니라 모든 row union 으로 박는 옵션 검토

## References

- CP223 ADR-0011 (snapshot 도입)
- CP237 `.github/workflows/ci.yml` (snapshot job)
- 본 CP 보고서: `docs/cp237_5_snapshot_redesign_report.md`
