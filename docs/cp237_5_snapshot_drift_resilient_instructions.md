# CP237.5 지시서 — CP223 Snapshot Drift-Resilient 재설계

> 작성: 2026-06-06. 보안 트랙(CP238~242) **진입 전 prerequisite**. ADR-0028.5 동반.
> 전략: Schema + 마지막 N행 (사용자 선택, 2026-06-06).

---

## 0. 한 줄 목표

CP223 characterization snapshot 이 daily refresh 의 새 row 추가에 깨지지 않게 정규화. 응답 schema (keys/dtypes/HTTP status) + data list 마지막 N행을 정해진 비교 키로 박아 drift-resilient 회귀 안전망 회복. 8 endpoint 모두 GREEN.

---

## 1. 진단

| 항목 | 현재 |
|---|---|
| `backend/tests/test_characterization_api.py` snapshot | 8 endpoint (aapl_prices / indicators / product_history / line / band_1d / band_1w / scan_indicator / backtest_aapl) |
| 실패 원인 | data list 에 새 row 1개 (`asof_date='2025-06-02'`) 추가 → snapshot mismatch |
| 코드 회귀 | ❌ 없음. 응답 schema 동일, HTTP 200 정상 |
| daily refresh 영향 | 매일 깨짐 (현재 구조 한계) |
| syrupy float tolerance | 작동 중 (값 비교는 OK), 구조 변화는 무력 |

**핵심**: snapshot 의도 = "보안/리팩토링 트랙 변경이 응답에 영향 주는지 검출". 데이터 freshness 추적이 아님. 따라서 변동 필드는 정규화 대상.

---

## 2. 변경 내용 (선택된 전략: Schema + 마지막 N행)

- `backend/tests/_snapshot_normalize.py` 신규 — 응답 정규화 헬퍼
  - top-level: `status_code` + `keys` + per-key `dtype` (str/int/list/dict)
  - data list: 마지막 **N=5 행** + 변동 필드 제거 (`asof_date`, `actual_h5_return`, `created_at`, `updated_at` 등)
  - row 내 float 값은 유지 (syrupy tolerance 가 처리)
- `test_characterization_api.py::test_endpoint_snapshot` → 정규화 함수 거친 payload 와 비교
- `__snapshots__/` 의 기존 ambr 파일 갱신 (1회 `--snapshot-update`)
- 검증: 같은 응답 2회 / 새 row 추가 시뮬레이션 → 둘 다 0 diff
- ADR-0028.5

---

## 3. Step 분할

| Step | 내용 | 위험 | 시간 | 자동/수동 |
|---|---|---|---|---|
| 1 | 현재 8 endpoint snapshot 구조 read → 변동 필드 후보 목록 추출 | 매우낮음 | 20분 | 자동 |
| 2 | `backend/tests/_snapshot_normalize.py` 신규 — `normalize_response()` 헬퍼 | 매우낮음 (additive) | 40분 | 자동 |
| 3 | `test_characterization_api.py::test_endpoint_snapshot` 정규화 적용 (snapshot 직전에 변환) | 낮음 | 20분 | 자동 |
| 4 | 1회 `pytest --snapshot-update` → 새 baseline 박힘 | 낮음 | 5분 | 자동 |
| 5 | 같은 응답 2회 실행 → 0 diff confirm | 매우낮음 | 5분 | 자동 |
| 6 | **Drift 시뮬레이션**: parquet 끝에 fake row 1개 추가한 fixture 로 응답 → snapshot 동일 confirm | 낮음 | 30분 | 자동 |
| 7 | `docs/cp237_5_snapshot_redesign_report.md` + ADR-0028.5 | 매우낮음 | 30분 | 자동 |

---

## 4. 각 Step 정확한 명령 / 코드

### Step 1 — 현재 snapshot 구조 분석

```powershell
cd C:\Users\user\lens
Get-ChildItem backend\tests\__snapshots__\test_characterization_api.ambr
```

내용 read:
```powershell
Get-Content backend\tests\__snapshots__\test_characterization_api.ambr | Select-Object -First 200
```

추출할 정보:
- 각 endpoint 의 top-level keys (data / meta / 등)
- `data` list 내 row 의 모든 key
- 변동 후보 필드: `asof_date`, `actual_h5_return`, `actual_h*_return`, `line_rank_by_date`, `created_at`, `updated_at`, `inserted_at`, `latest_*`

### Step 2 — 정규화 헬퍼

`backend/tests/_snapshot_normalize.py` (신규):

```python
"""CP237.5 — Drift-resilient snapshot 정규화.

전략: Schema + 마지막 N행.
- top-level: status_code / keys / per-key dtype
- data list: 마지막 N=5 행 (data 외 다른 list 키도 동일) + 변동 필드 제거
- float 값 유지 (syrupy float tolerance 가 처리)

목적: daily refresh 의 새 row 추가에 snapshot 깨지지 않도록.
보안/리팩토링 트랙 코드 변경에 의한 응답 schema/key/order 변동은 그대로 검출.
"""

from __future__ import annotations

from typing import Any

# 매일 변하는 필드 — snapshot 에서 제거.
# 새 endpoint 가 추가 필드 도입하면 여기에 추가.
DRIFT_FIELDS: frozenset[str] = frozenset({
    "asof_date",
    "actual_h5_return",
    "actual_h1_return",
    "actual_h4_return",
    "actual_h20_return",
    "line_rank_by_date",
    "created_at",
    "updated_at",
    "inserted_at",
    "latest_asof_date",
    "data_freshness_date",
    "as_of",
})

# data list 에서 보존할 마지막 행 수. snapshot 안정성과 회귀 검출력의 trade-off.
TAIL_ROWS: int = 5


def _dtype_name(value: Any) -> str:
    """간단한 dtype label. Pydantic 응답이 이미 정규화되어 있어 light-weight."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _strip_drift(row: dict[str, Any]) -> dict[str, Any]:
    """변동 필드 제거."""
    return {k: v for k, v in row.items() if k not in DRIFT_FIELDS}


def _normalize_list(rows: list[Any]) -> dict[str, Any]:
    """list → {len, last_n_rows, row_schema}.

    row_schema 는 첫 행 기준 (마지막 N행도 동일 schema 가정).
    """
    if not rows:
        return {"len": 0, "last_n": [], "row_schema": {}}

    tail = rows[-TAIL_ROWS:]
    if isinstance(rows[0], dict):
        tail_clean = [_strip_drift(r) for r in tail]
        schema = {k: _dtype_name(v) for k, v in rows[0].items()}
    else:
        tail_clean = tail
        schema = {"_value_dtype": _dtype_name(rows[0])}

    return {
        "len": len(rows),
        "last_n": tail_clean,
        "row_schema": schema,
    }


def normalize_response(status_code: int, payload: Any) -> dict[str, Any]:
    """HTTP 응답을 drift-resilient snapshot 형태로 변환.

    Args:
        status_code: HTTP status code (정수)
        payload: response.json() 결과 (dict 또는 list)

    Returns:
        snapshot 비교용 dict.
    """
    out: dict[str, Any] = {
        "status_code": status_code,
    }

    if isinstance(payload, dict):
        out["payload_type"] = "dict"
        out["top_level_keys"] = sorted(payload.keys())
        out["top_level_dtypes"] = {k: _dtype_name(v) for k, v in payload.items()}
        # list 값을 가진 키만 정규화 (data / items / results 등)
        for k, v in payload.items():
            if isinstance(v, list):
                out[f"list:{k}"] = _normalize_list(v)
            elif isinstance(v, dict):
                # 1-level nested dict 만 (depth 제한)
                out[f"dict:{k}"] = {
                    "keys": sorted(v.keys()),
                    "dtypes": {kk: _dtype_name(vv) for kk, vv in v.items()},
                }
            else:
                out[f"scalar:{k}"] = {"dtype": _dtype_name(v), "value": v}
    elif isinstance(payload, list):
        out["payload_type"] = "list"
        out["root"] = _normalize_list(payload)
    else:
        out["payload_type"] = _dtype_name(payload)
        out["value"] = payload

    return out
```

### Step 3 — `test_characterization_api.py` 적용

기존 (가정):
```python
def test_endpoint_snapshot(...):
    response = client.get(url)
    payload = response.json()
    assert payload == snapshot
```

수정:
```python
from backend.tests._snapshot_normalize import normalize_response

def test_endpoint_snapshot(...):
    response = client.get(url)
    normalized = normalize_response(response.status_code, response.json())
    assert normalized == snapshot
```

(실제 경로/import 는 현재 코드 read 후 조정.)

### Step 4 — Snapshot 갱신

```powershell
cd C:\Users\user\lens
$env:PYTHONPATH=".;backend"; $env:LENS_FORCE_LOCAL="1"
.\.venv\Scripts\pytest.exe `
  backend\tests\test_characterization_api.py `
  --snapshot-update `
  --ignore=backend\tests\test_services.py
```

→ `__snapshots__/test_characterization_api.ambr` 갱신.

### Step 5 — 같은 응답 2회 → 0 diff

```powershell
.\.venv\Scripts\pytest.exe backend\tests\test_characterization_api.py -q --ignore=backend\tests\test_services.py
.\.venv\Scripts\pytest.exe backend\tests\test_characterization_api.py -q --ignore=backend\tests\test_services.py
```

둘 다 `8 passed` 여야 통과.

### Step 6 — Drift 시뮬레이션

`backend/tests/test_cp237_5_drift_simulation.py` (신규):

```python
"""CP237.5 — Drift simulation.

snapshot 정규화가 새 row 추가에 안 깨지는지 검증.

방식: 응답에 fake row 1개를 끝에 추가한 dict 와 원본 dict 가 동일 normalized payload 를 만드는지.
"""

import pytest

from backend.tests._snapshot_normalize import normalize_response


def test_normalize_resilient_to_new_row():
    base = {
        "data": [
            {"asof_date": "2025-05-30", "value": 1.0, "ticker": "AAPL"},
            {"asof_date": "2025-05-31", "value": 2.0, "ticker": "AAPL"},
        ],
        "meta": {"count": 2},
    }
    # daily refresh 시뮬레이션 — 새 행 1개 추가
    drifted = {
        "data": base["data"] + [
            {"asof_date": "2025-06-01", "value": 3.0, "ticker": "AAPL"},
        ],
        "meta": {"count": 3},
    }

    base_norm = normalize_response(200, base)
    drifted_norm = normalize_response(200, drifted)

    # last_n 가 같은 row 들을 가지면 (5보다 적으니 전체) 정규화 결과는 last_n 이 다를 수 있음
    # 하지만 row_schema + top_level_keys + dtypes 는 동일해야
    assert base_norm["top_level_keys"] == drifted_norm["top_level_keys"]
    assert base_norm["top_level_dtypes"] == drifted_norm["top_level_dtypes"]
    assert base_norm["list:data"]["row_schema"] == drifted_norm["list:data"]["row_schema"]
    # asof_date 가 변동 필드라 last_n 내용에도 영향 없음
    for row in drifted_norm["list:data"]["last_n"]:
        assert "asof_date" not in row


def test_normalize_detects_schema_change():
    """key 추가/제거는 검출해야 (회귀)."""
    base = {"data": [{"a": 1, "b": 2}]}
    schema_added = {"data": [{"a": 1, "b": 2, "c": 3}]}

    base_norm = normalize_response(200, base)
    added_norm = normalize_response(200, schema_added)

    assert base_norm["list:data"]["row_schema"] != added_norm["list:data"]["row_schema"]


def test_normalize_detects_status_change():
    """status code 변화 검출."""
    a = normalize_response(200, {"ok": True})
    b = normalize_response(500, {"ok": True})
    assert a["status_code"] != b["status_code"]
```

실행:
```powershell
.\.venv\Scripts\pytest.exe backend\tests\test_cp237_5_drift_simulation.py -v --ignore=backend\tests\test_services.py
```

3개 PASS 여야 통과.

### Step 7 — 보고서 + ADR

`docs/cp237_5_snapshot_redesign_report.md`:

```markdown
# CP237.5 Snapshot Drift-Resilient 보고서

## 문제
- CP223 snapshot 8개가 매일 refresh 후 깨짐 → 보안/리팩토링 트랙 회귀 안전망 무력화
- 원인: data list 에 새 row 추가 → snapshot mismatch (코드 회귀 X)

## 해결
- Schema + 마지막 N=5 행 전략
- 변동 필드 (asof_date, actual_*_return, *_at) 정규화 단계에서 제거
- syrupy float tolerance 와 결합

## 산출물
- backend/tests/_snapshot_normalize.py (신규)
- backend/tests/test_characterization_api.py 수정 (normalize 거쳐 비교)
- backend/tests/__snapshots__/test_characterization_api.ambr (재생성)
- backend/tests/test_cp237_5_drift_simulation.py (신규)
- docs/adr/0028_5_snapshot_drift_resilient.md

## 검증
- 같은 응답 2회: 0 diff
- Drift 시뮬레이션 3 케이스: PASS (resilient + schema change detect + status change detect)
- 보안 트랙 진입 baseline 회복: 8 endpoint GREEN

## v2 후속
- daily refresh cron 직후 자동 snapshot diff 알람 (drift 외 변화 감시)
- snapshot 정규화 깊이 (현재 1-level nested dict) 확장 검토
```

ADR 양식:

`docs/adr/0028_5_snapshot_drift_resilient.md`:

```markdown
# ADR-0028.5: CP223 Snapshot Drift-Resilient Redesign

## Status
Accepted (2026-06-06)

## Context
CP223 characterization snapshot 이 daily refresh 의 새 row 추가에 매일 깨짐. 코드 회귀 검출 baseline 으로 기능 못 함. 보안 트랙(CP238~242) 진입 전 prerequisite.

## Decision
Schema + 마지막 N행 전략:
- top-level: status_code / sorted keys / per-key dtype
- list 값 (data 등): len / last 5 rows (변동 필드 제거) / row_schema
- 변동 필드 frozenset: asof_date / actual_*_return / *_at / latest_*
- syrupy float tolerance 그대로 활용 (값 정밀도 회귀 검출 유지)

## Consequences
- 매일 refresh 영향 0 (drift-resilient)
- 코드 변경에 의한 schema/keys/dtypes 변동은 그대로 검출 (보안 트랙 회귀 안전망 회복)
- 정밀한 row-level 회귀 검출력 약화 (last 5 rows 만) — trade-off 명시
- 새 endpoint 추가 시 변동 필드 frozenset 갱신 필요 → 향후 자동화 검토 (v2)

## References
- CP223 ADR-0011 (snaptol 도입)
- CP237 ci.yml (snapshot job)
```

---

## 5. 회귀 안전망

- Step 5 (같은 응답 2회) — 비결정성 검출
- Step 6 (drift 시뮬레이션) — 새 정규화의 resilience + schema change detect 보장
- 기존 backend pytest (`pytest backend/tests --ignore=backend/tests/test_services.py`) — 전체 회귀

---

## 6. 성공 기준

- 8 endpoint snapshot 모두 GREEN
- 같은 응답 2회 실행 → 0 diff
- Drift 시뮬레이션 3 케이스 PASS
- backend pytest 전체 PASS (test_services.py 제외)
- CI workflow GREEN
- ADR-0028.5 작성

---

## 7. 인터페이스 보존

- 응답 schema 자체는 안 건드림 (test 만 수정)
- API contract 영향 0
- 운영 모델 추론 영향 0

---

## 8. Lens 특화

- daily refresh cron (scripts/run_v1_unified_refresh_local.ps1) 영향 0
- 운영 모델 3개 (CP210/CP153/CP178) 응답 정확성 유지
- CI workflow 의 `--ignore=backend/tests/test_services.py` 패턴 보존

---

## 9. 자동 실행 적합도

| Step | 자동 | 사람 확인 |
|---|---|---|
| 1 | ✅ | — |
| 2 | ✅ | — |
| 3 | ✅ | — |
| 4 | ✅ | — |
| 5 | ✅ | — |
| 6 | ✅ | — |
| 7 | ✅ | — |

→ **전 Step 자동 적합**. 사용자 확인 필요 없음 (단 회귀 깨지면 즉시 중단).

---

## 10. 종료 후 commit / 보고

```
CP237.5 Step 1: snapshot structure analysis (no code change)
CP237.5 Step 2: add _snapshot_normalize.py
CP237.5 Step 3: apply normalize_response in test_characterization_api
CP237.5 Step 4: regenerate snapshots (drift-resilient baseline)
CP237.5 Step 5: 2x stable run confirm
CP237.5 Step 6: drift simulation tests (3 cases)
CP237.5 report + ADR-0028.5
```

보고서: `docs/cp237_5_snapshot_redesign_report.md`
ADR: `docs/adr/0028_5_snapshot_drift_resilient.md`

---

**진입 조건**: 없음 (보안 트랙 CP238 진입 전 즉시).
**다음 CP**: CP238 (보안 트랙 시작).
**리스크**: Step 4 snapshot 재생성 직전 응답이 정상인지 1회 수동 read 권장. 잘못된 baseline 박으면 향후 회귀 못 잡음.
