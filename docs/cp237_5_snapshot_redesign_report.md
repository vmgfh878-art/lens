# CP237.5 — Snapshot Drift-Resilient 재설계 보고서

작성: 2026-06-07. 보안 트랙(CP238~242) prerequisite. ADR-0028.5 동반.

## 0. 한 줄 요약

CP223 의 9 endpoint snapshot 이 daily refresh 의 데이터 drift 로 매일 깨지는
문제를 응답 값 → 응답 shape 비교로 정규화해서 해결. 9 endpoint GREEN +
drift simulation 11/11 PASS. 보안 트랙 진입 baseline 회복.

## 1. 핵심 컴포넌트 존재 체크리스트

CP237.5 는 테스트 정규화만 손대며 운영 코드 / ML 모델 컴포넌트 무변경.
ML 컴포넌트 체크리스트(RevIN denorm, CI aggregate, ticker emb 등) 는 N/A.
대신 본 CP 핵심 컴포넌트 체크:

- [x] `backend/tests/_snapshot_normalize.py` 존재 (`normalize_response`, `_shape`, `DRIFT_FIELDS`)
- [x] `_shape(value)` 가 dict / list / scalar 재귀 분기
- [x] dict → `{dict_keys, dict_schema}`, list → `{list_row_schema}`, scalar → dtype 이름
- [x] `normalize_response(status_code, payload)` 가 `{status_code, shape}` 반환
- [x] `backend/tests/test_characterization_api.py::test_endpoint_snapshot` 가 `normalize_response` 결과로 snapshot 비교
- [x] `backend/tests/__snapshots__/test_characterization_api.ambr` 가 새 shape 만 박힘 (45056 줄 → ~700 줄)
- [x] `backend/tests/test_snapshot_normalize.py` 11 케이스 (drift 면역 4 + 회귀 검출 7)
- [x] `.gitignore` 의 `test_cp*.py` 패턴과 무관한 파일명 (영구 안전망 정책 일관)
- [x] 운영 코드 / ML 모델 / 응답 schema 자체 0 수정 (정규화 함수와 테스트만 수정)

## 2. 새 테스트 결과

### 2.1 Characterization snapshot (CP223 회복)

```
pytest backend/tests/test_characterization_api.py -q
9 snapshots passed. 9 passed
```

9 endpoint:
- stocks_list ✅
- aapl_prices ✅
- aapl_indicators ✅
- aapl_product_history ✅
- aapl_line ✅
- aapl_band_1d ✅
- aapl_band_1w ✅
- scan_indicator ✅
- backtest_aapl ✅

### 2.2 Drift simulation (정규화 자체 검증)

```
pytest backend/tests/test_snapshot_normalize.py -v
11 passed in 0.09s
```

| # | 테스트 | 분류 |
|---|---|---|
| 1 | test_new_row_added_does_not_break | Drift 면역 |
| 2 | test_scalar_value_change_does_not_break | Drift 면역 |
| 3 | test_cumulative_list_grows_does_not_break | Drift 면역 |
| 4 | test_nested_scalar_change_does_not_break | Drift 면역 |
| 5 | test_row_key_added_is_detected | 회귀 검출 |
| 6 | test_row_key_removed_is_detected | 회귀 검출 |
| 7 | test_dtype_change_is_detected | 회귀 검출 |
| 8 | test_nested_schema_change_is_detected | 회귀 검출 |
| 9 | test_top_level_key_added_is_detected | 회귀 검출 |
| 10 | test_status_code_change_is_detected | 회귀 검출 |
| 11 | test_empty_list_to_populated_is_detected | 회귀 검출 |

### 2.3 비결정성 검증 (Step 5: 2x stable run)

같은 응답을 2회 실행 → 0 diff 확인 (둘 다 9 passed). 정규화 결정성 OK.

## 3. Dry-run / 시뮬레이션 결과

CP237.5 는 ML 모델 forward 호출 없음 (테스트 정규화만 수정). 대신 drift
simulation 으로 정규화 헬퍼의 의도 검증:

- daily refresh 시뮬 (test #1~4): 같은 schema 의 새 row 추가, scalar 값
  변동, cumulative list 길이 증가, nested scalar 변동 → 모두 normalized
  결과 동일 (drift 면역 ✅)
- 회귀 시뮬 (test #5~11): row key 추가/제거, dtype 변경, nested schema
  변경, top-level key 추가, status code 변경, 빈 list → row 채움 → 모두
  normalized 결과 다름 (회귀 검출 ✅)

## 4. 기존 회귀 통과 건수

### 4.1 회귀 안전망 (사용자 명시: CP223 + CP230)

| 안전망 | 결과 |
|---|---|
| CP223 characterization snapshot (9 endpoint) | ✅ 9 passed |
| CP237.5 drift simulation (11 케이스) | ✅ 11 passed |
| CP230 frontend smoke (Vitest) | ✅ 8 files / 166 passed (4 todo, 1 skipped) |

### 4.2 backend pytest 전체 (참고)

```
pytest backend/tests --ignore=backend/tests/test_services.py -q
118 passed, 11 failed, 2 skipped
```

**11 failed 는 모두 CP237.5 와 무관한 pre-existing**:
- `test_api.py::test_legacy_predict_*` / `test_prediction_*` (7개):
  `app.routers.v1.stocks` 의 `get_latest_prediction_data` 속성 부재
  (이전 라우터 리팩토링 결과)
- `test_market_data_providers.py::test_provider_config_defaults_yfinance_fallback_to_eodhd`:
  `settings.market_data_fallback_provider` 가 `'yahoo'` 반환 (test 는
  `'eodhd'` 기대 — settings 변경 결과)
- `test_product_prediction_history_api.py::test_product_history_*` (3개):
  product prediction history API 변경 결과

CP237.5 의 변경 파일 (`_snapshot_normalize.py`, `test_characterization_api.py`,
`.ambr`, `test_snapshot_normalize.py`) 는 위 fail 들과 import / fixture
의존성 없음. 별도 CP 에서 정리 권고.

## 5. 진행 중 발견된 함정 4 개

지시서 §2 와 실제 응답 schema mismatch / drift 패턴 mismatch.

1. **DRIFT_FIELDS mismatch**: 지시서의 `actual_h{1,4,20}_return` /
   `created_at` / `latest_*` / `as_of` 등은 실제 응답에 없음. 반대로
   `actual_return` / `actual_return_available` / `forecast_date` /
   `date` / `asofDate` (camelCase) 가 누락. 실측 후 갱신.

2. **last_n=5 fragile**: rolling window 8 endpoint 에서 last 5 가 매일
   row swap. drift 필드 제거해도 row 내 `band_lower` / `line_score` /
   `band_upper` 등이 매일 새 값 → 매일 mismatch.

3. **scalar value 박는 정규화 fragile**: backtest_aapl 의 top-level
   metric (`averageHoldingDays` 등) 이 daily refresh 마다 변동 → 매일
   mismatch.

4. **1-level shallow dict 부족**: 응답이 `{data: dict{data: list}}`
   중첩이라 inner list 의 row schema 가 비교 대상에서 빠짐. 보안 트랙
   회귀 안전망 절반 무력.

해결: 함정 1 은 `DRIFT_FIELDS` 실측 갱신 (현 정규화 전략에선 실용 가치
약하지만 v2 fixture 기반 row 비교 시 활용 위해 기록 보전). 함정 2/3/4
는 정규화 함수 통째 재작성 — depth 무제한 재귀, `len` 도 안 박음, scalar
는 dtype 만. 자세한 결정 근거는 ADR-0028.5.

## 6. 산출물

신규:
- `backend/tests/_snapshot_normalize.py` (138 줄)
- `backend/tests/test_snapshot_normalize.py` (drift sim 11 케이스)
- `docs/adr/0028_5_snapshot_drift_resilient.md`
- `docs/cp237_5_snapshot_redesign_report.md` (본 보고서)

수정:
- `backend/tests/test_characterization_api.py` (import + normalize 호출)
- `backend/tests/__snapshots__/test_characterization_api.ambr` (재생성 ×2)

운영 코드 / ML 모델 / 응답 schema 자체 / dataloader / cron / API contract 0 수정.

## 7. 인터페이스 보존

| 항목 | 상태 |
|---|---|
| API contract (응답 schema) | 0 수정 |
| 모델 forward 인터페이스 | 0 수정 |
| dataloader / calendar features | 0 수정 |
| daily refresh cron (`run_v1_unified_refresh_local.ps1`) | 0 수정 |
| 운영 모델 3 개 (CP210 / CP153 / CP178) 응답 정확성 | 유지 |
| CI workflow의 `--ignore=backend/tests/test_services.py` 패턴 | 보존 |

## 8. commit 이력 (5 commit)

```
4f2e852 CP237.5 Step 1+2: add _snapshot_normalize.py (drift-resilient helper)
91c4679 CP237.5 Step 3: apply normalize_response in test_characterization_api
200bef4 CP237.5 Step 4+5: regenerate snapshots (drift-resilient baseline) + 2x stable
2d3bad1 CP237.5 Step 4 보강: 재귀 정규화로 helper 재작성 + baseline 두 번째 갱신
4eeb8ee CP237.5 Step 6: drift simulation 테스트 (11 케이스)
```

본 commit (보고서 + ADR) 이 6번째.

## 9. 다음 CP

CP238 (Dependency CVE audit). 진입 조건 (CP237.5 완료) 충족.
