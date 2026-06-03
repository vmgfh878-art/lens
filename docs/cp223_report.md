# CP223 보고서 — 백엔드 Characterization 스냅샷 안전망

- **상태**: ✅ 완료
- **기간**: 2026-06-03 (단일 세션, CP222 직후)
- **모드**: code (구현 + 자가 점검)
- **선행 의존**: CP222 (안전망 도구) ✅
- **마지막 그린 커밋**: e730db3 (Step 3 baseline) → Step 7 커밋 후 갱신

---

## 요구

CP225+ 백엔드 분리 리팩토링을 시작하려면 분리 전후 응답이 "1바이트도(허용 tolerance 내) 안 바뀌었다"를 기계적으로 증명할 안전망 필요. 기존 `backend/tests/test_api.py`는 전부 mock 기반이라 실제 parquet → 직렬화 경로 미통과. **운영 코드 0 변경**으로 9개 read-path endpoint를 syrupy + float normalize 조합으로 박제.

---

## 한 일 (Sub-step 별)

### Step 1 — 도구 설치 + conftest (커밋 `42d669c`)
- **snaptol PyPI 부재** (`Could not find a version that satisfies the requirement snaptol`) → 차단 트리거 3 발동 → 사용자 결정 = **syrupy + float normalize** 폴백.
- `requirements-dev.txt` 보강: `syrupy==5.3.1` 추가 (snaptol 주석 정정).
- `backend/tests/conftest.py` 신규:
  - sys.path 이중 보정 (ROOT + BACKEND)
  - `LENS_FORCE_LOCAL=1` 강제 (Supabase 차단)
  - `client` fixture (session scope, TestClient + 4개 캐시 cold reset: `parquet_store.clear_all` / `local_market_svc.clear_caches` / `strategy_backtest_svc.clear_strategy_cache` / `product_prediction_history_svc.clear_product_history_cache`)
  - `FIXED_HEADERS = {"X-Request-Id": "test-fixed"}`
  - `normalize_floats(obj, ndigits=9)` 헬퍼 (NaN/Inf→None + round 재귀)

### Step 2 — 9 endpoint 스냅샷 테스트 (커밋 `1fc1d15`)
- `backend/tests/test_characterization_api.py` 신규. `@pytest.mark.parametrize` 9 케이스:

| id | path | params |
|---|---|---|
| stocks_list | `/api/v1/stocks` | `{"limit": 50}` |
| aapl_prices | `/api/v1/stocks/AAPL/prices` | `{"start":"2025-01-02","end":"2025-06-30"}` |
| aapl_indicators | `/api/v1/stocks/AAPL/indicators` | `{"limit": 300}` |
| aapl_product_history | `/api/v1/stocks/AAPL/predictions/product-history` | `{"lookback_days": 60}` |
| aapl_line | `/api/v1/predictions/line/AAPL` | `{"days": 365}` |
| aapl_band_1d | `/api/v1/predictions/band/1d/AAPL` | `{"days": 365}` |
| aapl_band_1w | `/api/v1/predictions/band/1w/AAPL` | `{"days": 730}` |
| scan_indicator | `/api/v1/strategies/indicator_balance_v2/scan` | `{"limit": 500}` |
| backtest_aapl | `/api/v1/strategies/indicator_balance_v2/backtest/AAPL` | `{}` |

- 가드: `meta.request_id == "test-fixed"` 미준수 시 fail (uuid leak 차단).
- 검증 결과: baseline 없이 실행 → **9 snapshots failed** (예상대로 — 200 통과 후 syrupy 박제 부재로 fail).
- conftest docstring `\l` escape DeprecationWarning 정정 (forward slash).

### Step 3 — baseline 생성 (커밋 `e730db3`)
- `pytest --snapshot-update` → **9 snapshots generated, 9 passed**.
- baseline 파일: `backend/tests/__snapshots__/test_characterization_api.ambr` (1.55MB, 44,786 라인).
- 비결정 흔적 검증:
  - request_id 9/9 모두 `'test-fixed'` 박제됨 ✅
  - uuid 패턴 (8-4-4-4-12 hex) 전체 0건 ✅
  - float 9자리 정규화 (예: `0.007783874`, `198.744185387`) ✅
  - 날짜는 역사 데이터 (`2025-06-02` 등), 오늘 날짜 누수 없음 ✅

### Step 4 — 재실행 diff=0 (커밋 없음, 검증만)
- `pytest` 단순 재실행 → **9 snapshots passed**, **9 passed**.
- `git status --porcelain backend/tests` → 빈 출력 (clean). 스냅샷 재생성 0.

### Step 5 — 2회 동일 + 회귀 0 (커밋 없음, 검증만)
- 연속 2회 실행: 각 9 passed, git clean.
- backend/tests 전체: **87 passed** (78 CP222 baseline + 9 characterization) / 11 failed / 1 error.
- CP222 baseline 78 passed **무손실**. 신규 실패 0. **회귀 0 확정**.

### Step 7 — 보고서 + ADR (이 커밋)
- `docs/adr/0011-characterization-snaptol-for-ml.md` 신규.
- 본 보고서 `docs/cp223_report.md` 신규.

---

## 인터페이스 보존 (성공 기준)

- **운영 코드 `backend/app/**` 0 라인 변경.** request_id.py / api_service.py 무수정.
- 함수 signature / API 응답 schema / props 무변경.
- 비결정성은 conftest의 호출 측 fixture(`FIXED_HEADERS`, `normalize_floats`)로만 해결.
- 기존 `backend/tests/test_api.py` 등 unittest 테스트 무수정. 회귀 0.

---

## 핵심 컴포넌트 존재 체크리스트 (메타 D21)

- `app.services.parquet_store.clear_all` ✅ (실측 backend/app/services/parquet_store.py:98)
- `app.services.local_market_svc.clear_caches` ✅ (:36)
- `app.services.strategy_backtest_svc.clear_strategy_cache` ✅ (:568)
- `app.services.product_prediction_history_svc.clear_product_history_cache` ✅ (:116)
- `backend/data/v1/*.parquet` 9개 모두 존재 ✅ (market 4 + predictions 3 + product 1 + 1 backup)
- syrupy 5.3.1 `snapshot` fixture 정상 동작 ✅
- `meta.request_id == "test-fixed"` 가드 ✅
- `LENS_FORCE_LOCAL=1` Supabase 차단 ✅
- 운영 코드 0 변경 ✅ (`git diff backend/app/` 비어있음)

---

## 새 테스트 결과 (메타 D21)

- 신규 테스트 9개: `test_endpoint_snapshot[stocks_list/aapl_prices/aapl_indicators/aapl_product_history/aapl_line/aapl_band_1d/aapl_band_1w/scan_indicator/backtest_aapl]`.
- 모두 200 + snapshot 박제 + 재실행 diff=0 + 2회 동일.

## Dry-run 결과 (메타 D21)

- baseline 생성: 9 snapshots generated, 9 passed (4.52s).
- 재실행: 9 snapshots passed (4.47s).
- 2회 연속 실행: 각 ~4.5s, 결과 동일.
- 운영 코드 변경 0줄 확인 (`git diff backend/app/`).

## 기존 회귀 통과 건수 (메타 D21)

- CP222 baseline backend/tests 78 passed → CP223 후 87 passed (+9 characterization, 기존 78 무손실).
- failed 11 / error 1 — 모두 pre-existing (CP222 baseline과 동일).

---

## 성공 기준 충족표

| 항목 | 기준 | 실측 | 결과 |
|---|---|---|---|
| 스냅샷 baseline 생성 | 9 케이스 git 추적 | 1.55MB ambr 파일 git 추적 | ✅ |
| endpoint 커버 | ≥6 | **9** | ✅ |
| 재실행 diff | 0 | 9 passed, git clean | ✅ |
| 2회 실행 동일 | 동일 | 각 9 passed | ✅ |
| 기존 테스트 회귀 | 0 | 78 → 78 (무손실) + 9 신규 | ✅ |
| 운영 코드 변경 | 0 라인 | 0 라인 | ✅ |

---

## 후속

- **CP225 (다음)**: `feature_svc.py` 591줄 분리. CP223 baseline diff=0 유지하면서 진행.
- **CP226**: `strategy_backtest_svc.py` 590줄 분리.
- **별도**: v1 parquet hash 가드 conftest (운영 데이터 보호).
- **별도**: test_api.py 7건 + DatasetPlan signature 4건 등 stale reference cleanup.

---

## 자가 점검

- **[Plan v3 정합]** **PASS** — CP223은 read-path 박제. 학습/calibration/fidelity/EODHD/α=1·β=2 등 Plan v3 의사결정에 영향 0.
- **[구조 결함]** **PASS** — conftest sys.path 이중 보정이 기존 mock 테스트와 충돌 없음 (78 passed 무손실). LENS_FORCE_LOCAL=1이 `test_ready_health` 류의 `patch.dict(clear=True)` 동작과 충돌 없음 확인. syrupy ambr 위치 backend/tests/__snapshots__/는 표준.
- **[모델 영향]** **PASS** — 운영 코드 0 라인. 학습 weights / calibration / preprocess / sufficiency gate 무관. parquet read-only. ForecastOutput, RevIN, channel layout 모두 무변경.
