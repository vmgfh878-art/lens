# CP223 백엔드 Characterization 스냅샷 안전망 (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고 검증하고 중단 판단을 한다. 추측 금지, 실제 출력 박제.

## 역할 고정
- 모드: **code** (구현 + 자가 점검만 보고).
- 권한: 코드 수정, 로컬 검증(pytest 실행, TestClient 호출, lint)만 허용.
- 금지:
  - 새 모델 학습 / 새 calibration / 가중치 재생성 금지.
  - DB write 금지. **Supabase 호출 금지** (이 CP는 `LENS_FORCE_LOCAL=1`로 local parquet 경로만 탄다).
  - 사용자가 직접 수정한 파일(`backend/app/**` 운영 코드)을 revert 하거나 동작을 바꾸지 마라. 이 CP는 **테스트/conftest/스냅샷 baseline만 추가**한다. 운영 코드는 한 줄도 수정하지 않는다.
- 자가 점검: 완료 후 [Plan v3 정합] [구조 결함] [모델 영향] 각각 PASS/WARN/FAIL + 사유 1줄.
- 커밋 메시지: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 환경
- 워킹 디렉토리: `C:\Users\user\lens` (git root, `git rev-parse --show-toplevel` 확인됨).
- venv: `C:\Users\user\lens\.venv` (Python 3.10.0, torch 2.11.0+cu128). Python 실행은 `C:\Users\user\lens\.venv\Scripts\python.exe`.
- 백엔드 코드 루트: `C:\Users\user\lens\backend`. 앱은 `app.*` 절대 import(예: `from app.main import app`)를 쓰므로 `backend/`가 sys.path에 있어야 하고, **동시에** `app/db.py:7`이 `from backend.collector.utils.network import sanitize_proxy_env`를 하므로 **repo root도** sys.path에 있어야 한다. → `PYTHONPATH`에 두 경로(`C:\Users\user\lens;C:\Users\user\lens\backend`)를 모두 넣는다. (이 사실은 conftest에서 코드로 보장한다. Sub-step 1 참조.)
- 백엔드 기동(이 CP는 서버 기동 불필요, TestClient in-process 사용): 참고용으로 `scripts\start_demo.ps1` 또는 `uvicorn app.main:app`.
- 프론트(이 CP 무관): `npm run dev`.
- 검증용 포트 충돌 회피: 이 CP는 `TestClient(app)`만 쓰므로 **포트를 점유하지 않는다**. 별도 uvicorn을 띄우지 마라(8000 등 충돌 방지). 만약 디버깅으로 띄운다면 사용 후 반드시 종료.

## 진단 (근거)
백엔드 구조 분리(CP225~)를 하려면, 분리 전후 **응답이 1바이트도 안 바뀌었다**를 기계적으로 증명할 안전망이 먼저 있어야 한다(Feathers, *Working Effectively with Legacy Code*; Fowler, *Refactoring* 2e — characterization test = "의도가 아니라 현재 실제 출력을 박제"). 현재 백엔드에는 characterization 스냅샷이 **전무**하다.

현재 테스트 자산(실측):
- `backend/tests/test_api.py` (529줄, `unittest.TestCase` 기반). 전부 **mock** 기반(`patch("app.routers.v1.stocks.get_price_response_data", return_value=...)`)이라 **실제 parquet → 실제 직렬화 경로를 한 번도 통과하지 않는다**. 즉 분리 리팩토링이 직렬화/정렬/필터 로직을 깨도 이 테스트는 초록색을 유지한다. 안전망이 못 된다.
- `backend/tests/test_services.py`, `test_feature_svc.py`, `test_collector_jobs.py` 존재(스냅샷 아님).
- `.venv`에 **pytest 미설치**, **snaptol 미설치**, **syrupy 미설치**(실측: `python -m pytest --version` → `No module named pytest`).
- `httpx==0.27.0`, `fastapi==0.111.0` 설치됨 → `starlette.testclient.TestClient` 동작 확인됨.

운영 데이터/엔드포인트 구조(실측, **스펙 원안과 다름 — 아래가 사실**):
- 모든 v1 market/strategy/prediction 응답은 **local parquet** (`C:\Users\user\lens\backend\data\v1\*.parquet`)에서 나온다. Supabase 경로는 v1에서 비활성(`backend/app/services/api_service.py:16` 주석 "Supabase 경로는 v1 에서 비활성. 모든 market/stocks 조회는 local parquet 직접 읽음").
- 실제 parquet 파일(실측, `C:\Users\user\lens\backend\data\v1\`):
  - `market_prices_1d.parquet`, `market_prices_1w.parquet`, `market_indicators_1d.parquet`, `market_stock_info.parquet`
  - `predictions_line_1d.parquet`, `predictions_band_1d.parquet`, `predictions_band_1w.parquet`
  - `product_prediction_history_1D.parquet` + `product_prediction_history_1D.manifest.json`
- 실제 mount(`backend/app/main.py:58-63`): `health, stocks, ai, admin, predictions, strategies` 전부 prefix `/api/v1`.
- 라우터 ↔ 서비스 ↔ parquet 매핑(실측):
  | 엔드포인트 | 라우터 파일:줄 | 서비스 | 데이터 |
  |---|---|---|---|
  | `GET /api/v1/stocks` | `backend/app/routers/v1/stocks.py:34-47` `list_stocks` | `api_service.get_stocks` → `local_market_svc.fetch_stocks_local` | `market_stock_info.parquet` |
  | `GET /api/v1/stocks/{t}/prices` | `stocks.py:50-70` `get_prices` | `api_service.get_price_response_data` → `fetch_price_rows_local` | `market_prices_1d.parquet` |
  | `GET /api/v1/stocks/{t}/indicators` | `stocks.py:73-87` `get_indicators` | `api_service.get_indicator_response_data` → `fetch_indicator_rows_local` | `market_indicators_1d.parquet` |
  | `GET /api/v1/stocks/{t}/predictions/product-history` | `stocks.py:90-115` `get_product_prediction_history` | `product_prediction_history_svc.get_product_prediction_history_data` | `product_prediction_history_1D.parquet` |
  | `GET /api/v1/predictions/line/{t}` | `backend/app/routers/v1/predictions.py:123-150` `get_line` | `parquet_store.get_raw("line_1d")` | `predictions_line_1d.parquet` |
  | `GET /api/v1/predictions/band/1d/{t}` | `predictions.py:153-184` `get_band_1d` | `parquet_store.get_raw("band_1d")` | `predictions_band_1d.parquet` |
  | `GET /api/v1/predictions/band/1w/{t}` | `predictions.py:187-218` `get_band_1w` | `parquet_store.get_raw("band_1w")` | `predictions_band_1w.parquet` |
  | `GET /api/v1/strategies/{id}/scan` | `backend/app/routers/v1/strategies.py:41-51` `scan_strategy` | `strategy_backtest_svc.get_strategy_scan` | `market_*` + `predictions_*` merge |
  | `GET /api/v1/strategies/{id}/backtest/{t}` | `strategies.py:54-64` `backtest_strategy_ticker` | `strategy_backtest_svc.get_strategy_backtest` | 동상 |
- 전략 ID(실측, `backend/app/strategies/strategy_rules.py:39-93`): `indicator_balance_v2`(line/band 미사용 — 가장 단순·견고), `ai_balance_v2`, `ai_band_defense_v1`.

**비결정성 원천 — 이 CP의 핵심 위험 (실측 2건):**
1. `backend/app/middleware/request_id.py:9` — 모든 응답 `meta.request_id`가 `str(uuid4())` 또는 헤더 `X-Request-Id` 에코. 매 호출 달라진다. → 스냅샷에서 반드시 제거/정규화하거나 헤더로 고정해야 함.
2. `backend/app/services/api_service.py:57-62` `resolve_price_window` — `end`가 None이면 `resolved_end = date.today()`. 즉 `/stocks/{t}/prices`를 start/end 없이 호출하면 **오늘 날짜에 의존** → 매일 응답이 달라진다. → 스냅샷 호출 시 `start`/`end`를 **명시 고정**해야 함.
- 그 외(실측): float NaN/Inf는 서비스단에서 모두 `None`으로 정규화됨(`_jsonable`/`_finite_or_none`). dict key 순서는 Python 3.7+ 삽입순 고정. parquet 디스크 행 순서는 `lru_cache`된 frame 위에서 안정.

**선행 실측 (이 CP가 안전망으로 성립하는 근거):** 위 9개 엔드포인트를 `LENS_FORCE_LOCAL=1` + `PYTHONPATH=repo;backend`로 in-process `TestClient` 2회 호출하고 `meta.request_id`만 제거해 비교한 결과 — **9/9 모두 status 200, 2회 호출 byte-동일(stable=True)**. (line/band/scan/backtest 포함, backtest/AAPL 응답 ~160KB.) 즉 동일 프로세스 내 결정성은 확인됨. **남은 위험은 (a) 프로세스 재시작 시 float 재계산 미세차, (b) pandas/numpy 버전 차이** — 이것이 snaptol tolerance가 필요한 이유.

조사 출처: 위 표/줄번호는 본 세션에서 `Read`로 직접 확인. 2회-동일 결과는 본 세션에서 `.venv` python + TestClient 실측. 원칙은 Feathers(Characterization Test) / Fowler Refactoring 2e.

## 선행 의존
- **없음.** CP223은 안전망 그 자체이므로 어떤 분리 CP보다 먼저 그린이 되어야 한다. (역으로: CP225~ 백엔드 분리는 CP223 스냅샷 그린이 선행 조건이다.)
- 단, 전제 데이터: `backend/data/v1/*.parquet`가 디스크에 존재해야 한다(실측 존재). 누락 시 차단 트리거 참조.

## 범위
**포함:**
- pytest + float-tolerance 스냅샷 도구 설치 및 `requirements-dev.txt` 고정.
- `backend/tests/conftest.py` 신규: `sys.path` 보정 + `LENS_FORCE_LOCAL` 강제 + cache 초기화 + `TestClient` fixture.
- `backend/tests/test_characterization_api.py` 신규: 9개 엔드포인트 응답 스냅샷.
- 스냅샷 baseline 생성 + git 커밋.
- 재실행 diff=0 확인, 2회 실행 동일성(비결정성) 점검.

**제외:**
- 운영 코드(`backend/app/**`) 수정 일절 금지(특히 `request_id.py`, `api_service.py`의 `date.today()`는 **고치지 말고** 스냅샷 측에서 회피). 동작 변경은 별도 CP.
- **Supabase 경로 스냅샷 보류** (v1에서 비활성이고 이 CP는 local만). Supabase 기반 엔드포인트가 부활하면 별도 CP에서 다룬다.
- `ai` 라우터(`/api/v1/ai/runs*`), `admin` 라우터 스냅샷 제외(각각 외부 mock / write 트리거 성격 → 안전망 핵심 아님. 후속 CP 여지).
- 에러 경로(404/422/503) 스냅샷 제외(이미 `test_api.py`가 mock으로 커버). 이 CP는 **정상 200 응답 박제**에 집중.

## Sub-step (Strangler Fig, 작은 단위)
> 이 CP는 "옛 코드 제거"가 없는 **순수 추가**다. Strangler 패턴은 "새 안전망을 옆에 세우고(공존) → 그 위에서 후속 CP가 옛 구조를 교체"로 적용된다. 각 Step 끝에 커밋 + 검증. 한 Step = 한 revert 단위.

### Step 1 — 도구 설치 + dev 의존성 고정 + conftest fixture
1. 스냅샷 도구 결정(가용성 확인 우선순위):
   - 1순위 **`snaptol`** (PlasmaFAIR; numpy `assert_allclose` rtol/atol 내장 — ML float 출력에 적합).
   - 설치 시도:
     ```powershell
     & C:\Users\user\lens\.venv\Scripts\python.exe -m pip install pytest snaptol
     & C:\Users\user\lens\.venv\Scripts\python.exe -c "import snaptol; print('snaptol', getattr(snaptol,'__version__','?'))"
     ```
   - **`snaptol` 설치/임포트 실패 시** → 2순위로 폴백하되 **임의 선택 금지, 차단 보고**(아래 차단 트리거 "도구 부재"). 폴백 후보 순서: (a) `syrupy` (결정론적 dict/스키마엔 OK이나 float tolerance 없음 → 정수/문자열 필드만), (b) **자체 fixture**: 표준 `unittest`/`pytest` + `math.isclose`/`pytest.approx(rel=1e-6, abs=1e-9)` 기반 재귀 비교 + baseline JSON을 `backend/tests/snapshots/`에 저장. 폴백을 쓸지 여부는 **사용자 판단**을 받는다.
2. `backend/requirements-dev.txt` 신규 작성(설치 성공한 버전 핀):
   ```
   pytest==<설치된버전>
   snaptol==<설치된버전>     # 폴백 시 이 줄 제거하고 사유 주석
   ```
3. `backend/tests/conftest.py` 신규. 반드시 다음을 보장:
   - **sys.path 보정**: repo root(`parents[2]`)와 `backend`(`parents[1]`)를 둘 다 `sys.path`에 삽입(진단의 이중 import 문제 해결).
   - `os.environ["LENS_FORCE_LOCAL"] = "1"` (Supabase 차단, 진단 1·`app/db.py:14-22` 근거).
   - 결정성 위해 **세션 시작 시 캐시 초기화**: `parquet_store.clear_all()`, `local_market_svc.clear_caches()`, `strategy_backtest_svc.clear_strategy_cache()`, `product_prediction_history_svc.clear_product_history_cache()` (모두 실측 존재하는 공개 함수).
   - `client` fixture: `TestClient(app)` 반환. **요청 시 `headers={"X-Request-Id": "test-fixed"}` 고정** → `meta.request_id`가 항상 `"test-fixed"`가 되어 비결정성 제거(진단 1, `request_id.py:9`가 헤더를 우선 에코함).
   - (참고 골격, 실행자가 검증 후 확정):
     ```python
     import sys, os
     from pathlib import Path
     import pytest
     ROOT = Path(__file__).resolve().parents[2]   # C:\Users\user\lens
     BACKEND = Path(__file__).resolve().parents[1] # ...\backend
     for p in (str(ROOT), str(BACKEND)):
         if p not in sys.path:
             sys.path.insert(0, p)
     os.environ["LENS_FORCE_LOCAL"] = "1"

     @pytest.fixture(scope="session")
     def client():
         from app.main import app
         from app.services import parquet_store, local_market_svc, strategy_backtest_svc
         from app.services import product_prediction_history_svc as pph
         parquet_store.clear_all(); local_market_svc.clear_caches()
         strategy_backtest_svc.clear_strategy_cache(); pph.clear_product_history_cache()
         from starlette.testclient import TestClient
         return TestClient(app)

     FIXED_HEADERS = {"X-Request-Id": "test-fixed"}
     ```
4. **검증**: `& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/conftest.py --collect-only -q` (에러 없이 수집) 또는 빈 더미 테스트로 fixture 임포트 확인.
5. **커밋**: `test(cp223): pytest+snaptol 도입 및 characterization conftest fixture` (운영 코드 변경 없음 — 이 커밋엔 `requirements-dev.txt`, `conftest.py`만).

### Step 2 — characterization 스냅샷 테스트 작성 (baseline 미생성, 호출만)
1. `backend/tests/test_characterization_api.py` 신규. 9개 케이스. **각 케이스는 파라미터를 고정**한다(비결정성 회피):
   | 케이스 | path | params(고정) |
   |---|---|---|
   | stocks_list | `/api/v1/stocks` | `{"limit": 50}` |
   | aapl_prices | `/api/v1/stocks/AAPL/prices` | `{"start":"2025-01-02","end":"2025-06-30"}` ← `date.today()` 회피 필수 |
   | aapl_indicators | `/api/v1/stocks/AAPL/indicators` | `{"limit": 300}` |
   | aapl_product_history | `/api/v1/stocks/AAPL/predictions/product-history` | `{"lookback_days": 60}` |
   | aapl_line | `/api/v1/predictions/line/AAPL` | `{"days": 365}` |
   | aapl_band_1d | `/api/v1/predictions/band/1d/AAPL` | `{"days": 365}` |
   | aapl_band_1w | `/api/v1/predictions/band/1w/AAPL` | `{"days": 730}` |
   | scan_indicator | `/api/v1/strategies/indicator_balance_v2/scan` | `{"limit": 500}` |
   | backtest_aapl | `/api/v1/strategies/indicator_balance_v2/backtest/AAPL` | `{}` |
   - 모든 요청에 `headers=FIXED_HEADERS`.
   - 각 테스트: `assert resp.status_code == 200` 먼저 검증한 뒤 **응답 JSON 전체를 스냅샷 대상**으로 넘긴다. `meta.request_id`는 고정 헤더로 이미 `"test-fixed"`라 박제 가능(또는 비교 전 pop). float 필드는 snaptol tolerance(`rtol=1e-6, atol=1e-9` 권장 출발값)로 비교.
   - snaptol 사용 예(도구 확정 후 실제 API에 맞춰 실행자가 조정):
     ```python
     def test_aapl_line(client, snapshot):  # snaptol fixture명은 설치 후 확인
         r = client.get("/api/v1/predictions/line/AAPL", params={"days":365}, headers=FIXED_HEADERS)
         assert r.status_code == 200
         snapshot.assert_match(r.json())   # 실제 API는 snaptol docs로 확정
     ```
2. **검증**: baseline 없이 한 번 실행 → snaptol이 "snapshot 없음"으로 실패/스킵하는지 확인(도구별 동작 차이 인지). 아직 `--snapshot-update` 안 함.
3. **커밋**: `test(cp223): 9개 endpoint characterization 스냅샷 테스트 작성` (baseline 파일 제외).

### Step 3 — baseline 생성 + 커밋
1. baseline 생성:
   ```powershell
   & C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py --snapshot-update -q
   ```
   (snaptol의 실제 update 플래그는 설치 후 `pytest -h | Select-String snapshot`로 확인해 사용. syrupy 폴백 시 `--snapshot-update` 동일.)
2. 생성된 baseline 파일 위치 확인(snaptol/syrupy는 보통 `__snapshots__/` 또는 `_snapshots/`). git 추적되도록 `git add`.
3. baseline JSON을 **사람 눈으로 1개 열어** 비결정 흔적(타임스탬프/uuid/오늘날짜) 없는지 확인. `meta.request_id`가 `"test-fixed"`로 박제됐는지 확인.
4. **커밋**: `test(cp223): characterization baseline 스냅샷 생성` (baseline 파일 포함).

### Step 4 — 재실행 diff=0 확인
1. update 없이 재실행:
   ```powershell
   & C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py -q
   ```
2. 기대: **9 passed, diff 0**. 실패 시 → 차단 트리거(비결정성).
3. 커밋 없음(검증 단계). 단 이 시점에서 `git status`가 clean이어야 한다(스냅샷 파일이 재생성으로 바뀌면 안 됨).

### Step 5 — 비결정성 점검 (2회 실행 동일)
1. 연속 2회 실행 후 종료코드/통과수 동일 확인:
   ```powershell
   & C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py -q
   & C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py -q
   ```
   둘 다 `9 passed`, `git status` clean.
2. (강화) 캐시 콜드 상태 차이 점검: 한 번은 그대로, 한 번은 프로세스 새로 띄워(별도 pytest invocation = 새 프로세스라 이미 콜드) 실행 — 위 2회가 이미 별도 프로세스이므로 충족. 그래도 미세 float 변동으로 실패하면 tolerance 조정 여부를 **차단 보고**.
3. 기존 회귀 0 확인: 전체 백엔드 테스트가 깨지지 않았는지(이 CP는 추가만 했으므로):
   ```powershell
   & C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests -q
   ```
   (주의: `test_api.py`는 unittest 스타일이지만 pytest가 수집·실행 가능. 만약 conftest의 `LENS_FORCE_LOCAL`/sys.path 변경이 기존 mock 테스트를 깨면 → 차단 보고. 깨지면 안 됨: 기존 테스트는 전부 mock이라 환경변수 영향 없음.)
4. 커밋 없음.

## 인터페이스 보존
- 이 CP는 **운영 코드의 함수 signature / API 응답 schema / 동작을 일절 바꾸지 않는다.** 추가하는 것은 테스트·conftest·baseline·dev 의존성뿐.
- `request_id.py`, `api_service.py`(특히 `date.today()`)를 **수정하지 않는다.** 비결정성은 호출 측(고정 헤더 + 고정 start/end)에서 회피한다. 만약 "스냅샷을 위해 운영 코드를 바꿔야 한다"는 판단이 들면 → **즉시 중단·보고**(그건 다른 CP의 동작 변경이다).
- `conftest.py`가 설정하는 `os.environ["LENS_FORCE_LOCAL"]="1"`는 테스트 프로세스 한정. 운영 기동에는 영향 없음. 그래도 기존 `test_api.py`(특히 `test_ready_health_returns_config_error_when_env_missing`이 `patch.dict(os.environ, {}, clear=True)`로 환경을 비움)와의 상호작용을 Step 5-3에서 반드시 확인.

## 성공 기준 (측정 가능)
| 항목 | 기준 |
|---|---|
| 스냅샷 baseline 생성 | 9개 케이스 baseline 파일 git 추적 |
| endpoint 커버 | 6개 이상 — 본 CP는 **9개**(stocks list/prices/indicators/product-history, predictions line/band1d/band1w, strategies scan/backtest) |
| 재실행 diff | 0 (Step 4: `9 passed`, `git status` clean) |
| 2회 실행 동일성 | 동일 (Step 5: 연속 2회 `9 passed`) |
| 기존 테스트 회귀 | 0 (`pytest backend/tests` 기존 케이스 통과수 유지) |
| 운영 코드 변경 | 0 줄 (`git diff` 상 `backend/app/**` 무변경) |
| 예상 시간 | 2~3시간(도구 설치/플래그 확인 포함) |

## 검증
```powershell
# 0) 도구 확인
& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest --version
& C:\Users\user\lens\.venv\Scripts\python.exe -c "import snaptol; print(snaptol.__version__)"

# 1) conftest 수집 OK
& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py --collect-only -q
# 기대: 9 tests collected, 0 errors

# 2) baseline 생성 (최초 1회)
& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py --snapshot-update -q
# 기대: 9 snapshots written

# 3) 재실행 diff 0
& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py -q
# 기대: 9 passed

# 4) 비결정성 2회 동일 + git clean
& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests/test_characterization_api.py -q
git -C C:\Users\user\lens status --porcelain backend/tests
# 기대: 두 번째 실행도 9 passed, status 출력 비어있음(스냅샷 안 바뀜)

# 5) 전체 회귀 0
& C:\Users\user\lens\.venv\Scripts\python.exe -m pytest backend/tests -q
# 기대: 기존 + 신규 모두 통과, 기존 통과수 감소 없음
```
참고(독립 sanity, 도구 무관): 본 directive 작성 시 `PYTHONPATH="C:\Users\user\lens;C:\Users\user\lens\backend"` + `LENS_FORCE_LOCAL=1`로 9개 endpoint를 TestClient 2회 호출한 결과 9/9 status 200·byte-stable 확인됨. 실행자가 같은 결과를 못 얻으면 환경 문제이니 차단 보고.

## 차단 트리거 (중요)
다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.** 정리해서(무엇이/어디서/왜) 보고.
1. **스냅샷이 매 실행 달라진다**(재실행 diff ≠ 0, 또는 Step 5의 2회 실행이 불일치): 비결정성. 원인 후보를 분류해 보고 — (a) `meta.request_id` uuid가 새는 경우(`request_id.py:9`, 고정 헤더 누락), (b) `date.today()`가 새는 경우(`api_service.py:58`, prices에서 start/end 미고정), (c) dict/리스트 정렬 비결정, (d) float 미세변동. **tolerance로 풀지 / 비결정 원천을 제거할지는 사용자 판단.** 임의로 운영 코드를 고치지 마라.
2. **float 미세 변동으로 재실행이 실패**(특히 `strategies/.../backtest`, `scan` 같은 numpy 집계): rtol/atol을 얼마로 둘지(예 1e-6→1e-4 완화) 또는 비결정 제거 여부를 보고. 임의 완화 금지.
3. **`snaptol` 설치/임포트 불가**(그리고 폴백 도구도 모호): 어떤 도구를 쓸지 사용자에게 확인받고 진행. 임의 선택 금지(Step 1-1).
4. **`backend/data/v1/*.parquet` 누락**으로 endpoint가 503/`not loaded`/`FileNotFoundError`: 데이터가 없으면 박제 불가. 어떤 파일이 없는지(`predictions_line_1d` 등) 명시해 보고. (실측 시점엔 모두 존재.)
5. **AAPL이 특정 슬롯에 없어 404**(예: line/band 캐시에 AAPL 미존재): 고정 ticker를 바꿔야 하므로 보고. (실측 시점 9개 모두 200.)
6. **conftest 환경 변경이 기존 테스트를 깨뜨림**(`pytest backend/tests`에서 기존 통과수 감소, 특히 `test_api.py`의 env clear 테스트): 안전망이 기존 자산을 망가뜨리면 안 됨. 어느 테스트가 왜 깨졌는지 보고.
7. **스냅샷을 통과시키려고 운영 코드(`backend/app/**`) 수정이 필요해 보이는 순간**: 그건 이 CP 범위 밖(동작 변경). 절대 고치지 말고 중단·보고.
8. **TestClient가 import 단계에서 실패**(`ModuleNotFoundError: backend` 또는 `app`): sys.path 이중 경로 보정이 안 된 것. conftest를 고치되, 운영 코드 import 구조는 건드리지 말고, 막히면 보고.

## ADR
완료 후 `docs/adr/0011-characterization-snaptol-for-ml.md` 1장(200~300단어) 작성.
기록할 것: **왜 characterization 스냅샷인가**(분리 리팩토링의 동작 보존 증명, Feathers/Fowler), **왜 snaptol(float tolerance)인가**(ML 수치 출력 — numpy 재계산/버전차 미세변동을 mock 테스트나 정확 일치 비교로는 못 잡거나 거짓 실패가 남 → rtol/atol 필요), **무엇을 박제했는가**(9개 local-parquet endpoint의 정상 200 응답 전체 JSON), **무엇을 박제 안 했는가 + 이유**(Supabase 경로 v1 비활성, ai/admin/에러경로 제외), **비결정성을 어떻게 제거했나**(고정 `X-Request-Id`, prices start/end 고정으로 `date.today()` 회피 — 운영 코드는 무수정). `docs/adr/` 디렉토리는 현재 없으므로 생성. 0011 번호가 충돌하면(기존 ADR 존재 시) 다음 번호로 조정하고 report에 기록.

## 자가 점검 결과 양식
- [Plan v3 정합]: PASS/WARN/FAIL — 사유: ____ (안전망 추가만, fidelity 우선/밴드 본체 등 Plan v3 결정에 영향 없음을 확인했는가)
- [구조 결함]: PASS/WARN/FAIL — 사유: ____ (conftest sys.path 이중 경로/캐시 초기화가 기존 테스트와 충돌 없는가, 스냅샷 baseline 위치가 git에 합리적으로 배치됐는가)
- [모델 영향]: PASS/WARN/FAIL — 사유: ____ (학습·calibration·가중치 무변경, parquet read-only, 모델 출력 수치 무변경을 확인했는가)

## 산출물
- 변경/추가 파일:
  - `backend/requirements-dev.txt` (신규)
  - `backend/tests/conftest.py` (신규)
  - `backend/tests/test_characterization_api.py` (신규)
  - 스냅샷 baseline 디렉토리/파일 (도구가 생성, git 추적) — 예 `backend/tests/__snapshots__/...` 또는 `backend/tests/snapshots/...`
  - `docs/adr/0011-characterization-snaptol-for-ml.md` (신규)
- `docs/cp223_report.md` 작성(요구 / 한일 / 결정 / 후속, 필요한 만큼만). 결정 항목엔 최소: 선택한 스냅샷 도구·버전, 적용 rtol/atol, baseline 파일 경로, 제외 endpoint와 사유, 발견한 비결정 원천 2건(request_id·date.today)을 호출 측에서 회피한 방식.
