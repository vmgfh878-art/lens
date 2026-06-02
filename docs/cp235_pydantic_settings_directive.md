# CP235 BE 설정 일원화 — Pydantic Settings (Directive)

> 이 문서는 런북(`docs/cp221_237_refactoring_runbook.md`)이 자동으로 꺼내 실행하는 단일 지시서다.
> 실행자는 이 문서만 읽고 코드를 고치고, 검증하고, 중단 여부를 판단한다. 추측 금지 — 아래 박힌 파일:줄번호를 직접 열어 확인하고 시작한다.
> 작성 시점 기준 코드 스냅샷: 2026-06-02. 실제 줄번호가 어긋나면 그건 선행 CP가 손댄 것이니 먼저 `git log -p` 로 확인하고, 어긋남이 크면 차단 트리거를 적용한다.

---

## CP235 BE 설정 일원화 — Pydantic Settings (Directive)

목적 한 줄: `os.environ.get(...)` 직접 호출이 백엔드 `app/` 전역 10곳에 산발해 있고 GZip/epsilon/lru_cache 매직넘버가 코드에 박혀 있다. 이를 **도메인별 설정 객체**로 일원화하고, 환경변수 정의를 `.env.example` 과 1:1 동기화한다. 동작은 한 비트도 바꾸지 않는다(순수 구조 리팩토링).

---

## 역할 고정

- 모드: **code** (구현 + 자가 점검만 보고).
- 권한: 코드 수정, 로컬 검증(pytest / uvicorn 기동 / 수동 curl)만.
- 금지(이 CP에서 절대 하지 않는다):
  - 새 모델 학습, 새 calibration, 모델 가중치/체크포인트 변경.
  - DB write, Supabase 네트워크 호출(읽기 포함). 로컬 parquet 경로만 사용.
  - 사용자가 직접 수정한 파일을 revert. (의심되면 멈추고 보고.)
  - 동작 변경. 기본값/필수 구분을 **현재 코드와 정확히 동일하게** 옮기는 것이 전부다. 매직넘버 "값"을 바꾸지 않는다(위치만 이동).
- 자가 점검 의무: 매 Step 후 [Plan v3 정합] [구조 결함] [모델 영향] 3축 점검(문서 끝 양식).
- 커밋 메시지: 간결. 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 환경

- 워킹 디렉토리: `C:\Users\user\lens`
- venv: `.venv` (Python 3.10.0, pydantic **2.13.3** 설치됨, torch 2.11.0+cu128). Python 실행은 `.venv\Scripts\python.exe`.
- 백엔드 기동: `scripts\start_demo.ps1` (uvicorn 래핑, 포트 8000 + 프론트 3000) 또는 직접
  `.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (cwd=`backend`, `PYTHONPATH=backend`).
- 프론트: 이 CP는 BE 전용이라 프론트 기동 불필요. `npm run dev` 는 회귀 확인이 필요할 때만.
- 검증용 포트 충돌 피하기: `start_demo.ps1` 은 8000 이 이미 떠 있으면 재기동을 건너뛴다. 깨끗한 기동 검증을 원하면 **임시 포트(예: 8011)** 로 uvicorn 을 띄워 기존 데모 프로세스를 죽이지 않는다.
- `start_demo.ps1` 이 기동 시 주입하는 env (38~44행): `PYTHONPATH`, `BACKEND_CORS_ORIGINS`, `MARKET_DATA_PROVIDER=yfinance`, `LENS_USE_LOCAL_SNAPSHOTS=1`, `LENS_LOCAL_SNAPSHOT_DIR=<root>\data\parquet`. 이 동작이 리팩토링 후에도 동일해야 한다.

> 선행 설치(차단 가능 항목): 아래 "선행 의존"의 pydantic-settings 미설치 문제를 **Step 1 이전에** 해결해야 한다.

---

## 진단 (근거)

조사 출처: 아래는 전부 `C:\Users\user\lens` 의 실제 파일을 Read/Grep 으로 확인한 것이다(2026-06-02).

### A. `os.environ` 직접 호출이 `app/` 전역에 산발 (총 10개 지점 / 5개 키 도메인)

`Grep "os\.environ|os\.getenv"` (`backend\app` 범위) 결과:

| # | 파일:줄 | 환경변수 | 기본값 | 도메인 |
|---|---------|----------|--------|--------|
| 1 | `backend\app\db.py:20` | `LENS_FORCE_LOCAL` | `"0"` | Database |
| 2 | `backend\app\db.py:22` | `SUPABASE_URL`, `SUPABASE_KEY` | (없음/필수) | Database |
| 3 | `backend\app\db.py:30-31` | `SUPABASE_URL`, `SUPABASE_KEY` | (없음) | Database |
| 4 | `backend\app\db.py:46-47` | `SUPABASE_URL`, `SUPABASE_KEY` | (없음) | Database |
| 5 | `backend\app\main.py:29` | `BACKEND_CORS_ORIGINS` | 4개 origin CSV(23~28행) | Cors |
| 6 | `backend\app\main.py:39` | `BACKEND_CORS_ORIGIN_REGEX` | `^https://lens(?:-[a-z0-9-]+)?\.vercel\.app$` | Cors |
| 7 | `backend\app\main.py:74` | `LENS_EAGER_V1_CACHE` | `"0"` | Cache |
| 8 | `backend\app\repositories\market_repo.py:61` | `MARKET_DATA_PROVIDER` | `"yfinance"` | Market |
| 9 | `backend\app\routers\v1\admin.py:28` | `LENS_ADMIN_RELOAD_TOKEN` | `""` | Admin |
| 10 | `backend\app\routers\v1\admin.py:37` | `LENS_ALLOW_LOCAL_ADMIN_RELOAD` | `"0"` | Admin |
| 11 | `backend\app\services\product_prediction_history_svc.py:49` | `LENS_LOCAL_SNAPSHOT_DIR` | (override only) | Market |

추가로 `admin.py:96-109` 의 `interesting_keys` 리스트는 **debug 출력 전용**(값 노출 X, 존재 여부만)이다. 이건 설정 객체로 옮길 대상이 아니라 "어떤 키가 존재하나"를 그대로 dump 하는 진단 코드이므로 **그대로 둔다**(범위 제외, 아래 명시).

또한 `backend\collector\repositories\local_snapshots.py:15,19,22,26` 가 `LENS_LOCAL_SNAPSHOT_DIR`, `LENS_DATA_BACKEND`, `LENS_REQUIRE_LOCAL_SNAPSHOTS`, `LENS_USE_LOCAL_SNAPSHOTS` 를 읽는다. 이건 **collector 도메인**이라 이 CP(app 설정) 범위 밖이다(아래 범위 제외). 단 `.env.example` 동기화 대상에는 포함한다.

### B. 매직넘버 박힘

- `backend\app\main.py:55`: `app.add_middleware(GZipMiddleware, minimum_size=512)` — GZip 최소 압축 바이트 `512`.
- `backend\app\services\feature_svc.py:77`: `_EPSILON = 1e-9` — RSI/MACD/ATR/BB/ratio 계산 분모 보호값. **154~365행에서 12회 사용**(모델 feature 계산 경로). → **모델 영향 주의 대상.** 아래 범위/차단에서 별도 취급.
- `lru_cache(maxsize=...)` 산발:
  - `backend\app\services\strategy_backtest_svc.py:125` (`maxsize=1`)
  - `backend\app\services\strategy_backtest_svc.py:226` (`maxsize=1`)
  - `backend\app\services\strategy_backtest_svc.py:458` (`maxsize=16`)
  - `backend\app\services\product_prediction_history_svc.py:82` (`maxsize=2`)
  - `backend\app\repositories\ai_repo.py:19` (`maxsize=1`)

### C. `.env.example` 가 실제 코드와 어긋나 있음

`.env.example` (현재 18줄) 에 정의된 키: `NEXT_PUBLIC_BACKEND_URL`, `BACKEND_CORS_ORIGINS`, `SUPABASE_URL`, `SUPABASE_KEY`, `FRED_API_KEY`, `FMP_API_KEY`.
**누락**되어 있고 코드는 읽는 키: `BACKEND_CORS_ORIGIN_REGEX`, `MARKET_DATA_PROVIDER`, `MARKET_DATA_FALLBACK_PROVIDER`, `LENS_FORCE_LOCAL`, `LENS_EAGER_V1_CACHE`, `LENS_ADMIN_RELOAD_TOKEN`, `LENS_ALLOW_LOCAL_ADMIN_RELOAD`, `LENS_LOCAL_SNAPSHOT_DIR`, `LENS_USE_LOCAL_SNAPSHOTS`, `LENS_DATA_BACKEND`, `LENS_REQUIRE_LOCAL_SNAPSHOTS`, `EODHD_API_KEY`. 이 불일치 해소가 Step 3.

### D. 선례: collector 도메인은 이미 설정 일원화 패턴이 있다 (재사용/정합의 기준)

`backend\collector\config.py:40` `def get_settings() -> CollectorSettings:` 는 **frozen dataclass 를 매 호출마다 os.environ 에서 새로 빌드**한다(40~69행). 캐시하지 않는다. 그리고 테스트가 이 "호출 시마다 재읽기" 동작에 의존한다:

- `backend\tests\test_market_data_providers.py:219-235`:
  ```python
  with patch.dict(os.environ, {"MARKET_DATA_PROVIDER": "yfinance"}, clear=True):
      settings = get_settings()              # patch 컨텍스트 안에서 호출 → 재읽기 전제
  self.assertEqual(settings.market_data_provider, "yfinance")
  self.assertEqual(settings.market_data_fallback_provider, "eodhd")
  ```

→ **이 CP의 app 설정도 같은 계약을 따른다: 설정 접근자는 호출 시점에 env 를 재평가한다(import 시 1회 고정 금지).** 근거는 다음 E.

### E. import-time 싱글톤이면 기존 테스트가 깨진다 (가장 중요한 제약)

`backend\tests\test_api.py:26-33`:
```python
def test_ready_health_returns_config_error_when_env_missing(self):
    with patch.dict(os.environ, {}, clear=True):     # env 전부 비움
        response = self.client.get("/api/v1/health/ready")
    self.assertEqual(response.status_code, 500)
    self.assertEqual(body["error"]["code"], "CONFIG_ERROR")
```
`/health/ready` → `db.check_supabase_ready()` (`db.py:44`) → `os.environ.get("SUPABASE_URL")` 를 **요청 처리 시점에** 읽기 때문에 통과한다. 만약 Settings 를 모듈 import 시 1회 인스턴스화해 캐시하면, 테스트가 env 를 비워도 이미 로드된 값이 남아 CONFIG_ERROR 가 안 나고 테스트가 깨진다.
→ **설정은 호출 시점 재평가(또는 cache_clear 훅 + 테스트 reset) 여야 한다.** (설계 결정 → ADR 기록.)

---

## 선행 의존

- **CP223 (백엔드 characterization 스냅샷) 그린**이 시작 전제다. CP223 이 그린이 아니면 이 CP를 시작하지 않는다. 안전망 없이 구조를 건드리지 않는다.
  - 확인: CP223 산출물(백엔드 응답 스냅샷 테스트)이 존재하고 전부 통과하는지 먼저 실행한다. 위치를 모르면 `Grep "snapshot"` (backend\tests 범위) 또는 CP223 report(`docs/cp223*`)로 식별한다.
  - **CP223 스냅샷 테스트를 식별할 수 없으면** → 차단 트리거 적용(아래). 안전망 부재 상태에서 진행 금지.
- **pydantic-settings 패키지 미설치** (차단성 선행 작업):
  - 현재 환경에서 `.venv\Scripts\python.exe -c "import pydantic_settings"` → `ModuleNotFoundError`. pydantic v2 에서는 `BaseSettings` 가 core 에서 분리되어 `pydantic-settings` 패키지가 필요하다.
  - 따라서 Step 1 이전에 `backend\requirements.txt` 에 `pydantic-settings` 핀을 추가하고 설치해야 한다. (구체 절차는 Step 0.)

---

## 범위

### 포함
- `app/` 도메인의 `os.environ` 직접 호출 10개 지점(진단 표 #1~#11, 단 admin debug dump 제외)을 **도메인별 설정 객체 + 접근자**로 이전.
- 도메인 분리: `DatabaseConfig` / `MarketConfig` / `CorsConfig` / `AdminConfig` / `CacheConfig`.
- 매직넘버 중 **값 변경 위험이 없는 것**의 설정화: GZip `minimum_size=512`(`CacheConfig` 또는 신설 `ServerConfig`), `LENS_EAGER_V1_CACHE` 게이트(`CacheConfig`).
- `.env.example` 를 실제 코드가 읽는 모든 키와 동기화(주석으로 기본값/필수 표기).
- 신설 설정 모듈 단위 테스트(env override 시 값 반영, 미설정 시 기본값) 추가.

### 제외 (이번에 건드리지 않음)
- **Supabase 호출 코드 자체**는 보류 상태다. 그러나 **환경변수 정의(`SUPABASE_URL`/`SUPABASE_KEY`/`LENS_FORCE_LOCAL`)는 유지**한다 — 다시 살릴 것이므로 `DatabaseConfig` 에 정의하고 `.env.example` 에도 남긴다. db.py 의 호출 위치만 설정 참조로 교체하고, Supabase 네트워크 동작/조건은 그대로.
- **`feature_svc.py:77` `_EPSILON = 1e-9`** 의 설정화는 **이번 범위에서 제외**(WARN). 이유: 12개 모델 feature 계산식의 분모 보호값이라 환경변수로 외부화하면 누군가 값을 바꿨을 때 모델 출력이 조용히 드리프트한다. 모듈 상수로 둔다. (꼭 옮겨야 하면 값 동결 + 별도 CP + characterization 으로.)
- **`lru_cache(maxsize=...)` 값**의 설정화는 **이번 범위에서 제외**(WARN). 캐시 크기를 런타임 env 로 빼면 메모리 거동/스냅샷이 환경마다 달라져 회귀 비교가 흔들린다. 매직넘버를 굳이 옮기지 말고 코드 상수로 둔다. (스펙에 "설정화" 언급은 있으나, 동작 보존 원칙이 우선 — 점검 결과 양식에 사유 기록.)
- **collector 도메인**(`backend\collector\config.py`, `local_snapshots.py`)의 env 읽기는 이 CP 범위 밖(이미 `get_settings()` 패턴 존재). 단 `.env.example` 동기화에는 키를 포함한다.
- `admin.py:96-109` debug-state 의 `interesting_keys` dump 는 진단 목적이라 그대로 둔다.
- DB 스크립트(`backend\db\scripts\*`)의 `os.environ` 은 운영 스크립트라 범위 밖.

---

## Sub-step (Strangler Fig, 작은 단위)

원칙: 각 Step = **옛 코드 옆에 새 설정 공존 → caller 한 곳씩 이전 → 옛 직접호출 제거**. 한 Step = 한 commit = 한 revert 단위. 추출 순서는 부작용 없는 것부터(Cors/Cache → Market → Admin → Database).

### Step 0 — pydantic-settings 설치 (인프라, 코드 변경 최소)
1. `backend\requirements.txt` 에 한 줄 추가: `pydantic-settings==2.x` (설치 후 실제 받은 버전으로 핀 고정; pydantic 2.13.3 과 호환되는 최신 2.x).
2. `.venv\Scripts\python.exe -m pip install "pydantic-settings"` 실행 → 받은 정확한 버전을 requirements 핀에 반영.
3. 검증: `.venv\Scripts\python.exe -c "import pydantic_settings; print(pydantic_settings.__version__)"` 가 버전 출력.
4. commit: `chore(be): add pydantic-settings dependency`.
- 이 Step 은 코드 동작 변경 없음(의존성만). pytest 회귀 0 확인 후 커밋.

### Step 1 — 설정 모듈 신설 (옛 코드 그대로, 새 코드 공존만)
1. 신설 `backend\app\config\__init__.py` 와 `backend\app\config\settings.py` (디렉토리 신설). 도메인별 클래스 정의:
   - `DatabaseConfig`: `supabase_url: str | None`, `supabase_key: str | None`, `force_local: bool`.
   - `MarketConfig`: `market_data_provider: str = "yfinance"`, `local_snapshot_dir: str | None`.
   - `CorsConfig`: `origins: list[str]`(기본값 = main.py 23~28행 4개 origin), `origin_regex: str`(기본 = main.py 41행 regex).
   - `AdminConfig`: `reload_token: str = ""`, `allow_local_reload: bool = False`.
   - `CacheConfig`: `eager_v1_cache: bool = False`, `gzip_minimum_size: int = 512`.
2. **단일 거대 Settings 금지.** 각 클래스는 독립 `BaseSettings`(또는 동등 패턴). 환경변수 이름 매핑은 각 필드의 alias/`env=` 로 **현재 키 이름 그대로**(예: `force_local` ↔ `LENS_FORCE_LOCAL`, `origins` ↔ `BACKEND_CORS_ORIGINS`).
3. **접근 패턴(중요)**: 각 도메인에 `get_database_config()`, `get_market_config()`, `get_cors_config()`, `get_admin_config()`, `get_cache_config()` 접근자를 둔다. 접근자는 **호출 시점에 새 인스턴스를 만들어 env 를 재평가**한다(진단 E 의 `test_api.py:26` `clear=True` 통과 보장). 만약 성능상 캐시가 필요하면 `@lru_cache` + `cache_clear()` 훅을 같이 제공하고, 테스트 reset 픽스처와 admin/reload 에 훅을 연결한다 — 그러나 **기본 권장은 무캐시(매 호출 재읽기), collector `get_settings()` 와 정합**.
4. CSV 파싱 보존: `CorsConfig.origins` 는 main.py 의 `_parse_cors_origins()`(19~30행)와 **동일 규칙**(콤마 split + strip + 빈 값 제거)이어야 한다. 빈 문자열/공백 처리 동작을 그대로 옮긴다.
5. bool 파싱 보존: `LENS_FORCE_LOCAL`/`LENS_ALLOW_LOCAL_ADMIN_RELOAD` 는 현재 `.strip().lower() in {"1","true","yes"}` (db.py:20, admin.py:37). `LENS_EAGER_V1_CACHE` 는 현재 `!= "1"` 게이트(main.py:74). 이 **정확한 truthy 규칙**을 각각 그대로 재현(달라지면 동작 변경 = 차단).
6. 이 시점에 caller 는 아직 안 바꾼다(공존만). 신설 모듈 단위 테스트 작성: 미설정 기본값 / env override / CSV 다중값 / truthy 변형("1","true","yes","0").
7. 검증: 신설 테스트 통과 + 전체 pytest 회귀 0 + import 정상.
8. commit: `refactor(be): add domain Settings (db/market/cors/admin/cache) alongside env reads`.

### Step 2 — caller 를 설정 참조로 이전 (도메인 1개씩, 공존→이전→옛 제거)
각 항목은 **개별 커밋** 권장(최소한 도메인 경계로 분리). 순서: Cors → Cache → Market → Admin → Database.

2a. **Cors** (`main.py:19-30, 39-42, 49-50, 55`):
   - `_parse_cors_origins()` 본문을 `get_cors_config().origins` 호출로 대체(또는 함수를 얇은 위임으로). `_CORS_ORIGIN_REGEX`(39행) → `get_cors_config().origin_regex`. GZip `minimum_size=512`(55행) → `get_cache_config().gzip_minimum_size`.
   - 옛 `os.environ.get("BACKEND_CORS_ORIGINS"...)` / `os.environ.get("BACKEND_CORS_ORIGIN_REGEX"...)` 직접 호출 제거.
   - 주의: `add_middleware` 는 import 시 1회 실행된다. 여기서는 "기동 시점 1회 읽기"가 맞다(요청별 재평가 불필요). 단 설정 접근자 자체는 무캐시여도 무방(기동 시 1회 호출).
2b. **Cache** (`main.py:74`): `os.environ.get("LENS_EAGER_V1_CACHE","0") != "1"` → `not get_cache_config().eager_v1_cache`. 로그 문구(75행)는 그대로 유지.
2c. **Market** (`market_repo.py:61`): `os.environ.get("MARKET_DATA_PROVIDER","yfinance")` → `get_market_config().market_data_provider`. `resolve_market_data_provider()`(49행)의 시그니처/정규화(`_normalize_provider_name`)는 **불변**. 그리고 `product_prediction_history_svc.py:49` 의 `LENS_LOCAL_SNAPSHOT_DIR` override → `get_market_config().local_snapshot_dir`. 단 `_snapshot_dir()`(45~54행)의 우선순위 로직(V1 우선 → override → LEGACY)을 **그대로** 유지하고, override 값만 설정 경유.
2d. **Admin** (`admin.py:28, 37`): `LENS_ADMIN_RELOAD_TOKEN` → `get_admin_config().reload_token`, `LENS_ALLOW_LOCAL_ADMIN_RELOAD` → `get_admin_config().allow_local_reload`. `_require_reload_allowed()`(27~44행)의 분기/예외 메시지 **불변**. (admin debug dump 96~109행은 건드리지 않음.)
2e. **Database** (`db.py:20, 22, 30-31, 46-47`): 네 곳의 `os.environ.get("SUPABASE_URL"/"SUPABASE_KEY")` 와 `LENS_FORCE_LOCAL` 를 `get_database_config()` 경유로. **반드시 호출 시점 재평가**(진단 E). `supabase_is_configured()`/`get_supabase()`/`check_supabase_ready()` 의 시그니처·예외(`ConfigError` 메시지 포함)·분기 **불변**. Supabase 네트워크 호출 로직은 손대지 않는다.
   - 각 caller 이전 직후 해당 도메인 `os.environ` import 의존이 사라졌는지 확인. `main.py`/`db.py`/`admin.py`/`market_repo.py`/`product_prediction_history_svc.py` 의 `import os` 가 다른 용도로 남아있는지 점검 후, 안 쓰면 제거.
   - 각 도메인 이전 후 검증: 관련 pytest(아래 "검증"의 타깃 테스트) 통과 + CP223 스냅샷 diff 0.
   - commit(도메인별): 예) `refactor(be): route CORS/GZip config through CorsConfig`, ... `refactor(be): route Supabase env through DatabaseConfig`.

### Step 3 — `.env.example` 동기화
1. `.env.example` 에 코드가 읽는 모든 키를 추가하고 주석으로 **기본값 / 필수 여부**를 표기. 누락 키(진단 C): `BACKEND_CORS_ORIGIN_REGEX`, `MARKET_DATA_PROVIDER`, `MARKET_DATA_FALLBACK_PROVIDER`, `LENS_FORCE_LOCAL`, `LENS_EAGER_V1_CACHE`, `LENS_ADMIN_RELOAD_TOKEN`, `LENS_ALLOW_LOCAL_ADMIN_RELOAD`, `LENS_LOCAL_SNAPSHOT_DIR`, `LENS_USE_LOCAL_SNAPSHOTS`, `LENS_DATA_BACKEND`, `LENS_REQUIRE_LOCAL_SNAPSHOTS`, `EODHD_API_KEY`.
2. 기존 키(`NEXT_PUBLIC_BACKEND_URL`, `BACKEND_CORS_ORIGINS`, `SUPABASE_URL`, `SUPABASE_KEY`, `FRED_API_KEY`, `FMP_API_KEY`)는 유지.
3. 기본값은 **코드의 실제 기본값과 동일**해야 한다(예: `MARKET_DATA_PROVIDER` 의 app 기본은 `yfinance`, collector 기본은 `yahoo` — 둘 다 다르므로 주석으로 구분 표기). 불일치 발견 시 차단 트리거.
4. 검증: 정적 점검(아래 스크립트) — 코드가 읽는 키 집합 ⊆ `.env.example` 정의 키 집합.
5. commit: `docs(be): sync .env.example with settings keys`.

### Step 4 — 기동 검증 + report/ADR
1. 임시 포트로 uvicorn 기동(아래 검증) → `/`, `/api/v1/health/live`, `/api/v1/health/ready` 응답 확인.
2. `docs/adr/0025-pydantic-settings.md` 작성, `docs/cp235_report.md` 작성.
3. commit: `docs(be): CP235 report + ADR 0025`.

---

## 인터페이스 보존

다음을 **바꾸지 않는다**. 바꿔야만 하면 호출자 영향 분석 + 즉시 차단 보고.

- 함수 시그니처 불변:
  - `db.supabase_is_configured() -> bool`, `db.get_supabase() -> Client`, `db.reset_supabase_client() -> None`, `db.check_supabase_ready() -> dict[str, bool]`.
  - `market_repo.resolve_market_data_provider(market_data_provider, source, *, warn_if_default) -> str`, `fetch_price_rows(...)`, `fetch_indicator_rows(...)`, `fetch_stocks(...)`.
  - `admin._require_reload_allowed(request, token) -> None`, `admin.reload_v1_predictions(...)`, `admin.debug_state(...)`.
  - `product_prediction_history_svc.get_product_prediction_history_data(...)`, `_snapshot_dir() -> Path`.
  - `main._parse_cors_origins() -> list[str]` (남겨두면 동일 반환, 본문만 위임).
- API 응답 schema 불변: `/`, `/api/v1/health/ready`, `/api/v1/admin/reload`, `/api/v1/admin/debug-state` 의 응답 키/값·에러 코드(`CONFIG_ERROR`, `VALIDATION_ERROR`, `INTERNAL_ERROR`)·status code. 특히 `debug-state` 의 `interesting_env` dict 키 목록(admin.py:96-109)은 그대로.
- 예외 메시지 문자열 불변(테스트가 substring 매칭): `db.py:33,55` 의 `"SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다."`, `admin.py:33,43` 의 reload 거부 메시지.
- 설정 접근 계약: 설정은 **호출 시점 env 재평가**(진단 E). import-time 싱글톤 캐시로 이 계약을 깨면 인터페이스 위반으로 간주.

---

## 성공 기준 (측정 가능)

| 항목 | 기준 |
|------|------|
| `app/` 내 `os.environ` 직접 호출 | 11개 지점 → **0** (단 admin.py debug dump `interesting_keys` 는 의도적 잔존, 별도 카운트) |
| `.env.example` 동기화 | 코드가 읽는 키 집합 ⊆ `.env.example` 정의 키 (정적 점검 PASS) |
| pytest 회귀 | 기존 테스트 전부 통과, **회귀 0** (특히 `test_api.py`, `test_services.py`, `test_market_data_providers.py`, `test_cp209_admin_rebuild_contracts.py`, `test_product_prediction_history_api.py`) |
| 신설 설정 단위 테스트 | 도메인 5개에 대해 기본값/override/CSV/truthy 케이스 추가, 전부 통과 |
| CP223 characterization 스냅샷 diff | **0** (동작 불변 증명) |
| 기동 | 임시 포트 uvicorn 정상 기동, `/health/ready` 가 env 유무에 따라 기존과 동일 분기 |
| mypy 신규 에러 | **0 추가** (저장소에 mypy 설정은 없음 → best-effort: `mypy backend/app/config` 에서 신규 에러 0) |
| 예상 시간 | 3~4시간 |

> tsc/screenshot 항목은 BE 전용 CP라 **해당 없음(생략)**.

---

## 검증

PowerShell 기준(Windows). cwd 는 `C:\Users\user\lens`.

1. **pydantic-settings 설치 확인** (Step 0 후):
   ```powershell
   .venv\Scripts\python.exe -c "import pydantic_settings; print(pydantic_settings.__version__)"
   ```
   기대: 버전 문자열 출력(에러 없음).

2. **전체 pytest 회귀** (각 Step 후):
   ```powershell
   $env:PYTHONPATH = "$PWD\backend"
   .venv\Scripts\python.exe -m pytest backend\tests -q
   ```
   기대: 이전(베이스라인)과 동일 pass 수, 신규 fail 0. **베이스라인을 Step 0 전에 한 번 떠두고 비교**한다.

3. **타깃 테스트 집중 실행** (caller 이전 직후):
   ```powershell
   .venv\Scripts\python.exe -m pytest backend\tests\test_api.py backend\tests\test_services.py backend\tests\test_market_data_providers.py backend\tests\test_cp209_admin_rebuild_contracts.py backend\tests\test_product_prediction_history_api.py -q
   ```
   기대: 전부 통과. 특히 `test_ready_health_returns_config_error_when_env_missing`(clear=True) 통과 = 호출시점 재평가 계약 보존 증명.

4. **CP223 스냅샷** (각 Step 후):
   - CP223 스냅샷 테스트 파일을 식별해 실행(예시): 
   ```powershell
   .venv\Scripts\python.exe -m pytest backend\tests -q -k "snapshot or characterization or cp223"
   ```
   기대: diff 0 / 전부 통과. (식별 불가 시 차단.)

5. **`.env.example` 정적 동기화 점검** (Step 3 후): 코드가 읽는 키가 모두 `.env.example` 에 있는지 확인. 임시 스크립트로 `backend\app` 의 `os.environ.get("KEY"...)` 키를 추출해 `.env.example` 키와 차집합이 비었는지 본다. (차집합 비어 있음 = PASS.)

6. **기동 검증** (Step 4): 데모 포트(8000)를 죽이지 않도록 임시 포트 사용:
   ```powershell
   $env:PYTHONPATH = "$PWD\backend"
   $env:LENS_USE_LOCAL_SNAPSHOTS = "1"
   $env:LENS_LOCAL_SNAPSHOT_DIR = "$PWD\data\parquet"
   $env:MARKET_DATA_PROVIDER = "yfinance"
   .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8011
   ```
   별 셸에서:
   ```powershell
   Invoke-WebRequest http://127.0.0.1:8011/api/v1/health/live -UseBasicParsing
   Invoke-WebRequest http://127.0.0.1:8011/api/v1/health/ready -UseBasicParsing
   ```
   기대: live=200, ready 는 Supabase env 미설정 시 기존과 동일하게 500/CONFIG_ERROR(또는 설정 시 정상). 확인 후 Ctrl+C 로 임시 서버 종료.

---

## 차단 트리거 (중요)

다음 상황이면 **즉시 중단하고 사용자에게 보고한다. 그냥 넘어가기 절대 금지.**

1. **CP223 스냅샷 테스트를 식별/실행할 수 없다.** → 안전망 부재. 구조 변경 시작 금지. 보고 후 대기.
2. **CP223 스냅샷 diff 가 발생**(어느 Step에서든). = 동작이 바뀌었다는 증거. 해당 Step revert, 원인 정리해 보고.
3. **pydantic-settings 설치 실패 / pydantic 2.13.3 과 버전 충돌.** = 기반 의존성 문제. 보고(임의 다운그레이드·업그레이드 금지).
4. **env 미설정/기본값 오류로 uvicorn 기동 실패.** 특히 신설 `BaseSettings` 에 필수 필드(기본값 없음)를 잘못 지정해 import 시점에 `ValidationError` 로 부팅이 죽는 경우 — `SUPABASE_URL`/`KEY` 는 **현재 None 허용(미설정 OK)** 이므로 필수로 만들면 안 된다. 기동 실패 시 즉시 멈추고 어떤 필드가 필수로 잘못 잡혔는지 보고.
5. **`test_ready_health_returns_config_error_when_env_missing`(test_api.py:26) 또는 provider 테스트(test_market_data_providers.py:219-235)가 깨진다.** = import-time 싱글톤으로 호출시점 재평가 계약을 위반. revert 후 무캐시 접근자로 교정. 그래도 안 되면 보고.
6. **`.env.example` 기본값이 실제 코드 기본값과 불일치**(예: `MARKET_DATA_PROVIDER` app=yfinance vs collector=yahoo 를 혼동, truthy 규칙 변형). 발견 시 보고하고 코드 기준으로 맞춘다 — 추측으로 한쪽을 바꾸지 않는다.
7. **기존 테스트 다수(2개 초과) 동시 실패** 또는 베이스라인 대비 pass 수 감소. 광범위 회귀 신호 → 멈추고 보고.
8. **인터페이스 보존 위반이 불가피**(시그니처/응답 schema/예외 문자열을 바꿔야만 동작이 유지됨). 임의 변경 금지, 영향 분석 붙여 보고.
9. **사용자 직접수정 파일 충돌**: 옮기려는 줄이 진단 표의 줄번호와 다르거나 최근 사용자 커밋이 같은 영역을 건드린 흔적이 있으면, revert 위험. 멈추고 보고.
10. **모델 영향 의심**: 실수로 `feature_svc.py:77 _EPSILON` 값이나 `lru_cache(maxsize)` 값을 바꾸는 변경이 들어가면 즉시 되돌리고 보고(이번 범위 제외 항목).

---

## ADR

완료 후 `docs/adr/0025-pydantic-settings.md` 1장(200~300단어) 작성.
- `docs/adr/` 디렉토리가 **현재 존재하지 않으므로** 디렉토리부터 생성한다.
- 기록할 결정: (1) 단일 거대 Settings 대신 **도메인별 설정 클래스 5개**(Database/Market/Cors/Admin/Cache)로 분리한 이유. (2) 설정 접근을 **import-time 싱글톤이 아니라 호출 시점 재평가 접근자**로 둔 이유 — `test_api.py:26` 의 `patch.dict(clear=True)` 계약과 collector `get_settings()` 선례 정합. (3) Supabase 코드는 보류지만 **환경변수 정의는 유지**(부활 대비)한 결정. (4) `_EPSILON`/`lru_cache maxsize` 를 **의도적으로 설정화하지 않은** 이유(모델/메모리 거동 드리프트 방지). 대안(거대 Settings, 싱글톤 캐시)과 트레이드오프, 결과를 적는다.

---

## 자가 점검 결과 양식

각 Step 완료 및 최종에 아래를 채운다(빈칸 금지).

- [Plan v3 정합] PASS / WARN / FAIL — 사유: ____ (밴드 본체·fidelity 우선·EODHD 유지·α/β·backtest cost 와 충돌 없는지. 설정 리팩토링이라 통상 PASS 예상.)
- [구조 결함] PASS / WARN / FAIL — 사유: ____ (도메인 경계 적절성, 순환 import 없음, os.environ 잔존 0, 접근자 무캐시 계약 준수.)
- [모델 영향] PASS / WARN / FAIL — 사유: ____ (`_EPSILON`/feature 계산/lru_cache 값 불변 확인. 값 변경 0이면 PASS.)

---

## 산출물

- 변경 파일(예상):
  - `backend\requirements.txt` (pydantic-settings 핀 추가)
  - `backend\app\config\__init__.py` (신설), `backend\app\config\settings.py` (신설)
  - `backend\app\main.py` (Cors/GZip/Cache caller 이전)
  - `backend\app\db.py` (Database caller 이전)
  - `backend\app\repositories\market_repo.py` (Market caller 이전)
  - `backend\app\routers\v1\admin.py` (Admin caller 이전)
  - `backend\app\services\product_prediction_history_svc.py` (snapshot dir override 이전)
  - `backend\tests\test_settings.py` (신설 단위 테스트)
  - `.env.example` (키 동기화)
  - `docs\adr\0025-pydantic-settings.md` (신설)
  - `docs\cp235_report.md` (신설)
- `docs\cp235_report.md`: 요구 / 한 일(Step별 commit 해시) / 결정(ADR 요약) / 후속(보류한 `_EPSILON`·lru_cache 설정화, collector 도메인 통합 여지) — 필요한 만큼만, 간결하게.
