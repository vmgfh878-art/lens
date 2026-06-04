# ADR-0025: BE 설정 일원화 — 도메인별 Pydantic Settings

Status: Accepted
Date: 2026-06-04
CP: CP235

## 결정

`backend/app/` 전역에 산발해 있던 11개 `os.environ.get(...)` 직접 호출을 **도메인별 5개 `BaseSettings` 클래스**로 일원화한다. `backend/app/config/settings.py`:

- `DatabaseConfig` (SUPABASE_URL / SUPABASE_KEY / LENS_FORCE_LOCAL)
- `MarketConfig` (MARKET_DATA_PROVIDER / LENS_LOCAL_SNAPSHOT_DIR)
- `CorsConfig` (BACKEND_CORS_ORIGINS / BACKEND_CORS_ORIGIN_REGEX)
- `AdminConfig` (LENS_ADMIN_RELOAD_TOKEN / LENS_ALLOW_LOCAL_ADMIN_RELOAD)
- `CacheConfig` (LENS_EAGER_V1_CACHE + 코드 상수 gzip_minimum_size=512)

## 단일 거대 Settings 대신 5개로 분리한 이유

도메인 경계가 명확하고 caller가 서로 다르다 — main.py(Cors/Cache), db.py(Database), market_repo.py(Market), admin.py(Admin), product_prediction_history_svc.py(Market). 거대 Settings 하나면 caller가 본인 도메인 외 필드까지 의도치 않게 import해 결합이 늘어난다. 5개 분리는 collector `get_settings()` 단일 dataclass 패턴과 다르지만 — collector는 모든 env가 한 파이프라인 안에서 묶이는 반면 app은 여러 라우터/서비스가 서로 독립적으로 일부 도메인만 읽는다.

## 호출 시점 재평가 (import-time 싱글톤 금지)

`get_*_config()` 접근자는 매 호출 새 `BaseSettings` 인스턴스를 만든다 → 매번 `os.environ` 재평가. 이유:

1. `test_api.py:26` `test_ready_health_returns_config_error_when_env_missing`이 `patch.dict(os.environ, {}, clear=True)` 안에서 `/health/ready`를 호출해 `CONFIG_ERROR`를 기대한다. import-time 싱글톤이면 이미 로드된 값이 남아 테스트가 깨진다.
2. `backend/collector/config.py:40 get_settings()` 선례도 매 호출 frozen dataclass를 새로 빌드 — app도 같은 계약.

캐시가 필요한 미래엔 `@lru_cache` + `cache_clear()` 훅 + admin/reload 연결을 추가하면 된다 — 그러나 현재는 **무캐시가 안전한 기본**.

## Supabase 코드는 보류, 환경변수 정의는 유지

CLAUDE.md §7 "Supabase 코드는 보류"에 따라 `db.py`의 supabase 호출 로직은 손대지 않았다. 그러나 `DatabaseConfig`에 SUPABASE_URL/KEY/LENS_FORCE_LOCAL을 정의하고 `.env.example`에도 남겼다 — 사용자가 다시 살릴 예정이라 정의가 끊기면 안 된다. `supabase_is_configured()` / `get_supabase()` / `check_supabase_ready()` 시그니처와 예외 메시지 ("SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")는 1글자도 안 바꿨다.

## 의도적 제외 — `_EPSILON` / `lru_cache(maxsize)` 설정화 안 함

- `feature_svc.py:77 _EPSILON = 1e-9`: 12개 모델 feature 계산식의 분모 보호값. 환경변수로 외부화하면 누군가 값을 바꿨을 때 모델 출력이 조용히 드리프트한다 → 코드 상수 유지.
- `lru_cache(maxsize=...)` 5개 지점: 캐시 크기를 env로 빼면 메모리 거동/스냅샷이 환경마다 달라져 회귀 비교가 흔들린다 → 코드 상수 유지.

`gzip_minimum_size=512`는 `CacheConfig`에 두긴 했으나 환경변수 alias 없이 코드 상수만 — 미래 외부화 여지를 열어두되 현재는 동작 동일.

## 결과

`backend/app/`의 `os.environ.get(...)` 직접 호출 11 → **1** (admin.py debug-state의 `interesting_keys` env 존재 dump만 의도적 잔존). `.env.example`이 코드가 읽는 모든 키와 1:1 동기화 (12개 신규 키 문서화). pytest 회귀 0 (baseline 19 fail / 85 pass → CP235 후 19 fail / 99 pass, +14 = 신규 test_settings). `test_ready_health_returns_config_error_when_env_missing` 통과 보존.
