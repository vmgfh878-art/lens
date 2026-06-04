# CP235 보고서 — BE 설정 일원화 (Pydantic Settings)

**완료일**: 2026-06-04
**선행 의존**: CP223 (백엔드 characterization 스냅샷 — 일부 pre-existing fail 존재. 추가 fail 0 가드)
**커밋 범위**: `c30e6dd` (CP234 끝) → `9a07f12` (Step 0) → `12182c4` (Step 1) → `7053101` (Step 2) → `9fdfa32` (Step 3) → 본 commit (Step 4)
**마지막 그린**: `9fdfa32`

## 요구

`backend/app/` 11개 `os.environ.get(...)` 직접 호출을 도메인별 `BaseSettings`로 일원화 + `.env.example`을 코드가 읽는 모든 키와 동기화. 동작 1bit 변경 0.

## 한 일 (Step별)

| Step | 내용 | commit |
|---|---|---|
| 0 | pydantic-settings==2.14.1 핀 추가 | `9a07f12` |
| 1 | 신설 `backend/app/config/{__init__,settings}.py` — 5 도메인 BaseSettings + `get_*_config()` 접근자 + `test_settings.py` 14 cases | `12182c4` |
| 2 | 5 파일 caller 이전 (main.py / db.py / admin.py / market_repo.py / product_prediction_history_svc.py) | `7053101` |
| 3 | `.env.example` 동기화 (12개 신규 키 문서화) | `9fdfa32` |
| 4 | 기동 검증 + ADR + report | 본 commit |

## 결정 (ADR-0025 요약)

- **단일 거대 Settings 대신 5개**: 도메인 경계 명확 + caller 독립성 보존.
- **호출 시점 재평가**: 매 `get_*_config()` 호출마다 새 BaseSettings 인스턴스 → `os.environ` 재평가. `test_api.py:26 patch.dict(clear=True)` 계약 + `collector/config.py:40 get_settings()` 정합.
- **Supabase 코드는 보류, 환경변수는 유지**: `DatabaseConfig`에 SUPABASE_URL/KEY/LENS_FORCE_LOCAL 정의 유지, `db.py` 호출 로직은 손대지 않음 (CLAUDE.md §7).
- **`_EPSILON`/`lru_cache maxsize` 설정화 제외**: 모델 출력 드리프트 + 메모리 거동 회귀 위험. 코드 상수 유지.
- **admin debug dump `interesting_keys` 잔존**: env 존재 여부만 dump하는 진단 코드 → 의도적 보존.

## 핵심 보존 체크리스트

| 항목 | 확인 |
|---|---|
| `app/` 내 `os.environ` 직접 호출 11 → 1 (admin debug dump만 잔존) | OK |
| `.env.example` ⊇ 코드가 읽는 모든 키 | OK (12개 신규: BACKEND_CORS_ORIGIN_REGEX / LENS_FORCE_LOCAL / MARKET_DATA_PROVIDER / MARKET_DATA_FALLBACK_PROVIDER / LENS_USE_LOCAL_SNAPSHOTS / LENS_LOCAL_SNAPSHOT_DIR / LENS_DATA_BACKEND / LENS_REQUIRE_LOCAL_SNAPSHOTS / LENS_EAGER_V1_CACHE / LENS_ADMIN_RELOAD_TOKEN / LENS_ALLOW_LOCAL_ADMIN_RELOAD / EODHD_API_KEY) |
| 함수 signature 보존 (`supabase_is_configured` / `get_supabase` / `check_supabase_ready` / `resolve_market_data_provider` / `_require_reload_allowed` / `_snapshot_dir` / `_parse_cors_origins`) | OK |
| 예외 메시지 문자열 보존 (`SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.` / `admin reload token이 올바르지 않습니다.` 등) | OK |
| truthy 파싱 규칙 보존 (`.strip().lower() in {"1","true","yes"}` for force_local/allow_local; `== "1"` strict for eager_v1_cache) | OK (test_settings로 박제) |
| CSV split 규칙 보존 (split + strip + 빈 값 제거) | OK |
| 호출 시점 재평가 계약 (`test_ready_health_returns_config_error_when_env_missing` 통과) | OK (test_api.py + test_settings.py `test_clear_env_repeated_call_reevaluates`) |
| `_EPSILON` / `lru_cache(maxsize)` 값 무변경 | OK (의도적 제외) |

## 새 테스트 결과 (test_settings.py)

14 passing — DatabaseConfigTestCase(4) / MarketConfigTestCase(2) / CorsConfigTestCase(3) / AdminConfigTestCase(3) / CacheConfigTestCase(2). 기본값/override/CSV/truthy 변형/매 호출 재평가 박제.

## pytest 회귀 (Step 0 baseline 대비)

| 시점 | passed | failed | skipped | notes |
|---|---|---|---|---|
| baseline (Step 0 직전) | 85 | 19 | 2 | pre-existing fail (CP223 characterization 일부 + market provider + product history) |
| Step 1 후 | 99 (+14) | 19 (=) | 2 | +14 = 신규 test_settings |
| Step 2 후 | 99 | 19 | 2 | caller 이전 회귀 0 |

회귀 0. baseline pre-existing fail은 CP235 책임 아님 (CP227+ 변경 후 잔여) — 후속 CP에서 별도 cleanup.

> `test_services.py`는 collection error로 baseline부터 ignore (--ignore 옵션). CP235와 무관.

## 자가 점검 결과

- **[Plan v3 정합]** PASS — 사유: BE 설정 일원화 only. 밴드 본체·fidelity·EODHD 유지·α/β·backtest cost와 무관. `_EPSILON`/feature 계산 코드 무변경.
- **[구조 결함]** PASS — 사유: 도메인 경계 명확 (Database/Market/Cors/Admin/Cache). 순환 import 없음 (config 모듈은 leaf). os.environ 잔존 = admin debug dump 1건 (의도적). 호출 시점 재평가 계약 준수 (test 박제).
- **[모델 영향]** PASS (N/A 확정) — 사유: 학습·calibration·예측 수치 무관. `_EPSILON` 의도적 보존. `lru_cache(maxsize)` 의도적 보존. backend/ai/ 코드 무수정.

## 후속 (별도 CP)

1. **pre-existing pytest fail 정리**: 19개 fail (CP223 characterization 일부 + market provider + product history). CP227 이후 시점 누적. 별도 cleanup CP에서 식별·수정.
2. **collector 도메인 통합**: 현재 `collector/config.py`는 frozen dataclass + 자체 `get_settings()`. 미래에 `app/config`와 통합 여지 — 단 도메인 의미가 달라 강제 통합은 부적절. 별도 검토.
3. **`gzip_minimum_size` env alias 추가**: 현재 `CacheConfig.gzip_minimum_size=512` 코드 상수. env로 외부화 여지 있음 (모델 영향 0이라 안전).
4. **`@lru_cache` + `cache_clear()` 훅**: 성능 측정 후 필요하면 추가. 현재 미사용.

## ADR

`docs/adr/0025-pydantic-settings.md` 작성. 5 도메인 분리 사유, 호출 시점 재평가 계약, Supabase 환경변수 유지, `_EPSILON`/`lru_cache` 의도적 제외 기록.
