# ADR 0011 — Characterization snapshot 안전망 (syrupy + float normalize)

- **상태**: 채택 (2026-06-03, CP223 Step 1~5)
- **컨텍스트**: CP225+ 백엔드 분리 리팩토링(`feature_svc` 591줄, `strategy_backtest_svc` 590줄 등)을 시작하려면 분리 전후 응답이 "1바이트도(허용 tolerance 내) 안 바뀌었다"를 기계적으로 증명할 안전망이 먼저 있어야 한다 (Feathers, *Working Effectively with Legacy Code*; Fowler, *Refactoring* 2e — characterization test). 기존 `backend/tests/test_api.py`는 전부 mock 기반이라 실제 parquet → 직렬화 경로를 한 번도 통과하지 않아 안전망이 못 된다.

## 결정 요지

| 항목 | 선택 | 근거 |
|---|---|---|
| 도구 | **syrupy 5.3.1** | 지시서 1순위 `snaptol`은 PyPI 부재(`Could not find a version that satisfies the requirement snaptol`). 2순위 `syrupy`로 폴백 — 사용자 결정. dict/list 자동 재귀 직렬화, 안정적, popular. |
| float tolerance | **`normalize_floats(json, ndigits=9)`** | syrupy 자체엔 float tolerance가 없음. 응답 dict를 syrupy에 넘기기 직전 `round(v, 9)` 재귀 적용. rtol≈1e-9 효과. numpy/pandas 버전 미세 변동 흡수. |
| 직렬화 형식 | syrupy 기본 `.ambr` (SingleFileAmberSnapshotExtension) | 사람 가독, parametrize id별 섹션 분리. git diff 가능. |
| baseline 위치 | `backend/tests/__snapshots__/test_characterization_api.ambr` (1.55MB, git 추적) | syrupy 기본 경로. |
| 비결정 회피 | (a) `X-Request-Id="test-fixed"` 헤더 (b) `aapl_prices` start/end 명시 | 운영 코드 무수정. 호출 측에서만 해결. |
| 커버리지 | **9개 endpoint** (목표 6개 이상 충족) | stocks list/prices/indicators/product-history, predictions line/band1d/band1w, strategies scan/backtest. AAPL 고정. |
| 제외 | Supabase, /ai/runs*, /admin/*, 404/422/503 에러경로 | v1에서 비활성 / 외부 mock / write 트리거 / 기존 `test_api.py`가 mock으로 이미 커버. |

## 비결정성 원천 2건 (실측 발견 + 회피 방식)

1. **`backend/app/middleware/request_id.py:9`** — 모든 응답 `meta.request_id`가 `str(uuid4())` 또는 `X-Request-Id` 헤더 에코. 매 호출 다름. → conftest의 `FIXED_HEADERS = {"X-Request-Id": "test-fixed"}`로 박제. 가드: snapshot 비교 직전 `meta.request_id == "test-fixed"` 명시 assert (uuid leak 방지).
2. **`backend/app/services/api_service.py:57-62`** — `resolve_price_window(end=None)`이 `date.today()`로 폴백. → `aapl_prices` 케이스의 `params={"start":"2025-01-02","end":"2025-06-30"}`로 명시 고정.

운영 코드는 **수정하지 않았다**. 비결정성은 호출 측에서만 회피.

## 대안과 거부 이유

- **자체 fixture (pytest.approx)** — 가능하나 baseline 직렬화/관리/diff 가독성을 처음부터 구축해야 함. syrupy의 ambr 직렬화가 사실상 같은 일을 해 줌. 거부.
- **syrupy 단독 (float normalize 없이)** — pandas/numpy 버전 미세 변동에서 false positive 가능. CP223 안전망의 신호 신뢰도 떨어짐. 거부.
- **snaptol GitHub 직접 설치** — 출처/리비전 리스크 + 사용자 결정 불요. 거부.

## 결과 (CP223 baseline)

| 항목 | 실측 |
|---|---|
| baseline 생성 | 9 snapshots generated, 9 passed (Step 3) |
| 재실행 diff=0 | 9 passed, git status clean (Step 4) |
| 2회 동일성 | 2회 연속 9 passed (Step 5) |
| backend/tests 회귀 | 78 passed (CP222 baseline) + 9 characterization = **87 passed**. 신규 실패 0. |
| 운영 코드 변경 | **0 라인** (`backend/app/**` 무변경) |
| uuid leak | 0건 (전체 1.55MB ambr 파일 grep) |
| baseline 크기 | 1.55MB (44,786 라인) — git이 안정적으로 다룰 사이즈 |

## 후속

- **CP225**: `feature_svc` 591줄 분리. CP223 baseline diff=0 유지하면서 모듈 분리.
- **CP226**: `strategy_backtest_svc` 590줄 분리. 동일.
- **운영 코드 변경 CP**(별도): `date.today()` 명시 vs 헤더 강제 같은 정책 정리.
- **v1 parquet 가드 conftest** (CP223 직전 권장이었던 별도 작업): `conftest.py`에 `backend/data/v1/*.parquet` SHA256 스냅샷 + 세션 종료 검증 추가. CP225 시작 전 권장.
- **테스트 stale reference 정리** (별도 cleanup CP): test_api.py 7건 + test_services.py / test_product_prediction_history_api.py / DatasetPlan signature 4건.
