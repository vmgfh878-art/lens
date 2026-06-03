# CP225 보고서 — feature_svc 4-module split

## 요구

`docs/cp225_feature_svc_split_directive.md`. 591 줄 (CP224b 직후 619 줄) `backend/app/services/feature_svc.py` 를 definition / validators / resampling / calculator 4 모듈로 분리. fidelity 1bit 보존. caller 인터페이스 (공개 + 사설 심볼) 무수정.

## 한 일

5 step, 5 commit:

| Step | 커밋 | 작업 | 검증 |
|---|---|---|---|
| 1 | a814f8a | feature_definition.py 추출 (140 줄) | ruff/mypy/pytest/snapshot PASS |
| 2 | f13d496 | validators.py 추출 (100 줄) | ruff/mypy/pytest/snapshot PASS |
| 3 | fd1ffe8 | resampling.py 추출 (166 줄) | ruff/mypy/pytest/snapshot PASS |
| 4 | b2672b0 | feature_calculator.py 추출 (338 줄) | ruff/mypy/pytest/snapshot PASS |
| 5 | 70f28cf | feature_svc.py → facade (106 줄) + __all__ 38 심볼 | ruff/mypy/pytest/snapshot PASS |

## 핵심 컴포넌트 존재 체크리스트

- `_apply_adjusted_ohlc_contract` → validators.py (이동), feature_svc 재공개.
- `_validate_ratio_feature_sanity` → validators.py (이동), feature_svc 재공개.
- `normalize_timeframe` → resampling.py (이동), feature_svc 재공개.
- `resample_price_frame` → resampling.py (이동), feature_svc 재공개.
- `_resample_context_frame` → resampling.py (이동), feature_svc 재공개.
- `_compute_rsi` → feature_calculator.py (이동), feature_svc 재공개.
- `_compute_features_for_single_ticker` → feature_calculator.py (이동), feature_svc 재공개. diagnostics caller 보호.
- `build_features` 출력 컬럼 = `_OUTPUT_COLUMNS` 그대로.
- `build_price_features` 출력 = `["ticker", "date", "timeframe", *PRICE_DERIVED_FEATURE_COLUMNS]` 그대로.
- merge_asof (`direction="backward"`, `allow_exact_matches=True`) 옵션 동일.
- `fundamental_quarter_count < 8` insufficiency mask 동일.
- VIX < 15 → calm, ≥ 25 → stress, otherwise neutral — 임계값 동일.
- ratio sanity `_P99_RATIO_ABS_LIMIT=1.0`, `_MAX_RATIO_ABS_LIMIT=5.0`, `enforce_distribution=timeframe != "1M"` 동일.

## 새 테스트 결과

이번 CP 는 신규 테스트 추가 없음 (구조 이동만). 기존 `backend/tests/test_feature_svc.py` 11 케이스가 매 Step 그린.

## dry-run 결과

`build_features` / `build_price_features` 의 실 호출 결과는 CP223 syrupy snapshot 으로 박제. 매 Step:
- `pytest backend/tests/test_characterization_api.py -q` → 9 snapshots passed, diff 0.
- `pytest backend/tests/test_feature_svc.py -q` → 11 passed, failed 0.

`lower<=upper` / `line_preserved` 는 forecast 출력 검증 항목으로 본 CP 범위 밖이지만, CP223 snapshot 이 line/band/price 응답 9 개를 박제하므로 간접 보장.

## 기존 회귀 통과 건수

- `pytest backend/tests` (test_services.py 제외): **87 passed** — CP224b baseline 그대로.
- `pytest backend/tests` (전체): 87 passed + 11 failed (전부 pre-existing 다른 파일, 본 CP 영향 0).
- import sanity (공개 + 사설 21 심볼 + diagnostics 2 파일): 통과.
- `ruff check`: 2 errors (둘 다 pre-existing E501, 위치만 모듈 이동에 따라 이동).
- `mypy`: 신규 error 0.

## 결정

- 4 모듈 단방향 의존: definition ← validators ← resampling ← calculator.
- `feature_svc.py` 는 re-export 파사드 (`__all__` 38 심볼, 공개 21 + 사설 17). 기존 caller 0 라인 수정.
- `feature_calculator.py` 338 줄 — 목표 200 초과. **차단 보고 대신 진행**: ① 본 CP 의 범위는 4 모듈 분리이며 추가 분리는 명시적 후속 권장 사안 ② 동작 변경 0 이라 fidelity 가 우선 ③ 추가 분리 (indicators 순수 함수 vs build_features 류 오케스트레이션) 는 ADR-0015 에 후속 권고로 기록.

## 후속

1. **`feature_calculator.py` 추가 분리 검토 CP** — `_compute_rsi`, `_compute_features_for_single_ticker` 를 `indicators.py` 로, `_apply_*` + `build_features` 류를 `feature_assembly.py` 로 분리. CP226~229 이후 우선순위 검토.
2. **caller 점진 이전** — 공개 심볼 caller (ai/preprocessing, collector pipelines, scripts/cp*) 와 사설 심볼 caller (scripts/diagnostics/*) 를 새 모듈 경로 (`backend.app.services.feature_definition` 등) 로 점진 이전. 별도 후속 CP. 본 CP 의 re-export 파사드 덕에 시한 압박 없음.
3. **줄 수 모니터링** — feature_calculator 338 줄이 향후 추가 indicators 추가로 더 늘어나지 않도록 모니터링.
