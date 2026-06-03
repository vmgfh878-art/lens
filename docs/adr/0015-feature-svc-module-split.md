# ADR-0015: feature_svc 4-module split (definition / validators / resampling / calculator)

Status: Accepted
Date: 2026-06-03
Context: CP225 (refactoring runbook CP221~237)

## Context

`backend/app/services/feature_svc.py` 가 619 줄 (CP224b 직후 기준) 단일 파일에 4 종의 책임을 묶고 있었다:

1. **정의 상수** — `FEATURE_COLUMNS`, `REQUIRED_FEATURE_COLUMNS`, `SUPPORTED_TIMEFRAMES`, ratio sanity 한도 등.
2. **OHLC / ratio 검증** — `_validate_adjusted_ohlc_contract`, `_apply_adjusted_ohlc_contract`, `_validate_ratio_feature_sanity`.
3. **타임프레임 정규화 / 리샘플링** — `normalize_timeframe`, `_resample_single_ticker`, `resample_price_frame`, `_resample_context_frame`, `latest_complete_period_end`, `drop_incomplete_resampled_periods`, `_ensure_datetime`.
4. **지표 계산 + 오케스트레이션** — `_compute_rsi`, `_compute_features_for_single_ticker`, `_apply_regime_columns`, `_apply_context_flags`, `_apply_fundamental_features`, `build_features`, `build_price_features`.

caller 인벤토리 (grep 실측):
- 공개 심볼: `backend/app/services/api_service.py`, `backend/collector/...`, `backend/db/scripts/...`, `ai/preprocessing.py`, `ai/cp202_1_band_baseline_comparison.py`, `scripts/cp*` 다수.
- **사설 (언더스코어) 심볼**: `scripts/diagnostics/data_length_audit.py`, `scripts/diagnostics/base_feature_nan_tracer.py` 가 `_BASE_FEATURE_COLUMNS`, `_CONTEXT_COLUMNS`, `_REGIME_FEATURE_COLUMNS`, `_compute_features_for_single_ticker`, `_resample_context_frame`, `_apply_fundamental_features`, `_apply_regime_columns` 등을 직접 import.

Plan v3 는 fidelity 우선이라 한 줄 계산 변경도 회귀로 본다. 단일 파일은 변경 영향 범위 추적이 어렵다.

## Decision

`feature_svc.py` 를 책임별 4 모듈로 분리한다:

```
feature_definition.py   (140 줄)  ← 무의존
        ↑
validators.py           (100 줄)  ← definition
        ↑
resampling.py           (166 줄)  ← definition + validators
        ↑
feature_calculator.py   (338 줄)  ← definition + validators + resampling
```

의존 방향은 단방향. 순환 import 없음.

`feature_svc.py` 는 **re-export 파사드 (106 줄)** 로 남긴다. 공개 + 사설 심볼 모두 (38 개) `__all__` 에 명시. 기존 caller 의 import 경로 (`backend.app.services.feature_svc.X`) 와 diagnostics 의 사설 심볼 import 가 무수정으로 통과한다.

## Consequences

긍정:
- 정의/검증/리샘플/계산을 따로 읽고 따로 테스트할 수 있다.
- 변경 영향 범위가 모듈 경계로 좁아진다 (예: ratio sanity 한도 조정은 validators 만 확인).
- caller 0 라인 수정. 외부 회귀 0.

부정 / 미해결:
- `feature_calculator.py` 가 338 줄로 200 목표를 초과. CP225 후속에서 indicators (`_compute_rsi`, `_compute_features_for_single_ticker`) vs assembly (`build_features` 류) 분리 검토 권장.
- 새 모듈로의 caller 이전은 별도 후속 CP. 지금은 호환 유지가 우선.

대안 및 기각 이유:
- (a) **파일 그대로 두기** — Plan v3 의 fidelity 추적 / 변경 영향 분석 비용을 계속 지불해야 함. SRP 위반 누적.
- (b) **caller 전부 새 경로로 이전** — 단일 CP 범위 초과. diagnostics 사설 심볼 caller 가 2 개 있어 즉시 전환 위험. re-export 파사드로 점진 이전이 안전.
- (c) **`from feature_definition import *` 같은 wildcard re-export** — 지시서 금지. 사설 심볼 명시 보존이 명확.

Fidelity 보장:
- CP223 syrupy snapshot 9 endpoint, 매 Step diff 0.
- `test_feature_svc.py` 11 케이스 매 Step pass.
- 계산식 / 컬럼 정의 / sanity 한도 / merge_asof 옵션 1 비트 변경 없음.
- `backend/tests` 87 passed (CP224b baseline 유지, 회귀 0).
