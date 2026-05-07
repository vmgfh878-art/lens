# CP60-M Composite Legacy AI Guard 보고서

## 1. composite AI 스크립트 격하
`ai/composite_inference.py`와 `ai/composite_policy_eval.py`는 Phase 1 기본 제품 계약이 아니라 legacy/research utility로 고정했다. Phase 1 본류 산출물은 `AI line layer`와 `AI band layer`이며, composite row는 과거 demo 또는 연구용 진단 산출물로만 남긴다.

## 2. `--save` 차단 정책
`ai.composite_inference`에서 `--save`를 사용하려면 `--allow-legacy-composite-save`를 반드시 함께 넣어야 한다. 플래그 없이 `--save`가 들어오면 실행 초기에 실패하며, 이유는 legacy composite row가 Phase 1 제품 기본 저장 계약이 아니기 때문이다.

## 3. 저장 meta 강제 필드
명시적으로 legacy 저장을 허용해도 저장 config/meta에는 다음 필드를 강제한다: `deprecated_for_phase1_product_contract=true`, `indicator_layer_replacement=overlay_bundle`, `role=composite_model`, `legacy_reason=line_and_band_layers_are_product-separated`, `composite_is_not_product_default=true`.

## 4. `line_inside_band` 의미 변경
`line_inside_band_ratio`는 제품 탈락 기준이 아니라 표시 진단값이다. `line_inside_band_required`와 같은 성공 조건 의미를 제거하고, `line_inside_band_diagnostic_only=true`, `line_inside_band_is_success_condition=false`로 계약을 바꿨다.

## 5. h5 하드코딩 제거
`series_length_all_5` 체크를 제거하고 `expected_horizon` 기준의 `series_length_matches_horizon`, `forecast_dates_length_matches_horizon`으로 바꿨다. 따라서 h5/h10/h20 모두 같은 체크 계약을 사용한다.

## 6. 중복 정책 정리
`include_line_clamp`와 `risk_first_lower_preserve`는 현재 구현상 동일한 legacy 표시 진단 정책이다. `risk_first_lower_preserve`는 Phase 1 제품 기본 정책이 아니라 `include_line_clamp`의 legacy alias로 문서화했다.

## 7. composite policy 진단 스크립트 정리
CP43/CP46류 upper buffer/clamp 비교는 모델 후보 통과/탈락 판정이 아니라 `diagnostic comparison`으로만 표시한다. `ai/composite_policy_eval.py`와 `ai/cp46_upper_calibration.py`의 출력 키도 `pass_flags`/`all_pass` 계열 대신 `diagnostic_flags`, `diagnostic_all_thresholds_met`로 바꿔 성능 개선 판정처럼 보이지 않게 했다.

## 8. 검증
`py_compile` 대상은 `ai/composite_inference.py`, `ai/composite_policy_eval.py`, `ai/cp46_upper_calibration.py`, `ai/tests/test_composite_legacy_guard.py`다. 단위 테스트는 `ai.tests.test_composite_legacy_guard`, `ai.tests.test_evaluation_targets`, `ai.tests.test_metric_definition_contract`, `ai.tests.test_inference_backtest`, `ai.tests.test_storage_contracts`를 실행했다.

## 9. 남은 과제
남은 작업은 overlay bundle API, line layer inference 저장 계약, band layer inference 저장 계약, rolling AI band history 설계다. 이번 CP에서는 제품/UI/백엔드 API와 DB schema를 건드리지 않았다.
