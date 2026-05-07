CP51은 모델 개선이 아니라, 보수적 line과 AI band의 역할에 맞게 평가 지표 계약을 재설계한 CP다.

## 목표

기존 평가는 coverage와 일부 line 지표가 섞여 있어, 밴드를 넓혀서 coverage만 맞추는 후보와 실제로 유용한 risk interval 후보를 구분하기 어려웠다. CP51에서는 line, band, composite 표시 지표를 분리하고, 기존 호출부와 저장 JSON 호환성을 유지했다.

## 구현 파일

- `ai/evaluation.py`: band calibration, sharpness, interval score, dynamic width, horizon bucket 지표 추가.
- `ai/train.py`: `q_low/q_high`를 평가 함수로 전달하고, test 결과에서 새 지표가 누락되지 않게 전체 metric dict를 보존.
- `ai/inference.py`: inference summary에 checkpoint quantile을 전달.
- `ai/band_calibration.py`: calibration summary에 후보 quantile을 전달.
- `ai/composite_inference.py`: composite summary에 band model quantile을 전달.
- `ai/composite_policy_eval.py`: composite policy 비교 summary에 band model quantile을 전달.
- `ai/cp46_upper_calibration.py`: upper calibration summary에 band model quantile을 전달.
- `ai/baselines.py`: baseline summary의 nominal coverage를 0.80으로 고정 전달.
- `ai/tests/test_evaluation_targets.py`: CP51 synthetic 지표 검증 3건 추가.

## Line 지표

일반 예측 지표는 기존 `spearman_ic`, `long_short_spread`, `mae`, `smape`를 유지한다. 보수적 line 지표는 `overprediction_rate`, `mean_overprediction`, `underprediction_rate`, `mean_underprediction`, `false_safe_rate`, `severe_downside_recall`, `downside_capture_rate`, `conservative_bias`, `upside_sacrifice`를 유지한다.

horizon bucket은 `all_horizon_*`, `h1_h5_*`, `h6_h10_*`, `h11_h20_*` prefix로 고정했다. CP49에서 빠졌던 `all_horizon_*` 계열은 `evaluate_bundle()` 결과 보존 로직을 넓혀 다시 빠지지 않게 했다.

## Band 지표

Calibration 지표는 `nominal_coverage`, `empirical_coverage`, `coverage_error`, `coverage_abs_error`, `lower_breach_rate`, `upper_breach_rate`다. 기존 `coverage`는 backward compatibility를 위해 유지하고, 새 `empirical_coverage`와 같은 의미로 둔다.

Sharpness 지표는 기존 `avg_band_width`에 `median_band_width`, `p90_band_width`를 추가했다. 밴드 폭 평균만으로 과도한 tail 폭을 놓치지 않기 위한 장치다.

Interval score는 `interval_score = width + lower_penalty + upper_penalty`로 계산한다. CP52에서 기본 penalty weight는 lower 2.0, upper 1.0으로 고정했으며, 하방 리스크를 더 보수적으로 본다. 세부 항목은 `interval_width_component`, `interval_lower_penalty`, `interval_upper_penalty`로 분리 저장한다.

Dynamic width 지표는 `band_width_ic`, `downside_width_ic`, width bucket별 realized volatility/downside rate, `squeeze_breakout_rate`를 추가했다. 목적은 밴드가 단순히 넓은지보다, 실제 변동성과 하방 위험이 큰 구간에서 더 넓어지는지 확인하는 것이다.

## Horizon Band Bucket

band 전용 horizon bucket은 `all_horizon_band_*`, `h1_h5_band_*`, `h6_h10_band_*`, `h11_h20_band_*` prefix로 분리했다. line bucket과 이름 충돌을 피하기 위해 `band` prefix를 명시했다.

## Composite 표시 지표

`line_inside_band_ratio`는 모델 탈락 기준이 아니라 제품 표시 보조 지표로 유지한다. 포인트 단위 비율은 `line_inside_band_point_ratio`로 분리했다. 표시 위험은 `product_display_warning_rate`로 계산한다.

`conservative_series_false_safe_rate`는 conservative series로 쓰는 lower band가 위험 구간에서 0 이상으로 안전하게 보이는 비율이다. composite 정책의 하방 보수성 확인용 보조 지표다.

## W&B 키 표준

train loop는 기존처럼 `train/<metric>`, `val/<metric>`, `test/<metric>` 형태로 기록한다. 새 지표도 모두 scalar key라 W&B, JSON, finite gate에서 별도 변환 없이 처리된다.

## 검증

- `C:\Users\user\lens\.venv\Scripts\python.exe -m py_compile ai/evaluation.py ai/train.py ai/inference.py ai/band_calibration.py ai/composite_inference.py ai/composite_policy_eval.py ai/cp46_upper_calibration.py ai/baselines.py`: 통과.
- `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_evaluation_targets`: 10건 통과.
- `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_checkpoint_selection`: 12건 통과.
- `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest discover -s ai\tests -p test_*.py`: 139건 통과.

## 산출물

- `docs/cp51_metric_redesign_report.md`
- `docs/cp51_metric_schema.json`

## 남은 리스크

`composite_width_increase_ratio`는 두 밴드 후보의 기준 폭이 필요하므로 `summarize_forecast_metrics()` 공통 함수가 아니라 composite probe 스크립트 계층에서 계속 계산한다. 다음 composite CP에서 metric schema의 composite 확장 항목으로 유지하면 된다.
