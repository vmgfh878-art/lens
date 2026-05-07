# CP61-M Line/Band Metric Split 보고서

## 1. CP51/52 지표 이동 표
| 지표군 | CP61 분류 | 처리 |
|---|---|---|
| IC, spread, MAE, SMAPE, direction_accuracy | `line_metrics` | 유지 |
| false-safe, severe recall, downside capture, conservative bias | `line_metrics` | 유지 |
| coverage, breach, band width, quantile calibration | `band_metrics` | 유지 |
| asymmetric interval score, lower/upper penalty | `band_metrics` | 유지 |
| band width IC, downside width IC, width bucket, squeeze breakout | `band_metrics` | 유지 |
| line_inside_band, product display warning | `legacy_overlay_diagnostics` | 기본 model metric에서 제거 |
| composition_policy, clamp, risk_first, upper_buffer | `legacy_overlay_diagnostics` | 모델 랭킹 사용 금지 |

## 2. line_metrics 최종 정의
`line_metrics`는 AI line layer 전용이다. 핵심 키는 `spearman_ic`, `ic_mean/std/ir/t_stat`, `long_short_spread`, `spread_mean/std/ir/t_stat`, `direction_accuracy`, `mae`, `smape`, `false_safe_negative_rate`, `false_safe_tail_rate`, `false_safe_severe_rate`, `severe_downside_recall`, `downside_capture_rate`, `conservative_bias`, `upside_sacrifice`다.

## 3. band_metrics 최종 정의
`band_metrics`는 AI band layer 전용이다. 핵심 키는 `nominal_coverage`, `empirical_coverage`, `coverage_error`, `coverage_abs_error`, `lower_breach_rate`, `upper_breach_rate`, `avg_band_width`, `median_band_width`, `p90_band_width`, `asymmetric_interval_score`, `interval_lower_penalty`, `interval_upper_penalty`, `band_width_ic`, `downside_width_ic`, quantile calibration table, width bucket, squeeze breakout 지표다.

## 4. legacy_overlay_diagnostics 최종 정의
`legacy_overlay_diagnostics`는 제품 표시/과거 composite 진단 전용이다. `line_inside_band_ratio`, `line_inside_band_point_ratio`, `product_display_warning_rate`, `conservative_series_false_safe_rate`, composition policy 계열 값은 모델 랭킹과 checkpoint selector에 사용하지 않는다.

## 5. selector 변경 내용
`line_gate`는 `line_metrics`만 우선 참조한다. `band_gate`는 `band_metrics`만 우선 참조한다. `combined_gate`와 `coverage_gate`는 CLI 호환을 위해 남기지만 deprecated이며, `coverage_gate`는 계속 `combined_gate` alias로 해석한다.

## 6. composite 지표 모델 랭킹 제거 확인
기본 `summarize_forecast_metrics` 반환값에서 `line_inside_band_ratio` 등 overlay 계열 flat metric은 빠진다. 필요할 때만 `include_legacy_overlay_diagnostics=True`로 `legacy_overlay_diagnostics`를 만든다. 따라서 기본 학습/평가 루프의 line/band selector는 composite 표시 지표를 보지 않는다.

## 7. 다음 모델 실험 평가표 예시
Line 실험 표는 `line_metrics.spearman_ic`, `line_metrics.ic_ir`, `line_metrics.long_short_spread`, `line_metrics.false_safe_tail_rate`, `line_metrics.severe_downside_recall`을 우선 본다. Band 실험 표는 `band_metrics.coverage_abs_error`, `band_metrics.asymmetric_interval_score`, `band_metrics.lower_breach_rate`, `band_metrics.upper_breach_rate`, `band_metrics.band_width_ic`, `band_metrics.downside_width_ic`를 우선 본다.
