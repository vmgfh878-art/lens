CP52는 새 모델 성능 개선이 아니라, CP51에서 추가한 line/band/composite 지표의 정의와 1차 기준선을 고정하는 CP다.

## 범위

full 473티커 학습, 대형 sweep, save-run, UI 수정, DB schema 변경, 모델 구조 변경, 기존 후보 재학습은 하지 않았다. 코드 변경은 평가 함수와 테스트에 한정했다.

## Line 지표 정의

| 지표 | 역할 | 계산식 | 기준 성격 | 누수 방지 |
|---|---|---|---|---|
| `spearman_ic` / `ic_mean` | line 일반 성능 | 날짜별 `Spearman(score, raw_future_return)` 평균 | v1 guideline | validation/test 실제 return만 사용 |
| `ic_std` | line 안정성 | 날짜별 IC의 sample std | 참고 | 동일 |
| `ic_ir` | line 안정성 | `ic_mean / ic_std` | v1 guideline | 동일 |
| `ic_t_stat` | line 안정성 | `ic_mean / (ic_std / sqrt(n_periods))` | 참고 | 동일 |
| `long_short_spread` / `spread_mean` | line 랭킹 성능 | 날짜별 top-k realized return - bottom-k realized return 평균 | v1 guideline | 동일 날짜 단면만 사용 |
| `spread_std` | line 안정성 | 날짜별 long-short spread sample std | 참고 | 동일 |
| `spread_ir` | line 안정성 | `spread_mean / spread_std` | v1 guideline | 동일 |
| `spread_t_stat` | line 안정성 | `spread_mean / (spread_std / sqrt(n_periods))` | 참고 | 동일 |
| `fee_adjusted_sharpe` | line 보조 성능 | fee-adjusted daily strategy return 평균 / 표준편차 | v1 guideline | 동일 |

## 보수적 Line 지표 정의

기존 `false_safe_rate`는 더 이상 단일 의미로 쓰지 않는다. backward compatibility를 위해 키는 유지하되, 보고서와 판단에서는 아래 세부 키를 우선 사용한다.

| 지표 | 역할 | 계산식 | 기준 성격 | 누수 방지 |
|---|---|---|---|---|
| `false_safe_negative_rate` | line 리스크 | `P(score >= 0 | actual < 0)` | 참고 | raw future return만 사용 |
| `false_safe_tail_rate` | line 리스크 | `P(score >= 0 | actual이 같은 asof_date 단면 하위 20%)` | v1 guideline | 같은 날짜 단면만 사용 |
| `false_safe_severe_rate` | line 리스크 | `P(score >= 0 | actual <= severe_downside_threshold)` | 참고 | threshold는 train split q10 |
| `severe_downside_recall` | line 리스크 | `P(score < 0 | actual <= severe_downside_threshold)` | v1 guideline | threshold는 train split q10 |
| `downside_capture_rate` | line 리스크 | `P(score가 같은 날짜 단면 하위 20% | actual이 같은 날짜 단면 하위 20%)` | v1 guideline | 같은 날짜 단면만 사용 |
| `conservative_bias` | line 리스크 | `mean(score - actual)`. 음수면 보수적 | 참고 | raw future return 기준 |
| `upside_sacrifice` | line 비용 | 실제 상위 20% sample에서 `actual - score` 평균 | 참고 | raw future return 기준 |

`severe_downside_threshold`는 기본적으로 train split raw future return의 하위 10%다. h5 -3%, h20 -7% 같은 절대 임계값은 보조 진단으로만 둔다.

## Horizon Bucket

line bucket은 다음 prefix를 사용한다.

| prefix | 의미 |
|---|---|
| `all_horizon_*` | 전체 horizon 평균 score 기준 |
| `h1_h5_*` | 1~5 step |
| `h6_h10_*` | 6~10 step |
| `h11_h20_*` | 11~20 step |

bucket이 horizon 밖이면 해당 키는 `null`로 남긴다. 이 규칙으로 CP49처럼 `all_horizon_*` 필드가 누락되는 문제를 방지한다.

## Band Calibration 정의

`nominal_coverage = q_high - q_low`로 고정한다.

| quantile | nominal_coverage |
|---|---:|
| q10/q90 | 0.80 |
| q15/q85 | 0.70 |
| q20/q80 | 0.60 |
| q25/q75 | 0.50 |

`empirical_coverage = P(lower <= band_target <= upper)`다. `coverage_abs_error = abs(empirical_coverage - nominal_coverage)`를 핵심 calibration 지표로 사용한다. 기존 `coverage` 단독 판정은 폐기하고, nominal 대비 오차로 판단한다.

## Asymmetric Interval Score

ranking용 interval score는 비대칭 하방 가중치를 적용한다.

```text
alpha = 1 - nominal_coverage
lower_penalty = lower_penalty_weight * (2 / alpha) * max(lower - y, 0)
upper_penalty = upper_penalty_weight * (2 / alpha) * max(y - upper, 0)
asymmetric_interval_score = width + lower_penalty + upper_penalty
```

기본값은 `lower_penalty_weight=2.0`, `upper_penalty_weight=1.0`이다. 코드 키는 backward compatibility를 위해 `interval_score`를 유지하지만, 문서상 의미는 asymmetric interval score다.

## Dynamic Width 정의

| 지표 | 역할 | 계산식 | baseline 필요 |
|---|---|---|---|
| `band_width_ic` | band 동적 폭 | `Spearman(upper - lower, abs(raw_future_return))` | 필요 |
| `downside_width_ic` | 하방 폭 | band 단독은 `band_center - lower`, composite는 `line - lower`; realized는 `max(-raw_future_return, 0)` | 필요 |
| `width_bucket_realized_vol_ratio` | 폭-변동성 정렬 | band width 상위 1/3 bucket realized_abs 평균 / 하위 1/3 평균 | 필요 |
| `width_bucket_downside_rate_ratio` | 폭-하방위험 정렬 | band width 상위 1/3 downside rate / 하위 1/3 downside rate | 필요 |
| `squeeze_breakout_rate` | squeeze 위험 | width 하위 20% 구간에서 `abs(realized)`가 train 기준 상위 20%를 넘는 비율 | 필요 |

`squeeze_breakout_threshold`는 train split의 `abs(raw_future_return)` 80% 분위수만 사용한다.

## Quantile Calibration Table

이번 CP에서는 PIT라는 용어를 쓰지 않고 quantile calibration table로 부른다.

| 지표 | 계산식 | 현재 지원 |
|---|---|---|
| `empirical_q_low` | `P(y <= predicted_lower)` | 지원 |
| `empirical_q_high` | `P(y <= predicted_upper)` | 지원 |
| `empirical_p10` | q10 출력이 있을 때만 채움 | 부분 지원 |
| `empirical_p25` | q25 출력이 있을 때만 채움 | 부분 지원 |
| `empirical_p50` | q50 head가 없으면 `null` | 미지원 |
| `empirical_p75` | q75 출력이 있을 때만 채움 | 부분 지원 |
| `empirical_p90` | q90 출력이 있을 때만 채움 | 부분 지원 |

현재 모델이 lower/upper만 출력하므로 full quantile table은 미지원이다.

## Composite 지표 위치

Composite 지표는 모델 탈락 기준이 아니라 제품 표시/정책 지표다.

| 지표 | 역할 | 기준 성격 |
|---|---|---|
| `line_inside_band_ratio` | 표시 정합성 참고 | 탈락 기준 아님 |
| `product_display_warning_rate` | 저장/화면 표시 정책 후 경고율 | 0에 가까워야 함 |
| `conservative_series_false_safe_rate` | 최종 보수적 예측선 리스크 | 중요 |
| `composite_width_increase_ratio` | 조합 정책으로 밴드를 얼마나 늘렸는지 | 성능 지표 아님 |

## Baseline 계약

band width와 interval score는 절대값만으로 판단하지 않는다. 다음 baseline을 같은 평가판에 올린다.

| baseline | 정의 | 다음 CP 구현 필요 |
|---|---|---|
| Bollinger return band | rolling 20 realized return mean ± 2σ | 예 |
| Historical quantile band | rolling window realized return q_low/q_high | 예 |
| Constant-width band | train 평균 width 또는 train quantile 기반 고정 폭 | 예 |

## v1 Guideline

hard gate가 아니라 1차 후보 판단 기준이다.

| Line 기준 | 생존 | 후보 |
|---|---:|---:|
| IC mean | > 0 | >= 0.02 |
| IC IR | > 0.25 | >= 0.5 |
| long_short_spread net | > 0 | >= 0.003 |
| fee_adjusted_sharpe | > 0 | >= 0.3 |
| false_safe_tail_rate | < 0.25 | < 0.15 |
| severe_downside_recall | >= 0.65 | >= 0.80 |
| downside_capture_rate | >= 0.30 | >= 0.35 |

| Band 기준 | 생존 | 후보 |
|---|---:|---:|
| coverage_abs_error | <= 0.15 | <= 0.08 |
| asymmetric_interval_score | baseline보다 낮음 | baseline보다 낮음 |
| band_width_ic | > 0 | >= 0.05 |
| downside_width_ic | > 0 | >= 0.05 |
| width_bucket_realized_vol_ratio | > 1.0 | >= 1.15 |
| squeeze_breakout_rate | baseline 이하 | baseline 이하 |

| Composite 기준 | 기준 |
|---|---|
| product_display_warning_rate | 거의 0 |
| conservative_series_false_safe_rate | 후보 < 0.10, fail > 0.20 |
| line_inside_band_ratio | 참고 지표 |

## 다음 CP 표준 Metric Set

기존 후보 재채점 시 최소 표준 표는 다음 열을 포함한다.

| 역할 | 표준 열 |
|---|---|
| line | `ic_mean`, `ic_std`, `ic_ir`, `ic_t_stat`, `spread_mean`, `spread_std`, `spread_ir`, `spread_t_stat`, `fee_adjusted_sharpe`, `false_safe_negative_rate`, `false_safe_tail_rate`, `false_safe_severe_rate`, `severe_downside_recall`, `downside_capture_rate`, `conservative_bias`, `upside_sacrifice` |
| band | `nominal_coverage`, `empirical_coverage`, `coverage_abs_error`, `lower_breach_rate`, `upper_breach_rate`, `avg_band_width`, `median_band_width`, `p90_band_width`, `interval_score`, `interval_lower_penalty`, `interval_upper_penalty`, `band_width_ic`, `downside_width_ic`, `width_bucket_realized_vol_ratio`, `width_bucket_downside_rate_ratio`, `squeeze_breakout_rate`, `empirical_q_low`, `empirical_q_high` |
| composite | `line_inside_band_ratio`, `product_display_warning_rate`, `conservative_series_false_safe_rate`, `composite_width_increase_ratio` |

## 검증

- `C:\Users\user\lens\.venv\Scripts\python.exe -m py_compile ai/evaluation.py ai/train.py ai/inference.py ai/composite_inference.py ai/composite_policy_eval.py ai/cp46_upper_calibration.py`
- `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_evaluation_targets`
- `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_metric_definition_contract`
- `C:\Users\user\lens\.venv\Scripts\python.exe -m unittest ai.tests.test_checkpoint_selection`

## 산출물

- `docs/cp52_metric_definition_contract_report.md`
- `docs/cp52_metric_definition_schema.json`
