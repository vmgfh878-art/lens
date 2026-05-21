# CP87-D yfinance 전환 3단계 검증 보고서

작성일: 2026-05-03

## 1. Executive Summary

최종 판정: **WARN**

CP87-D는 운영 DB 전체 overwrite 없이 yfinance 1D 가격을 shadow 데이터로 받아 데이터, 피처, 모델 smoke를 순서대로 검증했다. EODHD 코드는 삭제하지 않았고, DB write, save-run, live inference 연결, full 모델 학습은 실행하지 않았다.

- 데이터 검증: passed=True, ticker_count=50, contract_violation_count=0, duplicate_date_count=0, min_provider_calendar_coverage=1.0
- 피처 검증: passed=True, feature_rows=59850, MODEL_N_FEATURES=36, atr_ratio_in_model_features=False
- 모델 smoke: passed=True, line_status=pass, band_status=pass

source_data_hash는 현재 DB max date/count 중심이라 provider/source 차이를 직접 포함하지 않는다. 그래서 데이터/피처/model smoke가 통과해도, 운영 write 전에는 제한 write와 cache 격리 정책을 먼저 고쳐야 한다.

## 2. 1단계 데이터 검증 결과

- 기간: 2021-05-02 ~ 2026-05-03
- 비교 티커 수: 50
- EODHD baseline rows: 1000
- adjusted OHLC sanity violation: 0
- duplicate ticker/date: 0
- split-like abnormal count: 0
- provider calendar coverage min: 1.0
- EODHD 대비 분류: {'baseline_missing': 2, 'dividend_adjustment_policy_diff': 4, 'pass': 30, 'split_adjustment_policy_diff': 10, 'unclassified_diff': 4}

2015년 이후 장기 샘플도 별도로 확인했다. 상세 row count와 violation은 metrics JSON의 `stage1_data.long_sample`에 남겼다.

## 3. 2단계 피처 검증 결과

- shadow price rows: 62800
- shadow feature rows: 59850
- feature build failures: 0
- finite summary: {'checked_columns': 36, 'checked_rows': 59850, 'inf_count': 0, 'nan_count': 0}
- open/high/low ratio p99: {'high_ratio': 0.07474499025260238, 'low_ratio': 0.07203106346902173, 'open_ratio': 0.047370252387899416}
- horizon 5 target: {'abs_return_gt_50pct_rate': 0.0, 'finite_count': 59600, 'nan_inf_count': 250, 'row_count': 59850, 'stats': {'max': 0.4693665875533162, 'mean': 0.0028272376869944646, 'p01': -0.12190145952703509, 'p50': 0.0030617730195861004, 'p99': 0.1336807280888009, 'std': 0.046547670013543326}, 'tail_rows_without_future_label': 250, 'model_sample_nan_inf_count': 0}
- split overlap: {'checked': True, 'eligible_ticker_count': 49, 'excluded_count': 1, 'gap': 20, 'overlap_count': 0, 'passed': True}
- indicators coverage: {'atr_ratio_non_null': 1000, 'latest_atr_ratio_non_null': 43, 'latest_date': '2021-05-26', 'latest_ticker_count': 43, 'ratio_abs_p99': {'high_ratio': 0.05761609804483139, 'low_ratio': 0.07439787716172712, 'open_ratio': 0.03147066373489351}, 'rows': 1000}
- shadow 저장: {'attempted': True, 'format': 'parquet', 'paths': ['C:\\Users\\user\\lens\\logs\\cp87_yfinance_validation\\yfinance_shadow_prices.parquet', 'C:\\Users\\user\\lens\\logs\\cp87_yfinance_validation\\yfinance_shadow_features.parquet']}
- EODHD feature 비교 상태: no_common_rows, common_rows=0, summary={'common_tickers': 0, 'eodhd_rows': 1000, 'yfinance_rows': 59850}

EODHD 기반 indicators와 yfinance shadow feature의 분포 차이 top 10은 다음과 같다.

| feature | difference_score | diff_abs_p99 | 분류 |
|---|---:|---:|---|
| 비교 불가 |  |  | EODHD indicators와 yfinance shadow feature의 공통 ticker/date가 없어 분포 비교를 보류 |

## 4. 3단계 모델 smoke 결과

line smoke는 PatchTST 1D h5, seq_len 252, patch_len 32, patch_stride 16, full_features, line_gate로 실행했다. band smoke는 CNN-LSTM 1D h5, seq_len 60, price_volatility_volume, q15/q85, lambda_band 2.0, direct band, band_gate로 실행했다.

- line exit_code: 0
- line metrics: {'asymmetric_interval_score': 2.6529293060302734, 'band_width_ic': 0.1848835998238524, 'coverage_abs_error': 0.16183371543884273, 'downside_width_ic': 0.03289912638187434, 'empirical_coverage': 0.9618337154388428, 'false_safe_tail_rate': 0.29865976241242764, 'forecast_loss': 0.3621216336121926, 'ic_mean': -0.02569033735120695, 'long_short_spread': -0.002005541406757782, 'lower_breach_rate': 0.020986901596188545, 'severe_downside_recall': 0.6942811330839124, 'total_loss': 0.3621216336121926, 'upper_breach_rate': 0.01717940904200077}
- band exit_code: 0
- band metrics: {'asymmetric_interval_score': 0.27223774790763855, 'band_width_ic': 0.03373482056031865, 'coverage_abs_error': 0.07901589870452885, 'downside_width_ic': 0.008321823671580828, 'empirical_coverage': 0.7790158987045288, 'false_safe_tail_rate': 0.3684737698760486, 'forecast_loss': 0.033134902419988066, 'ic_mean': -0.0022405377177394394, 'long_short_spread': 0.0003030912643884154, 'lower_breach_rate': 0.09350194036960602, 'severe_downside_recall': 0.6329713171818435, 'total_loss': 0.033134902419988066, 'upper_breach_rate': 0.12748216092586517}

이번 smoke는 제품 run_id를 교체하지 않았고, save-run도 하지 않았다. CP87 shadow smoke는 checkpoint 파일도 만들지 않도록 전용 경로로 실행했다.

## 5. EODHD 대비 차이

EODHD 대비 차이는 metrics JSON의 ticker별 `comparison`과 `classification`에 저장했다. 차이는 `pass`, `dividend_adjustment_policy_diff`, `split_adjustment_policy_diff`, `baseline_missing`, `coverage_mismatch`, `unclassified_diff`로 분류한다.

## 6. 전환 가능/불가 판정

최종 판정은 **WARN**이다.

- PASS는 yfinance primary 제한 전환 가능을 뜻한다.
- WARN은 개인 로컬 dry-run은 가능하지만 운영 write 전 보완이 필요하다는 뜻이다.
- FAIL은 yfinance 전환 금지를 뜻한다.

## 7. 전환 전 반드시 고칠 것

- price_data에 provider/source provenance가 없어서 yfinance row와 EODHD row를 schema상 구분하지 못한다.
- source_data_hash가 provider를 직접 반영하지 않아 캐시 격리 정책이 필요하다.
- 제한 write 후 indicators 재계산과 atr_ratio coverage 재검증이 필요하다.
- stock_info는 yfinance가 흔들릴 수 있으므로 price_data distinct 기반 검색 fallback을 계속 유지해야 한다.

## 8. 다음 단계

다음 CP에서는 10~20티커 제한 write를 수행하고, 즉시 compute_indicators, data quality check, feature cache hash 격리 확인까지 한 번에 묶어야 한다. live inference 연결은 그 이후 단계로 둔다.

## 실행한 읽기/검증 명령

- `python -m ai.cp87_yfinance_three_stage_validation ...`
- `scripts/run_daily_local_market_sync.ps1 -DryRun ...`
- `python -m py_compile ...`
- `python -m unittest backend.tests.test_market_data_providers`
