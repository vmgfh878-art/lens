# CP123-DG 1W 데이터 충분성 및 과적합 가드 감사

생성일: 2026-05-06

## 1. 요약

최종 판정: `PASS_WITH_GUARDS`

현재 1W yfinance 로컬 snapshot은 1W line/band 후보 실험을 계속할 수 있을 만큼의 기간, ticker, sector coverage를 갖고 있다. 다만 CP112~CP119에서 동일 test window를 반복 참조하며 후보를 좁힌 흔적이 커서, 다음 단계부터는 test를 중간 선택 기준으로 쓰면 안 된다.

핵심 결론:
- 1W indicator 기간: `2016-02-19` ~ `2026-05-01`
- input ticker: `100`, eligible ticker: `97`
- sector 수: 전체 `11`
- split: train `2018-02-09` ~ `2023-05-12`, val `2023-08-11` ~ `2024-09-20`, test `2024-12-20` ~ `2026-02-06`
- CP112~CP119 test/candidate 참조 수: `22`
- 과적합 위험: `HIGH`
- 주요 경고: `stock_info sector UNKNOWN 비중이 높아 sector holdout 감사 신뢰도가 제한된다.; 동일 test set 반복 참조가 많아 후보 선택 bias 위험이 높다.`

## 2. 데이터 coverage

| 항목 | 값 |
|---|---:|
| 1W price rows | 59200 |
| 1W indicator rows | 53300 |
| price ticker 수 | 100 |
| indicator ticker 수 | 100 |
| eligible ticker 수 | 97 |
| excluded ticker | LMT, MS, T |
| price duplicate `(ticker,date,source)` | 0 |
| indicator duplicate `(ticker,date,source)` | 0 |

sector별 eligible ticker 수:

```json
{
  "UNKNOWN": 43,
  "Technology": 13,
  "Financial Services": 12,
  "Healthcare": 8,
  "Industrials": 7,
  "Consumer Cyclical": 4,
  "Consumer Defensive": 3,
  "Energy": 3,
  "Communication Services": 2,
  "Basic Materials": 1,
  "Real Estate": 1
}
```

주의: `UNKNOWN` sector가 `43`개라서 현재 stock_info만으로 sector holdout을 설계하면 편향이 생길 수 있다. sector holdout은 stock_info sector 보강 후 적용하는 편이 안전하다.

## 3. Split 기간과 sample 분포

| split | rows | tickers | sectors | asof_min | asof_max |
| --- | --- | --- | --- | --- | --- |
| train | 26675 | 97 | 11 | 2018-02-09 | 2023-05-12 |
| val | 5723 | 97 | 11 | 2023-08-11 | 2024-09-20 |
| test | 5820 | 97 | 11 | 2024-12-20 | 2026-02-06 |

티커별 총 sample 수 분포:

```json
{
  "count": 97,
  "mean": 394.0,
  "std": 0.0,
  "p01": 394.0,
  "p05": 394.0,
  "p10": 394.0,
  "p50": 394.0,
  "p90": 394.0,
  "p95": 394.0,
  "p99": 394.0,
  "min": 394.0,
  "max": 394.0
}
```

sector별 split sample 수:

```json
{
  "Basic Materials": {
    "train": 275,
    "val": 59,
    "test": 60
  },
  "Communication Services": {
    "train": 550,
    "val": 118,
    "test": 120
  },
  "Consumer Cyclical": {
    "train": 1100,
    "val": 236,
    "test": 240
  },
  "Consumer Defensive": {
    "train": 825,
    "val": 177,
    "test": 180
  },
  "Energy": {
    "train": 825,
    "val": 177,
    "test": 180
  },
  "Financial Services": {
    "train": 3300,
    "val": 708,
    "test": 720
  },
  "Healthcare": {
    "train": 2200,
    "val": 472,
    "test": 480
  },
  "Industrials": {
    "train": 1925,
    "val": 413,
    "test": 420
  },
  "Real Estate": {
    "train": 275,
    "val": 59,
    "test": 60
  },
  "Technology": {
    "train": 3575,
    "val": 767,
    "test": 780
  },
  "UNKNOWN": {
    "train": 11825,
    "val": 2537,
    "test": 2580
  }
}
```

## 4. Target 분포 차이

아래 표는 h4 terminal raw future return 기준이다. 모델은 h1~h4 벡터를 쓰므로 metrics JSON에는 h1~h4 flatten 분포도 같이 기록했다.

| split | rows | asof | h4_mean | h4_p05 | h4_p50 | h4_p95 | h4 severe <= -5% | h1~h4 severe rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 26675 | 2018-02-09 ~ 2023-05-12 | 0.013831 | -0.117570 | 0.014150 | 0.148785 | 5088 | 0.146261 |
| val | 5723 | 2023-08-11 ~ 2024-09-20 | 0.019641 | -0.090235 | 0.017426 | 0.140599 | 800 | 0.103966 |
| test | 5820 | 2024-12-20 ~ 2026-02-06 | 0.013073 | -0.130232 | 0.011664 | 0.157858 | 1171 | 0.150687 |

판단:
- severe downside는 train/val/test 모두 존재한다.
- test는 2024-12-20 이후라 2020 crash나 2022 bear를 직접 포함하지 않는다.
- 따라서 test tail 성능이 과거 crisis tail 일반화까지 보장한다고 해석하면 안 된다.

## 5. Volatility regime 분포

| split | high abs-return | low abs-return | high atr | low atr | stress mean | calm mean |
| --- | --- | --- | --- | --- | --- | --- |
| train | 5579 | 5238 | 5937 | 4968 | 0.000000 | 1.000000 |
| val | 861 | 1253 | 486 | 1804 | 0.000000 | 1.000000 |
| test | 1204 | 1153 | 1221 | 872 | 0.000000 | 1.000000 |

split마다 high/low volatility sample은 충분히 있다. 하지만 2020/2022 같은 큰 구조적 stress는 train에만 존재하므로, 제품 후보 판단에는 regime별 metric을 별도로 붙여야 한다.

## 6. 역사적 regime의 split 위치

```json
{
  "2020_crash": {
    "calendar_range": {
      "start": "2020-02-21",
      "end": "2020-04-10"
    },
    "sample_rows": 776,
    "split_counts": {
      "train": 776
    },
    "ticker_count": 97,
    "asof_min": "2020-02-21",
    "asof_max": "2020-04-10"
  },
  "2022_bear": {
    "calendar_range": {
      "start": "2022-01-07",
      "end": "2022-12-30"
    },
    "sample_rows": 5044,
    "split_counts": {
      "train": 5044
    },
    "ticker_count": 97,
    "asof_min": "2022-01-07",
    "asof_max": "2022-12-30"
  },
  "2023_2024_bull": {
    "calendar_range": {
      "start": "2023-01-06",
      "end": "2024-12-27"
    },
    "sample_rows": 7760,
    "split_counts": {
      "test": 194,
      "train": 1843,
      "val": 5723
    },
    "ticker_count": 97,
    "asof_min": "2023-01-06",
    "asof_max": "2024-12-27"
  }
}
```

해석:
- 2020 crash: train
- 2022 bear: train
- 2023~2024 bull: train 일부, val 대부분, test 극소 구간
- test는 주로 2025~2026 최근 regime이다.

## 7. Overlapping window에 따른 과대평가

```json
{
  "nominal_split_samples": {
    "test": 5820,
    "train": 26675,
    "val": 5723
  },
  "nominal_total_samples": 38218,
  "adjacent_input_window_overlap_ratio": 0.9903846153846154,
  "adjacent_horizon_label_overlap_ratio": 0.75,
  "label_nonoverlap_effective_sample_estimate": 9555,
  "label_effective_ratio": 0.25001308284054635,
  "sequence_block_effective_sample_estimate": 367.4807692307692,
  "sequence_block_effective_ratio": 0.009615384615384616,
  "interpretation": "nominal rows는 교차단면 ticker 수와 overlapping weekly windows 때문에 독립 표본 수보다 크게 보인다."
}
```

`seq_len=104`인 1W 인접 입력 window는 103주를 공유한다. 인접 label h1~h4도 3/4가 겹친다. 따라서 nominal sample `38218`개를 독립 표본처럼 해석하면 과대평가다.

## 8. CP112~CP119 test 반복 사용 감사

| metrics file | test/candidate refs | run estimate | reused refs |
| --- | --- | --- | --- |
| cp112_bm_1w_band_smoke_metrics.json | 1 | 1 | 0 |
| cp112_lm_1w_line_smoke_metrics.json | 1 | 1 | 0 |
| cp113_bm_1w_band_limited_validation_metrics.json | 1 | 1 | 0 |
| cp113_lm_1w_line_rescue_metrics.json | 3 | 3 | 0 |
| cp114_bm_1w_band_candidate_expansion_metrics.json | 5 | 4 | 1 |
| cp114_lm_1w_line_candidate_expansion_metrics.json | 4 | 4 | 0 |
| cp118_bm_1w_band_feature_target_audit_metrics.json | 1 | 0 | 0 |
| cp119_bm_1w_band_feature_group_experiment_metrics.json | 6 | 6 | 0 |

요약:
- test/candidate 참조 총계: `22`
- 신규 또는 재실행 model run 추정: `20`
- unique run_id 수: `20`
- bias 등급: `HIGH`

판단:
- CP112~CP119는 smoke/제한 검증 목적이었기 때문에 test 확인 자체는 이해 가능하다.
- 하지만 이미 test가 후보 좁히기에 반복 노출됐다.
- 다음 1W 후보 저장 CP에서 test가 좋은 후보만 고르는 방식은 제품 후보 금지로 봐야 한다.

## 9. 과적합 방지 제안

필수 가드:
1. test set은 최종 후보 확인용으로만 쓴다.
2. 중간 실험과 후보 narrowing은 validation 중심으로 판단한다.
3. candidate registry에 `validation_stability`, `seed_stability`, `regime_stability`, `test_exposure_count`를 추가한다.
4. 후보가 test에서만 좋고 validation/regime 안정성이 없으면 제품 후보 금지다.
5. 모든 1W 보고서에 train/val/test target 분포와 regime별 metric을 필수로 붙인다.

추가 권장:
- anchored split 2개 이상 또는 rolling time split 추가
- sector holdout 또는 ticker group holdout 소규모 검증
- 2020 crash, 2022 bear, 2023~2024 bull, 2025~2026 recent를 별도 regime bucket으로 평가
- final test는 한 번만 열고, 그 전 후보 선택은 val과 rolling validation으로 제한

## 10. 다음 CP 권장

1. 1W 후보 재현 CP는 validation stability 중심으로 설계한다.
2. CP119 추천 band 후보는 seed 2~3개에서 val coverage, interval, downside_width_ic 안정성을 먼저 본다.
3. line 후보도 test IC만 보지 말고 val IC, tail recall, fee-adjusted proxy가 같은 방향인지 확인한다.
4. 제품 저장 전 `candidate_registry`에 test 노출 횟수와 split stability를 기록한다.

## 11. 금지 작업 확인

| 금지 항목 | 발생 |
|---|---|
| 모델 학습 | false |
| DB write | false |
| inference 저장 | false |
| Supabase 대량 read | false |
| 프론트 수정 | false |
| EODHD 호출 | false |

## 12. 읽기 전용 근거

- local price: `C:\Users\user\lens\data\parquet\price_data_yfinance_1W.parquet`
- local indicators: `C:\Users\user\lens\data\parquet\indicators_yfinance_1W.parquet`
- stock info: `C:\Users\user\lens\data\parquet\stock_info.parquet`
- split 근거: `ai/splits.py`, `ai/preprocessing.py`
- CP112~CP119 metrics: `docs/cp112*`, `docs/cp113*`, `docs/cp114*`, `docs/cp118*`, `docs/cp119*`
