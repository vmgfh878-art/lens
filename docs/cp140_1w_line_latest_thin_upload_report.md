# CP140-D 1W line latest-only thin upload 보고서

## 1. 요약

판정은 **PASS**이다.

1W line latest-only prediction 저장이 완료됐고 API/repository 조회가 통과했다. band/composite/bulk 저장은 발생하지 않았다.

## 2. 실행 범위

- run_id: `patchtst-1W-fe7f05a84c93`
- timeframe: `1W`
- horizon: `4`
- role: `line_model`
- feature_set: `price_volatility_volume`
- source/provider: yfinance local parquet

## 3. local 1W snapshot

| snapshot | rows | tickers | latest date | duplicate | source | provider |
|---|---:|---:|---|---:|---|---|
| price 1W | 59200 | 100 | 2026-05-01 | 0 | ['yfinance'] | ['yfinance'] |
| indicators 1W | 53300 | 100 | 2026-05-01 | 0 | ['yfinance'] | ['yfinance'] |

## 4. inference 결과

| 항목 | 값 |
|---|---|
| usable ticker count | 97 |
| skipped tickers | {} |
| asof dates | ['2026-05-01'] |
| forecast date lengths | [4] |
| line series lengths | [4] |
| n_features | 11 |

## 5. latest-only 저장 결과

DB 스키마상 `lower_band_series`와 `upper_band_series`는 not-null이라 line-only row에도 값을 넣어야 한다. CP140은 의미 있는 1W band를 저장하지 않기 위해 두 필드를 `line_series`와 동일한 퇴화 구간으로 저장하고, meta에 `band_saved_in_cp140=False`, `band_fields_policy=schema_required_degenerate_equal_to_line`을 남긴다.

| 항목 | 값 |
|---|---|
| attempted | True |
| predictions attempted rows | 97 |
| evaluations attempted rows | 0 |
| prediction row delta | 97 |
| evaluation row delta | 0 |
| before predictions | {'rows': 0, 'ticker_count': 0, 'asof_dates': [], 'asof_date_count': 0, 'layer_counts': {}, 'composite_count': 0} |
| after predictions | {'rows': 97, 'ticker_count': 97, 'asof_dates': ['2026-05-01'], 'asof_date_count': 1, 'layer_counts': {'line': 97}, 'composite_count': 0} |
| before evaluations | {'rows': 0, 'ticker_count': 0, 'asof_dates': [], 'asof_date_count': 0, 'layer_counts': {}, 'composite_count': 0} |
| after evaluations | {'rows': 0, 'ticker_count': 0, 'asof_dates': [], 'asof_date_count': 0, 'layer_counts': {}, 'composite_count': 0} |

## 6. API/repository 조회

| ticker | found | asof_date | timeframe | horizon | layer | role | storage_contract | line length | meaningful band series |
|---|---:|---|---|---:|---|---|---|---:|---:|
| AAPL | True | 2026-05-01 | 1W | 4 | line | line_model | product_latest_only | 4 | False |
| MSFT | True | 2026-05-01 | 1W | 4 | line | line_model | product_latest_only | 4 | False |
| NVDA | True | 2026-05-01 | 1W | 4 | line | line_model | product_latest_only | 4 | False |

## 7. 금지 작업 확인

- 1W band inference/storage: False
- composite 저장: False
- bulk evaluation save: False
- 모델 학습: False
- W&B: False
- yfinance live fetch: False
- EODHD 호출: False
- Supabase price_data/indicators 대량 read: False
- 프론트 수정: False
