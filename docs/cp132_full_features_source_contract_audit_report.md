# CP132-DG full_features source/provider 계약 감사 보고서

## 1. Executive Summary

판정은 **WARN**이다.

CP125의 1W BM 최종 후보 `cnn_full_q10_direct_lower_guard_w1p5`는 `full_features` 기반이지만, 현재 `data/parquet/indicators_yfinance_1W.parquet`에서 macro / market breadth / fundamentals 계열은 실제 정보 신호가 아니라 전부 0-fill 상태다. `has_macro`, `has_breadth`, `has_fundamentals` true rate도 모두 0%다.

따라서 이 후보를 1W BM 제품 후보로 저장하는 것은 가능하다. 다만 제품/발표 문구에서 “시장/섹터/펀더멘털 정보를 적극 활용했다”고 설명하면 안 된다. 현재 해석은 “가격/기술 지표 중심의 1W band 후보이며, full_features preset을 사용했지만 외부 context 계열은 missing/zero 상태였다”가 맞다.

## 2. 감사 대상과 근거

- 후보: `cnn_full_q10_direct_lower_guard_w1p5`
- timeframe/horizon: `1W` / `4`
- model_family: `cnn_lstm`
- feature_set: `full_features`
- raw run ids: `cnn_lstm-1W-022d19d7af97, cnn_lstm-1W-82bd1d4d654a`
- local indicator snapshot: `data\parquet\indicators_yfinance_1W.parquet`
- local price snapshot: `data\parquet\price_data_yfinance_1W.parquet`

코드 근거:

- `MODEL_FEATURE_COLUMNS = SOURCE_FEATURE_COLUMNS + CALENDAR_FEATURE_COLUMNS`: `ai/preprocessing.py:49`
- `MODEL_N_FEATURES = len(MODEL_FEATURE_COLUMNS)`: `ai/preprocessing.py:51`
- source feature 정의: `backend/app/services/feature_svc.py:15`, `backend/app/services/feature_svc.py:34`, `backend/app/services/feature_svc.py:39`, `backend/app/services/feature_svc.py:67`
- price/technical feature set: `backend/app/services/feature_svc.py:97`
- macro/breadth 0-fill flag 처리: `backend/app/services/feature_svc.py:423` - `backend/app/services/feature_svc.py:435`
- fundamentals `filing_date` backward merge 및 0-fill: `backend/app/services/feature_svc.py:439` - `backend/app/services/feature_svc.py:502`
- 1W/1M context resample last: `backend/app/services/feature_svc.py:299`
- indicators source/provider 저장: `backend/collector/jobs/compute_indicators.py:241` - `backend/collector/jobs/compute_indicators.py:242`
- source-aware indicators unique key: `backend/db/schema.sql:147`, `backend/db/schema.sql:187`
- macro/breadth/fundamentals upstream source/provider 부재: `backend/db/schema.sql:59`, `backend/db/schema.sql:102`, `backend/db/schema.sql:124`
- sector_returns source/provider 부재 및 현재 모델 미사용: `backend/db/schema.sql:110`, `backend/db/schema.sql:119`

## 3. full_features vs price_volatility_volume

| 구분 | 컬럼 수 | 컬럼 |
|---|---:|---|
| price_volatility_volume | 11 | log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position |
| full_features | 36 | log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position, us10y, yield_spread, vix_close, credit_spread_hy, nh_nl_index, ma200_pct, regime_calm, regime_neutral, regime_stress, revenue, net_income, equity, eps, roe, debt_ratio, has_macro, has_breadth, has_fundamentals, day_of_week_sin, day_of_week_cos, month_sin, month_cos, is_month_end, is_quarter_end, is_opex_friday |
| full_features 추가분 | 25 | us10y, yield_spread, vix_close, credit_spread_hy, nh_nl_index, ma200_pct, regime_calm, regime_neutral, regime_stress, revenue, net_income, equity, eps, roe, debt_ratio, has_macro, has_breadth, has_fundamentals, day_of_week_sin, day_of_week_cos, month_sin, month_cos, is_month_end, is_quarter_end, is_opex_friday |

`sector_returns`는 현재 `MODEL_FEATURE_COLUMNS` 36개에 포함되지 않는다. 즉 CP125의 `full_features` 후보는 현재 코드 계약상 sector return 값을 직접 보지 않았다.

## 4. local 1W snapshot 상태

| 항목 | 값 |
|---|---:|
| indicators rows | 53300 |
| indicators tickers | 100 |
| indicators date range | 2016-02-19 ~ 2026-05-01 |
| indicators source counts | {"yfinance":53300} |
| indicators provider counts | {"yfinance":53300} |
| indicators duplicate ticker/timeframe/date/source | 0 |
| price rows | 59200 |
| price tickers | 100 |
| price date range | 2015-01-02 ~ 2026-05-01 |
| price source counts | {"yfinance":59200} |
| price provider counts | {"yfinance":59200} |
| price duplicate ticker/date/source | 0 |

1W local indicators는 `source=yfinance`, `provider=yfinance`로 고정되어 있다. 따라서 현재 모델 입력 parquet 자체에서 EODHD/yfinance 직접 혼합 증거는 없다.

## 5. 0-fill / missing flag 감사

| 계열 | 컬럼 | 결과 |
|---|---|---|
| macro | us10y, yield_spread, vix_close, credit_spread_hy | 전부 zero_rate=1.0, `has_macro` true rate=0.0 |
| market_breadth | nh_nl_index, ma200_pct | 전부 zero_rate=1.0, `has_breadth` true rate=0.0 |
| fundamentals | revenue, net_income, equity, eps, roe, debt_ratio | 전부 zero_rate=1.0, `has_fundamentals` true rate=0.0 |
| regime | regime_calm, regime_neutral, regime_stress | `regime_calm`은 상수 1.0, neutral/stress는 상수 0.0 |

중요한 해석은 두 가지다.

1. 현재 full_features 성능은 실질적으로 macro / breadth / fundamentals 신호를 사용한 결과라고 보기 어렵다.
2. `vix_close`가 missing 후 0-fill되면서 regime이 항상 calm으로 만들어지는 artifact가 있다. 이 값은 상수라 성능을 크게 만든 신호일 가능성은 낮지만, 의미 있는 시장 regime feature로 설명하면 틀린 설명이다.

## 6. source/provider 및 asof 계약 위험

| 데이터 계열 | 현재 계약 | 위험 | 판단 |
|---|---|---|---|
| price_data | source/provider/policy, unique `(ticker,date,source)` | 낮음 | yfinance local 1W 기준 정상 |
| indicators | source/provider, unique `(ticker,timeframe,date,source)` | 낮음 | yfinance local 1W 기준 정상 |
| macroeconomic_indicators | source/provider/release_date/asof 없음 | 중간 | 현재는 0-fill이라 직접 신호 없음 |
| market_breadth | source/provider/asof 없음 | 중간 | 현재는 0-fill이라 직접 신호 없음 |
| sector_returns | source/provider 없음 | 중간 | 현재 모델 feature에는 미포함 |
| company_fundamentals | source/provider 없음, filing_date 있음 | 중간 | backward merge는 있으나 현재는 0-fill |

## 7. lookahead 감사

- fundamentals는 `filing_date` 기준 `merge_asof(direction="backward")`이므로 코드상 직접적인 미래 filing_date 사용은 막고 있다.
- 그러나 `company_fundamentals` 자체에는 source/provider 및 provider별 확정 시각 계약이 없다. 향후 실제 값을 넣으면 provider 차이와 restatement 정책을 분리해야 한다.
- macro/breadth는 1W resample에서 `last()`를 사용한다. release/asof 계약이 없는 상태에서 실제 값을 넣으면 같은 주의 사후 확정값을 주간 feature에 당겨 쓸 위험이 있다.
- sector_returns는 현재 36개 모델 feature에 없으므로 CP125 후보에는 직접 영향이 없다. 단, 향후 feature 승격 전 source/provider/asof 계약이 필요하다.

## 8. 1W BM 후보 영향 판단

`cnn_full_q10_direct_lower_guard_w1p5`를 full_features라는 이름만 보고 “시장/섹터/펀더멘털까지 활용한 후보”로 해석하면 안 된다. 현재 local 1W snapshot 기준으로 이 후보가 본 유효한 추가 정보는 calendar feature와 missing/constant context 정도다.

제품 후보 저장은 가능하지만 조건이 필요하다.

- 설명 문구: “가격/기술 지표 중심 1W BM 후보”로 제한한다.
- 금지 문구: “macro/breadth/fundamental alpha를 활용했다”, “시장 regime을 학습했다”.
- 제품 저장 전 P1: `regime_*`이 missing VIX에서 상수로 생기는 정책을 막거나 문서화한다.
- Phase 1.5: macro/breadth/fundamentals/sector_returns source/provider/asof 계약을 별도로 수리한다.

PVV 후보로 무조건 되돌릴 필요는 없다. CP125에서 PVV 대안은 lower breach 쪽에서 밀렸고, 현재 full_features의 위험 컬럼들이 실제 신호로 작동하지 않았기 때문이다. 다만 해석 리스크를 피해야 하면 PVV 후보가 더 보수적인 선택이다.

## 9. 리스크 분류

| 등급 | 항목 | 조치 |
|---|---|---|
| P0 | 없음 | 현재 local 1W model input에서 source 혼합이나 명백한 lookahead 사용 증거 없음 |
| P1 | `regime_*` missing artifact | 제품 후보 설명 전 “시장 regime 신호”라고 부르지 말고, 다음 CP에서 missing gate 또는 제외 검토 |
| P1 | full_features 명칭 오해 | 제품 저장 메타/보고서에 “external context inactive” 문구 추가 |
| P2 | macro/breadth/fundamentals source/provider/asof 부재 | Phase 1.5 데이터 계약 수리 필요 |
| P2 | sector_returns 향후 승격 위험 | 현재 미사용이나 source/provider/asof 계약 없이 feature 승격 금지 |
| P3 | calendar feature 포함 | 정상. source/provider 계약 불필요 |

## 10. 최종 판정

**WARN: 제품 후보 저장 가능, 단 설명 제약 필요.**

`cnn_full_q10_direct_lower_guard_w1p5`는 1W BM 후보로 저장해도 된다. 하지만 저장/발표/제품 문구에서는 다음처럼 제한해야 한다.

> 이 후보는 full_features preset을 사용했지만, 현재 yfinance local 1W snapshot에서는 macro, breadth, fundamentals 값이 모두 missing/0-fill 상태였다. 따라서 성능은 시장/섹터/펀더멘털 정보 활용 결과가 아니라, 가격/기술 지표와 calendar/constant context를 포함한 결과로 해석한다.

## 11. 실행한 읽기 전용 명령 목록

- `Select-String -Path ai\preprocessing.py -Pattern ...`
- `Select-String -Path backend\app\services\feature_svc.py -Pattern ...`
- `Select-String -Path backend\db\schema.sql -Pattern ...`
- `Select-String -Path backend\collector\jobs\compute_indicators.py -Pattern ...`
- `Get-Content backend\db\schema.sql | Select-Object ...`
- `Get-Content backend\collector\jobs\compute_indicators.py | Select-Object ...`
- local parquet read: `data/parquet/indicators_yfinance_1W.parquet`, `data/parquet/price_data_yfinance_1W.parquet`
- local docs read: `docs/cp125_bm_1w_band_final_registry.json`

## 12. 금지 작업 미발생 확인

모델 학습, inference 실행, save-run, DB write/delete/update, Supabase price_data/indicators 대량 read, 프론트 수정, EODHD 호출, fake data 생성은 수행하지 않았다.
