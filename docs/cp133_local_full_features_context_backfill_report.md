# CP133-DG local full_features context backfill 보고서

## 1. Executive Summary

판정은 **WARN**이다.

CP132에서 0-fill이었던 full_features context 중 macro, market_breadth, fundamentals를 Supabase가 아니라 local parquet 기준으로 채웠다. 1D/1W indicators도 context를 반영해 재생성했고, feature NaN/Inf와 duplicate key는 0이다.

다만 credit_spread_hy는 FRED coverage가 부분적이고, fundamentals는 SEC EDGAR filing_date gate 때문에 feature true rate가 1D 0.05455816204010681, 1W 0.06030018761726079 수준이다. market_breadth도 전체 미국 시장 breadth가 아니라 local 100티커 universe breadth다. 그래서 full_features 재실험은 가능하지만 “완전한 시장/펀더멘털 context”라고 말하면 안 된다.

## 2. 생성/갱신한 local parquet

| 파일 | 역할 |
|---|---|
| data/parquet/context/macroeconomic_indicators.parquet | FRED 기반 macro context |
| data/parquet/context/market_breadth.parquet | local yfinance 100티커 breadth |
| data/parquet/context/company_fundamentals.parquet | SEC EDGAR filing_date 기반 fundamentals |
| data/parquet/context/sector_returns.parquet | local yfinance sector return, 현재 모델 feature 미포함 |
| data/parquet/indicators_yfinance_1D.parquet | context 반영 후 재생성 |
| data/parquet/indicators_yfinance_1W.parquet | context 반영 후 재생성 |

기존 1D/1W indicator parquet는 data/parquet/backups 아래에 백업했다.

## 3. context source 결과

| 계열 | 상태 | rows | date range | 비고 |
|---|---|---:|---|---|
| macro | PASS | 13036 | 1976-06-01 ~ 2026-05-04 | provider=fred |
| market_breadth | PASS | 2650 | 2015-10-16 ~ 2026-05-01 | local 100티커 breadth |
| fundamentals | PASS | 1135 | 2014-12-31 ~ 2026-04-03 | provider=sec_edgar, 98/100 ticker 성공 |
| sector_returns | PASS | 31330 | 2015-01-05 ~ 2026-05-04 | 현재 MODEL_FEATURE_COLUMNS에는 미포함 |

macro non-null rate:

| column | non-null rate |
|---|---:|
| us10y | 0.2174746854863455 |
| yield_spread | 0.2174746854863455 |
| vix_close | 0.7040503221847192 |
| credit_spread_hy | 0.06014114759128567 |

## 4. 1D/1W feature sanity

| 항목 | 1D | 1W |
|---|---:|---:|
| rows | 279005 | 53300 |
| tickers | 100 | 100 |
| date range | 2015-03-30 ~ 2026-05-04 | 2016-02-19 ~ 2026-05-01 |
| feature non-finite count | 0 | 0 |
| duplicate key count | 0 | 0 |
| has_macro true rate | 1 | 1 |
| has_breadth true rate | 0.9498216877833731 | 1 |
| has_fundamentals true rate | 0.05455816204010681 | 0.06030018761726079 |

context 반영 후 regime도 더 이상 constant calm이 아니다.

| column | 1D non-zero rate | 1W non-zero rate |
|---|---:|---:|
| regime_calm | 0.36594326266554367 | 0.3714821763602252 |
| regime_neutral | 0.49391587964373396 | 0.4934333958724203 |
| regime_stress | 0.14014085769072238 | 0.13508442776735463 |
| nh_nl_index | 0.8724037203634344 | 0.924953095684803 |
| ma200_pct | 0.9498216877833731 | 1 |
| revenue | 0.05455816204010677 | 0.06030018761726075 |

## 5. lookahead 방지 계약

- macro: FRED observation date만 사용한다. 1W는 완료된 W-FRI bucket에서 last 값을 사용하므로, 주간 asof가 금요일 종가 이후라는 전제에서 안전하다.
- market_breadth: 같은 날짜까지 존재하는 local yfinance adjusted close만 사용한다.
- fundamentals: filing_date 기준 merge_asof backward만 허용한다. report period date가 아니라 filing_date가 feature asof gate다.
- sector_returns: 현재 모델 feature가 아니며, 향후 승격 전 source/provider/asof 계약을 별도 모델 feature 계약에 추가해야 한다.

## 6. cache/hash 영향

context 반영 전후 indicator value checksum이 모두 바뀌었다. CP96 이후 feature fingerprint는 indicator value checksum을 포함하므로, 기존 feature cache를 그대로 재사용하면 안 된다.

| timeframe | before checksum | after checksum | changed |
|---|---|---|---|
| 1D | abd92ff690cfc2db | 08158c217c672a6e | true |
| 1W | fe876245e7727b19 | b83cb84767f8c357 | true |

## 7. 최종 판단

full_features 재실험은 가능하다. 다만 결과 해석은 다음 제약을 붙여야 한다.

- macro는 FRED 기반으로 일부 장기 coverage 차이가 있다.
- breadth는 전체 시장 breadth가 아니라 100티커 local universe breadth다.
- fundamentals는 SEC filing_date gate 때문에 최근 구간 일부에만 활성화된다.
- sector_returns는 생성했지만 아직 36개 모델 feature에는 포함되지 않는다.

## 8. 금지 작업 미발생 확인

Supabase 대량 read/write, DB write/delete/update, 모델 학습, inference 실행, save-run, 프론트 수정, EODHD 호출, fake data 생성은 수행하지 않았다.
