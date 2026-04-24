# CP2.6 완료

## 1. 컨텍스트 테이블 분포
- `macroeconomic_indicators`
  - `min_date=2015-01-01`
  - `max_date=2026-04-23`
  - `row_count=4131`
  - null 비율
    - `us10y=0.3157`
    - `yield_spread=0.3157`
    - `vix_close=0.3062`
    - `credit_spread_hy=0.8095`
- `market_breadth`
  - `min_date=2016-04-22`
  - `max_date=2026-04-22`
  - `row_count=2514`
  - null 비율
    - `nh_nl_index=0.0000`
    - `ma200_pct=0.0000`
- `price_data`의 `per/pbr` null 비율 (`AAPL`)
  - `per=0.9870`
  - `pbr=0.9870`

## 2. AAPL base feature 컬럼별 first_non_null 날짜
- `log_return`: `2015-01-05`
- `open_ratio`: `2015-01-05`
- `high_ratio`: `2015-01-05`
- `low_ratio`: `2015-01-05`
- `vol_change`: `2015-01-05`
- `ma_5_ratio`: `2015-01-08`
- `ma_20_ratio`: `2015-01-30`
- `ma_60_ratio`: `2015-03-30`
- `rsi`: `2015-01-23`
- `macd_ratio`: `2015-01-02`
- `bb_position`: `2015-01-30`
- `us10y`: `2015-01-02`
- `yield_spread`: `2015-01-02`
- `vix_close`: `2015-01-02`
- `credit_spread_hy`: `2023-04-24`
- `nh_nl_index`: `2016-04-22`
- `ma200_pct`: `2016-04-22`
- `per/pbr`: `2026-03-03` (`_BASE_FEATURE_COLUMNS`에는 포함되지 않아 이번 병목의 직접 원인은 아님)

## 3. 원인 결정
- 분기: `복합(A+B)`
- 근거
  - `macroeconomic_indicators`는 테이블 자체는 2015년부터 있지만 `credit_spread_hy`의 첫 실값이 `2023-04-24`라서 exact/forward context merge 후 과거 구간을 비워버렸다.
  - `market_breadth`는 `2016-04-22`부터 시작해서 그 이전 구간을 비워버렸다.
  - `per/pbr`도 늦게 채워지지만 현재 `_BASE_FEATURE_COLUMNS`에는 포함되지 않아 이번 `N1 -> N2` 급감의 직접 원인은 아니다.

## 4. 적용한 조치
- [ ] macro 백필 실행: `N`
  - 이유: 테이블 길이 자체는 충분했고, 이번 CP에서는 `has_macro` 플래그와 `0` 대체로 과거 구간 탈락을 먼저 막는 쪽이 더 직접적이었다.
- [ ] breadth 백필 실행: `N`
  - 이유: `market_breadth` 시작일이 `2016-04-22`로 제한되어 있지만, 이번 CP에서는 `has_breadth` 플래그와 `0` 대체로 모델 입력 생존성을 먼저 회복했다.
- [x] `feature_svc`에 `has_macro` 플래그 추가: `Y`
- [x] `feature_svc`에 `has_breadth` 플래그 추가: `Y`
- [ ] `per/pbr`용 `price metrics` 플래그 추가: `N`
  - 이유: 이번 원인 분석에서 직접 병목이 아니어서 범위에 넣지 않았다.
- [ ] `indicators` recompute full range 완료: `N`
  - 상태: `1D`, `1W`는 실제 복구를 확인했고, `1M`까지 포함한 전체 청크 재계산은 후반부 `RemoteProtocolError`로 종료됐다.

## 5. 결과 수치
- `AAPL 1D row_count`: `2784`
  - 이전 `753`
  - 목표 `2200`
  - 실제 `2784`
- `AAPL 1W row_count`: `532`
  - 이전 `157`
  - 목표 `450`
  - 실제 `532`
- 전체 유니버스 `1D row_count p10/p50/p90`: `1730.2 / 2783 / 2784.0`
- 전체 유니버스 `1W row_count p10/p50/p90`: `314.4 / 532.0 / 532.0`
- `indicators 1D min_date p10`: `2015-03-27`
- `has_macro=1` time-weighted 비율
  - `1D=0.9837`
  - `1W=0.9822`
- `has_breadth=1` time-weighted 비율
  - `1D=0.8933`
  - `1W=0.9664`
- `has_fundamentals=1` time-weighted 비율
  - `1D=0.0624`
  - `1W=0.0693`
- `FEATURE_COLUMNS count`: `29`
- 기존 테스트 + 추가 테스트 all green: `Y`
- 실행한 테스트 수: `21`
- 신규/수정 테스트: `5건`

## 6. CP3 준비 상태
- `26` ticker 제외 대상 확정: `Y`
- `seq_len=252 (1D)` 가능한 ticker 수: `476 / 477`
- `seq_len=104 (1W)` 가능한 ticker 수: `472 / 477`

## 메모
- 이번 CP의 핵심 복구는 `feature_svc.build_features()`에서 컨텍스트 결측을 `dropna` 원인으로 쓰지 않도록 바꾼 것이다.
- `compute_indicators.py`에는 동일 키 중복 upsert로 Supabase가 실패하지 않도록 `drop_duplicates(["ticker", "timeframe", "date"])` 보호를 추가했다.
- `1M`까지 포함한 완전한 full-range 재계산은 아직 닫히지 않았으므로, 다음 단계에서 안정적인 청크 재실행이 한 번 더 필요하다.
