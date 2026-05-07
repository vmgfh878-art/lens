# 무료 주식 데이터 소스 전환 감사 보고서

작성일: 2026-05-03

## 1. 요약

결론부터 말하면, EODHD를 지금 바로 무료 소스로 대체하면 안 된다. Lens의 핵심 가격 계약은 단순한 종가 수집이 아니라 `raw OHLC + adjusted_close`를 함께 저장하고, 이후 `adj_factor = adjusted_close / close`로 adjusted OHLC를 재구성한 뒤 1D/1W/1M 지표와 학습 피처를 만든다는 구조다. 따라서 무료 소스 전환의 성패는 "가격이 대충 비슷한가"가 아니라 "분할, 배당, raw/adjusted 의미, 날짜, 거래량이 EODHD 대비 충분히 같은가"로 판단해야 한다.

현재 코드 기준으로 가격 수집의 1차 공급자는 EODHD다. `backend/collector/sources/eodhd_prices.py:24`에서 EODHD EOD 엔드포인트를 호출하고, `backend/collector/jobs/sync_prices.py:239`에서 `price_data`에 upsert한다. yfinance fallback은 이미 코드에 있지만 `allow_yahoo_fallback=False`가 기본이고, daily/backfill 파이프라인은 `EODHD_API_KEY`가 없으면 시작 자체를 막는다. 즉 "fallback 구현은 있음"과 "운영 대체 가능"은 다르다.

추천 순서는 다음이다.

1. 운영 `price_data`는 당분간 EODHD를 기준값으로 유지한다.
2. 무료 후보는 adapter 방식으로 shadow 비교부터 한다.
3. 20티커 샘플에서 yfinance, Stooq, Alpha Vantage를 EODHD와 비교한다.
4. adjusted_close, split/dividend 반영, open/high/low ratio, volume 차이가 기준을 넘으면 full 전환을 금지한다.
5. stock_info와 fundamentals는 Phase 1 핵심 병목이 아니므로 가격 전환보다 우선순위를 낮춘다.

무료 후보 중 코드 적합성은 yfinance가 가장 높다. 다만 yfinance 공식 문서는 Yahoo Finance API가 개인 사용 목적이며 yfinance가 Yahoo의 공식 제품이 아니라고 고지한다. 따라서 제품 운영이나 발표 데모의 장기 안정성을 맡기기에는 법적, 운영 리스크가 있다. Stooq는 대용량 무료 historical data가 있지만 adjusted_close와 corporate action 계약이 Lens에 충분히 명문화되어 있지 않다. Alpha Vantage는 adjusted daily 계약은 가장 명확하지만 무료 요청량이 25회/일 수준이고 daily adjusted의 장기 full 출력도 premium 제약이 있어 473티커 운영 cron에는 맞지 않는다. Nasdaq Data Link의 무료 WIKI 가격 데이터는 2018년 이후 갱신되지 않아 현재 Lens 가격 수집 대체재가 아니다.

## 2. 현재 EODHD 의존 지점

### 가격 수집

- `backend/collector/sources/eodhd_prices.py:14-21`: Lens ticker를 EODHD 심볼로 바꾼다. 일반 미국 주식은 `.US`를 붙이고, `BRK.B` 같은 점 표기는 `BRK-B.US`류로 바뀔 수 있다. 무료 소스 전환 시 심볼 매핑 차이가 가장 먼저 터질 수 있는 지점이다.
- `backend/collector/sources/eodhd_prices.py:24-75`: EODHD `/api/eod/{symbol}` 호출 결과를 `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Amount`로 표준화한다. `adjusted_close`가 없으면 `Adj Close = Close`로 보정한다.
- `backend/collector/sources/eodhd_prices.py:78-112`: `fetch_ohlcv()`는 EODHD key가 있으면 EODHD를 먼저 쓰고, key가 없거나 결과가 비었을 때만 yfinance fallback을 쓴다. yfinance 호출은 `auto_adjust=False`라서 raw OHLC와 `Adj Close`를 동시에 받으려는 의도와 맞는다.
- `backend/collector/jobs/sync_prices.py:41-103`: 수집 프레임의 OHLC 정합성, 거래량, 극단 수익률을 검증한다. `Adj Close`가 있으면 raw close jump와 adjusted close jump를 함께 본다.
- `backend/collector/jobs/sync_prices.py:106-134`: DB 저장 레코드에 `open`, `high`, `low`, `close`, `adjusted_close`, `volume`, `amount`, `per`, `pbr`를 넣는다.
- `backend/collector/jobs/sync_prices.py:195-239`: ticker별 가격을 받아 검증 후 `price_data`에 `ticker,date` 기준으로 upsert한다.

### stock_info

stock_info는 현재 EODHD 의존이 아니다.

- `backend/collector/sources/yahoo_stock_info.py:12-46`: FMP profile을 먼저 조회한다.
- `backend/collector/sources/yahoo_stock_info.py:49-80`: FMP가 없거나 실패하면 yfinance `Ticker.info`에서 `sector`, `industry`, `marketCap`을 가져온다.
- `backend/collector/jobs/sync_stock_info.py:12-23`: 정보가 없더라도 ticker placeholder를 만든다. 따라서 stock_info는 가격 수집보다 무료 전환 리스크가 낮다.

### split, dividend, adjusted_close

현재 DB에는 별도 split/dividend 테이블이 없다. `backend/db/schema.sql:19-34`의 `price_data`는 raw OHLC와 `adjusted_close`만 저장한다. 즉 Lens는 corporate action을 직접 저장해 검산하지 않고, 공급자의 `adjusted_close`를 신뢰한 뒤 feature layer에서 adjusted OHLC를 재구성한다.

이 구조에서는 새 공급자가 다음 중 하나라도 다르면 곧바로 feature drift가 생긴다.

- `adjusted_close`가 split만 반영하는지, split과 dividend를 모두 반영하는지
- raw OHLC가 이미 split-adjusted인지, truly as-traded raw인지
- volume이 split-adjusted인지 raw volume인지
- dividend ex-date 이전 가격 보정 방식이 EODHD와 같은지

### fundamentals

fundamentals도 현재 EODHD 의존이 아니다.

- `backend/collector/jobs/sync_fundamentals.py:8-19`: EDGAR, FMP, Yahoo fundamental 경로를 쓴다.
- `backend/collector/sources/fundamentals.py:78-147`: Yahoo financial statement baseline을 만든다.
- `backend/collector/sources/fundamentals.py:150-170`: FMP 상세 financial statement를 조회한다.
- `backend/app/services/feature_svc.py:401-465`: fundamentals가 없거나 filing_date가 부족하면 fundamental feature를 0으로 채우고 `has_fundamentals=False`로 둔다. Phase 1에서 fundamentals는 가격 계약만큼 필수는 아니다.

### 1D, 1W, 1M resample과 indicators

- `backend/app/services/feature_svc.py:161-178`: `adjusted_close / close` 비율로 adjusted open/high/low를 만든다.
- `backend/app/services/feature_svc.py:239-258`: 1D는 adjusted OHLC 계약을 그대로 적용하고, 1W/1M은 ticker별 resample을 수행한다.
- `backend/app/services/feature_svc.py:286-334`: `log_return`, `open_ratio`, `high_ratio`, `low_ratio`, MA, RSI, MACD, Bollinger, ATR ratio를 adjusted OHLC 기준으로 계산한다.
- `backend/app/services/feature_svc.py:111-114`: ratio sanity 기준은 `open_ratio/high_ratio/low_ratio`의 p99 절댓값 1.0 이하, max 절댓값 5.0 이하로 정의되어 있다.
- `backend/collector/jobs/compute_indicators.py:163-193`: `price_data`를 읽어 `build_features()`로 indicators를 생성한다.
- `backend/collector/jobs/compute_indicators.py:218-226`: indicators는 `ticker,timeframe,date` 기준으로 upsert한다.

### cron, backfill job

- `backend/collector/pipelines/daily_market_sync.py:104-105`: `EODHD_API_KEY`가 없으면 일일 가격 동기화를 중단한다.
- `backend/collector/pipelines/daily_market_sync.py:134-145`: daily sync에서 `run_prices()`를 호출하고 EODHD key와 `allow_yahoo_fallback`을 넘긴다.
- `backend/collector/pipelines/bootstrap_backfill.py:204-269`: 초기 백필도 EODHD key를 필수로 본다.
- `backend/collector/pipelines/daily_sync.py:123-135`: 구 daily sync 경로도 EODHD key 없이는 가격 동기화를 시작하지 않는다.
- `render.yaml:21-27`: Render cron `lens-daily-market-sync`는 `python -m backend.collector.pipelines.daily_market_sync --indicator-lookback-days 60`을 실행한다.
- `render.yaml:40`: Render 환경 변수에 `EODHD_API_KEY`가 명시되어 있다.

### 기존 sisc, 깃 기반 수집 코드

현재 작업트리에서 운영 collector source로 `sisc`, `Stooq`, `Alpha Vantage`, `Nasdaq Data Link` 어댑터는 발견되지 않았다. 다만 과거 SISC/원본 DB 연동 흔적은 있다.

- `backend/db/scripts/import_kaggle_snapshot.py:21`: `C:\Users\user\projects\sisc-web\AI\data\kaggle_data`의 parquet snapshot을 Lens로 적재하는 스크립트가 있다.
- `backend/db/scripts/run_legacy_crawler.py:18-103`: 외부 `crawler_bot.py`를 실행하고 parquet export까지 이어가는 wrapper가 있다.
- `backend/db/scripts/sync_source_to_lens.py:31-48`: 원본 DB에서 `stock_info`, `price_data`, `macro`, `breadth`, `fundamentals`, `indicators`를 Lens로 동기화하는 경로가 있다.

이 경로는 "현재 운영 수집 소스"가 아니라 legacy/source import 경로다. 원본 수집기가 어떤 무료 소스를 썼는지, adjusted_close 계약이 무엇인지, corporate action이 어떻게 반영됐는지 확인되지 않았으므로 바로 재사용하면 안 된다.

## 3. 무료 후보 비교표

| 후보 | raw OHLC | adjusted_close | split/dividend | volume | ticker/date 계약 | rate limit와 최신성 | 사용 제한과 안정성 | Lens 적합도 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yfinance | `auto_adjust=False`이면 raw OHLC와 `Adj Close`를 받을 수 있다. 현재 코드도 이 옵션을 쓴다. | 가능. 단 Yahoo 쪽 adjusted 의미와 버전별 동작을 split/dividend 샘플로 검증해야 한다. | `actions=True`, `Ticker.actions`, `dividends`, `splits` 경로가 있다. 현재 가격 fallback은 actions를 저장하지 않는다. | 가능. 다만 provider별 volume 보정 차이 검증 필요. | Yahoo 심볼을 쓴다. `BRK.B`, `BF.B`, `GOOG/GOOGL` 매핑 확인 필요. 일봉 timezone은 yfinance 문서상 day 이상에서 timezone ignore 기본값이 다르다. | 공식 rate limit이 명확하지 않고 Yahoo 응답 변화에 취약하다. 최신 EOD는 대체로 빠르지만 운영 보장 없음. | yfinance 문서는 Yahoo 공식 제품이 아니며 연구/교육 목적, Yahoo Finance API는 개인 사용 목적이라고 고지한다. | 코드 적합성 1순위. 운영 primary 전환은 법무/약관/장기 안정성 확인 전 금지. |
| Stooq | 무료 historical OHLCV 파일과 CSV 다운로드가 있다. Stooq 페이지에 미국 daily 데이터 묶음이 제공된다. | 별도 `adjusted_close` 컬럼 계약이 명확하지 않다. close가 조정값인지 raw인지 샘플 검증 필요. | split/dividend 이벤트를 Lens가 필요한 형태로 받을 수 있는지 명확하지 않다. | 가능. 단 거래소별/파일별 volume 의미 확인 필요. | 미국 주식은 대체로 `.US` 접미사를 쓰는 별도 nomenclature가 필요하다. 날짜 timezone 문서가 부족하다. | Stooq historical page는 최근 업데이트 시각과 대용량 파일을 제공한다. API SLA는 없다. | 명확한 상업 사용 조건과 adjusted 계약 문서가 부족하다. | EODHD 대체 primary가 아니라 2차 cross-check 후보. adjusted 계약 불명확성이 크다. |
| Nasdaq Data Link 무료 | 현재 무료로 쓸 수 있는 최신 US EOD OHLCV 전면 coverage가 보이지 않는다. | 과거 WIKI는 adjusted 관련 데이터가 있었지만 2018년 이후 중단. | WIKI는 중단, 신뢰 불가. EOD/Sharadar는 premium 영역. | premium feed는 가능하지만 무료 후보가 아니다. | Nasdaq Data Link 자체 API 계약은 탄탄하지만 dataset별로 다르다. | 인증 사용자 API quota는 넉넉하지만, 필요한 최신 US EOD 가격 dataset이 무료가 아니다. | 공식 문서는 professional application에는 premium을 권고한다. | 현재 Lens 가격 대체재로 부적합. macro/경제 무료 데이터 탐색 정도만 가능. |
| Alpha Vantage free tier | `TIME_SERIES_DAILY`는 raw daily OHLCV를 제공한다. | `TIME_SERIES_DAILY_ADJUSTED`는 raw OHLCV, adjusted close, dividend, split coefficient를 제공한다고 문서화되어 있다. | 계약은 가장 명확한 편이다. 다만 daily adjusted가 premium으로 표시되고 full output도 premium 제약이 있다. | 가능. | 심볼 검색 API가 있고 글로벌 심볼 suffix 체계가 있다. 날짜는 EOD 기준. | 무료는 25 requests/day로 안내된다. 473티커 daily cron에는 절대 부족하다. | realtime/delayed 상업 이용은 별도 sales/premium 성격이 강하다. | 20티커 검증 또는 spot audit 후보. 운영 전환 후보로는 부적합. |
| legacy SISC/Kaggle/source DB | parquet 또는 외부 crawler 결과를 가져오는 흔적만 있다. | 계약 미확인. | 계약 미확인. | 계약 미확인. | 원본 DB와 parquet schema 검증 필요. | 현재 운영 SLA 없음. | 원본 코드, 데이터 출처, 라이선스 확인 전 재사용 불가. | 과거 snapshot 복구/비교용 후보. 무료 primary 후보로 분류하면 안 된다. |
| SEC EDGAR/FRED/내부 파생 | 가격 OHLCV 대체는 아니다. | 해당 없음. | 해당 없음. | 해당 없음. | fundamentals/macro/breadth 보조 데이터. | EDGAR/FRED는 무료 성격이 강하고 현재 코드에 이미 존재한다. | 가격 전환과 별개로 유지 가능. | macro/fundamentals 보완용. price_data 대체 불가. |

외부 문서 근거:

- yfinance 문서는 Yahoo Finance API가 개인 사용 목적이며 yfinance가 Yahoo 공식 제품이 아니라고 고지한다. 또한 `download()`는 `auto_adjust`, `actions`, day 이상 timezone 처리 옵션을 문서화한다.
- Alpha Vantage 문서는 daily raw와 daily adjusted의 차이를 명시하고, support 문서는 무료 API가 25 requests/day라고 안내한다.
- Nasdaq Data Link 문서는 equity price의 QuoteMedia EOD가 premium이고, 무료 WIKI Prices가 2018년에 deprecated되어 현재 분석용으로 권장되지 않는다고 안내한다.
- Stooq는 `Free Historical Market Data` 페이지에서 daily/hourly/5 minute 대용량 historical data 묶음을 제공하지만, Lens가 요구하는 adjusted_close와 corporate action 계약은 충분히 명문화되어 있지 않다.

## 4. 데이터 계약 리스크

### raw OHLC와 adjusted_close 혼용 리스크

CP29 이후 Lens는 adjusted OHLC 계약에 민감하다. `feature_svc.py:161-178`은 raw `close`와 `adjusted_close`의 비율을 open/high/low에 곱한다. 새 공급자의 `close`가 이미 split-adjusted인데 `adjusted_close`도 별도로 제공되면 조정이 두 번 들어갈 수 있다. 반대로 `adjusted_close`가 dividend를 반영하지 않으면 배당주에서 return feature와 MA/RSI/MACD가 EODHD 대비 다르게 움직인다.

전환 시 가장 위험한 컬럼은 `open`, `high`, `low`, `close`, `adjusted_close`다. `stock_info`나 fundamentals보다 훨씬 중요하다.

### split/dividend 원장 부재

현재 Lens DB는 split/dividend 이벤트를 별도 저장하지 않는다. 이 말은 공급자 교체 후 "왜 adjusted_close가 달라졌는지"를 DB만 보고는 추적할 수 없다는 뜻이다. 전환 전 shadow 비교에서는 최소한 provider별 corporate action 원자료를 별도 파일로 저장해 검산해야 한다. 이 파일은 운영 DB에 쓰지 않고 감사 산출물로만 유지해도 된다.

### volume 차이

`vol_change`는 모델 feature다. volume이 split-adjusted인지 raw인지 공급자마다 다르면 `vol_change` 분포가 바뀐다. 가격 차이가 작아도 volume p95 차이가 크면 full 전환을 막아야 한다.

### ticker symbol mapping

특히 위험한 ticker는 `BRK.B`, `BF.B`, `GOOG/GOOGL`, class share, hyphen/period 표기, 상장 거래소 suffix다. 현재 EODHD 경로는 점 표기를 하이픈으로 바꿀 수 있고 `.US`를 붙인다. yfinance와 Stooq는 서로 다른 표기를 요구할 수 있다.

### timezone과 날짜

Lens는 `date` 단위로 `price_data`와 indicators를 맞춘다. 공급자가 UTC timestamp를 주거나, 현지 거래일 기준을 다르게 주거나, 수정 데이터가 다음 날 늦게 반영되면 1W `W-FRI`와 1M month-end resample이 달라질 수 있다.

### 최신성

Render cron은 일일 동기화 후 indicators를 계산한다. 무료 소스가 장 종료 직후에는 stale이고 다음날 correction이 들어오는 구조라면, 최근 forecast와 차트 지표가 흔들린다. 최신성은 "당일 데이터가 있나"뿐 아니라 "수정 데이터가 언제 안정화되나"로 봐야 한다.

### rate limit

Alpha Vantage 무료 25 requests/day는 20티커 검증도 여러 날로 나눠야 할 수 있다. 473티커 운영에는 불가능하다. yfinance는 명시적 무료 quota가 아니라 비공식 endpoint 안정성 문제다. Stooq 대용량 파일은 network load와 사용 조건이 문제다.

### 상업/개인 사용 제한

무료 데이터는 비용이 0이어도 사용권이 0이 아니다. yfinance/Yahoo는 개인 사용 목적 고지가 강하고, Alpha Vantage도 realtime/delayed 상업 사용은 별도 sales 성격이 있다. 발표 데모, 제품 배포, 상업 운영의 사용권은 별도 확인 없이는 통과로 보면 안 된다.

## 5. 추천 전환 순서

1. 현재 운영 기준값은 EODHD로 유지한다.
   무료 소스는 검증 전까지 `price_data` primary로 쓰지 않는다.

2. adapter 계약을 먼저 정의한다.
   후보 adapter는 최소한 `ticker`, `date`, `open`, `high`, `low`, `close`, `adjusted_close`, `volume`, `amount`, `source`, `source_symbol`, `adjustment_policy`, `retrieved_at`, `source_data_hash`를 반환해야 한다. DB schema는 최대한 유지하되, shadow 비교 결과는 운영 DB가 아니라 파일 산출물로 남긴다.

3. yfinance를 1차 shadow 후보로 둔다.
   이유는 현재 `eodhd_prices.py:93-99`에 fallback 호출이 이미 있고, `auto_adjust=False`로 raw OHLC와 adjusted close를 분리하려는 구조가 있기 때문이다. 단 운영 primary가 아니라 비교용으로만 시작한다.

4. Stooq를 2차 가격 cross-check 후보로 둔다.
   Stooq는 비용 절감 잠재력은 크지만 adjusted 계약이 불명확하다. split/dividend 샘플에서 EODHD/yfinance와 맞는지 보기 전에는 `adjusted_close` 대체재가 아니다.

5. Alpha Vantage는 계약 검산용 spot source로 쓴다.
   adjusted daily가 raw OHLC, adjusted close, dividend, split coefficient를 함께 설명하므로 corporate action 검산에는 좋다. 하지만 무료 quota와 premium 제약 때문에 운영 cron 후보가 아니다.

6. Nasdaq Data Link 무료 가격은 현재 후보에서 제외한다.
   최신 US EOD full coverage는 premium에 가깝고, 무료 WIKI는 2018년 이후 갱신되지 않는다. 최신 가격 수집 비용 절감에는 맞지 않는다.

7. 20티커 검증이 통과하면 dual-read 기간을 둔다.
   최소 2주 동안 EODHD와 무료 후보를 동시에 읽어 파일 비교만 수행한다. 이후 차이가 기준 이하이고, 사용권 리스크가 해결된 후보만 부분 전환한다.

8. 부분 전환은 dev/local 또는 비핵심 보조 데이터부터 시작한다.
   운영 `price_data` 전체 전환은 가장 마지막이다.

## 6. 바로 바꿔도 되는 것

이번 CP에서는 코드 수정 금지이므로 실제 변경은 하지 않는다. 정책상 "바꿔도 되는 영역"은 다음 정도다.

- stock_info의 의존도 판단: 이미 FMP/Yahoo/placeholder 구조라서 EODHD 비용 절감 대상이 아니다. 가격 전환보다 우선순위가 낮다.
- fundamentals의 Phase 1 지위: `feature_svc.py:407-412`와 `feature_svc.py:461-465`를 보면 fundamentals가 없어도 0과 `has_fundamentals=False`로 처리된다. 따라서 paid fundamentals를 성능 병목으로 보기 전에 가격 계약을 먼저 고정하는 게 맞다.
- macro/breadth 유지: macro는 별도 FRED/FMP 계열이고, breadth/sector는 현재 `price_data`와 `stock_info`에서 파생된다. 가격 소스 교체 검증이 끝나기 전에는 유지한다.
- yfinance local smoke 또는 shadow 비교: 운영 DB에 쓰지 않고 파일 비교만 한다면 현재 코드와 가장 잘 맞는 무료 후보 실험이다.

## 7. 절대 바로 바꾸면 안 되는 것

- EODHD 없이 yfinance fallback만 켜서 Render daily cron을 운영 primary로 돌리는 것.
- Stooq의 `Close`를 `adjusted_close`로 간주하는 것.
- Alpha Vantage free tier로 473티커 daily cron을 돌리는 것.
- Nasdaq Data Link WIKI를 현재 가격 데이터로 쓰는 것.
- split/dividend 원장 없이 provider를 섞어 같은 `price_data` 테이블에 upsert하는 것.
- `price_data` 안에서 EODHD row와 무료 source row를 구분 불가능하게 섞는 것.
- adjusted_close 검증 없이 feature cache나 indicators를 재생성하는 것.
- 1W/1M resample을 공급자별 raw 데이터 위에서 따로 수행하고, 결과만 비교하는 것. 반드시 1D raw/adjusted 계약을 먼저 맞춘 뒤 Lens의 기존 resample 경로로 비교해야 한다.
- "무료니까 같다"는 가정으로 full 전환하는 것.

## 8. 20티커 검증 계획

### 샘플 universe

권장 20티커:

`AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `GOOG`, `META`, `TSLA`, `AMD`, `NFLX`, `JPM`, `XOM`, `JNJ`, `PG`, `KO`, `UNH`, `AVGO`, `GE`, `BRK.B`, `BF.B`

선정 이유:

- 대형 성장주: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`
- 고변동/고거래대금: `TSLA`, `AMD`, `NFLX`
- 배당주: `KO`, `PG`, `JNJ`, `XOM`, `JPM`
- split 이력 확인: `AAPL`, `NVDA`, `TSLA`, `AVGO`, `GE`
- class share와 symbol mapping: `GOOG`, `GOOGL`, `BRK.B`, `BF.B`

### 비교 범위

- 최소 기간: 최근 3년 일봉 전체
- corporate action 검증 기간: 각 ticker의 주요 split/dividend 이벤트 전후 30거래일
- 가능하면 기간 확장: 2015-01-01 이후 전체. 단 Alpha Vantage 무료 quota 때문에 Alpha는 spot 검산으로 제한한다.
- DB 쓰기 금지: 모든 후보 데이터는 임시 파일 또는 report artifact로만 저장한다.
- 대량 네트워크 호출 금지: yfinance/Stooq도 20티커 제한으로 실행하고, Alpha는 quota 내에서 나눠 실행한다.

### 비교 방법

1. EODHD를 기준 프레임으로 둔다.
2. 후보 source별 raw frame을 표준 컬럼으로 맞춘다.
3. 날짜 교집합과 차집합을 분리한다.
4. raw OHLC diff와 adjusted_close diff를 따로 계산한다.
5. `adj_factor = adjusted_close / close`를 비교한다.
6. Lens 방식으로 adjusted OHLC를 재구성한다.
7. `open_ratio`, `high_ratio`, `low_ratio`, `log_return`, `vol_change`, `atr_ratio`를 계산해 EODHD 결과와 비교한다.
8. 1D 통과 후에만 1W/1M resample 비교를 한다.

### 통과 기준 초안

| 항목 | 통과 기준 |
| --- | --- |
| 날짜 coverage | EODHD 공통 거래일 대비 99% 이상. 최신일 lag는 1거래일 이하. |
| OHLC 기본 정합성 | finite 100%, `high >= low`, adjusted 재구성 후에도 `high >= low`. |
| raw close 차이 | median 상대오차 5bp 이하, p95 20bp 이하, p99 50bp 이하. 단 exchange correction day는 별도 표기. |
| adjusted_close 차이 | median 5bp 이하, p95 20bp 이하, p99 50bp 이하. split/dividend window에서 튀면 full 전환 금지. |
| adjustment factor | EODHD 대비 p95 20bp 이하. split 전후 factor step 방향과 크기가 일치해야 함. |
| volume 차이 | p95 상대오차 10% 이하. 초과 시 `vol_change` feature drift로 분류. |
| ratio sanity | 기존 `feature_svc.py:111-114` 기준인 p99 절댓값 1.0 이하, max 절댓값 5.0 이하를 반드시 통과. |
| feature finite | 1D/1W/1M `open_ratio/high_ratio/low_ratio/log_return/atr_ratio` NaN/Inf 0. |
| symbol mapping | 20티커 모두 source symbol이 명시되고, 실패 ticker 0개. |

### 실패 시 판정

- adjusted_close 불일치가 split/dividend 종목에 집중되면 provider full 전환 금지.
- volume만 실패하면 가격 primary는 보류하고 volume feature 민감도 감사를 먼저 한다.
- ticker mapping 실패가 class share에 집중되면 adapter mapping table 없이는 전환 금지.
- latest date lag가 반복되면 Render daily cron primary로 전환 금지.

## 9. 예상 운영비 절감 효과

정확한 금액은 현재 EODHD 청구서나 plan 정보를 읽지 않았으므로 산정할 수 없다. 따라서 아래는 비용 구조 기준의 정성 추정이다.

| 시나리오 | API 비용 절감 | 운영 리스크 | 판단 |
| --- | --- | --- | --- |
| EODHD 유지, stock_info/fundamentals만 무료화 | 낮음 | 낮음 | 이미 대부분 FMP/Yahoo/EDGAR/FRED 구조라 절감 여지가 작다. |
| yfinance를 dev/local/shadow 비교용으로 사용 | 중간 | 낮음 | EODHD 호출을 개발 검증에서 줄일 수 있다. 운영 primary는 아님. |
| Stooq bulk를 historical backfill 보조로 사용 | 중간에서 높음 | 중간에서 높음 | adjusted 계약 검증 전에는 DB 반영 금지. |
| Alpha Vantage free를 검산용으로 사용 | 낮음 | 낮음 | 25회/일 제약 때문에 비용 절감보다 audit source에 가깝다. |
| EODHD price primary를 무료 소스로 전면 교체 | 높음 | 매우 높음 | 20티커 검증, 사용권 확인, shadow 기간 전에는 금지. |
| Nasdaq Data Link premium EOD로 교체 | 불명확 | 낮음에서 중간 | 무료 전환 목표와 맞지 않는다. EODHD보다 싸다는 보장도 없다. |

현실적인 절감 전략은 "EODHD 즉시 제거"가 아니라 "EODHD를 기준값으로 유지하면서 무료 source를 shadow로 붙여 호출량을 점진적으로 줄일 수 있는지 검증"이다. adjusted OHLC 계약이 통과하면 dev/local, 비핵심 backfill, 일부 ticker group 순서로 비용을 줄일 수 있다. 하지만 현재 단계에서 production daily price primary를 무료 source로 전환하는 것은 데이터 오염 리스크가 비용 절감보다 크다.

## 참고한 외부 문서

- [yfinance documentation](https://ranaroussi.github.io/yfinance/): Yahoo Finance API 사용권 고지, 연구/교육 목적, 개인 사용 목적 관련 고지.
- [yfinance.download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): `auto_adjust`, `actions`, timezone 옵션.
- [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/): daily raw, daily adjusted, split/dividend-adjusted data 설명.
- [Alpha Vantage Support](https://www.alphavantage.co/support/): 무료 API 25 requests/day 안내.
- [Nasdaq Data Link data organization](https://docs.data.nasdaq.com/docs/data-organization): equity price dataset의 free/premium 구분.
- [Nasdaq Data Link WIKI Prices help](https://help.data.nasdaq.com/article/506-why-does-wiki-prices-only-go-up-to-march-2018): WIKI Prices deprecated와 2018년 이후 미갱신 안내.
- [Stooq Free Historical Market Data](https://stooq.com/db/h/): 무료 historical data 묶음, 미국 daily 데이터와 업데이트 시각.

## 감사 중 실행한 작업

- 코드 수정 없음.
- DB 쓰기 없음.
- 모델 학습 없음.
- 대량 가격 데이터 호출 없음.
- 신규 산출물은 이 보고서 하나뿐이다.

