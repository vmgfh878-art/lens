# 로컬 일일 업데이트 런북

## 목적

local parquet를 원천 저장소로 유지하고, Supabase는 제품 표시용 thin DB로만 사용한다.

## 매일 실행 순서

1. 환경 고정
   - `MARKET_DATA_PROVIDER=yfinance`
   - `MARKET_DATA_FALLBACK_PROVIDER=`
   - `EODHD_API_KEY=`
   - `LENS_DATA_BACKEND=local`
   - `LENS_REQUIRE_LOCAL_SNAPSHOTS=1`

2. dry-run
   - `.\scripts\run_local_daily_update.ps1 -DryRun`
   - 신규 완료 거래일이 없으면 `PASS_WITH_NO_NEW_DAY`로 종료해도 정상이다.
   - yfinance fetch가 빈 응답이면 신규 거래일 없음이 아니라 중단 조건으로 본다.

3. price append
   - yfinance에서 `row.date < current_date`인 완료 거래일만 append한다.
   - duplicate `(ticker,date,source)`는 0이어야 한다.
   - adjusted OHLC contract가 실패하면 즉시 중단한다.

4. context refresh
   - breadth: local price universe 기준 재계산
   - sector_returns: local price + stock_info 기준 재계산
   - macro: FRED 최신 observation만 append/update
   - fundamentals: SEC EDGAR `filing_date` 기준 신규 filing만 반영

5. indicator refresh
   - 1D: append된 ticker/date 주변 lookback만 refresh
   - 1W: 완료된 W-FRI period만 refresh
   - 1M: daily job에서는 skip

6. cache/hash gate
   - indicator/context checksum이 바뀌면 기존 feature cache를 재사용하지 않는다.
   - manifest provider/source/context_hash mismatch면 재생성한다.

7. product inference와 thin upload
   - 1D line/band latest inference만 실행한다.
   - 저장은 `save_product_latest_predictions()`만 허용한다.
   - bulk prediction history와 composite 저장은 금지한다.

8. readiness
   - local price latest date
   - local indicator latest date
   - context latest/asof coverage
   - product prediction latest asof_date
   - EODHD fallback 0
   - Supabase bulk read/write 0

## 실패 시 중단 조건

- EODHD fallback 발생
- adjusted OHLC contract 위반
- partial current-date row append 후보 발생
- 1W partial period row 발생
- feature NaN/Inf 발생
- Supabase bulk read/write 발생
- product latest-only row 수 제한 초과
