# Lens

Lens는 미국 주식 데이터를 수집해 Supabase에 적재하고, 모델 입력용 파생 지표까지 생성하는 데이터 파이프라인 프로젝트다.

## 현재 운영 방향

- 유니버스: `S&P 500`
- 가격 소스: `EODHD`
- 거시 소스: `FRED` + 일부 `FMP`
- 재무/종목 메타데이터: `FMP`
- 저장소: `Supabase`

Yahoo 증분 수집은 로컬과 Render 양쪽에서 모두 실패가 재현되어, 운영용 가격 소스로는 사용하지 않는다.

## 디렉터리 구조

```text
lens/
  ai/                  학습 전처리와 학습 엔트리포인트
  backend/
    app/               FastAPI API
    db/                스키마와 DB 스크립트
  collector/           데이터 수집기와 일일 파이프라인
  data/
    universe/          유니버스 파일
    parquet/           parquet 스냅샷
    cache/             외부 소스 캐시
  frontend/            Next 프론트엔드
  ops/                 운영 스크립트와 cron 예시
```

## 핵심 테이블

### 원천 테이블

- `stock_info`
- `price_data`
- `company_fundamentals`
- `macroeconomic_indicators`

### 파생 테이블

- `sector_returns`
- `market_breadth`
- `indicators`

### 운영 상태 테이블

- `sync_state`

## 파이프라인

### 1. 초기 백필

```powershell
python collector/pipelines/bootstrap_backfill.py
```

- `stock_info`, `company_fundamentals`, `price_data`를 채운다.
- 가격은 `EODHD`로 우선 적재한다.
- 재무 데이터가 아직 다 안 차도 `price_data` 적재를 막지 않는다.
- `per`, `pbr`는 재무 데이터가 들어온 종목부터 순차적으로 채워진다.

### 2. 일일 시장 증분

```powershell
python collector/pipelines/daily_market_sync.py --indicator-lookback-days 60
```

- `price_data`를 EODHD 증분으로 갱신한다.
- 최신 거래일 커버리지를 확인한다.
- 기준 미달이면 `sector_returns`, `market_breadth`, `indicators` 계산을 중단한다.

### 3. 통합 일일 수집

```powershell
python collector/pipelines/daily_sync.py
```

- `macro`
- `stock_info`
- `fundamentals`
- `prices`
- `sector_returns`
- `market_breadth`
- `indicators`

순서로 실행한다.

### 4. 가격 중심 일일 모드

```powershell
python collector/pipelines/daily_sync.py --market-only --indicator-lookback-days 60
```

- 일일 운영에서는 이 경로를 기본으로 권장한다.
- 핵심은 `EODHD 가격 증분 -> 커버리지 검사 -> 파생 계산`이다.

## 환경 변수

### 필수

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `EODHD_API_KEY`

### 권장

- `FRED_API_KEY`
- `FMP_API_KEY`

### 선택

- `LENS_UNIVERSE_FILE`
- `LENS_TICKERS`
- `LENS_PRICE_START_DATE`
- `LENS_PRICE_LOOKBACK_DAYS`
- `LENS_MACRO_LOOKBACK_DAYS`
- `LENS_SECTOR_LOOKBACK_DAYS`
- `LENS_INDICATOR_LOOKBACK_DAYS`
- `LENS_BREADTH_MIN_TICKERS`
- `FMP_DAILY_LIMIT`
- `LENS_ALLOW_YAHOO_FALLBACK`
- `LENS_USE_YAHOO_FUNDAMENTALS_BASELINE`

## EODHD 메모

- API 키가 필요하다.
- 무료 플랜은 일일 호출 수와 과거 기간이 제한적이라 운영용으로는 부족하다.
- 유료 `EOD Historical Data` 플랜은 공식 문서 기준 월 `$19.99`부터 시작한다.
- 공식 문서 기준 `EOD Historical API`는 요청 1회로 해당 티커의 긴 가격 이력을 받을 수 있다.
- 공식 문서 기준 `Bulk API`도 있어, 나중에는 거래일 하루치 전체 시장 수집으로 최적화할 수 있다.

참고:

- [EOD Historical Data API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes/)
- [Bulk API](https://eodhd.com/knowledgebase/bulk-download-api/)
- [Pricing](https://eodhd.com/pricing-quantpedia)

## Render 배포

현재 `render.yaml`에는 아래 두 서비스가 정의되어 있다.

- `lens-backend`
- `lens-daily-market-sync`

cron 서비스에는 다음 환경 변수가 필요하다.

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `FRED_API_KEY`
- `FMP_API_KEY`
- `EODHD_API_KEY`

배포 후에는 먼저 `lens-daily-market-sync`를 수동 실행해서 커버리지와 실행 시간을 확인하는 것을 권장한다.
