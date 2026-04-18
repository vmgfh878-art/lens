# Lens

Lens는 독립적인 주식 데이터 수집 파이프라인을 통해 raw 데이터를 직접 적재하고, 서비스와 모델에 맞는 파생 피처를 다시 계산하는 프로젝트입니다.

## 목표

- 원본 프로젝트의 수집 아이디어와 피처 계산 규칙은 참고하되, 운영은 완전히 독립적으로 가져간다.
- 외부 데이터 소스에서 직접 데이터를 받아 Lens DB에 적재한다.
- raw 테이블과 파생 테이블을 분리해 관리한다.
- 일일 수집, 증분 업데이트, 장애 추적을 하나의 파이프라인으로 운영한다.

## 데이터 흐름

1. 유니버스 파일에서 대상 종목 목록을 읽는다.
2. `stock_info`, `company_fundamentals`, `price_data`, `macroeconomic_indicators`를 적재한다.
3. `price_data`와 `stock_info`를 바탕으로 `sector_returns`를 계산한다.
4. `price_data`를 바탕으로 `market_breadth`를 계산한다.
5. 가격, 거시, breadth를 합쳐 `indicators`를 생성한다.
6. 각 작업 상태를 `sync_state`에 남긴다.

## 현재 사용하는 테이블

### 원천 데이터 테이블

- `stock_info`
- `price_data`
- `company_fundamentals`
- `macroeconomic_indicators`

### 내부 파생 테이블

- `sector_returns`
- `market_breadth`
- `indicators`

### 예측 결과 테이블

- `predictions`

### 운영 상태 테이블

- `sync_state`

## 유니버스 관리

현재는 고정 파일 방식으로 유니버스를 관리합니다.

- 기본 경로: `data/universe/sp500.csv`
- 기본 포맷: `ticker, company_name, sector, industry`
- 현재 기본 스냅샷: S&P 500 구성종목 503개

파이프라인의 티커 우선순위는 아래와 같습니다.

1. `--tickers` CLI 인자
2. `LENS_TICKERS` 환경변수
3. `LENS_UNIVERSE_FILE` 또는 기본 `data/universe/sp500.csv`
4. `stock_info`에 이미 적재된 티커
5. 기본 Big Tech 목록

향후 계획:

- 별도 `universe_source` 또는 `universe_memberships` 테이블을 만들어 DB 기반 유니버스 관리로 확장
- 인덱스별 버전 관리와 편입/편출 이력 관리 추가

## 디렉터리

- `db/schema.sql`
  - Lens Supabase 스키마
- `collector/pipelines/daily_sync.py`
  - 독립 일일 수집 메인 엔트리포인트
- `collector/pipelines/daily_market_sync.py`
  - Yahoo 증분 가격 수집 중심의 일일 운영 엔트리포인트
- `collector/pipelines/bootstrap_backfill.py`
  - S&P 500 초기 백필 전용 엔트리포인트
- `collector/jobs`
  - 동기화 및 파생 계산 작업
- `collector/sources`
  - FMP, FRED, Yahoo 소스 어댑터
- `collector/repositories`
  - Supabase 조회 및 upsert 계층
- `collector/universe.py`
  - 고정 유니버스 파일 로더
- `data/universe/sp500.csv`
  - 현재 고정 S&P 500 유니버스 스냅샷
- `ops/run_daily_collector.sh`
  - 운영용 실행 스크립트
- `ops/lens-collector.cron.example`
  - cron 예시

## 환경 변수

### 필수

- `SUPABASE_URL`
- `SUPABASE_KEY`

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

## 설치

```powershell
pip install -r backend/requirements.txt
pip install -r collector/requirements.txt
```

## 실행

### 사전 점검

```powershell
python collector/pipelines/preflight.py
```

### 소수 종목 테스트

```powershell
python collector/pipelines/daily_sync.py --tickers AAPL MSFT NVDA
```

### 유니버스 전체 실행

```powershell
python collector/pipelines/daily_sync.py
```

### 일일 시장 증분 운영

```powershell
python collector/pipelines/daily_market_sync.py
```

또는 통합 엔트리포인트에서:

```powershell
python collector/pipelines/daily_sync.py --market-only
```

샘플 테스트:

```powershell
python collector/pipelines/daily_market_sync.py --tickers AAPL MSFT NVDA --breadth-min-tickers 1 --indicator-lookback-days 60
```

### 초기 백필 실행

```powershell
python collector/pipelines/bootstrap_backfill.py
```

### 전체 재계산

```powershell
python collector/pipelines/daily_sync.py --repair
```

### 피처 계산 생략

```powershell
python collector/pipelines/daily_sync.py --skip-indicators
```

## 운영 메모

- 초기 세팅 단계에서는 `bootstrap_backfill.py`를 사용해 유니버스를 천천히 채운다.
- 초기 백필이 끝난 뒤 일일 운영은 `daily_sync.py`로 전환한다.
- 무료 운영 기준 일일 가격 수집은 `daily_market_sync.py`에서 Yahoo 증분 수집을 우선 사용한다.
- `daily_market_sync.py`는 최신 가격 일자 커버리지를 검사한 뒤 기준 미달이면 파생 계산을 중단한다.
- `price_data`는 lookback 버퍼를 두고 다시 읽어 증분 적재한다.
- `company_fundamentals`는 FMP 보강을 우선 사용한다.
- `sector_returns`는 외부 ETF를 직접 저장하지 않고, 내부 `price_data + stock_info` 기준으로 계산한다.
- `market_breadth`는 내부 가격 이력 기준으로 계산한다.
- `indicators`는 `1D`, `1W`, `1M` 세 타임프레임으로 생성한다.
- `sync_state`는 각 작업의 성공 여부, 마지막 커서 날짜, 메시지를 저장한다.

## API 예시

### 가격 조회

```text
GET /prices/AAPL?timeframe=1D&start=2024-01-01
GET /prices/AAPL?timeframe=1W&start=2023-01-01
GET /prices/AAPL?timeframe=1M&start=2020-01-01
```

### 최신 예측 조회

```text
GET /predict/AAPL?model=patchtst&timeframe=1D&horizon=10
GET /predict/AAPL?model=patchtst&timeframe=1W&horizon=12
GET /predict/AAPL?model=patchtst&timeframe=1M&horizon=6
```
