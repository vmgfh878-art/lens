# Lens

Lens는 미국 주식 데이터를 직접 수집해 Supabase에 적재하고, 서비스와 모델에서 바로 사용할 수 있는 파생 피처까지 생성하는 데이터 파이프라인 프로젝트다.

## 목표

- S&P 500 유니버스를 기준으로 원천 데이터와 파생 데이터를 분리해 관리한다.
- 초기 백필과 일일 증분 수집을 분리해 운영한다.
- 데이터 적재, 파생 피처 계산, API 서빙 구조를 명확히 나눈다.
- 운영 상태를 `sync_state`에 기록해 실패 구간과 재시도 상태를 추적한다.

## 현재 디렉터리 구조

```text
lens/
  ai/                  모델 학습 코드와 전처리 유틸
  backend/
    app/               FastAPI 앱
    db/                스키마, 부트스트랩 유틸, DB 운영 스크립트
  collector/           외부 데이터 수집과 일일 파이프라인
  data/
    universe/          고정 유니버스 파일
    parquet/           스냅샷 parquet 저장소
    cache/             외부 수집 캐시
  frontend/            프론트엔드
  ops/                 운영 스크립트와 cron 예시
```

## 폴더 책임

### `data/`

데이터만 저장한다.

- `data/universe/sp500.csv`
  - 현재 기본 유니버스
- `data/parquet/`
  - 스냅샷 parquet
- `data/cache/`
  - 외부 수집 캐시

`data/` 아래에는 실행 코드나 비즈니스 로직을 두지 않는다.

### `backend/app/`

서비스 API 계층이다.

- Supabase 클라이언트
- 가격 조회 API
- 예측 조회 API
- 피처 조회용 서비스 로직

### `backend/db/`

DB 스키마와 적재 유틸, 운영 스크립트를 둔다.

- `backend/db/schema.sql`
  - Lens DB 스키마
- `backend/db/bootstrap.py`
  - parquet 스냅샷 적재 공통 유틸
- `backend/db/scripts/`
  - 연결 테스트
  - parquet export
  - Kaggle 스냅샷 import
  - 원본 DB 동기화 스크립트

### `collector/`

실제 운영 수집기다.

- `collector/sources/`
  - Yahoo, FRED, FMP 어댑터
- `collector/jobs/`
  - 단위 수집 잡과 파생 계산 잡
- `collector/pipelines/bootstrap_backfill.py`
  - 초기 백필
- `collector/pipelines/bootstrap_snapshot.py`
  - parquet 스냅샷 부트스트랩
- `collector/pipelines/daily_market_sync.py`
  - 일일 가격 중심 증분 수집
- `collector/pipelines/daily_sync.py`
  - 통합 일일 수집 엔트리포인트

## 데이터 흐름

1. 유니버스 파일에서 대상 종목을 읽는다.
2. `stock_info`, `company_fundamentals`, `price_data`, `macroeconomic_indicators`를 적재한다.
3. `price_data + stock_info`로 `sector_returns`를 계산한다.
4. `price_data`로 `market_breadth`를 계산한다.
5. `price_data + macro + breadth`로 `indicators`를 생성한다.
6. 각 작업의 진행 상태를 `sync_state`에 기록한다.

## 주요 테이블

### 원천 테이블

- `stock_info`
- `price_data`
- `company_fundamentals`
- `macroeconomic_indicators`

### 파생 테이블

- `sector_returns`
- `market_breadth`
- `indicators`

### 운영 테이블

- `sync_state`

### 예측 결과 테이블

- `predictions`

## 실행 경로

### 사전 점검

```powershell
python collector/pipelines/preflight.py
```

### 소수 종목 테스트

```powershell
python collector/pipelines/daily_sync.py --tickers AAPL MSFT NVDA
```

### 일일 시장 증분 수집

```powershell
python collector/pipelines/daily_market_sync.py
```

또는 통합 엔트리포인트에서:

```powershell
python collector/pipelines/daily_sync.py --market-only
```

### 초기 백필

```powershell
python collector/pipelines/bootstrap_backfill.py
```

### 스냅샷 parquet 부트스트랩

```powershell
python collector/pipelines/bootstrap_snapshot.py
```

### 원본 DB 동기화 스크립트

```powershell
python backend/db/scripts/sync_source_to_lens.py --indicator-lookback-days 2400
```

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

## 운영 메모

- 초기 세팅은 `bootstrap_backfill.py`로 천천히 채운다.
- 일일 운영은 `daily_market_sync.py` 또는 `daily_sync.py --market-only`로 가격 중심 증분 수집을 수행한다.
- `stock_info`, `company_fundamentals`는 초기 백필과 점진 갱신 대상이다.
- `price_data`는 일일 운영의 핵심이다.
- `daily_market_sync.py`는 최신 가격 일자 커버리지를 확인하고 기준 미달이면 후속 계산을 중단한다.
- 무료 운영 기준에서는 Yahoo 증분 수집의 실행 환경 차이가 크므로, 실제 배포 환경 검증이 필요하다.

## 다음 배포 방향

현재 구조상 가장 자연스러운 배포 조합은 아래와 같다.

- DB: Supabase
- 일일 수집: Render Cron Job
- 백엔드 API: Render Web Service
- 프론트엔드: Vercel

## Render 배포 초안

레포 루트의 `render.yaml`은 아래 두 서비스를 전제로 둔다.

- `lens-backend`
  - FastAPI 웹 서비스
- `lens-daily-market-sync`
  - 평일 오전 7시 30분 KST 기준으로 도는 일일 시장 수집 cron

Render에서 Blueprint로 연결한 뒤, 환경 변수 값만 대시보드에서 채우면 된다.
단, Render cron job은 무료 인스턴스가 아니라 `Starter` 이상 플랜을 써야 한다.
