# CP104-3-P 주식 보기 가격 조회/empty state 버그 수정 보고서

## 1. 목표

주식 보기 화면에서 AAPL 1D/1M 가격 조회가 404로 남아 차트가 비는 문제와, `Failed to construct 'URL': Invalid URL` 같은 raw JS 오류가 사용자 화면에 노출될 수 있는 문제를 함께 정리했다.

이번 CP는 리포트 화면 추가가 아니라 주식 보기 가격 조회 안정화 작업이다.

## 2. 가격 404의 실제 원인

원인은 가격 API가 현재 데모 데이터 위치와 다른 provider를 보고 있던 것이다.

- 실제 로컬 가격 데이터는 `data/parquet/price_data_yfinance.parquet`에 존재한다.
- 기존 백엔드 기본 provider는 `eodhd`였다.
- 데모 서버를 별도 env 없이 띄우면 `/api/v1/stocks/AAPL/prices`가 yfinance local parquet가 아니라 Supabase/eodhd 가격 경로를 보게 된다.
- Supabase 쪽 eodhd price_data에는 AAPL 가격 row가 없어 404 `RESOURCE_NOT_FOUND`가 발생했다.
- 반면 prediction/evaluation/history는 제품 run 기준으로 존재해 “예측은 있는데 가격이 없는” 상태가 만들어졌다.

## 3. 프론트 URL 문제 여부

`frontend/src/api/client.ts`에는 이미 `NEXT_PUBLIC_BACKEND_URL`이 비어 있거나 프로토콜이 없는 경우를 보정하는 코드가 있었다.

이번 CP에서는 주식 보기 화면에서 URL 관련 raw JS 오류가 노출되지 않도록 한 번 더 방어했다.

- `Network Error`, `ECONNREFUSED`는 백엔드 연결 실패 안내로 변환
- `Failed to construct 'URL'`, `Invalid URL`은 `NEXT_PUBLIC_BACKEND_URL` 확인 안내로 변환
- raw JS error 문자열을 그대로 빨간 배너에 보여주지 않도록 수정

## 4. 현재 주식 보기 가격 조회 경로

데모 기준 가격 조회 경로는 다음으로 정리했다.

- provider: `yfinance`
- source: local parquet snapshot
- snapshot dir: `C:\Users\user\lens\data\parquet`
- API: `/api/v1/stocks/{ticker}/prices`

`scripts/start_demo.ps1`의 백엔드 runner에 아래 env를 추가했다.

- `MARKET_DATA_PROVIDER=yfinance`
- `LENS_USE_LOCAL_SNAPSHOTS=1`
- `LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet`

또한 `backend/app/repositories/market_repo.py`의 기본 provider를 `yfinance`로 변경했다.

테스트 격리를 위해 `backend/collector/repositories/local_snapshots.py`는 parquet 폴더가 존재한다는 이유만으로 자동 활성화하지 않게 바꿨다. 이제 local snapshot은 `LENS_USE_LOCAL_SNAPSHOTS=1` 또는 local 강제 env가 있을 때만 활성화된다.

## 5. 가격 데이터가 없을 때의 최종 UX

가격 404는 치명 오류로 보지 않는다.

가격 데이터가 없으면:

- 상단 현재가/등락률/거래량은 `-`로 유지
- 차트 영역은 `가격 데이터 없음` empty state 표시
- 상세 문구: `{ticker} {timeframe} 가격 데이터가 아직 연결되지 않았습니다. 티커 또는 데이터 소스를 확인해주세요.`
- 저장된 AI 예측이 있더라도 가격 차트 위에 억지로 표시하지 않음
- 예측 패널에는 `예측 결과는 저장되어 있지만 가격 데이터가 없어 차트 위에 표시할 수 없습니다. 가격 데이터가 연결되면 표시됩니다.` 표시
- raw JS 오류 문구는 노출하지 않음
- 빨간 백엔드 오류 배너는 실제 연결 실패나 5xx 계열 문제에만 사용

## 6. 수정 파일 목록

- `backend/app/repositories/market_repo.py`
  - 기본 market data provider를 `yfinance`로 변경
  - query mock 또는 제한된 query 객체에서 source filter가 터지지 않도록 방어
- `backend/collector/repositories/local_snapshots.py`
  - parquet 폴더 존재만으로 local snapshot이 자동 활성화되지 않게 변경
- `scripts/start_demo.ps1`
  - 데모 백엔드 runner에 yfinance/local snapshot env 추가
- `frontend/src/components/StockView.tsx`
  - 가격 로딩 상태를 `PriceState`로 분리
  - 가격 404와 API 연결 실패를 구분
  - 가격이 없어도 indicator/prediction 조회는 가능한 만큼 진행
  - 저장된 예측은 있으나 가격이 없는 상태를 별도 안내
  - raw URL error 메시지 정리
- `frontend/src/app/globals.css`
  - stacked empty state 스타일 추가
- `docs/cp104_3_stock_price_empty_state_bugfix_report.md`
  - 본 보고서

## 7. 검증 명령

```powershell
cd C:\Users\user\lens

# 백엔드 관련 unittest
$env:PYTHONPATH="C:\Users\user\lens\backend"
.\.venv\Scripts\python.exe -m unittest backend.tests.test_services backend.tests.test_api

# 프론트 빌드
cd C:\Users\user\lens\frontend
npm run build

# 데모 readiness
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

## 8. 검증 결과

- `npm run build`: 통과
- `backend.tests.test_services`, `backend.tests.test_api`: 37개 테스트 통과
- `scripts/check_demo_readiness.ps1`: 통과
  - health OK
  - CORS OK
  - frontend OK
  - AAPL 1D prices OK
  - AAPL indicators OK
  - AAPL 1M prices OK
  - stock search OK
  - LM/BM prediction/evaluation/history OK
- 브라우저 확인
  - AAPL 1D: 가격 차트 상태 OK
  - AAPL 1W: 가격 차트 상태 OK
  - AAPL 1M: 가격 전용 상태 OK
  - 존재하지 않는 ticker: `가격 데이터 없음` empty state OK
  - raw `Failed to construct 'URL'` 노출 없음
  - 빨간 백엔드 연결 실패 배너 없음
  - console error/warn 0건

## 9. 남은 blocker 여부

이번 CP 기준 blocker는 없다.

단, 사용자가 `scripts/start_demo.ps1`이 아니라 직접 uvicorn을 실행할 경우에도 같은 데모 경로를 보려면 아래 env가 필요하다.

```powershell
$env:MARKET_DATA_PROVIDER="yfinance"
$env:LENS_USE_LOCAL_SNAPSHOTS="1"
$env:LENS_LOCAL_SNAPSHOT_DIR="C:\Users\user\lens\data\parquet"
```

이 env 없이 실행하면 백엔드는 yfinance provider를 기본으로 보지만, local snapshot 강제 사용은 켜지지 않는다. 데모 문서에서는 `start_demo.ps1` 또는 위 env를 표준 실행 경로로 유지해야 한다.

## 10. 금지 사항 준수

- fake data 생성 없음
- DB write 없음
- Supabase 대량 read 없음
- 모델 학습 없음
- inference 실행 없음
- composite 제품 기본 복구 없음
