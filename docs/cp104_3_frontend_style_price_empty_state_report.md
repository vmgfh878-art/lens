# CP104-3-P 프론트 스타일 및 가격 empty state 수정 보고서

작성 시각: 2026-05-04 15:32:27 KST

## 1. 요약

이번 CP104-3-P는 새 기능 추가가 아니라 데모 실행 안정화 작업이다.

- 프론트가 무스타일 HTML처럼 보이던 원인을 확인하고 dev 서버를 정상 상태로 복구했다.
- AAPL 가격 조회가 404였던 원인을 확인하고, 데모 실행 경로에서는 local yfinance snapshot을 사용하도록 정리했다.
- 가격 데이터가 없을 때 raw JavaScript 오류나 빨간 장애 배너가 아니라 조용한 empty state가 보이도록 프론트 UX를 정리했다.
- DB write, fake data 생성, 모델 학습, inference 실행, Supabase 대량 read는 하지 않았다.

## 2. 스타일 미적용 원인

원인은 CSS 파일 자체가 깨진 것이 아니라 Next.js dev 서버와 `.next` 산출물 상태가 어긋난 것이었다.

확인한 현상:

- `http://127.0.0.1:3000` HTML은 200으로 응답했다.
- 하지만 dev 서버가 실행 중인 상태에서 `npm run build`가 같은 `frontend/.next` 디렉터리를 덮어쓰면, dev 서버가 참조하던 `_next/static/...` asset 경로와 실제 파일이 맞지 않게 되었다.
- 이 상태에서 다음 asset들이 404로 재현됐다.
  - `/_next/static/css/app/layout.css`
  - `/_next/static/chunks/main-app.js`
  - `/_next/static/chunks/app-pages-internals.js`
  - `/_next/static/chunks/app/page.js`
  - `/_next/static/chunks/polyfills.js`

결론:

- readiness의 frontend 200만으로는 스타일 적용 여부를 보장할 수 없다.
- dev 서버가 켜진 상태에서 `npm run build`를 실행했다면 dev 서버를 재시작해야 한다.

## 3. 프로세스 종료 및 재시작

무스타일 상태를 만든 stale dev 서버 계열 프로세스를 종료하고 프론트 dev 서버를 다시 시작했다.

종료한 프론트 관련 프로세스:

- `cmd`: PID 40048, 29520
- `node`: PID 37672, 33140, 16536, 29632

재시작한 프론트 dev 서버:

```powershell
cd C:\Users\user\lens\frontend
$env:NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:8000"
npm run dev -- --hostname 127.0.0.1 --port 3000
```

현재 확인 상태:

- 프론트 HTML: 200
- `_next/static/css/app/layout.css`: 200
- `_next/static/chunks/main-app.js`: 200
- `_next/static/chunks/app/page.js`: 200
- 브라우저에서 스타일 적용된 화면 확인

## 4. `.next` 삭제 여부

이번 CP에서는 `frontend/.next`를 삭제하지 않았다.

이유:

- asset 404 원인이 `.next` 영구 손상이 아니라 dev 서버와 build 산출물의 런타임 불일치였기 때문이다.
- dev 서버 재시작만으로 `_next/static` asset이 200으로 복구됐다.
- `node_modules` 삭제나 재설치도 하지 않았다.

## 5. URL/env 확인

프론트 dev 서버는 다음 환경값으로 실행했다.

```powershell
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
```

프론트에서는 백엔드 URL 조립 시 프로토콜이 없는 값이나 잘못된 값이 들어와도 raw `Failed to construct 'URL': Invalid URL` 메시지가 그대로 사용자에게 노출되지 않도록 방어했다.

최종 UX:

- 실제 연결 실패나 설정 오류일 때만 백엔드 연결 안내를 표시한다.
- 단순 가격 404는 장애가 아니라 데이터 없음 상태로 처리한다.

## 6. AAPL 가격 404 원인

가격 404의 핵심 원인은 가격 데이터 소스 전환 상태와 API 기본 조회 경로가 맞지 않았기 때문이다.

확인한 구조:

- 실제 데모용 가격 snapshot은 `C:\Users\user\lens\data\parquet\price_data_yfinance.parquet`에 존재한다.
- 기존 백엔드 기본 provider는 `eodhd` 경로를 우선했다.
- local snapshot env가 없으면 주식 보기 가격 API는 local yfinance parquet가 아니라 Supabase 또는 provider 기준 price data 경로를 보게 되어 AAPL 가격을 찾지 못할 수 있었다.

이번 CP에서 정리한 실행 기준:

- 기본 시장 데이터 provider: `yfinance`
- 데모 실행 시 local snapshot 사용:
  - `LENS_USE_LOCAL_SNAPSHOTS=1`
  - `LENS_LOCAL_SNAPSHOT_DIR=C:\Users\user\lens\data\parquet`
- start demo 스크립트에도 위 env를 넣어 같은 문제가 반복되지 않게 했다.

현재 확인 결과:

- `/api/v1/stocks/AAPL/prices?timeframe=1D&limit=5`: 200
- `/api/v1/stocks/AAPL/prices?timeframe=1W&limit=5`: 200
- `/api/v1/stocks/AAPL/prices?timeframe=1M&limit=5`: 200

## 7. 현재 주식 보기 가격 조회 경로

현재 데모 기준 주식 보기 가격 조회 경로는 다음과 같다.

1. 프론트 `StockView`가 `/api/v1/stocks/{ticker}/prices`를 호출한다.
2. 백엔드는 market repository를 통해 가격을 조회한다.
3. 데모 실행 env가 켜져 있으면 local yfinance parquet snapshot을 사용한다.
4. local snapshot이 꺼져 있으면 기존 backend provider/Supabase 경로로 동작한다.

중요한 점:

- Supabase `price_data` 또는 `indicators`를 대량 read하지 않았다.
- 없는 가격을 fake data로 채우지 않았다.
- 가격이 없으면 empty state로 처리한다.

## 8. 가격 empty state UX

가격 데이터가 없을 때의 주식 보기 화면 동작을 정리했다.

최종 UX:

- 상단 현재가, 등락률, 거래량은 `-`로 표시한다.
- 차트 영역에는 `가격 데이터 없음` 상태를 표시한다.
- 가격이 없지만 예측 row가 있는 경우에는 `가격 데이터가 연결되면 차트 위에 표시됩니다` 성격의 안내를 보여준다.
- 404 데이터 없음은 빨간 백엔드 장애 배너로 처리하지 않는다.
- raw JavaScript 오류 메시지를 사용자에게 그대로 보여주지 않는다.

확인한 안정성:

- AAPL 1D, 1W, 1M 전환 중 raw URL error 없음
- 없는 티커 입력 시에도 화면이 깨지지 않음
- 백엔드 연결 실패 배너가 가격 없음 상황에 잘못 뜨지 않음

## 9. 수정 파일

- `backend/app/repositories/market_repo.py`
  - 기본 market data provider를 yfinance 기준으로 조정했다.
  - mock query처럼 `eq`가 없는 객체에서도 source filter가 터지지 않도록 방어했다.

- `backend/collector/repositories/local_snapshots.py`
  - `data/parquet` 폴더 존재만으로 local snapshot이 자동 활성화되지 않게 했다.
  - `LENS_USE_LOCAL_SNAPSHOTS=1` 또는 local 강제 env가 있을 때만 local snapshot을 사용하게 했다.

- `scripts/start_demo.ps1`
  - 백엔드 실행 env에 `MARKET_DATA_PROVIDER=yfinance`를 추가했다.
  - `LENS_USE_LOCAL_SNAPSHOTS=1`과 `LENS_LOCAL_SNAPSHOT_DIR`을 추가했다.

- `frontend/src/components/StockView.tsx`
  - 가격 조회 상태를 분리했다.
  - 가격 404와 백엔드 연결 실패를 구분했다.
  - 잘못된 URL raw 오류를 사용자 친화 문구로 바꿨다.
  - 가격 없음 상태에서도 prediction/indicator 상태가 화면을 깨지 않도록 정리했다.

- `frontend/src/app/globals.css`
  - 가격 없음 empty state가 여러 줄 안내를 안정적으로 보여줄 수 있게 스타일을 추가했다.

- `docs/cp104_3_frontend_style_price_empty_state_report.md`
  - 이번 CP 보고서를 추가했다.

## 10. 검증 결과

실행한 검증:

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 통과

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

결과:

- backend health OK
- CORS OK
- frontend OK
- AAPL 1D prices OK
- AAPL 1M prices OK
- indicators OK
- stock search OK
- LM/BM run, prediction, evaluation, history OK
- band width OK

```powershell
cd C:\Users\user\lens
.\.venv\Scripts\python.exe -m unittest backend.tests.test_services backend.tests.test_api
```

결과:

- 37개 테스트 통과

브라우저 확인:

- AAPL 1D 확인
- AAPL 1W 확인
- AAPL 1M 확인
- 스타일 적용 확인
- `_next/static` asset 200 확인
- 백엔드 연결 실패 배너 없음
- browser console error/warn 0건 확인

## 11. 남은 blocker 및 주의점

남은 blocker:

- dev 서버가 켜져 있는 상태에서 `npm run build`를 다시 실행하면 같은 `.next` 산출물 불일치가 재발할 수 있다.
- build 후에는 프론트 dev 서버 재시작이 필요하다.

운영 주의:

- 데모 가격 조회는 local yfinance snapshot env에 의존한다.
- start demo 스크립트나 수동 실행 시 아래 env를 유지해야 한다.

```powershell
$env:MARKET_DATA_PROVIDER="yfinance"
$env:LENS_USE_LOCAL_SNAPSHOTS="1"
$env:LENS_LOCAL_SNAPSHOT_DIR="C:\Users\user\lens\data\parquet"
```

하지 않은 것:

- DB write 없음
- fake data 생성 없음
- 모델 학습 없음
- inference 실행 없음
- Supabase price_data/indicators 대량 read 없음
- `node_modules` 삭제/재설치 없음
- `.next` 삭제 없음
