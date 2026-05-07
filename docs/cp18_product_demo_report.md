# CP18-P 데모 실행 안정화 보고서

## 1. 데모 실행 방법

### 방법 A: 스크립트 실행

Windows PowerShell에서 실행한다.

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

스크립트는 별도 PowerShell 창 2개를 열어 백엔드와 프론트를 실행한다.

- 백엔드: `http://127.0.0.1:8000`
- 프론트: `http://127.0.0.1:3000`
- 프론트 환경값: `NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000`
- 백엔드 환경값: `PYTHONPATH=C:\Users\user\lens\backend`

### 방법 B: 터미널 2개로 직접 실행

터미널 1, 백엔드:

```powershell
cd C:\Users\user\lens
$env:PYTHONPATH="C:\Users\user\lens\backend"
C:\Users\user\lens\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

터미널 2, 프론트:

```powershell
cd C:\Users\user\lens\frontend
$env:NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:8000"
npm run dev -- --hostname 127.0.0.1 --port 3000
```

브라우저:

```text
http://127.0.0.1:3000
```

### 방법 C: readiness 점검

백엔드가 켜진 상태에서 현재 데모 데이터가 충분한지 확인한다.

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

이 스크립트는 데이터를 만들지 않는다. `prediction`, `evaluation`, `backtest` row가 없으면 그대로 `부족`으로 표시한다.

## 2. scripts 추가 여부

추가했다.

- 파일: `scripts/start_demo.ps1`
- 복잡한 health check나 자동 종료는 넣지 않았다.
- 실패 시에도 사용자가 터미널 2개 방식으로 바로 대체할 수 있게 문서화했다.
- 파일: `scripts/check_demo_readiness.ps1`
- 백엔드가 켜진 상태에서 health, AAPL 가격, 티커 검색, latest completed run, prediction, evaluation, backtest row 존재 여부를 점검한다.
- fake data를 만들지 않고 현재 산출물 부족 상태를 그대로 표시한다.

## 3. 발표 동선

1. 주식 보기
   - `AAPL`, `1D` 상태에서 가격 차트를 확인한다.
   - `캔들` / `라인` 전환을 보여준다.
   - `AI 밴드`, `보수적 예측선` 토글을 보여준다.
   - `커버리지`, `평균 밴드 폭`, `보수적 예측선` 카드를 확인한다.
   - 현재 로컬 데이터에서는 AAPL prediction이 비어 있어 `저장된 AI 예측 없음` 상태를 보여주는 동선도 가능하다.
2. 1M 전환
   - 가격 차트가 정상 표시되는지 확인한다.
   - AI 레이어가 비활성화되고 `월봉은 가격 지표만 제공됩니다.` 문구가 보이는지 확인한다.
3. 백테스트
   - 선택한 차트 단위 기준 latest completed run 조회를 확인한다.
   - 수수료 반영 수익률/샤프와 수수료 전 수익률/샤프가 분리된 것을 확인한다.
   - 평가 지표 패널에서 `coverage`, `avg_band_width`가 위쪽에 있는지 확인한다.
4. 모델 학습
   - `완료(completed)` run 탭을 확인한다.
   - `NaN 실패(failed_nan)` 탭을 확인한다.
   - `val_metrics / test_metrics` 품질 지표와 설정 요약을 확인한다.

## 4. 1D / 1W / 1M 확인 결과

로컬 백엔드 API 기준 확인 결과:

- `1D` 가격: `GET /api/v1/stocks/AAPL/prices?timeframe=1D&limit=5` 정상 응답.
- `1W` 가격: `GET /api/v1/stocks/AAPL/prices?timeframe=1W&limit=5` 정상 응답.
- `1M` 가격: `GET /api/v1/stocks/AAPL/prices?timeframe=1M&limit=5` 정상 응답.
- `1D` completed PatchTST run: 1건 확인.
- `1W` completed PatchTST run: 현재 로컬 DB에서는 없음.
- `1M` completed PatchTST run: 현재 로컬 DB에서는 없음. 화면은 가격 전용으로 동작한다.

## 5. 데모 readiness check

백엔드가 켜진 상태에서 추가 점검해야 하는 항목과 현재 확인 결과는 다음과 같다.

| 항목 | 현재 결과 | 비고 |
| --- | --- | --- |
| completed PatchTST run 존재 여부 | 존재 | `patchtst-1D-94d61c4e84d3` |
| 해당 run의 prediction row 존재 여부 | 없음 | `GET /api/v1/stocks/AAPL/predictions/latest?run_id=patchtst-1D-94d61c4e84d3`가 404 |
| 해당 run의 evaluation row 존재 여부 | 없음 | `GET /api/v1/ai/runs/patchtst-1D-94d61c4e84d3/evaluations?ticker=AAPL&timeframe=1D&limit=1`가 빈 배열 |
| 해당 run의 backtest row 존재 여부 | 없음 | `GET /api/v1/ai/runs/patchtst-1D-94d61c4e84d3/backtests?timeframe=1D&limit=1`가 빈 배열 |
| AAPL 1D 가격 조회 가능 여부 | 가능 | 가격 차트 표시 가능 |
| stock search 가능 여부 | 실패 | `GET /api/v1/stocks?search=AAPL`가 503, `stock_info` 기반 검색만 실패 |
| 1M 가격 전용 가능 여부 | 가능 | `GET /api/v1/stocks/AAPL/prices?timeframe=1M&limit=5` 정상 응답 |

현재 데모 산출물은 가격 조회와 completed run 조회까지는 준비되어 있지만, AAPL prediction/evaluation row와 1D backtest row는 없다. 따라서 주식 보기의 AI overlay, 백테스트 상세 수치, 평가 지표는 실제 row가 생기기 전까지 empty state로 보이는 것이 정상이다.

## 6. empty / error state 확인 결과

- 백엔드 꺼짐 / 프론트만 켜짐
  - 실제 실행 중인 8000번 서버는 사용자 작업 보호를 위해 중지하지 않았다.
  - 대신 `Network Error` / `ECONNREFUSED`를 한국어 문구로 매핑했다.
  - 표시 문구: `백엔드에 연결할 수 없습니다. 127.0.0.1:8000 서버가 켜져 있는지 확인해주세요.`
- stock search 실패
  - `GET /api/v1/stocks?search=AAPL`가 503으로 확인됐다.
  - 주식 보기 화면은 검색 실패를 전체 화면 오류로 띄우지 않고 `티커 검색을 사용할 수 없습니다. 티커를 직접 입력하면 가격 조회는 계속 가능합니다.`를 표시한다.
  - 사용자가 `AAPL`을 직접 입력하고 `조회`를 누르면 기존 가격 조회 경로는 계속 동작한다.
- completed run 없음
  - `1W`, `1M` completed run 조회가 빈 배열로 확인됐다.
  - 백테스트/평가 패널은 완료 run 없음 empty state를 표시한다.
- prediction 없음
  - `AAPL` + `patchtst-1D-94d61c4e84d3` prediction 조회가 404로 확인됐다.
  - 주식 보기 화면은 가격 차트는 유지하고 `저장된 AI 예측 없음: 선택한 run에 이 티커 prediction row가 없습니다.`를 표시한다.
  - API 오류인 경우에는 `AI 예측 API 오류: ...`로 표시해 prediction row 부재와 구분한다.
- 1M 가격 전용
  - 1M 가격 API 정상 응답 확인.
  - 프론트는 1M에서 prediction/evaluation API를 호출하지 않고 AI 토글을 비활성화한다.
- backtest 결과 없음
  - 현재 로컬 `patchtst-1D-94d61c4e84d3` backtest 조회가 빈 배열로 확인됐다.
  - 화면은 `최근 완료 run에 저장된 백테스트 결과가 없습니다.`를 표시한다.
- failed_nan run 탭
  - 현재 로컬 DB에서는 `failed_nan` run 조회가 빈 배열로 확인됐다.
  - 모델 학습 화면은 빈 목록 상태에서도 탭 자체가 유지된다.

## 7. 모바일 / 데스크톱 QA 결과

- 데스크톱 폭
  - 첫 화면은 주식 보기로 바로 진입한다.
  - 왼쪽 내비게이션은 사이드바 형태를 유지한다.
  - 차트와 레이어 패널은 좌우 구조로 표시된다.
- 모바일 폭
  - `860px` 이하에서 사이드바가 상단 탭 내비게이션으로 접힌다.
  - metric card는 1열로 접혀 줄바꿈 충돌을 줄인다.
  - `520px` 이하에서 카드 높이와 글자 크기를 조금 줄였다.

브라우저 자동 스크린샷 도구와 Playwright 의존성은 현재 환경에 없어 화면 캡처 검증은 수행하지 못했다. 대신 HTTP 응답, API 응답, CSS breakpoint 경로를 확인했다.

## 8. 수정한 polish 목록

- `scripts/start_demo.ps1` 추가.
- `scripts/check_demo_readiness.ps1` 추가.
- 백엔드 연결 실패 문구를 한국어로 정리.
- stock search 503 시 전체 화면 오류 대신 작은 안내 문구로 fallback 처리.
- prediction 404와 일반 AI API 오류를 구분해 표시.
- metric card 값 글자 크기를 `clamp`로 조정해 긴 숫자 줄바꿈 안정화.
- 차트 legend에 상단 여백을 추가해 차트와 겹침을 줄임.
- 모바일 폭에서 workspace padding과 metric card 높이를 조정.
- 데모 발표 동선을 문서화.

## 9. npm run build 결과

샌드박스 내부에서는 Next.js worker 생성이 `spawn EPERM`으로 실패했다. CP18 추가 지시 반영 후 승인된 실행에서는 정상 통과했다.

```text
npm run build
Compiled successfully
Linting and checking validity of types ...
Generating static pages (4/4)
Route (app) / 79.1 kB, First Load JS 166 kB
```

## 10. 실행 확인 결과

현재 확인한 로컬 응답:

```text
GET http://127.0.0.1:8000/api/v1/health/live
200 OK

GET http://127.0.0.1:3000
200 OK
```

프론트 dev 서버는 `NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000`로 재시작한 뒤 200 응답을 확인했다.

`scripts/check_demo_readiness.ps1` 실행 결과:

```text
OK   health - live 200
OK   1D prices - AAPL 5 rows
OK   1M prices - price-only view available, 5 rows
FAIL stock search - 503 UPSTREAM_UNAVAILABLE
OK   completed run - patchtst-1D-94d61c4e84d3
WARN prediction - AAPL prediction row missing, RESOURCE_NOT_FOUND
WARN backtest - stored backtest rows missing
WARN evaluation - AAPL evaluation row missing
```

## 11. 남은 데모 리스크

- 현재 로컬 DB 기준으로 AAPL prediction, backtest, evaluation이 비어 있어 AI overlay가 실제로 보이지 않을 수 있다.
- `stock_info` 기반 티커 검색 API가 503을 반환한다. 프론트는 fallback 처리했지만, 검색 자동완성 자체는 백엔드 원인 해결 전까지 사용할 수 없다.
- 1W/1M completed run이 없는 상태라 백테스트/평가 패널은 empty state 중심으로 보인다.
- `scripts/start_demo.ps1`은 새 PowerShell 창을 띄우는 단순 스크립트라 포트 충돌을 자동 해결하지 않는다.
- 브라우저 시각 QA는 수동 확인이 필요하다.

실제 AI overlay와 백테스트 수치를 보여주려면 fake data가 아니라 기존 completed run의 checkpoint로 inference와 backtest 산출물을 저장해야 한다. 현재 코드 기준 후보 명령은 다음과 같다.

```powershell
cd C:\Users\user\lens
$env:PYTHONPATH="C:\Users\user\lens"
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.inference --run-id patchtst-1D-94d61c4e84d3 --split test --tickers AAPL --save
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.backtest --run-id patchtst-1D-94d61c4e84d3 --timeframe 1D --strategy-name band_breakout_v1 --save
```

위 명령은 CP18에서 실행하지 않았다. 데모 데이터 생성이 필요한 경우에만 별도 작업으로 실행한다.
