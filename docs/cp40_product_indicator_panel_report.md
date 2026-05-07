“CP40-P는 디자인 전체 리워크가 아니라, 주식 보기 화면의 보조지표 구조를 TradingView식 패널로 분리하는 CP다.”

## 변경 파일

- `backend/app/repositories/market_repo.py`
- `backend/app/services/api_service.py`
- `backend/app/schemas/stocks.py`
- `backend/app/routers/v1/stocks.py`
- `backend/tests/test_api.py`
- `backend/tests/test_services.py`
- `frontend/src/api/client.ts`
- `frontend/src/components/StockView.tsx`
- `frontend/src/components/Chart.tsx`
- `frontend/src/components/IndicatorPanel.tsx`
- `frontend/src/app/globals.css`
- `scripts/check_demo_readiness.ps1`
- `scripts/start_demo.ps1`

## 구현 요약

- 주식 보기 화면의 메인 가격 차트 아래에 독립 보조지표 패널을 추가했다.
- 오른쪽 패널의 단일 보조지표 버튼을 제거하고, AI 보조지표와 파생 보조지표를 그룹별 체크 목록으로 선택하게 바꿨다.
- 기본 표시 지표는 `RSI`, `AI 밴드 폭`이다.
- 여러 지표를 동시에 켜면 보조지표 패널 안에서 세로로 쌓이며, 각 지표는 독립 축과 최신값을 가진다.
- 1M에서는 가격 중심 화면을 유지하고 AI 보조지표 선택은 비활성화한다.
- 메인 가격 차트의 예측 overlay는 유지하되, 가격 스케일을 크게 망가뜨리는 예측 포인트는 차트 표시에서 제외한다.

## 추가 API

`GET /api/v1/stocks/{ticker}/indicators?timeframe=1D&limit=300`

응답 필드:

- `date`
- `rsi`
- `macd_ratio`
- `bb_position`
- `ma_5_ratio`
- `ma_20_ratio`
- `ma_60_ratio`
- `vol_change`
- `atr_ratio`
- `regime_label`

DB schema는 변경하지 않았다. 실제 Supabase에 없는 컬럼은 조회에서 제외하고 응답에서 `null`로 채운다. 현재 로컬 Supabase 기준 `atr_ratio` 컬럼이 없어 `null` fallback이 적용된다.

## 보조지표 분류

AI 보조지표:

- AI 밴드 폭
- 보수적 변화
- 하방 위험
- Coverage

파생 보조지표:

- RSI
- MACD
- BB 위치
- MA 5
- MA 20
- MA 60
- 거래량 변화
- ATR

## 검증 결과

- `cd C:\Users\user\lens\frontend; npm run build`
  - 통과
- `$env:PYTHONPATH="C:\Users\user\lens\backend"; python -m unittest discover backend\tests`
  - 통과, 43개 테스트 OK
- `powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`
  - `health`: OK
  - `cors`: OK
  - `frontend`: OK
  - `1D prices`: OK
  - `indicators`: OK
  - `1M prices`: OK
  - `completed runs`: OK
  - `prediction`: OK
  - `evaluation`: OK
  - `stock search`: FAIL, 기존 `stock_info` 조회 503
  - `backtest`: WARN, 저장된 backtest row 없음

## 화면 QA

- AAPL 1D 가격 차트가 표시된다.
- 보조지표 패널이 메인 가격 차트 아래에 분리되어 표시된다.
- RSI와 AI 밴드 폭이 기본 패널로 표시된다.
- 오른쪽 패널에서 AI/파생 보조지표를 개별 선택할 수 있다.
- prediction row가 있으면 가격 스케일에 맞는 completed run을 골라 overlay를 표시한다.
- 가격 스케일에 맞지 않는 일부 예측 포인트는 표시에서 제외한다.
- `stock search` 503은 작은 안내 문구로만 표시되고, 직접 입력한 AAPL 가격 조회는 유지된다.
- 화면 캡처: `artifacts/cp40_aapl_1d_full.png`

## 남은 리스크

- `stock_info` 기반 티커 검색은 여전히 503이다. 가격 조회와 보조지표 조회는 정상이라 전체 화면 오류로 올리지는 않는다.
- demo readiness 기준 latest run의 backtest row가 없다.
- 실제 prediction band 일부 값은 가격 범위와 차이가 커서 차트 표시 시 필터링된다. 데이터는 수정하지 않았고, 화면 표시만 가격 스케일 보호 목적으로 제한했다.
- in-app browser의 `tab.goto()`가 로컬 dev 서버에서 timeout되어, 화면 검증은 Chrome headless 스크린샷으로 대체했다.
