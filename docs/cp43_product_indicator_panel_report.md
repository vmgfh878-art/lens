CP43-P는 전체 리디자인이 아니라, 주식 보기 화면에서 가격 차트와 실제 값이 있는 보조지표를 함께 보여주는 CP다.

# 1. 변경 요약

- 주식 보기 화면의 하단 보조지표 패널을 실제 indicator API 값 기반으로 정리했다.
- 첫 화면 기본 보조지표는 RSI, MACD 2개로 고정했다.
- AI 관련 선택지는 오른쪽 패널에서 AI 밴드, 보수적 예측 2개만 남겼다.
- 회원, 관심종목, 알림, 설정, 데이터셋, 리포트, 라이브러리 진입점은 이번 CP 범위에서 숨겼다.
- `volume`은 `indicators` 테이블 컬럼이 없어도 `price_data`에서 read-only로 병합되도록 정리했다.

# 2. 변경 파일

- `backend/app/repositories/market_repo.py`
- `backend/app/schemas/stocks.py`
- `backend/tests/test_api.py`
- `backend/tests/test_services.py`
- `frontend/src/api/client.ts`
- `frontend/src/components/AppShell.tsx`
- `frontend/src/components/Chart.tsx`
- `frontend/src/components/IndicatorPanel.tsx`
- `frontend/src/components/StockView.tsx`
- `frontend/src/app/globals.css`

# 3. 실제 사용 가능한 indicator 목록

AAPL 1D 기준 `/api/v1/stocks/AAPL/indicators?timeframe=1D&limit=2`에서 확인한 값:

- `rsi`
- `macd_ratio`
- `bb_position`
- `ma_5_ratio`
- `ma_20_ratio`
- `ma_60_ratio`
- `vol_change`
- `volume`
- `atr_ratio`

프론트는 실제 non-null 값이 있는 지표만 보조지표 메뉴에 노출한다. 값이 모두 null이면 해당 지표는 숨긴다.

# 4. ATR 처리

초기 지시와 달리 현재 AAPL 1D API 응답에서는 `atr_ratio` 값이 존재한다.
따라서 ATR은 메뉴에 노출될 수 있다.
다만 구현 규칙은 값 기반이므로, 다른 ticker/timeframe에서 `atr_ratio`가 전부 null이면 자동으로 숨겨진다.
ATR 백필이나 새 계산 로직은 추가하지 않았다.

# 5. 미지원 또는 미노출 지표

- 하방 위험 상태, coverage, upper breach, lower breach 요약은 AI 보조지표처럼 보이지 않도록 주식 보기 패널에서 제외했다.
- AI 밴드 폭과 보수적 예측선 변화는 하단 보조지표가 아니라 메인 차트 overlay와 오른쪽 예측 요약으로만 다룬다.
- `regime_label`은 API 응답 후보로 남아 있지만 이번 CP에서는 차트 지표로 노출하지 않았다.
- 없는 지표를 프론트에서 임의 계산하지 않았다.

# 6. 화면 동작 확인

- AAPL 1D 가격 차트 표시 확인.
- AI 밴드와 보수적 예측 overlay 유지 확인.
- 하단 보조지표 패널에서 RSI, MACD 기본 표시 확인.
- 보조지표 추가 메뉴에서 기본, 추세, 변동성/위치 그룹 확인.
- 1M 전환 시 가격 차트가 유지되고 AI 밴드/보수적 예측 토글이 비활성화되는 것 확인.
- stock search 503 상태에서도 빨간 전체 오류 대신 작은 안내가 표시되고, 직접 티커 입력 가격 조회는 유지됨을 확인.

# 7. 검증 명령

- `cd C:\Users\user\lens\frontend; npm run build`
  - 샌드박스에서는 `spawn EPERM`으로 1회 실패.
  - 권한 승인 경로로 같은 명령 재실행 후 성공.
- `cd C:\Users\user\lens; $env:PYTHONPATH="C:\Users\user\lens\backend"; python -m unittest discover backend\tests`
  - 45개 테스트 통과.
- `cd C:\Users\user\lens; powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`
  - health, CORS, frontend, 1D prices, indicators, 1M prices, completed runs, demo prediction, evaluation 통과.
  - stock search는 503으로 실패.
  - latest run은 AAPL prediction이 없어 demo run을 사용한다는 경고 유지.
  - backtest row 없음 경고 유지.

# 8. 남은 리스크

- `stock_info` 기반 검색 API는 여전히 503이다. 화면에서는 직접 티커 입력 fallback으로 흡수한다.
- latest completed run인 `patchtst-1D-41d584bcb3cb`에는 AAPL prediction이 없어, 데모 overlay는 `patchtst-1D-239b58ab90f0`를 사용한다.
- 저장된 backtest row가 없어 백테스트 readiness 경고는 남아 있다.
- 현재 데모 prediction의 일부 값이 가격 범위와 크게 떨어져 차트 스케일에 영향을 줄 수 있다. 이번 CP에서는 overlay 계약을 바꾸지 않았다.
