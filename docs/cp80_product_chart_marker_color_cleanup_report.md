CP80-P는 작은 차트 표시 보정과 보조지표 구조 점검을 위한 안정화 CP다.

## 1. 오늘 날짜 marker 적용 방식

- `frontend/src/components/Chart.tsx`에서 차트 기준선 표시용 날짜를 클라이언트의 오늘 날짜로 계산하도록 추가했다.
- 오늘 날짜는 `YYYY-MM-DD` 로컬 날짜 문자열로 만들고, 가격/예측 데이터가 없어도 차트 time scale에 들어가도록 whitespace 날짜에만 추가한다.
- `timeToCoordinate(today)` 결과가 차트 폭 밖이면 marker를 `display: none` 처리한다.
- 따라서 사용자가 과거 구간으로 pan해서 오늘 날짜가 화면 밖으로 나가면 `오늘` marker도 사라진다.
- 1M은 기존 정책대로 price-only이므로 오늘 이후 예측 marker를 표시하지 않는다.

## 2. prediction.asof_date와 표시 기준선 분리

- prediction의 `forecast_dates`, `asof_date`, `line_series`, `upper_band_series`, `lower_band_series` 값은 변경하지 않았다.
- 차트 위 vertical marker만 `오늘`로 표시한다.
- 우측 예측 출처 패널은 실제 저장된 모델 기준일을 `모델 기준일`로 표시한다.
- AAPL 1D 브라우저 확인 기준 모델 기준일은 `2026-04-02`로 표시됐다.

## 3. 색상 변경 전/후 역할 정리

- 가격 라인: 기존 어두운 회색을 유지하되 라인 차트에서 `#111827`로 더 선명하게 조정했다.
- 예측선: 최신 예측을 진한 청록 `#047857`, 과거 예측을 같은 계열 낮은 opacity로 조정했다.
- AI 밴드: 최신 upper/lower를 남색 `#1e40af`와 두께 3으로 강화했다.
- 과거 AI 밴드: 남색 계열 opacity를 조금 올려 최신 밴드와 같은 계열로 읽히게 했다.
- 보수적 기준선: slate 계열 dotted line으로 유지해 주 예측선보다 덜 강조되게 했다.
- legend 색상도 위 역할에 맞춰 함께 조정했다.

## 4. 보조지표 텍스트 겹침 수정

- `frontend/src/components/IndicatorPanel.tsx`에서 하단 최소값 축 라벨을 제거했다.
- 기존에는 시작일 라벨과 최소값 라벨이 같은 위치에 겹칠 수 있었다.
- 현재는 상단 최대값과 하단 시작일/종료일만 표시해 작은 패널에서도 텍스트 충돌을 줄였다.
- RSI, MACD, AI 밴드 폭 동시 표시 상태에서 시작일/축 라벨 겹침이 완화됐다.

## 5. 보조지표 대개편 backlog

- 보조지표가 여러 개 켜지면 가격 차트와 함께 보기 어렵다.
- 패널이 각각 독립 카드로 쌓여 화면 밀도가 낮다.
- TradingView처럼 가격 chart 아래에 volume, RSI, MACD를 compact stack으로 쌓는 구조와 아직 차이가 있다.
- Bollinger band는 가격 chart overlay로 들어가는 편이 자연스럽다.
- Volume은 가격 chart 바로 아래 bar로 들어가는 편이 자연스럽다.
- AI band width는 별도 compact 보조지표로 유지하는 편이 좋다.
- 다음 후보: CP81-P 보조지표 대개편
  - Bollinger overlay
  - Volume bar
  - compact indicator stack
  - indicator picker UX 정리

## 6. 검증 결과

- `cd C:\Users\user\lens\frontend; npm run build`
  - 통과
- `cd C:\Users\user\lens; powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1`
  - 통과
  - backend health OK
  - frontend OK
  - AAPL 1D prices OK
  - indicators OK
  - 1M prices OK
  - LM/BM prediction, evaluation, history OK
- 브라우저 확인
  - AAPL 1D: 가격 차트 표시, 예측선, AI 위험 범위, 오늘 이후 예측, RSI, MACD 표시 확인
  - AAPL 1W: 런타임 오류 없이 가격 표시, `주간 AI 예측은 준비 중입니다.` 확인
  - AAPL 1M: 런타임 오류 없이 가격 표시, `월간 화면은 현재 가격 전용입니다.` 확인, 오늘 이후 예측 미표시 확인
  - `1D -> 1W -> 1M -> 1D -> 1W -> 1M -> 1D` 반복 전환 후 console error/warn 0건
  - 1D에서 과거 방향 pan 후 `오늘` marker가 화면 밖으로 사라지는 것 확인
- backend unittest
  - 이번 CP는 프론트 표시 보정만 했고 API/백엔드 코드를 수정하지 않아 생략했다.

## 7. 남은 아쉬운 점

- lightweight-charts의 band fill은 아직 구현하지 않았다. 이번 CP는 안정화 우선이라 upper/lower 라인 강조만 적용했다.
- 보조지표는 아직 독립 카드형 구조라 한 화면 밀도가 낮다.
- ATR은 값이 내려오는 경우에만 선택지로 보이지만, 기본 검증은 RSI/MACD/AI 밴드 폭 중심으로 진행했다.
- 프론트 dev 서버가 한 번 HTML 200, `_next/static` CSS/JS 404 상태로 떠 있어 브라우저가 무스타일 HTML만 보여줬다. 포트 3000의 기존 node 프로세스를 종료하고 dev 서버를 재기동한 뒤 정상 확인했다.
