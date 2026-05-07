# CP82-P 단일 티커 Risk Guard v1 백테스트 데모

CP82-P는 백테스트 화면을 포트폴리오 백테스트가 아니라 단일 티커 룰 기반 전략 실험 화면으로 정리한 작업이다.

## 1. Risk Guard v1 규칙 정의

- 목적: 수익 극대화보다 하방 위험 회피를 먼저 보는 단순 룰 기반 전략
- 사용 지표:
  - 보수적 예측선
  - AI 밴드 하단
  - RSI
- 진입/보유 조건:
  - 보수적 예측선 >= 0%
  - AI 밴드 하단 > -3%
  - RSI < 75
- 회피/청산 조건:
  - 위 조건 중 하나라도 깨지면 현금
  - RSI 80 이상은 과열 회피 사유로 별도 표시
- 포지션:
  - 현금 100% 또는 단일 티커 100%
  - 부분 비중 조절 없음
  - 다종목 동시 보유 없음
  - 포트폴리오 리밸런싱 없음
- 거래 비용:
  - 기존 백테스트 기본값과 맞춰 `10bp` 사용

## 2. 단일 티커 백테스트로 제한한 이유

- Phase 1의 목적은 포트폴리오 최적화가 아니라 “이 티커에서 이 규칙을 썼다면 어땠을까?”를 확인하는 것이다.
- Lens의 현재 제품 후보는 line layer와 band layer가 1D 기준으로 준비되어 있다.
- 다종목/리밸런싱/유니버스 구성은 성능 해석과 데이터 계약이 더 복잡하므로 이번 CP에서는 제외했다.

## 3. 사용한 데이터/API

프론트에서 read-only API만 사용했다.

- 가격:
  - `GET /api/v1/stocks/{ticker}/prices`
- 보조지표:
  - `GET /api/v1/stocks/{ticker}/indicators?timeframe=1D&limit=1000`
- 보수적 예측선 history:
  - `GET /api/v1/stocks/{ticker}/predictions/history?run_id=patchtst-1D-efad3c29d803&limit=200`
- AI 밴드 history:
  - `GET /api/v1/stocks/{ticker}/predictions/history?run_id=cnn_lstm-1D-d0c780dee5e8&limit=200`
- 티커 검색:
  - `GET /api/v1/stocks?search=...`

DB 쓰기, fake data 생성, inference 실행은 하지 않았다.

## 4. 신호 생성 방식

- 신호 날짜는 prediction `asof_date` 기준이다.
- 보수적 예측선 값은 `conservative_series`를 우선 사용한다.
- `conservative_series`가 없으면 `line_series`를 fallback으로 사용한다.
- AI 밴드 하단은 `lower_band_series`의 horizon 내 최저값을 사용한다.
- 보수적 예측선 수익률과 AI 밴드 하단 수익률은 해당 `asof_date`의 종가 대비 비율로 계산한다.
- RSI는 같은 날짜의 indicator row를 사용하고, 0~1 값이면 0~100 스케일로 변환한다.
- prediction이 없는 날짜에는 새 신호를 만들지 않고 직전 신호를 유지한다.
- 최초 신호 전에는 현금 상태로 둔다.

## 5. lookahead leakage 방지 방식

- 신호는 `asof_date`에 이미 저장된 prediction과 해당 날짜의 가격/RSI만 사용한다.
- 미래 실현 수익률은 신호 생성에 사용하지 않는다.
- 신호는 다음 거래 구간 수익률에 적용한다.
- horizon 5의 미래 가격 실현값은 규칙 판단에 사용하지 않는다.

## 6. 티커 선택 UX

- AAPL 고정 화면을 제거하고 티커 직접 입력을 추가했다.
- 검색 결과가 있으면 버튼으로 선택할 수 있다.
- stock search가 실패해도 직접 입력 조회는 유지한다.
- 가격 데이터가 없으면 조용한 empty state를 보여준다.
- 해당 티커의 AI prediction row가 없으면 가격/RSI만 표시하고 아래 문구를 보여준다.
  - `이 티커는 아직 AI 예측 저장 결과가 없습니다. 가격/RSI만 표시합니다.`

## 7. 표시한 지표

상단 핵심 카드:

- 최대 낙폭
- 현금 대기 비율
- 손실 회피율
- 전략 수익률
- 단순 보유 수익률
- 초과 수익률

리스크/거래 카드:

- 수수료 반영 수익률
- 수수료 반영 샤프
- 최악 거래 손실
- 평균 보유 기간
- 거래 횟수
- 회피한 하락일

수익률보다 최대 낙폭, 현금 대기, 손실 회피율을 먼저 배치했다.

## 8. 차트 표시 내용

- 가격 차트
- 진입/청산 마커
- 보유/현금 구간 strip
- 전략 누적 수익률 vs 단순 보유 누적 수익률
- RSI 차트
- 최근 신호 테이블

화려한 차트보다는 데모에서 규칙과 결과를 해석할 수 있는 최소 구성을 우선했다.

## 9. 처리하지 못한 항목

- 저장된 backend backtest row를 생성하거나 갱신하지 않았다.
- read-only 계산 API는 추가하지 않았다.
- 거래 마커는 단순 SVG 마커이며 TradingView식 정교한 marker는 아니다.
- 전략 결과는 프론트 계산 결과라, 추후 서버 계산 API와 수치 일치 테스트가 필요하다.
- 1W/1M은 현재 Risk Guard v1 계산 대상이 아니며 가격/RSI 확인 중심으로 둔다.

## 10. Phase 2 포트폴리오 백테스트 backlog

- 포트폴리오/다종목 백테스트
- 리밸런싱 규칙
- universe 선택
- 종목별 비중 조절
- 거래 비용/슬리피지 고도화
- 서버 측 read-only 계산 API
- 전략 파라미터 저장과 비교 리포트

## 11. 검증 결과

- `npm run build`: 통과
- `scripts/check_demo_readiness.ps1`: 통과
- 백엔드 API 수정 없음. backend unittest는 생략했다.
- 브라우저 확인:
  - AAPL 백테스트 화면 확인
  - Risk Guard v1 전략 설명 패널 확인
  - 수익률/최대 낙폭/거래 횟수 카드 표시 확인
  - 가격 차트와 누적 수익률 차트 표시 확인
  - MSFT 직접 입력 조회 확인
  - AMP prediction 없음 상태 확인
  - 1W/1M 전환 시 1D 제품 후보 기준 안내 확인
  - 브라우저 console error/warn 0건 확인

