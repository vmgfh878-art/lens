# CP111-P Lens Balance v1 백테스트 제품 반영 보고서

CP111-P는 CP109/CP110에서 탐색한 Line + Band Balance 계열 결과를 바탕으로, 백테스트 화면의 기본 전략을 `Lens Balance v1` 하나로 정리한 작업이다.

## 1. 적용 전략

- 전략명: `Lens Balance v1`
- 목적: 보수적 예측선으로 방향을 보고, AI 밴드로 불확실성과 하방 위험을 확인해 진입과 청산을 조절하는 단일 티커 실험 전략
- line layer: `patchtst-1D-efad3c29d803`
- band layer: `cnn_lstm-1D-d0c780dee5e8`
- composite 사용: 없음
- 포지션: 단일 티커 100% 또는 현금 100%
- 성격: 제품 기본 후보가 아니라 AI 지표 해석용 실험 전략 v1

## 2. 전략 룰

| 항목 | 값 |
| --- | --- |
| line_entry_threshold | -0.2% |
| line_hold_threshold | -1.4% |
| lower_risk_threshold | -5% |
| width_expansion_threshold | 1.10 |
| width_percentile_threshold | 0.75 |
| entry_band_filter | block_expansion |
| exit_risk_mode | lower_or_width |
| confirm_days | 2 |
| reentry_confirm_days | 2 |
| position_mode | binary |

구현상 line은 진입과 보유 판단의 중심으로 쓰고, band는 신규 진입 차단과 청산 확인 신호로 사용했다. 밴드 폭 확장만으로 즉시 매도하지 않고, line 약화와 함께 나타날 때 청산 후보로 본다.

## 3. 화면 반영

### 기본 전략 교체

- 기존 Risk Guard / Trend Guard / Line Trend / Band Risk 계열은 기본 UI 선택지에서 제거했다.
- 전략 선택지는 `Lens Balance v1`만 남겼다.
- 기존 포트폴리오 관점 문구가 아니라 단일 티커 전략 실험 문구를 유지했다.

### 전략 설명 패널

전략 설명 패널에 다음 정보를 표시했다.

- 어떤 AI 지표를 쓰는지: `보수적 예측선 v1`, `AI 밴드 v1`
- 어떤 제품 layer를 쓰는지: 예측선 버전과 밴드 버전
- 왜 현금 보유를 허용하는지: 하방 위험과 불확실성이 커졌을 때 시장 참여를 잠시 줄이는 방식으로 설명
- 판정: 제품 기본 후보가 아니라 실험 전략 v1
- 한계: AAPL/MSFT/NVDA 같은 대표 티커별 일관성이 약했던 점을 숨기지 않고 표시

직전 화면 피드백을 반영해, 별도 큰 `검증 요약` 패널은 만들지 않았다. 대신 전략 설명 패널 안에 낮은 위계의 작은 검증 결과 strip으로 넣었다.

## 4. 검증 결과 요약 표시

화면에는 아래 수치를 작은 요약으로 표시한다.

| 지표 | 값 |
| --- | --- |
| 전략 수익률 | 3.98% |
| 단순 보유 수익률 | 2.37% |
| 전략 MDD | -5.93% |
| 단순 보유 MDD | -13.08% |
| 손실 회피율 | 55.6% |
| 시장 참여율 | 44.7% |
| 강한 상승 방어율 | 67.1% |
| 검증 티커 | 95개 |

이 수치는 제품 성과 보장 문구가 아니라 CP109/CP110 실험 결과를 설명하는 참고값으로 표시했다.

## 5. 현재 티커 계산

백테스트 화면의 메인 카드와 차트는 선택한 티커 기준으로 프론트에서 read-only 계산한다.

- 가격: local snapshot 기반 가격 API
- 보수적 예측선: 제품 LM run prediction history
- AI 밴드: 제품 BM run prediction history
- 보조지표: indicator API
- DB write: 없음
- inference 실행: 없음
- 모델 학습: 없음
- Supabase price_data/indicators 대량 read: 없음

prediction이 없는 티커는 fake data를 만들지 않고 empty state로 처리한다.

## 6. UI 변경 파일

- `frontend/src/components/BacktestView.tsx`
  - Lens Balance v1 단일 전략 정의
  - line/band 제품 run 기반 prediction history 조회
  - 단일 티커 백테스트 계산
  - 전략 설명, 검증 요약, 현재 티커 결과 카드, 가격/성과 차트, 매매 기록 표시
  - Risk Guard/Trend Guard 기본 UI 제거
- `frontend/src/app/globals.css`
  - 백테스트 툴바, 전략 설명 패널, 검증 결과 strip, 차트/매매 기록 레이아웃 스타일 추가

## 7. 검증 결과

- `npm run build`: 통과
- `scripts/check_demo_readiness.ps1`: 통과
- 브라우저 확인:
  - AAPL 백테스트 화면 확인
  - MSFT 백테스트 화면 확인
  - NVDA 백테스트 화면 확인
  - 전략명 `Lens Balance v1` 표시 확인
  - 예측선/밴드 버전 표시 확인
  - 별도 큰 검증 요약 패널 미노출 확인
  - console error/warn 0건 확인

## 8. 남은 한계

- `Lens Balance v1`은 매수/매도 추천이 아니라 AI 지표 해석용 룰 실험이다.
- AAPL/MSFT/NVDA 같은 대표 티커별 결과가 항상 일관적이지 않았다.
- CP112 이후에는 전략 선택 UI를 늘리기보다, 이 전략이 어떤 구간에서 실패하는지 설명하는 분석 UX가 먼저 필요하다.
