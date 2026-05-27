# 단일 티커 백테스트 전략 v2 계획

작성 시각: 2026-05-27

## 1. 결론

이번 백테스트 전략은 포트폴리오 회전 전략이 아니다.

제품 본류 전략은 각 티커마다 독립적으로 실행되는 `long/cash` 전략이다. 예를 들어 AAPL은 AAPL 자체의 신호로 100% 보유 또는 100% 현금을 결정하고, MSFT도 같은 방식으로 독립 실행한다. 500티커 평가는 포트폴리오 수익률이 아니라 같은 룰을 500개 티커에 각각 적용한 뒤 평균, 중앙값, 통과율, 손실 회피율, 시장 참여율을 보는 방식이다.

따라서 CP160 계열의 500티커 횡단면 회전 전략은 현재 백테스트 상세 화면의 기본 전략으로 쓰지 않는다. 해당 전략은 나중에 별도 scanner 또는 포트폴리오 연구 화면으로 분리해야 한다.

## 2. 이전 CP 기준 정리

| CP | 의미 | 제품 반영 판단 |
|---|---|---|
| CP157 | AI 없이 가격/보조지표만 쓰는 단일 티커 long/cash 기준선 | 지표-only 기본 전략 후보 |
| CP158 | 더 깊은 지표-only 탐색. 방어형 후보는 손실 회피가 강하지만 참여율이 낮음 | 방어형 리스크 필터 후보 |
| CP106 | AI band 단독 위험 전략 | band 단독은 보조지표 결합 없이는 약함 |
| CP107 | AI line 단독 추세 전략 | line 단독은 참여율이 낮은 문제 |
| CP109 | AI line + band balance 탐색 | 구조는 맞지만 대표 티커 참여율 문제가 있었음 |
| CP110 | AI balance에 RSI/ATR/추세/거래량을 결합 | 과방어 후보는 많았으나 제품 기본으로는 신중 |
| CP122 | 12티커 신호 화면은 subset scanner이며 상세는 단일 티커 백테스트 | 현재 UI 계약의 원형 |
| CP160 | 500티커 횡단면 선별/회전 전략 | 단일 티커 상세에 붙이면 안 됨 |

## 3. 평가 계약

공통 계약:

- 단일 티커별 100% 보유 또는 100% 현금.
- 포트폴리오 top-k, 횡단면 랭킹, 종목군 동시 보유는 이번 전략에서 제외.
- 신호일 `t`의 포지션은 다음 거래일 수익률부터 적용.
- 기본 거래비용은 왕복이 아니라 포지션 변경 시 10bp로 둔다.
- 신규 진입은 2일 확인, 청산은 2~3일 확인을 기본값으로 둔다.
- 500티커 검증은 티커별 백테스트 결과의 평균/분포로 표시한다.

필수 집계 지표:

- 전략 수익률
- 단순 보유 수익률
- 전략 MDD
- 단순 보유 MDD
- MDD 개선폭
- 손실 회피율
- 시장 참여율
- Sharpe
- 거래 횟수
- 평균 보유 기간
- 통과 티커 비율

대표 티커 필수 점검:

- AAPL
- MSFT
- NVDA
- AMZN
- GOOGL
- META
- NFLX
- TSLA

대표 티커에서 시장 참여율이 0~15%로 무너지는 전략은 제품 기본 후보에서 제외한다.

## 4. 전략 1: 지표 균형 v2

역할:

- AI 없이 작동하는 기준선 전략.
- 모든 AI 전략은 이 전략보다 무엇이 나은지 설명할 수 있어야 한다.

사용 지표:

- 60일 추세: `ma_60_ratio`
- 20일 추세: `ma_20_ratio`
- MACD 비율: `macd_ratio`
- RSI
- ATR 비율: `atr_ratio`
- Bollinger 위치: `bb_position`

진입 조건:

```text
trend_entry =
  ma_60_ratio >= 0.02
  and ma_20_ratio >= -0.02
  and macd_ratio >= 0
  and rsi < 75

pullback_entry =
  ma_60_ratio >= 0.02
  and bb_position <= 0.35
  and rsi < 55

entry = trend_entry or pullback_entry
entry_confirm_days = 2
```

청산 조건:

```text
trend_exit =
  ma_60_ratio <= -0.05
  or ma_20_ratio <= -0.05

volatility_exit =
  atr_ratio >= 0.07
  and ma_20_ratio < 0

exit = trend_exit or volatility_exit
exit_confirm_days = 3
```

이 전략을 남기는 이유:

- CP157의 900개 후보 중 균형 점수가 가장 좋았던 구조다.
- 기존 프론트의 `지표 기준선 v1`과 가장 가까워 설명 가능성이 높다.
- 시장 참여율이 약 60%대라 AAPL/MSFT 같은 대형주가 완전히 사라지는 문제를 줄인다.

주의:

- 강한 상승장에서는 단순 보유 평균을 이기지 못할 수 있다.
- 따라서 이 전략은 “수익 극대화”가 아니라 “AI 없는 비교 기준선”으로 표시해야 한다.

기존 CP157 기준:

| 지표 | 값 |
|---|---:|
| 평균 전략 수익률 | 22.95% |
| 평균 단순 보유 수익률 | 30.91% |
| 평균 전략 MDD | -20.13% |
| 평균 단순 보유 MDD | -25.95% |
| 평균 MDD 개선폭 | +5.82%p |
| 평균 손실 회피율 | 37.15% |
| 평균 시장 참여율 | 63.52% |
| 통과 티커 비율 | 31.40% |

판정:

`기준선 전략으로 채택 가능`

## 5. 전략 2: AI 균형 v2

역할:

- AI line과 AI band를 같이 쓰는 단일 티커 전략.
- AI line은 방향성/보유 의지, AI band는 하방 위험과 불확실성 확인용으로만 쓴다.
- line을 band 안에 넣거나 composite로 합치지 않는다.

사용 지표:

- AI line score: CP175 1D line
- AI band lower/upper: CP153 1D band historical
- 60일 추세
- 20일 추세
- ATR 비율
- band width expansion

진입 조건:

```text
line_entry_ok =
  line_score >= -0.02

trend_support =
  ma_60_ratio >= 0.00
  and ma_20_ratio >= -0.04

band_not_extreme =
  band_lower_return >= -0.06
  or band_width_expansion < 1.25

entry = line_entry_ok and trend_support and band_not_extreme
entry_confirm_days = 2
```

청산 조건:

```text
line_weak =
  line_score < -0.06

band_risk =
  band_lower_return < -0.06
  or band_width_expansion > 1.25

price_break =
  ma_20_ratio < -0.10

volatility_break =
  atr_ratio > 0.12
  and ma_20_ratio < 0

exit =
  (line_weak and band_risk)
  or price_break
  or volatility_break

exit_confirm_days = 3
```

이 전략을 남기는 이유:

- CP109/110에서 확인한 방향이 맞다.
- line 단독은 참여율이 너무 낮아질 수 있고, band 단독은 위험 회피력이 약했다.
- line은 진입/보유 방향, band는 위험 확인으로 분리해 쓰는 편이 제품 의미에 맞다.

주의:

- 기존 CP109에서는 AAPL/MSFT 참여율 문제가 있었다.
- 현재 빠른 로컬 재평가에서도 AI 균형 전략은 평균 단순 보유를 이기지는 못했다.
- 따라서 이 전략은 “AI가 지표-only보다 항상 우월하다”가 아니라 “AI line/band를 함께 썼을 때 방어 신호가 어떻게 달라지는지 보여주는 전략”으로 시작해야 한다.

빠른 로컬 재평가 참고값:

| 지표 | 값 |
|---|---:|
| 평가 티커 | 501 |
| 평가 기간 | 2025-05-01 ~ 2026-05-01 |
| 평균 전략 수익률 | 19.6% |
| 평균 단순 보유 수익률 | 42.3% |
| 평균 전략 MDD | -16.8% |
| 평균 단순 보유 MDD | -25.2% |
| 평균 MDD 개선폭 | +8.4%p |
| 평균 손실 회피율 | 39.8% |
| 평균 시장 참여율 | 59.7% |
| 통과 티커 비율 | 9.6% |

판정:

`제품 전략 후보로는 보류, AI 설명용/방어형 후보로 유지`

## 6. 전략 3: AI 밴드 방어 v1

역할:

- AI 지표 중 band만 제품 전략 입력으로 사용하는 전략.
- AI line은 쓰지 않는다.
- 방향성은 가격/보조지표가 담당하고, AI band는 진입 veto와 청산 확인만 담당한다.

사용 지표:

- AI band lower/upper: CP153 1D band historical
- band lower return
- band width expansion
- 60일 추세
- 20일 추세
- RSI
- Bollinger 위치
- ATR 비율

진입 조건:

```text
indicator_trend_entry =
  ma_60_ratio >= 0.02
  and ma_20_ratio >= -0.03
  and rsi < 82

indicator_pullback_entry =
  ma_60_ratio >= 0.02
  and bb_position <= 0.45
  and rsi < 60

band_veto_clear =
  band_lower_return >= -0.08
  or band_width_expansion < 1.60

entry =
  (indicator_trend_entry or indicator_pullback_entry)
  and band_veto_clear

entry_confirm_days = 2
```

청산 조건:

```text
band_stress =
  band_lower_return < -0.08
  and band_width_expansion > 1.60

trend_break =
  ma_60_ratio < -0.05
  or ma_20_ratio < -0.08

volatility_break =
  atr_ratio > 0.12
  and ma_20_ratio < 0

exit =
  band_stress
  or trend_break
  or volatility_break

exit_confirm_days = 3
```

이 전략을 남기는 이유:

- “AI 지표 중 하나만 쓰는 전략” 조건에 가장 잘 맞는다.
- line을 쓰지 않기 때문에 line/band 해석이 섞이지 않는다.
- CP106의 band 단독 실패를 그대로 반복하지 않기 위해 방향성은 가격 지표가 담당하고, AI band는 위험 veto로만 쓴다.

빠른 로컬 재평가 참고값:

| 지표 | 값 |
|---|---:|
| 평가 티커 | 501 |
| 평가 기간 | 2025-05-01 ~ 2026-05-01 |
| 평균 전략 수익률 | 28.3% |
| 평균 단순 보유 수익률 | 42.3% |
| 평균 전략 MDD | -20.2% |
| 평균 단순 보유 MDD | -25.2% |
| 평균 MDD 개선폭 | +5.0%p |
| 평균 손실 회피율 | 30.9% |
| 평균 시장 참여율 | 68.7% |
| 통과 티커 비율 | 23.0% |

판정:

`세 번째 전략 후보로 채택 가능`

## 7. 전략 3개 최종 후보 요약

| 전략 | AI 사용 | 주 목적 | 제품 상태 |
|---|---|---|---|
| 지표 균형 v2 | 없음 | AI 없는 기준선, 보조지표-only | 기본 전략 후보 |
| AI 균형 v2 | line + band | AI line/band 결합 방어 신호 | 보류/설명형 후보 |
| AI 밴드 방어 v1 | band만 사용 | band risk veto + 가격 추세 | 후보 |

## 8. UI 연결 원칙

전략 선택 드롭다운에는 위 3개만 제품 본류로 둔다.

제거 또는 격리할 항목:

- `indicator_rotation_500_v1`
- `ai_hybrid_rotation_500_v1`
- `ai_line_only_rotation_500_v1`

위 3개는 CP160 계열의 횡단면 선별 전략이므로 단일 티커 상세 백테스트에 붙이지 않는다. 삭제하지 않더라도 `scanner_research` 또는 `legacy_research`로 격리한다.

신호판은 포트폴리오 추천이 아니라 아래 상태만 보여준다.

- 매수 후보
- 보유 유지
- 위험 확대
- 관망

각 그룹은 기본 5개만 노출하고 나머지는 접는다.

## 9. 새 데이터 최신화 원칙

새 데이터가 들어오면 전략 자체를 매번 바꾸는 것이 아니라, 같은 룰을 최신 price/indicator/AI parquet에 다시 적용한다.

필요한 최신화 대상:

- price parquet
- indicator parquet
- CP204 line payload 또는 후속 line payload
- CP204 band historical payload 또는 후속 band payload
- 500티커 신호 cache
- 티커별 상세 백테스트 cache

전략 파라미터를 바꾸려면 별도 CP로 재탐색해야 한다.

## 10. 다음 작업

1. 기존 포트폴리오 회전 전략을 제품 백테스트 UI에서 제거한다.
2. 위 3개 전략을 단일 티커 long/cash 엔진으로 붙인다.
3. 500티커 scan은 같은 룰을 각 티커에 독립 적용한 결과로 만든다.
4. AAPL/MSFT/NVDA/AMZN/GOOGL/META/NFLX/TSLA 대표 티커 상세 결과를 표로 고정한다.
5. 프론트에는 “포트폴리오 추천 아님, 단일 티커 룰 기반 시뮬레이션” 문구를 유지한다.
