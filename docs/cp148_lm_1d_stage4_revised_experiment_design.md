# CP148-LM-1D Stage 4 Revised Experiment Design

## 목적

Stage 4-0 risk-aware rerank 결과를 반영해 `cp148_s2_patchtst_no_fund_p32_s16`을 주 베이스로 두고 하방 안정성 개선 원인을 분해한다.

## 고정 조건

- beta=2.0, alpha/beta/delta=1.0/2.0/1.0 유지
- tail 전용 loss 추가 없음
- 기존 line_gate 생존 의미 변경 없음
- product save, DB write, inference 저장, live fetch 없음
- band/composite 실험 없음
- EODHD 500 local parquet 기준

## 후보 역할

- primary_stage4_base: `cp148_s2_patchtst_no_fund_p32_s16`
- secondary_stage4_base: `cp148_s2_patchtst_pvv_p32_s16`
- alpha/stress 참고선: `cp148_s2_patchtst_pvv_p16_s8`
- CNN-LSTM: 이번 LM 실험 제외, risk_only_reference로 보관

## A/B/C/D

| 실험 | 추가 피처 | 질문 |
| --- | --- | --- |
| A_selector_only | 없음 | checkpoint 선택 기준만 바꿔도 false_safe와 severe가 개선되는가? |
| B_atr_only | atr_ratio | 종목의 단기 불안정성, 갭, 장중 진폭을 넣으면 h1 약점과 false_safe가 개선되는가? |
| C_stress_delta | atr_ratio, vix_change_5d, credit_spread_change_20d, ma200_pct_change_20d | 시장 stress 변화와 시장 내부 붕괴 신호를 넣으면 stress 구간 하방 안정성이 개선되는가? |
| D_stock_fragility | atr_ratio, drawdown_20, downside_vol_20 | 개별 종목의 추세 훼손과 하락 변동성을 넣으면 false_safe가 개선되는가? |

## 평가 분해

전체 validation, calm/neutral/stress, vix_rising, breadth_worsening, h1, h2_h3, h4_h5를 기록한다.

## 성공 기준

- line_gate_pass=True
- fee_adjusted_return > 0
- false_safe_tail_rate < 0.30866056266521946
- severe_downside_recall >= 0.6850190785352241
- spread >= 0.008 권장
- upside_sacrifice 과도 증가 없음
