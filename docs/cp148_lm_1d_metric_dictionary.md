# CP148-LM 1D Metric Dictionary

이 문서는 1D line_model 후보를 평가할 때 사용할 line_metrics의 의미와 제품 의사결정 가중치를 정의한다. band coverage, lower_breach_rate, upper_breach_rate, line_inside_band, composite/overlay 지표는 line_model ranking에 사용하지 않는다.

| metric | 의미 | 방향 | 단독 판단 금지 여부 | 제품 의사결정 가중치 |
|---|---|---|---|---|
| spearman_ic | 같은 날짜 cross-section에서 예측 line과 실제 raw_future_return의 순위 상관 | 높을수록 좋음, 제품 후보는 > 0 | 단독 판단 금지. risk metric과 함께 봄 | 높음 |
| ic_mean | 기간별 IC 평균 | 높을수록 좋음, 제품 후보는 > 0 | 단독 판단 금지 | 높음 |
| ic_ir | IC 평균을 변동성으로 나눈 안정성 지표 | 높을수록 좋음 | 표본 수와 regime에 영향 | 중간 |
| ic_t_stat | IC가 0보다 큰지에 대한 통계적 강도 참고 | 높을수록 좋음 | 금융 시계열 자기상관 때문에 과신 금지 | 중간 |
| long_short_spread | 예측 상위와 하위 그룹의 실제 수익률 차이 | 높을수록 좋음, 제품 후보는 > 0 | 단독 판단 금지 | 높음 |
| spread_ir | spread 평균 대비 변동성 안정성 | 높을수록 좋음 | 단독 판단 금지 | 중간 |
| spread_t_stat | spread 양수성의 통계적 참고값 | 높을수록 좋음 | 비용/turnover 반영 전이면 과신 금지 | 중간 |
| fee_adjusted_return | 거래비용 또는 fee proxy 반영 후 long-short 성과 | 높을수록 좋음, 제품 후보는 > 0 | fee proxy 가정 확인 필요 | 높음 |
| fee_adjusted_sharpe | fee 반영 성과의 변동성 대비 효율 | 높을수록 좋음 | low turnover 후보와 함께 비교 필요 | 중간 |
| direction_accuracy | 예측 방향과 실제 방향 일치율 | 높을수록 좋음 | 수익률 크기와 순위 정보를 잃으므로 단독 판단 금지 | 낮음~중간 |
| mae | 예측 line과 실제 target의 절대오차 | 낮을수록 좋음 | point forecast 지표라 보조만 사용 | 낮음 |
| smape | scale 보정 상대 오차 | 낮을수록 좋음 | return near zero에서 왜곡 가능 | 낮음 |
| false_safe_tail_rate | 실제 tail downside인데 line이 안전하게 보인 비율 | 낮을수록 좋음, 제품 후보는 <= 0.20 | 단독 판단 금지. severe recall과 함께 봄 | 매우 높음 |
| false_safe_severe_rate | severe downside 구간에서 안전 오판 비율 | 낮을수록 좋음 | threshold 정의 확인 필요 | 매우 높음 |
| severe_downside_recall | 큰 하락을 위험으로 포착한 비율 | 높을수록 좋음, 제품 후보는 >= 0.75 | 보수화로 모든 것을 위험 처리하는지 함께 봄 | 매우 높음 |
| downside_capture_rate | downside 구간을 line이 얼마나 포착하는지 | 높을수록 좋음 | false_safe와 conservative_bias 함께 해석 | 높음 |
| conservative_bias | 예측이 실제보다 얼마나 낮게 치우쳤는지 | 음수면 보수적 | 너무 낮으면 over_conservative_fail | 높음 |
| upside_sacrifice | 보수화 때문에 상승 기회를 희생한 정도 | 낮을수록 좋음 | false_safe 개선과 tradeoff로 판단 | 높음 |
| turnover | 예측 ranking 변화로 발생하는 매매 회전율 | 낮을수록 비용에 유리 | 낮기만 하면 signal 약화 가능 | 중간 |
| fee proxy | turnover와 비용 가정으로 산출한 성과 차감치 | 낮을수록 유리 | 비용 가정 명시 필요 | 중간 |
| line_gate_pass | line_model 후보가 CP61/CP52 line 목적 기준을 통과했는지 | PASS 필요 | PASS만으로 제품 후보 아님 | 필수 hard gate |

## 의사결정 우선순위

제품 v1 후보는 다음 순서로 본다.

1. source/provider/cache 계약이 명확한가
2. line_gate를 통과하는가
3. IC, spread, fee_adjusted_return이 모두 양수인가
4. false_safe_tail_rate가 0.20 이하인가
5. severe_downside_recall이 0.75 이상인가
6. 기존 1D 제품 후보와 baseline보다 위험 지표가 개선됐는가
7. upside_sacrifice와 conservative_bias가 과도하지 않은가
8. seed stability가 유지되는가

## 해석 금지 규칙

- line_model을 band coverage 실패로 탈락시키지 않는다.
- lower_band / upper_band가 출력되어도 line 평가에는 사용하지 않는다.
- line_inside_band는 모델 성능 지표가 아니다.
- composite prediction, overlay, risk_first_lower_preserve, include_line_clamp, upper_buffer 정책으로 line 후보를 살리지 않는다.
- test에서만 좋은 후보를 제품 후보로 승격하지 않는다.
