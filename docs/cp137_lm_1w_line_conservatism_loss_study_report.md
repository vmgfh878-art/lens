# CP137-LM 1W 라인 보수 손실 실험

## 요약

- 상태: `PASS`
- recommended_default: `beta_2p0` / run_id `patchtst-1W-36ad14278e44`
- 후보 선택은 validation 중심이며 test는 직접 선택에 쓰지 않았다.
- save-run, DB write, inference 저장, W&B, composite, 프론트 수정, live yfinance fetch, EODHD 호출은 하지 않았다.

## 데이터 기준

- source_data_hash: `13a7f83d` / expected `13a7f83d`
- context_checksum: `ecb532122fca5eee` / expected `ecb532122fca5eee`
- source/context CP136 일치: `True` / `True`
- feature/target NaN/Inf: `0` / `0`
- test_exposure_count: `23280`

## beta 후보 결과

| candidate | class | run_id | beta | gate | val_ic | d_ic | val_spread | val_fee | val_false_safe | d_false_safe | val_severe | bias | upside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beta_1p5 | selectable_verified | patchtst-1W-50231a0606dc | 1.5 | True | 0.037746 | 0.008346 | 0.005807 | 0.156807 | 0.122516 | 0.043459 | 0.853642 | -0.131313 | 0.218361 |
| beta_2p0 | recommended_default | patchtst-1W-36ad14278e44 | 2.0 | True | 0.029400 | 0.000000 | 0.005852 | 0.183052 | 0.079057 | 0.000000 | 0.901293 | -0.171929 | 0.262746 |
| beta_2p5 | selectable_verified | patchtst-1W-86de6a2fa298 | 2.5 | True | 0.020313 | -0.009087 | 0.004109 | 0.069744 | 0.048701 | -0.030356 | 0.940095 | -0.204189 | 0.298000 |
| beta_3p0 | risk_conservative_variant | patchtst-1W-ff04c1574652 | 3.0 | True | 0.015174 | -0.014225 | 0.000644 | -0.130108 | 0.033195 | -0.045862 | 0.957794 | -0.229940 | 0.325757 |

## CP136 기준 대비 validation 개선/악화

### beta_1p5

| metric | CP136 val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.029400 | 0.037746 | 0.008346 | improved |
| ic_ir | 0.158777 | 0.200423 | 0.041647 | improved |
| ic_t_stat | 1.219586 | 1.539480 | 0.319894 | improved |
| long_short_spread | 0.005852 | 0.005807 | -0.000045 | worsened |
| spread_ir | 0.133667 | 0.114194 | -0.019474 | worsened |
| spread_t_stat | 1.026718 | 0.877137 | -0.149580 | worsened |
| fee_adjusted_return | 0.183052 | 0.156807 | -0.026245 | worsened |
| fee_adjusted_sharpe | 0.087691 | 0.074577 | -0.013114 | worsened |
| false_safe_tail_rate | 0.079057 | 0.122516 | 0.043459 | worsened |
| false_safe_severe_rate | 0.098707 | 0.146358 | 0.047651 | worsened |
| severe_downside_recall | 0.901293 | 0.853642 | -0.047651 | worsened |
| conservative_bias | -0.171929 | -0.131313 | 0.040617 | less_conservative |
| upside_sacrifice | 0.262746 | 0.218361 | -0.044386 | improved |
| direction_accuracy |  | 0.427398 |  | unknown |

### beta_2p0

| metric | CP136 val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.029400 | 0.029400 | 0.000000 | flat |
| ic_ir | 0.158777 | 0.158777 | 0.000000 | flat |
| ic_t_stat | 1.219586 | 1.219586 | 0.000000 | flat |
| long_short_spread | 0.005852 | 0.005852 | 0.000000 | flat |
| spread_ir | 0.133667 | 0.133667 | 0.000000 | flat |
| spread_t_stat | 1.026718 | 1.026718 | 0.000000 | flat |
| fee_adjusted_return | 0.183052 | 0.183052 | 0.000000 | flat |
| fee_adjusted_sharpe | 0.087691 | 0.087691 | 0.000000 | flat |
| false_safe_tail_rate | 0.079057 | 0.079057 | 0.000000 | flat |
| false_safe_severe_rate | 0.098707 | 0.098707 | 0.000000 | flat |
| severe_downside_recall | 0.901293 | 0.901293 | 0.000000 | flat |
| conservative_bias | -0.171929 | -0.171929 | 0.000000 | flat |
| upside_sacrifice | 0.262746 | 0.262746 | 0.000000 | flat |
| direction_accuracy |  | 0.402237 |  | unknown |

### beta_2p5

| metric | CP136 val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.029400 | 0.020313 | -0.009087 | worsened |
| ic_ir | 0.158777 | 0.112089 | -0.046688 | worsened |
| ic_t_stat | 1.219586 | 0.860971 | -0.358615 | worsened |
| long_short_spread | 0.005852 | 0.004109 | -0.001743 | worsened |
| spread_ir | 0.133667 | 0.095600 | -0.038067 | worsened |
| spread_t_stat | 1.026718 | 0.734316 | -0.292401 | worsened |
| fee_adjusted_return | 0.183052 | 0.069744 | -0.113307 | worsened |
| fee_adjusted_sharpe | 0.087691 | 0.048288 | -0.039403 | worsened |
| false_safe_tail_rate | 0.079057 | 0.048701 | -0.030356 | improved |
| false_safe_severe_rate | 0.098707 | 0.059905 | -0.038802 | improved |
| severe_downside_recall | 0.901293 | 0.940095 | 0.038802 | improved |
| conservative_bias | -0.171929 | -0.204189 | -0.032260 | more_conservative |
| upside_sacrifice | 0.262746 | 0.298000 | 0.035254 | worsened |
| direction_accuracy |  | 0.397519 |  | unknown |

### beta_3p0

| metric | CP136 val | current val | delta | 판정 |
| --- | --- | --- | --- | --- |
| ic_mean | 0.029400 | 0.015174 | -0.014225 | worsened |
| ic_ir | 0.158777 | 0.083953 | -0.074823 | worsened |
| ic_t_stat | 1.219586 | 0.644857 | -0.574729 | worsened |
| long_short_spread | 0.005852 | 0.000644 | -0.005208 | worsened |
| spread_ir | 0.133667 | 0.014636 | -0.119031 | worsened |
| spread_t_stat | 1.026718 | 0.112423 | -0.914294 | worsened |
| fee_adjusted_return | 0.183052 | -0.130108 | -0.313159 | worsened |
| fee_adjusted_sharpe | 0.087691 | -0.032100 | -0.119791 | worsened |
| false_safe_tail_rate | 0.079057 | 0.033195 | -0.045862 | improved |
| false_safe_severe_rate | 0.098707 | 0.042206 | -0.056501 | improved |
| severe_downside_recall | 0.901293 | 0.957794 | 0.056501 | improved |
| conservative_bias | -0.171929 | -0.229940 | -0.058011 | more_conservative |
| upside_sacrifice | 0.262746 | 0.325757 | 0.063011 | worsened |
| direction_accuracy |  | 0.392102 |  | unknown |

## h1_h4 버킷 지표

| candidate | bucket | ic_mean | long_short_spread | false_safe_tail_rate | severe_downside_recall |
| --- | --- | --- | --- | --- | --- |
| beta_1p5 | h1_h4 | 0.022771 | 0.003818 | 0.083898 | 0.852140 |
| beta_2p0 | h1_h4 | 0.014303 | 0.004068 | 0.043220 | 0.937743 |
| beta_2p5 | h1_h4 | 0.006760 | 0.002246 | 0.027966 | 0.964981 |
| beta_3p0 | h1_h4 | 0.001573 | 0.000802 | 0.021186 | 0.968872 |

## 판단

- beta tradeoff: beta 2.0가 validation 기준 수익 추종성과 false-safe tradeoff가 가장 좋았다. IC 0.029400, spread 0.005852, false_safe 0.079057, severe_recall 0.901293.
- 다음 CP 추천: beta_2p0 설정을 CP136 h4 pvv 후보의 저장 전용 재현 CP로 검토한다.
