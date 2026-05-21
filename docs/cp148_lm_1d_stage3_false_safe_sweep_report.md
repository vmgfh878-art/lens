# CP148-LM-1D Stage 3 false-safe narrow sweep 보고서

- 생성 시각: 2026-05-08T17:36:13
- 범위: EODHD 500 local parquet 기준 1D h5 PatchTST line_model Stage 3 narrow sweep
- 금지 준수: save-run 없음, DB write 없음, inference 저장 없음, product promotion 없음, band/composite 없음, beta 변경 없음, live fetch 없음
- beta/alpha/delta: 2.0 / 1.0 / 1.0 유지
- 목표: Stage 2 best false_safe_tail_rate 0.308661보다 낮추고, severe_downside_recall 0.685019보다 높이는 후보 탐색

## 1. Stage 2 대비 기준

- Stage 2 false_safe best: `cp148_s2_patchtst_no_fund_p32_s16` = 0.308661
- Stage 2 severe_recall best: `cp148_s2_patchtst_no_fund_p32_s16` = 0.685019
- Stage 3 목표 관찰선: false_safe_tail_rate <= 0.27, severe_downside_recall >= 0.70
- product target은 false_safe_tail_rate <= 0.20, severe_downside_recall >= 0.75로 유지하지만 이번 보고서는 제품 결론을 내리지 않는다.

## 2. Trial 결과

| trial | base | hp | line_gate | score | IC | spread | fee | false_safe | FS 개선 | severe | severe 개선 | 분류 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s3_pvv_p16_d35_wd10_lr5e5 | patchtst_pvv_p16_s8 | d=0.35, wd=0.1, lr=5e-05 | True | 0.778571 | 0.073080 | 0.016486 | 12216.557601 | 0.393621 | -0.084960 | 0.590096 | -0.094923 | 탈락 후보 |
| s3_pvv_p16_d25_wd05_lr1e4 | patchtst_pvv_p16_s8 | d=0.25, wd=0.05, lr=0.0001 | True | 0.650000 | 0.072878 | 0.016272 | 10493.350737 | 0.402969 | -0.094309 | 0.581938 | -0.103081 | 탈락 후보 |
| s3_no_fund_d15_wd01_lr1e4 | patchtst_no_fund_p32_s16 | d=0.15, wd=0.01, lr=0.0001 | True | 0.621429 | 0.067490 | 0.010936 | 193.558523 | 0.329803 | -0.021142 | 0.657569 | -0.027450 | 탈락 후보 |
| s3_pvv_p16_d15_wd01_lr1e4 | patchtst_pvv_p16_s8 | d=0.15, wd=0.01, lr=0.0001 | True | 0.521429 | 0.072843 | 0.016174 | 9622.676116 | 0.407521 | -0.098860 | 0.568441 | -0.116578 | 탈락 후보 |
| s3_no_fund_d25_wd05_lr1e4 | patchtst_no_fund_p32_s16 | d=0.25, wd=0.05, lr=0.0001 | True | 0.435714 | 0.065752 | 0.010534 | 151.616428 | 0.358881 | -0.050221 | 0.628424 | -0.056595 | 탈락 후보 |
| s3_no_fund_d25_wd10_lr2e4 | patchtst_no_fund_p32_s16 | d=0.25, wd=0.1, lr=0.0002 | True | 0.421429 | 0.064434 | 0.009902 | 98.356080 | 0.335761 | -0.027100 | 0.654493 | -0.030526 | 탈락 후보 |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | patchtst_pvv_p32_s16_reference | d=0.25, wd=0.05, lr=0.0001 | True | 0.392857 | 0.070157 | 0.012709 | 737.377406 | 0.418287 | -0.109627 | 0.565963 | -0.119056 | 탈락 후보 |
| s3_no_fund_d35_wd10_lr5e5 | patchtst_no_fund_p32_s16 | d=0.35, wd=0.1, lr=5e-05 | True | 0.178571 | 0.054787 | 0.012445 | 701.104653 | 0.432054 | -0.123393 | 0.552708 | -0.132311 | 탈락 후보 |

## 3. IC/spread/fee 희생 여부

Stage 3 후보는 false-safe만 낮아져도 IC, spread, fee_adjusted_return 중 하나가 무너지면 seed 재평가 후보로 넘기지 않는다.

| trial | IC/spread/fee 양수 | failure |
| --- | --- | --- |
| s3_pvv_p16_d35_wd10_lr5e5 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_pvv_p16_d25_wd05_lr1e4 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d15_wd01_lr1e4 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_pvv_p16_d15_wd01_lr1e4 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d25_wd05_lr1e4 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d25_wd10_lr2e4 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d35_wd10_lr5e5 | True | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |

## 4. Top 2 seed 재평가 후보

_없음_

## 5. 탈락 후보와 이유

| trial | base | 분류 | failure |
| --- | --- | --- | --- |
| s3_pvv_p16_d35_wd10_lr5e5 | patchtst_pvv_p16_s8 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_pvv_p16_d25_wd05_lr1e4 | patchtst_pvv_p16_s8 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d15_wd01_lr1e4 | patchtst_no_fund_p32_s16 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_pvv_p16_d15_wd01_lr1e4 | patchtst_pvv_p16_s8 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d25_wd05_lr1e4 | patchtst_no_fund_p32_s16 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d25_wd10_lr2e4 | patchtst_no_fund_p32_s16 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_pvv_p32_ref_d25_wd05_lr1e4 | patchtst_pvv_p32_s16_reference | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |
| s3_no_fund_d35_wd10_lr5e5 | patchtst_no_fund_p32_s16 | 탈락 후보 | false_safe_not_improved, false_safe_target_miss, severe_recall_not_improved, severe_recall_0p70_miss |

## 6. 다음 단계 추천

- seed 재평가 후보 없음.
- 이번 축의 dropout/weight_decay/lr 조정만으로는 false-safe와 severe recall을 동시에 개선하지 못한 것으로 기록한다.
- 다음 LM은 beta 변경 없이 false-safe-aware selector, downside sample weighting, feature 추가 축소를 별도 CP로 검토한다.

## 7. 잔여 python/pythonw 프로세스 확인

```json
{
  "status": "none_visible",
  "processes": []
}
```

## 8. 산출물

- metrics: `docs\cp148_lm_1d_stage3_false_safe_sweep_metrics.json`
- summary csv: `docs\cp148_lm_1d_stage3_false_safe_sweep_summary.csv`
- logs: `docs\cp148_lm_1d_stage3_false_safe_sweep_logs`
