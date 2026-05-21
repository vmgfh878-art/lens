# CP146-LM EODHD 500 Line baseline full training

## 상태

- status: `PASS`
- provider/context: `eodhd` / `eodhd_500`
- context_checksum: `1aa6452d82369cc6`
- W&B run link 수: `6`

## 후보 결과

| candidate | status | class | line_gate | ic_mean | spread | fee | false_safe | severe | wandb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1d_patchtst_h5_pvv_p32_s16_beta2 | PASS | rejected | True | 0.053091 | 0.007208 | 11.491260 | 0.311823 | 0.684179 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-eodhd500/runs/rnw4uxjz |
| 1d_patchtst_h5_no_fundamentals_p32_s16_beta2 | PASS | rejected | True | 0.052596 | 0.009178 | 47.330809 | 0.301973 | 0.688892 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-eodhd500/runs/gqosn2u0 |
| 1d_patchtst_h5_pvv_dense_beta2 | PASS | rejected | True | 0.082525 | 0.009612 | 57.153860 | 0.352381 | 0.643858 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-eodhd500/runs/d3jlbkqu |
| 1w_patchtst_h4_pvv_p16_s8_beta2 | PASS | selectable_verified | True | 0.025911 | 0.009724 | 0.588526 | 0.159205 | 0.843923 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-eodhd500/runs/vllo4r3x |
| 1w_patchtst_h4_no_fundamentals_p16_s8_beta2 | PASS | recommended_default | True | 0.030282 | 0.011879 | 0.787057 | 0.128305 | 0.874389 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-eodhd500/runs/gmi39hnq |
| 1w_patchtst_h6_pvv_p16_s8_beta2 | PASS | selectable_verified | True | 0.010767 | 0.018887 | 1.548625 | 0.070312 | 0.923980 | https://wandb.ai/vmfhdirn2014-sejong-university/lens-eodhd500/runs/ugyd3709 |

## 메모

- save-run, DB write, inference 저장, composite, 프론트 수정은 사용하지 않는다.
- W&B online 실행은 사용자 터미널에서 수행한다.
- local log와 summary.json을 기준으로 report/registry/csv를 재생성할 수 있다.
- 터미널 실행 중 출력이 멈추거나 GPU compute가 보이지 않으면 정상 학습으로 가정하지 말고, 후보별 progress.log와 python 프로세스 CPU delta를 확인한다. 20~30초 동안 같은 stage에서 CPU delta가 0이면 잔여 실행을 컷하고 병목 stage를 수정한 뒤 재실행한다.
