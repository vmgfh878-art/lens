# CP50-G 기획값 대비 구현 기본값 감사 보고서

감사일: 2026-04-30

## 1. 결론

현재 Lens는 `h_max` 기반 split/gap과 저장 schema는 비교적 잘 정리되어 있지만, 제품 기본 horizon, CLI 기본 `seq_len`, checkpoint 기본값, AI 밴드 표시 범위가 원 기획/SoT와 충돌한다. CP48/CP49 이후에는 `h5`, `h10`, `h20`이 같은 후보가 아니라 서로 다른 branch라는 점이 더 명확해졌다.

특히 `horizon=5`는 지금 제품 기본 line 후보로 올라왔지만, 원 기획의 `1D=20`, `1W=12` 제품 horizon을 대체한 것이 아니라 short-horizon fallback 또는 Phase 1 제품 후보로 분리해 기록해야 한다. `h20`은 CP48/CP49 기준 학습 가능성은 확인됐지만 Phase 1.5 branch이며, 현재 구현/API 기본값은 아직 `1D=5`, `1W=4`다.

이번 CP에서는 코드 수정, 테스트 수정, 포맷팅, 모델 학습, DB 쓰기를 하지 않았다.

## 2. 근거 파일

주요 SoT/기획 근거:

- `README.md:151` `gap = h_max (1D=20, 1W=12)`
- `README.md:161` `1D seq=252 h∈{1,5,20}`, `1W seq=104 h∈{1,4,12}`
- `README.md:162` `1M` 표시 전용
- `README.md:175` 초기 loss 식에 `lambda_width` 포함
- `docs/cp3_instruction.md:8-12` CP2.6 기반 `seq_len`, `h_max`, row_count 근거
- `docs/cp3_instruction.md:24-26` `70/15/15`, `gap=h_max` 구현 지시
- `docs/cp3_instruction.md:88-90` `seq_len=252/104`, `h_max=20/12`
- `docs/training_hyperparameters.md:221`, `233`, `243` `gap=h_max` 채택
- `docs/training_hyperparameters.md:347`, `353`, `540` q10/q90 및 `lambda_width` 레거시화
- `docs/training_hyperparameters.md:762-764` batch ladder `64/128/256`
- `docs/cp_product_demo_plan.md:72`, `84-86`, `320-322`, `346-349` latest prediction 기반 제품 표시와 1M price-only

주요 코드 근거:

- `ai/preprocessing.py:614-616` `default_horizon`: `1D=5`, `1W=4`
- `ai/preprocessing.py:776`, `921` dataset 기본 `seq_len=60`
- `ai/preprocessing.py:855-888`, `985-996` `forecast_dates` 길이가 `resolved_horizon`을 따름
- `ai/preprocessing.py:1031-1068` split plan에서 `h_max=MAX_HORIZON_BY_TIMEFRAME`
- `ai/splits.py:8-116` `SPLIT_RATIO`, `MAX_HORIZON_BY_TIMEFRAME`, `gap=h_max`
- `ai/train.py:84-90` checkpoint/gate 기준
- `ai/train.py:486-538` CLI 기본값
- `ai/train.py:517-522` RevIN/PatchTST geometry CLI 기본값
- `ai/loss.py:128-180` loss weight 기본값과 `lambda_width` 미사용
- `ai/models/patchtst.py:16-30` PatchTST 구조 기본값
- `ai/models/cnn_lstm.py:18-27`, `35-75`, `98-100` CNN-LSTM 구조, fp32/cudNN 안정화
- `ai/models/tide.py:17-28`, `40-48` TiDE 구조 기본값
- `backend/app/services/model_svc.py:15-55` API 기본 horizon
- `backend/app/services/api_service.py:102-174` latest/run_id prediction 조회와 composite meta 병합
- `backend/app/repositories/prediction_repo.py:8-55` latest/run_id prediction 조회
- `frontend/src/components/StockView.tsx:725-756`, `783-792` 1M 비활성, latest/fallback, forecast segment timeline
- `frontend/src/components/Chart.tsx:65-116` `forecast_dates` 구간만 AI overlay
- `backend/db/schema.sql:183-210`, `266-288` predictions/evaluations schema
- `backend/db/scripts/ensure_runtime_schema.py:37-50`, `140-174` runtime schema 보강
- `ai/storage.py:20-24`, `35-39` 저장 upsert key

최근 실험 근거:

- `docs/cp48_h20_feasibility_smoke_report.md:1-5` h20 feasibility smoke 성격
- `docs/cp48_h20_feasibility_smoke_report.md:78-92` h20 `forecast_dates`/series length 20 확인
- `docs/cp48_h20_feasibility_smoke_report.md:113-128` h20은 가능하지만 Phase 1.5 보류
- `docs/cp49_patchtst_horizon_rescue_report.md:34-47` h5/h10/h20/seq504 matrix
- `docs/cp49_patchtst_horizon_rescue_report.md:93-106` CP49 결과표
- `docs/cp49_patchtst_horizon_rescue_report.md:125-135` h5 제품 후보, h20 Phase 1.5 판단
- `docs/cp49_patchtst_horizon_rescue_metrics.json:9-18` CP49 scope: `line_gate`, `batch_size=256`, save-run false, W&B metric

## 3. 불일치 표

| 항목 | 기획값 | 코드 기본값 | 최근 실험값 | 차이 원인 | 분류 | 다음 조치 |
| --- | --- | --- | --- | --- | --- | --- |
| Horizon: `h_max` | `1D=20`, `1W=12` (`README.md:151`, `docs/cp3_instruction.md:12`) | `ai.splits.MAX_HORIZON_BY_TIMEFRAME`: `1D=20`, `1W=12` (`ai/splits.py:8-16`) | CP48/49도 h20 branch에서 h_max 원칙을 유지 | CP3에서 leakage 방지 계약으로 정착 | 의도된 변경 | 유지 |
| Horizon: 제품/API 기본 | 원 기획은 `1D h∈{1,5,20}`, `1W h∈{1,4,12}`이고 제품 기본은 `1D=20`, `1W=12` 후보 (`README.md:161`) | `default_horizon`: `1D=5`, `1W=4` (`ai/preprocessing.py:614-616`), API도 동일 (`backend/app/services/model_svc.py:16-55`) | CP49 결론은 h5 longer_context를 제품 기본 line 후보로 올리고 h20은 Phase 1.5 (`docs/cp49_patchtst_horizon_rescue_report.md:133-135`) | 추측: CP3.5 dry-run `h5/h4`와 이후 smoke branch가 API 기본으로 굳어짐 | 추가 결정 필요 / 제품 오류 위험 | `h5`를 제품 Phase 1 기본으로 공식 채택할지, 원 기획 `h20/h12`를 별도 제품 모드로 남길지 SoT를 갱신 |
| Horizon branch 구분 | h5/h10/h20은 모두 존재하나 같은 후보군으로 섞으면 안 됨 | CLI는 `--horizon None`이면 default `5/4`; 명시 실행만 분기 (`ai/train.py:486`, `1803`) | CP49 matrix: h5/h10/h20 및 h20 seq504 비교 (`docs/cp49_patchtst_horizon_rescue_report.md:36-47`) | CP49에서 horizon별 line 가능성과 risk line을 분리 | 의도된 변경 | 실험명, run meta, 제품 UI에 `short-horizon`, `long-horizon`, `risk-only` 태그 추가 여부 결정 |
| Sequence length: 제품/SoT | `1D=252`, `1W=104` (`README.md:161`, `docs/cp3_instruction.md:88-90`) | train/preprocessing 기본 `60` (`ai/train.py:487`, `ai/preprocessing.py:776`, `921`) | PatchTST: `252`, CP49 h20에서 `504` 1회; CNN-LSTM: `60/90/120`; TiDE: 주로 `252` | CNN-LSTM local-volatility branch와 smoke 편의값이 전역 CLI 기본으로 남음 | 임시 smoke | 모델별 기본값을 문서화. PatchTST 기본 `252`, CNN-LSTM band branch `60`, h20 장기 후보 `252/504`로 분리 |
| Batch size | CP50 지시서 기획값: `1D=128`, `1W=64`. 저장소 문서에서는 `README.md:119` 예시 `128`, `docs/training_hyperparameters.md:101` 1D PatchTST batch 128 여유, `docs/cp13_patchtst_solo_plan.md:214-215` smoke 64 및 ladder 확인 | CLI 기본 `64` (`ai/train.py:489`) | CP29 이후 다수 `256`; CP49 scope도 `batch_size=256` (`docs/cp49_patchtst_horizon_rescue_metrics.json:12`) | 처리량 최적화. `256`은 성능 기획값이 아니라 GPU throughput 값 | 의도된 변경 / 추가 결정 필요 | 제품 재현 기본은 `1D=128`, `1W=64`로 둘지, 실험 throughput 기본을 `256`으로 둘지 역할별 분리. 저장소에서 `1W=64` 직접 근거는 찾지 못했으므로 SoT에 명시 필요 |
| Split ratio | `70/15/15` (`docs/cp3_instruction.md:24-26`) | `SPLIT_RATIO=(0.7,0.15,0.15)` (`ai/splits.py:8`, `ai/preprocessing.py:31`) | 동일 | CP3 계약 반영 | 의도된 변경 | 유지 |
| Split/gap 일관성 | `gap=h_max`, `1D=20`, `1W=12` (`docs/training_hyperparameters.md:221-243`) | split에서 sample count를 `h_max`로 계산하고 val/test 시작에 `h_max`를 더함 (`ai/splits.py:65-116`) | h5/h10/h20 모두 split plan은 timeframe h_max 기준 (`ai/preprocessing.py:1031-1068`) | horizon별 실제 `h`가 아니라 timeframe 최대 horizon으로 purge | 의도된 변경 | 유지. 단 h>h_max 연구가 생기면 `MAX_HORIZON_BY_TIMEFRAME` 확장 필요 |
| Band quantile nominal | 초기 기획 q10/q90 (`docs/training_hyperparameters.md:347`) | CLI/loss 기본 `q_low=0.1`, `q_high=0.9` (`ai/train.py:495-496`, `ai/loss.py:128-144`) | q15/q85, q20/q80, q25/q75 반복. CP49 line rescue는 q25/q75 (`docs/cp49_patchtst_horizon_rescue_metrics.json:33-55`) | calibration과 역할 분리 실험 | 의도된 변경 | q는 “nominal coverage”와 “product target coverage”를 분리해 문서화 |
| Coverage gate | 제품 band는 목표 coverage 0.85가 반복 등장. CP36 이후 band gate는 `0.75~0.95`, upper `<=0.15`, lower `<=0.20` (`docs/training_hyperparameters.md:1211`) | 코드도 동일 (`ai/train.py:86-90`, `309-321`) | composite/product는 `0.75~0.90`, lower `<=0.12`, upper `<=0.15` (`ai/composite_policy_eval.py:103-105`, `ai/cp45_band_sweep.py:237-239`, `ai/cp46_upper_calibration.py:229-231`) | 학습 gate와 제품 composite gate가 분리됨 | 의도된 변경 / 추가 결정 필요 | `training band_gate`와 `product composite_gate` 이름을 분리해 SoT화 |
| `lambda_line` | SoT loss 식 포함 (`README.md:175`) | 기본 `1.0` (`ai/train.py:500`, `ai/loss.py:133`) | 대부분 `1.0` | 유지 | 의도된 변경 | 유지 |
| `lambda_band` | SoT loss 식 포함 | 기본 `1.0` (`ai/train.py:501`, `ai/loss.py:134`) | 최근 band/line 후보 대다수 `2.0` (`docs/training_hyperparameters.md:992-993`, `1048`, CP49 command) | band 신호 강화 실험이 표준 후보화 | 의도된 변경 | 재현 명령에는 항상 명시. CLI 기본을 바꿀지는 별도 CP |
| `lambda_width` | 초기 SoT loss 식에 포함 (`README.md:175`, `docs/training_hyperparameters.md:353`) | CLI/loss 기본값은 있으나 현재 손실 계산에는 미사용 (`ai/train.py:502`, `ai/loss.py:149-150`, `171-180`) | 실험값과 무관하게 dead/legacy | CP10 이후 호환 인자로만 남음 | 근거 없는 임의값 | CLI help와 문서에는 “학습 미반영”을 계속 명시. 나중에 제거/복원 중 하나 결정 |
| `lambda_cross` | SoT loss 식 포함 | 기본 `1.0` (`ai/train.py:503`, `ai/loss.py:136`, `171-173`) | 대부분 기본 유지 | crossing 방지 | 의도된 변경 | 유지 |
| `lambda_direction` | 초기 SoT에는 없음. CP11 이후 보조 head 실험 | 기본 `0.1` (`ai/train.py:504`, `ai/loss.py:137`, `180`) | CNN-LSTM direction head는 불확실, 기본 `use_direction_head=False` (`ai/train.py:523`, `ai/models/cnn_lstm.py:26`) | CP11 실험 잔재를 loss 기본으로 남김 | 추가 결정 필요 | direction head를 본류에서 쓸지 확정 전까지 명시적 off branch로 유지 |
| PatchTST 구조 기본 | CP13 solo 기준 `seq_len=252`, `patch_len=16`, `stride=8`; 원 구현 fidelity 중시 | 모델 기본 `seq_len=252`, `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=8`, `n_layers=3`, `use_revin=True`, `ci_aggregate=target` (`ai/models/patchtst.py:16-30`, `ai/train.py:517-522`) | CP49 h5/h10/h20에서 baseline 16/8, longer 32/16, dense 16/4; h20 seq504 1회 | 구조 기본은 유지하되 geometry branch 확장 | 의도된 변경 | h5 제품 후보가 32/16으로 이동했으므로 PatchTST “제품 geometry”와 “기본 구현 geometry”를 분리 |
| CNN-LSTM 기본/실험 | Phase 1.5 보조 모델. seq_len 기획은 명확하지 않음 | 모델 기본 `seq_len=252`이나 내부에서 `del seq_len`; conv/LSTM 구조는 `cnn_channels=64`, `lstm_hidden=128`, `n_layers=2`, `fp32_modules=none` (`ai/models/cnn_lstm.py:18-35`) | band 후보는 `seq_len=60`, 일부 `90/120`, `fp32_modules=lstm,heads` 기록 (`docs/training_hyperparameters.md:1390`, `docs/model_architecture.md:911-914`) | CNN-LSTM은 local band 후보로 재정의됐지만 모델 기본은 252처럼 보임 | 임시 smoke / 추가 결정 필요 | CNN-LSTM에서 `seq_len`이 실제 구조에는 직접 안 쓰인다는 점을 문서화하고, dataset lookback 기본을 별도 지정 |
| TiDE 구조/기본값 | CP3.5 이후 최소 TiDE 골격 후보 | 모델 기본 `feature_dim=16`, `enc_dim=256`, `dec_dim=128`, `n_enc_layers=4`, `n_dec_layers=2`, `future_cov_dim=0` (`ai/models/tide.py:17-28`) | CP35/37에서 `seq_len=252`, q10/q90, direct/param band smoke | Phase 1.5 보류 모델 | 의도된 변경 | TiDE는 제품 기본이 아니라 비교 후보로 유지 |
| `use_revin` | PatchTST fidelity/ablation 대상 | CLI 기본 true, 모델 기본 true (`ai/train.py:517`, `ai/models/patchtst.py:26`) | CP33에서 true/false ablation. false도 밴드 문제 해결 못 함 (`docs/model_architecture.md:787-796`) | fidelity 기본 유지, ablation만 분리 | 의도된 변경 | 유지. RevIN 수정은 별도 ablation CP에서만 |
| Product display: 1M | `1M` 표시 전용, AI 예측/밴드/시그널 비활성 (`README.md:162`, `docs/cp_product_demo_plan.md:72`, `86`) | UI에서 1M이면 AI disabled 및 layer off (`frontend/src/components/StockView.tsx:725-745`, `frontend/src/components/Chart.tsx:65-67`) | 제품 demo도 1M price-only 확인 | 계약 일치 | 의도된 변경 | 유지 |
| Product display: latest forecast | 기획 문서 일부는 latest completed run 기준 표시를 허용 (`docs/cp_product_demo_plan.md:320-322`, `346-349`) | API는 latest prediction 또는 run_id prediction 1건을 조회 (`backend/app/services/api_service.py:131-169`, `backend/app/repositories/prediction_repo.py:14-55`) | CP45/CP46 계열은 latest/fallback composite 5일 구간 표시 | 데모 완성 우선으로 latest forecast 표시가 제품 기본처럼 굳어짐 | 의도된 변경 / 추가 결정 필요 | latest forecast는 Phase 1 표시 계약으로 유지 가능. 다만 full rolling band와 혼동 금지 |
| Product display: full rolling AI band | 사용자가 전체 차트 AI 밴드 계약 감사를 명시. 원 제품 서사는 가격 + AI band overlay (`docs/cp_product_demo_plan.md:9`, `84`) | 차트는 `forecast_dates`만 overlay하고 실제 가격 구간 전체 rolling band는 없음 (`frontend/src/components/Chart.tsx:77-116`, `StockView.tsx:746-756`) | 저장 prediction도 latest/asof row 단위. rolling history endpoint 없음 | latest demo 계약과 full rolling 계약이 분리되지 않음 | 제품 오류 / 추가 결정 필요 | rolling AI band가 제품 계약이면 prediction history API와 UI layer를 새로 정의 |
| predictions.horizon | horizon은 저장 계약의 필수 컬럼 (`backend/db/schema.sql:188`) | 저장 upsert key에도 horizon 포함 (`ai/storage.py:20-24`) | CP48 h20에서 series length 20 확인 (`docs/cp48_h20_feasibility_smoke_report.md:78-81`) | 저장 schema는 horizon 일반화 가능 | 의도된 변경 | 유지 |
| forecast_dates 길이 | horizon 길이와 일치해야 함 | preprocessing이 `future_start:future_end`로 horizon 길이 생성 (`ai/preprocessing.py:855-888`, `985-996`) | h5 demo 길이 5, h20 probe 길이 20. 단 `series_length_all_5` 체크명은 하드코딩 잔재 (`docs/cp48_h20_feasibility_smoke_report.md:92`) | 저장 값은 일반화됐으나 체크 이름이 h5 중심 | 추가 결정 필요 | h20 본류 전 `series_length_all_horizon` 체크로 일반화 |
| composite meta | line/band run과 policy 추적 필요 | API meta merge 대상에 `line_model_run_id`, `band_model_run_id`, `composition_policy` 포함 (`backend/app/services/api_service.py:102-116`); composite 저장도 동일 (`ai/composite_inference.py:508-511`) | CP44-M smoke에서 meta 저장 확인, CP46-M에서 policy calibration | 저장 계약은 대체로 정리됨 | 의도된 변경 | 유지. run 상세 UI/보고서에서 meta를 항상 노출 |
| line/band model run_id | composite provenance 필수 | `predictions.meta`에 저장, `predictions.run_id`는 composite run_id (`docs/model_architecture.md:942-949`, `ai/composite_inference.py:482-511`) | CP41/44 이후 composite 저장 smoke 통과 | CP40~44 저장 계약 수리 결과 | 의도된 변경 | 유지 |
| composition_policy | `risk_first_lower_preserve` 등 정책 기록 필요 | 기본 `risk_first_lower_preserve` (`ai/composite_inference.py:419`), record meta에도 기록 (`ai/composite_inference.py:508-511`) | CP46-M 이후 upper buffer 후보가 있으나 저장 기본 변경 여부는 별도 | policy 실험과 저장 기본이 완전히 동기화되지 않음 | 추가 결정 필요 | 제품 기본 policy를 CP46-M 후보로 바꿀지 별도 결정 |
| predictions unique/upsert | run별 prediction history 보존 필요 | schema/upsert key가 `run_id,ticker,model_name,timeframe,horizon,asof_date` (`backend/db/schema.sql:203`, `223`, `ai/storage.py:24`) | CP30 이후 run_id 포함 계약 | 과거 unique key에서 run_id 포함으로 수리 | 의도된 변경 | 유지 |
| W&B/save-run | 제품/재현 run 외에는 금지 또는 제한 | CLI W&B 기본 true, save-run false (`ai/train.py:530-534`) | CP49는 W&B metric only true, save-run false (`docs/cp49_patchtst_horizon_rescue_report.md:34`, metrics scope) | 실험별 운영 정책이 CLI 기본과 다름 | 추가 결정 필요 | 금지 CP에서는 `--no-wandb` 명시. W&B 기본 true 변경 여부는 별도 |

## 4. 항목별 감사

### 4.1 Horizon

기획값의 핵심은 두 가지다. 첫째, leakage 방지용 `h_max`는 `1D=20`, `1W=12`다. 둘째, 제품 horizon 후보는 `1D h∈{1,5,20}`, `1W h∈{1,4,12}`다. 구현은 `h_max`에는 맞지만 default horizon은 `1D=5`, `1W=4`다.

CP48은 `1D horizon=20`이 학습 가능한지 확인한 smoke다. 결과적으로 h20은 CUDA/bf16/checkpoint/composite 길이 20까지는 가능했지만, line/band/composite 지표가 본류 기준을 통과하지 못했다. CP49는 h5/h10/h20을 같은 후보군으로 섞지 않고, h5를 제품 기본 line 후보로 올리고 h20을 Phase 1.5 branch로 보류했다.

판단:

- `h5`는 현재 Phase 1 제품 후보 또는 short-horizon fallback이다.
- `h20`은 원 기획의 long-horizon branch지만 아직 제품 기본으로 승격되지 않았다.
- 현재 API 기본 `1D=5`, `1W=4`가 원 기획을 바꾼 것인지 임시 default인지 SoT에 명시해야 한다.

### 4.2 Sequence length

SoT의 `1D=252`, `1W=104`는 CP3부터 안정적이다. 하지만 학습 CLI와 preprocessing의 기본값은 `60`이다. 이는 PatchTST/제품 기본값과 맞지 않는다.

최근 실험은 이미 모델별로 분화됐다.

- PatchTST line: `252` 기본, CP49에서 h20 `504` 1회.
- CNN-LSTM band: `60` 생존, `90/120` sweep 또는 smoke.
- TiDE: 대체로 `252`.

판단:

- `seq_len=60`은 전역 기본값으로 부적절하다.
- `60`은 CNN-LSTM band/local-volatility branch의 값으로 분류해야 한다.
- PatchTST 제품/SoT 기본은 여전히 `252`이며, h20 장기 branch에서 `504`는 탈락 기록이 있다.

### 4.3 Batch size

CP50 지시서는 기획값을 `1D=128`, `1W=64`로 준다. 저장소 안에서는 `1D=128`에 가까운 근거는 확인되지만, `1W=64`라는 직접 문서 라인은 찾지 못했다. 따라서 `1W=64`는 이번 CP50 지시서가 최신 SoT로 준 값으로 기록한다.

코드 기본은 `64`이고, 최근 실험은 대부분 `256`이다. `256`은 모델 품질 기본값이 아니라 throughput 최적화 값이다.

판단:

- 재현 기본과 throughput 기본을 분리해야 한다.
- `256`을 제품/기획값처럼 쓰면 안 된다.

### 4.4 Split/gap

이 영역은 양호하다. split ratio는 `70/15/15`, gap은 `h_max`이며, 코드도 `MAX_HORIZON_BY_TIMEFRAME`을 통해 timeframe별 최대 horizon으로 gap을 둔다.

주의점은 `horizon=5`를 학습해도 gap은 `1D=20`이라는 점이다. 이는 보수적인 purge 계약으로 볼 수 있다. h20 이상의 연구가 생기면 `MAX_HORIZON_BY_TIMEFRAME`을 확장해야 하지만, 현재 h5/h10/h20 범위에서는 일관된다.

### 4.5 Band quantile / coverage

초기 nominal quantile은 q10/q90이다. 현재 실험은 q15/q85, q20/q80, q25/q75 등으로 분화됐다. 이 변화는 coverage와 width 균형을 맞추려는 의도된 실험이다.

다만 두 종류의 coverage가 섞인다.

- 학습 checkpoint band gate: `0.75~0.95`, upper `<=0.15`, lower `<=0.20`.
- product/composite gate: `0.75~0.90`, upper `<=0.15`, lower `<=0.12`.

판단:

- q10/q90은 nominal quantile 기본값이다.
- product target coverage는 0.85 중심 calibration 목표다.
- 학습 gate와 제품 gate를 한 이름으로 부르면 해석이 꼬인다.

### 4.6 Loss weights

현재 실제 loss에 들어가는 항은 `lambda_line`, `lambda_band`, `lambda_cross`, `lambda_direction`이다. `lambda_width`는 CLI와 config에는 남아 있지만 `ForecastCompositeLoss.forward`에서는 사용되지 않는다.

판단:

- `lambda_width`는 dead/legacy argument다.
- `lambda_band=2.0`은 최근 실험 표준 후보지만 CLI 기본은 `1.0`이다.
- `lambda_direction=0.1`은 direction head 실험의 잔재이며, direction head 자체는 기본 off다.

### 4.7 Model defaults

PatchTST 구현 기본은 SoT와 꽤 가깝다. `seq_len=252`, `patch_len=16`, `stride=8`, `d_model=128`, `n_heads=8`, `n_layers=3`, `use_revin=True`, channel independence가 기본이다. 다만 CP49 제품 line 후보는 `patch_len=32`, `stride=16`이라서 “구현 기본”과 “제품 후보 geometry”가 갈라졌다.

CNN-LSTM은 모델 생성자 기본 `seq_len=252`를 받지만 내부에서 `del seq_len`한다. 즉 모델 구조 자체는 lookback 길이에 직접 의존하지 않고, dataset window 길이만 바뀐다. 최근 band 후보가 `seq_len=60`인 이유는 모델 기본이 아니라 데이터 window branch다.

TiDE는 최소 골격이 들어가 있으며, 기본 `future_cov_dim=0`이다. 최근에는 Phase 1.5 비교 후보에 가깝다.

RevIN은 PatchTST 기본 true다. CP33 ablation에서 false가 문제를 해결하지 못했으므로, 현재는 유지가 맞다.

### 4.8 Product display contract

`1M` price-only는 기획과 구현이 일치한다. 문제는 AI 밴드 범위다.

현재 구현은 latest 또는 fallback prediction 1건의 `forecast_dates`만 그린다. 과거 전체 차트를 덮는 rolling AI band history는 없다. `docs/cp_product_demo_plan.md:320-322`는 latest completed run 표시를 Phase 1 데모 기본으로 허용하지만, 사용자가 말한 “전체 차트를 덮는 AI 밴드”와는 다른 계약이다.

판단:

- latest forecast 표시는 Phase 1 demo/product fallback 계약으로는 가능하다.
- full rolling AI band가 제품 계약이라면 현재 구현은 제품 오류다.
- rolling band를 하려면 prediction history API와 UI layer가 필요하다.

### 4.9 Storage/evaluation contract

저장 schema는 CP30 이후 비교적 건강하다.

- `predictions.horizon`은 필수 컬럼이다.
- unique/upsert key에 `run_id`가 포함된다.
- `forecast_dates`와 series는 horizon 길이를 따른다.
- composite meta에는 `line_model_run_id`, `band_model_run_id`, `composition_policy`가 들어간다.
- `prediction_evaluations`에는 lower/upper breach rate가 있다.

남은 위험은 두 가지다.

- `composite_inference` 내부 contract check 이름이 `series_length_all_5`로 남아 있어 h20에서는 false가 정상처럼 기록된다. 실제 series length는 20이지만 체크 이름이 h5 중심이다.
- `composition_policy` 기본이 CP46-M upper buffer 후보와 동기화됐는지는 추가 결정이 필요하다.

## 5. 우선순위

| 우선순위 | 항목 | 이유 |
| --- | --- | --- |
| P0 | horizon 제품 계약 정리 | `h5`가 Phase 1 제품 기본인지, `h20/h12`가 long-horizon 제품 계약인지 명시가 필요 |
| P0 | full rolling AI band 여부 | latest forecast와 rolling band는 API/UI/저장 계약이 다름 |
| P1 | `seq_len=60` 전역 기본 | PatchTST/SoT 기본과 충돌하고 CNN-LSTM branch 값으로 보임 |
| P1 | checkpoint 기본 `val_total` | role gate 체계와 맞지 않음 |
| P1 | `lambda_width` dead argument | SoT loss 식과 코드 동작이 다름 |
| P1 | h20 contract check 이름 | 실제 h20 series는 되지만 체크 이름이 h5로 고정 |
| P2 | batch 기본 계층화 | 기획값, CLI 기본, throughput 값이 섞임 |
| P2 | W&B 기본 true | 금지 CP와 충돌 가능 |

## 6. 최종 판단

CP50 기준 가장 큰 불일치는 모델 구조가 아니라 기본값 계층이다.

1. `h5`는 현재 제품 후보지만 원 기획 `h20/h12`를 폐기한 근거는 아니다.
2. `h20`은 학습 가능성이 확인됐지만 Phase 1.5 branch다.
3. `seq_len=60`, `batch_size=256`, `q25/q75`, `lambda_band=2`는 모두 역할별 실험값이지 범용 기획값이 아니다.
4. 저장 계약은 `run_id`, `horizon`, composite meta 중심으로 정리됐지만, 제품 화면은 아직 latest forecast segment 중심이다.
5. full rolling AI band를 제품 계약으로 유지한다면 다음 CP는 모델 학습이 아니라 prediction history 조회와 UI 표시 계약 정리가 먼저다.

추측으로 분류한 부분:

- `default_horizon=5/4`가 CP3.5 dry-run과 smoke 흐름에서 제품/API 기본으로 굳어졌다는 원인.
- `seq_len=60`이 CNN-LSTM local band branch 값에서 전역 CLI 기본으로 남았다는 원인.
- `batch_size=256`이 품질 기본이 아니라 throughput 최적화 값이라는 해석. 단, CP13과 CP49 근거상 가능성이 높다.

이번 보고서는 코드 수정, 테스트 수정, 포맷팅, 모델 학습, DB 쓰기 없이 작성했다.
