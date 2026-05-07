# CP148-LM 1D 실험 프로토콜

작성일: 2026-05-07  
목적: EODHD 500 기준 1D h5 보수적 예측선 후보를 연구형 절차로 검증하기 위한 실행 프로토콜.  
주의: 이 문서는 계획서이며, 이번 CP에서는 학습, sweep, DB write, inference 저장을 실행하지 않는다.

## 1. 공통 계약

| 항목 | 값 |
|---|---|
| role | line_model |
| provider/source | eodhd |
| timeframe | 1D |
| horizon | 5 |
| target | raw_future_return |
| line_target_type | raw_future_return |
| feature_version | v3_adjusted_ohlc |
| checkpoint_selection | line_gate |
| alpha/beta/delta | 1.0 / 2.0 / 1.0 |
| ci_aggregate | target |
| composite | 금지 |
| band experiment | 금지 |
| save-run | 별도 제품 저장 CP 전까지 금지 |
| inference 저장 | 금지 |

## 2. Stage 0: data/cache gate

입력:

- `data/parquet/price_data_eodhd_500.parquet`
- `data/parquet/indicators_eodhd_1D_500.parquet`
- `data/parquet/context/eodhd_500/`
- CP148-0/CP148-0-DG2/DG3 gate 결과

출력:

- provider/source가 eodhd인지 확인
- source_data_hash, context_checksum, feature contract 기록
- eligible ticker count 기록
- feature/target NaN/Inf 0 확인
- open_ratio/high_ratio/low_ratio p99 sanity 확인
- split overlap 0 확인
- `atr_ratio`가 MODEL_FEATURE_COLUMNS 36개에 없는지 확인

통과 조건:

- 1D LM eligible ticker >= 473 또는 변동 사유 명확
- feature NaN/Inf 0
- target h5 NaN/Inf 0
- split overlap 0
- provider/source/cache가 yfinance/generic과 섞이지 않음

중단 조건:

- provider/source 불명확
- cache manifest가 EODHD가 아님
- feature/target nonfinite 발생
- split overlap 또는 purge gap 실패

다음 stage 후보 수: data gate가 PASS해야 전체 진행.

## 3. Stage 1: baseline

입력:

- Stage 0에서 확정된 train/val/test split
- raw_future_return h5 target

실행할 baseline:

- zero return
- historical mean
- 5D momentum
- 20D momentum
- short reversal
- rolling mean
- existing 1D product line `patchtst-1D-efad3c29d803`
- random/shuffled baseline, 가능하면 seed 고정

출력:

- baseline별 IC, spread, fee_adjusted_return, false_safe_tail, severe_downside_recall, direction_accuracy
- 딥러닝 후보가 넘어야 할 최소 기준선

통과 조건:

- 최소 zero/historical/momentum/reversal baseline 산출
- random/shuffled baseline이 가능하지 않으면 불가 사유 기록

중단 조건:

- baseline metric 자체가 생성되지 않음
- existing 1D product line 로드 불명확

다음 stage 후보 수: baseline은 후보가 아니라 비교 기준 전체 유지.

## 4. Stage 2: coarse model family comparison

목적은 처음부터 넓은 sweep을 하지 않고 큰 구조 차이를 비교하는 것이다.

1차 실행 후보:

| experiment_id | model_family | seq_len | patch_len | patch_stride | feature_set | 목적 |
|---|---|---:|---:|---:|---|---|
| cp148_s2_patchtst_pvv_p32_s16 | PatchTST | 252 | 32 | 16 | price_volatility_volume | CP146 pvv 기준 재검증 |
| cp148_s2_patchtst_no_fund_p32_s16 | PatchTST | 252 | 32 | 16 | no_fundamentals | fundamentals 제거 효과 |
| cp148_s2_patchtst_pvv_p16_s8 | PatchTST | 252 | 16 | 8 | price_volatility_volume | dense patch 효과 |
| cp148_s2_tide_pvv | TiDE | 252 | n/a | n/a | price_volatility_volume | covariate dense model 비교 |
| cp148_s2_cnn_lstm_pvv | CNN-LSTM | 252 | n/a | n/a | price_volatility_volume | 국소 패턴 + 순차 기억 비교 |
| cp148_s2_tcn_pvv | TCN | 252 | n/a | n/a | price_volatility_volume | temporal convolution 비교 |

출력:

- validation 중심 line_metrics
- test 참고 metric
- 후보별 failure taxonomy
- top 2~3 후보 선별

통과 조건:

- line_gate pass
- IC > 0
- spread > 0
- fee_adjusted_return > 0
- false_safe_tail <= 0.20
- severe_downside_recall >= 0.75

중단 조건:

- 같은 model family가 두 feature set 이상에서 false_safe_fail
- 학습은 끝나지만 metric NaN/Inf 발생
- provider/source/cache 기록 누락

다음 stage 후보 수: 최대 3개.

## 5. Stage 3: feature set comparison

Stage 2 top model family를 대상으로 feature set만 제한적으로 연다.

비교 순서:

1. price_volatility_volume
2. no_fundamentals
3. technical_only, pvv와 완전히 같으면 생략 가능
4. full_features, WARN 비교 1회
5. context_light, 미정의면 실행하지 않고 design_needed

출력:

- feature_set별 개선/악화
- fundamentals coverage WARN 해석
- 제품 설명 가능성 점수

통과 조건:

- pvv 또는 no_fundamentals 중 하나가 hard gate에 근접
- full_features가 이길 경우 fundamentals low coverage에도 설명 가능한 근거가 있어야 함

중단 조건:

- context 확장 후보가 false_safe를 악화하고 IC/spread 개선도 없음

다음 stage 후보 수: 최대 2~3개.

## 6. Stage 4: narrow Optuna/W&B sweep

coarse 이후 top 2~3개만 sweep한다. W&B는 사용자가 터미널에서 online으로 실행한다.

고정:

- beta=2.0
- target/raw_future_return
- horizon h5
- checkpoint_selection=line_gate
- provider/source=eodhd
- feature contract v3_adjusted_ohlc

작게 열 항목:

- lr: 1e-4, 3e-4, 1e-3 주변
- weight_decay: 0, 1e-5, 1e-4
- dropout: 0.05, 0.10, 0.20
- batch_size: 128, 256
- PatchTST d_model: 64, 128
- PatchTST n_layers: 2, 3
- seq_len: 120, 180, 252 중 제한 비교

열지 않을 항목:

- beta
- target type
- 1W/1M
- band objective
- calibration
- product save
- composite
- frontend 연결

objective 기록:

- primary: validation line_gate pass
- secondary score: IC, spread, fee_adjusted_return 보상, false_safe_tail penalty, severe_recall reward, upside_sacrifice penalty

주의: train code의 checkpoint 선택은 line_gate 기준을 유지한다. 별도 composite score는 실험 분석용 기록일 뿐 checkpoint selection을 대체하지 않는다.

중간 중단 기준:

- false_safe_tail이 0.30 이상으로 지속
- IC와 spread가 모두 음수
- metric NaN/Inf
- val_total만 좋고 line 목적 metric이 무너짐

## 7. Stage 5: seed stability

sweep best 3개 후보를 seed 3~5개로 재실행한다.

기준:

- top1~top5 차이가 seed variance보다 작으면 동률권으로 판정
- seed 안정성 전에는 제품 v1 확정 금지
- 단일 seed 결과는 후보이지 결론이 아님

출력:

- seed별 validation/test metric
- 평균, 표준편차, worst seed
- hard gate 통과율
- seed_unstable 여부

## 8. Stage 6: final candidate selection

제품 v1 provisional candidate는 다음을 모두 만족해야 한다.

- line_gate pass
- IC > 0
- spread > 0
- fee_adjusted_return > 0
- false_safe_tail <= 0.20
- severe_downside_recall >= 0.75
- baseline 3개 이상 핵심 metric에서 우위
- 기존 1D product line 대비 false_safe 또는 severe_recall 개선
- seed stability 유지
- feature_set 설명 가능

분류:

- recommended_default: 제품 v1 저장 CP 후보
- selectable_verified: 제품 기본은 아니지만 사용자가 선택 가능한 후보
- risk_conservative_variant: 수익 추종성은 약하지만 위험선 가치가 있는 후보
- experiment_record: 결과 보존
- rejected: 탈락
- design_needed: feature/model 정의가 아직 부족

## 9. Stage 7: product save는 별도 CP

Stage 6에서 후보가 확정되면 별도 CP에서만 save-run을 수행한다.

별도 CP에서 확인할 것:

- model_runs.status=completed
- role=line_model
- checkpoint 존재
- config에 feature_set/horizon/beta/source_data_hash/context_checksum 기록
- predictions/inference 저장은 별도 지시가 있을 때만 수행
- band/composite 저장 금지

## 10. W&B naming 규칙

권장 project: `lens-eodhd500-line` 또는 `lens-cp148-lm-1d`

run name 형식:

`cp148_1d_{model}_h5_{feature_set}_seq{seq_len}_{patch}_beta2_seed{seed}`

예:

- `cp148_1d_patchtst_h5_pvv_seq252_p32s16_beta2_seed42`
- `cp148_1d_tide_h5_pvv_seq252_beta2_seed42`

W&B init 실패 시 학습 자체는 local log 기준으로 계속 가능해야 한다. local log는 필수다.

## 11. 재현성 기록

각 run은 다음을 반드시 남긴다.

- provider/source/hash
- source_data_hash
- context_checksum
- feature_set
- feature columns
- seed
- train/val/test date 또는 split manifest
- checkpoint_selection
- line_loss alpha/beta/delta
- W&B run URL
- local log path
- code commit/hash, 가능하면 기록
- cache manifest

## 12. 실행 후 보고서 템플릿

나중에 실행 CP의 결과 보고서는 다음 구조를 따른다.

1. 실험 목적
2. 데이터/피처 계약
3. baseline 결과
4. coarse model 결과
5. Optuna sweep 결과
6. seed stability 결과
7. 실패 유형 분석
8. 최종 후보
9. 왜 이 모델을 v1로 고르는가
10. 왜 탈락 후보는 탈락했는가
11. 다음 v2 개선 방향

## 13. 운영 주의사항

터미널 실행 중 출력이 멈추거나 GPU compute가 보이지 않으면 정상 학습으로 가정하지 않는다. 후보별 progress/local log와 python 프로세스 CPU delta를 확인하고, 20~30초 동안 같은 stage에서 CPU delta가 0이면 잔여 실행을 컷한 뒤 병목 stage를 수정하고 재실행한다.

`.venv\\Scripts\\python.exe`만 사용한다. 시스템 Python으로 실행된 흔적이 있으면 결과를 폐기하거나 별도 WARN으로 분류한다.
