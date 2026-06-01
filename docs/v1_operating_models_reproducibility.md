# Lens v1 운영 모델 재현성 매니페스트

목적: 화면에 붙어있는 운영 모델 3종(예측선 / 1D 밴드 / 1W 밴드)을 **나중에 다시 만들 수 있도록** 학습 시점/데이터/하이퍼/환경을 한 장에 모아둠.

v1 전체 실험을 다루지 않는다. **v1의 결론으로 살아남은 3개 모델만** 정리한다. 전체 실험 타임라인은 CP216(라인) / CP217(밴드)에서 별도 정적 문서로 작성.

> 재현 사고 예방: 라인 v1 성능을 한 번 재현 시도하다 넘어진 적이 있다 (데이터 윈도우가 매일 갱신되니까). 도커는 환경만 잠그고, 데이터/seed/config는 별도 lock이 필요하다.

---

## 0. 환경 (공통)

### Python / 가상환경

| 항목 | 값 |
|---|---|
| Python | `3.10.0` |
| 가상환경 | `C:\Users\user\lens\.venv` (Windows venv) |
| 활성화 | `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`) |
| 의존 정의 | `requirements.txt` (사용자 직접 관리, pin 일부만) |

핵심 패키지 (pip freeze 시점 = 2026-05-30):

| 패키지 | 버전 |
|---|---|
| torch | `2.11.0+cu128` |
| numpy | `1.26.4` |
| pandas | `2.2.2` |
| pyarrow | `16.1.0` |
| scipy | `1.15.3` |
| fastapi | `0.111.0` |
| uvicorn | `0.30.1` |
| optuna | `4.8.0` |

전체 freeze 스냅샷 권장 캡처 (수동):

```powershell
.venv\Scripts\pip.exe freeze > docs\v1_pip_freeze_20260530.txt
```

### GPU / CUDA

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| compute capability | `sm_120` (12.0) |
| CUDA runtime | `12.8` |
| PyTorch CUDA build | `cu128` |
| 추가 환경 변수 | `KMP_DUPLICATE_LIB_OK=TRUE`, `TORCHDYNAMO_DISABLE=1` |
| DataLoader | `num_workers=0` (Windows + sm_120 호환 위해 폴백) |

> `sm_120` 은 일반 PyTorch 휠에 항상 들어있지 않다. nightly 또는 자체 빌드가 필요할 수 있어 v2 도커 train 단계에서 base image 잠그기가 까다롭다. (§v2 master plan 11.6)

### 데이터 소스 / 시점

| 항목 | 값 |
|---|---|
| 시장 데이터 provider | `yfinance` (운영 활성) |
| 보조 provider | `eodhd` (코드는 있으나 운영 비활성, `EODHD_API_KEY=""`) |
| 유니버스 | S&P 500 (`data/parquet/price_data_yfinance_500.parquet`) |
| feature parquet (1D) | `data/parquet/indicators_yfinance_1D_500.parquet` |
| feature parquet (1W) | `data/parquet/indicators_yfinance_1W_500.parquet` |
| feature_version | `v3_adjusted_ohlc` |
| feature_set | `price_volatility_volume` |
| 갱신 주기 | 매일 / 매주 (cron + `run_v1_unified_refresh_local.ps1`) |
| 마지막 일일 refresh (이 문서 시점) | `2026-05-29` |

> 학습 시점의 데이터 윈도우는 **매일 가변**이다. 운영 모델이 학습된 시점의 데이터셋을 그대로 재현하려면 (a) 그 시점의 parquet 스냅샷이 보존되어 있어야 하고 (b) `source_data_hash` 가 일치해야 한다.

---

## 1. 예측선 — CP210 F4 β=4 5-seed Ensemble (현재 운영)

> 정확히는 CP210은 ensemble forward verification만 수행했고, 새 학습은 CP209에서 끝났다. CP210은 CP209의 5 seed checkpoint를 묶어 ship 판정.

### 모델 식별

| 항목 | 값 |
|---|---|
| run_id (parquet model_id) | `cp210_F4_b4_ensemble_mean` |
| source_cp | `CP208Z_CP209_F4B4` |
| ship 판정 | `NO_SHIP` (WF IC range 0.0457 > 기준 0.040) — 사용자가 일단 채택 |
| 백본 | PatchTST `p32/s16` (CP175/CP208 backbone lock 동일) |
| 출력 contract | score형 `line_score`, `safe_line_score` (수익률 단위, 화면에서 종가에 환산) |
| 손실 | Asymmetric MSE, α=1, β=4 (낙관 오차에 더 큰 페널티) |
| ensemble | 5 seed mean (median은 부산물) |
| timeframe / horizon | 1D / h5 (5거래일) |

### 데이터

| 항목 | 값 |
|---|---|
| split | `calendar_aligned` (시간 기준 train/val/test) |
| feature_pack | `F4_stress_delta_plus_yield_curve` |
| feature 컬럼 | `atr_ratio`, `vix_change_5d`, `credit_spread_change_20d`, `ma200_pct_change_20d`, `yield_curve` + 기본 가격 채널 |
| feature coverage | 거의 1.0 (vix_change_5d 0.9983, credit_spread 0.9932, ma200_pct 0.9932) |
| source data hash (CP208Z) | `11bf3a4831d54815` |
| seeds (CP209 5-seed) | `7, 13, 23, 42, 71` |
| walk-forward fold 수 | 4 (W1~W4) |

### 학습/검증/테스트 윈도우 (CP209 4-fold WF)

| fold | train_end | test_start | test_end |
|---|---|---|---|
| W1 | 2024-10-29 | 2024-10-30 | 2025-02-28 |
| W2 | 2025-02-28 | 2025-03-01 | 2025-06-30 |
| W3 | 2025-06-30 | 2025-07-01 | 2025-10-31 |
| W4 | 2025-10-31 | 2025-11-01 | 2026-05-01 |

> train_start / val_start 는 CP209 verification metrics 에 미수록. CP209 학습 스크립트 진행 로그에서 추가 추출 가능.

### 베이스라인 (커트라인 근거)

| baseline | role | IC | false-safe | severe recall |
|---|---|---:|---:|---:|
| CP175 beta=5 | 제품 기준선 | 0.0420 | 0.1972 | 0.7921 |
| CP164 calendar line | 알파 기준선 | 0.0436 | 0.2056 | 0.6732 |
| CP175 beta=2 | 중립 기준선 | 0.0444 | 0.2126 | 0.6152 |

### 성능 (test ensemble mean)

| metric | 값 | ship 기준 |
|---|---:|---|
| IC | 0.0325 | ≥ 0.030 PASS |
| false-safe | 0.2048 | ≤ 0.210 PASS |
| severe recall | 0.7727 | ≥ 0.750 PASS |
| spread | 0.0055 | — |
| WF IC range (4 fold) | 0.0457 | ≤ 0.040 FAIL |

### 산출물 경로

- ensemble report: `docs/cp210_ensemble_report.md`
- progress: `docs/cp210_progress_latest.md`
- artifact: `data/artifacts/cp210/` (5 seed checkpoint)
- feature lock: `docs/cp208z_feature_pack_lock.md`
- baseline lock: `docs/cp208z_baseline_lock_report.md`
- environment lock: `docs/cp208z_environment_lock.md`
- 운영 parquet: `backend/data/v1/predictions_line_1d.parquet`

### 재현 절차 (개략)

1. `.venv` 활성, 위 핵심 패키지 버전 동일하게.
2. 학습 시점 parquet 스냅샷 복원 — `source_data_hash = 11bf3a4831d54815` 일치해야 함.
3. CP209 학습 스크립트 5 seed 재실행 (시드 리스트는 CP209 진행 로그 참조).
4. CP210 ensemble forward 스크립트로 mean 묶기.

---

## 2. AI 밴드 1D — CP153 TiDE (현재 운영)

### 모델 식별

| 항목 | 값 |
|---|---|
| run_id (parquet model_id) | `tide-1D-ea54dcae654d` |
| source_cp | `CP153` |
| 백본 | TiDE (Time-series Dense Encoder, Google 2023) |
| 출력 contract | quantile pair (q_low/q_high) → conformal 보정 |
| q_low / q_high | `0.15 / 0.85` (= target coverage 70%) |
| calibration | `lower_focused`, validation-only fit |
| timeframe / horizon | 1D / h5 (5거래일) |

> ⚠️ **PPT 초기 계획서는 밴드 포함율 목표 90% 였으나, v1 운영은 70% (q15~q85)로 좁혔다.** 1D 5거래일짜리에 90% CI는 너무 넓어 실용성이 떨어진다고 판단한 결과. CP215에서 정직하게 적을 것.

### 데이터

| 항목 | 값 |
|---|---|
| provider | `yfinance` |
| source_data_hash | `90666b44cbfb8e5c` |
| feature_version | `v3_adjusted_ohlc` |
| feature_set | `price_volatility_volume` |
| feature 컬럼 | `log_return, open_ratio, high_ratio, low_ratio, vol_change, ma_5_ratio, ma_20_ratio, ma_60_ratio, rsi, macd_ratio, bb_position` |
| target | `raw_future_return` |
| seeds (stage5T) | `7, 42, 123` |
| save-run asof | `2026-05-08` |

### 학습/검증/테스트 윈도우 (CP153 Stage 5T true walk-forward, 3-fold)

| fold | train_start | train_end | val_start | val_end | test_start | test_end |
|---|---|---|---|---|---|---|
| fold_1 | 2019-05-01 | 2024-05-01 | 2024-05-01 | 2024-11-01 | 2024-11-01 | 2025-05-01 |
| fold_2 | 2019-11-01 | 2024-11-01 | 2024-11-01 | 2025-05-01 | 2025-05-01 | 2025-11-01 |
| fold_3 | 2020-05-01 | 2025-05-01 | 2025-05-01 | 2025-11-01 | 2025-11-01 | 2026-05-09 |

- 각 fold 마다 fresh checkpoint 학습 (replay 아닌 true WF)
- calibration: fold validation 에서만 fit, test 에 고정 적용 (`lower_focused`)

### 성능 (save-run test)

| metric | 값 | Stage 5T TiDE ref | delta |
|---|---:|---:|---:|
| coverage_abs_error | 0.009887 | 0.025440 | -0.0156 (개선) |
| lower_breach_rate | 0.158586 | 0.142540 | +0.0160 |
| upper_breach_rate | 0.151300 | 0.156973 | -0.0057 |
| band_width_ic | 0.375986 | 0.373995 | +0.0020 |
| downside_width_ic | 0.086588 | 0.086193 | +0.0004 |

product gate: PASS.

### 산출물 경로

- primary report: `docs/cp153_bm_1d_band_primary_product_candidate_save_run_report.md`
- run meta: `docs/cp153_bm_1d_band_primary_product_candidate_run_meta.json`
- calibration params: `docs/cp153_bm_1d_band_primary_product_candidate_calibration_params.json`
- latest artifact: `docs/cp153_bm_1d_band_primary_product_candidate_latest_predictions.json`
- 운영 parquet: `backend/data/v1/predictions_band_1d.parquet`
- Stage 별 보고서: `docs/cp153_bm_1d_band_500_stage{0_1,2,2_5,3,4,5}_*_report.md`

### 재현 절차 (개략)

1. `.venv` 활성, 핵심 패키지 동일.
2. 학습 시점 yfinance parquet 스냅샷 복원 — `source_data_hash = 90666b44cbfb8e5c` 일치.
3. Stage 1 baseline → Stage 2 model zoo → Stage 2.5 expansion → Stage 4 seed stability → Stage 5 walk-forward → primary save-run 순서로 재현.
4. calibration은 validation-only `lower_focused` 적용.

---

## 3. AI 밴드 1W — CP178 TiDE WFLOCK (현재 운영)

### 모델 식별

| 항목 | 값 |
|---|---|
| run_id (parquet model_id) | `tide_s60_q10_q90_param` |
| source_cp | `CP178-WFLOCK` |
| 백본 | TiDE (1D와 동일) |
| q_low / q_high | `0.10 / 0.90` (목표 coverage 80%) |
| calibration | walk-forward lower calibration |
| timeframe / horizon | 1W / h4 (주 4주) |

### 데이터

| 항목 | 값 |
|---|---|
| provider | `yfinance` |
| feature parquet | `data/parquet/indicators_yfinance_1W_500.parquet` |
| seeds (stage5) | `7, 42, 123` |
| 운영 parquet asof 범위 | `2024-11-01 ~ 2026-04-17` (138,908 rows) |

### 학습/검증/테스트 윈도우 (CP178 Stage 5 true walk-forward, 3-fold)

CP153 1D 와 동일한 fold 정의 사용 (1W 표본 수 적음을 감안한 동일 calendar 정렬):

| fold | train_end | test_start | test_end |
|---|---|---|---|
| fold_1 | 2024-05-01 | 2024-11-01 | 2025-05-01 |
| fold_2 | 2024-11-01 | 2025-05-01 | 2025-11-01 |
| fold_3 | 2025-05-01 | 2025-11-01 | 2026-05-09 |

운영 모델은 WFLOCK (`cp178_wflock_1w_band_walk_forward_lower`) 채택. lower calibration 별도 적용.

### 성능 (운영 시점 기준 — TrainingView에 박힌 수치)

| metric | 1W 값 | 1D 비교 |
|---|---:|---:|
| coverage_abs_error | 0.039 | 0.007 |
| band_width_ic | 0.34 | 0.40 |

1W 밴드가 1D 보다 coverage 오차 크고 band_width_ic 낮은 이유 = 주간 난이도 + 주봉 표본 수가 적음.

### 산출물 경로

- adaptive calibration: `docs/cp178_alt_1w_band_adaptive_calibration_report.md`
- walk-forward lower: `docs/cp178_wflock_1w_band_walk_forward_lower_report.md`
- lower calibration: `docs/cp178_cal_1w_band_lower_calibration_report.md`
- rescue expansion: `docs/cp178_bm_1w_band_500_rescue_expansion_report.md`
- Stage 별 보고서: `docs/cp178_bm_1w_band_500_stage{0,1,2,3,4,5}_*_report.md`
- 운영 parquet: `backend/data/v1/predictions_band_1w.parquet`

### 재현 절차 (개략)

1. `.venv` 활성, 핵심 패키지 동일.
2. 1W feature parquet 스냅샷 복원.
3. Stage 1~5 순차 재현 + walk-forward lower calibration 적용 (WFLOCK).
4. adaptive calibration 적용 시 별도 보고서 절차 참조.

---

## 4. 보강 필요 (TODO)

다음 정보가 보고서에 미수록되어 있어 추후 보강:

- [x] CP209 라인 학습 윈도우 — fold W1~W4 추출 완료 (train_start 일부만 미확정)
- [x] CP153 1D 밴드 학습 윈도우 — Stage 5T fold_1~3 추출 완료
- [x] CP178 1W 밴드 학습 윈도우 — Stage 5 fold_1~3 추출 완료 (1D와 동일 정의)
- [ ] 각 학습 시점의 git sha (자동 박는 구조는 v2)
- [ ] 학습 wall clock time, peak GPU memory
- [ ] 학습 시점 parquet 스냅샷 보존 여부 (없으면 정확 재현 불가)

위 항목은 v2 도입 시 모든 학습 스크립트가 표준 manifest JSON을 자동 생성하도록 잡으면 자연스럽게 해결된다. (§v2 master plan 11.7)

---

## 4.5 PPT 초기 계획 평가지표 vs v1 실제

초기 계획서 (`OneDrive/Desktop/2026-1/딥러닝실습/딥러닝실습_프로젝트계획서_22011859_김지형.pptx` 슬라이드 10) 평가지표를 v1 운영에 어떻게 반영/대체했는지. CP216 정적 평가 화면 작성 시 정직 근거.

| PPT 지표 | PPT 목표 | v1 운영 대응 | 차이/사유 |
|---|---|---|---|
| MAPE (예측가 vs 실제가) | < 5% | **미사용** | 라인이 가격 절대값이 아니라 수익률 score 라 직접 산정 안 함. 대신 IC / severe recall 채택 |
| 방향 정확도 (horizon 기준) | > 55% | **IC 0.0325** (라인) | rank correlation 이 방향성+크기를 같이 본다. IC > 0 으로 약하지만 정통과 |
| 밴드 포함율 | **> 90%** | **70%** (q15~q85) 1D, **80%** (q10~q90) 1W | 5거래일 90% CI 가 너무 넓어 실용성 떨어짐 → q-pair 좁히고 conformal 보정 |
| 모델 간 성능 비교 | 각각 확인 | PatchTST(라인) / TiDE(밴드) 채택, CNN-LSTM/TCN 비교 reserve | CP153 model zoo + CP208Z baseline lock |
| 지지/저항 일치 (정성) | 시각적 확인 | 미진행 | 시각 검증은 데모 화면에서 사용자 확인. 별도 정량 평가 없음 |
| API 응답 속도 | < 3초 | 추정 PASS (lru_cache + columns filter) | 정량 측정 미수록. v2 관측성 트랙에서 박을 것 |

**v1 에서 추가로 사용한 지표 (PPT 미수록)**

| 지표 | 의미 | 운영 값 | 커트라인 근거 |
|---|---|---|---|
| IC | 순위 상관 | 0.0325 (라인) | CP175 baseline 0.042 / CP164 0.044 |
| severe recall | 큰 하락 포착률 | 0.7727 (라인) | CP175 0.79, CP164 0.67, CP175 β=2 0.62 |
| false-safe rate | 위험 오판율 | 0.2048 (라인) | CP175 0.197 |
| band_width_ic | 밴드 폭 vs 실제 변동성 상관 | 0.376 (1D) / 0.34 (1W) | TiDE ref 0.374 |
| coverage_abs_error | 목표 포함율 - 실제 포함율 | 0.0099 (1D save-run) / 0.039 (1W) | Stage 5T ref 0.025 (1D) |
| lower_breach_rate | 하단 이탈률 | 0.159 (1D) | target q15 |

## 5. 한 줄 요약

운영 모델 3개의 학습 시점 환경(Python 3.10 / torch 2.11.0+cu128 / RTX 5060 Ti sm_120) · 데이터 소스(yfinance 500 / `v3_adjusted_ohlc`) · feature set(F4 라인 / `price_volatility_volume` 밴드) · 출력 contract(score 라인 / q15-q85 밴드 1D / q10-q90 밴드 1W)는 잠겼다. **학습 윈도우 정확 일자는 보고서에 미수록** — v2 manifest 자동화에서 해결.
