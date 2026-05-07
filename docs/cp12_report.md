# CP12 보고서

## 목적
CUDA/bf16 autocast 경로에서 발생하던 NaN/Inf로 GPU 학습 결과 해석이 오염되는 문제를 차단한다.
NaN 결과가 체크포인트·W&B·DB에 저장되지 못하도록 finite gate를 단계별로 깐다.
실패한 run은 `model_runs.status="failed_nan"`로 메타만 남기고, 결과 테이블 (predictions / prediction_evaluations / backtest_results)과 checkpoint는 저장하지 않는다.

## 1. 변경 파일
- `C:\Users\user\lens\ai\finite.py` (신규)
- `C:\Users\user\lens\ai\train.py`
- `C:\Users\user\lens\ai\inference.py`
- `C:\Users\user\lens\ai\backtest.py`
- `C:\Users\user\lens\backend\db\schema.sql`
- `C:\Users\user\lens\backend\db\scripts\ensure_runtime_schema.py`
- `C:\Users\user\lens\ai\tests\test_cp12_finite_gate.py` (신규)
- `C:\Users\user\lens\docs\cp12_report.md` (이 파일)

## 2. finite gate 구조

`ai/finite.py`에 다음 유틸을 두었다.

- `check_metrics_finite(metrics, *, phase, run_id, epoch, batch) -> FiniteCheckResult`
  - dict 안의 스칼라(float/int/0-dim Tensor)들을 isfinite 검사. 첫 실패 항목과 phase/metric/run_id/epoch/batch 정보를 담은 보고서를 반환.
  - `None` 값(예: spearman_ic가 표본 부족으로 None인 경우)은 통과시킨다.
- `assert_finite_metrics(...)`: 위 결과를 즉시 RuntimeError로 던진다.
- `tensor_finite_summary({name: tensor})`: 입력 tensor의 finite_ratio, has_nan, has_inf, min/max를 계산. None은 무시.
- `is_nan_safe_better(candidate, best, mode, min_delta)`: NaN candidate는 절대 best가 될 수 없고, NaN best는 어떤 finite candidate에게도 진다. EarlyStopping의 NaN 안전 비교용.

## 3. train.py 적용

다음 단계에 finite gate를 깔았다.

- `run_epoch`: 매 batch에서 `losses.total` + 각 loss component (line/band/cross/direction/forecast/total)의 isfinite 검사.
  - 첫 NaN batch에서 `_dump_nonfinite_diagnostics`가 입력 tensor (features, line_target, band_target, raw_future_returns, prediction.line/lower_band/upper_band/direction_logit) 와 loss component 의 finite 통계를 한 번 stderr로 덤프.
  - `direction_logit`이 None인 경우 (PatchTST/TiDE) 건너뛴다.
  - eval phase는 첫 NaN에서 즉시 RuntimeError.
  - train phase는 단발성 NaN을 흡수하기 위해 `NAN_STREAK_LIMIT=3` 연속 시 RuntimeError. streak 안에서 정상 batch가 한 번이라도 나오면 리셋.
  - `batch_count == 0` 가드: 모든 batch가 NaN이거나 loader가 비어 있는 경우 averaged dict가 0.0으로 위장 통과하지 못하도록 즉시 실패.
- `evaluate_loader`/`evaluate_bundle`: `amp_dtype` 인자 추가. `run_training`이 `config.amp_dtype`을 그대로 넘긴다.
- `run_training`:
  - 매 epoch val_summary에 `check_metrics_finite(phase="val")` 적용. 실패 시 W&B finish + `_persist_failed_run` + RuntimeError.
  - `checkpoint_metrics`에도 같은 검사. 실패 시 `save_checkpoint` 호출 자체를 차단.
  - test phase 결과(`test_quality`)에도 같은 검사.
  - 정상 종료 시 `save_model_run`에 `status="completed"`을 명시.
  - 실패 경로의 `_persist_failed_run`은 `val_metrics`/`test_metrics`를 빈 dict로 저장하고 `config.failure`에 phase/metric/epoch/batch를 함께 남긴다. `checkpoint_path=None`.
- `EarlyStopping`: NaN-aware 비교(`is_nan_safe_better`)로 갈아치웠다. NaN epoch는 best가 되지 않고, NaN epoch에서도 patience 카운트는 증가한다.

## 4. CLI 옵션
- `--amp-dtype {bf16,fp16,off}` (기본 `bf16`).
  - `off`: train/val/test 모든 phase에서 autocast 미사용 (`nullcontext`).
  - `bf16`/`fp16`: CUDA에서만 autocast 적용. CPU 디바이스에선 무시된다.
  - 부분 fp32 강제 (예: `--fp32-modules lstm,heads`)는 본 CP에서 노출하지 않는다. 매트릭스 결과상 필요하면 별도 CP로 발주.
- `--detect-anomaly`: `torch.autograd.set_detect_anomaly(True)` 활성화. backward NaN 발생 op를 추적하지만 학습 속도 상당히 느려지므로 평소엔 off.
- `TrainConfig`에 `amp_dtype`, `detect_anomaly` 필드 추가. 기본값 `"bf16"` / `False`로 기존 호출 호환 유지.

## 5. 결과 테이블 차단

| phase 발생 | model_runs | predictions | prediction_evaluations | backtest_results | checkpoint |
|---|---|---|---|---|---|
| 정상 종료 | status=`completed` | inference 단계에서 저장 | 동일 | backtest 단계에서 저장 | 저장 |
| NaN 실패 | status=`failed_nan`, `config.failure` 메타, `checkpoint_path=NULL` | **차단** | **차단** | **차단** | **차단** |

- `ai/inference.py`: `run_inference` 진입부에서 `model_runs.status != 'completed'`이면 `ValueError`로 거부. 결과적으로 `save_predictions` / `save_prediction_evaluations` 호출이 일어나지 않는다.
- `ai/backtest.py`: `run_backtest` 진입부에서 같은 status 가드. `save_backtest_results` 차단.

## 6. DB 스키마

`public.model_runs`에 status 컬럼 추가.

```sql
ALTER TABLE public.model_runs
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed';
```

CHECK 제약 (`status IN ('completed', 'failed_nan')`)은 신규 테이블 생성문에만 두었다. 기존 운영 DB는 `ensure_runtime_schema.py`의 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`로만 추가하고 CHECK 제약은 별도 수동 운영 단계에서 적용한다 (NULL 데이터가 없는 것을 확인한 뒤). 본 CP는 사용 코드에서 두 값만 기록한다.

마이그레이션 명령:

```
C:\Users\user\lens\.venv\Scripts\python.exe -m backend.db.scripts.ensure_runtime_schema
```

## 7. 검증 (코드 단)

- `ai/tests/test_cp12_finite_gate.py` 18 test (신규). 모두 통과.
- AI 테스트 전체: `Ran 110 tests in 4.691s — OK` (CP11 92 + CP12 18).
- backend 테스트: `Ran 23 tests in 0.122s — OK`.

검증 명령:

```
C:\Users\user\lens\.venv\Scripts\python.exe -m unittest discover -s ai\tests -p "test_*.py"
PYTHONPATH="C:\Users\user\lens\backend;C:\Users\user\lens" C:\Users\user\lens\.venv\Scripts\python.exe -m unittest backend.tests.test_feature_svc backend.tests.test_collector_jobs backend.tests.test_api
```

## 8. 매트릭스 (사용자 GPU 실행)

같은 조건: `seed=42`, `timeframe=1D`, `epochs=1`, `batch-size=64`, `line-target-type=raw_future_return`, `band-target-type=raw_future_return`, `--no-wandb`, `--no-compile`. 차이는 device · amp-dtype · `--use-direction-head` · ticker 수.

### 5티커 매트릭스 (원인 분리)

```
[1] CPU fp32 baseline
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cpu --amp-dtype off --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return

[2] CUDA amp off baseline
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cuda --amp-dtype off --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return

[3] CUDA bf16 baseline
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cuda --amp-dtype bf16 --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return

[4] CUDA amp off direction_head
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cuda --amp-dtype off --use-direction-head --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return

[5] CUDA bf16 direction_head
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 5 --seed 42 --device cuda --amp-dtype bf16 --use-direction-head --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return
```

### 50티커 (재현 확인)

```
[6] 50티커 CUDA bf16 baseline
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train --model cnn_lstm --timeframe 1D --epochs 1 --batch-size 64 --limit-tickers 50 --seed 42 --device cuda --amp-dtype bf16 --no-wandb --no-compile --line-target-type raw_future_return --band-target-type raw_future_return
```

각 명령은 다음 중 하나로 끝난다:
- 정상 종료 → 마지막에 `result` JSON 출력 (run_id, checkpoint_path, best_metrics, test_metrics).
- NaN 실패 → `[NaN-DIAG ...]` + `[NaN-GATE ...]` 한 줄 후 `RuntimeError`. checkpoint와 결과 테이블은 비어 있고, `save-run` 모드였다면 `model_runs.status='failed_nan'` 한 행만 남는다.

## 9. W&B 지표 키 클린업
- mape 잔여:
  - `ai/train.py`, `ai/evaluation.py`, `ai/inference.py`, `ai/baselines.py`, `ai/postprocess.py`, `ai/sweep.py`를 grep한 결과 `mape`/`MAPE` 잔여 없음. 모두 `smape`/`mae` 키로 정리되어 있다.
  - `wandb.log` 호출 4곳 (`ai/benchmark_gpu.py:111`, `ai/sweep.py:235`, `ai/train.py:814`, `ai/train.py:865`)에서 mape 키 미발견.
- 수정 여부: 코드 경로엔 수정할 잔여 없음. 과거 W&B sweep (`patchtst_lr_main`, `patchtst_lr_v2`)에 `test/mape`만 보였던 것은 CP10 이전 빌드로 학습이 돌아갔기 때문. 이후 학습은 자동으로 smape/mae로 들어간다.

## 10. 매트릭스 결과 보고 양식 (사용자가 채워서 보낼 자리)

각 명령 실행 후 마지막 stdout 한 덩어리 (정상이면 result JSON, 실패면 RuntimeError + 직전 NaN-DIAG/GATE 줄)를 아래 표에 담아서 회신.

| # | 조건 | 결과 (PASS/NAN) | first NaN phase/metric/batch | 비고 |
|---|---|---|---|---|
| 1 | 5티커 CPU fp32 baseline | | | |
| 2 | 5티커 CUDA amp off baseline | | | |
| 3 | 5티커 CUDA bf16 baseline | | | |
| 4 | 5티커 CUDA amp off direction_head | | | |
| 5 | 5티커 CUDA bf16 direction_head | | | |
| 6 | 50티커 CUDA bf16 baseline | | | |

판정 기준 (지시서 기준 그대로):
- CUDA amp off PASS + CUDA bf16 NAN → autocast/LSTM/Conv 정밀도 문제. 다음 CP에서 fp32 강제 모듈 옵션 검토.
- 양쪽 다 NAN → 모델 출력 또는 데이터 GPU 이동 문제. NaN-DIAG 의 features/prediction.* finite ratio가 단서.
- direction_head만 NAN → direction loss 또는 logits 문제. CP11 변경(BCEWithLogits 비대칭 가중치)을 의심.
- 50티커 bf16 PASS 시에만 CP11 direction head 재실험으로 진행. NAN이면 direction head 반복 실험은 금지.

## 11. 남은 리스크 / 다음 단계 후보

- 본 CP는 *학습 단계 NaN 차단*까지만 다룬다. 모델 구조 자체에 NaN 원인이 있다면 매트릭스 결과로 분리되며, fix는 별도 CP12.x로 분리한다 (지시서 5번 임시 해결 항목).
- `--detect-anomaly`는 학습 속도가 크게 느려지므로 매트릭스에는 기본 미적용. NaN 위치가 forward인지 backward인지 추가 분리가 필요할 때만 켠다 (예: `[3]`이 NaN이면 그 case만 anomaly mode로 한 번 더 실행).
- CP11 direction head 재평가는 매트릭스 [6] 50티커 bf16 baseline이 PASS된 이후로 미룬다. 동시에 lambda_direction sweep (다음 CP에서 0.0/0.05/0.1/0.2)과 PatchTST/TiDE direction head 미러 (fairness)가 후속 후보.
- 메모리 갱신 (`project_lens_cp_state.md`)은 매트릭스 결과 회신 후 사용자 closure 승인과 함께 수행한다.

## 12. 자가 리뷰 결론

- Plan v3 정합: 본 CP는 모델 손실 함수·밴드 본체 원칙을 변경하지 않는다. NaN 결과가 *밴드 평가*를 오염시키는 것을 막는 정합 보강이라 충돌 없음.
- 데이터 계약: `model_runs.status` 컬럼 추가만, 다른 결과 테이블 스키마는 그대로. 기존 reader가 status 컬럼을 모르면 문제없음 (default `'completed'`이라 기존 row와 동일하게 해석된다).
- 모델 영향: 학습 코드 path 변경분은 *NaN이 아닌 정상 batch에 대해선 동작 동일* 하도록 신경 썼다. amp_dtype 기본값 `bf16`은 기존과 동일.
- 회귀: 기존 `ai/tests/test_cp9_5.py`의 NaN 테스트는 eval phase 즉시 raise 경로로 통과. CP11 direction head 통합 테스트도 회귀 없음.
