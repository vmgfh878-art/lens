# CP13 PatchTST Solo Track 재정의 + 논문 정합 감사

## 1. 방향 재정의
- 이번 단계의 목적은 모델 수를 늘리는 것이 아니라, PatchTST 하나를 논문 기준으로 다시 정렬하고 실험 축을 깨끗하게 재정의하는 것이다.
- TiDE, CNN-LSTM은 당분간 freeze한다.
- 새 head, 새 target, 새 sweep은 PatchTST 외 모델에 열지 않는다.

## 2. Freeze 범위
### 2.1 TiDE freeze
- 새 head 추가 금지
- 새 target 추가 금지
- 새 sweep 추가 금지
- 기존 회귀 테스트와 기존 실행 가능성만 유지

### 2.2 CNN-LSTM freeze
- 새 head 추가 금지
- 새 target 추가 금지
- 새 sweep 추가 금지
- 기존 회귀 테스트와 기존 실행 가능성만 유지
- CP12.6에서 잡은 CUDA 종료 안정성은 유지하되, 성능 실험 축은 더 늘리지 않는다

## 3. Phase 1.5 backlog
- CNN-LSTM direction head 확장 실험 재개
- TiDE 미래 공변량 본격 실험
- rank target 실험
- direction target 실험
- TiDE/CNN-LSTM 전용 sweep 재개
- 공통 multi-model 비교표 확대

## 4. PatchTST 현재 구현 감사표
| 항목 | 현재 구현 | 판정 | 분류 | 메모 |
|---|---|---|---|---|
| RevIN 적용 | 입력 전체 채널에 `norm`, 출력은 `target_channel_idx` 기준 `denormalize_target` | 부분 일치 | 의도적 차이 | 원 논문 취지와 대체로 맞지만, 현재는 target 채널 복원만 강하게 쓰는 구조다 |
| Channel Independence | `channel_independent=True`일 때 채널별 patch 후 encoder 공유 | 일치 | 의도적 구현 | PatchTST 핵심 아이디어와 맞다 |
| `ci_aggregate=target` | target 채널 hidden만 사용 | 부분 일치 | 의도적 차이 | 논문의 기본 PatchTST와 완전히 같은 표현은 아니고, Lens 목적에 맞춘 target 채널 우선 집계다 |
| `ci_aggregate=mean` | 채널 평균 집계 | 차이 | 의도적 차이 | 실험용 변형이다 |
| `ci_aggregate=attention` | 채널 attention 집계 | 차이 | 의도적 차이 | 실험용 변형이다 |
| `ci_target_fast` | target 채널만 backbone 입력 | 차이 | 의도적 차이 | 속도 최적화용이다. 논문 정합성은 낮아진다 |
| patching | 모델 init에는 `patch_len`, `stride`가 있으나 train/sweep CLI에는 아직 미노출 | 부분 일치 | 실험 인프라 미완 | PatchTST 핵심 sweep 축이므로 CP14 초반에 CLI 또는 전용 sweep entry를 열어야 한다 |
| positional embedding | learned positional embedding 사용 | 대체로 일치 | 의도적 구현 | 통상 구현과 호환적이다 |
| dropout | input, encoder, output dropout 존재 | 대체로 일치 | 의도적 구현 | 과하지 않다 |
| head 구조 | 공통 line/band head 사용 | 차이 | 의도적 차이 | 논문 본체보다 Lens의 밴드 예측 목적이 강하다 |
| ticker embedding | backbone 뒤 concat | 차이 | 의도적 차이 | 논문 바깥 Lens 전용 확장이다 |
| target type | 현재 raw future return 중심 | 일치 가능 | 의도적 선택 | PatchTST solo 실험의 1순위 축으로 적절하다 |

## 5. 공식 구현 대비 차이 구분
### 5.1 의도적 차이
- `ci_aggregate=target/mean/attention`
- `ci_target_fast`
- line + band 다중 head
- ticker embedding
- raw return 기반 밴드 예측 평가 체계

### 5.2 실수 가능성이 있는 차이
- `ci_aggregate=target`가 기본값이라, channel independence의 full 정보 활용보다 target 채널 편향이 먼저 들어간다
- RevIN이 전체 입력에 적용되지만, 출력 복원은 target 채널에만 강하게 의존한다
- `ci_target_fast=True`일 때는 사실상 Channel Independence 실험이 아니라 단일 채널 근사 모델에 가까워진다
- 현재 sweep 축이 LR, regularization 쪽에 비해 target type과 patch geometry 우선순위가 낮게 배치되어 있었다

## 6. PatchTST Solo Track 원칙
- 먼저 PatchTST를 논문 기반 단일 모델 트랙으로 세운다
- target type을 제일 먼저 검증한다
- patch 구조와 입력 길이를 그 다음에 검증한다
- LR 대탐색보다 구조적으로 설명 가능한 축을 우선한다
- direction head, rank head는 이번 단계에서 열지 않는다

## 7. PatchTST sweep 축 재설계
### 7.1 우선순위
1. `target type`
2. `patch_len`, `stride`
3. `seq_len`
4. `ci_aggregate`
5. `d_model`, `n_layers`, `dropout`

### 7.2 지금 당장 금지
- direction head 추가
- rank head 추가
- TiDE/CNN-LSTM 미러링
- 긴 LR sweep 선행

## 8. 실험 매트릭스 v1
### 8.1 Stage A: target type 고정 실험
- 모델: `patchtst`
- timeframe: `1D`
- `ci_aggregate=target`
- `ci_target_fast=False`
- `patch_len=16`, `stride=8`
- `seq_len=252`
- target 후보
  - `raw_future_return`
  - `volatility_normalized_return`

### 8.2 Stage B: patch geometry 실험
- target type은 Stage A best 1개로 고정
- 실험 축
  - `(patch_len=8, stride=4)`
  - `(patch_len=16, stride=8)`
  - `(patch_len=24, stride=12)`
- 주의:
  - 현재 `ai/train.py` CLI에는 `patch_len`, `stride`, `d_model`, `n_layers`, `n_heads` 인자가 아직 직접 노출돼 있지 않다
  - 따라서 Stage B는 CP14 초반에 PatchTST 전용 실험 인자 노출 또는 전용 sweep entry 정리 후 실행한다

### 8.3 Stage C: seq_len 실험
- target type, patch geometry 고정
- 실험 축
  - `seq_len=126`
  - `seq_len=252`

### 8.4 Stage D: ci_aggregate 실험
- `target`
- `mean`
- `attention`
- 단, `ci_target_fast=True`는 속도 비교용 보조 실험으로만 둔다

### 8.5 Stage E: 경량 용량 조정
- `d_model`
- `n_layers`
- `dropout`
- 이 단계는 앞 축이 고정된 뒤에만 연다

## 9. 평가 기준
- 공통 평가는 이미 있는 체계를 그대로 쓴다
  - `direction_accuracy`
  - `spearman_ic`
  - `top_k_long_spread`
  - `top_k_short_spread`
  - `long_short_spread`
  - `coverage`
  - `avg_band_width`
  - `mae`
  - `smape`
  - `fee_adjusted_return`
  - `fee_adjusted_sharpe`
  - `fee_adjusted_turnover`
- PatchTST solo 트랙에서는 성능보다 먼저, 이 지표들이 실험 축에 따라 얼마나 일관되게 움직이는지를 본다

## 10. CP14에서 바로 돌릴 baseline 명령
### 10.1 Baseline A: raw return
```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train `
  --model patchtst `
  --timeframe 1D `
  --seq-len 252 `
  --epochs 5 `
  --batch-size 64 `
  --device cuda `
  --no-wandb `
  --no-compile `
  --ci-aggregate target `
  --line-target-type raw_future_return `
  --band-target-type raw_future_return
```

### 10.2 Baseline B: volatility normalized return
```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train `
  --model patchtst `
  --timeframe 1D `
  --seq-len 252 `
  --epochs 5 `
  --batch-size 64 `
  --device cuda `
  --no-wandb `
  --no-compile `
  --ci-aggregate target `
  --line-target-type volatility_normalized_return `
  --band-target-type volatility_normalized_return
```

### 10.3 Baseline C: ci_aggregate 비교 시작점
```powershell
C:\Users\user\lens\.venv\Scripts\python.exe -m ai.train `
  --model patchtst `
  --timeframe 1D `
  --seq-len 252 `
  --epochs 5 `
  --batch-size 64 `
  --device cuda `
  --no-wandb `
  --no-compile `
  --ci-aggregate mean `
  --line-target-type raw_future_return `
  --band-target-type raw_future_return
```

### 10.4 CP14 초반 선행 작업
- PatchTST 전용 실험 인자 노출 여부 결정
  - `patch_len`
  - `stride`
  - `d_model`
  - `n_heads`
  - `n_layers`
- 이 인자들이 열리기 전까지는 Stage B, Stage E를 실제 명령으로 돌리지 않는다

## 11. CP14 실행 순서 권고
1. `target type` 2개만 먼저 비교
2. best target 고정
3. PatchTST 전용 실험 인자 노출 또는 전용 sweep entry 준비
4. best geometry 고정
5. `seq_len` 비교
6. 마지막에만 `ci_aggregate` 비교

## 12. 학습 시간 ETA 운영
- 이후 PatchTST Research CP는 실행 전 예상 소요 시간을 반드시 포함한다
- 첫 epoch가 끝나면 실제 epoch time 기준으로 남은 시간을 다시 계산한다
- 보고서에는 `batch_size`, VRAM 최대 사용량, GPU util, epoch time, 총 실행 시간을 남긴다
- 입력 NaN 정합이 닫힌 뒤에는 `batch_size=64/128/256` ladder로 처리량을 확인한다
- batch 증량은 VRAM을 채우기 위한 목적이 아니라 같은 실험을 더 빨리 끝내기 위한 처리량 최적화로 본다

## 13. 입력 정합 복구 후 실행 원칙
- 재무 6개 컬럼 NaN imputation과 `has_fundamentals` 플래그 경로가 복구됐다
- `raw_future_return` 전체 학습은 NaN 없이 완료됐지만 473티커 5epoch 기준 약 80분이 걸렸다
- 이후 새 target/구조는 전체 학습으로 바로 가지 않는다
- 모든 새 조건은 먼저 `--limit-tickers 50 --epochs 1 --batch-size 64` smoke를 통과해야 한다
- smoke 통과 후 `batch_size=64/128/256` ladder로 처리량을 측정한다
- full run은 후보가 좁혀진 뒤에만 허용한다

## 14. 최종 정리
- 이제 모델을 늘리지 않는다
- PatchTST 하나를 논문 기준으로 다시 세운다
- 그 뒤에 백엔드/프론트 데모 루프를 닫는다
- TiDE/CNN-LSTM은 freeze하고, Phase 1.5 backlog로만 남긴다
