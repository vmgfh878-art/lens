# CP46-G 기획값 대비 구현 기본값 감사

감사일: 2026-04-29

## 1. 한 줄 결론

현재 Lens는 데이터 split의 `h_max` 계약은 잘 지키고 있지만, 제품 horizon 기본값, CLI 기본 `seq_len`, checkpoint 선택 기본값, 최신 예측 표시 방식이 원 기획과 다르게 흘러갔다. 특히 `horizon=5` 결과는 성능 결론이 아니라 **short-horizon branch**로 분리해야 하며, 제품 기본을 `1D=20`, `1W=12`로 복구할지 별도 결정해야 한다.

## 2. 감사 범위와 준수 사항

- 코드 수정 없음.
- 모델 학습 없음.
- DB 쓰기 없음.
- 확인 대상은 문서, CLI 기본값, 서비스 기본값, 최근 CP 보고서에 한정했다.

주요 근거 파일:

- `README.md`
- `docs/cp3_instruction.md`
- `docs/cp3.5_instruction.md`
- `docs/cp13_patchtst_solo_plan.md`
- `docs/cp_product_demo_plan.md`
- `docs/training_hyperparameters.md`
- `docs/project_journal.md`
- `ai/train.py`
- `ai/preprocessing.py`
- `ai/splits.py`
- `ai/loss.py`
- `ai/evaluation.py`
- `ai/backtest.py`
- `backend/app/services/model_svc.py`
- `backend/app/services/api_service.py`
- `frontend/src/api/client.ts`
- `frontend/src/components/StockView.tsx`
- `frontend/src/components/Chart.tsx`

## 3. 기본값 불일치 총괄

| 항목 | 기획서/SoT 값 | 코드 기본값 | 최근 실험 사용값 | 차이 원인 추정 | 분류 |
| --- | --- | --- | --- | --- | --- |
| horizon | `1D h∈{1,5,20}`, `1W h∈{1,4,12}`. 제품 장기 기본 후보는 `1D=20`, `1W=12` | `ai.preprocessing.default_horizon`: `1D=5`, `1W=4`; `model_svc.DEFAULT_HORIZONS`: `1D=5`, `1W=4` | CP29 이후 대부분 `1D horizon=5` | CP3.5 dry-run, smoke, short-horizon 실험이 제품/API 기본으로 굳어진 것으로 보임 | 임시 smoke / 제품 오류 위험 |
| h_max | `1D=20`, `1W=12` | `ai.splits.MAX_HORIZON_BY_TIMEFRAME`: `1D=20`, `1W=12` | split gap은 h_max 기준 | CP3 계약이 코드에 정착됨 | 의도된 변경, 일치 |
| seq_len | `1D=252`, `1W=104` | `ai.train --seq-len`: `60`; preprocessing 기본도 `60` | PatchTST line은 `252`, CNN-LSTM band는 `60/45/90/120` | CNN-LSTM band rescue와 smoke 편의값이 전역 CLI 기본으로 남음 | 임시 smoke |
| batch_size | README 예시는 `128`, CP13 baseline은 `64`, ladder는 `64/128/256` | `ai.train --batch-size`: `64` | CP32 이후 주요 실험은 `256` | GPU 처리량 최적화와 반복 실험 편의 | 의도된 변경 |
| train/val/test ratio | `70/15/15` | `SPLIT_RATIO=(0.7,0.15,0.15)` | 동일 | CP3 설계 반영 | 의도된 변경, 일치 |
| train/val/test gap | `gap=h_max` | split spec `gap=h_max`; `1D=20`, `1W=12` | 동일 | CP3 leakage 방지 계약 반영 | 의도된 변경, 일치 |
| q_low/q_high | README/초기 기획은 q10/q90 중심 | CLI 기본 `0.1/0.9` | PatchTST line `0.25/0.75`, CNN-LSTM band `0.20/0.80`, 일부 `0.15/0.85` | coverage 과보수/미달 보정 실험으로 좁은 밴드 후보가 생김 | 의도된 변경 |
| lambda_line | SoT loss 식에 포함 | CLI/loss 기본 `1.0` | 대부분 `1.0` | 유지 | 의도된 변경, 일치 |
| lambda_band | SoT loss 식에 포함 | CLI/loss 기본 `1.0` | CP29 이후 대다수 `2.0` | 밴드 학습 신호 강화 실험이 사실상 표준 후보가 됨 | 의도된 변경 |
| lambda_width | SoT loss 식에 포함 | CLI 기본 `0.1`, 하지만 loss 계산에는 미사용 | 실험값과 무관하게 미사용 | CP10 이후 레거시 호환 인자만 남음 | 근거 없는 임의값 |
| lambda_cross | SoT loss 식에 포함 | CLI/loss 기본 `1.0` | 대부분 `1.0` | lower/upper crossing 방지 유지 | 의도된 변경, 일치 |
| coverage gate | 최근 SoT는 band gate `0.75<=coverage<=0.95`, upper `<=0.15`, lower `<=0.20`, target coverage `0.85` | `ai.train` 동일. `coverage_gate`는 `combined_gate` alias | CP36 이후 `line_gate`, `band_gate`, `combined_gate` 사용 | 역할별 gate 분리 | 의도된 변경 |
| composite/product breach gate | CP43 이후 composite 기준은 coverage `0.75~0.90`, lower `<=0.12`, upper `<=0.15`, width ratio `<=1.25`, line-inside `>=0.95` | 학습 CLI 기본 gate와 별도 | CP43/CP46-M calibration 검증에 사용 | 학습 gate와 제품 composite gate가 분리됨 | 의도된 변경, SoT 정리 필요 |
| fee bps | 제품 기획은 fee-adjusted 지표 표시를 요구하나 숫자 기본값은 명확하지 않음 | evaluation/backtest 기본 `10.0` bps | 최근 실험도 사실상 10 bps | 코드 관습값이 SoT 없이 굳어짐 | 근거 없는 임의값 |
| target type | CP13은 `raw_future_return`과 `volatility_normalized_return` 비교. 현재 1차 기준은 raw | `raw_future_return` | 최근 주요 실험 모두 `raw_future_return` | clean feature 이후 raw branch 유지 | 의도된 변경 |
| market/sector excess target | 데이터 감사에서 필요성 제기 | `market_excess_return` 지원 흔적은 있으나 기본값 아님. sector excess 기본 없음 | 사용 안 함 | 다음 target branch로 남음 | 의도된 변경, 미완 |
| band_mode | 명시 SoT는 약함. CP35 이후 `direct/param` 비교 | CLI 기본 `direct` | CNN-LSTM band는 `direct`, TiDE 일부 `param` | direct가 저장/제품 후보로 살아남음 | 의도된 변경 |
| checkpoint_selection | CP36 이후 역할별 `line_gate`, `band_gate`, `combined_gate`; `coverage_gate`는 호환 alias | CLI 기본 `val_total` | 최근 유효 실험은 `line_gate` 또는 `band_gate`; 과거 `coverage_gate` | CP36 이전 기본값이 남음 | 근거 없는 임의값 / 제품 오류 위험 |
| W&B 정책 | README 예시는 `--use-wandb`; 최근 CP는 W&B sweep 금지 반복 | CLI 기본 W&B true, dry-run에서는 비활성 | CP29 이후 대부분 `--no-wandb` | 운영 안전상 끈 값이 실험 표준이 됨 | 의도된 변경, CLI 기본 위험 |
| save-run 정책 | README 예시는 `--save-run`; CP14 이후 후보 확정 전 저장 금지 원칙 | CLI 기본 false | 대부분 false. CP41/CP44-M만 저장 smoke | 저장/재현 계약 검증 시만 허용 | 의도된 변경 |
| product 표시 | 제품 기획은 가격 + AI band/line overlay. 사용자는 전체 차트 AI 밴드 계약을 별도 감사 대상으로 지정 | API는 latest prediction 1건을 반환. UI는 latest/fallback run의 `forecast_dates` 구간만 그림 | CP45 제품 확인도 latest composite run의 5거래일 예측 표시 | 저장된 rolling prediction series를 읽는 제품 계약이 아직 없음 | 제품 오류 |

## 4. 항목별 상세 판단

### 4.1 horizon / h_max

기획 기준은 `1D`에서 `h∈{1,5,20}`, `1W`에서 `h∈{1,4,12}`이며, split leakage 방지 기준은 `h_max`다. 코드는 `ai.splits.MAX_HORIZON_BY_TIMEFRAME`에서 `1D=20`, `1W=12`를 유지하고, split 경계도 `gap=h_max`를 사용한다. 이 부분은 정상이다.

문제는 기본 예측 horizon이다. `ai.preprocessing.default_horizon`과 `backend/app/services/model_svc.py`는 모두 `1D=5`, `1W=4`를 기본값으로 사용한다. CP3.5에서 `horizon=5`, `horizon=4`는 dry-run과 smoke에 등장했기 때문에, 현재 `horizon=5` 결과는 제품 장기 horizon 결론이 아니라 **short-horizon branch**로 분류해야 한다.

별도 판단 항목:

- 제품 기본을 `1D=20`, `1W=12`로 복구할지 결정해야 한다.
- 복구한다면 API 기본, UI 기본 선택, 저장 run 선택, inference 기본값을 함께 맞춰야 한다.
- split gap은 이미 `h_max`를 쓰므로, 이 결정은 leakage 방지보다 제품 horizon 계약 문제에 가깝다.

### 4.2 seq_len

기획 기준은 `1D seq=252`, `1W seq=104`다. PatchTST solo plan도 `seq_len=252`, `patch_len=16`, `stride=8`을 baseline으로 둔다. 반면 `ai.train --seq-len` 기본값과 preprocessing 기본값은 `60`이다.

최근 실험은 모델 역할별로 갈렸다.

- PatchTST line: 대체로 `seq_len=252`.
- CNN-LSTM band: `seq_len=60`이 생존 후보, CP45에서는 `45/60/90/120` sweep.
- TiDE rescue: `seq_len=252`.

따라서 `seq_len=60`을 전역 기본값으로 보는 것은 위험하다. 이는 CNN-LSTM band/smoke branch 값이지, PatchTST 또는 제품 SoT 기본값이 아니다.

### 4.3 batch_size

README 학습 예시는 `batch-size=128`이고, CP13 baseline은 `64`, ladder는 `64/128/256`이다. 코드 기본값은 `64`다. 최근 GPU 실험은 대부분 `256`을 사용했다.

이 차이는 성능 의미보다 처리량과 실험 반복성의 차이다. 다만 문서 예시, CLI 기본, 최근 실험 표준이 모두 달라서 재현 명령을 작성할 때 명시하지 않으면 결과 비교가 흔들릴 수 있다.

### 4.4 q_low/q_high

초기 기획과 CLI 기본은 q10/q90이다. 하지만 CP22 이후 coverage와 band width를 맞추는 과정에서 q20/q80, q25/q75, q15/q85가 반복 검증됐다. 최근 역할은 다음처럼 정리된다.

- PatchTST line 후보: q25/q75, `lambda_band=2`.
- CNN-LSTM band 후보: q20/q80 또는 q15/q85, `lambda_band=2`.
- composite calibration: raw q 자체보다 postprocess scale과 policy가 중요.

이 변경은 근거 없는 임의값이 아니라 coverage calibration을 위한 의도된 실험 변화다. 다만 제품 기본 q preset은 아직 하나로 고정됐다고 보기 어렵다.

### 4.5 lambda 계열

`lambda_line`, `lambda_band`, `lambda_cross`는 코드와 실험에서 실제 사용된다. `lambda_band=2.0`은 최근 실험의 사실상 표준 후보지만, CLI 기본은 여전히 `1.0`이다.

가장 큰 불일치는 `lambda_width`다. README와 초기 SoT loss 식에는 `lambda_width * width`가 포함되어 있지만, 현재 `ai.loss.ForecastCompositeLoss`에서는 width 항이 total loss에 들어가지 않는다. 문서에는 CP10 이후 레거시 호환 인자로 남겼다고 기록되어 있다.

판단:

- `lambda_width`는 현재 학습 제어값이 아니다.
- CLI에 남아 있어도 사용자가 band width를 loss로 제어한다고 오해할 수 있다.
- 분류는 `근거 없는 임의값`이다. 값 자체보다 “있는 척하는 비활성 노브”가 위험하다.

### 4.6 coverage/breach gate

학습 checkpoint gate는 CP36 이후 역할별로 분리됐다.

- `line_gate`: line 지표 중심.
- `band_gate`: coverage `0.75~0.95`, upper breach `<=0.15`, lower breach `<=0.20`, width와 band loss finite.
- `combined_gate`: 두 gate 모두 통과.
- `coverage_gate`: 호환 alias.

코드도 이 정책을 반영한다. 다만 CLI 기본 `checkpoint_selection=val_total`은 현재 실험 체계와 어긋난다. `val_total`은 loss가 좋아도 제품 band 조건이 망가진 checkpoint를 고를 수 있으므로, current branch에서는 명시 selector가 필요하다.

또 하나의 분리는 composite/product gate다. CP43 이후 composite calibration은 coverage `0.75~0.90`, lower breach `<=0.12`, upper breach `<=0.15`, width ratio `<=1.25`, line-inside `>=0.95`를 본다. 이는 학습용 band gate보다 제품 보수성이 강하다. 이 자체는 의도된 변경이지만, “학습 gate”와 “제품 composite gate”라는 이름으로 문서화하지 않으면 같은 coverage 숫자를 서로 다르게 해석할 위험이 있다.

### 4.7 fee bps

`ai.evaluation`과 `ai.backtest` 기본 fee는 `10.0` bps다. 제품 기획은 fee-adjusted 지표 표시를 요구하지만, 10 bps라는 숫자의 SoT 근거는 확인되지 않았다.

판단:

- 현재 값은 재현 가능한 코드 기본값이다.
- 하지만 기획값이라기보다는 관습값이다.
- 분류는 `근거 없는 임의값`이다.

### 4.8 target type

현재 기본 target은 `raw_future_return`이고, 최근 주요 실험도 모두 raw branch다. CP13에서 `volatility_normalized_return` 비교가 제안됐고, CP28/CP29 계열에서 market/sector excess target 필요성이 제기됐지만 기본값으로 승격되지는 않았다.

판단:

- `raw_future_return` 기본은 현재 의도된 변경으로 볼 수 있다.
- market excess, sector excess는 다음 target branch 후보이며 현 제품 기본은 아니다.

### 4.9 band_mode

CLI 기본은 `direct`다. TiDE rescue에서 `param`이 일부 검증됐지만, 저장/제품 후보는 대부분 `direct` 계열이다.

판단:

- 현재 `direct` 기본은 타당하다.
- 다만 `param`은 TiDE 전용 rescue 성격이라 제품 기본값으로 보기는 어렵다.

### 4.10 checkpoint_selection

현재 CLI 기본은 `val_total`이다. 그러나 CP36 이후 실험 설계는 line 모델은 `line_gate`, band 모델은 `band_gate`, composite 후보는 별도 composite policy로 판단한다.

판단:

- `val_total` 기본은 현재 실험 계약에 맞지 않는다.
- 이 값은 legacy default로 보이며 `근거 없는 임의값 / 제품 오류 위험`으로 분류한다.
- 최소한 재현 명령에는 checkpoint selector를 항상 명시해야 한다.

### 4.11 W&B / save-run

README 예시는 `--use-wandb --save-run`을 사용하지만, 최근 CP는 W&B sweep과 save-run을 반복적으로 금지했다. 코드 기본은 W&B true, save-run false다.

판단:

- save-run 기본 false는 안전하다.
- W&B 기본 true는 로컬 재현이나 금지 CP에서 놀랄 수 있는 기본값이다.
- 최근 운영 정책은 “후보 확정 전 no-wandb/no-save-run, 저장 계약 smoke에서만 save-run”이다.

### 4.12 product 표시 정책

제품 기획은 가격 차트 위에 AI 밴드와 보수적 예측선을 overlay하는 것이다. 하지만 현재 구현은 “전체 rolling AI band”가 아니라 “latest 또는 fallback run의 최신 prediction forecast segment”를 표시한다.

확인된 구현 흐름:

- 프론트는 완료 run 목록에서 최신 또는 fallback run을 고른다.
- API는 `run_id`가 없으면 latest prediction 1건을 조회한다.
- 차트는 `prediction.forecast_dates` 구간만 AI line/band로 그린다.
- `StockView`는 첫 forecast date 이전의 실제 가격과 forecast date를 합쳐 timeline을 만든다.

따라서 현재 제품은 전체 과거 차트 구간을 AI band가 덮지 않는다. 사용자가 지정한 “AI 밴드가 전체 차트를 덮어야 하는 제품 계약” 기준으로 보면 이 항목은 `제품 오류`다.

필요한 후속 판단:

- full rolling AI band가 제품 계약이면, API는 단일 latest prediction이 아니라 as-of date별 prediction series를 반환해야 한다.
- UI는 최신 forecast segment overlay와 historical rolling band overlay를 구분해야 한다.
- 저장 정책도 ticker/date/horizon/run 기준 rolling prediction을 조회할 수 있어야 한다.

## 5. 최근 실험값 요약

| CP 범위 | 대표 값 | 성격 |
| --- | --- | --- |
| CP29-D smoke | PatchTST, `1D`, `seq_len=252`, `horizon=5`, `batch_size=256`, `q20/q80`, `lambda_band=2`, `coverage_gate`, no W&B, no save-run | clean feature 후 50티커 1epoch smoke |
| CP32/33 | PatchTST, `seq_len=252`, `horizon=5`, `q25/q75`, `lambda_band=2`, `coverage_gate`, `batch_size=256` | PatchTST clean/revin branch |
| CP35/37 | TiDE/CNN-LSTM rescue, `horizon=5`, `batch_size=256`, no W&B, no save-run | 대체 모델 rescue smoke |
| CP39/42 | PatchTST line `seq_len=252 q25/q75`, CNN-LSTM band `seq_len=60 q20/q80`, `horizon=5`, `batch_size=256` | 100/200 ticker 제한 검증 |
| CP43/46-M | composite calibration, `horizon=5`, coverage/breach/width/line-inside policy | 후처리 calibration 검증 |
| CP44-M | 5 ticker save-run smoke, composite run 저장 | 저장 계약 smoke, 성능 판단 아님 |
| CP45 | CNN-LSTM band sweep, `limit_tickers=100`, `epochs=3`, `batch_size=256`, no W&B summary 기본 | band 후보 탐색 |

이 표의 `horizon=5` 결과는 모두 short-horizon branch다. 제품 기본 horizon 성능으로 읽으면 안 된다.

## 6. 별도 판단 항목

### 6.1 제품 기본 horizon 복구

현재 코드/API 기본은 `1D=5`, `1W=4`다. 원 기획의 장기 제품 기본을 살리려면 `1D=20`, `1W=12` 복구 여부를 결정해야 한다.

판단 옵션:

| 옵션 | 의미 | 장점 | 위험 |
| --- | --- | --- | --- |
| 현행 유지 | 제품 기본도 short horizon 유지 | 현재 저장 run과 UI가 바로 맞음 | 원 기획의 장기 band 제품과 불일치 |
| `1D=20`, `1W=12` 복구 | 제품 기본을 h_max horizon으로 맞춤 | SoT와 제품 서사에 맞음 | 현재 학습/저장 후보 대부분이 h5라 재학습/재저장 필요 |
| dual 기본 | UI에서 short/long을 명시 분리 | 실험 branch와 제품 branch를 동시에 보존 | API, run selection, UX 복잡도 증가 |

감사 판단은 “복구 여부 미결정”이다. 다만 `horizon=5`를 제품 기본으로 계속 쓰려면 SoT 문서를 바꿔야 한다.

### 6.2 전체 rolling AI band 계약

현재 구현은 latest forecast segment 표시다. 전체 차트를 덮는 rolling AI band가 계약이면, 현재 제품은 계약 미충족이다.

필요 변경 범위는 다음과 같다.

- prediction API: latest 1건이 아니라 rolling prediction series 조회.
- DB 조회 계약: ticker/timeframe/horizon/run/asof_date별 여러 prediction row 반환.
- UI 차트: 실제 가격 구간 위의 historical predicted band와 미래 forecast segment를 구분.
- provenance: latest run, fallback run, rolling window의 as-of 기준 표시.

이번 CP는 코드 수정 금지이므로 변경하지 않았다.

### 6.3 역할별 기본값 분리

현재 단일 CLI 기본값으로는 Lens의 실제 branch를 표현하기 어렵다. 최소한 문서상 역할별 기본값을 분리해야 한다.

| 역할 | 권장 기본 해석 |
| --- | --- |
| 제품 표시 | horizon은 `1D=20`, `1W=12` 복구 여부 결정 전까지 미정. 현재 구현은 h5 latest forecast segment |
| PatchTST line 연구 | `1D`, `horizon=5`, `seq_len=252`, q25/q75, `lambda_band=2`, `line_gate`, no W&B/save-run |
| CNN-LSTM band 연구 | `1D`, `horizon=5`, `seq_len=60`, q20/q80 또는 q15/q85, `lambda_band=2`, `band_gate`, no W&B/save-run |
| 저장 smoke | limit ticker 소량, save-run 허용, 성능 판단 금지 |
| full/product candidate | 아직 금지 또는 미정. h5 branch와 h20 product branch를 분리해야 함 |

## 7. 위험도 우선순위

| 우선순위 | 항목 | 이유 |
| --- | --- | --- |
| P0 | product horizon 기본값 불일치 | 코드/API 기본 `5/4`가 원 기획의 `20/12` 제품 계약을 덮고 있다 |
| P0 | latest forecast segment vs full rolling AI band | 사용자가 보는 제품이 전체 AI band 계약을 만족하지 않는다 |
| P1 | `checkpoint_selection=val_total` 기본 | role gate 체계와 맞지 않아 잘못된 checkpoint를 대표 후보로 고를 수 있다 |
| P1 | `lambda_width` 비활성 노브 | SoT loss 식과 코드 동작이 다르고, 사용자가 width loss를 조절한다고 오해할 수 있다 |
| P1 | `seq_len=60` 전역 기본 | PatchTST/제품 SoT의 `252/104`와 다르며 CNN-LSTM branch 값으로 보인다 |
| P2 | fee bps 10 근거 부족 | backtest 숫자는 재현되지만 기획 근거가 약하다 |
| P2 | W&B 기본 true | 최근 CP 금지 정책과 반대라 재현 명령에서 명시가 필요하다 |

## 8. 감사 결론

이번 감사에서 “모델이 실패했다”는 결론보다 먼저 정리해야 할 것은 기본값 계층이다. 데이터 split과 leakage 방지 계약은 대체로 살아 있지만, 제품 horizon과 표시 계약은 short-horizon 실험 branch에 끌려 내려왔다.

다음 CP에서 가장 먼저 해야 할 일은 다음 3개다.

1. `horizon=5` branch와 제품 기본 branch를 공식 분리한다.
2. 제품이 `1D=20`, `1W=12` 및 full rolling AI band를 요구하는지 확정한다.
3. 학습 CLI 기본값과 재현 명령에서 `seq_len`, `checkpoint_selection`, W&B/save-run 정책을 명시적으로 고정한다.

이번 보고서는 코드 수정, 학습, DB 쓰기를 수행하지 않았다.
