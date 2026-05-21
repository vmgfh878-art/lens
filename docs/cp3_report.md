# CP3 완료

## 1. sufficiency gate 구현
- 위치 함수
  - [`eligible_tickers`](/C:/Users/user/lens/ai/splits.py)
  - [`make_splits`](/C:/Users/user/lens/ai/splits.py)
  - [`build_dataset_plan`](/C:/Users/user/lens/ai/preprocessing.py)
  - [`prepare_dataset_splits`](/C:/Users/user/lens/ai/preprocessing.py)
- Gate 기준
  - Gate A: `sample_count = row_count - seq_len - h_max + 1 >= 1`
  - Gate B: `train_count >= min_fold_samples`
  - Gate C: `val_count >= min_fold_samples AND test_count >= min_fold_samples`
- `min_fold_samples` 결정값: `50`

## 2. split 구현
- 비율: `70/15/15`
- gap 적용: `h_max`
  - `1D = 20`
  - `1W = 12`
- 구현 방식
  - 먼저 `sample_count`를 계산
  - train-val, val-test 경계마다 `h_max`만큼 샘플 인덱스를 비워 두고 split
- 경계 분리 테스트 통과: `Y`

## 3. 최종 학습 대상
### 1D (seq_len=252, h_max=20)
- 입력 유니버스: `477`
- 통과: `473`
- 제외: `30`
  - Gate fundamentals: `26`
  - Gate A: `1`
  - Gate B: `0`
  - Gate C: `3`
- 제외 ticker 목록
  - `AMP`, `APA`, `BALL`, `BK`, `BXP`, `CHTR`, `CINF`, `CPAY`, `DUK`, `EA`, `EME`, `EXE`, `GEV`, `GLW`, `HOOD`, `INVH`, `KEYS`, `LMT`, `MS`, `PRU`, `Q`, `RF`, `SNDK`, `SOLV`, `T`, `TDG`, `UBER`, `VICI`, `VLTO`, `XYZ`

### 1W (seq_len=104, h_max=12)
- 입력 유니버스: `477`
- 통과: `421`
- 제외: `81`
  - Gate fundamentals: `25`
  - Gate A: `6`
  - Gate B: `30`
  - Gate C: `20`
- 제외 ticker 목록
  - `ABNB`, `AMP`, `APA`, `APP`, `BALL`, `BK`, `BXP`, `CARR`, `CEG`, `CHTR`, `CINF`, `COIN`, `CPAY`, `CRWD`, `CTVA`, `CVNA`, `DASH`, `DDOG`, `DELL`, `DOW`, `DUK`, `EA`, `EME`, `EXE`, `FOX`, `FOXA`, `FTV`, `GEHC`, `GEV`, `GLW`, `HOOD`, `HWM`, `INVH`, `IR`, `KEYS`, `KVUE`, `LMT`, `MRNA`, `MS`, `OTIS`, `PLTR`, `PRU`, `RF`, `SNDK`, `SOLV`, `T`, `TDG`, `TTD`, `UBER`, `VICI`, `VLTO`, `VRT`, `VST`, `VTR`, `VTRS`, `VZ`, `WAB`, `WAT`, `WBD`, `WDAY`, `WDC`, `WEC`, `WELL`, `WFC`, `WM`, `WMB`, `WMT`, `WRB`, `WSM`, `WST`, `WTW`, `WY`, `WYNN`, `XEL`, `XOM`, `XYL`, `XYZ`, `YUM`, `ZBH`, `ZBRA`, `ZTS`

## 4. 샘플 수 집계
### 1D
- ticker별 train 샘플 p10/p50/p90: `1315 / 1731 / 1731`
- ticker별 val 샘플 p10/p50/p90: `281 / 370 / 370`
- ticker별 test 샘플 p10/p50/p90: `283 / 372 / 372`
- 전체 train 샘플 수: `759124`
- 전체 val 샘플 수: `162351`
- 전체 test 샘플 수: `163205`

### 1W
- ticker별 train 샘플 p10/p50/p90: `275 / 275 / 275`
- ticker별 val 샘플 p10/p50/p90: `58 / 58 / 58`
- ticker별 test 샘플 p10/p50/p90: `60 / 60 / 60`
- 전체 train 샘플 수: `115653`
- 전체 val 샘플 수: `24395`
- 전체 test 샘플 수: `25233`

## 5. 테스트
- 기존 21 테스트 green: `Y`
- 신규 테스트: `4건`
  - `test_make_splits_rejects_gate_a_when_rows_too_short`
  - `test_make_splits_rejects_gate_b_and_gate_c_at_thresholds`
  - `test_make_splits_keeps_gap_between_folds`
  - `test_timeframe_filters_are_independent`
- 경계 분리 테스트: `Y`
- timeframe 독립성 테스트: `Y`
- 전체 테스트 수: `25`

## 6. 학습 진입점 연결
- [`ai/train.py`](/C:/Users/user/lens/ai/train.py)에서 `build_dataset_plan -> prepare_dataset_splits` 체인 반영: `Y`
- dry-run으로 split 구성 출력 확인: `Y`
  - `python -m ai.train --dry-run --timeframe 1D --seq-len 252 --horizon 5`
  - `python -m ai.train --dry-run --timeframe 1W --seq-len 104 --horizon 4`
- 실제 학습 실행: 금지 준수 `Y`

## 7. CP4 준비 상태
- ticker embedding 입력 인터페이스가 3모델(PatchTST/CNN-LSTM/TiDE)에 이미 반영돼 있는지: `N`
- 학습 대상 ticker 수 기준 embedding 차원 결정 가능 여부
  - `1D = 473`
  - `1W = 421`

## 메모
- `1D`는 CP2.6의 `seq_len` 가능 수치 `476`과 비교해 `473`으로 줄었다. 이 차이는 fundamentals 제외 26개 외에 `Gate A 1개`, `Gate C 3개`가 추가로 걸렸기 때문이다.
- `1W`는 CP2.6의 `seq_len` 가능 수치 `472`에서 `421`로 크게 줄었다. 이유는 이번 CP에서 `min_fold_samples=50`과 `gap=12`를 실제 split 규칙으로 적용하면서 `Gate B 30개`, `Gate C 20개`가 추가로 걸렸기 때문이다.
