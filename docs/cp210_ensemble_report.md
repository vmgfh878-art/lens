# CP210 Ensemble Ship Verification

- 목적: CP209가 학습한 F4 beta=4, F6 beta=7의 5 seed checkpoint를 새 학습 없이 ensemble forward로 ship 판정한다.
- Ensemble: mean을 ship 판정 기준으로 사용하고, median은 부산물로 기록했다.
- 금지 준수: 새 학습, 새 seed, backbone/beta/feature 변경, pinball/curriculum/hybrid 없음.
- 진행 로그: `C:\Users\user\lens\logs\cp210_ensemble_ship_verification\progress.log`

## Ship 기준

| metric | 기준 |
|---|---:|
| test IC | >= 0.030 |
| test false-safe | <= 0.210 |
| test severe recall | >= 0.750 |
| WF fold IC | 4 fold 중 3 fold >= 0 |
| WF IC range | <= 0.040 |

## Test Ensemble Mean

| 후보 | IC | false-safe | severe recall | spread | fee | test 기준 |
|---|---:|---:|---:|---:|---:|---|
| F4 beta=4 | 0.032490 | 0.204808 | 0.772749 | 0.005477 | 0.004477 | PASS |
| F6 beta=7 | 0.030723 | 0.196408 | 0.835395 | 0.004904 | 0.003904 | PASS |

## Test Ensemble Median

| 후보 | IC | false-safe | severe recall | spread | fee |
|---|---:|---:|---:|---:|---:|
| F4 beta=4 | 0.032874 | 0.204200 | 0.775123 | 0.005620 | 0.004620 |
| F6 beta=7 | 0.029581 | 0.197734 | 0.834164 | 0.004924 | 0.003924 |

## Walk-Forward Ensemble Mean

| 후보 | fold | IC | false-safe | severe recall | spread |
|---|---|---:|---:|---:|---:|
| F4 beta=4 | W1 | 0.013158 | 0.249715 | 0.651072 | -0.001734 |
| F4 beta=4 | W2 | 0.008174 | 0.188755 | 0.807666 | -0.003318 |
| F4 beta=4 | W3 | 0.046086 | 0.160441 | 0.829097 | 0.008158 |
| F4 beta=4 | W4 | 0.053874 | 0.211729 | 0.795786 | 0.013835 |
| F6 beta=7 | W1 | 0.016722 | 0.235731 | 0.734277 | -0.001061 |
| F6 beta=7 | W2 | 0.000891 | 0.183735 | 0.872231 | -0.004664 |
| F6 beta=7 | W3 | 0.041652 | 0.153975 | 0.880368 | 0.006705 |
| F6 beta=7 | W4 | 0.054040 | 0.202991 | 0.852172 | 0.013532 |

## 판정

- F4 beta=4는 test 기준은 통과했지만 WF IC range가 `0.045700`으로 기준 `0.040`을 초과했다.
- F6 beta=7도 test 기준은 통과했지만 WF IC range가 `0.053149`로 기준 `0.040`을 초과했다.
- 두 후보 모두 4 fold 중 4 fold IC는 0 이상이지만, fold 간 IC range 기준에서 탈락했다.

최종 라벨: `NO_SHIP`

권장: ensemble이 test에서는 납득 가능한 수준까지 올라왔지만, 완화된 WF range 기준을 넘지 못했다. 지시서 기준대로라면 CP175 frozen 유지 또는 별도 hybrid/v2 재고가 맞다.

판단은 사용자 몫이다. 이 보고서는 ensemble이 완화된 ship 기준을 만족하는지 보조한다.