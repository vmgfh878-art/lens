CP38은 새 모델 비교가 아니라, quantile 기반 밴드 후보의 validation/test calibration gap을 줄이는 CP다.

# CP38-M Band Calibration Gap 보고서

## 1. 목표
CP37에서 역할 기반으로 살아난 후보를 유지하되, 밴드 모델의 validation/test gap을 후처리 calibration으로 줄일 수 있는지 확인했다. full 473티커, W&B sweep, UI 수정, 신규 모델, DLinear/NLinear, save-run은 하지 않았다.

## 2. 대상 후보
원본 gap 진단은 CP37의 세 band 후보를 대상으로 했다.

| 후보 | val coverage | test coverage | val width | test width | test breach 특징 | 진단 |
|---|---:|---:|---:|---:|---|---|
| TiDE q10-b2 param | 0.823158 | 0.589787 | 0.107054 | 0.105727 | lower breach 0.274450 | 중심/분포 이동, lower breach 우세 |
| TiDE q10-b2 direct | 0.776830 | 0.529882 | 0.160960 | 0.134663 | lower breach 0.312600 | test 밴드 축소 + lower breach 우세 |
| CNN-LSTM seq60 q20-b2 direct | 0.755517 | 0.698842 | 0.068763 | 0.082414 | upper/lower 비교적 균형 | test undercoverage |

TiDE 계열은 test에서 밴드 폭만의 문제가 아니라 actual이 lower 쪽으로 밀리는 분포/중심선 이동 문제가 크다. CNN-LSTM seq60은 breach가 비교적 균형이라 width calibration으로 해결될 가능성이 더 높다.

## 3. 구현한 calibration 방식
`ai/band_calibration.py`를 추가했다. checkpoint를 로드해 val/test 예측 텐서를 모으고, 학습 loss를 바꾸지 않은 채 후처리 보정만 비교한다.

구현한 방식:

- scalar width calibration: validation residual/breach 기준으로 line 기준 lower/upper 폭에 각각 단일 scale을 곱한다.
- conformal residual calibration: validation residual의 분위수로 line 기준 lower/upper offset을 산출해 test band에 적용한다.

두 방식 모두 quantile loss 학습 자체를 대체하지 않는다.

## 4. TiDE param 결과
| 방식 | val coverage | test coverage | test lower breach | test upper breach | test width | test band_loss | coverage gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 원본 | 0.823158 | 0.589787 | 0.274450 | 0.135763 | 0.105727 | 0.056600 | 0.233371 |
| scalar width | 0.849983 | 0.648969 | 0.210411 | 0.140620 | 0.116588 | 0.055147 | 0.201013 |
| conformal residual | 0.849913 | 0.598300 | 0.270886 | 0.130814 | 0.126687 | 0.054563 | 0.251613 |

scalar width는 test coverage를 0.589787에서 0.648969로 올렸지만, 여전히 0.75 기준에 못 미치고 lower breach가 0.210411로 기준 0.20을 넘는다. conformal residual은 validation coverage만 맞추고 test gap은 오히려 커졌다.

TiDE param 판정: calibration 후에도 band 후보 생존 기준 미달이다.

## 5. CNN-LSTM seq60 결과
| 방식 | val coverage | test coverage | test lower breach | test upper breach | test width | test band_loss | coverage gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 원본 | 0.755174 | 0.699801 | 0.142628 | 0.157572 | 0.082385 | 0.026366 | 0.055373 |
| scalar width | 0.849949 | 0.829036 | 0.092571 | 0.078392 | 0.128096 | 0.031031 | 0.020913 |
| conformal residual | 0.849541 | 0.708474 | 0.138792 | 0.152734 | 0.111810 | 0.037466 | 0.141067 |

scalar width calibration은 test coverage를 0.829036으로 올리고 upper/lower breach를 모두 기준 안으로 낮췄다. avg_band_width는 0.082385에서 0.128096으로 커졌지만 폭발적 증가는 아니다. band_loss는 0.026366에서 0.031031로 소폭 악화했다.

CNN-LSTM seq60 판정: scalar width calibration 적용 시 band 후보 생존이다.

## 6. Calibration 계수
| 후보 | 방식 | 계수 |
|---|---|---|
| TiDE param | scalar width | lower_scale 1.266029, upper_scale 0.938776 |
| TiDE param | conformal residual | lower_offset -0.059220, upper_offset 0.067540 |
| CNN-LSTM seq60 | scalar width | lower_scale 1.095193, upper_scale 1.420661 |
| CNN-LSTM seq60 | conformal residual | lower_offset -0.049270, upper_offset 0.062674 |

CNN-LSTM seq60은 upper scale이 크게 올라가면서 upper breach를 해결했다. TiDE param은 lower scale을 키워도 test lower breach가 여전히 높아 중심/분포 이동이 남아 있다.

## 7. 판정
밴드 후보 생존 기준은 test coverage 0.75~0.95, upper breach <= 0.15, lower breach <= 0.20, width 과도 폭증 없음, validation/test coverage gap 감소다.

| 후보 | 방식 | 생존 여부 | 이유 |
|---|---|---|---|
| TiDE param | scalar width | 탈락 | test coverage 0.648969, lower breach 0.210411 |
| TiDE param | conformal residual | 탈락 | test coverage 0.598300, gap 증가 |
| CNN-LSTM seq60 | scalar width | 생존 | test coverage 0.829036, breach 기준 통과, gap 감소 |
| CNN-LSTM seq60 | conformal residual | 탈락 | test coverage 0.708474 |

## 8. 결론
CP38 기준 band model 후보는 CNN-LSTM seq60 + scalar width calibration이다. TiDE param은 validation에서는 좋아 보이지만 test에서 lower breach가 크게 늘어나는 분포 이동 문제가 있어, 단순 calibration으로는 구조적 gap을 닫지 못했다.

다음 단계는 full run이 아니라 CNN-LSTM seq60 scalar calibration을 100티커 제한으로 확인하는 것이다.
