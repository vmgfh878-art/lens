# CP29-D 가격/피처 계약 수리 보고서

## 한 줄 판단

지금은 “PatchTST가 실패했다”가 아니라 “PatchTST가 깨진 가격 피처를 먹고 있었을 가능성이 크다”입니다. 그래서 이건 오히려 좋은 발견이에요. 모델 갈아엎기 전에 잡아야 할 진짜 병목을 찾은 쪽에 가깝습니다.

## 적용한 가격 기준 계약

모델 입력 가격 피처는 adjusted 기준으로 통일했다.

- `adj_factor = adjusted_close / close`
- `adj_open = open * adj_factor`
- `adj_high = high * adj_factor`
- `adj_low = low * adj_factor`
- `adj_close = adjusted_close`

`log_return`, `open_ratio`, `high_ratio`, `low_ratio`, 이동평균 이격도, RSI, MACD, Bollinger, ATR 계산 경로는 adjusted OHLC를 사용하도록 수정했다. `1D`는 즉시 adjusted OHLC로 변환하고, `1W`/`1M`은 일별 adjusted OHLC를 만든 뒤 주봉/월봉으로 resample한다.

## 코드 변경

- `backend/app/services/feature_svc.py`
  - adjusted OHLC 변환과 가격 정합성 검증을 추가했다.
  - `high >= low`, `high >= open/close`, `low <= open/close` 계약을 검증한다.
  - `open_ratio/high_ratio/low_ratio`의 finite 여부와 폭주 여부를 검증한다.
  - `build_price_features`를 추가해 DB의 오래된 indicator를 price 데이터 기준으로 다시 덮어쓸 수 있게 했다.
  - `vol_change`의 `Inf`를 제거해 finite contract를 통과하도록 했다.
- `ai/preprocessing.py`
  - feature contract version을 `v3_adjusted_ohlc`로 bump했다.
  - `fetch_training_frames`에서 `price_data`로 가격 파생 피처를 재계산해 기존 `indicators`의 깨진 가격 ratio를 덮어쓴다.

이번 CP에서는 `has_fundamentals`, macro, breadth 구조는 바꾸지 않았다.

## 캐시 무효화

무작정 삭제하지 않고 contract version bump로 새 캐시 이름을 만들었다. 기존 v2 캐시는 남겨 두었고, v3 학습 경로는 아래 새 캐시를 사용한다.

| universe | old feature cache | new feature cache | old index cache | new index cache |
|---:|---|---|---|---|
| 50 | `features_1D_40f0c40a15a4_f77c891d.pt` | `features_1D_90e245c45310_f77c891d.pt` | `feature_index_1D_9f3468f07706_f77c891d.pt` | `feature_index_1D_1a967362529f_f77c891d.pt` |
| 100 | `features_1D_0e369e48b09f_f77c891d.pt` | `features_1D_82bda25b318d_f77c891d.pt` | `feature_index_1D_261af1e1df8d_f77c891d.pt` | `feature_index_1D_cb26f4024d0e_f77c891d.pt` |
| 200 | `features_1D_ca978dc79a3b_f77c891d.pt` | `features_1D_abbd75353d4f_f77c891d.pt` | `feature_index_1D_565acb0d9022_f77c891d.pt` | `feature_index_1D_92d8fcad896b_f77c891d.pt` |

feature index는 날짜/티커 eligibility만 담고 있어 값 수리 대상은 아니지만, cache key에 contract version이 들어가므로 v3 이름으로 다시 생성했다.

## CP28 대비 분포 정상화

200 universe 기준 `open_ratio/high_ratio/low_ratio` 폭주는 제거됐다.

| 피처 | 이전 mean | 이전 std | 이전 p99 | 이전 max | 수리 후 mean | 수리 후 std | 수리 후 p99 | 수리 후 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `open_ratio` | 0.840277 | 4.125401 | 23.470229 | 56.188089 | 0.000408 | 0.012402 | 0.033812 | 0.594203 |
| `high_ratio` | 0.862225 | 4.176397 | 23.797772 | 61.684083 | 0.012339 | 0.017881 | 0.074500 | 0.701449 |
| `low_ratio` | 0.818189 | 4.075289 | 22.646632 | 55.861157 | -0.011606 | 0.017725 | 0.018251 | 0.550827 |

50/100 universe에서도 같은 패턴으로 정상화됐다. 100 universe의 `open_ratio` p99는 24.769831에서 0.032638로 내려갔고, 50 universe의 `high_ratio` p99는 19.084899에서 0.071703으로 내려갔다.

## Finite Contract

feature contract 적용 후 50/100/200 universe 모두 non-finite model feature count가 0이다.

| universe | rows after repair | rows after required contract | non-finite |
|---:|---:|---:|---:|
| 50 | 129,762 | 129,762 | 0 |
| 100 | 252,026 | 252,026 | 0 |
| 200 | 505,991 | 505,991 | 0 |

`docs/cp28_feature_inventory.csv`도 v3 adjusted OHLC 기준으로 재생성했다. `vol_change`는 finite가 되었지만 max가 15,420으로 여전히 꼬리가 두꺼워 다음 CP에서 로그 변환 또는 winsorization 후보로 남긴다.

## 50티커 1epoch Smoke

허용 범위 안에서만 실행했다.

- 명령: `python -m ai.train --model patchtst --timeframe 1D --seq-len 252 --horizon 5 --epochs 1 --batch-size 256 --limit-tickers 50 --patch-len 16 --patch-stride 8 --q-low 0.20 --q-high 0.80 --lambda-band 2.0 --band-mode direct --ci-aggregate target --checkpoint-selection coverage_gate --no-wandb --num-workers 0`
- `--save-run`은 사용하지 않았다.
- full 473티커 학습은 실행하지 않았다.

결과 JSON과 `after_stdio_flush` marker까지 출력되어 학습/평가는 완료됐다. 다만 프로세스가 출력 후 종료되지 않아 셸 timeout으로 exit code는 124가 기록됐다. smoke 판단은 “학습/평가 완료, 종료 처리 timeout”이다.

주요 지표:

| 항목 | 값 |
|---|---:|
| train samples | 80,497 |
| val samples | 17,238 |
| test samples | 17,174 |
| val coverage | 0.994953 |
| val spearman_ic | 0.013894 |
| test coverage | 0.994690 |
| test spearman_ic | 0.010485 |
| test long_short_spread | -0.000501 |

이번 smoke의 목적은 성능 개선 판정이 아니라 finite/sanity contract 통과 확인이다. 그 기준은 통과했다.

## 검증

- `python -m unittest backend.tests.test_feature_svc`: 7 tests OK
- `python -m unittest backend.tests.test_feature_svc ai.tests.test_preprocessing ai.tests.test_cp9_final`: 22 tests OK
- `python -m unittest ai.tests.test_checkpoint_selection ai.tests.test_evaluation_targets ai.tests.test_patchtst_cli_config`: 12 tests OK
- 50/100/200 v3 feature finite contract: non-finite 0
- 50티커 PatchTST 1epoch smoke: 학습/평가 완료, 종료 처리 timeout

## 다음 판단

가격 피처 폭주는 제거됐다. 이제 CP28에서 본 50→100→200 확장 불안정성을 다시 해석해야 한다. 이전 결과 일부는 PatchTST 구조 문제가 아니라 깨진 OHLC ratio와 거래량 `Inf`를 먹은 결과였을 가능성이 크다. 다음 단계는 v3 feature cache 기준으로 50→100 안정성만 다시 확인하고, 그 전까지 full 473티커는 계속 금지하는 쪽이 맞다.
