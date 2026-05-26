# Band MLOps 계획

이 문서는 기존 1D/1W band 모델에 MLOps를 적용해 자동 재평가, 자동 재학습, 자동 승격 추천까지 이어지는 구조를 정리한다.

목표는 모델을 자동으로 갈아끼우는 것이 아니다. 새 데이터가 계속 들어오는 금융 도메인에서 현재 product slot 모델이 여전히 유효한지 확인하고, 더 나은 후보가 생겼을 때 사람이 판단할 수 있는 근거를 자동으로 만드는 것이다.

## 왜 band부터 MLOps인가

Line은 false-safe, break composition, stop-loss 해석처럼 제품 의미가 복잡하다. 반면 band는 coverage, lower/upper breach, width, stress regime coverage처럼 자동 검증 가능한 지표가 비교적 명확하다.

따라서 Phase 2 MLOps의 첫 대상은 기존 1D/1W band가 적합하다.

## 핵심 원칙

- 자동 데이터 검증은 허용한다.
- 자동 champion 재평가는 허용한다.
- 자동 challenger 학습은 허용한다.
- 자동 calibration 갱신은 제한적으로 허용한다.
- 자동 승격 추천은 허용한다.
- product slot 자동 교체는 금지한다.
- 최종 promotion은 사람이 승인한다.

## 대상 슬롯

| 슬롯 | 현재 역할 | MLOps 목표 |
|---|---|---|
| 1D Band | 단기 위험 범위 신호 | daily monitor, monthly challenger retrain, conformal calibration |
| 1W Band | 주간 위험 범위 신호 | payload와 serving 계약을 고정한 뒤 동일 루프 적용 |
| ETF Band | Phase 2 확장 후보 | US ETF universe가 생긴 뒤 별도 champion/challenger 운영 |
| Korea Band | Phase 2 이후 확장 후보 | 데이터 소스와 캘린더가 안정된 뒤 별도 운영 |

## 자동 루프

```text
새 가격 데이터 수집
-> snapshot/manifest 생성
-> data validation
-> feature build
-> champion band 재평가
-> baseline 재평가
-> challenger 후보 학습
-> calibration fit
-> seed stability 평가
-> walk-forward 평가
-> stress regime 평가
-> 비용/속도/저장 크기 확인
-> recommendation report 생성
-> 사람이 승인하면 product slot promotion
```

## 두 가지 자동화 트랙

### 1. Recalibration Track

기존 모델을 새로 학습하지 않고 calibration만 갱신한다.

주기:

- daily: coverage/breach monitor
- weekly: conformal Q 후보 계산
- monthly: calibration 교체 추천

통과 기준:

- calibration set leakage 없음
- coverage_abs_error 개선
- lower/upper breach 비대칭 악화 없음
- stress regime coverage 붕괴 없음
- band width가 과도하게 넓어지지 않음

이 트랙은 비용이 낮고 안정적이므로 Phase 2 초기에 먼저 붙인다.

### 2. Retraining Track

새 snapshot 기준으로 challenger 모델을 다시 학습한다.

주기:

- monthly: 제한된 모델 family retrain
- quarterly: 넓은 sweep 또는 새 feature family 검토
- event-driven: provider 변경, 큰 drift, coverage 붕괴 발생 시 수동 승인 후 실행

초기 후보:

| 후보 | 역할 |
|---|---|
| 기존 champion 재학습 | 같은 구조가 새 데이터에서 유지되는지 확인 |
| TiDE band challenger | 기존 1D band primary 계열 |
| TCN band challenger | secondary 후보 또는 1W 참고 후보 |
| GARCH baseline | stress regime 비교 기준 |
| Historical Quantile baseline | 단순 통계 기준 |
| Bollinger baseline | 사용자가 이해하기 쉬운 chart 기준 |

## 검증 gate

### Data Gate

- provider/source/hash 고정
- ticker universe 변경 기록
- duplicate 0
- non-finite 0
- adjusted OHLC sanity 통과
- split overlap 0
- 최신 asof_date 확인

### Metric Gate

| 지표 | 의미 |
|---|---|
| coverage_abs_error | 목표 coverage와 실제 coverage 차이 |
| lower_breach_rate | 하방 이탈률 |
| upper_breach_rate | 상방 이탈률 |
| asymmetric_interval_score | 폭과 이탈 penalty를 함께 본 점수 |
| band_width_ic | band 폭이 실제 변동성과 관련 있는지 |
| downside_width_ic | 하방 폭이 실제 하방 위험과 관련 있는지 |
| stress_cov_abs | stress regime coverage 안정성 |
| vix_gt30_lower | 고위험 구간 하방 이탈 안정성 |

### Stability Gate

- seed 3개 이상
- walk-forward window 5개 이상 권장
- worst seed/fold 붕괴 없음
- validation/test coverage gap 제한
- month worst coverage 기록

### Product Gate

- inference latency 허용 범위
- payload schema 유지
- frontend 표시 계약 유지
- latest-only product contract 유지
- rollback 가능한 이전 champion 보존
- 배포 비용과 저장 크기 허용 범위 확인

## Champion/Challenger 판단

| 판단 | 조건 |
|---|---|
| 유지 | challenger가 champion보다 명확히 낫지 않음 |
| calibration만 교체 추천 | 모델 자체는 유지하되 coverage/breach가 개선됨 |
| retrain 후보 추천 | challenger가 metric/stability/product gate를 모두 통과 |
| 연구 후보 | 일부 지표는 좋지만 제품 gate 미달 |
| 탈락 | data gate, stability gate, stress gate 중 하나라도 치명적 실패 |

## Registry 기록

각 run은 다음 정보를 남긴다.

- snapshot id
- provider/source/hash
- universe id
- timeframe
- horizon
- model role
- model family
- seed
- calibration method
- metric summary
- gate result
- recommendation
- promotion decision
- rollback target

## 운영 화면에서 보고 싶은 것

관리자 화면 또는 리포트에는 다음을 보여준다.

- 현재 champion
- 최근 30일 coverage/breach 추이
- latest challenger 목록
- champion vs challenger 비교표
- gate 실패 이유
- promotion 추천 사유
- rollback 가능한 이전 모델

## Phase 2 실행 순서

1. Supabase 사용량 제한 해제 후 thin product DB를 재연결하고 egress guard를 확인한다.
2. 1D band champion 재평가 리포트를 자동화한다.
3. GARCH/Historical/Bollinger baseline을 자동 재계산한다.
4. weekly recalibration candidate를 생성한다.
5. monthly TiDE/TCN challenger를 retrain한다.
6. seed 3개와 walk-forward 5개 이상으로 검증한다.
7. recommendation report를 생성한다.
8. 사람이 승인하면 product slot promotion을 수행한다.
9. promotion 이후 monitor와 rollback 기록을 남긴다.

## 포트폴리오 관점

이 트랙은 단순 모델링보다 실무 MLOps 역량을 보여주기 좋다.

- 데이터 검증
- 실험 재현성
- 모델 registry
- champion/challenger 운영
- 자동 재학습
- drift monitoring
- promotion gate
- rollback 설계
- 비용과 배포 안정성 고려

Lens의 Phase 2에서 band MLOps는 제품 안정성과 취업 포트폴리오 양쪽 모두에 의미 있는 핵심 축으로 둔다.
