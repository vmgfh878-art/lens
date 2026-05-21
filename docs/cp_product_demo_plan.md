# Lens Product Demo Track 계획

## 1. 목적

모델 연구 CP와 백엔드/프론트 CP를 분리한다. PatchTST 학습은 시간이 오래 걸리고 실패 가능성이 높으므로, 그 사이에 데모 화면과 조회 API를 병렬로 완성한다.

핵심 목표는 다음이다.

- 일반 사용자가 주식 가격과 AI 밴드를 자연스럽게 볼 수 있는 완성형 화면을 만든다.
- 예측선, 밴드, 평가 지표, 백테스트, run 상태를 한 화면에서 연결한다.
- 모델 성능 개선이 늦어져도 발표 가능한 제품 루프를 먼저 닫는다.

첫 화면은 연구자용 실험판이 아니라 주식 앱처럼 보여야 한다. 학습 로그와 백테스트는 별도 작업 공간으로 분리한다.

## 2. 트랙 분리 원칙

| 트랙 | 담당 범위 | 금지 |
|---|---|---|
| Research CP | PatchTST target, patch geometry, seq_len, ci_aggregate, loss/eval 개선 | 프론트 UI 수정과 API 대공사 동시 진행 금지 |
| Product CP | API 조회, run/evaluation/backtest 노출, 프론트 리서치 콘솔 | 모델 구조, target/loss 변경 금지 |

Research CP와 Product CP는 같은 주차에 병렬로 갈 수 있지만, 한 CP 안에서 섞지 않는다.

## 3. 현재 프론트 진단

현재 `frontend/src/components/DashboardPage.tsx`는 가격 차트, 티커 검색, 예측 요약을 이미 갖고 있다. 다만 Phase 1 데모 목표에는 부족하다.

| 항목 | 현재 | 판단 |
|---|---|---|
| 화면 성격 | hero 중심 대시보드 | 완성형 주식 화면으로 전환 필요 |
| 모델 선택 | PatchTST/CNN-LSTM/TiDE 선택 가능 | 첫 화면에서는 모델 선택을 숨기고 기본 모델만 사용. 학습 화면에서는 추후 다중 모델 지원 |
| 차트 | 가격 캔들만 표시 | 예측선과 band overlay 필요 |
| 평가 지표 | 없음 | coverage, band width, IC, fee-adjusted 지표 필요 |
| run 상태 | 없음 | completed/failed_nan, target type, config 표시 필요 |
| 백테스트 | 없음 | return, sharpe, turnover, fee bps 표시 필요 |
| 파라미터 조정 | 없음 | 별도 학습 화면에서 read-only config부터 시작 |

결론: 현재 화면은 버리지 않는다. 다만 첫 화면은 소개형 hero나 연구자 콘솔이 아니라 **주식 가격 화면**으로 재구성한다. 학습과 백테스트는 왼쪽 내비게이션의 별도 화면에서 다룬다.

## 4. 앱 정보 구조

왼쪽 내비게이션을 기준으로 3개 주요 작업 공간을 둔다.

| 화면 | 목적 | 주요 사용자 |
|---|---|---|
| 주식 보기 | 가격, 보조지표, AI 예측선/밴드 확인 | 일반 사용자, 발표 데모 |
| 백테스트 | 전략 성과와 수수료 반영 결과 확인 | 사용자, 연구자 |
| 모델 학습 | run, config, W&B/Optuna 결과, checkpoint 상태 확인 | 연구자, 개발자 |

Phase 1 첫 화면은 **주식 보기**다.

## 5. 주식 보기 화면

TradingView의 차트 조작력과 토스증권식 간결한 정보 구조를 섞는다. 정확한 클론이 아니라 장점을 흡수한다.

### 5.1 상단 헤더

목적: 조회 조건을 빠르게 바꾸는 영역.

포함:

- 티커 검색
- timeframe: 1D / 1W / 1M
- 현재가, 등락률, 거래량
- 관심 종목 추가 자리
- AI 상태 배지: 사용 가능 / 가격만 / 결과 없음
- 조회 버튼

주의:

- 첫 화면에서 모델명과 target type을 전면 노출하지 않는다.
- 1M은 AI 밴드가 없어도 가격 화면으로는 완성도 있게 표시한다.
- 모델명, target type, run_id는 상세 패널이나 모델 학습 화면에서 확인한다.

### 5.2 메인 차트

목적: Lens의 차별점인 보수적 예측선과 AI band를 시각적으로 보여준다.

포함:

- 캔들 / 라인 차트 전환
- conservative line overlay
- lower/upper band overlay
- forecast 구간 배경 구분
- signal marker는 보조로만 표시
- 1M에서는 가격 차트만 표시하고 AI layer는 비활성 안내

우선순위:

1. 가격 + line + band overlay
2. forecast 구간 강조
3. hover tooltip
4. horizon별 band table
5. 보조지표 layer toggle

### 5.3 보조지표와 layer toggle

기본은 가격만 보여주고, 필요한 지표를 사용자가 켜는 방식으로 간다.

초기 후보:

- 거래량
- 이동평균선
- RSI
- MACD
- Bollinger Band
- Lens AI band
- Lens conservative line

초기 구현에서는 거래량, 이동평균선, RSI부터 우선한다. MACD와 Bollinger Band는 후속으로 둔다.

### 5.4 AI 요약 카드

첫 화면에서 보여줄 AI 정보는 단순해야 한다.

- AI band 상태
- 보수적 예측선 마지막 값
- 하단/상단 밴드 마지막 값
- coverage
- avg_band_width
- fee-adjusted return 요약

target type은 일반 사용자에게 바로 노출하지 않는다. 다만 상세 정보 펼치기에서 `raw_future_return`, `volatility_normalized_return` 같은 학습 target을 확인할 수 있게 한다.

### 5.5 target type 표시의 의미

target type은 "사용자가 사고파는 신호 종류"가 아니라, 모델이 무엇을 맞추도록 학습됐는지를 뜻한다.

| target type | 의미 | 사용자 화면 노출 |
|---|---|---|
| `raw_future_return` | 실제 미래 수익률을 직접 예측 | 상세 정보에서만 표시 |
| `volatility_normalized_return` | 변동성으로 나눈 미래 수익률 점수 예측 | 상세 정보에서만 표시. 가격 시그널 생성은 제한 |
| `market_excess_return` | 시장 대비 초과수익률 예측 | 아직 미구현. 미래 후보 |

첫 화면에는 "AI 밴드", "보수적 예측선", "가격만 제공"처럼 사용자 언어를 쓴다. target type은 모델 학습 화면과 run 상세에서 주로 표시한다.

## 6. 백테스트 화면

백테스트는 주식 보기 화면의 작은 패널이 아니라 별도 화면으로 둔다.

목적:

- 모델 또는 전략별 성과 확인
- 수수료 반영 후 남는지 확인
- 발표에서 "예측이 실제 전략에 어떤 의미가 있는지" 보여주기

포함:

- 전략 선택
- timeframe
- fee bps
- return / sharpe / mdd / win_rate
- fee-adjusted vs gross 비교
- turnover
- trade count
- equity curve
- drawdown chart

초기 구현은 read-only 조회다. 화면에서 백테스트 실행 버튼은 만들지 않는다.

## 7. 모델 학습 화면

모델 학습 화면은 W&B를 대체하지 않고, 프로젝트 내부의 요약 콘솔 역할을 한다.

포함:

- run 목록
- run status: completed / failed_nan
- model name
- target type
- key metrics
- checkpoint path
- config summary
- W&B link 자리
- Optuna study summary 자리

Phase 1에서는 PatchTST만 강조하지만, 화면 구조는 TiDE/CNN-LSTM 추가를 전제로 만든다.

## 8. 백엔드 API 계획

### 8.1 현재 있는 API

| API | 상태 |
|---|---|
| `GET /api/v1/stocks` | 사용 가능 |
| `GET /api/v1/stocks/{ticker}/prices` | 사용 가능 |
| `GET /api/v1/stocks/{ticker}/predictions/latest` | 사용 가능 |

### 8.2 추가할 API

| API | 목적 |
|---|---|
| `GET /api/v1/ai/runs` | model_runs 목록과 status 조회 |
| `GET /api/v1/ai/runs/{run_id}` | run config, best/test metrics 조회 |
| `GET /api/v1/ai/runs/{run_id}/evaluations` | prediction_evaluations 조회 |
| `GET /api/v1/ai/runs/{run_id}/backtests` | backtest_results 조회 |
| `GET /api/v1/stocks/{ticker}/predictions/latest?run_id=` | 특정 run 기준 예측 조회 옵션 |

초기 Product CP에서는 read-only 조회 API만 만든다. 학습 실행 API는 만들지 않는다.

## 9. 핵심 지표

포함:

- coverage
- avg_band_width
- band_loss
- mae
- smape
- spearman_ic
- top_k_long_spread
- long_short_spread
- fee_adjusted_return
- fee_adjusted_sharpe
- fee_adjusted_turnover

표시 원칙:

- direction_accuracy는 보조 위치에 둔다.
- coverage와 band_width를 line/IC보다 위에 둔다.
- fee-adjusted 지표는 별도 묶음으로 둔다.

## 10. run 상태와 파라미터

목적: 실험 신뢰성을 보여준다.

포함:

- run_id
- status: completed / failed_nan
- model_ver
- created_at
- target type
- checkpoint_path 존재 여부
- best_epoch
- best_val_total
- config hash

failed_nan run은 빨간 오류 화면이 아니라, 별도 상태로 명확히 표시한다.

포함:

- seq_len
- patch_len
- stride
- d_model
- n_heads
- n_layers
- ci_aggregate
- dropout
- lr
- weight_decay
- line_target_type
- band_target_type

나중에만 editable로 전환한다. Phase 1에서는 화면에서 학습을 직접 실행하지 않는다.

## 11. CP 분리안

| CP | 트랙 | 목표 |
|---|---|---|
| CP14-R | Research | PatchTST target baseline 2종 비교 |
| CP14-P | Product | AI run/evaluation/backtest 조회 API 설계 및 최소 구현 |
| CP15-R | Research | PatchTST patch_len/stride 실험 |
| CP15-P | Product | 주식 보기 화면 레이아웃 전환 |
| CP16-R | Research | seq_len, ci_aggregate 실험 |
| CP16-P | Product | 차트에 line/band overlay |
| CP17-P | Product | 백테스트/평가 지표 패널 연결 |
| CP18-P | Product | 발표용 데모 플로우 고정 |

## 12. 프론트 구현 우선순위

1. 왼쪽 내비게이션: 주식 보기 / 백테스트 / 모델 학습.
2. 주식 보기 첫 화면 재구성.
3. 캔들/라인 차트 전환.
4. 1D/1W/1M 가격 표시.
5. 가격 차트에 Lens line/band overlay.
6. 보조지표 layer toggle.
7. 백테스트 별도 화면.
8. 모델 학습 별도 화면.
9. 모바일에서는 가격 → AI 요약 → 지표 → 세부 정보 순서로 세로 배치.

## 13. 데모 성공 기준

- AAPL 1D 기준으로 가격, line, band가 한 화면에 보인다.
- 1M은 AI가 없어도 가격 화면으로 자연스럽게 보인다.
- 캔들/라인 차트 전환이 가능하다.
- run status가 completed인지 확인 가능하다.
- coverage와 avg_band_width가 바로 보인다.
- fee-adjusted return과 sharpe가 바로 보인다.
- failed_nan run은 예측/백테스트에 섞이지 않고 상태로만 보인다.
- 화면에서 TiDE/CNN-LSTM은 Phase 1.5로 숨겨져 있다.

## 14. 다음 오더 후보

### CP14-R

PatchTST target baseline 2종 비교.

- raw_future_return
- volatility_normalized_return

### CP14-P

Product API 최소 조회판.

- model_runs 목록
- run 상세
- evaluation 요약
- backtest 요약

두 CP는 병렬 진행 가능하다. 단, 한 구현 에이전트가 둘을 동시에 맡으면 안 된다.

## 15. CP14-P 검수 후 API 제한

CP14-P에서 read-only API 최소판은 추가됐다. 다만 Product 화면 설계 시 다음 제한을 반영한다.

- aggregate 성격의 `spearman_ic`, `top_k_*`, `fee_adjusted_*`는 `/runs/{run_id}`의 `val_metrics` 또는 `test_metrics`에서 우선 읽는다.
- `/runs/{run_id}/evaluations`는 ticker/asof 단위 평가 테이블로 본다.
- `predictions/latest?run_id=...`는 현재 DB unique key 구조상 모든 과거 run prediction 이력을 보장하지 않는다.
- 첫 제품 화면에서는 latest completed run 기준 표시를 기본으로 둔다.
- 특정 run별 prediction 비교는 DB 저장 정책을 정리한 뒤 연다.

## 16. CP15-P closure와 CP16-P 목표

CP15-P 결과:

- 첫 화면을 주식 보기 화면으로 전환했다.
- 왼쪽 내비게이션으로 `주식 보기`, `백테스트`, `모델 학습`을 분리했다.
- 1M은 가격 전용으로 처리하고 AI layer를 비활성화했다.
- 모델 선택과 target type은 주식 보기 화면에서 숨겼다.
- 차트 overlay는 props 자리만 열렸고 실제 렌더링은 아직 없다.

CP16-P 목표:

- AI 밴드와 보수적 예측선을 실제 차트 overlay로 렌더링한다.
- 사용자 화면 문구를 한국어 중심으로 정리한다.
- 모델 학습 화면은 리서치 콘솔이므로 상태값 원문을 괄호 병기할 수 있다.
- overlay가 없거나 1M인 경우에도 가격 차트가 정상 동작해야 한다.
- 주식 보기 화면에 coverage, avg_band_width, 보수적 예측선 상태를 작은 요약으로 붙인다.

## 17. CP16-P 결과와 다음 제품 과제

CP16-P 결과:

- 주식 보기에서 최신 완료 PatchTST run을 먼저 선택하고 `run_id` 기준 prediction을 조회한다.
- 차트에 `line_series` 기반 보수적 예측선과 `upper_band_series`/`lower_band_series` 기반 AI 밴드 상하단선을 표시한다.
- `forecast_dates`와 series 길이가 맞지 않거나 값이 유효하지 않으면 overlay를 생략한다.
- 1M은 가격 전용으로 유지한다.
- 커버리지, 평균 밴드 폭, 보수적 예측선 요약 카드를 추가했다.
- band fill은 보류하고 상하단 점선으로 마감했다.

다음 제품 과제:

1. 백테스트 화면에서 latest completed run 조회 시 선택 timeframe을 같이 넘긴다.
2. 백테스트 결과 화면에 fee-adjusted 지표와 비용 정보를 더 명확히 표시한다.
3. 모델 학습 화면은 원문 상태값 병기 정책에 맞춰 라벨을 정리한다.
4. 차트 overlay legend와 layer toggle의 모바일 배치를 점검한다.
5. band fill은 안정 구현 방식을 찾은 뒤 별도 CP에서 연다.

## 18. CP17-P 결과와 다음 제품 과제

CP17-P 결과:

- 백테스트 화면은 선택한 차트 단위 기준으로 latest completed PatchTST run을 고른다.
- 백테스트 조회도 같은 timeframe으로 이어진다.
- 수수료 반영 수익률/샤프와 수수료 전 수익률/샤프를 분리 표시한다.
- 백테스트 화면에 `val_metrics`/`test_metrics` 품질 패널을 추가했다.
- 모델 학습 화면에도 `val_metrics`/`test_metrics` 품질 패널을 추가했다.
- 상태 라벨은 `완료(completed)`, `NaN 실패(failed_nan)` 방식으로 정리했다.
- 주식 보기 화면의 단순함은 유지했다.

다음 제품 과제:

1. 백테스트 equity curve와 drawdown series 저장 구조가 확정되면 실제 차트로 연결한다.
2. metric별 단위 표시 규칙을 정리한다.
3. 모델 학습 화면의 config/run 라벨을 발표용으로 더 다듬는다.
4. dev 서버 화면을 브라우저에서 직접 보며 모바일/데스크톱 시각 QA를 한다.
5. 발표 데모 동선을 `주식 보기 → 백테스트 → 모델 학습` 순서로 고정한다.

## 19. CP18-P 결과와 다음 제품 과제

CP18-P 결과:

- `scripts/start_demo.ps1`을 추가해 백엔드/프론트 개발 서버를 빠르게 실행할 수 있게 했다.
- 터미널 2개로 직접 실행하는 수동 절차도 문서화했다.
- 발표 동선을 `주식 보기 → 1M 가격 전용 → 백테스트 → 모델 학습` 순서로 정리했다.
- 백엔드 연결 실패 문구와 metric card/legend/mobile spacing polish를 반영했다.
- `http://127.0.0.1:8000/api/v1/health/live`와 `http://127.0.0.1:3000`의 200 응답을 확인했다.
- 현재 로컬 DB의 AAPL prediction/backtest/evaluation 부재를 데모 리스크로 명시했다.

다음 제품 과제:

1. 실제 AI overlay가 보이는 completed PatchTST run/ticker를 확보한다.
2. 해당 run의 prediction, evaluation, backtest 저장 여부를 readiness check로 확인한다.
3. fake data 없이 inference/backtest 산출물을 생성하는 절차를 문서화한다.
4. 사용자의 일반 브라우저에서 데스크톱/모바일 시각 QA를 수행한다.
5. 발표용 스크린샷 또는 짧은 녹화 동선을 고정한다.

CP18 추가 안정화:

- `scripts/check_demo_readiness.ps1`을 추가했다.
- stock search 503은 전체 오류가 아니라 직접 티커 입력 fallback으로 처리한다.
- prediction row 없음과 AI prediction API 오류를 구분한다.
- prediction row가 없어도 가격 차트는 유지한다.
- readiness 결과상 현재 부족 산출물은 prediction, backtest, evaluation이다.

다음 단계는 제품 기능 추가가 아니라 실제 AI 산출물 확보가 우선이다.

## 20. CP19 결과와 다음 제품 과제

CP19 결과:

- fake data 없이 실제 AI overlay 데모 산출물을 확보했다.
- demo run은 `patchtst-1D-fc096a026a1e`, demo ticker는 `AAPL`이다.
- prediction row가 존재하고 `forecast_dates`, `line_series`, `upper_band_series`, `lower_band_series`가 모두 길이 5로 일치한다.
- evaluation row가 존재한다.
- backtest row는 실제 `ai.backtest --save` 경로로 생성했다.
- 주식 보기 화면은 최신 completed run에 prediction이 없으면 최근 completed run들을 순서대로 확인해 실제 prediction이 있는 run을 사용한다.

현재 데모 동선:

1. `AAPL` 1D 주식 보기에서 가격 차트와 AI 밴드/보수적 예측선 overlay 확인.
2. 1M으로 전환해 가격 전용 처리 확인.
3. 백테스트 화면에서 demo run의 수수료 반영 지표 확인.
4. 모델 학습 화면에서 completed run과 val/test 품질 지표 확인.

남은 제품 과제:

1. `stock_info` 기반 티커 검색 503 원인을 별도 확인한다.
2. 최신 completed run과 사용 가능한 prediction run이 다를 때 화면에 어떤 수준으로 안내할지 결정한다.
3. 발표용 시각 QA와 스크린샷/녹화 동선을 고정한다.
4. checkpoint 호환성 문제는 모델 구조 변경 없이 다룰 수 있는지 별도 Research/Artifact CP에서 판단한다.
