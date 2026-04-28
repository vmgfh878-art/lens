# Lens 프로젝트 전반 개선 리서치 보고서

작성일: 2026-04-27  
범위: 코드 수정 없이 모델, 데이터, 백엔드, 프론트, 백테스트, 운영, 사업성 관점에서 Phase 1 완성도와 다음 발주 후보를 정리한다.

## 한 줄 결론

Lens의 Phase 1 병목은 모델을 더 늘리는 문제가 아니라, PatchTST 예측을 데이터 정합성, run 이력, band 검증, 제품 화면, 백테스트 흐름으로 반복 가능하게 연결하는 제품-운영 루프의 얇음이다.

## 가장 큰 병목 3개

| 순위 | 병목 | 판단 | 왜 중요한가 |
|---|---|---|---|
| 1 | 제품-운영 루프 부재 | 모델 학습 결과가 예측 저장, 평가, 백테스트, 화면, 데모까지 안정적으로 이어지는 구조가 아직 얇다. | 좋은 모델 결과가 나와도 발표와 반복 실험에서 신뢰를 만들기 어렵다. |
| 2 | 데이터 정합과 실험 재현성 | 입력 NaN은 일부 finite gate로 막고 있지만, 재무 결측, hard-coded ticker 제외, cache fingerprint, run별 prediction 이력은 아직 리스크다. | 모델 성능 판단 전에 데이터가 같은지, 예측이 어느 run에서 왔는지 설명 가능해야 한다. |
| 3 | PatchTST target/denorm/calibration | PatchTST 단일 전략은 타당하지만 RevIN denorm, target 스케일, band coverage 보정이 아직 Lens 정체성의 핵심 위험이다. | 하방 보수적 line과 AI band가 틀리면 Lens의 차별점 자체가 약해진다. |

질문에 대한 직접 답: 지금은 순수 모델 성능보다 데이터 정합, 제품 루프, 운영 체계가 더 큰 병목이다. 모델 성능은 포기할 문제가 아니라, 앞의 병목을 얇게 정리한 뒤 PatchTST 안에서 target, calibration, patch geometry 순서로 열어야 한다.

## 전략 판단

Phase 1을 PatchTST 단일 모델과 백엔드/프론트 완성으로 닫는 전략은 타당하다. PatchTST는 공식 논문과 구현에서 patching, channel independence, 긴 lookback 활용, self-supervised 확장 가능성을 이미 가진 모델이다. 반대로 TiDE와 CNN-LSTM을 지금 다시 추가하면 target, split, band, UI, backtest 원인 분석 축이 동시에 늘어난다.

TiDE와 CNN-LSTM을 Phase 1.5로 미루는 판단도 맞다. TiDE는 dynamic covariate와 빠른 MLP 계열 장점이 있고, CNN-LSTM은 국소 패턴과 순차 구조를 비교 baseline으로 쓸 수 있다. 그러나 현재 Lens에는 모델 비교보다 “하나의 run이 제품에서 어떻게 믿을 수 있게 보이는가”가 먼저다.

Lens 정체성인 하방 보수적 예측선과 AI band는 코드와 화면에 들어와 있지만 아직 충분히 중심에 있지는 않다. 화면에서는 band와 conservative line이 보이기 시작했으나, 사용자가 믿을 근거인 coverage, avg_band_width, regime별 calibration, 비용 반영 백테스트, “투자 조언 아님” 표현이 같은 흐름으로 묶여야 한다.

direction, rank, IC, backtest는 보조 지표로 두는 설계가 맞다. 특히 direction accuracy는 금융 예측에서 쉽게 과대평가될 수 있으므로, Lens에서는 coverage, band width, 하방 과대예측 억제, fee-adjusted return, IC, top-k spread 순서로 해석해야 한다.

## 즉시 적용 후보 10개

| 우선순위 | 후보 | 기대 효과 | 위험 | 발주 형태 |
|---|---|---|---|---|
| 1 | run별 prediction 이력 보존 | 특정 run의 예측과 화면/백테스트를 재현 가능하게 만든다. | DB unique key 변경 필요 | `predictions` unique key에 `run_id` 포함 또는 별도 snapshot 테이블 설계 |
| 2 | model_runs 상태 확장 | 학습 진행률, 실패 원인, 취소, prune을 운영 화면에서 설명 가능하다. | 상태 전이 규칙 필요 | `running`, `failed_error`, `cancelled`, `pruned` 추가 설계 |
| 3 | progress jsonl와 로그 조회 API | Codex 터미널에서 학습 진행률을 못 보는 피로를 줄인다. | 로그 파일 경로/보안 관리 필요 | 학습 epoch JSONL 저장, read-only 조회 API |
| 4 | AI band calibration CP | Lens 정체성인 band를 “그럴듯한 선”이 아니라 검증된 리스크 지표로 만든다. | 과도한 후처리로 폭만 넓어질 수 있음 | validation residual 기반 conformal 또는 scale 보정 |
| 5 | 하방 보수 line 안정화 CP | overprediction 억제와 실제 투자 판단 설명력이 좋아진다. | 지나치게 보수적이면 수익 지표가 약화 | asymmetric loss beta sweep, validation offset |
| 6 | RevIN/target sanity CP | PatchTST 핵심 병목을 빠르게 확인한다. | 결과가 나쁘면 기존 baseline 해석이 흔들림 | RevIN denorm on/off, target scale, raw return decode 검증 |
| 7 | 백테스트 equity/drawdown/cost sweep 저장 | 발표 화면이 요약 숫자에서 실제 곡선으로 진화한다. | 저장량 증가 | 5/10/15/20/30 bps 결과와 곡선 저장 |
| 8 | 데이터 품질 대시보드 | NaN, 결측, stale cache, ticker 제외 이유를 연구 전에 확인한다. | 화면 범위 증가 | finite count, eligible/excluded ticker, source max date 표시 |
| 9 | 사용자용 문구 정리 | 신뢰와 발표 완성도를 빠르게 올린다. | 실제 UI 인코딩 확인 필요 | “AI 밴드”, “보수적 예측선”, “검증 coverage” 중심 문구 |
| 10 | 데모용 고정 시나리오와 fallback run | 발표 리스크를 줄인다. | 최신 run과 demo run 혼동 가능 | AAPL 1D 기준 고정 흐름, demo tag 또는 champion alias |

입력 NaN 해결 직후 바로 실행할 PatchTST 실험은 6번이다. 그다음 4번, 5번이 Lens 정체성을 직접 강화한다.

## 하지 말아야 할 것 10개

| 번호 | 하지 말아야 할 것 | 이유 |
|---|---|---|
| 1 | Phase 1에서 TiDE/CNN-LSTM을 다시 확장 | 원인 분석 축이 늘고 제품 루프 완성이 늦어진다. |
| 2 | direction accuracy만 보고 개선 채택 | 방향만 맞고 band와 downside가 망가질 수 있다. |
| 3 | rank target을 바로 제품 신호로 노출 | 현재 rank target 생성 경로가 미완성이고 사용자에게 의미 설명이 어렵다. |
| 4 | self-supervised pretraining을 먼저 착수 | 데이터 정합, calibration, run 이력 문제가 먼저다. |
| 5 | UI에서 raw return, target type을 전면 노출 | 일반 사용자 신뢰를 낮추고 연구 화면과 제품 화면이 섞인다. |
| 6 | 프론트에서 학습 실행 버튼부터 만들기 | Windows/CUDA/장시간 작업 실패를 사용자 화면으로 가져온다. |
| 7 | 데이터 벤더를 지금 갈아타기 | 비용과 license 이슈만 커지고 현재 병목 해결이 아니다. |
| 8 | intraday/real-time 데이터로 확장 | EOD 기반 Phase 1 검증 전에 비용과 법적 부담이 커진다. |
| 9 | calibration 없는 band를 “리스크 지표”로 강하게 주장 | coverage 검증 없이 리스크 표현을 하면 신뢰와 법적 리스크가 생긴다. |
| 10 | 성과를 투자 조언처럼 표현 | SEC가 AI 금융 표현의 과장과 오해를 강하게 문제 삼고 있다. |

## 모델 개선안

현재 PatchTST 단일 방향은 유지한다. 공식 PatchTST는 patching과 channel independence를 핵심 설계로 두며, 긴 lookback에서 이점을 주장한다. Lens의 모델 CP는 새 모델 추가가 아니라 이 설계를 Lens target과 band 목적에 맞게 맞추는 작업이어야 한다.

우선순위는 다음 순서다.

1. RevIN denorm과 raw return target의 스케일 정합 확인
2. band calibration과 conformal 후처리
3. conservative line의 asymmetric loss와 validation offset
4. `seq_len=126/252/504`, `patch_len/stride=8/4, 16/8, 24/12`
5. `ci_aggregate=target/mean/attention`
6. batch size 64/128/256 처리량과 일반화
7. d_model, n_layers는 마지막

target은 raw future return을 Phase 1 기본으로 둔다. volatility-normalized target은 학습 안정화에는 유리할 수 있지만 가격 series decode와 사용자 화면 연결이 더 어렵다. market excess return은 벤치마크 수익률 경로가 안정화된 뒤 열어야 한다. rank target은 Phase 1.5 이후 cross-sectional 전략 화면이 준비될 때 여는 편이 맞다.

self-supervised pretraining은 지금 보류가 맞다. PatchTST 공식 구현은 masked pretraining과 fine-tuning 경로를 제공하지만, Lens는 아직 데이터 정합, 실험 추적, 예측 이력, band calibration이 선행 과제다. pretraining은 “데이터가 충분하고, raw/vol target baseline이 명확하고, 동일 split에서 유의미하게 비교 가능할 때” Phase 1.5 후보로 둔다.

금융 예측에서 PatchTST가 약할 수 있는 지점은 비정상성, regime shift, 낮은 신호대잡음비, 비용 민감성, cross-sectional ranking 약화다. 방어법은 모델 추가가 아니라 coverage calibration, regime별 평가, fee-adjusted backtest, purged/embargo split 검토, 데이터 품질 게이트다.

## 데이터와 피처 개선안

데이터 정합은 Lens 전체 속도의 상위 병목이다. 현재 finite contract, fundamental imputation, `has_fundamentals`, split gate, cache fingerprint가 들어와 있는 것은 좋은 방향이다. 다만 다음 리스크가 남아 있다.

| 영역 | 현재 판단 | 개선 제안 |
|---|---|---|
| 입력 NaN | finite gate가 생겼지만 모델 실험을 이미 막은 핵심 리스크다. | 학습 전 데이터 품질 리포트를 필수 산출물로 만든다. |
| 재무 결측 | 결측값 0과 `has_fundamentals` 플래그는 현실적인 선택이다. | “0이 실제 0인지 결측 대체인지”를 UI/메타에서 분리해 추적한다. |
| hard-coded 제외 ticker | `FUNDAMENTAL_INSUFFICIENT_TICKERS`는 빠른 방어지만 확장성이 낮다. | 동적 gate와 제외 사유 리포트로 전환한다. |
| stale cache | data fingerprint가 있지만 값 수정까지 완전히 잡는지는 추가 확인 필요다. | source max date, row count, feature schema, 핵심 통계 hash를 함께 저장한다. |
| 1D/1W/1M | 제품에서는 자연스럽다. 1M 가격 전용도 맞다. | 연구에서는 1D/1W만 AI 학습, 1M은 화면 전용으로 명확히 분리한다. |
| universe | ticker별 sufficiency gate는 필요하다. | 발표용 universe와 연구용 universe를 분리하고 제외 사유를 저장한다. |
| leakage | 현재 split에는 horizon gap이 있다. | 금융 관행에 맞춰 purged/embargo 개념을 문서화하고 split report에 gap을 표시한다. |

데이터 벤더는 Phase 1에서 바꾸지 않는 편이 낫다. 2026-04-27 기준 확인한 공개 가격은 EODHD가 EOD/글로벌/펀더멘털을 낮은 월 비용으로 제공하고, FMP는 미국/펀더멘털/비율 endpoint가 편하지만 plan별 범위 차이가 있다. Alpha Vantage는 무료 호출 제한이 낮아 대량 학습용보다는 보조 확인용이다. 실제 상용화 전에는 표시권, 재배포권, 실시간/지연 데이터 entitlement를 별도로 확인해야 한다.

## 백엔드/API 개선안

read-only API 중심 전략은 Phase 1에 적절하다. 학습 실행 API보다 조회 API, 상태 API, 재현 가능한 run API가 먼저다.

가장 큰 구조 리스크는 `predictions`의 unique key가 `run_id`를 포함하지 않는 점이다. 현재 구조에서는 같은 ticker, model, timeframe, horizon, asof_date 조합에서 run별 prediction 이력을 보존하기 어렵다. 발표와 연구 추적을 위해서는 “이 차트가 어느 run에서 왔는가”가 재현되어야 한다.

`model_runs`, `prediction_evaluations`, `backtest_results`의 방향은 맞다. 다만 Phase 1 데모와 운영에는 다음 보강이 필요하다.

| 항목 | 개선 제안 |
|---|---|
| `model_runs.status` | `running`, `completed`, `failed_nan`, `failed_error`, `cancelled`, `pruned`로 확장 |
| 진행률 | epoch별 `progress.jsonl`과 마지막 heartbeat 저장 |
| prediction 이력 | run별 snapshot 보존, latest 조회는 별도 view 또는 query로 처리 |
| backtest 결과 | summary 외에 equity curve, drawdown series, cost sweep meta 저장 |
| 페이지네이션 | `/ai/runs`의 `total=len(rows)`는 전체 total이 아니므로 count 정책 정리 |
| 캐싱 | prediction/latest는 public cache 1시간이 제품 데모에서 stale result를 만들 수 있어 ETag 또는 짧은 TTL 검토 |
| 에러 처리 | 사용자용 메시지와 연구자용 details를 분리 |
| legacy API | `/prices`, `/predict`와 `/api/v1`의 역할을 문서화하거나 숨김 처리 |

## 프론트/제품 경험 개선안

첫 화면을 주식 보기로 둔 방향은 맞다. Lens는 연구 콘솔이 아니라 사용자가 먼저 “가격과 AI 리스크 범위”를 보는 제품이어야 한다.

TradingView식 조작력과 토스증권식 간결함을 섞는 방식은 다음처럼 구체화하면 좋다.

| 화면 요소 | 방향 |
|---|---|
| 메인 차트 | candlestick/line 전환, zoom/pan, hover tooltip, forecast 구간 음영 |
| AI band | 상단/하단 선만이 아니라 가능하면 옅은 band fill로 리스크 범위를 직관화 |
| conservative line | 검은색 또는 절제된 색으로 “기대 수익선”이 아니라 “보수적 기준선”으로 표현 |
| 신뢰 카드 | coverage, avg_band_width, 최근 run, 검증 기간을 작은 카드로 표시 |
| 1M | AI가 없는 실패 상태가 아니라 “월봉은 가격 전용” 상태로 자연스럽게 표현 |
| 연구 용어 | target type, raw return, ci_aggregate는 모델 학습 화면에만 둔다 |
| 모바일 | 차트, AI 요약, ticker 검색 순서로 쌓고 metric grid는 줄 수를 줄인다 |

현재 로컬에서 프론트 문자열이 깨진 형태로 읽히는 파일이 있었다. 실제 브라우저에서도 깨지면 발표 신뢰도를 즉시 해치는 문제이므로, UI 문구 QA는 모델 성능보다 먼저 잡아야 한다. 이것이 단순 콘솔 인코딩 문제인지 실제 소스 문자열 문제인지는 확인이 필요하다.

주식 보기, 백테스트, 모델 학습 3화면 분리는 맞다. 주식 보기는 사용자 화면, 백테스트는 설득 화면, 모델 학습은 read-only 연구 콘솔이어야 한다. Phase 1에서는 모델 학습 화면을 실행 콘솔로 만들지 말고, run 이력과 실패 원인을 읽는 화면으로 둔다.

## 백테스트와 투자 지표 개선안

현재 지표인 fee-adjusted return, sharpe, turnover, coverage, avg_band_width, IC, top-k spread는 Phase 1 기준 충분하다. 부족한 것은 지표 종류가 아니라 저장과 시각화다.

우선 보강할 것은 다음이다.

1. 5/10/15/20/30 bps cost sweep 저장
2. equity curve와 drawdown series 저장
3. long-only baseline과 long-short research 지표 분리
4. band width와 realized volatility/downside의 관계 검증
5. coverage를 전체 평균이 아니라 regime별, ticker별, horizon별로 분해
6. turnover와 fee drag를 성과 카드에서 분리 표시

전략 확장 순서는 long-only, long-short, risk-off, band 기반 position sizing 순서가 좋다. Phase 1 발표에서는 long-only와 fee-adjusted 결과가 가장 안전하다. long-short는 연구자에게 설득력은 있지만 일반 발표에서는 공매도, 차입, 비용 설명이 늘어난다.

하방 보수적 line은 투자 판단에서 “매수 가격 목표”가 아니라 “과대 낙관을 제한하는 기준선”으로 써야 한다. 예를 들어 현재가가 하단 band와 conservative line 대비 어디에 있는지, 예측선이 위로 좋아 보여도 하방 band가 넓으면 risk-off로 해석하는 식이다.

## 운영/배포 개선안

지금처럼 사용자가 GPU에서 직접 학습하는 구조는 빠른 연구에는 좋지만, 반복 가능성과 관찰 가능성이 낮다. 장시간 학습 중 터미널이 끊기거나 Codex에서 진행률을 보기 어려우면 CP 속도가 급격히 떨어진다.

우선순위는 작업 큐보다 progress jsonl이다.

1. 학습 프로세스가 epoch별 JSONL을 남긴다.
2. `model_runs`에 `running`과 heartbeat를 저장한다.
3. read-only 로그 API와 학습 화면 tail view를 만든다.
4. 그다음에 cancel API와 queue를 검토한다.

W&B와 Optuna는 이미 방향이 좋다. 다만 지금은 W&B를 “필수 운영 시스템”으로 키우기보다, dataset fingerprint, config hash, checkpoint path, best metrics, failure meta를 Lens DB에서 먼저 읽을 수 있어야 한다. W&B Artifacts나 MLflow Model Registry 같은 체계는 Phase 1.5 이후 팀/배포가 커질 때 도입해도 된다.

Windows + CUDA + PyTorch 환경 리스크는 계속 남는다. DataLoader worker, torch compile, AMP dtype, CUDA cleanup, PyTorch/CUDA wheel 버전이 흔들릴 수 있다. 따라서 발표용으로는 학습을 실시간으로 돌리는 대신, 검증된 checkpoint와 demo run을 준비하고 학습 화면은 진행률/이력 콘솔로 보여주는 편이 안전하다.

## 사업성/수익화 개선안

Lens의 차별점은 “주가를 맞히는 앱”이 아니다. “AI가 보는 위험 범위와 보수적 기준선을 통해 과도한 낙관을 줄이는 투자 리스크 렌즈”가 더 설득력 있다.

사용자 가치 표현은 다음이 좋다.

- AI band: 다음 구간에서 가격이 흔들릴 수 있는 위험 범위
- 보수적 예측선: 낙관적 평균선이 아니라 하방을 더 조심스럽게 본 기준선
- coverage: 과거 검증에서 실제 값이 band 안에 들어온 비율
- fee-adjusted backtest: 수수료를 빼고도 신호가 남는지 보는 검증

무료/유료 기능 분리는 다음처럼 자연스럽다.

| 구분 | 기능 |
|---|---|
| 무료 | 지연 EOD 가격, 기본 차트, 최신 AI band, 제한된 ticker, 기본 백테스트 요약 |
| 유료 개인 | 관심종목, band 이탈 알림, 더 넓은 universe, regime별 리스크, 상세 백테스트 |
| 유료 연구자 | run 비교, CSV export, cost sweep, calibration report, target/전략 실험 리포트 |
| B2B 후보 | API 제공, white-label risk overlay, 내부 리서치 대시보드 |

공개 배포 시에는 투자 조언이 아니라 정보 제공/연구 도구임을 명확히 해야 한다. AI를 쓴다는 표현도 실제 검증 지표와 한계를 함께 제시해야 한다.

## 발표 데모 전략

가장 설득력 있는 흐름은 `주식 보기 -> AI band 해석 -> 백테스트 검증 -> 모델 학습 콘솔`이다.

1. 주식 보기에서 AAPL 1D를 연다.
2. 가격 차트 위에 AI band와 보수적 예측선을 보여준다.
3. “이 선은 수익 보장이 아니라 검증된 위험 범위와 보수적 기준”이라고 설명한다.
4. coverage와 avg_band_width 카드를 보여준다.
5. 백테스트 화면으로 이동해 fee-adjusted return, sharpe, turnover, drawdown을 보여준다.
6. 모델 학습 화면으로 이동해 latest completed run, config, failed_nan 이력을 보여준다.
7. 마지막에 “Phase 1은 모델 수가 아니라 제품 루프 완성에 집중했고, Phase 1.5에서 TiDE/CNN-LSTM을 같은 루프에 꽂을 수 있다”고 닫는다.

발표용으로 과장 없이 보여줄 지표는 coverage, avg_band_width, fee-adjusted return, fee-adjusted sharpe, turnover, IC, top-k spread다. 수익률 하나만 전면에 두면 투자 조언처럼 보이고, direction accuracy만 보여주면 연구 설득력이 약하다.

## 2주 실행 로드맵

### 1주차: 신뢰 루프 고정

| 날짜 | 목표 | 산출물 |
|---|---|---|
| 1일차 | prediction 이력과 run status 설계 | DB/API 변경 CP 지시서 |
| 2일차 | progress jsonl와 로그 API 설계 | 학습 진행률 CP 지시서 |
| 3일차 | RevIN/target sanity 실험 준비 | 입력 NaN 해결 후 실행할 PatchTST 실험 명령 |
| 4일차 | UI 문구와 band 표현 QA | 주식 보기 trust copy, 1M 가격 전용 문구 |
| 5일차 | 백테스트 곡선/비용 sweep 설계 | equity/drawdown/cost sweep 저장안 |

### 2주차: 발표 가능한 검증 패키지

| 날짜 | 목표 | 산출물 |
|---|---|---|
| 6일차 | band calibration CP | validation coverage 보정 리포트 |
| 7일차 | conservative line CP | overprediction, mean_signed_error, fee-adjusted 변화 비교 |
| 8일차 | patch geometry 최소 sweep | 3개 조합 결과표 |
| 9일차 | 데모 run 고정 | AAPL 1D 기준 화면 흐름과 fallback run |
| 10일차 | 발표 리허설 문서화 | 데모 스크립트, 한계, 다음 단계 |

이 순서는 모델을 포기하는 순서가 아니다. 모델을 제품과 검증 루프에 묶어서, 다음 모델을 추가해도 비교 가능한 구조를 먼저 만드는 순서다.

## 확인이 더 필요한 질문

- 실제 브라우저에서 프론트 한글 문구가 깨지는가, 아니면 터미널 출력 인코딩 문제인가?
- 운영 DB의 `predictions` unique key와 실제 upsert 정책이 run별 이력을 덮어쓰고 있는가?
- 현재 cache fingerprint가 max date/count 외의 값 변경까지 감지하는가?
- EODHD/FMP 사용 조건이 발표, 공개 배포, 상용 배포에서 각각 어디까지 허용되는가?
- 실제 demo run으로 쓸 수 있는 completed PatchTST run이 있으며, prediction/evaluation/backtest가 모두 연결되어 있는가?
- 1W 예측은 발표에 포함할 만큼 데이터와 화면 상태가 안정적인가?
- PyTorch/CUDA 버전과 GPU 환경은 재설치 없이 반복 학습 가능한 상태인가?
- `failed_nan` 외 실패 원인 중 가장 잦은 것은 무엇인가?
- band coverage 목표는 80% 구간으로 둘 것인가, 90% 구간으로 둘 것인가?
- 일반 사용자용 signal은 BUY/SELL/HOLD를 유지할 것인가, 아니면 risk 상태 중심으로 바꿀 것인가?

## 참고한 자료 또는 근거

### 로컬 코드/문서 근거

- `ai/models/patchtst.py`: PatchTST 단일 backbone, RevIN, channel independence, `ci_aggregate`, `ci_target_fast` 확인.
- `ai/models/revin.py`: target channel 기준 denorm 구조 확인.
- `ai/loss.py`: asymmetric Huber, quantile pinball, band cross penalty 확인.
- `ai/evaluation.py`: coverage, avg_band_width, IC, top-k spread, fee-adjusted 지표 확인.
- `ai/preprocessing.py`: finite contract, cache fingerprint, split 준비 경로 확인.
- `ai/splits.py`: horizon gap과 sufficiency gate 확인.
- `ai/targets.py`: raw, volatility-normalized, market excess, rank target 상태 확인.
- `backend/db/schema.sql`: `model_runs`, `predictions`, `prediction_evaluations`, `backtest_results`, `job_runs`, `sync_state` 구조 확인.
- `backend/app/routers/v1/ai.py`, `backend/app/services/api_service.py`: read-only AI API와 prediction 조회 방식 확인.
- `frontend/src/components/StockView.tsx`, `BacktestView.tsx`, `TrainingView.tsx`, `Chart.tsx`: 3화면 구성과 chart overlay 확인.
- `docs/cp13_patchtst_solo_plan.md`, `docs/cp16_research_patchtst_improvement_intel.md`, `docs/cp_product_demo_plan.md`: PatchTST solo, product loop, overlay 방향 확인.

### 외부 근거

- PatchTST 공식 구현: https://github.com/yuqinie98/PatchTST  
  patching, channel independence, supervised/self-supervised 경로, 긴 lookback 효율 근거.
- PatchTST 논문 요약: https://huggingface.co/papers/2211.14730  
  multivariate forecasting, self-supervised pretraining, 긴 lookback 장점 근거.
- TiDE 공식 Google Research: https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/  
  TiDE는 covariate와 빠른 MLP 계열 장점이 있으나 Phase 1.5 비교 후보로 충분하다.
- Conformalized Quantile Regression: https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html  
  quantile band를 coverage 보장 방향으로 보정할 근거.
- Adaptive Conformal Inference: https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html  
  distribution shift와 regime 변화에서 adaptive coverage를 고려할 근거.
- mlfinlab purged/embargo cross validation: https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html  
  금융 시계열 split 누수 방지와 embargo 개념 근거.
- W&B Artifacts: https://docs.wandb.ai/models/artifacts  
  dataset, model checkpoint, run artifact versioning 근거.
- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry  
  model lineage, versioning, alias, tags, governance 근거.
- Optuna Artifacts: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html  
  trial별 큰 산출물 저장, SQLite와 artifact 분리 근거.
- TradingView Lightweight Charts: https://www.tradingview.com/lightweight-charts/  
  lightweight financial chart, markers, indicators, mobile-ready chart 방향 근거.
- EODHD 가격: https://eodhd.com/pricing  
  2026-04-27 확인 기준 EOD/펀더멘털/글로벌 데이터 비용 판단 근거.
- FMP 가격: https://site.financialmodelingprep.com/pricing-plans  
  free/basic, starter, premium, ultimate의 호출량과 데이터 범위 판단 근거.
- Alpha Vantage premium: https://www.alphavantage.co/premium/  
  무료 호출 제한과 premium 호출량 판단 근거.
- SEC AI washing 제재: https://www.sec.gov/newsroom/press-releases/2024-36  
  AI 금융 표현을 과장하지 않아야 하는 근거.
- SEC Investor Bulletin Robo-Advisers: https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-45  
  자동화 투자 도구의 위험, 비용, 사용자 적합성 설명 필요 근거.
