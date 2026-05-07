# Lens Phase 2: 연구용 Lens와 배포용 Lens Signal 분리 계획

작성일: 2026-05-07

이 문서는 Phase 1 모델 실험 이후 진행할 제품 분리와 사용자 경험 검증 계획이다. 실험 시간 계산, 개별 CP 성과, 모델 sweep 계획은 이 문서에 섞지 않는다.

## 1. Phase 2 한 줄 목적

Phase 2의 목적은 연구용 Lens를 계속 실험실로 유지하면서, 외부 사용자에게 보여줄 Lens Signal을 별도 제품으로 분리하고, 클릭 데이터와 사용자 피드백으로 앱의 목적성과 사용성을 검증하는 것이다.

핵심 질문은 다음 하나다.

> 사용자는 Lens Signal을 보고 오늘 이 종목을 더 볼지, 위험을 점검할지, 넘길지 판단할 수 있는가?

## 2. Phase 2 진입 조건

Phase 2는 모델을 완전히 끝냈다는 뜻이 아니다. 다만 사용성 검증을 시작할 수 있을 만큼 제품 후보가 닫혔다는 뜻이다.

진입 조건:

- 1D 보수적 예측선 제품 후보가 하나 이상 있다.
- 1D AI 밴드 제품 후보가 하나 이상 있다.
- 1W 보수적 예측선이 제품 후보 또는 준비 중 상태로 연결 가능하다.
- 1W AI 밴드는 후보가 약하면 준비 중으로 닫고 더 끌지 않는다.
- 1M은 가격 전용 또는 미지원으로 명확히 둔다.
- 모델 run, source/provider/hash, asof_date가 제품 화면에 전달 가능하다.
- Supabase는 원천 가격/지표 저장소가 아니라 thin 제품 DB로 유지한다.

1W 실험이 애매하면 Phase 2를 미루기보다, 1W를 "주간 흐름 참고 준비 중"으로 표시하고 사용자 검증으로 넘어간다.

## 3. 두 제품의 역할

### 3.1 연구용 Lens

연구용 Lens는 내부 실험실이다.

포함해도 되는 것:

- 원시 가격 차트
- 보조지표 전체
- AI 예측선과 AI 밴드 실험 결과
- 모델 학습 화면
- run_id, checkpoint, feature contract, source hash
- local parquet, Supabase thin DB 점검 화면
- CP 문서, 실험 로그, 실패 기록
- 백테스트 상세와 내부 metric

연구용 Lens의 목표:

- 모델과 데이터 계약을 검증한다.
- 실패한 실험의 이유를 남긴다.
- 배포용으로 넘길 수 있는 결과물만 선별한다.
- 개인 연구와 개발 속도를 유지한다.

연구용 Lens는 외부 사용자에게 그대로 배포하지 않는다.

### 3.2 배포용 Lens Signal

Lens Signal은 외부 사용자용 제품이다.

제품 한 줄 정의:

> Lens Signal은 보조지표와 AI 리스크 밴드, 백테스트 요약을 바탕으로 오늘 이 종목을 더 볼 가치가 있는지 판단하게 돕는 전략 점검 도구다.

보여줄 것:

- 오늘 볼 만한 종목 목록
- 신호 그룹: 진입 검토, 위험 점검, 전략 유지, 신호 약함
- 신호 이유 한 줄
- 기준일과 데이터 최신성
- 1D/1W 관점 구분
- 백테스트 요약
- 모델 신뢰 상태
- 투자 권유가 아니라는 고지

기본으로 숨길 것:

- 원시 OHLC 가격 히스토리
- 대형 가격 차트
- run_id와 checkpoint 상세
- 내부 metric 원문
- 학습 콘솔
- CP 로그
- Supabase/Parquet 운영 정보

가격 차트가 필요하면 기본 화면이 아니라 상세의 보조 영역에 둔다. 배포용 핵심은 가격을 보여주는 것이 아니라, 신호를 이해하고 다음 행동을 정하는 흐름이다.

## 4. 데이터 전달 경계

연구용 Lens에서 Lens Signal로 넘기는 값은 승인된 제품 산출물만으로 제한한다.

허용 데이터:

| 항목 | 설명 |
|---|---|
| ticker | 종목 코드 |
| asof_date | 모델과 데이터 기준일 |
| timeframe | 1D 또는 1W |
| signal_group | 진입 검토, 위험 점검, 전략 유지, 신호 약함 |
| signal_label | 사용자에게 보이는 짧은 신호 |
| reason_code | 판단 근거 코드 |
| reason_text | 사용자용 한 줄 설명 |
| line_summary | 보수적 예측선 요약 |
| band_summary | AI 밴드/하방 위험 요약 |
| backtest_summary | 수익률, MDD, 샤프, 승률 등 제한된 요약 |
| model_version | 사용자가 이해 가능한 버전 |
| data_freshness | 최신성 상태 |

금지 데이터:

- 전체 `price_data`
- 전체 `indicators`
- 전체 prediction history bulk
- 학습 feature tensor
- 내부 로그 원문
- 실패 run 전체 목록
- provider가 섞인 원천 데이터

원칙:

- local parquet가 연구와 학습의 원천이다.
- Supabase는 Lens Signal 표시용 얇은 DB다.
- top-k 신호와 선택 ticker 최신 요약만 올린다.
- raw 가격/지표 대량 read/write는 Phase 2에서도 금지한다.

## 5. Lens Signal 핵심 사용자 루프

Lens Signal의 기본 루프는 다음 순서다.

1. 사용자가 오늘의 신호 목록을 본다.
2. 진입 검토 또는 위험 점검 종목을 클릭한다.
3. 왜 이 신호인지 한 줄 근거를 읽는다.
4. 백테스트 요약과 주의점을 확인한다.
5. 이 종목을 더 볼지, 넘길지 결정한다.
6. 필요하면 피드백을 남긴다.

이 루프 밖의 기능은 Phase 2 초기에는 뒤로 미룬다.

뒤로 미룰 것:

- 포트폴리오 저장
- 자동 매매
- 알림
- 커스텀 전략 빌더
- 고급 차트
- 모델 실험 비교 화면
- 유료 구독

## 6. 초기 화면 구조

### 6.1 오늘의 신호

첫 화면은 대시보드가 아니라 신호 목록이다.

구성:

- 기준일
- 데이터 최신성
- 스캔 범위
- 진입 검토 상위 종목
- 위험 점검 상위 종목
- 전략 유지 종목
- 신호 약함 요약

주의:

- "매수", "판매" 같은 직접 행동 문구는 기본 라벨로 쓰지 않는다.
- "진입 검토", "청산 검토", "위험 점검"처럼 판단 보조 언어를 쓴다.
- 전체 universe가 아니면 대표 종목 데모라고 명시한다.

### 6.2 종목 상세

종목 상세는 가격 차트보다 신호 해석을 먼저 보여준다.

구성:

- 종목명과 기준일
- 현재 신호 그룹
- 왜 이 신호인지
- 하방 위험 요약
- 1D와 1W 관점 차이
- 백테스트 요약
- 모델 신뢰 상태
- 투자 판단 주의 문구

가격 차트는 넣더라도 접힌 보조 영역으로 둔다.

### 6.3 방법론/주의사항

방법론 화면은 너무 길게 만들지 않는다.

포함할 것:

- 투자 권유가 아님
- 모델 기준일
- 데이터 출처와 한계
- 백테스트는 미래 수익 보장이 아님
- 수수료와 거래비용 가정
- AI 밴드는 목표가가 아니라 위험 참고 범위임

## 7. 클릭 데이터 분석 계획

분석의 목적은 방문자 수 자랑이 아니라 사용자가 제품 루프를 이해하는지 확인하는 것이다.

필수 이벤트:

| 이벤트 | 목적 |
|---|---|
| `page_view` | 어떤 화면을 보는지 |
| `signal_card_click` | 신호 카드가 실제로 눌리는지 |
| `ticker_search` | 사용자가 직접 종목을 찾는지 |
| `signal_detail_open` | 카드 클릭 후 상세까지 가는지 |
| `backtest_expand` | 백테스트 요약을 더 보는지 |
| `timeframe_toggle` | 1D/1W를 구분해서 쓰는지 |
| `risk_group_filter` | 진입 검토/위험 점검 중 무엇을 보는지 |
| `methodology_open` | 방법론/주의사항을 확인하는지 |
| `feedback_submit` | 정성 피드백이 들어오는지 |
| `signup_interest` | 로그인/대기자 수요가 있는지 |

이벤트 속성:

| 속성 | 예시 |
|---|---|
| ticker | AAPL |
| signal_group | risk |
| timeframe | 1D |
| asof_date | 2026-05-06 |
| source_surface | today_signal |
| chart_visible | false |
| user_type_answer | 개인투자자, 퀀트 성향 |

초기에는 개인정보를 최소화한다. 로그인 없는 공개 데모라면 익명 이벤트와 피드백만 받는다.

## 8. 사용성 판단 지표

초기 검증 지표:

| 지표 | 해석 |
|---|---|
| 신호 카드 클릭률 | 첫 화면 신호가 호기심을 만드는지 |
| 상세 진입률 | 카드 클릭 후 맥락이 이어지는지 |
| 백테스트 펼침률 | 사용자가 근거를 확인하려 하는지 |
| 1W 토글 사용률 | 주간 관점이 실제로 필요한지 |
| 피드백 제출률 | 제품이 의견을 남길 만큼 이해되는지 |
| 재방문률 | 하루 뒤 다시 볼 이유가 있는지 |
| 방법론 열람률 | 신뢰/주의사항을 확인하려는 수요 |

초기 성공 기준 초안:

- 신호 카드 클릭률 20% 이상
- 상세 진입 후 30초 이상 체류 비율 30% 이상
- 백테스트 펼침률 15% 이상
- 피드백 또는 설문 응답 5명 이상
- "무슨 앱인지 모르겠다" 피드백이 반복되지 않을 것

숫자는 초기 가정이다. 실제 트래픽이 작으면 절대값보다 사용자 세션 녹취와 피드백을 더 중요하게 본다.

## 9. 구글 설문과 사용자 인터뷰

구글 설문은 제품 목적 검증용으로 쓴다.

물어볼 것:

- 주식 종목을 고를 때 가장 먼저 보는 것은 무엇인가?
- 하루에 몇 종목을 확인하는가?
- 보조지표를 어느 정도 이해하는가?
- 백테스트 결과를 투자 판단에 참고하는가?
- AI 예측선보다 AI 위험 범위가 더 유용하게 느껴지는가?
- "오늘 볼 만한 종목" 목록이 있으면 쓰겠는가?
- 가격 차트 없이 전략 신호와 근거만 있어도 이해 가능한가?
- 진입 검토, 위험 점검, 전략 유지, 신호 약함 문구가 안전하게 느껴지는가?
- 어떤 경우에 다시 방문할 것 같은가?

5명 사용자 테스트 과제:

1. 오늘 볼 만한 종목 하나를 고르게 한다.
2. 왜 그 종목을 골랐는지 설명하게 한다.
3. 위험 점검 종목 하나를 보고 넘길지 더 볼지 결정하게 한다.
4. 1D와 1W 차이를 이해했는지 물어본다.
5. 가격 차트가 없거나 접혀 있어도 판단이 가능한지 확인한다.

테스트 중 사용자가 막히는 문구는 바로 기록하고 다음 UX 수정 후보로 올린다.

## 10. 로그인 전략

Phase 2 초기에는 로그인 없이 시작한다.

로그인 없이 하는 이유:

- 진입 장벽이 낮다.
- 사용성 검증이 빠르다.
- 초기에는 저장 기능보다 이해 여부가 중요하다.
- 개인정보와 약관 부담을 줄일 수 있다.

로그인이 필요한 시점:

- 관심종목 저장
- 사용자별 재방문 추적
- 알림
- 비공개 베타
- 유료화
- 사용자별 약관 동의 기록

권장 순서:

1. 로그인 없는 공개 데모
2. 피드백 폼과 이메일 관심 등록
3. 클로즈드 베타에서 로그인 추가
4. watchlist와 저장 기능 추가
5. 유료화 가능성 검토

## 11. 운영자용 대시보드

Phase 2에는 작은 운영자용 화면 또는 리포트가 필요하다.

봐야 할 것:

- latest asof_date
- 데이터 최신성
- 신호 생성 성공/실패 수
- top-k 신호 수
- Supabase egress
- API 응답 크기
- 이벤트 수집 정상 여부
- 피드백 수
- 모델 버전
- 데이터 source/provider/hash

운영 중단 조건:

- asof_date가 오래됨
- source/provider가 섞임
- Supabase에서 가격/지표 대량 조회 발생
- 신호가 전부 같은 그룹으로 쏠림
- 모델 기준일과 표시 기준일이 다름
- 사용자에게 투자 권유처럼 읽히는 문구가 발견됨

## 12. 법률/데이터 리스크 가드

Phase 2에서 반드시 보수적으로 볼 것:

- 무료 데이터 원천의 상업 배포 가능 여부
- 가격 차트와 원시 시계열 재배포 여부
- 신호가 투자자문처럼 읽히는지
- 백테스트가 수익 보장처럼 보이는지
- 매수/매도 문구 사용 여부
- 사용자 피드백과 이메일 수집 시 개인정보 고지

문구 원칙:

- "매수 추천" 대신 "진입 검토"
- "매도 추천" 대신 "위험 점검" 또는 "청산 검토"
- "목표가" 대신 "보수적 참고값"
- "예측 성공" 대신 "과거 검증 기준"
- "수익 가능" 대신 "백테스트 기준 결과"

법률 확정 판단은 별도 검토가 필요하다. 개발 문서에서는 안전한 제품 표현과 데이터 경계만 먼저 고정한다.

## 13. MLOps와 자동 버전 업데이트 방향

Phase 1에서 진행한 실험은 단순 학습 기록이 아니다. 그대로 정리하면 Lens의 모델 승격 파이프라인 원형이 된다.

현재 수동 실험 흐름:

1. 데이터 스냅샷 고정
2. baseline 평가
3. 후보 모델 학습
4. metric gate 평가
5. 실패 유형 분류
6. best 후보 seed 재검증
7. product slot 승격 여부 판단
8. registry 기록

이 흐름을 자동화하면 Lens의 MLOps가 된다. Phase 2에서는 완전 자동 교체가 아니라, 먼저 "자동 평가와 교체 추천"까지를 목표로 둔다. product slot 교체는 사람이 승인한다.

### 13.1 지금 실험 단계와 나중 MLOps 역할

| 지금 실험 단계 | 나중 MLOps 역할 |
|---|---|
| data/cache gate | data validation |
| baseline 평가 | regression test |
| coarse model 비교 | candidate generation |
| Optuna/W&B sweep | hyperparameter search |
| seed stability | robustness check |
| hard gate | model validation |
| product slot 결정 | model registry promotion |
| 실패 taxonomy | model monitoring issue label |

이 매핑을 유지하면 CP 문서는 단발성 실험 로그가 아니라, 나중에 자동 파이프라인의 실행 기록과 설계 근거가 된다.

### 13.2 주기적 모델 평가 루프

나중에는 매주 또는 매월 다음 루프를 돌린다.

1. 새 데이터 snapshot 생성
2. 기존 product model 재평가
3. 후보 모델 vNext 학습
4. baseline과 기존 모델 비교
5. gate 통과 여부 확인
6. seed stability 확인
7. 자동으로 "교체 추천" 또는 "유지" 판정
8. 사람이 승인하면 product slot 교체
9. Lens Signal에 새 model_version 반영
10. 이전 버전과 새 버전의 사용자 지표를 비교

Phase 2의 자동화 목표는 1번부터 7번까지다. 8번 product slot 교체는 수동 승인으로 유지한다.

### 13.3 product slot 기준

MLOps는 모델 하나를 잘 뽑는 시스템이 아니라, 제품 슬롯을 안전하게 유지하는 시스템이다.

Phase 2 기준 product slot:

| 슬롯 | 역할 | 자동 평가 기준 |
|---|---|---|
| 1D line | 오늘 판단용 보수적 예측선 | 기존 모델 대비 false-safe 감소, IC/spread 유지 |
| 1D band | 오늘 판단용 AI 리스크 범위 | coverage, band width, downside guard 유지 |
| 1W line | 주간 흐름 참고 예측선 | 1D와 충돌하지 않는 방향성, 과적합 guard |
| 1W band | 주간 리스크 범위 | 후보가 약하면 준비 중 유지 |

slot 교체 원칙:

- 새 모델이 baseline만 이겨서는 부족하다.
- 기존 product model을 이겨야 한다.
- hard gate를 통과해야 한다.
- seed stability가 흔들리면 교체하지 않는다.
- 사용자 화면의 신호 분포가 갑자기 깨지면 교체하지 않는다.
- 교체 후 rollback 가능한 registry가 있어야 한다.

### 13.4 registry에 남길 것

registry는 모델 파일 목록이 아니라 제품 승격 판단 기록이다.

필수 기록:

| 항목 | 설명 |
|---|---|
| model_version | 사용자와 운영자가 볼 수 있는 버전 |
| slot | 1D line, 1D band, 1W line, 1W band |
| provider/source | eodhd, yfinance 등 |
| source_data_hash | 데이터 snapshot fingerprint |
| feature_version | feature contract |
| train_range | 학습 기간 |
| val_range | 검증 기간 |
| test_range | 테스트 기간 |
| baseline_metrics | 기준 모델 성능 |
| incumbent_metrics | 기존 product model 성능 |
| candidate_metrics | 후보 모델 성능 |
| hard_gate_result | PASS 또는 FAIL |
| seed_stability | seed 재검증 결과 |
| failure_label | 실패 시 taxonomy label |
| promotion_decision | promote, keep, reject |
| approved_by | 수동 승인자 |
| approved_at | 승인 시각 |
| rollback_target | 되돌릴 이전 model_version |

### 13.5 자동 버전 업데이트 원칙

자동 버전 업데이트는 "모델이 좋으니 바로 배포"가 아니다.

안전한 순서:

1. 새 후보를 shadow model로 평가한다.
2. 기존 product model과 같은 snapshot에서 비교한다.
3. hard gate와 seed stability를 통과한 경우에만 교체 추천을 만든다.
4. 사람이 registry diff를 확인한다.
5. product slot pointer만 새 version으로 바꾼다.
6. Lens Signal은 새 pointer를 읽어 model_version을 표시한다.
7. 이상 징후가 있으면 이전 pointer로 rollback한다.

초기에는 자동으로 PR 또는 report를 만들고, 실제 교체는 수동으로 한다. 자동 배포는 최소 2~3회 수동 승격이 안정적으로 반복된 뒤 검토한다.

### 13.6 사용자 경험 지표와 연결

모델 승격은 offline metric만으로 결정하지 않는다. Lens Signal에서는 사용자 경험 지표도 같이 본다.

함께 볼 지표:

- 신호 카드 클릭률
- 상세 진입률
- 백테스트 펼침률
- 위험 점검 카드 체류 시간
- 피드백의 이해도
- 신호 그룹 쏠림
- 재방문률

예를 들어 offline metric은 좋아졌는데 사용자가 위험 점검 카드만 보고 바로 이탈하면, 모델이 너무 보수적이거나 문구가 불안하게 읽히는 것일 수 있다. 반대로 클릭률이 높아도 hard gate가 약하면 product slot으로 승격하지 않는다.

### 13.7 Phase 2에서 만들 최소 MLOps 산출물

Phase 2에서 바로 거대한 MLOps 시스템을 만들지는 않는다. 하지만 나중에 자동화할 수 있도록 최소 산출물은 남긴다.

만들 것:

- `model_registry` 또는 registry JSON/Parquet 초안
- product slot pointer 파일
- snapshot manifest
- baseline regression report
- candidate vs incumbent 비교 report
- hard gate 결과 report
- seed stability report
- promotion decision note
- rollback 절차 문서

하지 않을 것:

- 사람 승인 없는 자동 교체
- 매일 full retraining
- 사용자 트래픽 전체를 새 모델에 즉시 전환
- offline metric만으로 제품 승격
- 실패 taxonomy 없는 자동 reject

### 13.8 내 판단

이 방향은 Lens와 잘 맞는다. 지금까지의 CP가 많았던 이유는 단순히 작업이 잘게 쪼개져서가 아니라, 데이터 계약, 모델 역할, 실패 유형, 제품 슬롯을 계속 분리해 왔기 때문이다. 이 기록은 나중에 MLOps로 바꾸기 좋은 형태다.

다만 우선순위는 조심해야 한다. Phase 2의 첫 목표는 여전히 Lens Signal의 사용자 경험 검증이다. MLOps는 그 검증을 망치지 않는 선에서 "교체 추천과 registry 기록"부터 시작한다.

정리하면 다음 순서가 좋다.

1. 1D/1W 제품 슬롯을 수동으로 한 번 닫는다.
2. Lens Signal MVP를 만든다.
3. 클릭 데이터로 사용자가 신호를 이해하는지 본다.
4. 동시에 registry와 promotion report를 남긴다.
5. 그 다음 반복부터 자동 평가와 교체 추천을 붙인다.

## 14. Phase 2 작업 순서

### 14.1 Phase 2A: 제품 경계 확정

- 연구용 Lens와 Lens Signal 메뉴/라우트 분리
- Lens Signal에 넘길 데이터 DTO 정의
- 숨길 내부 정보 목록 확정
- 가격 차트 기본 숨김 여부 결정
- 투자 권유 방지 문구 초안 작성

완료 기준:

- 연구용 화면과 배포용 화면의 목적이 섞이지 않는다.
- Lens Signal 첫 화면에서 앱 목적을 10초 안에 이해할 수 있다.

### 14.2 Phase 2B: Lens Signal MVP 화면

- 오늘의 신호 목록 화면
- 신호 그룹 카드
- 종목 상세 화면
- 백테스트 요약 영역
- 1D/1W 상태 표시
- 방법론/주의사항 화면

완료 기준:

- 사용자가 종목을 클릭하고 신호 근거를 확인할 수 있다.
- 가격 차트 없이도 화면 목적이 유지된다.

### 14.3 Phase 2C: 클릭 데이터 계측

- 이벤트 스키마 정의
- 분석 도구 선택
- 필수 이벤트 심기
- 개발/배포 환경 이벤트 분리
- 이벤트 누락 점검

완료 기준:

- 신호 카드 클릭부터 상세 진입까지 흐름이 이벤트로 남는다.
- 개인정보 없이도 초기 사용성 판단이 가능하다.

### 14.4 Phase 2D: 피드백 검증

- 구글 설문 작성
- 피드백 버튼 연결
- 5명 과제형 테스트 진행
- 막힌 문구와 혼란 지점 정리
- UX 수정 우선순위 작성

완료 기준:

- 앱 목적을 이해하지 못하는 지점이 문서화된다.
- 가격 차트 숨김/축소가 가능한지 판단한다.

### 14.5 Phase 2E: 공개 데모

- 로그인 없는 공개 데모 배포
- 대표 종목 또는 제한 universe로 시작
- 데이터 최신성 표시
- 사용량과 egress 모니터링
- 피드백/관심 등록 수집

완료 기준:

- 실제 외부 사용자가 들어와도 비용과 데이터 경계가 깨지지 않는다.
- 다음 단계가 로그인인지, UX 수정인지, 모델 개선인지 판단할 수 있다.

### 14.6 Phase 2F: MLOps 최소 골격

- registry 초안 작성
- product slot pointer 정의
- candidate vs incumbent 비교 report 생성
- hard gate와 seed stability report 표준화
- 자동 교체 추천 report 생성
- 수동 승인과 rollback 절차 작성

완료 기준:

- 새 모델 후보가 기존 product slot을 이겼는지 같은 형식으로 비교할 수 있다.
- 자동으로 교체하지 않아도, 사람이 승인할 수 있는 promotion package가 남는다.

## 15. Phase 2에서 하지 않을 것

- 앱스토어 출시
- 유료 결제
- 자동 매매
- 전체 가격 데이터 공개 API
- 고급 퀀트 전략 빌더
- 개인 포트폴리오 저장
- 실시간 가격 제공
- 과도한 모델 추가 실험
- 1M 예측 억지 연결
- 사람 승인 없는 product slot 자동 교체
- 매일 full retraining

Phase 2의 목표는 큰 제품을 완성하는 것이 아니라, 작게 배포해서 사용자가 신호를 이해하는지 확인하는 것이다.

## 16. 최종 정리

Phase 2는 두 갈래를 분명히 나누는 단계다.

연구용 Lens는 계속 깊게 판다. 실패 로그, 모델 실험, 원천 데이터, 내부 metric을 모두 품는다.

Lens Signal은 작게 보여준다. 오늘 볼 종목, 위험 점검, 신호 근거, 백테스트 요약, 주의사항만 남긴다.

그 위에 클릭 데이터와 피드백을 붙여서, 다음 질문에 답한다.

> 이 앱은 사용자가 실제로 이해하고 다시 보고 싶은 전략 점검 도구인가?

동시에 모델 registry와 promotion report를 남겨, 반복 실험이 나중에 자동 평가와 교체 추천으로 이어지게 만든다.

이 질문에 답하기 전까지는 기능을 크게 늘리지 않는다. 모델 교체도 처음에는 자동화하지 않고, 자동 추천과 사람 승인을 분리한다.
