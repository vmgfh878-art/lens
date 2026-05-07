# CP58-G 기획 의도 대비 구현 해석 전수 감사 보고서

작성일: 2026-04-30

감사 성격: 코드 수정, 테스트 수정, 포맷팅, DB 쓰기, 모델 학습, `save-run`, UI 수정 없이 수행한 읽기 전용 의도 오해 감사다. 산출물 작성만 예외로 이 파일을 신규 생성했다.

## 1. 핵심 요약

이번 감사의 핵심 결론은 명확하다. Lens의 원 기획은 `AI line`과 `AI band`를 하나의 강제 합성 모델로 해석하는 것이 아니라, RSI와 MACD처럼 함께 보는 서로 다른 AI 보조지표 layer로 해석하는 쪽이다. `backend/docs/project_plan.md:13-19`는 AI를 예측 도구가 아니라 보조지표로 정의하고, `README.md:219-223`도 AI는 보조지표이며 밴드 품질과 규칙의 투명성이 본체라고 정리한다.

가장 큰 불일치는 CP40-CP46 구간에서 `line_band_composite`가 하나의 모델/run처럼 굳어지고, `line_inside_band`와 clamp 정책이 성능 또는 통과 조건처럼 해석된 부분이다. 이는 사용자 의도와 불일치다. 이후 CP51, CP52, CP56, CP57에서 이 해석을 상당 부분 되돌렸고, 특히 `docs/cp57_ai_indicator_layer_contract_report.md:5-19`는 composite model/prediction을 Phase 1 본류에서 내리고 `AI line layer`, `AI band layer`, `overlay bundle`로 분리한다고 명시한다.

다만 현재 코드와 제품 노출 경로에는 legacy composite 잔재가 남아 있다. `ai/composite_inference.py:374`, `428`, `607-612`는 composite run을 만들 수 있고 기본 policy도 `risk_first_lower_preserve`다. 동시에 `frontend/src/components/StockView.tsx:80`, `571`과 `frontend/src/components/TrainingView.tsx:56`, `102`는 `line_band_composite`를 표시 후보로 포함한다. 코드 내부에 `deprecated_for_phase1_product_contract=True`와 `indicator_layer_replacement="overlay_bundle"` 보정이 들어간 것은 맞지만, 제품 화면이나 보고서에서 composite를 최종 모델처럼 읽으면 여전히 P0급 해석 왜곡이 발생한다.

이번 CP의 판정은 “모델 구현이 틀렸다”보다 “해석 층이 한때 틀렸고, 일부 legacy 경로가 아직 그 오해를 재생산할 수 있다”에 가깝다. 따라서 다음 작업은 학습이 아니라 용어, 저장/조회, UI provenance, role별 output 차단 계약을 먼저 닫는 쪽이 맞다.

## 2. 기획 의도 핵심 재정의

Lens의 제품 의도는 가격을 중심에 두고, AI가 의사결정을 대신하는 것이 아니라 판단 근거를 늘려주는 구조다. 초기 기획 문서의 핵심 문장은 다음과 같다.

| 근거 | 기획 의도 |
| --- | --- |
| `backend/docs/project_plan.md:9` | 미국 주식 데이터, AI 예측, 보조지표를 결합한 의사결정 지원 시스템 |
| `backend/docs/project_plan.md:13-14` | AI는 예측 도구가 아니라 보조지표이며, 최종 의사결정은 사용자 몫 |
| `backend/docs/project_plan.md:18-19` | AI 결과를 RSI, 이동평균, 변동성 지표와 함께 해석 |
| `backend/docs/project_plan.md:34`, `197` | AI 하단밴드는 ATR, RSI 같은 조건과 함께 쓰는 신호 |
| `README.md:5`, `22-25` | AI 보조지표 기반 플랫폼, calibrated band와 규칙 엔진 |
| `README.md:219-223` | AI는 보조지표이고 밴드/규칙 투명성이 승부처 |

따라서 CP58 기준 용어는 아래처럼 고정하는 편이 맞다.

| 용어 | 의도 |
| --- | --- |
| `AI line layer` | 하방 보수적 예측선 또는 순위/방향 점수 layer다. 제품에서는 `line_series`만 본다. |
| `AI band layer` | 가격 범위 자체가 의미 있는 위험 보조지표 layer다. 제품에서는 `lower_band_series`, `upper_band_series`를 본다. |
| `overlay bundle` | 화면에서 line layer와 band layer를 함께 보여주는 표시 묶음이다. 모델도 아니고 성능 평가 단위도 아니다. |
| `composite model` | Phase 1 본류에서는 쓰지 않는 legacy 표현이다. |
| `line_inside_band` | 모델 탈락 기준이 아니라 overlay 표시 참고 지표다. |

## 3. 의도와 구현이 맞는 부분

| 영역 | 맞는 부분 | 근거 |
| --- | --- | --- |
| 원 기획 | AI를 보조지표로 보는 정체성이 분명하다. | `backend/docs/project_plan.md:13-19`, `README.md:219-223` |
| 제품 표시 | 가격 + 보수적 예측선 + AI band overlay 방향은 기획과 일치한다. | `docs/cp_product_demo_plan.md:77-96`, `docs/project_journal.md:727-728`, `814-815` |
| 1M 정책 | 1M은 가격 전용이고 AI layer 비활성이라는 계약이 유지된다. | `README.md:162`, `docs/cp_product_demo_plan.md:72`, `86`, `347-349`, `frontend/src/components/StockView.tsx:725`, `744-745`, `frontend/src/components/Chart.tsx:65-67` |
| 역할 분리 학습 | `line_gate`, `band_gate`가 분리되어 line 후보와 band 후보를 다르게 고른다. | `ai/train.py:284-321`, `1595-1599` |
| 평가 재설계 | CP51/CP52는 line, band, composite 표시 지표를 분리했고 `line_inside_band`를 탈락 기준에서 제외했다. | `docs/cp51_metric_redesign_report.md:39-43`, `docs/cp52_metric_definition_contract_report.md:104-113` |
| CP57 방향 수정 | composite model/prediction을 Phase 1 본류에서 내리고 indicator layer로 분리했다. | `docs/cp57_ai_indicator_layer_contract_report.md:5-19`, `21-29`, `41-52`, `143` |
| composite 코드 보정 | current composite 도구는 결과에 Phase 1 legacy 안내와 `overlay_bundle` 대체 지위를 기록한다. | `ai/composite_inference.py:607-612`, `ai/composite_policy_eval.py:221-222` |
| provenance 일부 노출 | prediction meta와 화면에서 line/band run provenance를 어느 정도 드러낸다. | `backend/app/services/api_service.py:102-119`, `frontend/src/components/StockView.tsx:793-796`, `923-944` |

## 4. 의도와 어긋난 부분 P0/P1/P2/P3 표

| 등급 | 항목 | 판정 | 근거 | 영향 | 다음 조치 |
| --- | --- | --- | --- | --- | --- |
| P0 | `line_band_composite`가 제품 후보처럼 남아 있는 경로 | 불일치. CP57 의도는 composite를 Phase 1 본류에서 내리는 것인데, 현재 코드와 화면 후보에 composite run이 남아 있다. | `docs/cp57_ai_indicator_layer_contract_report.md:13-19`, `ai/composite_inference.py:374`, `428`, `607-612`, `frontend/src/components/StockView.tsx:80`, `571`, `frontend/src/components/TrainingView.tsx:56`, `102` | 제품 데모나 최근 run 조회에서 composite를 최종 모델처럼 읽으면 line과 band의 독립 layer 의미가 사라진다. | 다음 CP에서 신규 composite 저장 금지, 제품 후보 목록 제외, legacy run 표시 라벨을 분리한다. |
| P1 | `line_inside_band`가 legacy pass flag로 남아 있음 | 불일치. CP51/52/57은 표시 참고 지표라고 했지만 CP43 legacy 도구는 통과 조건으로 계산한다. | `ai/composite_policy_eval.py:95-107`, `214-215`, `docs/cp51_metric_redesign_report.md:39-43`, `docs/cp52_metric_definition_contract_report.md:104-113`, `docs/cp57_ai_indicator_layer_contract_report.md:15`, `87-93` | 모델 성능이 아니라 표시 정합성인 값을 모델 탈락/통과처럼 해석할 수 있다. | legacy 도구 명칭과 출력에 “성능 gate 아님”을 더 강하게 표시하고, 본류 리포트에서 제외한다. |
| P1 | standalone inference가 role별 output을 차단하지 않음 | 불일치. line run의 lower/upper가 제품 band로, band run의 line이 제품 예측선으로 오해될 수 있다. | `ai/inference.py:299-316`, `docs/cp56_line_band_contract_audit_report.md:74`, `131-135`, `docs/cp57_ai_indicator_layer_contract_report.md:31-39` | line layer와 band layer의 provenance가 분리되어도 저장 row 구조가 다시 합성처럼 보일 수 있다. | `role=line_model`은 band output 무시, `role=band_model`은 line output 무시 정책을 저장/조회 계약에 반영한다. |
| P1 | `include_line_clamp`와 `risk_first_lower_preserve`가 같은 구현 | 불일치. 이름은 다른 정책이지만 실제로는 둘 다 line을 band 안에 넣는다. | `ai/composite_inference.py:237-241`, `docs/cp56_line_band_contract_audit_report.md:102`, `109`, `120-125`, `docs/cp57_ai_indicator_layer_contract_report.md:15-19` | clamp로 늘어난 폭이 band 모델의 신호인지, 제품 표시 후처리인지 구분되지 않는다. | Phase 1 본류에서 clamp 정책을 성능 개선으로 기록하지 않는다. 유지한다면 legacy utility로만 둔다. |
| P1 | full rolling AI band와 latest forecast 표시 계약이 분리되지 않음 | 추가 결정 필요. full rolling이 제품 계약이면 현재 latest forecast 방식은 불일치다. | `docs/cp50_plan_vs_implementation_audit_report.md:85-86`, `177-187`, `frontend/src/components/StockView.tsx:746`, `755`, `frontend/src/components/Chart.tsx:77-116` | 첫 화면에서 AI 밴드가 전체 차트를 덮는 지표인지, 최신 예측 구간만 보여주는 forecast인지 제품 해석이 달라진다. | 제품 기본을 `latest forecast overlay`로 둘지 `rolling AI band history`로 확장할지 별도 결정한다. |
| P1 | Training 화면에서 composite가 run 모델처럼 보이고 일부 모델명이 고정 표시됨 | 불일치 가능성이 높다. composite는 legacy인데 training run 후보로 포함되고, line/band 모델 표시가 고정값으로 보인다. | `frontend/src/components/TrainingView.tsx:56`, `85`, `102`, `245-253`, `292`, `299`, `docs/cp57_ai_indicator_layer_contract_report.md:47-52` | TiDE band 또는 다른 후보가 들어가도 PatchTST/CNN-LSTM 조합처럼 보일 수 있고, composite가 본류 모델처럼 보일 수 있다. | 화면 표시 계약은 별도 CP에서 `legacy composite`와 `layer run`을 분리한다. |
| P2 | CP43-CP46 문서에 composite policy가 후보/기본처럼 남아 있음 | 불일치였으나 CP57에서 뒤집힌 legacy 기록이다. | `docs/project_journal.md:1513-1524`, `1537-1539`, `1557-1571`, `docs/cp57_ai_indicator_layer_contract_report.md:19`, `41-52` | 과거 보고서를 단독으로 읽으면 `risk_first_lower_preserve`나 `upper_buffer`가 모델 개선으로 오해될 수 있다. | CP57 이후 보고서와 SoT에서 “과거 legacy 기록”이라고 연결 주석을 둔다. |
| P2 | `risk_first_upper_buffer_1.10` 문서와 현재 코드 상태가 다름 | 의도 불일치라기보다 이력 해석 주의가 필요하다. CP46은 확장 후보라고 했지만 CP57은 구현을 남기지 않았다고 정리했다. | `docs/project_journal.md:1571`, `docs/cp57_ai_indicator_layer_contract_report.md:91-93`, `133`, `ai/composite_inference.py:41` | 어느 CP를 기준으로 삼느냐에 따라 “상단 buffer가 기본 후보”인지 “legacy 제외 정책”인지 혼동된다. | CP57을 최신 SoT로 명시하고 CP46은 폐기 전 후보로만 인용한다. |
| P2 | backend 평가 조회가 CP51/52 확장 지표보다 legacy metric 중심 | 해석 혼동 위험. API 요약은 coverage, breach, avg width 중심이라 band를 coverage-only interval로 축소해 읽기 쉽다. | `backend/app/repositories/ai_repo.py:11-13`, `backend/app/routers/v1/ai.py:122-125`, `ai/evaluation.py:12-47`, `418-453` | band width IC, interval score, dynamic width 같은 “밴드 자체 지표”가 제품/운영 요약에서 약하게 보인다. | 조회 API와 리포트에서 CP51/52 지표를 role별로 노출할지 결정한다. |
| P2 | band calibration이 coverage target 중심으로 읽힐 수 있음 | 부분 불일치 위험. calibration 자체는 필요하지만, AI band 의도가 “coverage만 맞추는 구간”으로 축소되면 틀린 해석이다. | `ai/band_calibration.py:163-205`, `351`, `docs/cp51_metric_redesign_report.md:29-43`, `docs/cp52_metric_definition_contract_report.md:81`, `104-113` | band 폭 자체가 위험 신호인지 보는 목적이 약해지고, 단순 넓은 구간도 좋은 band처럼 보일 수 있다. | coverage, interval score, dynamic width, downside width IC를 함께 본다고 SoT에 고정한다. |
| P2 | h5/h20 제품 기본 해석이 아직 열려 있음 | 추가 결정 필요. h5는 short-horizon branch 또는 Phase 1 제품 후보이지, h20/h12 원 기획 폐기는 아니다. | `docs/cp50_plan_vs_implementation_audit_report.md:7-9`, `66-68`, `97-106`, `docs/project_journal.md:1575-1583` | h5 결과를 long-horizon 제품 의도까지 대표하는 것처럼 말할 위험이 있다. | 제품 기본 horizon과 long-horizon branch를 명시적으로 분리한다. |
| P2 | `series_length_all_5` 하드코딩 명칭 | h5 중심 해석 잔재다. h20 smoke에서는 실제 길이 20이 가능했으나 check 이름은 5에 고정됐다. | `ai/composite_inference.py:642-645`, `docs/cp56_line_band_contract_audit_report.md:103`, `111`, `149` | horizon 일반화가 되어도 리포트/contract check가 h5 기준처럼 보인다. | composite legacy 경로를 유지한다면 `series_length_all_horizon`으로만 표현한다. |
| P2 | baseline-aware band의 위치 | 의도 혼동 위험. baseline-aware는 본류 AI 모델로 확정된 것이 아니라 다음 설계/비교 branch다. | `docs/cp54_baseline_metric_comparison_report.md:61-81`, `docs/cp55_fair_baseline_and_band_candidate_expansion_report.md:42-45`, `90-93`, `docs/project_journal.md:1641-1657` | Bollinger/rolling quantile이 Phase 1 AI 본류인지, 경쟁 기준선인지 섞일 수 있다. | baseline은 비교 기준선, baseline-aware는 후속 후보로 문서화한다. |
| P3 | 용어가 여러 시대의 흔적을 함께 담고 있음 | 정리 필요. `conservative_series`, `line_band_composite`, `patchtst_line__cnn_lstm_calibrated_band`, `overlay bundle`, `AI band`가 섞인다. | `ai/composite_inference.py:545`, `backend/app/services/api_service.py:170-173`, `docs/cp57_ai_indicator_layer_contract_report.md:21-29`, `frontend/src/components/Chart.tsx:375` | 발표나 문서에서 설명을 잘못하면 사용자가 “하나의 AI 예측”으로 이해할 수 있다. | CP57 용어를 glossary로 승격한다. |
| P3 | Plan 문서의 “AI 예측” 표현 | 보조지표 의도와 함께 읽으면 문제 없지만 단독으로는 예측 모델 중심으로 보일 수 있다. | `backend/docs/project_plan.md:9`, `17-19`, `README.md:22-25` | 새 참여자가 예측 정확도 중심으로 해석할 수 있다. | “AI 예측 산출물은 보조지표로 사용”이라는 문장을 SoT 상단에 반복한다. |

## 5. composite 관련 해석 오류 정리

composite 해석 오류는 구현 버그라기보다 제품/평가 단위의 오해로 시작했다. CP40에서는 PatchTST line과 CNN-LSTM band를 하나의 prediction row로 저장하는 smoke가 열렸고, CP43에서는 `line_inside_band_ratio`와 coverage/breach를 묶어 pass flag를 계산했다. CP44에서는 `risk_first_lower_preserve`가 저장 기본 후보가 되었고, CP46에서는 상단 buffer 정책이 후보로 기록됐다.

이 흐름의 문제는 다음 네 가지다.

| 오류 | 왜 문제인가 | 근거 |
| --- | --- | --- |
| line과 band를 하나의 `line/lower/upper` 모델 출력처럼 묶음 | 서로 다른 보조지표 layer를 하나의 모델로 강제 해석한다. | `docs/cp57_ai_indicator_layer_contract_report.md:7-19`, `29` |
| `line_inside_band`를 통과 조건처럼 봄 | 표시 정합성 참고 지표를 모델 성능 기준으로 바꾼다. | `ai/composite_policy_eval.py:95-107`, `docs/cp51_metric_redesign_report.md:39-43` |
| clamp/upper buffer를 모델 개선처럼 기록 | 실제 모델이 아니라 제품 표시 후처리 또는 legacy policy다. | `ai/composite_inference.py:237-241`, `docs/project_journal.md:1513-1524`, `1571`, `docs/cp57_ai_indicator_layer_contract_report.md:91-93` |
| composite run을 최종 후보처럼 노출 | product provenance가 있어도 사용자에게 하나의 최종 AI 모델처럼 보일 수 있다. | `frontend/src/components/StockView.tsx:80`, `571`, `frontend/src/components/TrainingView.tsx:56`, `102` |

현재 최신 SoT는 CP57이다. CP57의 결론은 `docs/cp57_ai_indicator_layer_contract_report.md:143`에 있는 것처럼 Phase 1에서는 line과 band를 하나의 composite prediction으로 합치지 않는다는 것이다. 따라서 CP58 기준으로 CP43-CP46의 composite 성과 수치는 성능 지표가 아니라 legacy 조합 실험 기록으로만 읽어야 한다.

## 6. 권장 용어

| 권장 용어 | 써야 하는 의미 | 피해야 할 표현 |
| --- | --- | --- |
| `AI line layer` | 하방 보수적 선, 순위/방향 점수, line-only 모델 산출물 | `최종 예측선 모델`, `band 중심선` |
| `AI band layer` | 위험 범위, 하방/상방 band, interval/dynamic width 지표 | `line을 반드시 포함해야 하는 예측 구간` |
| `overlay bundle` | 화면에서 line layer와 band layer를 같이 보여주는 표시 묶음 | `composite model`, `합성 모델`, `최종 모델` |
| `legacy composite utility` | CP40-CP46 기록 재현 또는 연구용 도구 | `Phase 1 본류`, `제품 기본` |
| `line_inside_band display check` | 표시 충돌 경고용 참고값 | `model gate`, `탈락 기준`, `성능 통과` |
| `baseline comparison` | rolling quantile, Bollinger 등 통계 기준선 | `AI band 본류 대체 모델` |

제품 문구는 “AI 예측”보다 “AI 보조지표”, “AI line layer”, “AI band layer”가 안전하다. 화면에서는 “이 line과 band는 서로 다른 run/model에서 온 layer일 수 있다”는 provenance가 드러나야 한다.

## 7. Horizon 관련 남은 의사결정

확정된 것과 미결정인 것을 분리해야 한다.

| 항목 | 현재 판단 | 근거 |
| --- | --- | --- |
| split/gap용 `h_max` | `1D=20`, `1W=12`는 유지되는 leakage 방지 계약이다. | `README.md:151`, `docs/cp50_plan_vs_implementation_audit_report.md:66`, `72` |
| 현재 구현/API 기본 horizon | `1D=5`, `1W=4`가 기본으로 남아 있다. | `docs/cp50_plan_vs_implementation_audit_report.md:31`, `43`, `67` |
| h5 | short-horizon branch 또는 Phase 1 제품 후보로 볼 수 있다. | `docs/cp50_plan_vs_implementation_audit_report.md:99-106`, `docs/project_journal.md:1583` |
| h20/h12 | 원 기획의 long-horizon branch지만 아직 제품 기본 승격은 아니다. | `docs/cp50_plan_vs_implementation_audit_report.md:99-106`, `docs/project_journal.md:1575-1583` |
| composite length check | h5 중심 잔재가 남아 있어 horizon 일반화 표현이 필요하다. | `ai/composite_inference.py:642-645`, `docs/cp56_line_band_contract_audit_report.md:111` |

남은 결정은 두 가지다. 첫째, Phase 1 제품 기본을 h5/h4로 공식 채택할지, 원 기획 h20/h12를 별도 long-horizon 모드로 남길지 결정해야 한다. 둘째, latest forecast 표시를 제품 기본으로 유지할지, rolling AI band history를 제품 계약으로 새로 열지 결정해야 한다.

## 8. Baseline / baseline-aware 위치 정리

baseline은 AI band를 평가하기 위한 경쟁 기준선이다. `docs/cp52_metric_definition_contract_report.md:115-122`는 Bollinger return band와 historical quantile band를 baseline으로 정의하고, CP54/CP55는 CNN-LSTM band가 일부 통계 baseline을 이기지 못했다는 사실을 기록한다. 따라서 baseline은 “Phase 1 AI 본류”가 아니라 “AI band가 최소한 넘어야 할 기준선”이다.

baseline-aware band는 다음 band 설계 후보로 보는 것이 맞다. `docs/cp55_fair_baseline_and_band_candidate_expansion_report.md:90-93`은 baseline-aware residual/scale band를 검토안으로 둔다. 이것을 바로 본류 모델로 부르면 불일치다. 올바른 문장은 “AI band 후보가 rolling/Bollinger baseline을 충분히 이기지 못했으므로, 다음 CP에서 baseline-aware 보정 또는 residual/scale 설계를 비교한다”이다.

band 평가도 coverage 하나로 축소하면 안 된다. CP51/CP52가 추가한 interval score, dynamic width, downside width IC, band width IC는 밴드 폭 자체가 위험 신호인지 확인하기 위한 지표다. `ai/evaluation.py:12-47`, `418-453`에 지표는 존재하지만, backend 조회 요약은 아직 `coverage`, `avg_band_width`, breach 중심이다(`backend/app/repositories/ai_repo.py:11-13`, `backend/app/routers/v1/ai.py:122-125`). 이 차이는 P2 해석 리스크다.

## 9. 다음 CP 추천

| 우선순위 | 추천 CP | 목적 | 금지/주의 |
| --- | --- | --- | --- |
| 1 | Composite legacy 차단 계약 정리 | 신규 Phase 1 경로에서 `line_band_composite`를 제품 후보처럼 쓰지 않도록 저장/조회/UI 후보 정책을 분리한다. | 학습 금지. legacy run 삭제 금지. |
| 2 | Role별 inference 저장 계약 | `role=line_model`은 lower/upper 제품 사용 차단, `role=band_model`은 line 제품 사용 차단을 저장 meta와 API에 명시한다. | 모델 구조 변경 금지. |
| 3 | Overlay bundle API/제품 계약 | line layer와 band layer를 별도 run에서 가져와 함께 표시하되, mismatch와 provenance를 노출한다. latest forecast와 rolling history 중 제품 기본을 결정한다. | 기존 UI를 고치기 전 API 계약부터 확정. |
| 4 | Band baseline-aware 비교 설계 | rolling quantile/Bollinger 대비 AI band의 interval/dynamic width 개선 가능성을 본다. | baseline-aware를 Phase 1 본류로 바로 승격하지 않기. |
| 5 | Horizon SoT 정리 | h5/h4 제품 기본, h20/h12 long-horizon branch, h_max gap을 서로 다른 개념으로 고정한다. | h5 성과를 h20 기획 폐기로 해석하지 않기. |

## 10. 수정 금지 파일을 건드리지 않았다는 확인

이번 CP에서 코드, 테스트, 포맷팅, DB, 모델 학습, `save-run`, UI 파일은 수정하지 않았다. 읽기 전용으로 확인한 파일은 다수지만, 변경한 파일은 신규 감사 산출물인 `docs/cp58_planning_intent_alignment_audit_report.md` 하나다.

`docs/cp58_planning_intent_alignment_matrix.json`은 별도 생성하지 않았다. 이 보고서의 P0/P1/P2/P3 표가 matrix 역할을 충분히 수행하므로 중복 산출물은 만들지 않았다.

## 11. 실행한 읽기 전용 명령 목록

아래 명령은 모두 읽기 전용이다. `rg`는 접근 거부로 실패했고, 이후 PowerShell `Select-String`, `Get-Content`, `Get-ChildItem`, `Test-Path`로 대체했다.

| 목적 | 명령 |
| --- | --- |
| 전체 키워드 검색 시도 | `rg -n "line_inside_band|risk_first|include_line_clamp|upper_buffer|composite|line_model|band_model|overlay|rolling|horizon=5|h20|baseline-aware|Bollinger|Plan v3|AI 밴드|보수적 예측선" docs ai backend frontend` |
| 추가 키워드 검색 시도 | `rg -n "composition_policy|line_model_run_id|band_model_run_id|forecast_dates|latest run|fallback run|1M|price-only|AI layer|final model|최종 모델" docs ai backend frontend` |
| CP 문서 목록 확인 | `Get-ChildItem docs -Filter "*cp*"` 계열 문서 목록 확인 |
| CP57 문서 존재 확인 | `Get-ChildItem docs -Filter "*cp57*"` |
| CP58 산출물 존재 여부 확인 | `Test-Path docs/cp58_planning_intent_alignment_audit_report.md` |
| 초기 기획 문서 검색 | `Select-String -Path backend/docs/project_plan.md -Pattern "AI = 보조지표|AI 결과|기존 기술적 지표|AI 하단밴드|의사결정"` |
| 제품 데모 문서 검색 | `Select-String -Path docs/cp_product_demo_plan.md -Pattern "AI|밴드|latest|1M|price-only|forecast|보조|overlay|전체"` |
| README 검색 | `Select-String -Path README.md -Pattern "AI|보조|horizon|1D=20|1W=12|1M|밴드|latest|rolling"` |
| 주요 감사 문서 검색 | `Select-String -Path docs/cp50_plan_vs_implementation_audit_report.md,docs/cp51_metric_redesign_report.md,docs/cp52_metric_definition_contract_report.md,docs/cp56_line_band_contract_audit_report.md -Pattern ...` |
| CP57 보고서/매트릭스 검색 | `Select-String -Path docs/cp57_ai_indicator_layer_contract_report.md -Pattern ...`, `Select-String -Path docs/cp57_ai_indicator_layer_contract_matrix.json -Pattern ...` |
| 프로젝트 저널 검색 | `Select-String -Path docs/project_journal.md -Pattern "보수적 예측선|AI band|AI 밴드|line_model|band_model|composite|risk_first_lower_preserve|line_inside_band|1M|rolling|최종 후보|최종 모델"` |
| composite 코드 검색 | `Select-String -Path ai/composite_inference.py -Pattern "COMPOSITION_POLICIES|risk_first_lower_preserve|include_line_clamp|line_inside_band|phase1|deprecated|overlay_bundle|model_name|line_model_run_id|band_model_run_id|series_length_all_5"` |
| composite policy 코드 검색 | `Select-String -Path ai/composite_policy_eval.py -Pattern "line_inside_band_ratio|passes_cp43|legacy|Phase 1|모델 성능|pass|phase1"` |
| inference/train/evaluation/calibration 검색 | `Select-String -Path ai/inference.py,ai/train.py,ai/evaluation.py,ai/band_calibration.py -Pattern ...` |
| backend API/repository 검색 | `Select-String -Path backend/app/repositories/ai_repo.py`, `backend/app/routers/v1/ai.py`, `backend/app/services/api_service.py`, `backend/app/repositories/prediction_repo.py` 키워드 검색 |
| frontend 표시 경로 검색 | `Select-String -Path frontend/src/components/StockView.tsx,frontend/src/components/TrainingView.tsx,frontend/src/components/Chart.tsx -Pattern ...` |
| 특정 코드 라인 범위 확인 | `Get-Content ai/composite_inference.py`, `ai/composite_policy_eval.py`, `ai/inference.py`, `frontend/src/components/StockView.tsx`, `frontend/src/components/TrainingView.tsx`의 필요한 라인 범위 출력 |
