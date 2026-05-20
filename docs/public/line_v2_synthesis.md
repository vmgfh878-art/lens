# CP154~CP175-LM 1D Line V2 실험 종합 보고서

작성일: 2026-05-15

## 0. 결론

1D line v2 계열의 최종 판정은 **제품 후보 없음, 연구 후보 유지**다.

다만 “완전 실패”로 정리하면 틀리다. 실험에서 확인한 핵심은 다음이다.

1. line의 수익성/순위 신호는 calendar split 기준에서도 살아 있었다.
2. line 단독으로 하방 위험까지 안정적으로 설명하는 것은 현재 feature와 목적함수로 부족했다.
3. rule overlay, regime head, conformal overlay, warning model, GBDT fallback 모두 제품형 warning 기준에는 못 닿았다.
4. 학습 차원에서 beta를 키우면 line 자체가 더 보수적으로 움직이고 false-safe가 줄었다.
5. 현재 가장 의미 있는 line 후보는 `beta_5`지만, 아직 save-run/DB attach 전 단계는 아니다.

따라서 다음 판단은 이렇다.

> 1D line은 버리지 않는다. 다만 Phase 1 제품 후보로 확정하려면 새 미래 데이터, beta narrow sweep, 그리고 line 해석 기준 재정의가 필요하다.

## 1. 원래 목표

초기 line 구상은 단순했다.

> 실제 가격 흐름보다 조금 낮게 따라가는 보수적 예측선이 있고, 가격이 그 선보다 위에 있으면 상승 기대, 선과 교차하거나 아래로 내려가면 주의/하락 기대를 해석할 수 있게 한다.

즉 line은 매수/매도 자동 신호가 아니라, 사용자가 가격 흐름을 해석할 때 참고하는 보수적 기준선이다.

하지만 CP148 이후 반복적으로 확인된 문제는 다음이었다.

| 문제 | 의미 |
|---|---|
| line alpha는 살아 있음 | IC, spread, fee 신호는 존재 |
| high-confidence false-safe | 강한 양수 예측을 했는데 실제로 크게 빠지는 사례 반복 |
| 문제 ticker 반복 | SATS, COHR, LITE, CIEN, CVNA, IBKR 등 |
| feature 추가 한계 | overextension/cross-sectional feature로 완전 해결 못 함 |
| seed amplifier | seed가 원인이 아니라 특정 blind spot을 증폭 |

이후 실험은 “line을 포기하지 않고, line 옆에 위험 해석을 붙이거나 line 자체를 더 보수적으로 만들 수 있는가”를 확인하는 방향으로 진행됐다.

## 2. 실험 흐름

### CP154: dual-head line v2

line score와 downside risk head를 분리했다.

| 결과 | 해석 |
|---|---|
| PatchTST는 line alpha 유지 | line 신호 자체는 죽지 않음 |
| risk recall 거의 0 | downside head가 위험을 못 켬 |
| TiDE/CNN-LSTM은 risk 신호는 일부 있으나 line alpha 약함 | 하나의 모델에서 두 목적을 동시에 만족하지 못함 |

결론:

> head만 분리한다고 line의 risk blind spot이 자동으로 해결되지는 않는다.

### CP158: 5-class regime head

downside risk를 단순 binary가 아니라 5단계 regime으로 바꿔 봤다.

| 결과 | 해석 |
|---|---|
| p32/s16 PatchTST line alpha 강함 | line 자체는 유지 |
| false-safe 감소는 약함 | test 기준 약한 보조 신호 |
| strong-down recall 낮음 | 강한 하락 색상으로 믿기 어려움 |
| p16/s8은 class 2 쏠림 | 색상이 안정적이라기보다 중앙으로 숨음 |

결론:

> regime head는 제품 위험 신호라기보다 약한 해석 보조 신호였다.

### CP159~CP160: conformal/statistical overlay 재판정

line을 그대로 두고 통계적 하방 여유를 붙이는 방식이었다.

| 결과 | 해석 |
|---|---|
| conformal lower는 coverage 자체는 가능 | 통계적 badge 후보 |
| false-safe 감소 작음 | line top decile 위험 제거에는 약함 |
| volatility baseline보다 약함 | 위험의 상당 부분은 단순 변동성으로 설명 |

결론:

> 통계 overlay는 설명은 쉽지만, 제품 warning으로 쓰기엔 위험 선별력이 부족했다.

### CP161~CP170: volatility/ATR/rule warning

volatility, ATR, self-normalized volatility, intraday range, drawdown 등을 rule로 조합했다.

중요한 발견은 두 가지다.

| 발견 | 의미 |
|---|---|
| ATR은 위험을 잘 잡음 | 하지만 고수익 후보도 많이 제거 |
| line 최상위 cohort에서는 다른 신호가 필요 | q5 영역에서는 intraday/microstructure 쪽이 더 맞음 |

CP169에서는 two-tier warning이 `STRONG_WARNING_BETA_CANDIDATE`까지 올라왔지만, 절대 개선폭이 작았다.

| 항목 | 값 |
|---|---:|
| no-warning baseline unwarned false-safe | 0.2066 |
| two-tier unwarned false-safe | 0.1990 |
| trust gap | 0.0328 |
| warning share | 0.2307 |
| spread retention | 0.8543 |
| fee retention | 0.8348 |

사용자 관점에서는 “warning이 꺼졌는데도 20% 가까이 빠진다”는 해석이 남았다. 이 지적은 맞다. 그래서 warning의 목표를 “의미 있는가”에서 “더 많이 잡는가”로 바꿔 CP170을 실행했다.

CP170 recall-first sweep:

| 항목 | best rule |
|---|---:|
| recall | 0.5202 |
| warning share | 0.5181 |
| spread retention | 0.6405 |
| fee retention | 0.5886 |
| random 대비 recall excess | 약 +0.0026 |

결론:

> 많이 잡기는 가능하지만, 그 순간 거의 랜덤에 가까운 넓은 경고가 되고 alpha를 크게 깎았다.

### CP171~CP174: warning model

규칙기반만으로 부족하니 별도 warning model을 붙였다. line_score는 입력하지 않고, line과 함께 평가만 했다.

CP171 smoke:

| 후보 | recall | warning share | spread retention | fee retention |
|---|---:|---:|---:|---:|
| TiDE risk-only | 0.5701 | 0.4846 | 1.4617 | 1.5284 |

TiDE가 규칙기반보다 가능성을 보였지만 warning share가 넓었다.

CP172 narrow:

| 항목 | 값 |
|---|---:|
| recall | 0.4574 |
| warning share | 0.3762 |
| precision | 0.2500 |
| spread retention | 1.3427 |
| fee retention | 1.3923 |

share를 줄이면 recall이 0.50 아래로 떨어졌다.

CP173 diagnostic:

| 항목 | 값 |
|---|---:|
| selected 후보 | `feature_combined_risk_pack` |
| recall | 0.4677 |
| warning share | 0.3689 |
| precision | 0.2606 |
| ROC-AUC | 0.5556 |

AUC 0.55대 plateau가 확인됐다. 즉 TiDE와 현재 feature 조합은 위험 ranking을 강하게 분리하지 못했다.

CP174 backbone/feature/target 진단:

| 항목 | 결과 |
|---|---|
| best diagnostic | RiskTabMLP |
| test recall | 0.7712 |
| warning share | 0.7278 |
| ROC-AUC | 0.5389 |
| 제품 후보 | 없음 |

RiskTabMLP는 recall은 높였지만 너무 많이 경고를 켜서 제품형 warning이 아니었다. GBDT는 LightGBM/XGBoost가 없어 sklearn fallback으로만 봤고, 해석 강도는 낮다.

결론:

> 현재 feature/target/backbone 조합에서는 warning model도 제품 기준에 못 닿았다.

### CP175: conservative line learning revisit

post-hoc warning이 plateau에 걸렸으므로 line 학습 자체를 다시 봤다. beta를 키우면 over-prediction penalty가 강해져 line이 더 보수적으로 내려간다.

핵심 결과:

| trial | false-safe | 감소 vs beta2 | IC retention | spread retention | fee retention | severe recall(line<0) | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| beta_2_baseline | 0.2126 | 0.00pp | 1.0000 | 1.0000 | 1.0000 | 0.6152 | 기준 |
| beta_5 | 0.1972 | 1.54pp | 0.9458 | 0.8409 | 0.8187 | 0.7921 | WARN_PARETO |
| beta_7 | 0.1929 | 1.97pp | 0.9079 | 0.8165 | 0.7908 | 0.8348 | WARN_PARETO |
| beta_7_plus_atr | 0.1893 | 2.33pp | 0.8397 | 0.6754 | 0.6300 | 0.8660 | FAIL |

결론:

> beta를 키우면 의도한 방향으로 움직인다. 하지만 beta/ATR만으로 제품 기준 3pp false-safe 감소와 alpha 보존을 동시에 만족하지는 못했다.

## 3. 지금까지 배운 점

### 3.1 line alpha는 버릴 신호가 아니다

calendar split 수리 후에도 p32/s16 계열 line alpha는 살아 있었다. CP164 기준 test IC는 약 0.0436, spread는 약 0.0079였다. CP175 beta_5도 IC retention 0.9458을 유지했다.

따라서 line을 버리는 결정은 아직 이르다.

### 3.2 risk를 line 하나에 모두 맡기면 안 된다

단일 line score 하나로 수익성 ranking과 하방 위험을 동시에 표현하려는 시도는 반복적으로 막혔다. 특히 high-confidence false-safe는 단순 feature 추가, head 추가, rule overlay로 쉽게 사라지지 않았다.

### 3.3 warning은 “넓게 켜기”가 되면 안 된다

CP170과 CP174에서 확인했다. recall만 올리면 warning share가 50~70%로 커지고, 이 경우 사용자에게는 주의 신호가 아니라 넓게 겁주는 신호가 된다.

### 3.4 beta 조정은 유효한 학습 축이다

post-hoc overlay보다 beta 조정이 더 직접적으로 line을 바꿨다. `beta_5`는 false-safe를 1.54pp 줄이면서 IC/spread/fee를 비교적 유지했다. 제품 후보는 아니지만, 다음 line v3의 출발점으로는 가장 정직하다.

### 3.5 현재 데이터만으로는 한계가 보인다

earnings, scheduled macro event, event proximity 같은 미래 일정형 데이터가 없으면 “왜 지금 위험한가”를 설명하는 데 한계가 있다. ATR/volatility는 위험을 잡지만, 원래 변동성 큰 고수익 후보도 같이 잘라낸다.

## 4. 최종 상태

| 축 | 최종 판단 |
|---|---|
| dual-head downside risk | 제품 후보 없음 |
| 5-class regime | 연구 후보, 제품 warning 아님 |
| conformal overlay | 연구 후보, 제품 warning 아님 |
| ATR/rule warning | 방향성은 있으나 제품 후보 아님 |
| TiDE warning model | 가능성은 있으나 AUC/precision/share 미달 |
| GBDT/MLP warning | 제품 후보 없음 |
| beta sweep | `beta_5`가 다음 line v3 출발점 |

현재 line v2는 save-run/DB attach 단계가 아니다.

## 5. 다음 계획

line v2를 이 상태로 제품에 붙이지 않는다. 다음 line 실험은 v3로 분리한다.

Line v3의 기본 방향:

| 항목 | 방향 |
|---|---|
| base line loss | beta 5 중심 narrow sweep |
| feature | earnings/macro event proximity 추가 |
| warning | 규칙기반은 baseline, 학습형 warning과 동시 비교 |
| split | calendar_aligned 유지 |
| source | yfinance 500 |
| product 기준 | line 단독 false-safe가 아니라 line+warning 해석 기준 |

stop-loss:

- beta 5 + 새 미래 데이터로도 false-safe가 3pp 이상 줄지 않으면 line warning Phase 1 후보는 보류한다.
- 그 경우 line은 수익성 보조선으로 남기고, 하방 위험의 primary는 band 또는 Phase 2 데이터 확장으로 넘긴다.

## 6. 보고서용 문장

최종 보고서에는 다음 문장으로 정리하는 것이 맞다.

> 1D line v2 실험은 제품 후보를 만들지는 못했지만, 실패 원인을 좁히는 데 성공했다. 단순 head 분리, regime 분류, 통계 overlay, 규칙기반 warning, 별도 warning model은 모두 현재 feature 체계에서 제품 기준에 미달했다. 반면 beta를 키운 보수적 line 학습은 false-safe를 줄이는 방향성을 보였으므로, 다음 단계는 beta 5 계열을 출발점으로 earnings/macro event 같은 사전 일정형 미래 데이터를 추가한 line v3 실험이다.

## 7. 참조 산출물

주요 참조 파일:

- `docs/cp154_lm_1d_line_v2_dual_head_plan.md`
- `docs/cp154_lm_1d_line_v2_stage2_smoke_report.md`
- `docs/cp158_lm_1d_line_regime_stage2_5_joint_signal_report.md`
- `docs/cp159_lm_1d_line_conformal_overlay_report.md`
- `docs/cp160_lm_1d_line_overlay_rejudgement_report.md`
- `docs/cp161_lm_1d_vol_anchored_risk_color_report.md`
- `docs/cp164_lm_calendar_split_line_risk_smoke_report.md`
- `docs/cp169_lm_1d_line_warning_trust_reframe_report.md`
- `docs/cp170_lm_1d_recall_first_warning_sweep_report.md`
- `docs/cp171_lm_1d_line_warning_stage2_smoke_report.md`
- `docs/cp172_lm_1d_tide_warning_narrow_sweep_report.md`
- `docs/cp173_lm_1d_tide_warning_diagnostic_report.md`
- `docs/cp174_lm_1d_warning_backbone_feature_target_diagnostic_report.md`
- `docs/cp175_lm_1d_conservative_line_learning_revisit_report.md`
