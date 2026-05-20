# CP204 — Band v2 Plan: Conformal Band with Concept-Aware Uncertainty

본 문서는 CP204 (Band v2) arc 의 reference plan 이다. v1 band 의 product/research 한계를 정리하고, v2 가 무엇을 목표로 하며 어떤 전략으로 달성하는지 명문화한다.

각 wave 의 dispatch 지시서는 별도이며, 본 문서는 의도 / 전략 / 평가 / baseline / concept / architecture 를 한 곳에서 꺼내볼 수 있도록 정리한다.

---

## 0. 배경 — 왜 v2 인가

### v1 결과 요약 (CP202 / CP202.1 / CP202.2)

- **CP202**: 1D / 1W band primary 가 학습 의도대로 작동 (coverage 정확) 단 stress regime 비대칭 + width 가 future severe 예측 약함 (AUC 0.51-0.52)
- **CP202.1**: 통계 baseline (BB / Historical Quantile / GARCH) 와 비교 결과
  - 정상 regime 우리 우위
  - Stress regime 에서 **GARCH(1,1) closed-form 과 동급** (deep 가치 marginal)
- **CP202.2**: Width 가 forward / backward vol correlation 비율 = **0.34** (1D), **0.38** (1W)
  - = **명확한 lagging**. 미래 vol 예측 거의 못 함

### v1 의 정직한 한계

1. **Lagging forecaster**: 과거 vol clustering 따라가는 수준. "변동성 커지기 전 미리 넓어짐" 없음
2. **Single-dimensional**: vol clustering 만 인코딩. 사건 / 섹터 / 매크로 위험은 무시
3. **No trust guarantee**: "90% coverage" 가 학습 의도일 뿐 수학적 보장 X
4. **Black box**: 왜 band 가 넓어졌는지 사용자에게 설명 불가
5. **No confidence signal**: 모델이 자신 없는 시점도 동일하게 표시. Over-trust 위험

### v2 의 정체성

> **Conformal-calibrated, multi-concept, uncertainty-aware band with selective output.**

학계 표현으로:
- Distribution-free coverage guarantee (수학적 trust)
- Aleatoric + epistemic uncertainty decomposition
- Event-aware + multi-signal feature stack
- Concept Bottleneck explainability
- Selective prediction (모르면 모른다고)

---

## 1. Product Intent — 사용자 frontend 에서 보는 것

### 표시되는 정보

```
종목: TSLA
오늘 Band: [-3.2%, +4.1%] (폭 7.3%, 평소 1.5배)
  ↑ Conformal-calibrated 90% coverage 보장

왜 넓어졌나 (Top 3 concept):
  1. earnings_imminent (3일 후 어닝) — 기여 +2.1%
  2. vix_term_inverted (9d > 3m) — 기여 +1.5%
  3. sector_stress (TECH 약세) — 기여 +0.9%

모델 신뢰도: ⭐⭐⭐⭐ (높음, epistemic 낮음)
```

또는 모델 자신 없는 경우:
```
종목: XYZ
오늘 Band: [표시 안 함 — 모델 신뢰 부족]
이유: epistemic uncertainty 임계 초과 (이런 패턴 학습 적게 봄)
```

### 사용자 기대 동작

- "Band 좁고 안정적이면 line 신뢰" — 좁은 band = 모델 자신 + 시장 안정
- "Band 넓으면 조심" — 넓은 band = vol 위험 + 사건 임박 등 명시적 reason
- "표시 안 됨 = 모델이 모르는 시점" — 강제 over-trust 방지
- "Top 3 concept 으로 의사결정 참고" — 모델 reasoning 가시화

---

## 2. v2 의 다섯 가지 기둥 (Pillars)

각 기둥은 **원하는 점** 과 **사용된 전략** 으로 명확히 분리.

### Pillar 1: Conformal Coverage Guarantee

**원하는 점:** "이 band 의 90% coverage 가 진짜 90% 임을 수학적으로 보장"

**사용된 전략:** Conformal Prediction (Vovk et al. 2005, Romano et al. 2019)

**구체 방법 (Split Conformal):**
1. 학습 데이터 분할: train / calibration / test
2. Train 으로 quantile model 학습 (현재 v1 처럼)
3. Calibration set 에서 prediction error 측정:
   ```
   s_i = max(predicted_q05_i - actual_i, actual_i - predicted_q95_i)
   ```
4. 모든 s_i 중 90% quantile = Q
5. Test 시 band 확장:
   ```
   conformal_band = [predicted_q05 - Q, predicted_q95 + Q]
   ```
6. 결과: **분포 가정 없이** 90% coverage 수학적 보장 (단 exchangeability 가정)

**진화 — Adaptive Conformal Inference (Gibbs & Candes 2021):**
- 분포 drift 발생 시 Q 자동 업데이트
- "이번 달 coverage 87% 면 Q 살짝 올림"
- Online learning 처럼 매월 재calibrate
- Lens 적용: 매월 calibration set 갱신

**진화 — Regime-Conditional Conformal:**
- VIX low/mid/high regime 별 Q 따로 계산
- "VIX > 25 시점엔 별도 Q 적용"
- 위기 regime 에서도 coverage 보장

**비용:** 학습 후 후처리. 코드 ~20 줄. 학습 부담 0.

**참고 논문:**
- Vovk, Gammerman, Shafer (2005) "Algorithmic Learning in a Random World" (원 저작)
- Romano, Patterson, Candes (2019) "Conformalized Quantile Regression"
- Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
- Angelopoulos & Bates (2023) "A Gentle Introduction to Conformal Prediction"

---

### Pillar 2: Deep Ensemble for Uncertainty Decomposition

**원하는 점:** "시장 자체 변동성 (aleatoric) 과 모델 무지 (epistemic) 를 분리 측정"

**사용된 전략:** Deep Ensemble (Lakshminarayanan et al. 2017)

**구체 방법:**
- 같은 backbone, 같은 데이터, **다른 random seed** 로 5 모델 학습
- Inference 시 5 모델 다 prediction → 평균과 분산 측정

```python
predictions = [model_i.predict(x) for i in range(5)]
# 각 prediction = (q05, q15, q50, q85, q95) tuple

mean_pred = mean(predictions, axis=0)  # 메인 예측
aleatoric = mean([m.predicted_std(x) for m in ensemble])  
# = quantile spread 의 평균 (시장 vol)
epistemic = std([m.predicted_mean(x) for m in ensemble])  
# = 모델 간 disagreement (모델 confidence)

total = sqrt(aleatoric**2 + epistemic**2)
```

**왜 작동:**
- Deep learning loss surface 는 non-convex
- 다른 seed = 다른 local minimum
- 학습 데이터로 잘 explained 되는 부분: 5 모델 동일 답
- 학습 데이터에 없는 부분 (OOD): 5 모델 다른 답
- → **모델 간 disagreement = epistemic uncertainty 자연 추정**

**Aleatoric vs Epistemic — 어원 + 의미:**

| 종류 | 어원 | 의미 | 줄이는 방법 |
|---|---|---|---|
| Aleatoric | 라틴어 alea = 주사위 | 세상 자체 무작위성 | **줄일 수 없음** |
| Epistemic | 그리스어 epistēmē = 지식 | 모델의 무지 | 더 많은 데이터 / 모델 개선 |

**Lens 적용:**
- Aleatoric 큰 종목: "이 종목 평소 vol 큼" (e.g., TSLA)
- Epistemic 큰 시점: "이런 patter 본 적 없음" (e.g., COVID 같은 regime)
- 다르게 표현: aleatoric = "시장 위험", epistemic = "모델 confidence"

**비용:**
- 학습 시간 5배 (real cost)
- Inference 5배 (parallelizable, 캐시 가능)
- 메모리 5배 (model state)

**저비용 대안 (필요 시):**
- MC Dropout (Gal 2016): dropout 켜둔 채 inference 여러 번
- SWAG (Maddox 2019): 학습 후반 weight 분산 활용
- 둘 다 단일 모델 학습 cost (5배 X), 단 epistemic 추정 정확도 낮음

**v2 선택:** Deep Ensemble 정석 (5배 비용 인정, v1 이미 배포돼 안정성 확보됐으니 v2 는 정확도 우선)

**참고 논문:**
- Lakshminarayanan, Pritzel, Blundell (2017) "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
- Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
- Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
- Lakshminarayanan recent review (2023)

---

### Pillar 3: Selective Prediction

**원하는 점:** "모델이 자신 있을 때만 사용자에게 표시. 모르면 명시적으로 모른다고."

**사용된 전략:** Selective Classification (Geifman & El-Yaniv 2017, El-Yaniv & Wiener 2010)

**구체 방법:**
1. Pillar 2 의 epistemic uncertainty 활용
2. Threshold 정함 (예: epistemic > τ → abstain)
3. Threshold 는 validation set 에서 결정:
   - 거부율 (abstention rate) 과 정확도 (accuracy on accepted) 의 trade-off
   - 학계 표준: risk-coverage curve

**선택 가능한 표시 정책:**
- **Hard abstain**: 표시 자체 안 함
- **Soft warning**: 표시하되 "낮은 신뢰도" 배지
- **Confidence gradient**: 별 1~5 개로 confidence 시각화

**Lens v2 추천:** 3-tier
```
epistemic < τ_low:    표시 + ⭐⭐⭐⭐⭐ (높은 신뢰)
τ_low ≤ ep < τ_high:  표시 + ⭐⭐ (낮은 신뢰, 참고만)
epistemic ≥ τ_high:   표시 안 함 ("이 종목 오늘 신뢰 부족")
```

**왜 user trust 의 핵심:**
- "100% 신뢰 못 줘서 사용자 안 씀" 문제의 정답
- = "100% 줄 수 있는 부분만 보여줌"
- 표시 안 한 부분은 모델이 모름 → 사용자 직접 판단
- 표시한 부분은 진짜 신뢰 가능 → product trust 상승

**참고 논문:**
- Geifman & El-Yaniv (2017) "Selective Classification for Deep Neural Networks"
- El-Yaniv & Wiener (2010) "On the Foundations of Noise-free Selective Classification"
- 최근: Selective forecasting in finance (Sun et al 2023)

---

### Pillar 4: Concept Bottleneck Models (CBM)

**원하는 점:** "Band 가 넓어진 이유를 사용자가 자연어로 이해 가능. 모델의 reasoning 이 frontend 에 직접 표시."

**사용된 전략:** Concept Bottleneck Models (Koh et al. 2020, Stanford) + Lens 확장

**핵심 발상:**

일반 모델:
```
입력 X → [블랙박스 신경망] → 출력 Y
```

CBM:
```
입력 X → [개념 예측층 C] → [최종 출력 Y]
        ↑ 사람이 정의한 개념
```

중간 layer 가 **literally** 사람이 이해할 수 있는 개념. 설명 = 모델의 실제 reasoning (post-hoc 근사 아님).

**왜 v1 의 post-hoc explanation 보다 우수:**

| | Post-hoc (SHAP, LIME, Attention) | CBM |
|---|---|---|
| 본질 | 블랙박스 모델 + 옆에서 추정 | 모델 자체의 mid-layer 가 개념 |
| 정확도 | 근사 | exact (model's real reasoning) |
| Frontend 표시 | "이 feature 가 아마 중요" | "이 개념이 이만큼 활성" |
| 사용자 수정 가능 | X | 가능 (concept intervention) |
| 학습 강제 | X | concept supervision 강제 |

**Lens v2 의 Concept 정의 (12개):**

Vol regime (3):
1. `vol_regime_high` — VIX > 25
2. `vol_regime_low` — VIX < 15
3. `vix_term_inverted` — VIX 9d > VIX 30d (backwardation)

Event proximity (3):
4. `earnings_imminent_5d` — 5 거래일 내 어닝
5. `macro_event_imminent_5d` — 5 거래일 내 FOMC / CPI / NFP / PPI
6. `opex_imminent_5d` — 5 거래일 내 옵션 만기

Recent event triggers (3):
7. `insider_sell_cluster_30d` — 30 일 내 insider sell 5+ (Form 4)
8. `eight_k_material_event_7d` — 7 일 내 8-K item 4.02 / 5.02 / 2.05
9. `post_earnings_window_3d` — 어닝 후 3 일 내

Cross-sectional stress (3):
10. `sector_stress` — 섹터 ETF -10% drawdown 활성
11. `cross_asset_stress` — 10y-2y yield 급변 + DXY momentum spike
12. `market_drawdown_active` — SPY 200d MA 하향 돌파

**학습 방식 (Semi-supervised CBM):**

각 concept 의 **ground truth label 은 rule-based** 로 생성:
- `vol_regime_high` GT = `1 if VIX > 25 else 0`
- `earnings_imminent_5d` GT = `1 if days_to_next_earnings <= 5 else 0`
- 등등

학습:
```python
loss = pinball_loss(predicted_q, actual)              # 메인 task
     + λ_concept * sum(BCE(predicted_C_i, GT_C_i))    # concept supervision
     + λ_forward * forward_vol_mse                    # leading 강제
```

**진화 — Probabilistic CBM (Kim et al. 2023):**
- 각 concept 에 confidence 까지 출력
- "earnings_imminent: 0.95 (high confidence)" 식
- v2 에 추가 검토

**진화 — Concept Intervention:**
- 사용자가 concept 값 직접 수정 가능
- 예: "earnings 가 사실 2 일 후" → concept 수정 → band 재예측
- v3 candidate

**Frontend 통합:**
- Concept activation top 3 자동 추출
- 각 concept 에 **사용자 친화 자연어 label** 매핑:
  ```
  vol_regime_high → "시장 변동성 높음"
  earnings_imminent_5d → "어닝 발표 임박"
  insider_sell_cluster_30d → "내부자 매도 다발"
  ```
- 표시: "Band 가 넓어진 이유: 어닝 발표 임박 + 시장 변동성 높음 + 섹터 약세"

**참고 논문:**
- Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang (2020) "Concept Bottleneck Models" (ICML)
- Espinosa Zarlenga et al. (2022) "Concept Embedding Models"
- Oikarinen et al. (2023) "Label-free Concept Bottleneck Models" (LLM auto-generates concepts)
- Kim et al. (2023) "Probabilistic Concept Bottleneck Models"
- Stammer et al. (2022) "Interactive Disentanglement"

---

### Pillar 5: Leading Volatility (not just lagging)

**원하는 점:** "변동성 커지기 전 band 가 미리 넓어짐. Lagging 이 아예 아닐 정도."

**사용된 전략:** Forward vol MSE loss + event-aware widening regularizer + leading feature stack

**구체 방법:**

**Loss 추가:**
```python
forward_vol_loss = MSE(predicted_band_width, future_realized_vol_5d)
# = "현재 width 가 미래 5d realized vol 을 잘 추정하도록 강제"

event_widening_loss = max(0, expected_widening - actual_widening) for scheduled events
# = "어닝/FOMC 직전엔 band 가 평소보다 넓어져야 함"

total_loss = pinball + λ_concept * concept_loss 
           + λ_forward * forward_vol_loss
           + λ_event * event_widening_loss
```

**Feature 추가 (leading 신호):**

기존 line cache features +
- **Form 4 features** (insider activity)
  - `insider_sell_count_30d`
  - `cluster_sell_flag`
  - `ceo_cfo_sell_flag`
  - 출처: SEC EDGAR 무료 XML
- **8-K features** (material events)
  - `eight_k_count_30d`
  - `item_4_02_flag_7d`, `item_5_02_flag_7d`, `item_2_05_flag_7d`
  - 출처: SEC EDGAR 무료 + FinBERT optional
- **VIX term structure dynamics**
  - `vix_term_spread_change_5d`
  - `vix_backwardation_persistence_10d`
  - 출처: 기존 line cache features 가공
- **Cross-asset stress index**
  - `yield_curve_inversion_intensity`
  - `dxy_momentum_zscore`
  - 출처: yfinance 추가
- **Realized vol change features**
  - `realized_vol_5d_change`
  - `realized_vol_20d_zscore`
  - 출처: 기존 OHLCV 가공

**왜 v1 이 lagging 이었나:**
- v1 loss = pinball 만
- 모델이 자연 수렴한 해 = "현재 vol regime 이 5 일 동안 유지" 가정 (vol clustering)
- Forward 강제 loss 없으니 lagging 학습이 가장 쉬운 길

**v2 가 leading 으로 학습되는 강제:**
- λ_forward 가 forward vol 직접 예측 압박
- λ_event 가 scheduled event 사전 widening 보상
- Feature stack 에 leading 신호 (Form 4, 8-K) 풍부
- → 모델이 lagging 만으로 loss 최소화 불가, leading 학습 자연 발생

**참고 논문:**
- 학계 직접 reference 약함 (이 design 은 Lens-specific)
- 관련: Carr & Wu (2009) "Variance Risk Premiums" (implied vol surface)
- 관련: Engle & Ghysels (2011) "MIDAS regressions with macro variables"

---

## 3. Architecture Stack

```
┌─────────────────────────────────────────────────┐
│ Input X (features + new leading features)        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Backbone (TFT / TiDE / PatchTST 비교)            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Concept Layer (12 concepts, supervised)          │
│ → predicted concept activations C_1 ... C_12     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Quantile Head (9 quantiles q05 ... q95)          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ × 5 Deep Ensemble (5 seeds)                      │
│ → ensemble mean + epistemic std                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Conformal Calibration Layer                      │
│ → coverage-guaranteed band                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Selective Output Decision                        │
│ → display / weak display / abstain               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Frontend Display                                  │
│ - Band (conformal)                               │
│ - Concept activation top 3                       │
│ - Confidence tier (1-5)                          │
│ - Aleatoric / Epistemic 분리 옵션 (advanced)    │
└─────────────────────────────────────────────────┘
```

---

## 4. Backbone 후보 (3개)

### TFT — Temporal Fusion Transformer (Lim et al. 2020, Google)
- Native quantile output (band 자연 구현)
- Variable selection network = feature importance per timestep (interpretability 무료)
- Multi-head attention = 시점/feature attention 시각화
- **CBM 과 자연 호환** (attention 이 concept activation 보강)
- 단점: 학습 시간 약간 큼

### TiDE — Time-series Dense Encoder (Das et al. 2023, Google)
- 현 v1 baseline
- 빠른 학습
- Quantile head 직접 추가 가능
- 단점: TFT 대비 interpretability 약함

### PatchTST — Patch Time Series Transformer (Nie et al. 2023, IBM)
- 최근 SOTA
- Patch 기반 input = 효율적
- Quantile head 직접 추가
- 단점: TFT 대비 interpretability 약함

### 비교 방식
- 같은 feature, 같은 loss, 같은 ensemble 구성
- 3 backbone × 5 seed = 15 model 학습
- 평가 metric 13 개로 graded 판정
- 최적 backbone 선정 후 Wave 3 sweep

---

## 5. Output Format

각 (ticker, date) 에 대해 출력:

```python
{
    # Quantile band (9 quantiles)
    "q05": float, "q10": float, "q20": float, "q35": float, "q50": float,
    "q65": float, "q80": float, "q90": float, "q95": float,
    
    # Conformal-adjusted band (final user-facing)
    "conformal_q05": float, "conformal_q95": float,
    
    # Concept activations (12 concepts, [0, 1])
    "concept_vol_regime_high": float,
    "concept_earnings_imminent_5d": float,
    "concept_insider_sell_cluster_30d": float,
    ... (12 total)
    
    # Top 3 contributing concepts (for frontend)
    "top_concepts": [
        {"name": "earnings_imminent_5d", "activation": 0.95, "label": "어닝 발표 임박"},
        ...
    ],
    
    # Uncertainty decomposition (from ensemble)
    "aleatoric": float,  # 시장 vol
    "epistemic": float,  # 모델 confidence
    
    # Selective output decision
    "confidence_tier": 1 | 2 | 3 | 4 | 5,
    "display": True | "weak" | False,
}
```

---

## 6. Training Strategy

### Total Loss

```python
total_loss = pinball_loss(predicted_q, actual)              # 메인
           + λ_concept * sum(BCE(predicted_C_i, GT_C_i))    # CBM
           + λ_forward * MSE(width, future_realized_vol_5d) # leading 강제
           + λ_event * event_widening_regularizer           # event 사전 wide
           + λ_asym * asymmetric_quantile_loss              # 하방 우세
```

**Hyperparam sweep 후보:**
- α (asymmetric quantile weight): 1.0 고정 (line v2 와 동일 convention)
- β (downside weight): sweep {2, 3, 5}
- λ_concept: sweep {0.1, 0.3, 1.0}
- λ_forward: sweep {0.1, 0.3, 1.0}
- λ_event: sweep {0.1, 0.5}
- λ_asym: sweep {0.0, 0.3, 1.0}

Smoke 단계에선 작은 sweep, full 단계에선 Optuna 탐색.

### Training Procedure

1. Train/Calib/Test split (calendar-aligned, CP163)
2. 각 backbone, 각 seed:
   - Train backbone + concept layer + quantile head with total loss
3. Conformal Calibration:
   - Hold-out calibration set 에서 nonconformity score 계산
   - 매월 재calibrate (Adaptive Conformal)
4. Selective threshold 결정:
   - Validation set 에서 risk-coverage curve
   - 3-tier threshold (high/mid/low confidence)

---

## 7. Evaluation Framework (13 metrics — 상세)

### Group A — Trust Core (신뢰도 핵심)

#### Metric 1: Conformal Coverage

**정의:** Conformal-calibrated band 의 실제 coverage 가 약속한 nominal (90%) 와 일치하는가

**수식:**
```python
conformal_coverage = mean(conformal_q05 <= actual <= conformal_q95)
nominal = 0.90
coverage_abs_error = abs(conformal_coverage - nominal)
```

**보조:**
- Regime-conditional coverage (VIX low / mid / high)
- Per-month coverage (drift 모니터)
- Per-ticker coverage

**의도:** 사용자에게 "90% 신뢰 가능" 약속이 진짜 지켜지는지 검증. 컷: abs_error ≤ 0.03 (3%p 이내).

---

#### Metric 2: Epistemic Separability

**정의:** Epistemic uncertainty 가 큰 시점이 실제로 모델이 어려워한 시점인가

**수식:**
```python
# 학습 데이터에 없는 패턴 = OOD-like
# 모델이 헷갈리는 시점 = prediction error 큰 시점
ep_sorted = sort by epistemic descending
high_ep_quintile = top 20% by epistemic
low_ep_quintile = bottom 20% by epistemic

epistemic_separability = 
    mean(|prediction_error|[high_ep_quintile]) /
    mean(|prediction_error|[low_ep_quintile])

# > 1.5: epistemic 이 진짜 어려운 시점 가리킴
# = 1.0: epistemic 과 어려움 무관 (epistemic 신호 약함)
```

**의도:** Ensemble disagreement 가 진짜 model confidence 와 일치하는지 검증. 컷: separability ≥ 1.5.

---

#### Metric 3: Selective Accuracy + Abstention Rate

**정의:** Selective threshold 적용 후, 표시한 시점의 정확도 / 거부한 시점의 비율

**수식:**
```python
# Tier 별 분리
tier_5_mask = (confidence_tier == 5)  # 가장 자신
tier_4_mask = (confidence_tier == 4)
tier_1_mask = (confidence_tier == 1)  # 표시 안 함

# 표시한 (tier ≥ 2) 케이스만의 coverage
displayed_coverage = mean(conformal_q05 <= actual <= conformal_q95 | displayed)

# 거부 비율
abstention_rate = mean(confidence_tier == 1)

# 거부한 케이스의 실제 어려움 (검증)
abstained_error = mean(|prediction_error|[abstained])
displayed_error = mean(|prediction_error|[displayed])
abstention_quality = abstained_error / displayed_error
# > 1.2: 거부한 케이스가 실제 어려움
```

**의도:** Selective output 의 정확성 + 거부 비율 적절성. 컷:
- displayed_coverage ≥ 0.92 (표시한 건 정말 신뢰)
- abstention_rate ≤ 0.20 (5 종목 중 1 종목 이하만 거부)
- abstention_quality ≥ 1.2 (거부한 케이스가 진짜 어려웠음)

---

### Group B — Vol / Widening Representation

#### Metric 4: Forward / Backward Width Correlation Ratio

**정의:** Width(T) 가 past vol 보다 future vol 과 더 강한 상관 (leading 정도)

**수식:**
```python
spearman_back = spearman(width_T, realized_vol[T-5:T-1])
spearman_fwd  = spearman(width_T, realized_vol[T+1:T+5])
lead_lag_ratio = spearman_fwd / spearman_back

# v1 baseline: 0.34 (1D)
# v2 목표: ≥ 1.0 (leading achieved)
```

**의도:** Lagging 거부 정량 검증. 컷: ratio ≥ 0.7 (v1 의 2 배 이상 개선).

---

#### Metric 5: Event-Anticipation Accuracy

**정의:** Scheduled event (어닝, FOMC) 5 일 전 band 가 평소보다 넓어졌나

**수식:**
```python
event_days = (days_to_next_earnings <= 5) | (days_to_next_fomc <= 5)
non_event_days = ~event_days

event_widening = mean(width[event_days]) / mean(width[non_event_days])
# > 1.2: event 직전 sustained widening
# = 1.0: event 무시 (lagging)
```

**의도:** v1 의 calendar feature 가 학습됐는지 + leading 강제 효과 검증. 컷: event_widening ≥ 1.15.

---

#### Metric 6: Width vs Realized Vol R²

**정의:** Band width 가 실제 realized vol 을 얼마나 잘 설명하는가

**수식:**
```python
r_squared = R²(width_T, realized_vol[T+1:T+5])
```

**의도:** 넓어진 게 진짜 vol 을 표현하는지. 컷: R² ≥ 0.4 (v1 ~0.06, 큰 개선 기대).

---

#### Metric 7: Asymmetric Coverage

**정의:** 하방 q05 와 상방 q95 의 coverage 가 product 의도 (하방 우세) 와 정합

**수식:**
```python
lower_coverage = mean(actual >= conformal_q05)  # target 0.95
upper_coverage = mean(actual <= conformal_q95)  # target 0.95
lower_upper_balance = lower_coverage - upper_coverage
# > 0: 하방 더 잘 막음 (Lens 의도)
# < 0: 상방 더 잘 막음 (반대)
```

**의도:** Lens product 가 stop-loss 위주니 하방 coverage 가 상방보다 우수해야. 컷: lower_upper_balance > 0.

---

### Group C — Explainability (CBM)

#### Metric 8: Concept Prediction Accuracy (per concept)

**정의:** 각 concept 의 model prediction 이 rule-based ground truth 와 일치하는가

**수식:**
```python
for each concept_i:
    accuracy_i = accuracy(predicted_C_i, GT_C_i)
    auc_i = roc_auc(predicted_C_i, GT_C_i)
```

**의도:** CBM 의 mid-layer 가 진짜 concept 을 학습했는지. 컷: 각 concept AUC ≥ 0.85 (rule-based 라 높아야 함).

---

#### Metric 9: Concept Attribution Stability

**정의:** 같은 종류의 widening event 에서 같은 concept 이 top 3 에 일관되게 나오는가

**수식:**
```python
# 예: 어닝 직전 widening event 모두 모아서
earnings_events = (days_to_next_earnings <= 5) & (width > p90_width)
top_3_concepts_per_event = extract_top_3(earnings_events)

# 'earnings_imminent_5d' 가 top 3 에 포함된 비율
expected_concept_consistency = mean('earnings_imminent_5d' in top_3 | earnings_events)
# > 0.9: 일관됨
```

**의도:** 같은 reason 의 widening 이 같은 concept 으로 설명되는지 (model reasoning 안정성). 컷: ≥ 0.85.

---

#### Metric 10: Concept-Width Correlation

**정의:** Concept activation 이 실제로 width 증가로 이어지는가 (CBM 의 효과 검증)

**수식:**
```python
for each concept_i:
    activated = predicted_C_i > 0.5
    width_with_concept = mean(width[activated])
    width_without_concept = mean(width[~activated])
    widening_effect_i = width_with_concept / width_without_concept
    # > 1.1: concept 이 실제로 band 넓힘
```

**의도:** Concept 이 frontend 표시용 뿐 아니라 모델 결정에 실제 기여. 컷: 12 concept 중 ≥ 8 개가 widening_effect ≥ 1.1.

---

### Group D — Robustness

#### Metric 11: Universe Consistency

**정의:** 평가지표 1, 2, 4 가 전 종목에서 균일하게 작동

**수식:**
```python
per_ticker_coverage = groupby(ticker).apply(empirical_coverage)
per_ticker_lead_lag = groupby(ticker).apply(lead_lag_ratio)

coverage_std = std(per_ticker_coverage)
coverage_worst = min(per_ticker_coverage)
lead_lag_std = std(per_ticker_lead_lag)
```

**의도:** "전 티커 신뢰" 직접 측정. 컷:
- coverage_std ≤ 0.10
- coverage_worst ≥ 0.75

---

#### Metric 12: Time Stability

**정의:** 평가지표 1, 2, 4 가 전 시점에서 균일하게 작동

**수식:**
```python
per_month_coverage = groupby(year_month).apply(empirical_coverage)
per_month_lead_lag = groupby(year_month).apply(lead_lag_ratio)

monthly_coverage_std = std(per_month_coverage)
months_below_target = sum(per_month_coverage < 0.85)
```

**의도:** "전 시점 신뢰" 직접 측정 + regime 별 안정성. 컷:
- monthly_coverage_std ≤ 0.05
- months_below_target ≤ 2 (전체 24 개월 중)

---

#### Metric 13: vs GARCH (B3) Baseline

**정의:** v2 가 classical GARCH(1,1) 대비 명확히 우수한가 (deep 가치 정량화)

**수식:**
```python
v2_metric_improvement = (v2_metric - garch_metric) / garch_metric * 100
# 모든 핵심 metric (1-7) 에 대해 측정
```

**의도:** v1 의 marginal-over-baseline 문제 해결 검증. 컷: 메인 KPI (1-7) 중 ≥ 5 개가 GARCH 대비 ≥ 15% 개선.

---

## 8. Baselines

| ID | 정의 | 역할 |
|---|---|---|
| **B1** | Bollinger Bands (SMA(20) ± k·std(20)) | 가장 단순 통계 floor |
| **B2** | GARCH(1,1) closed-form | Strong classical floor (CP202.1 에서 검증) |
| **B3** | Lens v1 band (TiDE quantile frozen) | 현 best, 개선 대상 |

Stage 0.5 에서 모든 baseline 에 13 metric 측정 → Stage 0.6 에서 컷 anchor 잡음.

---

## 9. Stage 구조 (Wave 분배)

| Stage | 내용 | Wave |
|---|---|---|
| 0 | 계약 잠금 (Plan 인용, metric 코드, baseline 코드, concept rule 정의) | 1 |
| 0.5 | Baseline pilot (B1/B2/B3 학습/평가, 13 metric 측정) | 1 |
| 0.6 | 컷 확정 (baseline + 마진) | 1 |
| 1 | Smoke (3 backbone × 5 seed = 15 model, 1 fold) | 2 |
| 2 | Sweep (Stage 1 살아남은 후보 + λ tuning) | 3 |
| 3 | Conformal calibration 설정 | 3 |
| 4 | Selective threshold 결정 (validation risk-coverage curve) | 3 |
| 5 | Walk-forward (multi-fold) | 4 |
| 6 | 판정 + frontend 통합 spec | 5 |
| 7 | Closure + 1W 확장 결정 | 5 |

---

## 10. Out of Scope (본 CP 미포함)

- Options data 활용 (free 한계, Phase 3)
- 1W timeframe (1D 검증 후 동일 framework 복사)
- 2008 위기 stress test (frontend 출시 후 개인 검증)
- Foundation model (Chronos, TimesFM) — Phase 3 검토
- LLM-generated concept (Label-free CBM) — Phase 3

---

## 11. Display 시점 정렬

- 차트 표시: 오늘 line = T−4 (4 거래일 전) 예측
- h5 horizon, delay-aligned (line v2 와 동일)
- Concept activation 도 같은 T−4 시점 기준
- 매일 누적

---

## 12. Frontend 통합 spec

### 표시 요소
- Band: conformal-adjusted [q05, q95]
- Center line: conformal-adjusted q50 (or 별도 line v2)
- Concept top 3 자연어 텍스트
- Confidence tier 시각화 (별 1-5 또는 색상)
- (Advanced) Aleatoric / Epistemic 분리 옵션 (별도 토글)

### Concept-자연어 매핑

| Concept | Frontend 표시 |
|---|---|
| vol_regime_high | 시장 변동성 높음 |
| vol_regime_low | 시장 안정 |
| vix_term_inverted | VIX 단기 역전 (스트레스) |
| earnings_imminent_5d | 어닝 발표 임박 (X일 후) |
| macro_event_imminent_5d | 매크로 이벤트 임박 |
| opex_imminent_5d | 옵션 만기 임박 |
| insider_sell_cluster_30d | 내부자 매도 다발 |
| eight_k_material_event_7d | 중대 사건 보고 (회계/경영진/구조조정) |
| post_earnings_window_3d | 어닝 직후 변동 구간 |
| sector_stress | 섹터 약세 |
| cross_asset_stress | 매크로 자산 변동 |
| market_drawdown_active | 시장 약세 진행 중 |

---

## 13. Phase 2+ Candidates (CP204 종료 후)

- 1W timeframe 확장
- 2008 stress test
- Foundation model 도입 검토 (Chronos, TimesFM)
- LLM-generated concept (자동 concept 발견)
- Probabilistic CBM (concept uncertainty)
- Interactive CBM (사용자 concept intervention)
- Options data 통합 (유료 검토)

---

## Appendix A — 참고 논문 정리

### Conformal Prediction
- Vovk, Gammerman, Shafer (2005) "Algorithmic Learning in a Random World"
- Romano, Patterson, Candes (2019) "Conformalized Quantile Regression" 
- Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
- Angelopoulos & Bates (2023) "A Gentle Introduction to Conformal Prediction"

### Uncertainty Decomposition
- Lakshminarayanan, Pritzel, Blundell (2017) "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
- Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
- Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning?"
- Maddox et al. (2019) "SWAG: Stochastic Weight Averaging Gaussian"

### Selective Prediction
- Geifman & El-Yaniv (2017) "Selective Classification for Deep Neural Networks"
- El-Yaniv & Wiener (2010) "On the Foundations of Noise-free Selective Classification"

### Concept Bottleneck Models
- Koh et al. (2020) "Concept Bottleneck Models" ICML
- Espinosa Zarlenga et al. (2022) "Concept Embedding Models"
- Oikarinen et al. (2023) "Label-free Concept Bottleneck Models"
- Kim et al. (2023) "Probabilistic Concept Bottleneck Models"

### Backbone
- Lim et al. (2020) "Temporal Fusion Transformers" Google
- Das et al. (2023) "Long-term Forecasting with TiDE" Google
- Nie et al. (2023) "PatchTST" IBM

### Vol Forecasting Foundation
- Engle (1982) "ARCH"
- Bollerslev (1986) "GARCH"
- Corsi (2009) "HAR"
- Bollerslev & Patton (2016) "HARQ"

---

## Appendix B — Concept-Rule 매핑 상세

| Concept | Rule (Ground Truth) | Data Source |
|---|---|---|
| vol_regime_high | VIX > 25 | line cache (vix30d_close) |
| vol_regime_low | VIX < 15 | line cache |
| vix_term_inverted | VIX 9d > VIX 30d | line cache (vix_9d_minus_30d > 0) |
| earnings_imminent_5d | days_to_next_earnings <= 5 | line cache |
| macro_event_imminent_5d | days_to_next_(cpi/fomc/nfp/ppi) <= 5 | line cache |
| opex_imminent_5d | days_to_next_opex <= 5 | line cache |
| insider_sell_cluster_30d | Form 4 sell count (last 30d) >= 5 | SEC EDGAR (신규 수집) |
| eight_k_material_event_7d | 8-K item 4.02/5.02/2.05 (last 7d) | SEC EDGAR (신규 수집) |
| post_earnings_window_3d | days_since_earnings <= 3 | line cache (post_earnings_window_3d) |
| sector_stress | sector ETF drawdown_20d <= -0.10 | line cache (sector_*) |
| cross_asset_stress | (yield_curve_inversion = 1) AND (dxy_return_5d zscore > 1.5) | line cache + 추가 가공 |
| market_drawdown_active | SPY price < SMA(200) | yfinance (신규 가공) |

---

## Appendix C — Stage 별 학습 비용 추정

- Stage 0.5 (Baseline pilot): GARCH closed-form 빠름, B3 reload 만, ~1일
- Stage 1 (Smoke): 3 backbone × 5 seed × 1 fold = 15 학습, 각 ~30분-1시간 → 약 8-15시간
- Stage 2 (Sweep): 선정 후보 × λ sweep × 5 seed ≈ 30-50 학습 → 약 20-40시간
- Stage 3 (Conformal): 학습 0회, calibration calculation ~몇 분
- Stage 4 (Selective threshold): validation 분석 ~몇 분
- Stage 5 (Walk-forward): 선정 후보 × 5 fold × 5 seed ≈ 100-150 학습 → 약 60-100시간
- 총 학습 시간 추정: 약 100-160 GPU 시간

---

본 문서는 CP204 Band v2 의 전체 reference. 각 wave dispatch 는 본 문서 인용으로 시작.
