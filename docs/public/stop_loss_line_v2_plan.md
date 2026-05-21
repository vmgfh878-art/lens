# CP203 — Conservative Stop-Loss Line v2 (1D) Plan

본 문서는 CP203 전체 arc 의 reference plan 이다. 각 wave 의 dispatch 지시서는 별도이며, 본 문서는 의도/평가지표/베이스라인/컷 철학을 한 곳에서 꺼내볼 수 있도록 정리한다.

---

## 0. 배경

CP148 ~ CP198 line 실험에서 다음 문제가 누적 확인됨:
- line 학습은 일부 CP에서 asymmetric loss (CP175 beta5) 적용됐으나, **평가가 product 의도를 직접 측정한 적이 없음**
- 평가는 top decile severe rate / ranking IC 위주 (학계 quant 표준)
- 결과: "보수성"이 학습 loss 에 반영돼도 그것이 product (stop-loss 기준선) 으로 작동하는지 검증 불가
- "β가 적용은 됐는데 평가가 안 됨" 상태

CP203 은 이 gap 을 한 번에 정리한다.

---

## 1. Product 의도

> **Line = h5 보수적 예측선 = 사용자 stop-loss 기준선**

핵심 동작:
- 실제 ≥ 예측: 정상 (대부분의 시점에 이래야)
- 실제 < 예측: 매도 트리거 신호
- 실제 < 예측 인데 actual 이 양수: false break (line 이 너무 아래라 의미 없이 트리거 — 줄여야 함)
- 실제 < 예측 인데 actual 이 음수: 정상 break (stop-loss 가치 입증)

품질 조건:
- 전 티커 일관: 평균만 좋고 일부 종목 부실 X
- 전 시점 일관: 평상시만 좋고 위기 시 무력 X
- 절대값 의미: score=0.03 이면 "약 3% 보수적 예측" 으로 해석 가능

---

## 2. Output 형식

- 단일 값 per (ticker, date)
- 의미: h5 수익률의 보수적 추정 (분포 중심보다 아래쪽 자리)
- 학습 loss: **Asymmetric MSE**

```python
def asymmetric_mse(y_true, y_pred, alpha=1.0, beta):
    diff = y_true - y_pred
    return mean(where(diff > 0, alpha * diff**2, beta * diff**2))
```

- α = 1.0 고정 (up-side weight)
- β sweep (down-side weight): {2, 3, 5}
- β 클수록 over-prediction penalty 강 → 더 보수적
- Display 시점 정렬: 오늘 표시 line = T−4 (4 거래일 전) 예측 (h5 = T−4 → T+1)

---

## 3. Out of Scope (본 CP 미포함)

- Warning detector (보류, line 단독 검증 우선)
- 외부 데이터 (Form 4, 8-K) — Phase 2
- 2008 위기 stress test — frontend 완성 후 개인 검증
- 1W timeframe — 본 CP 검증 후 동일 framework 복사 적용
- Band 재학습 — 별도 검증 트랙

---

## 4. 베이스라인 (3종)

평가 컷의 기준점. Stage 0.5 에서 학습/평가하여 측정.

| ID | 정의 | 역할 | 다음 단계와의 gap 의미 |
|---|---|---|---|
| **B1** | 종목별 trailing 60d mean return | 통계 floor (학습 0회) | B1→B2 gap = "Multivariate + asymmetric loss" 가치 |
| **B2** | 선형 회귀, line cache 전체 feature, asymmetric MSE α=1 β=3 고정 | ML floor (선형 + 같은 loss) | B2→B3 gap = "Deep architecture" 가치 |
| **B3** | CP164 frozen line + CP175 beta5 frozen line | 기존 best (deep, frozen) | B3→CP203 후보 gap = "새 design (loss + eval 재정렬)" 가치 |

**B2 디테일:**
- 모델: `sklearn.linear_model` 또는 PyTorch 1-layer linear
- 입력: line cache 전체 feature
- Loss: asymmetric MSE α=1, β=3 (sweep 중간값 고정, fair comparison)
- Train/val/test split: 동일 (calendar-aligned, CP163)
- 학습 시간: fold당 분 단위

---

## 5. 평가지표 (9 개)

### 5.1 Empirical Coverage [메인 KPI]

**정의:** 실제 h5 수익률이 예측 line 보다 큰 비율

```python
coverage = mean(actual_h5_return > predicted_line)
```

**의도:** 보수성 기본 검증. 0.70 부근이 product target.

**범위 해석:**
- 0.50: 평균 예측 (대칭)
- 0.70: 적절한 보수 (target)
- 0.95: 너무 보수적 (신호 가치 약함)

**컷:** Stage 0.6 에서 베이스라인 측정 후 확정. 일반 가이드:
- target ± margin 안에 들면 채택 후보 (target=0.70, margin=0.05 정도)
- 단 graded — 약간 벗어나도 다른 지표 강하면 채택 가능

**Sub-metrics:**
- per-ticker coverage 분포 → Universe Consistency 와 연동
- per-month coverage → Time Stability 와 연동
- VIX regime 별 coverage

---

### 5.2 Break Composition [메인 KPI]

**정의:** Break event (actual < predicted) 발생 시 actual 의 부호/분포

**주력 sub-metric (primary):**
```python
break_positivity_rate = mean(actual_h5_return > 0 | actual < predicted)
# 양수 break = false break. 낮을수록 좋음
```

**보조 sub-metrics:**
```python
break_severity_mean = mean(actual_h5_return | break event)
# 음수면 OK, 양수면 line 자체 문제. 깊을수록 강한 신호

break_severity_median = median(...)  # outlier robust
break_count = sum(break events)       # sample size
```

**의도:** Break 가 "진짜 stop-loss 가치 있는 신호" 인지 검증. 양수 break = 의미 없이 트리거된 false signal.

**컷:** Stage 0.6 확정. 일반 가이드:
- break_positivity_rate ≤ B1 baseline − margin (예: 0.05)
- break_severity_mean < 0 (양수면 자동 FAIL)
- break_count ≥ 100 (sample size 확보)

**Sub-metrics:**
- per-ticker break_positivity_rate 분포
- per-month break_positivity_rate
- regime 별 break severity

---

### 5.3 Universe Consistency [메인 KPI]

**정의:** 평가지표 1, 2 가 전 종목에서 균일하게 작동하는가

```python
per_ticker_coverage = groupby(ticker).apply(coverage)
per_ticker_break_pos = groupby(ticker).apply(break_positivity_rate)

coverage_std = std(per_ticker_coverage)
coverage_p10_p90 = p90 - p10
coverage_below_threshold = mean(per_ticker_coverage < 0.55)
worst_ticker_coverage = min(per_ticker_coverage)
```

**의도:** "전 티커에서 높은 신뢰도" 직접 측정. 평균이 좋아도 분산 크면 일부 종목엔 무용.

**컷:** Stage 0.6 확정. 가이드:
- coverage_std ≤ B1 coverage_std (통계 베이스보다 일관)
- coverage_below_threshold 비율 낮게 (예: 10% 이내)

**Sub-metrics:**
- 섹터별 평균/std
- worst-ticker case 분석

---

### 5.4 Time Stability [메인 KPI]

**정의:** 평가지표 1, 2 가 전 시점에서 균일하게 작동하는가

```python
per_month_coverage = groupby(year_month).apply(coverage)
per_month_break_pos = groupby(year_month).apply(break_positivity_rate)

monthly_coverage_std = std(per_month_coverage)
monthly_coverage_range = max - min
months_below_target = sum(per_month_coverage < target - 0.10)

# regime
coverage_by_vix_regime = groupby(vix_regime).apply(coverage)
```

**의도:** "전 범위에서 높은 신뢰도" 직접 측정. 평상시만 잘 작동하고 위기엔 무력하면 product 위험.

**컷:** Stage 0.6 확정. 가이드:
- monthly_coverage_std ≤ B1 monthly_coverage_std
- months_below_target 베이스라인보다 적게
- regime 별 coverage 격차 좁게

**Sub-metrics:**
- VIX bin (low <15, mid 15-25, high >25) 별 coverage
- drawdown -10% 이상 기간 coverage
- 학습 vs test 기간 coverage drift

---

### 5.5 Calibration [메인 KPI, **컷 strict 가능**]

**정의:** 예측선 절대값이 실제 분포와 일치하는가 — 사용자가 score 0.03 보고 "약 3% 의미" 로 해석 가능한지

```python
# Decile-wise calibration
predicted_deciles = qcut(predicted_line, 10)
calibration_table = groupby(predicted_deciles).agg(
    mean_predicted = mean(predicted_line),
    mean_actual = mean(actual_h5_return)
)

# Expected Calibration Error
ece = mean(abs(mean_predicted - mean_actual)) per decile
```

**의도:** Score 절대값 의미. **Frontend 가장 직접적으로 보이는 정보**. 다른 지표보다 strict 컷 OK.

**보수성 조건 (구조적):**
- 모든 decile 에서 mean_predicted < mean_actual (보수성 visual proof)

**컷:** Stage 0.6 확정. **이 지표만 베이스라인보다 strict 컷 허용**:
- ECE ≤ B3 ECE × 0.7 (즉 30% 더 좋아야)
- 단 도달 불가능한 수준은 아님 — 베이스라인 측정 결과 보고 조정

**Sub-metrics:**
- decile plot (시각화 필수)
- monotonicity check (predicted 분위 ↑ → actual 분위 ↑)
- 종목별 ECE → Universe Consistency 와 연결
- regime 별 ECE

---

### 5.6 Pinball Loss at q=0.30 [검증 KPI]

**정의:** 예측선을 "q=0.30 quantile predictor" 로 해석 시 정확도 — coverage target 0.70 에 정합

```python
def pinball_loss(actual, predicted, tau=0.30):
    diff = actual - predicted
    return mean(where(diff >= 0, tau * diff, (tau - 1) * diff))
```

**의도:** β 의도가 결과적으로 q=0.30 quantile 역할을 잘 하는지 학계 표준 metric 으로 외부 검증. Quantile regression 직접 학습 모델과 공정 비교 가능성 확보.

**컷:** Stage 0.6 확정. 가이드:
- pinball_loss ≤ B1 pinball_loss
- 절대값 컷 없음, 상대 비교

**Sub-metrics:**
- Multiple quantile (q=0.10, 0.20, 0.30, 0.40, 0.50) pinball loss
- 어느 β 가 어느 q 를 가장 잘 잡는지 매핑

---

### 5.7 Asymmetry of Error [검증 KPI]

**정의:** β 의도가 학습에 실제 반영됐는지 직접 검증 — over-pred vs under-pred 비대칭

```python
errors = actual_h5 - predicted_line

over_pred_mask = errors < 0   # 예측 너무 높음 (위험)
under_pred_mask = errors > 0  # 예측 너무 낮음 (보수)

under_to_over_ratio = sum(under_pred_mask) / sum(over_pred_mask)
mean_over_magnitude = mean(abs(errors[over_pred_mask]))
mean_under_magnitude = mean(abs(errors[under_pred_mask]))
magnitude_asymmetry = mean_under_magnitude / mean_over_magnitude
```

**의도:** β=5 학습이 실제로 over-prediction 줄였는지 structural proof.

**컷:** Stage 0.6 확정. 가이드:
- under_to_over_ratio > 1.0 (대칭보다 보수 쪽)
- magnitude_asymmetry > 1.0 (아래 머물 때 더 여유)
- 둘 다 충족 강함 / 하나만 충족 부분 인정

**Sub-metrics:**
- error histogram
- 종목/시점/regime 별 ratio

---

### 5.8 Magnitude Error (MAE / RMSE) [견제 KPI]

**정의:** 예측이 실제와 얼마나 떨어져 있는가 — 보수성과 정확도 트레이드오프 견제

```python
mae = mean(abs(actual_h5 - predicted_line))
rmse = sqrt(mean((actual_h5 - predicted_line) ** 2))
```

**의도:** 보수성 강해도 MAE/RMSE 가 베이스라인 대비 너무 나쁘면 "신호 아니라 노이즈" 의심. 정확도 floor 견제.

**컷:** Stage 0.6 확정. 가이드:
- MAE ≤ B1 MAE × 1.3 (30% 까지 여유)
- RMSE ≤ B1 RMSE × 1.3
- 2배 이상 차이 시 자동 FAIL

**Sub-metrics:**
- per-ticker MAE 분포
- per-month MAE drift
- predicted vs actual scatter

---

### 5.9 Ranking IC [호환 KPI]

**정의:** 종목 간 순위 정합성 — 기존 CP148~194 평가와 호환

```python
per_day_ic = groupby(date).apply(
    spearman_corr(predicted_line, actual_h5_return)
)
ranking_ic_mean = mean(per_day_ic)
ranking_ic_ir = mean / std
```

**의도:** 기존 평가 호환용 보조 sanity check. 채택/탈락 직접 영향 X.

**컷:** Stage 0.6 확정. 가이드:
- B3 IC 의 70% 이상 유지

**Sub-metrics:**
- 일별 IC 분포
- top decile IC

---

## 6. 평가 컷 철학

**모든 지표 공통:**
1. **Tolerant** — black-and-white 금지. 한 지표 미달도 graded 판정
2. **Baseline-relative** — 절대 임계값 X, 베이스라인 + 합리적 마진
3. **현실적** — 도달 불가능한 컷 X. 학습이 자연스럽게 도달하는 범위 안
4. **장점 우선** — 한 지표 강하면 다른 지표 약해도 채택 가능

**Baseline 우선순위:**
- 메인 컷 anchor: **B2 (선형 ML)** — "deep architecture 정당화" 최소 조건
- 추가 reference: B1 (통계 floor), B3 (기존 best)
- 메인 KPI 후보가 B2 를 못 넘으면 FAIL (선형으로도 충분 = deep 의미 없음)
- B3 까지 넘으면 "기존 best 개선" 입증

**예외 — 평가지표 5 (Calibration):**
- Frontend 가장 직접적으로 보이는 정보
- 다른 지표보다 strict 컷 허용 (B3 의 70% 수준)
- 단 베이스라인 측정 결과 보고 조정

**판정 형식 (graded):**
- PASS_CP203 — 메인 KPI 1~5 다 통과 + 검증 KPI 6~7 충족
- WARN_CP203 — 메인 KPI 1~5 중 1개 borderline 또는 검증 KPI 약함
- FAIL_CP203 — 메인 KPI 2개 이상 미달 또는 견제 KPI 8 자동 FAIL

---

## 7. Stage 구조 (wave 단위)

| Stage | 내용 | Wave |
|---|---|---|
| 0 | 계약 잠금 (Plan 인용, 평가지표 코드, 베이스라인 코드) | 1 |
| 0.5 | 베이스라인 (B1/B2/B3) 학습/평가 | 1 |
| 0.6 | 컷 확정 (베이스라인 + 마진) | 1 |
| 1 | Smoke (3 backbone × 3 β = 9 run) | 2 |
| 2 | 작은 sweep (Stage 1 살아남은 후보) | 2 |
| 3 | Seed stability | 3 |
| 4 | 시점 stability | 3 |
| 5 | Walk-forward (multi-fold) | 4 |
| 6 | 판정 + closure | 4 |

각 wave 끝나면 다음 wave 발주 여부 결정. 중간 closure 가능.

---

## 8. 고정 제약

- Horizon: h5 (5 trading days)
- Universe: 500 ticker yfinance, 11년 historical (2015~2026, 2008 미포함)
- Backbone 후보: PatchTST, TiDE, CNN-LSTM
- Walk-forward: calendar-aligned split (CP163 spec)
- Coverage target: 0.70 (조정 여지 있음, Stage 0.6 후 재확정 가능)
- 학습 환경: 5060 Ti sm_120 cu128, KMP/TORCHDYNAMO 폴백, num_workers=0
- DB write 없음, save-run 없음, live fetch 없음 (각 wave dispatch 에서 명시)

---

## 9. Display 시점 정렬

- 오늘 차트에 표시되는 line 값 = T−4 거래일 전에 만든 예측
- h5 horizon = T−4 → T+1 (5 trading days)
- 매일 새 예측 누적, 표시는 5일 지연

---

## 10. CP203 종료 후 (Phase 2 candidates)

- Frontend 통합 + 실서비스 트리거
- 2008 위기 stress test (개인 검증, 부담 없이)
- 1W timeframe 동일 framework 복사 적용
- Warning detector 재검토 (line 부족분 식별 후)
- 외부 데이터 (Form 4, 8-K) line feature enrichment

---

## Appendix A. 메트릭 분류

| 분류 | 지표 | 역할 |
|---|---|---|
| 메인 KPI | 1, 2, 3, 4, 5 | 채택 결정 |
| 검증 KPI | 6, 7 | β 의도 작동 증거 |
| 견제 KPI | 8 | 트레이드오프 monitor |
| 호환 KPI | 9 | 이전 평가 비교 |

## Appendix B. 베이스라인 데이터 path

- B1: raw OHLCV 에서 trailing 60d mean 계산 (학습 0회)
- B2: 선형 회귀 학습 — input 은 기존 line cache feature 전체, β=3 고정
- B3_A: `C:\Users\user\lens\data\tmp\cp194r2_safe_line_sets\line_a_*.parquet` (CP164)
- B3_B: `C:\Users\user\lens\data\tmp\cp194r2_safe_line_sets\line_b_*.parquet` (CP175 beta5)
