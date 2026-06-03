# Lens 운영 3모델 평가지표 — 수학 정의

> 운영에 실제 붙은 3개 모델(CP210 예측선 / CP153 1D 밴드 / CP178 1W 밴드)의 화면 노출 평가지표만 다룬다.
> 모든 산식은 `ai/evaluation.py` 실제 구현 기준(라인 번호 명시). 실험·진단용 보조지표(전체 109개)는 제외.
> 수식은 코드블록(등폭) 안에 일반 기호로 작성 — VSCode 기본 미리보기 포함 어디서나 보인다.
> (더 예쁘게 보려면 VSCode 확장 `Markdown Preview Enhanced` 또는 GitHub 업로드 시 LaTeX 렌더)

기호 안내:
- `Σ_t` = t에 대한 합(시그마), `·` = 곱, `|A|` = 집합 A의 원소 수
- `∩` = 교집합(둘 다 참), `≤ ≥` = 이하/이상, `1[조건]` = 조건 참이면 1 아니면 0(지시함수)
- `ρ(X,Y)` = X와 Y의 Spearman 순위상관, `rank(x)` = x의 순위

---

## 0. 공통 수학 기초

이 문서의 지표들이 쓰는 연산만 정의한다.

### 0.1 ReLU (한쪽 방향만 남기기)

```
ReLU(x) = max(0, x)

       = x   (x > 0 일 때)
       = 0   (x ≤ 0 일 때)
```

용도: "한쪽만" 남기고 반대쪽은 0으로 죽인다. 손실만 보려면 `ReLU(-r)` → 수익(r>0)은 0, 손실(r<0)은 크기 `|r|`로 남는다. 코드의 `torch.relu(...)` 또는 `torch.clamp(x, min=0)`.

### 0.2 Spearman 순위상관 (rank correlation) — 가장 많이 쓰임

두 변수 X, Y를 값이 아니라 순위(rank)로 바꾼 뒤 상관을 잰다.

```
1단계) 각 값을 순위로 변환
        rank(x_i) = x_i 가 전체에서 몇 번째로 큰가
2단계) 순위끼리 Pearson 상관

           Σ_i ( rank(x_i) − R̄x )( rank(y_i) − R̄y )
  ρ = ─────────────────────────────────────────────────────
       sqrt(Σ_i ( rank(x_i) − R̄x )²) · sqrt(Σ_i ( rank(y_i) − R̄y )²)

   (R̄x, R̄y = 순위의 평균)
```

동순위(tie)가 없으면 더 간단한 형태로도 같다:

```
  ρ = 1 − ( 6 · Σ_i d_i² ) / ( n · (n² − 1) )

  d_i = rank(x_i) − rank(y_i),   n = 표본 수
```

- 범위: `ρ ∈ [−1, +1]`. `+1`=순위 완전 일치, `0`=무관, `−1`=역순.
- Pearson과 차이: Pearson은 값의 선형관계, Spearman은 순위(단조)관계만 본다. 가격을 정확히 못 맞혀도 "큰 것끼리 큰 순서"만 맞으면 높게 나온다 → 금융 예측에 적합.
- 코드: `pandas.Series.corr(method="spearman")` (`evaluation.py:448, 461`의 `_spearman_corr`, `:364`의 IC).
- 안전장치: 유한값 2개 미만이거나 한쪽이 전부 같은 값(분산 0)이면 `None`.

### 0.3 분위수 (quantile)

```
Q(q) = 데이터의 하위 100·q % 지점 값
     = inf { x : F(x) ≥ q }     (F = 누적분포함수)
```

예: `Q(0.20)` = 하위 20% 경계(이보다 작은 게 전체의 20%). 코드 `torch.quantile(t, 0.20)`. 밴드 분위수 `q_low, q_high`와 tail 컷(하위 20%)에 쓰인다.

### 0.4 조건부 비율 (`_safe_rate`)

```
rate(A | B) = |A ∩ B| / |B|
            = ( Σ_i 1[ A_i 그리고 B_i ] ) / ( Σ_i 1[ B_i ] )
```

분모 `|B| = 0`이면 `None`. 코드 `evaluation.py:201-205`. recall / false-safe가 전부 이 형태.

### 0.5 평균·표준편차·정보비율·t-통계량

```
평균    x̄ = (1/n) · Σ_i x_i
표준편차 s  = sqrt( (1/(n−1)) · Σ_i (x_i − x̄)² )   (표본, ddof=1)
정보비율 IR = x̄ / s
t-통계량  t = x̄ / ( s / sqrt(n) )
```

코드 `evaluation.py:177-198`. IC 안정성(ic_ir, ic_t_stat)에 쓰이지만 화면엔 IC 평균만 노출.

---

## 1. 예측선 — CP210 (PatchTST F4 β=4, 5-seed ensemble)

기호: `s` = 모델 점수(`safe_line_score`, 수익률 단위), `r` = 실제 미래 수익률(h5 = 5거래일 누적). 날짜 t의 cross-section = 그날 500종목 묶음.

### 1.1 IC (순위 예측력) — 화면값 0.0325

```
IC_t = ρ( s_t , r_t )          # 날짜 t: 그날 종목들의 점수순위 vs 실제수익순위
IC   = (1/T) · Σ_t IC_t        # 모든 날짜의 IC 평균
```

`evaluation.py:364, 405, 417`. 도출: 점수 절대값이 아니라 종목 간 상대 순위가 실제와 얼마나 맞나. 시장 효율성 때문에 0.03~0.05면 약하지만 의미 있는 수준(Grinold-Kahn fundamental law).

### 1.2 severe recall (큰 하락 포착률) — 화면값 0.7727

```
severe_mask_i = 1[ r_i ≤ θ ]        # θ = horizon별 컷
recall = |{ i : s_i < 0 } ∩ { i : r_i ≤ θ }|  /  |{ i : r_i ≤ θ }|
```

θ = horizon별 컷(`evaluation.py:208-213`): `h≤5 → θ=0.05`, `h≤10 → 0.08`, 그 외 `0.12`. 예측선은 h5라 `θ=0.05`.
`evaluation.py:295, 311`. 도출: "실제로 심각 하락(`r≤θ`)한 구간 중, 모델이 미리 위험(`s<0`)으로 본 비율" = 재현율(recall). 비대칭 손실 β=4로 낙관을 억제해 이 값을 끌어올린 게 예측선의 정체성.
주: θ 컷의 구체 의미는 `r`(actual_h5_return) 정규화 방식에 의존. 산식은 코드 그대로.

### 1.3 false-safe (위험 오판율) — 화면값 0.2048

```
날짜별 tail:  tail_i^(t) = 1[ r_i^(t) ≤ Q_t(0.20) ]      # 그날 하위 20% = 큰 하락

false_safe = ( Σ_t |{ i : s_i^(t) ≥ 0 } ∩ tail^(t)| )  /  ( Σ_t |tail^(t)| )
```

`evaluation.py:240-253`. 날짜마다 하위 20%(큰 하락)를 고른 뒤, 그중 모델이 안전(`s≥0`)이라 본 것을 날짜 전체에 합산/비율. 도출: severe recall의 거울 — "위험을 놓친 비율". 낮을수록 좋음. CP175(0.197) 수준 유지가 목표.

### 1.4 spread (상위-하위 수익차) — 화면값 0.0055

```
spread_t = mean( 상위그룹 r ) − mean( 하위그룹 r )    # 날짜 t
spread   = (1/T) · Σ_t spread_t
```

`evaluation.py:376-378, 404`. 도출: 점수로 고른 좋은 종목이 나쁜 종목보다 실제로 얼마나 더 벌었나. 양수면 종목 구분력 존재(0 이하면 random과 동등 = 모델 최소 존재이유).

### 1.5 WF IC range (구간 안정성) — 화면값 0.0457

```
WF_IC_range = max(fold별 IC) − min(fold별 IC)        # walk-forward 4 fold
```

CP210 ensemble verification. 도출: 시장 국면이 바뀌어도 IC가 일관된가. 폭이 ship 기준 0.040 초과(0.0457) → CP210 `NO_SHIP` 사유. 낮을수록 안정.

---

## 2. AI 밴드 1D — CP153 (TiDE, q_low=0.15, q_high=0.85)

기호: `ℓ` = 밴드 하단(lower), `u` = 밴드 상단(upper), `a` = 실제값(밴드 비교 대상), `r` = 원시 수익률.

```
nominal = q_high − q_low = 0.85 − 0.15 = 0.70      # 목표 포함률
```

### 2.1 coverage_abs_error (포함률 오차) — 화면값 0.0099

```
covered_i = 1[ ℓ_i ≤ a_i ≤ u_i ]
empirical = (1/n) · Σ_i covered_i                  # 실제 밴드 안에 들어온 비율

coverage_abs_error = | empirical − nominal |
```

`evaluation.py:542, 551, 578-580`. 도출: "밴드가 목표(70%)만큼 실제로 덮었나"의 절대 오차. conformal 보정이 제대로 작동했는지의 핵심 척도. 0에 가까울수록 정확.

### 2.2 band_width_ic (밴드 폭 변동성 반응도) — 화면값 0.376

```
w_i = u_i − ℓ_i                                    # 밴드 폭
band_width_ic = ρ( max(0, w) , |r| )               # 폭 vs 실제 변동크기의 순위상관
```

`evaluation.py:545-546, 553, 599`. 도출: "밴드 폭이 실제 변동성과 같은 순위로 움직이나". 양수 = 변동성 큰 날 밴드가 넓어짐 = 위험 신호 기능. 밴드의 핵심 가치(정확도가 아니라 변동성 동조).

### 2.3 lower / upper breach rate (하단·상단 이탈률) — 화면값 0.1586 / 0.1513

```
lower_breach_rate = (1/n) · Σ_i 1[ a_i < ℓ_i ]     # 목표 = q_low = 0.15
upper_breach_rate = (1/n) · Σ_i 1[ a_i > u_i ]     # 목표 = 1 − q_high = 0.15
```

`evaluation.py:543-544, 581-584, 775`. 도출: 분위수 정의 자체의 검증 — q15로 학습한 하단이 정확하면 실제 이탈률도 0.15여야. 목표에서 멀수록 calibration 어긋남. lower+upper로 coverage 0.70의 좌우 대칭 점검.

### 2.4 downside_width_ic (하방 폭 손실 반응도) — 화면값 0.0866

```
downside_width_i    = ReLU( line_i − ℓ_i ) = max(0, line_i − ℓ_i)   # 기준선→하단 거리
downside_realized_i = ReLU( −r_i )         = max(0, −r_i)           # 손실 크기만(수익은 0)

downside_width_ic = ρ( downside_width , downside_realized )
```

`evaluation.py:547, 552, 554, 600`. 도출: 큰 손실 구간에서 밴드 아래쪽이 넓어지는지 = 하방 위험 반영. 양수면 위험 회피 도구로 기능.

---

## 3. AI 밴드 1W — CP178 (TiDE WFLOCK, q_low=0.10, q_high=0.90)

산식은 1D와 100% 동일. 분위수만 다르다:

```
nominal = q_high − q_low = 0.90 − 0.10 = 0.80
lower_breach 목표 = q_low = 0.10
upper_breach 목표 = 1 − q_high = 0.10
```

### 화면 노출 값
- `coverage_abs_error = 0.039` ( |empirical − 0.80|, 1D보다 큼 — 주봉 난이도 )
- `band_width_ic = 0.34` ( 1D 0.376보다 약함 — 주봉 표본 수 적음 )

### fold별 평가 (1W만 화면 노출)
1W는 walk-forward 3-fold이고 fold마다 분포가 달라 fold별 coverage/breach를 따로 본다.

```
coverage_fold_f = (1/|f|) · Σ_{i ∈ f} 1[ ℓ_i ≤ a_i ≤ u_i ]   # fold f 내부, 3 seed 평균
```

- `coverage_fold2 = 0.8002` (목표 0.80 정확 달성)
- `coverage_fold1 = 0.7462` (분포 이동 구간이라 미달 — 정직 노출)
- `lower_breach_fold2 = 0.0922` (목표 0.10 근접)
- `bootstrap 95% CI 폭 = ±0.003 ~ 0.005p` (같은 seed 내 모델 일관성)

bootstrap CI: fold/seed별 coverage를 재표집(resampling)해 95% 신뢰구간 폭을 잰 것. 좁을수록 추론 안정. CP216.2 통계 검정 bootstrap과 같은 원리.

---

## 4. 검증 결론 — "이게 다인가"

- `ai/evaluation.py`는 총 109개 지표를 정의한다(실험·진단·regime 분석·bucket 분해 등 포함).
- 운영 보고서(CP210/153/178 metrics.json)에는 그중 수십 개가 계산된다 (ic_ir, ic_t_stat, false_safe_negative/tail/severe_rate, high_confidence_*, width_bucket_*, interval_score 등).
- 이 문서의 12개(라인 5 + 1D 밴드 5 + 1W 밴드 fold 포함)는 그중 화면에 노출되고 운영 3모델 판정에 쓰인 핵심만 추린 것. 나머지는 실험 과정에서 세팅한 보조·진단 지표로, 운영 모델 평가의 결론에는 위 12개가 사용된다.

| 모델 | 화면 노출 핵심 지표 | 핵심 수학 도구 |
|---|---|---|
| 예측선 CP210 | IC, severe recall, false-safe, spread, WF IC range | Spearman + 조건부 비율 + 분위수 |
| 1D 밴드 CP153 | coverage_abs_error, band_width_ic, lower/upper breach, downside_width_ic | 포함률 카운트 + Spearman(폭 vs 변동성) + ReLU |
| 1W 밴드 CP178 | 위 + fold별 coverage/breach + bootstrap CI | 동일 산식(q10/q90) + fold 분해 |

한 줄: 예측선은 순위·재현율(점수 정확도 아님), 밴드는 포함률·변동성 동조(가격 예측 아님). 둘 다 "맞히기"가 아니라 "위험을 일관되게 신호하나"를 재도록 설계됐다.
