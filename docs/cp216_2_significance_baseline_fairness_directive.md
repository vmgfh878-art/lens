# CP216.2 통계 검정 베이스라인 공정성 + regime 보강 (Directive)

## 한 줄

CP216 결과의 3가지 약점 — (A) GARCH baseline in-sample 누설 (B) 1D asof 1년 short → regime 표본 부족 GW NaN (C) 1W baseline 가격 100 ticker subset — 을 동시에 보강해 CP216 결과 표를 다시 만든다. CP217 정적 표의 최종 출처가 이 결과.

## 역할 고정

- 모드: code
- 권한
  - 베이스라인 prediction 생성/재생성, GPU 사용
  - 운영 모델 (CP153 1D 밴드) **predict-only** 재추론 — backfill용. **새 학습/calibration/checkpoint 변경 절대 금지**
  - `data/parquet/` 에 신규 가공 parquet 작성 (`price_data_yfinance_1W_500.parquet`)
  - `ai/eval/significance/` 모듈 확장
- 금지
  - **운영 parquet (`backend/data/v1/predictions_*.parquet`) 수정/덮어쓰기 절대 금지**. backfill prediction은 별도 파일로
  - 새 학습 / calibration / checkpoint 가중치 변경
  - DB write, Supabase 호출
  - 사용자 직접 수정 파일 의미 단위 revert
- 자가 점검 (구현 후 같은 턴)
  - Plan v3 정합: coverage / IC / pinball 정의 운영 동일
  - 구조 결함: GARCH walk-forward fold 정의가 운영 모델 fold 정의와 1:1 일치
  - 모델 영향: 운영 parquet mtime 무변경, 운영 checkpoint 무변경
- 커밋 메시지 간결

## 환경

- 워킹 디렉토리: `C:\Users\user\lens`
- venv `.venv` (Python 3.10.0 / torch 2.11.0+cu128 / numpy 1.26.4 / pandas 2.2.2 / pyarrow 16.1.0 / scipy 1.15.3 / arch / statsmodels)
- GPU: RTX 5060 Ti `sm_120` cu128. `KMP_DUPLICATE_LIB_OK=TRUE`, `TORCHDYNAMO_DISABLE=1`, `num_workers=0`
- fold 정의 (운영 모델과 1:1 동일)
  - **CP153 / CP178 stage5(T) 3-fold** (`docs/v1_operating_models_reproducibility.md` §2~3 참조)
    - fold_1: train `2019-05-01 ~ 2024-05-01`, val `2024-05-01 ~ 2024-11-01`, test `2024-11-01 ~ 2025-05-01`
    - fold_2: train `2019-11-01 ~ 2024-11-01`, val `2024-11-01 ~ 2025-05-01`, test `2025-05-01 ~ 2025-11-01`
    - fold_3: train `2020-05-01 ~ 2025-05-01`, val `2025-05-01 ~ 2025-11-01`, test `2025-11-01 ~ 2026-05-09`
  - CP209 라인 4-fold (W1~W4) 도 같이 박혀있음. 라인 베이스라인 보강 시 사용

## 사용자 요구 (원문)

> ABC 다 묶어서 하자. 결과 오면 정확히 어떤 상태인지 그거까지 해서 해석을 좀 자세히 해줘.

## 범위 (3축 동시)

### 축 A. GARCH baseline walk-forward 재실행 (필수, 공정성)

**왜**
- CP216 보고서: "GARCH는 전체 시계열 1회 fit + 정상상태 σ로 h-step → walk-forward 보다 단순화"
- 운영 밴드는 true walk-forward fold마다 fresh fit. GARCH는 in-sample fit → 베이스라인이 운영보다 유리한 조건 (data leakage)
- "GARCH 우위 (large effect)" 결론이 baseline 부풀림으로 흔들릴 수 있음 → 학계 무기로서 신뢰도 흔들림

**해야 할 것**
- 각 fold 의 train 구간으로 GARCH(1,1) fit, val 구간으로 h-step quantile forecast 생성
- per ticker 별 fit (panel GARCH 아니라 ticker 단위 univariate, CP216 방식 유지)
- 분위수: 1D `q15 / q85`, 1W `q10 / q90` (운영 모델과 일치)
- 결과 parquet: `docs/cp216_2_significance_baselines/garch_walkforward_band_1d.parquet`, `garch_walkforward_band_1w.parquet`
- 비교 unit (ticker, asof_date, horizon) panel은 운영 prediction과 일치하게 정렬

### 축 B. 1D asof backfill (predict-only)

**왜**
- 운영 `predictions_band_1d.parquet` asof 범위 `2025-05-20 ~ 2026-05-08` (1년)
- VIX>30 일수 = 2일 / drawdown -10% 일수 = 0일 → GW test NaN
- 1D selective output trigger 의 통계 근거가 없는 상태 (1W만 GW p<0.001 확보)

**해야 할 것**
- CP153 운영 checkpoint (`tide-1D-ea54dcae654d`) 그대로 사용. **새 학습 X**
- predict-only 로 fold_1 / fold_2 test 구간 (`2024-11-01 ~ 2025-05-01`, `2025-05-01 ~ 2025-11-01`) 까지 inference 확장
  - 그 이전 fold_1 val 구간 (`2024-05-01 ~ 2024-11-01`) 도 OOS 라 추가 가능
  - 그 이전 (fold_1 train 구간 = `2019-05-01 ~ 2024-05-01`) 은 in-sample → **건드리지 마** (data leakage)
- 결과 parquet: `docs/cp216_2_significance_baselines/predictions_band_1d_backfilled.parquet` (운영 파일 무변경)
- backfill 영역에서만 추가 regime 표본 산입. fold_3 영역은 운영 parquet 그대로

**열린 결정 (자가판단 금지, 보고)**
- val 구간 (fold_1 `2024-05-01 ~ 2024-11-01`) 포함할지 (strict OOS 만 vs val 포함). 둘 다 시도해 결과 차이 보고 권장.

### 축 C. 1W price 500 ticker weekly resample

**왜**
- `price_data_yfinance_1W.parquet` 100 ticker (옛 dump)
- 1W baseline (`historical_quantile`, `bollinger`) 만 95 ticker subset 으로 평가됨
- 1W 운영 모델 (445 ticker) 와 표본 불일치 → 통계 검정 unit 일관성 깨짐

**해야 할 것**
- 입력: `data/parquet/price_data_yfinance_500.parquet` (1D, 501 ticker)
- 변환: ticker × week 단위 OHLC aggregate (**금요일 종가 표준**, open=Monday open, high=max, low=min, close=Friday close, volume=sum)
- 출력: `data/parquet/price_data_yfinance_1W_500.parquet` (신규)
- 1W 베이스라인 (`historical_quantile`, `bollinger`) 재생성 → 500 ticker base
- CP178 운영 prediction 445 ticker 와 ticker overlap 분석 (몇 ticker overlap, 어느 ticker 빠지는지)

**열린 결정 (자가판단 금지, 보고)**
- weekly resample anchor (금요일 종가 vs 일요일 종가 / 토요일 종가). 미국 시장은 금요일 표준 → 금요일로 가되 보고서에 명시
- 휴장 주 (Thanksgiving 등) 처리: 마지막 거래일 사용

## 작업 항목

### 1. 데이터 보강 (축 C 먼저)

`ai/eval/significance/baselines.py` 또는 `scripts/cp216_2_weekly_resample.py` 신설
- 1D → 1W resample 함수
- `price_data_yfinance_1W_500.parquet` 생성
- 500 ticker × week count 검증
- manifest JSON (`data/parquet/price_data_yfinance_1W_500.manifest.json`)

### 2. GARCH walk-forward (축 A)

`ai/eval/significance/baselines/garch_walkforward.py` 신설
- fold 정의 import (CP153 stage5T fold_1~3, CP178 stage5 fold_1~3)
- 각 ticker × fold:
  - train 구간으로 GARCH(1,1) fit
  - val 구간에 대해 h-step ahead conditional sigma / quantile forecast
  - test 구간 backtest (운영 모델은 test 에서 inference 했으므로 baseline 도 test 에서)
- ticker 단위 univariate, panel 아님
- 결과 parquet 두 개 (1D / 1W)

### 3. 1D backfill (축 B)

`scripts/cp216_2_band_1d_backfill_predict_only.py` 신설
- 운영 checkpoint `tide-1D-ea54dcae654d` 로드
- inference only. checkpoint 가중치 hash 보존 (학습 X confirm)
- fold_1 test + fold_2 test (옵션: fold_1 val) 구간 inference
- 결과 parquet: `docs/cp216_2_significance_baselines/predictions_band_1d_backfilled.parquet`
- 운영 calibration 그대로 적용 (CP153 `lower_focused`)

### 4. CP216 통계 검정 재실행

`ai/eval/significance/cli.py` 옵션 확장
- `--use-walkforward-garch` flag
- `--backfilled-band-1d <parquet>` flag
- `--price-1w-parquet <parquet>` flag
- 출력: `docs/cp216_2_significance_report.md`, `cp216_2_significance/{summary.csv, metrics.json}`

### 5. 비교: CP216 (원래) vs CP216.2

- summary.csv 양쪽 join
- delta 표 (점추정 차이, p-value 변화, Cohen's d 변화, CI shift)
- 어느 결론이 뒤집혔는지 / 강화됐는지 명시

## 검증

### 단위

- 운영 parquet mtime 무변경 (`backend/data/v1/predictions_band_1d.parquet`, `predictions_band_1w.parquet`)
- 운영 checkpoint hash 무변경 (CP153/CP178 weights)
- backfilled 1D prediction이 fold_3 test 구간 (운영 영역) 과 운영 parquet 같은 ticker/asof 에서 **수치 일치** (re-run 일관성 검증)
- weekly resample 출력의 ticker 500개 / week count 일관성
- GARCH walk-forward fold 정의가 CP153 stage5T metrics.json 안 fold 정의와 1:1 일치 (assert)

### 결과 sanity

- backfill 후 1D regime 표본
  - VIX>30 일수 >= 30 (목표)
  - drawdown -10% 일수 >= 30 (목표)
- 1W baseline ticker count = 500
- GARCH walk-forward 결과가 in-sample GARCH 보다 일반적으로 **약간 더 나쁨** 예상 (walk-forward는 더 보수적)

### dry-run

- 50 ticker × fold_3 만 먼저
- PASS 후 full run

## 산출물

- `data/parquet/price_data_yfinance_1W_500.parquet` (신규)
- `docs/cp216_2_significance_baselines/garch_walkforward_band_1d.parquet`
- `docs/cp216_2_significance_baselines/garch_walkforward_band_1w.parquet`
- `docs/cp216_2_significance_baselines/predictions_band_1d_backfilled.parquet`
- `docs/cp216_2_significance/summary.csv`
- `docs/cp216_2_significance/metrics.json`
- `docs/cp216_2_significance_report.md` — 본 보고서
- `docs/cp216_2_significance_runs/full_run.log`

## 보고서 narrative (구현 후 채울 것)

### 결과 표 한 장
- 모델 × 베이스라인 × test 종류 × (statistic / p_raw / p_bonferroni / CI_cluster / CI_block / Cohen's d / 결론 라벨)
- CP216 (원래) vs CP216.2 delta 컬럼

### 어떤 결론이 뒤집혔는지
- 예: "GARCH walk-forward 적용 후 1D 밴드 vs GARCH effect size Cohen's d 0.69 → 0.21로 축소. 통계적 유의 유지하나 effect medium → small로 약화. 'GARCH 우위' narrative 약화."
- 또는: "그대로 유지. baseline 공정성 보강 후에도 운영 밴드 약점 진짜."

### 1D GW 결과
- backfill 후 VIX>30 / drawdown 일수, GW wald p-value
- 1D selective output trigger 의 통계 근거 정의 (예: "VIX>30 regime 에서 1D 밴드는 GARCH walk-forward 대비 통계적 우위 없음 → 침묵 trigger")

### 1W 일반화
- 500 ticker base 결과. ticker overlap 분석 + 빠진 ticker 영향

### selective output trigger 최종 정의 (v2 master plan §0 박을 문장)
- regime 별 통계적 결론 표
- 침묵 trigger condition (regime + p-value 임계값)

## 자가 점검 결과 양식

```
[Plan v3 정합]
- coverage / IC / pinball 정의 운영 동일: PASS/WARN/FAIL + 사유
- GARCH walk-forward fold 정의 운영 일치: PASS/WARN/FAIL + 사유
[구조 결함]
- baseline 모듈 단일화 (walkforward 추가 후): PASS/WARN/FAIL + 사유
[모델 영향]
- 운영 parquet mtime 무변경: PASS + mtime 캡처
- 운영 checkpoint weights hash 무변경: PASS + hash 캡처
- 새 학습 / calibration 없음: PASS
```

## 후속 / 열린 질문 (작업 끝나면 보고)

- 1D backfill val 포함 여부 (strict OOS vs val 포함)
- weekly resample anchor (금요일 표준 vs 다른)
- GARCH walk-forward 결과가 CP216 결론을 약화/유지/뒤집기 중 어느 쪽인지
- 1W 500 ticker 보강 후 ticker overlap (445 운영 vs 500 baseline → ticker subset/superset 비교)
- selective output trigger 의 통계 임계값 (Bonferroni p<0.05 vs p<0.001 vs CI overlap)
- CP217 정적 표에 어느 결과를 박을지 (CP216 원본 / CP216.2 보강 / 둘 다 비교)

## 다음 단계 약속

이 CP가 끝나면 사용자에게 **정확히 어떤 상태인지** 다음 6 축으로 자세 해석 보고:

1. **모델 자체의 평가** — 운영 3모델이 통계 베이스라인 대비 어디서 우위/동등/열위
2. **베이스라인 공정성 보강 효과** — A (GARCH walkforward) 적용 후 결론 변화 magnitude
3. **regime 별 결론** — 1D + 1W 둘 다 GW p-value, selective output trigger 정의 가능 여부
4. **표본 일반화** — 1W 500 ticker base 결과, ticker overlap 영향
5. **v1 정체성 영향** — "리스크 인식 도구" narrative 가 통계로 받쳐지는지 / 어디가 약한지 정직 평가
6. **CP217 정적 표 박을 narrative 초안** — 모델별/지표별 1줄 결론
