# CP3 지시서 — Sufficiency 제외 ticker 정착 + train/val/test split 구현

## 배경

CP2.6에서 데이터 기반 복구 완료. 이제 학습 대상 ticker를 **timeframe별로 확정**하고, 기획안 §3.6 · D15(gap=h_max) · D4(train/val/test 70/15/15) 에 따른 split을 코드로 정착한다.

CP2.6 확정 수치:
- 전체 유니버스 503 ticker, fundamentals 8-quarter gate 통과 477
- seq_len=252 (1D) 가능 476 / 477 (1 ticker 탈락)
- seq_len=104 (1W) 가능 472 / 477 (5 ticker 탈락)
- row_count p10: 1D 1730 / 1W 314
- gap = h_max: 1D=20, 1W=12

이 CP는 **코드 작업만**. 학습 실행은 CP4~ 이후.

---

## 목표

1. **학습 대상 ticker 리스트 결정 로직**을 `ai/preprocessing.py`(또는 동급 지점)에 상설화.
2. **timeframe별 독립 필터**: 1D·1W에 각각 다른 제외 리스트 적용.
3. **3단계 sufficiency gate 코드 정착**:
   - Gate 1: fundamentals 8-quarter (CP2.6까지 이미 처리, 477 통과)
   - Gate 2: `row_count >= seq_len + h_max + min_fold_samples` 보장
   - Gate 3: split 후 val/test fold가 최소 샘플 수(권고 50개) 이상
4. **70/15/15 time-ordered split + gap=h_max** 로직 구현.
5. **샘플 수 집계 보고**: (ticker × timeframe)별 train/val/test 샘플 수, 전체 합계, 제외 ticker 전량 목록.

---

## 예상 시간

| 단계 | 시간 |
|---|---|
| 기존 preprocessing/training 경로 파악 | 20~30분 |
| sufficiency gate 함수 작성 + 테스트 | 30~45분 |
| split 함수 작성 + 테스트 (경계·gap·fold 최소치) | 45~60분 |
| 집계·로깅·CLI 진입점 | 20~30분 |
| 전체 유니버스 dry-run 집계 | 10분 |
| 테스트 정비 + 회귀 | 20~30분 |
| **총 체감** | **2.5~3.5시간** |

기준: CPU 작업 위주, 풀 학습 미실행.

---

## 권한 번들 (사전 승인)

**파일 쓰기 범위**:
- `ai/preprocessing.py`, `ai/train.py`, `ai/inference.py` (해당 시)
- `ai/datasets.py` 또는 신규 `ai/splits.py` 생성 허용
- `scripts/diagnostics/sufficiency_report.py` (신규 집계 스크립트)
- `tests/**`
- `docs/cp3_report.md` (종료 리포트)

**허용 명령**:
- `uv run pytest tests/`
- `uv run python scripts/diagnostics/sufficiency_report.py` (읽기 전용 집계)
- `uv run python -m ai.train --dry-run` 형태의 split 검증 (학습 실제 실행 금지)

**금지**:
- 학습 실제 실행(train) 금지
- DB 쓰기 금지 (예측·model_run 저장 등)
- `indicators`·`price_data` 수정 금지
- `git push` 금지

**휴먼 개입 트리거**:
- Gate 2/3 임계 결정이 애매 (val fold 최소 50 권고지만 실측 분포 보고 조정 제안 환영)
- seq_len=252 가능 ticker가 CP2.6 보고(476) 대비 5개 이상 차이
- 예상 시간 3배 초과 (10시간+)

---

## 실행 상세

### 1단계 — 현 상태 파악

`ai/preprocessing.py`, `ai/train.py`에서 ticker iteration·split 관련 코드를 찾고, 현재 어떤 방식으로 학습 대상을 고르는지 확인. (이미 기초 스모크 학습은 돌았으므로 어딘가 split 로직이 있을 것)

### 2단계 — 3단계 sufficiency gate 함수

`ai/preprocessing.py` (또는 신규 `ai/splits.py`)에 다음 함수 구현:

```python
def eligible_tickers(
    feature_frame: pd.DataFrame,   # build_features 결과 전체
    *,
    timeframe: str,                # "1D" | "1W"
    seq_len: int,                  # 252 | 104
    h_max: int,                    # 20 | 12
    min_fold_samples: int = 50,    # val/test fold 최소 샘플
) -> tuple[list[str], dict[str, str]]:
    """
    반환:
      - 통과 ticker 리스트
      - 제외 ticker → 탈락 사유 매핑 (dict)

    3단계 gate:
      Gate A: row_count >= seq_len + h_max (샘플 1개 이상)
      Gate B: 70% 구간 >= seq_len + h_max + min_fold_samples (train fold 최소)
      Gate C: 15% val, 15% test fold가 각각 min_fold_samples 이상
    """
```

**Gate 계산 로직**:
- 총 사용 가능 샘플 수 = `row_count - seq_len - h_max` (= 샘플 수)
- Train 샘플 = `floor(total * 0.70)`
- Val 샘플 = `floor(total * 0.15)`
- Test 샘플 = `total - train - val` (약 15%)
- Gate B: `train >= min_fold_samples`
- Gate C: `val >= min_fold_samples AND test >= min_fold_samples`

**핵심 주의: gap 이중 카운팅 방지**
- gap=h_max는 "train 마지막 입력과 val 첫 입력 사이의 경계"용. 샘플 수 계산 시 train과 val 경계에서 h_max만큼 빠지는 것.
- 구현: train 마지막 샘플 타깃 끝 = val 첫 샘플 입력 시작 (= 교집합 없음) 이 되도록 경계 index를 h_max 빼고 자르기.
- test 경계도 동일.

### 3단계 — split 함수

```python
def make_splits(
    ticker_frame: pd.DataFrame,    # 단일 ticker의 feature frame
    *,
    seq_len: int,
    h_max: int,
    min_fold_samples: int = 50,
) -> SplitsSpec:
    """
    반환 dataclass/dict: train/val/test 각각의 (sample 인덱스 리스트 or (start, end) 경계)
    """
```

- Time-ordered 70/15/15
- Gap = h_max 적용: train 끝 index + h_max < val 시작 index
- 검증 포인트: 인접 fold 간 입력/타깃 윈도우 교집합 0 확인.

### 4단계 — 테스트 필수

최소 3건 추가:

1. **Gate 테스트**: row_count가 임계 바로 위/아래인 dummy ticker로 A/B/C 판정 분기 확인.
2. **split 경계 누수 테스트**: dummy ticker에서 train 마지막 샘플 타깃 끝과 val 첫 샘플 입력 시작의 시점 차이가 **`>= h_max`** 인지.
3. **timeframe 독립성 테스트**: 동일 유니버스에 1D/1W 필터 돌려서 결과 리스트가 다를 수 있음을 확인 (CP2.6 보고 기준 1D 476·1W 472).

### 5단계 — 집계 스크립트

`scripts/diagnostics/sufficiency_report.py`:

```
# 전체 유니버스 × 1D/1W 에 대해 eligible_tickers 호출
# 출력:
#   - 최종 학습 대상 ticker 수 (timeframe별)
#   - 제외 ticker 리스트 (timeframe별, 탈락 사유 포함)
#   - train/val/test 샘플 수 합계 (timeframe별)
#   - 각 ticker별 샘플 수 분포 p10/p50/p90
```

리포트는 `docs/cp3_report.md`에 붙임.

### 6단계 — 학습 진입점 (`ai/train.py`) 연결

- 학습 시작 전 `eligible_tickers` 호출 → 통과 ticker만 학습 데이터로 사용
- `make_splits` 결과에 따라 DataLoader 구성
- **실제 학습 실행 금지**. `--dry-run` 모드나 `check-splits` 서브커맨드로 split 구성 검증까지만.

---

## 종료 보고 포맷

```
[CP3] 완료

## 1. sufficiency gate 구현
- 위치 함수: _ (경로:함수명)
- Gate A/B/C 각각 임계치: _
- min_fold_samples 결정값: _ (권고 50, 최종 _)

## 2. split 구현
- 비율: 70/15/15 (확정)
- gap 적용: h_max (1D=20, 1W=12)
- 경계 누수 테스트 통과: Y/N

## 3. 최종 학습 대상
### 1D (seq_len=252, h_max=20)
- 입력 유니버스: 477
- 통과: _
- 제외: _ (사유별 breakdown)
  - Gate A 탈락: _
  - Gate B 탈락: _
  - Gate C 탈락: _
- 제외 ticker 리스트: _

### 1W (seq_len=104, h_max=12)
- 입력 유니버스: 477
- 통과: _
- 제외: _ (사유별 breakdown)
- 제외 ticker 리스트: _

## 4. 샘플 수 집계
### 1D
- ticker당 train 샘플 p10/p50/p90: _
- ticker당 val 샘플 p10/p50/p90: _
- ticker당 test 샘플 p10/p50/p90: _
- 전체 train 샘플 합: _
- 전체 val 샘플 합: _
- 전체 test 샘플 합: _

### 1W
- (동일 포맷)

## 5. 테스트
- 기존 21 테스트: green Y/N
- 신규 테스트: _건 (이름 리스트)
- 경계 누수 테스트: Y/N
- timeframe 독립성 테스트: Y/N

## 6. 학습 진입점 연결
- `ai/train.py`에서 eligible_tickers + make_splits 호출 체인 정착: Y/N
- dry-run으로 split 구성 출력 확인: Y/N
- 실제 학습 실행: 금지 준수 (Y)

## 7. CP4 준비 상태
- ticker embedding 도입 대상 모델 (PatchTST/CNN-LSTM/TiDE) 입력 인터페이스 확인: Y/N
- 학습 대상 ticker 수 기준 embedding 차원 결정 가능: 1D=_, 1W=_
```

---

## 체크포인트 준수

- 이 CP는 **단일 종료 보고**. 중간 보고 없음.
- 위 종료 보고 7개 섹션 전부 숫자로. "추후 확정"·"파일 참고" 금지.
- CP4(ticker embedding)는 이 보고 통과 후 지시서 따로 발주.
- 1M recompute (CP2.7) 은 이 CP와 독립 진행. 이 CP에서 건드리지 말 것.
