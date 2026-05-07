# CP148-0-DG Sufficiency Gate Repair Report

작성일: 2026-05-07

## 1. Executive Summary

최종 판정: **WARN**

1D/1W split plan 생성 단계에 timeframe별 absolute minimum history gate를 추가했다. 1D는 최소 450 trading rows, 1W는 최소 78 weekly rows를 요구한다. 기존 `seq_len + h_max` 조건과 별개로 적용되며, 최종 필요 row 수는 `max(seq_len + h_max, timeframe_abs_min_rows)`로 판단한다.

기존 EODHD 500 1D/1W parquet 기준 eligible count는 감소하지 않았다. 다만 일부 짧은 history ticker의 탈락 사유가 기존 `Gate A/C`에서 명시적인 `insufficient_absolute_history_*`로 바뀐다. cache manifest 감사에서는 feature/index cache 117개 중 75개가 manifest 없이 남아 있어 삭제는 하지 않고 수동 정리 후보로만 기록한다. 이 때문에 판정은 PASS가 아니라 WARN이다.

금지 작업 확인:

- 모델 학습 없음
- W&B/Optuna 실행 없음
- inference 저장 없음
- DB/Supabase write 없음
- 프론트 수정 없음
- EODHD/yfinance fetch 없음
- cache 삭제 없음

## 2. 수정 내용

### `ai/splits.py`

추가:

- `TIMEFRAME_ABSOLUTE_MIN_ROWS = {"1D": 450, "1W": 78}`
- `absolute_min_rows_for_timeframe(timeframe)`
- `required_history_rows(timeframe, seq_len, h_max)`

`make_splits()`에서 다음 순서로 사전 탈락한다.

1. `row_count < seq_len + h_max`이면 기존처럼 `Gate A`
2. `row_count >= seq_len + h_max`이지만 absolute minimum 미달이면:
   - `insufficient_absolute_history_1D_min_450`
   - `insufficient_absolute_history_1W_min_78`
3. 이후 기존 effective sample, fold size, embargo gap gate 적용

### `ai/preprocessing.py`

`DatasetPlan`과 split diagnostics에 다음 필드를 추가했다.

- `absolute_min_rows`
- `required_history_rows`

`resolve_prepared_splits_cache_key()` payload에도 같은 값을 포함해, split gate 기준이 바뀌면 in-memory prepared split cache key도 달라진다. feature cache contract version은 그대로 `v3_adjusted_ohlc`를 유지했다.

## 3. 테스트 추가

수정한 테스트:

- `ai/tests/test_splits.py`

추가/보강한 테스트:

| 테스트 | 검증 |
|---|---|
| `test_make_splits_rejects_1d_absolute_history_below_450` | 1D에서 `seq_len+h_max`는 만족하지만 450 rows 미만이면 제외 |
| `test_make_splits_rejects_1w_absolute_history_below_78` | 1W에서 `seq_len+h_max`는 만족하지만 78 rows 미만이면 제외 |
| `test_absolute_history_gate_keeps_sufficient_ticker_eligible` | 충분한 1D row ticker는 기존처럼 통과 |
| `test_absolute_history_exclusion_reason_is_kept_in_split_specs` | exclusion reason이 split specs 결과에 남음 |
| `test_absolute_history_required_rows_are_separate_from_sequence_requirement` | `max(seq_len+h_max, abs_min)` 계약과 1M 제외 정책 확인 |

## 4. EODHD 500 Eligible Count 비교

읽기 전용 입력:

- `data/parquet/indicators_eodhd_1D_500.parquet`
- `data/parquet/indicators_eodhd_1W_500.parquet`

| 대상 CP | timeframe | seq_len | h_max | required rows | before eligible | after eligible | 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CP148-LM-1D | 1D | 252 | 20 | 450 | 473 | 473 | 0 |
| CP149-BM-1D | 1D | 60 | 20 | 450 | 476 | 476 | 0 |
| CP150-LM-1W | 1W | 104 | 12 | 116 | 447 | 447 | 0 |
| CP151-BM-1W | 1W | 60 | 12 | 78 | 453 | 453 | 0 |

1D/1W 모두 eligible count 자체는 줄지 않았다. 짧은 ticker는 기존 gate에서도 이미 탈락했지만, 이제 원인이 명시적으로 기록된다.

## 5. 탈락 Ticker 수와 주요 Reason

| 대상 CP | excluded count | 주요 reason |
|---|---:|---|
| CP148-LM-1D | 30 | `Gate fundamentals`: 26, `Gate C`: 3, `Gate A`: 1 |
| CP149-BM-1D | 27 | `Gate fundamentals`: 26, `insufficient_absolute_history_1D_min_450`: 1 |
| CP150-LM-1W | 55 | `Gate fundamentals`: 25, `Gate C`: 21, `Gate A`: 6, `Gate B`: 3 |
| CP151-BM-1W | 49 | `Gate fundamentals`: 25, `Gate C`: 17, `Gate A`: 3, `Gate B`: 3, `insufficient_absolute_history_1W_min_78`: 1 |

명시적 absolute history reason으로 재분류된 ticker:

- `SNDK`: 1D BM 기준 248 rows, `insufficient_absolute_history_1D_min_450`
- `VLTO`: 1W BM 기준 77 rows, `insufficient_absolute_history_1W_min_78`

## 6. 1M 정책

1M은 현재 full run 대상이 아니다. 코드상 `MAX_HORIZON_BY_TIMEFRAME["1M"] = 3`은 유지하지만, `TIMEFRAME_ABSOLUTE_MIN_ROWS`에는 1M 기준을 추가하지 않았다. 문서상 최소 기준을 둔다면 `seq_len + h_max` 기준부터 시작하되, CP148~CP151 sweep 대상에서는 제외한다.

## 7. Cache Manifest 감사

삭제 없이 목록만 감사했다.

| 항목 | count |
|---|---:|
| feature/index cache 파일 | 117 |
| manifest valid | 42 |
| manifest missing | 75 |
| manifest legacy/invalid | 0 |

manifest 없는 cache 예시:

- `ai/cache/feature_index_1D_1a967362529f_0c1d7f52.pt`
- `ai/cache/feature_index_1D_1a967362529f_71a1d998.pt`
- `ai/cache/feature_index_1D_42a1fd663092_3ac43945.pt`
- `ai/cache/feature_index_1D_65aa31458e4a_960f3bc8.pt`
- `ai/cache/feature_index_1D_92d8fcad896b_f77c891d.pt`

권장:

- 이번 CP에서는 삭제 금지.
- sweep 전에 manifest 없는 `features_*.pt`, `feature_index_*.pt`는 수동 삭제 후보로만 표시.
- 새 gate가 적용된 plan은 cache key에 `absolute_min_rows`, `required_history_rows`가 들어가므로 prepared split cache 혼동 가능성은 낮다.

## 8. Seed Stability 계획

이번 CP에서는 seed stability를 구현하지 않았다. sweep은 단일 `seed=42`로 진행 가능하다. 단, best trial은 다음 기준을 반드시 통과해야 제품 후보나 hp gain으로 해석한다.

1. sweep top 후보를 3~5 seed로 재실행한다.
2. top1~top5 평균 차이가 seed variance보다 작으면 hp gain으로 보지 않는다.
3. validation stability를 candidate registry에 기록한다.
4. test set은 최종 후보 확인용으로만 사용한다.
5. sector/regime split metric을 함께 보고한다.

## 9. Sweep 진행 가능 여부

| CP | 진행 가능 여부 | 조건 |
|---|---|---|
| CP148-LM-1D | 가능 | 1D LM eligible 473 유지, gate 적용 확인 |
| CP149-BM-1D | 가능 | 1D BM eligible 476 유지, SNDK absolute history 탈락 명시 |
| CP150-LM-1W | 가능 | 1W LM eligible 447 유지 |
| CP151-BM-1W | 가능 | 1W BM eligible 453 유지, VLTO absolute history 탈락 명시 |

단, cache manifest missing이 많으므로 sweep 전 cache 정리 여부를 사람이 한 번 확인하는 것이 좋다. 삭제 자동화는 이번 CP에서 하지 않았다.

## 10. 검증

통과:

- `.venv\Scripts\python.exe -m py_compile ai\splits.py ai\preprocessing.py ai\tests\test_splits.py`
- `.venv\Scripts\python.exe -m unittest ai.tests.test_splits`: 9 tests OK
- `.venv\Scripts\python.exe -m unittest ai.tests.test_preprocessing`: 6 tests OK
- `docs/cp148_0_sufficiency_gate_repair_metrics.json` parse PASS

참고:

- `ai.tests.test_preprocessing_cache_isolation`은 별도 실행 시 Windows `torch` direct import에서 `c10.dll` 초기화 오류가 발생할 수 있다. 이 CP의 gate 보장 테스트는 `test_splits`에 추가했다.

## 11. 최종 판정

**WARN**

absolute minimum history gate는 적용됐고 테스트로 보장된다. 기존 EODHD 500 1D/1W eligible count도 과도하게 줄지 않았다. 다만 manifest 없는 cache가 75개 있어 sweep 전 수동 정리 후보를 확인해야 한다.
