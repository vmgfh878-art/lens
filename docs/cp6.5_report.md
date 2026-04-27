## 캘린더 피처 7개 정의/계산 위치
`/C:/Users/user/lens/ai/preprocessing.py`에서 `build_calendar_feature_frame()`로 계산합니다. `day_of_week_sin/cos`, `month_sin/cos`, `is_month_end`, `is_quarter_end`, `is_opex_friday` 7개를 날짜만으로 결정론적으로 만듭니다.

## 입력 채널 변화 (29→36, 채널 인덱스 매핑)
모델 입력은 기존 `SOURCE_FEATURE_COLUMNS` 29개 뒤에 캘린더 7개를 붙여 총 36채널로 확장했습니다. `target_channel_idx=0`은 그대로 유지되어 `log_return` 채널 인덱스는 변하지 않습니다.

## TiDE forward 시그니처 변경
`/C:/Users/user/lens/ai/models/tide.py`의 `forward`를 `forward(x, ticker_id=None, future_covariate=None)`로 확장했습니다. `future_cov_dim=0`이면 기존 경로 그대로 동작하고, 0보다 크면 `future_covariate` 입력을 요구합니다.

## Future covariate 주입 라인
TiDE는 decoder 출력 뒤에 `ticker embedding`과 `future_covariate`를 순서대로 concat한 뒤 `temporal_decoder`에 넣습니다. 미래 캘린더는 horizon 축으로 이미 정렬된 `Tensor[B, H, 7]` 형식으로 전달됩니다.

## DataLoader collate 변경 사항
`/C:/Users/user/lens/ai/train.py`의 `make_loader()`는 `features`, `line_targets`, `band_targets`, `ticker_ids`에 더해 `future_covariates`도 함께 배치로 묶습니다. `TiDE + use_future_covariate`일 때만 학습/평가/드라이런 forward에 전달합니다.

## 6건 테스트 결과
신규 6건은 모두 통과했습니다. `test_calendar_features_deterministic_from_date`, `test_future_calendar_aligns_with_target_dates`, `test_future_calendar_no_value_leak`, `test_tide_future_covariate_affects_output`, `test_tide_without_future_covariate_still_works`, `test_n_features_36_forward_all_models` 기준으로 캘린더 결정성, 정렬 일치, 누수 방지, TiDE 주입 경로를 확인했습니다.

## Leakage 검증 결과 (test 2, 3 강조)
`test_future_calendar_aligns_with_target_dates`로 `future_calendar[h]`가 정확히 목표 `h+1`일자 캘린더와 일치함을 확인했습니다. `test_future_calendar_no_value_leak`로 미래 공변량에는 가격, 수익률, 거시 값이 없고 7개 캘린더 채널만 들어가며 값 범위도 캘린더 특성 범위 안에 있음을 확인했습니다.

## Dry-run 5건 결과
`tide 1D direct future_cov=on`, `tide 1D direct future_cov=off`, `tide 1W direct future_cov=on`, `patchtst 1D direct`, `cnn_lstm 1D direct`를 모두 통과했습니다. 전부 `lower<=upper`, `line_preserved=true`, `NaN/Inf 없음`을 확인했습니다.

## 메모/잔여 issue
AI 전체 테스트는 59건, backend 회귀 테스트는 23건 모두 통과했습니다. `pandas.read_sql_query`의 DBAPI 경고는 계속 보이지만 기능 실패는 없었고, 이번 작업 범위상 SQLAlchemy 전환은 하지 않았습니다.
