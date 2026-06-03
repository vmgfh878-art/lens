# CP227 보고서 — BE 안정성 (admin 민감정보 / pandera 경계 / except 로깅)

## 요구

`docs/cp227_error_standardization_pandera_directive.md`. (1) admin `debug-state` 응답에서 traceback / 예외 메시지 / 예외 타입 제거, (2) pandera 5 종 스키마로 parquet read/merge 경계 보호, (3) 광범위 `except Exception` 9 곳 중 5 곳에 로깅 추가 / 일부 좁힘 — 모두 정상 경로 응답 0 변경 (CP223 snapshot diff 0).

## 한 일

4 step, 4 commit:

| Step | 커밋 | 작업 | 검증 |
|---|---|---|---|
| 1 | 33283a5 | admin debug-state 응답 민감정보 제거 (3 곳 — market probe / parquet stat / memory) + request_id 로깅 | snapshot diff 0, ruff baseline 동일 |
| 2 | c6b5638 | pandera 0.20.4 추가, `app/schemas/frames.py` 신설 (5 DataFrameModel) | 운영 parquet 5 종 검증 통과 (스모크) |
| 3 | 9e37b4f | parquet_store._load / local_market_svc._load / strategy_scan._load_frame 머지 직전에 검증 연결 | snapshot diff 0, 87 passed |
| 4 | 1a7ef83 | market_repo 5 곳 + predictions.py load_caches 1 곳에 logger.warning / debug 추가 | snapshot diff 0, 87 passed |

## 핵심 컴포넌트 존재 체크리스트

- **admin 민감정보 제거**: `admin.py` 의 market probe 응답 키 `exc_type` / `exc_msg` / `traceback_tail` 부재, parquet stat 응답 키 `error` 부재, memory probe 응답 키 `error` 부재. `traceback` import 사용처 0 → 제거.
- **pandera 스키마 5 종**: `LineDailyFrame`, `Band1dFrame`, `Band1wFrame`, `MarketPrices1d`, `MarketIndicators1d`. 날짜 컬럼 (`asof_date`, `forecast_date`, `date`) 은 모두 `Series[str]` + `coerce=True` (CP214 회귀 방지). `strict=False` 로 disk 의 추가 컬럼 (`actual_h5_return`, `*_flag` 등) 허용.
- **pandera import 경로**: `import pandera as pa` (0.20.4 — `pandera.pandas` 모듈 미존재).
- **read 경계 검증 연결**: parquet_store `_validate_slot` 으로 슬롯명 → 모델 매핑, local_market_svc `_FILE_MODELS` 로 파일명 → 모델 매핑. 매핑에 없는 슬롯/파일은 검증 우회 (신규 슬롯 의도 통과).
- **머지 경계 dtype 보호**: strategy_scan `_load_frame` 머지 직전 4 프레임 (price/indicators/line/band) 의 `date` dtype 이 `datetime64` 임을 assert.
- **lru_cache 보존**: strategy_scan `_load_frame` / `_sector_map` / `_strategy_results` 의 `.cache_clear` attribute CP226 검증 그대로 유효.
- **except 로깅**: market_repo 5 곳, predictions.py 1 곳. 재발생 경로 (fetch_price_rows, fetch_indicator_rows outer, fetch_stocks outer) 는 `logger.warning(..., exc_info=exc)` + raise 유지. best-effort 삼킴 (`_merge_indicator_volume`, `_fetch_stock_info_rows`) 은 `logger.debug(..., exc_info=exc)` + 사유 주석 + 응답 동작 0 변경.

## 새 테스트 결과

신규 테스트 0. 본 CP 는 구조 / 검증 / 로깅만 추가하며, 기존 `backend/tests` 87 passed 가 유효성 검증 기준선.

운영 parquet 5 종 검증 스모크 (Step 2 직후, 일회성 인라인 스크립트):
- line_1d (182,732 행) OK
- band_1d (597,160 행) OK
- band_1w (186,900 행) OK
- market_prices_1d (137,632 행) OK
- market_indicators_1d (137,565 행) OK

## dry-run 결과

CP223 syrupy snapshot 9 endpoint, 매 Step:
- `pytest backend/tests/test_characterization_api.py` → 9 passed, diff 0.
- `pytest backend/tests/test_feature_svc.py` → 11 passed.

정상 경로 응답이 1바이트도 바뀌지 않음 (pandera coerce 가 disk dtype 과 일치하므로). 비정상 경로 (admin debug-state error 분기) 는 의도된 schema 축소 — `exc_type` / `exc_msg` / `traceback_tail` 키 부재. CP223 이 debug-state error 분기를 박제하지 않으므로 snapshot 영향 0.

## 기존 회귀 통과 건수

- `pytest backend/tests` (test_services.py 제외): **87 passed** — CP226 baseline 그대로.
- `pytest backend/tests` (전체): 87 passed + 11 failed (전부 pre-existing).
- `ruff check`: market_repo + predictions.py 22 errors (baseline 동일, Step 4 fix 후). frames.py / parquet_store / local_market_svc / admin.py 신규 위반 0.
- `mypy`: frames.py 신규 error 0.
- facade / lru_cache / 라우터 import 모두 그대로.

## 결정

- pandera 0.20.4 + `import pandera as pa` (`pandera.pandas` 미존재).
- 날짜 컬럼 `Series[str]` + `coerce=True` — CP214 fix 와 정합.
- `strict=False` — disk 의 추가 컬럼 허용 (line_1d 의 `actual_h5_return` 등).
- `strategy_scan._load_frame` 의 `pd.read_parquet(market_*)` 직접 호출 경로는 read 직후 스키마 강제 안 함 (자체 정규화로 dtype 변형). 대신 머지 직전 datetime64 assert 로 가드. 후속에서 local_market_svc 경유 통일 검토.
- HTTPException → AppError 표준화는 범위 밖 (응답 schema 변경 위험). 별도 CP.
- Supabase 호출 except 좁히기는 클라이언트 예외 타입 불명확 + Plan v3 Supabase 보류 → 로깅만 추가, 좁히기는 Supabase 재개 CP.
- numpy `.item()` 폴백 (predictions.py:85 등) 은 의도적 직렬화 폴백 → 유지.

## 후속

1. **CP228 — structlog + correlation-id**: lens.admin / lens.predictions logger 가 dict context 를 자동 첨부하도록 structlog 도입. request_id 가 응답 + 로그 모두에 일관되게.
2. **CP229 — async blocking + 캐시 안전성**: lru_cache(maxsize=1/16) 동시 호출 안전성, def 전환 시 schema 보존.
3. **HTTPException → AppError 표준화** (별도 CP): 진단 3 의 비표준 `{"detail":...}` 응답 → 표준 `{"error":{"code":...,"message":...},"meta":{...}}` 전환. 프론트 fetch 영향 분석 필수.
4. **strategy_scan 의 직접 parquet read 통일**: `local_market_svc.get_prices_1d` / `get_indicators_1d` 경유로 통일하면 read 경계 스키마가 그쪽도 보호. 단 `_load_frame` 의 정규화 (groupby pct_change 등) 가 disk 컬럼과 다르므로 후처리 분리 필요.
5. **Supabase 도입 재개 CP**: market_repo 재발생 except 의 Supabase 클라이언트 예외 타입 (HTTPError / TimeoutError 등) 으로 좁히기. 본 CP 는 로깅만.
