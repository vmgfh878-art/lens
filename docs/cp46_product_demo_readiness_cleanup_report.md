CP46-P는 데모 readiness의 남은 FAIL/WARN을 줄이고, 검색과 백테스트 점검 기준을 실제 데모 run 기준으로 맞춘 정리 작업이다.

# 1. 현재 마감 상태

사용자 지시에 따라 CP46-P는 여기서 중단하고 closure 처리한다.

프론트 dev 서버 `http://127.0.0.1:3000` 응답 확인은 더 기다리지 않았고, residual risk로 남긴다.

# 2. Stock Search 503 원인

문제 경로:

```text
GET /api/v1/stocks?search=AAPL&limit=6
```

원인 파일/함수:

- `backend/app/routers/v1/stocks.py`
  - `list_stocks()`
- `backend/app/services/api_service.py`
  - `get_stocks()`
- `backend/app/repositories/market_repo.py`
  - `fetch_stocks()`

기존 `fetch_stocks()`는 `stock_info`만 조회했고, 검색어가 있을 때 Supabase `ilike("ticker", "%AAPL%")` 쿼리가 실패하면 `UpstreamUnavailableError`로 감싸져 API 503이 반환됐다.

가격 조회는 `price_data` 경로를 쓰기 때문에 정상이고, 티커 검색만 실패하는 구조였다.

# 3. Stock Search 수정 방식

`backend/app/repositories/market_repo.py`에서 read-only fallback을 추가했다.

수정 방식:

- `stock_info`를 우선 조회한다.
- 검색어가 있으면 Supabase `ilike`에 직접 의존하지 않고, 제한된 범위의 `stock_info`를 읽은 뒤 Python에서 ticker 필터링한다.
- `stock_info`가 비어 있거나 실패하면 `price_data`의 ticker를 fallback source로 사용한다.
- fallback 결과는 ticker 중심으로 반환하고, `sector`, `industry`, `market_cap`은 없으면 `null`로 안전 처리한다.
- fake data는 만들지 않았다.

확인 결과:

- AAPL 검색 정상
- MSFT 검색 정상
- 직접 티커 입력 가격 조회 유지

# 4. Backtest WARN 원인

기존 readiness는 아래 기준으로 demo run을 찾았다.

```text
/api/v1/ai/runs?model_name=patchtst&status=completed&timeframe=1D&limit=20
```

하지만 현재 주식 보기 화면은 composite prediction run인 아래 run을 실제 데모 run으로 사용한다.

```text
composite-1D-3a44b5e51ed2
```

따라서 readiness가 PatchTST run 기준으로 backtest row를 확인하면서, 실제 표시 중인 composite run에는 backtest row가 있어도 `backtest row missing` WARN이 남을 수 있었다.

# 5. Readiness 수정 여부

`scripts/check_demo_readiness.ps1`을 수정했다.

수정 내용:

- run 조회를 `model_name=&status=completed&timeframe=1D&limit=50` 기준으로 변경
- `line_band_composite`, `patchtst`를 demo candidate model로 필터링
- 실제 usable prediction row가 있는 run을 demo run으로 선택
- prediction/evaluation/backtest를 같은 demo run 기준으로 확인
- stock search 성공 시 `stock_info or price_data fallback available`로 표시

최종 readiness 결과 요약:

```text
OK   health
OK   cors
FAIL frontend - 원격 서버에 연결할 수 없습니다.
OK   1D prices
OK   indicators
OK   1M prices
OK   stock search
OK   completed runs
OK   demo run - composite-1D-3a44b5e51ed2
OK   prediction
OK   latest run
OK   evaluation
OK   backtest
```

# 6. 프론트 영향

`frontend/src/components/BacktestView.tsx`를 최소 수정했다.

수정 내용:

- 백테스트 화면도 `line_band_composite`, `patchtst` completed run 후보를 같은 기준으로 본다.
- 선택 timeframe 기준으로 backtest row가 있는 run을 우선 사용한다.
- backtest가 없으면 기존처럼 조용한 empty state를 유지한다.

대규모 UI 리디자인은 하지 않았다.

# 7. 검증 결과

프론트 빌드:

```text
cd C:\Users\user\lens\frontend
npm run build
```

결과:

- 성공
- `/` size: 85.6 kB
- First Load JS: 173 kB

백엔드 테스트:

```text
$env:PYTHONPATH="C:\Users\user\lens\backend"
python -m unittest discover backend\tests
```

결과:

- 49 tests OK

추가 확인:

- AAPL 검색 정상
- MSFT 검색 정상
- readiness에서 stock search OK
- readiness에서 backtest OK

# 8. 남은 WARN/FAIL 목록

남은 항목:

- `frontend` readiness: `FAIL - 원격 서버에 연결할 수 없습니다.`

사유:

- 프론트 dev 서버 `127.0.0.1:3000` 응답 확인 단계에서 프로세스 시작/응답 대기가 지연됐다.
- 사용자 지시에 따라 더 이상 기다리지 않고 CP46-P를 closure 처리했다.
- `npm run build`는 성공했으므로 코드 빌드 자체의 실패로 보지는 않는다.

# 9. Fake Data 미사용 확인

이번 CP에서 fake data를 생성하지 않았다.

하지 않은 일:

- 모델 학습 실행
- 예측값 조작
- backtest row 임의 삽입
- DB schema 변경
- ATR/indicator 전체 재계산

# 10. 변경 파일

CP46-P에서 직접 수정한 파일:

- `backend/app/repositories/market_repo.py`
- `backend/tests/test_services.py`
- `frontend/src/components/BacktestView.tsx`
- `scripts/check_demo_readiness.ps1`
- `docs/cp46_product_demo_readiness_cleanup_report.md`

작업 트리에는 이전 CP에서 남은 변경 파일이 다수 있다. 이번 closure에서는 되돌리지 않았다.
