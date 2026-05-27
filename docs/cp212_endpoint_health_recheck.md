# CP212-LG 후속 검증 - endpoint health recheck

검증 시각: 2026-05-27 KST

## 검증 범위

- 대상 ticker: AAPL
- local backend base URL: `http://127.0.0.1:8000`
- Render backend base URL: `https://lens-backend-7stj.onrender.com`
- 목적: CP212 serving 상태 확인 전, 동일 backend 인스턴스에서 필수 endpoint가 동시에 응답하는지 확인

## 현재 backend 상태

- local 8000 listener: 있음
- local backend health: 200
- Render backend health: 200
- 새 refresh/새 학습/DB write: 수행하지 않음

## local endpoint 결과

| endpoint | 결과 | 확인값 |
|---|---:|---|
| `/api/v1/health/live` | 200 | health 응답 정상 |
| `/api/v1/stocks/AAPL/prices?timeframe=1D&limit=10` | 200 | latest date `2026-05-26` |
| `/api/v1/predictions/line/AAPL?days=30` | 200 | latest asof `2026-05-22`, model `cp210_F4_b4_ensemble_mean` |
| `/api/v1/predictions/band/1d/AAPL?days=30&horizon=5` | 200 | latest asof `2026-05-26`, latest forecast `2026-06-02` |
| `/api/v1/predictions/band/1w/AAPL?days=60&horizon=4` | 200 | latest asof `2026-05-22`, model `tide_s104_q10q90_param` |

## Render endpoint 결과

| endpoint | 결과 | 확인값 |
|---|---:|---|
| `/api/v1/health/live` | 200 | health 응답 정상 |
| `/api/v1/stocks/AAPL/prices?timeframe=1D&limit=10` | 200 | latest date `2026-05-26` |
| `/api/v1/predictions/line/AAPL?days=30` | 200 | latest asof `2026-05-22`, model `cp210_F4_b4_ensemble_mean` |
| `/api/v1/predictions/band/1d/AAPL?days=30&horizon=5` | 200 | latest asof `2026-05-26`, latest forecast `2026-06-02` |
| `/api/v1/predictions/band/1w/AAPL?days=60&horizon=4` | 200 | latest asof `2026-05-22`, model `tide_s104_q10q90_param` |

## 로그 참고

가장 최근 CP212 통합 refresh 로그는 다음 상태였다.

- `append`: PASS
- `market_snapshot`: PASS
- `band_refresh`: PASS
- `line_export`: PASS
- `product_history_rebuild`: PASS
- `ai_runs_mock_rebuild`: PASS
- `admin reload`: SKIPPED_TOKEN_MISSING
- 최종: `WARN_UNIFIED_REFRESH_PARTIAL`

즉 로컬 serving 산출물 생성은 통과했고, 현재 local/Render endpoint도 모두 200을 반환한다. 다만 reload token 미설정으로 자동 실행 직후 live backend cache reload는 건너뛰어진 이력이 있다.

## 판정

`PASS_CP212_ENDPOINT_HEALTH`

local과 Render 모두 AAPL 가격, 1D line, 1D band, 1W band endpoint가 200을 반환한다.
