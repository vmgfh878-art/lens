# CP212-LG 후속 검증 - AAPL alignment recheck

검증 시각: 2026-05-27 KST

## 질문

CP212 기준으로 AAPL의 가격, 1D line, 1D band, 1W band, product history가 실제로 같은 serving 상태에서 맞물려 응답하느냐?

## 사용 backend

- local base URL: `http://127.0.0.1:8000`
- Render base URL: `https://lens-backend-7stj.onrender.com`
- local/Render 모두 같은 endpoint 결과를 반환함

## endpoint status

| 항목 | endpoint | local | Render |
|---|---|---:|---:|
| health | `/api/v1/health/live` | 200 | 200 |
| 가격 | `/api/v1/stocks/AAPL/prices?timeframe=1D&limit=10` | 200 | 200 |
| 1D line | `/api/v1/predictions/line/AAPL?days=30` | 200 | 200 |
| 1D band | `/api/v1/predictions/band/1d/AAPL?days=30&horizon=5` | 200 | 200 |
| 1W band | `/api/v1/predictions/band/1w/AAPL?days=60&horizon=4` | 200 | 200 |

## AAPL 날짜 정합성

| 항목 | 최신 기준일 | 비고 |
|---|---:|---|
| 가격 latest date | `2026-05-26` | `market_prices_1d.parquet` 및 API 기준 |
| 1D line latest asof | `2026-05-22` | `cp210_F4_b4_ensemble_mean`, `CP208Z_CP209_F4B4` |
| 1D band latest asof | `2026-05-26` | `tide-1D-ea54dcae654d`, `CP153` |
| 1W band latest asof | `2026-05-22` | `tide_s104_q10q90_param`, `CP178-WFLOCK` |
| product history latest asof | `2026-05-26` | AAPL rows 1,633 |

## serving parquet 확인

| 파일 | 전체 row | 전체 max | AAPL row | AAPL max |
|---|---:|---:|---:|---:|
| `market_prices_1d.parquet` | 137,630 | `2026-05-26` | 275 | `2026-05-26` |
| `predictions_line_1d.parquet` | 178,037 | `2026-05-22` | 378 | `2026-05-22` |
| `predictions_band_1d.parquet` | 594,800 | `2026-05-26` | 1,255 | `2026-05-26` |
| `predictions_band_1w.parquet` | 186,896 | `2026-05-22` | 420 | `2026-05-22` |
| `product_prediction_history_1D.parquet` | 772,837 | `2026-05-26` | 1,633 | `2026-05-26` |

## mismatch

가격과 1D band는 `2026-05-26`으로 맞는다. 1D line은 `2026-05-22`로 남아 있어 CP212의 엄격한 PASS 조건인 `price = 1D line = 1D band`는 만족하지 않는다.

다만 local과 Render endpoint가 같은 값을 반환하고, product history는 `2026-05-26`까지 생성되어 있다. 따라서 현재 문제는 backend endpoint 장애가 아니라 1D line serving artifact 기준일이 price/band보다 뒤처진 상태다.

## 화면 확인

브라우저에서 `http://127.0.0.1:3000/` 기준 AAPL 화면을 확인했다.

- 1D 가격 표시: 확인
- 1D 보수적 기준선 표시: 확인
- 1D AI 밴드 표시: 확인
- 1W AI 밴드 표시: 확인
- 1W 보수적 기준선 비표시: 확인
- frontend static CSS 200: 확인
- browser console error/warn: 확인 시점 기준 0건

## 최종 판정

`WARN_CP212_AAPL_PARTIAL`

사유: endpoint와 화면은 복구되었지만, AAPL 기준 1D line latest asof가 `2026-05-22`이고 가격/1D band latest date는 `2026-05-26`이다. 즉 line/band/price가 모두 같은 거래일이라는 CP212 엄격 PASS에는 아직 도달하지 않았다.
