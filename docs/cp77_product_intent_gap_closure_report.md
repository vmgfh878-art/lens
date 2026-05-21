CP77-P는 CP76의 Line/Band layer 분리를 제품 의도와 다시 대조하고, fake 없이 바로 닫을 수 있는 화면/조회 계약 gap을 반영한 CP다.

## 1. 의도 대비 완료/부분/미완료 표

### 주식 보기

| 항목 | 상태 | 확인 결과 |
|---|---|---|
| Line layer 최신 h1~h5 forecast | 완료 | `patchtst-1D-efad3c29d803` latest prediction에서 `forecast_dates=5`, `line_series=5` 확인 |
| Band layer 최신 h1~h5 forecast | 완료 | `cnn_lstm-1D-d0c780dee5e8` latest prediction에서 `forecast_dates=5`, upper/lower 각각 5개 확인 |
| rolling line history | 부분 완료 | 새 read-only history API로 최근 90 row 표시. 현재는 h1 rolling point 중심의 얇은 history 선 |
| rolling band history | 부분 완료 | 새 read-only history API로 최근 90 row의 upper/lower h1 point를 얇은 history 선으로 표시 |
| AI band width history | 완료 | BM history upper-lower로 `AI 밴드 폭` 보조지표 표시 |
| 1M에서 AI run_id 숨김 | 완료 | 브라우저 1M 확인: Line/Band run_id 미노출, price-only 문구 표시 |
| composite 기본 후보 차단 | 완료 | StockView 제품 기본 조회에서 composite 미사용, readiness는 `LEGACY_OK`로만 표시 |

### 보조지표

| 항목 | 상태 | 확인 결과 |
|---|---|---|
| RSI/MACD x축 동기화 | 완료 | 가격 차트 timeline과 같은 날짜 범위로 하단 패널 표시 |
| ATR/atr_ratio 표시 가능 | 완료 | AAPL 1D `atr_ratio_non_null=300/300`, 값이 있으면 메뉴에 표시 가능 |
| 초심자용 설명 | 완료 | RSI, MACD, ATR, AI 밴드 폭 설명 문구 정리 |
| 어색한 “3개 선택” 문구 제거 | 완료 | 현재 화면에 해당 문구 없음 |

### 모델 학습 화면

| 항목 | 상태 | 확인 결과 |
|---|---|---|
| Line/Band/Legacy 분리 | 완료 | `Line 후보`, `Band 후보`, `Legacy / 기타` 그룹 표시 |
| run_id/model/role/horizon/feature_set/파라미터 | 완료 | 제품 후보 카드와 상세 패널에 표시 |
| line metric과 band metric 분리 | 완료 | 선택 run role에 맞는 metric block 우선 표시, 제품 후보 카드도 role별 metric만 표시 |
| baseline 비교 | 미완료 | 실제 baseline API/DB 응답이 없어 placeholder만 표시 |
| W&B 상태 안전 표시 | 완료 | `wandb_status` 객체 저장 run도 status 문자열로 정규화해 표시 |

### 데이터셋/라이브러리/리포트

| 항목 | 상태 | 확인 결과 |
|---|---|---|
| 데이터셋 화면 | 미완료 | 이번 CP에서 신규 페이지 구현 금지. backlog로 정리 |
| 라이브러리 화면 | 미완료 | 이번 CP에서 신규 페이지 구현 금지. backlog로 정리 |
| 리포트 화면 | 미완료 | 이번 CP에서 신규 페이지 구현 금지. backlog로 정리 |

## 2. 이번 CP에서 실제 반영한 항목

- `GET /api/v1/stocks/{ticker}/predictions/history?run_id=...&limit=...` read-only API 추가.
- StockView에서 Line/Band rolling history를 각각 조회.
- Chart에서 latest forecast와 rolling history를 구분해 표시.
- 예측선 후보가 현재 가격 범위를 벗어나면 값은 바꾸지 않고 line layer만 숨김.
- BM history 기반 `AI 밴드 폭` 보조지표 추가.
- 오른쪽 예측 패널에 기준일, 예측 5거래일, Line 모델, Band 모델, Line 판정 표시.
- TrainingView에 제품 후보 카드, baseline placeholder, role별 metric 표시 보강.
- readiness에 LM/BM history row와 AI band width 계산 가능 여부 추가.

## 3. 반영하지 못한 항목과 이유

- rolling history는 현재 h1 point 중심으로 표시한다. h1~h5 전체 forecast fan을 과거 row마다 모두 그리면 화면 밀도와 성능 부담이 커서 다음 설계에서 별도 표시 정책이 필요하다.
- baseline 비교는 fake 값을 만들 수 없어 placeholder로만 표시했다.
- 데이터셋/라이브러리/리포트 화면은 이번 CP 범위가 아니므로 신규 페이지를 만들지 않았다.

## 4. 필요한 API/backlog

- overlay bundle API: Line/Band latest + history + evaluation을 한 번에 내려주는 제품용 read-only endpoint.
- baseline metric API: 모델 후보와 단순 baseline, 기존 후보를 같은 기준으로 비교하는 응답.
- rolling history 표시 정책: h1 point, h1~h5 fan, 기준일별 trail 중 제품 기본 표현 결정.
- local training log API: `config.json`, `metrics.jsonl`, `summary.json`을 화면에서 read-only로 읽는 경로.

## 5. Line/Band 제품 후보 run_id

| layer | run_id | 상태 |
|---|---|---|
| Line layer | `patchtst-1D-efad3c29d803` | completed, `completed_line_watch`로 화면 해석 |
| Band layer | `cnn_lstm-1D-d0c780dee5e8` | completed |
| Composite | `composite-1D-3a44b5e51ed2` | `LEGACY_OK`, 제품 기본 미사용 |

## 6. rolling history 표시 여부

- LM history: AAPL 기준 90 rows 확인.
- BM history: AAPL 기준 90 rows 확인.
- 차트 legend에 “얇은 선은 최근 rolling prediction history” 표시.
- composite row는 history 조회에 사용하지 않는다.

## 7. AI band width 표시 여부

- BM history upper/lower band에서 `upper - lower`를 계산한다.
- AAPL 1D 기준 90개 BM row에서 계산 가능.
- 보조지표 패널에 `AI 밴드 폭`으로 표시한다.
- 설명: “예측 범위가 넓을수록 모델이 보는 변동/불확실성이 큰 구간입니다.”

## 8. composite legacy 처리 확인

- 제품 기본 조회와 StockView overlay는 composite를 사용하지 않는다.
- 모델 학습 화면에서는 Legacy/기타 그룹으로 분리한다.
- readiness는 composite를 정상 제품 후보가 아니라 `LEGACY_OK`로만 표시한다.
- legacy run 삭제나 DB 변경은 하지 않았다.

## 9. 데이터셋/라이브러리/리포트 backlog

- 데이터셋 화면: 가격 데이터, 보조지표, 재무/거시/시장 폭, feature_set, 결측/백필 상태.
- 라이브러리 화면: PatchTST 설명, CNN-LSTM 설명, Line model과 Band model 차이, composite를 제품 기본에서 내린 이유.
- 리포트 화면: 실패에서 배운 점, 데이터 오류 수정, line/band 분리 결정, 현재 제품 후보, 남은 리스크.

## 10. 검증 결과

- `python -m unittest discover backend\tests`: 54개 통과.
- `npm run build`: 통과.
- `scripts/check_demo_readiness.ps1`: health/CORS/frontend/가격/indicator/search/LM/BM run/prediction/evaluation/history/band width 모두 OK, composite는 `LEGACY_OK`.
- 브라우저 확인:
  - AAPL 1D: Line/Band run 표시, AI 밴드 폭 표시, rolling history legend 표시, 예측 5거래일 표시, 백엔드 오류 없음.
  - 1M: price-only 문구 표시, Line/Band run_id 숨김, 백엔드 오류 없음.
  - 모델 학습: 제품 후보 카드, Line/Band/Legacy 분리, baseline placeholder, W&B status 표시 확인.

## 11. 금지 사항 준수

- 모델 학습 실행 없음.
- inference 실행 없음.
- DB 쓰기 없음.
- fake data 생성 없음.
- composite 제품 기본 복구 없음.
- 대규모 디자인 리워크 없음.
