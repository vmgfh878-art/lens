CP76-P는 composite 없이 1D 제품 후보를 Line layer와 Band layer로 분리해 화면에 표시하는 프론트 정리 CP다.

## 1. Line/Band layer 분리 결과

- 주식 보기 1D 화면에서 예측선은 PatchTST line 후보 run을, AI 밴드는 CNN-LSTM band 후보 run을 별도 prediction row로 조회한다.
- `Chart`는 `prediction`과 `bandPrediction`을 분리 props로 받아 line과 band를 각각 검증하고 렌더링한다.
- band prediction은 `upper_band_series` / `lower_band_series`만으로 표시 가능하게 했고, line prediction이 없어도 band layer가 유지된다.
- 1M은 price-only 정책을 유지하며, 1M 전환 시 제품 후보 run_id가 우측 provenance에 노출되지 않도록 정리했다.

## 2. 사용한 run_id

| 역할 | run_id | model | timeframe | horizon | feature_set | 상태 |
|---|---|---|---|---:|---|---|
| Line 후보 | `patchtst-1D-efad3c29d803` | PatchTST | 1D | 5 | `full_features` | completed |
| Band 후보 | `cnn_lstm-1D-d0c780dee5e8` | CNN-LSTM | 1D | 5 | `price_volatility_volume` | completed |

## 3. LM prediction 저장 여부

- readiness 기준 `patchtst-1D-efad3c29d803`의 AAPL 1D prediction row는 존재한다.
- 확인 결과: `forecast_dates=5`, `line_series=5`.
- AAPL evaluation row도 1건 확인됐다.
- 이번 CP에서는 inference 실행, DB 쓰기, fake data 생성을 하지 않았다.

## 4. BM prediction 저장 여부

- readiness 기준 `cnn_lstm-1D-d0c780dee5e8`의 AAPL 1D prediction row는 존재한다.
- 확인 결과: `forecast_dates=5`, `upper_band_series=5`, `lower_band_series=5`.
- AAPL evaluation row도 1건 확인됐다.
- 주식 보기에서는 해당 run을 AI 밴드 전용 layer로 사용한다.

## 5. composite 제거/legacy 처리 위치

- StockView의 제품 기본 조회에서 composite fallback 탐색을 제거하고, 고정된 Line/Band 제품 후보 run을 별도로 조회한다.
- TrainingView는 run 목록을 `Line 후보`, `Band 후보`, `Legacy / 기타`로 나누며 composite는 제품 기본 후보가 아니라 legacy/기타 영역에만 놓는다.
- readiness 스크립트는 composite를 정상 제품 후보로 요구하지 않고 `LEGACY_OK`로만 표시한다.
- legacy composite run은 삭제하지 않았다.

## 6. 남은 API 부족분

- 로컬 학습 로그 파일(`config.json`, `metrics.jsonl`, `summary.json`)은 브라우저 프론트에서 직접 읽지 않는다.
- 모델 학습 화면에는 로그 영역과 경로 placeholder를 만들었고, 실제 연결에는 read-only log API가 필요하다.
- overlay bundle API가 아직 없으므로 현재는 프론트가 Line/Band prediction endpoint를 각각 호출한다.
- role별 inference meta 정규화는 다음 모델/백엔드 CP에서 더 단단히 맞출 필요가 있다.

## 7. 검증 결과

- `python -m unittest discover backend\tests`: 53개 통과.
- `npm run build`: 통과.
- `scripts/check_demo_readiness.ps1`: health, CORS, frontend, AAPL 1D 가격, indicator, 1M 가격, stock search, LM run/prediction/evaluation, BM run/prediction/evaluation 모두 OK. composite는 `LEGACY_OK`.
- 브라우저 `http://127.0.0.1:3000` 확인:
  - AAPL 1D: 가격 화면 표시, Line/Band run_id 분리 표시, AI 밴드/보수적 예측 토글 표시, 백엔드 오류 배너 없음.
  - 1M: 가격 전용 문구 표시, 백엔드 오류 배너 없음, Line/Band run_id 노출 없음.
  - 모델 학습: Line 후보/Band 후보/Legacy 기타 그룹 표시, composite는 제품 기본 후보처럼 보이지 않음.

## 8. 다음 CP에서 해야 할 일

- overlay bundle API 설계.
- role별 AI inference meta 정규화.
- rolling AI band history 연결.
- h5/h20 horizon 표시 정책 확정.
- 로컬 학습 로그 read-only API 연결.
