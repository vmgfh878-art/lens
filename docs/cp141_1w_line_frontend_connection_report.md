CP141-P는 CP140-D에서 저장된 1W line-only latest prediction을 제품 화면에 연결하되, DB schema 때문에 들어간 degenerate band 필드를 AI 밴드로 표시하지 않게 막은 작업이다.

## 1. 변경 파일

- `frontend/src/components/StockView.tsx`
  - 1W 제품 line run `patchtst-1W-fe7f05a84c93` 상수를 추가했다.
  - 1W 선택 시 기존 “주간 AI 예측 준비 중” early return을 제거하고, 1W line latest prediction을 조회하도록 변경했다.
  - 1W에서는 product-history를 조회하지 않고, latest line forecast만 차트로 넘긴다.
  - `meta.role === "band_model"`이 아니거나 `band_saved_in_cp140=false`, `band_fields_policy=schema_required_degenerate_equal_to_line`이면 band로 인정하지 않는 guard를 추가했다.
  - 1W 사이드 패널에서 `1W 보수적 예측선 모델 v1`, `예측 기간 4주`, `1W AI 밴드 검증 중`을 표시하도록 정리했다.
  - 1W에서는 AI 밴드 toggle을 disabled로 유지하고 “1W AI 밴드는 검증 중입니다. 저장된 1W 예측은 보수적 예측선만 표시합니다.” 문구를 표시한다.
  - 1W에서는 AI 밴드 폭 보조지표를 후보에서 제외해 band가 있는 것처럼 보이지 않게 했다.

- `frontend/src/components/Chart.tsx`
  - Chart 내부에도 band prediction guard를 추가했다.
  - `role=band_model`이 아닌 prediction row는 upper/lower series가 있어도 band layer로 그리지 않는다.
  - CP140 1W line-only row의 degenerate lower/upper가 차트 AI 밴드로 오인되지 않도록 했다.
  - 1W line forecast legend를 `최신 4주 예측`처럼 timeframe에 맞춰 표시한다.

- `frontend/src/components/TrainingView.tsx`
  - 1W 보수적 예측선 슬롯을 `준비 중`에서 `사용 중` 가능한 제품 슬롯으로 변경했다.
  - 1W line run `patchtst-1W-fe7f05a84c93`를 연결했다.
  - 1W AI 밴드 슬롯은 계속 준비 중 상태로 유지했다.
  - 1W line 상세 설명을 “중기 방향과 하방 위험을 보수적으로 보기 위한 참고선”으로 바꿨다.
  - 1W 상세에서는 horizon을 `4주`로 설명한다.

## 2. 실제 API 확인

- AAPL 1W latest prediction:
  - run_id: `patchtst-1W-fe7f05a84c93`
  - timeframe: `1W`
  - horizon: `4`
  - forecast_dates: 4개
  - conservative_series: 4개
  - line_series: 4개
  - meta.role: `line_model`
  - meta.storage_contract: `product_latest_only`
  - meta.band_saved_in_cp140: `false`
  - meta.band_fields_policy: `schema_required_degenerate_equal_to_line`

- MSFT 1W latest prediction도 같은 계약으로 확인했다.

결론: CP140 row의 lower/upper는 실제 AI 밴드가 아니므로 이번 CP에서 차트 band layer에서 제외했다.

## 3. 주식 보기 동작

- AAPL 1W:
  - 가격 차트 표시 확인
  - `최신 4주 예측` 표시 확인
  - `1W 보수적 예측선 모델 v1` 표시 확인
  - `최신 AI 위험 범위` 미표시 확인
  - AI 밴드 toggle disabled 확인
  - `1W AI 밴드는 검증 중입니다` 안내 확인

- MSFT 1W:
  - 가격 차트 표시 확인
  - `최신 4주 예측` 표시 확인
  - AI 밴드 미표시 확인

- 1D:
  - 기존 `최신 5일 예측` 표시 유지
  - 기존 `최신 AI 위험 범위` 표시 유지
  - 1D line/band product layer 동작 유지

- 1M:
  - 가격 전용 정책 유지
  - AI 예측선/AI 밴드 미표시 확인

## 4. AI 모델 화면

- 1D 보수적 예측선: 사용 중 유지
- 1D AI 밴드: 사용 중 유지
- 1W 보수적 예측선: 사용 중 표시
- 1W AI 밴드: 준비 중 표시
- 1W 보수적 예측선 상세:
  - 제목: `1W 보수적 예측선 v1`
  - 설명: `1W 보수적 예측선은 중기 방향과 하방 위험을 보수적으로 보기 위한 참고선입니다.`
  - 모델 역할 설명에서 horizon을 `4주`로 표시
  - 실행 ID는 상세 정보 접기 영역에만 둔다.

## 5. 검증 결과

- `npm run build`
  - 첫 실행은 Codex sandbox의 Next.js worker `spawn EPERM`으로 실패
  - 승인 후 샌드박스 밖에서 재실행했고 통과

- `scripts/check_demo_readiness.ps1`
  - health OK
  - CORS OK
  - frontend 200 OK
  - frontend-static stylesheet 200 OK
  - 1D price/indicator OK
  - 1M price-only OK
  - 1D LM/BM run, prediction, evaluation, history OK

- 브라우저 확인
  - AAPL 1W 확인
  - MSFT 1W 확인
  - 1D 기존 line/band 확인
  - 1M price-only 확인
  - AI 모델 화면 1W line 슬롯 사용 중 확인
  - browser console error/warn 0건
  - frontend static CSS 200 확인

## 6. 남은 리스크

- 1W AI 밴드는 아직 제품 후보가 없으므로 표시하지 않는다.
- 1W product-history는 아직 연결하지 않았다. 이번 CP는 latest line-only forecast 연결만 수행했다.
- 1W line row에는 schema_required degenerate band 필드가 계속 존재하므로, 앞으로도 band guard를 유지해야 한다.
- `scripts/start_demo.ps1` 실행 중 기존 `logs/frontend_dev.out.log` 파일 잠금으로 프론트 재시작 스크립트가 한 번 실패했다. 이후 수동으로 백엔드/프론트를 다시 띄워 readiness와 브라우저 검증은 완료했다.
