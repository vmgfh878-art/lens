# CP122-5-P 예측 history 단절 수정 후 차트 UX 확인 보고서

## 1. 목표

CP122-4에서 들어간 prediction history gap 처리와 product latest-only 제외 로직이 실제 주식 보기 화면에서 자연스럽게 보이는지 확인했다.

이번 CP에서는 모델 값, prediction DB, inference, 전략 룰, 백엔드 schema를 수정하지 않았다.

## 2. 수정 파일

- `frontend/src/components/Chart.tsx`
- `docs/cp122_5_chart_history_gap_ux_verification_report.md`

## 3. 최소 UX 보정

차트 legend 문구만 짧게 보정했다.

- 변경 전: `과거 예측`
- 변경 후: `과거 예측 이력`

이유:

- 최신 forecast와 rolling history가 분리되어 표시되므로, 단순히 `과거 예측`보다 `과거 예측 이력`이 사용자가 보기 더 자연스럽다.
- 긴 설명은 추가하지 않았다.

## 4. 현재 차트 history 계약 확인

`Chart.tsx` 기준으로 아래 계약이 유지되어 있음을 확인했다.

- rolling history는 `h1~h5` 전체를 한 series로 이어 그리지 않는다.
- 각 prediction row의 `asof_date`별 대표 horizon 하나만 사용한다.
- 현재 대표 horizon은 h5 우선이다.
- 10일을 초과하는 history gap이 있으면 최신 contiguous 구간만 사용한다.
- `product_latest_only`, `thin_upload`, `storage_contract=product_latest_only` row는 rolling history에서 제외한다.
- future forecast는 latest prediction 1건의 forecast dates만 사용한다.
- future forecast는 최신 가격일 이후 날짜만 표시한다.

## 5. 데이터 확인

API 기준으로 1D BM history에는 아래 gap이 존재했다.

- NVDA: `2026-04-01 -> 2026-05-01`, 30일 gap
- AAPL: `2026-04-02 -> 2026-05-01`, 29일 gap
- MSFT: `2026-04-01 -> 2026-05-01`, 30일 gap

최신 forecast 기준은 세 종목 모두 아래와 같았다.

- 최신 가격일: `2026-05-04`
- line model 기준일: `2026-05-04`
- band model 기준일: `2026-05-04`
- forecast dates: `2026-05-05, 2026-05-06, 2026-05-07, 2026-05-08, 2026-05-11`

따라서 최신 forecast는 최신 캔들 이후에만 표시되는 계약이다.

## 6. 브라우저 확인

### NVDA 1D

- `2026-04-01 -> 2026-05-01` gap이 긴 대각선으로 이어지는 모습은 보이지 않았다.
- 최신 forecast는 최신 캔들 이후 구간에 붙는 형태로 보였다.
- AI 밴드 upper/lower가 gap을 가로지르는 한 선으로 연결되는 문제는 보이지 않았다.
- console error/warn: 0건

### AAPL 1D

- 동일한 history gap이 과거 구간과 최신 구간을 대각선으로 연결하지 않았다.
- rolling history와 future forecast는 선 굵기/스타일로 구분된다.
- console error/warn: 0건

### MSFT 1D

- 1D 차트 표시와 전환은 정상이다.
- history legend는 표시되지 않았는데, 현재 표시 가능한 history layer가 충분하지 않거나 별도 축/범위 조건으로 제한된 상태로 보인다.
- fatal 오류나 console error/warn은 없었다.

## 7. Marker 확인

- 오늘 marker는 표시하지 않는 상태를 유지했다.
- 차트 marker는 `모델 기준일`만 사용한다.
- `모델 기준일`은 예측 시작일이 아니라 prediction의 asof 기준임을 나타내므로, 오늘 marker보다 덜 헷갈린다.

## 8. 1W/1M 전환 확인

- 1W 전환: 과도한 AI 위험 범위 표시 없음
- 1M 전환: 가격 전용 안내 유지
- 1D 복귀: 정상
- 전환 중 console error/warn: 0건

## 9. 검증 명령

```powershell
cd C:\Users\user\lens\frontend
npm run build
```

결과: 통과.

참고:

- 샌드박스 안에서는 Next.js child process spawn이 `EPERM`으로 실패했다.
- 권한 밖에서 같은 `npm run build`를 실행해 실제 빌드 통과를 확인했다.

```powershell
cd C:\Users\user\lens
powershell -ExecutionPolicy Bypass -File .\scripts\check_demo_readiness.ps1
```

최종 결과: 통과.

주요 항목:

- backend health: OK
- CORS: OK
- frontend root: OK
- frontend-static stylesheet: OK
- AAPL 1D prices: OK
- AAPL indicators: OK
- AAPL 1M prices: OK
- LM run/prediction/history/evaluation: OK
- BM run/prediction/history/evaluation: OK

## 10. 실행 중 관찰

- `npm run build` 이후 한 번 readiness에서 백엔드 연결 실패가 발생했다.
- 백엔드를 다시 띄운 뒤 readiness는 정상 통과했다.
- `scripts/start_demo.ps1` 실행 중 `logs/frontend_dev.out.log` 파일이 이미 사용 중이라는 메시지가 있었지만, frontend root와 static CSS는 모두 200이었다.

## 11. 남은 아쉬운 점

- history gap을 최신 contiguous 구간만 남기는 방식이라, gap 이전 과거 예측 이력은 화면에서 생략된다.
- 장기 history를 모두 보이면서 gap만 끊으려면 history segment를 여러 line series로 나누는 방식이 더 좋다.
- 이번 CP에서는 대규모 시각 구조 변경 없이, 현재 안정화된 계약을 검증하고 문구만 보정했다.

## 12. 금지사항 준수

- 모델 값 수정 없음
- prediction DB 수정 없음
- inference 재실행 없음
- 전략 룰 변경 없음
- DB/API/schema 변경 없음
- fake data 생성 없음
- composite 사용 없음
