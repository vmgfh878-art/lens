# R2 서빙 스냅샷 전환 기준선

기록일: 2026-09-22 KST

## 저장소 기준

- 기준 커밋: `f068785c238be2821b8c6aa657857a02d8e207e8`
- 기준 브랜치: `codex/r2-serving-snapshots`
- 2026-09-01 이후 `backend/data/v1` 변경 커밋: 33개
- 로컬 Git 느슨한 객체: 4.06 GiB
- 로컬 Git 압축 객체: 216.21 MiB

## 운영 서빙 묶음

고정 백업 `predictions_line_1d_cp175_frozen_backup.parquet`을 제외한 운영 파일의 합계는
45,021,920바이트다.

| 파일 | 바이트 | SHA-256 |
| --- | ---: | --- |
| `ai_runs_mock.json` | 1,006,485 | `08bbac2d505f2692330080611f373f714ff76bb92a2e3d2633615cf8cd92938b` |
| `market_indicators_1d.parquet` | 10,647,009 | `574c38c5861cfb8a673b84ff1c28278ab68a4100512de18bb31366370774b119` |
| `market_prices_1d.parquet` | 3,162,468 | `f484915ed8261fb29538da72e2db421c0173a19ba8a4e5a2b70f22f056bdaa35` |
| `market_prices_1w.parquet` | 189,336 | `97b6891e7b2846a7987a0444995daaffeba3295b9d0cbd8be6158d24ee696875` |
| `market_stock_info.parquet` | 5,312 | `fe4344e1d484ada5bff6fa8b9a1c8653be9beefceacb197eb30bca77992a07fe` |
| `predictions_band_1d.parquet` | 14,078,475 | `c89a5643cba03e22430391096518b32f0c860469fb20d2abbb9b85defc9b6fbb` |
| `predictions_band_1w.parquet` | 4,630,115 | `ee6704bc17cb8b287e4940c1ee68907ec74b281ff2e08ffcec49c8318d1192e0` |
| `predictions_line_1d.parquet` | 4,995,093 | `1b8b245b785079998b00082bc076d5e6dc066e73e749eb4495b91fe84b2c2a4f` |
| `product_prediction_history_1D.manifest.json` | 402 | `60e99631efbda160e2043a4baf79598ca12fb4804810432e8e9a4b98cc6ff46b` |
| `product_prediction_history_1D.parquet` | 6,307,225 | `227ffd5d3bfff2346c6f75ba4cda5df6013bed0344c81522142c927dd5a2d3ac` |

## 데이터 기준

| 자료 | 행 수 | 종목 수 | 최소 날짜 | 최대 날짜 |
| --- | ---: | ---: | --- | --- |
| 1일 가격 | 137,506 | 502 | 2025-08-18 | 2026-09-21 |
| 1주 가격 | 3,700 | 100 | 2025-08-22 | 2026-05-01 |
| 1일 지표 | 137,435 | 501 | 2025-08-18 | 2026-09-21 |
| 1일 라인 예측 | 218,417 | 472 | 2024-11-12 | 2026-09-21 |
| 1일 밴드 예측 | 594,140 | 474 | 2025-09-22 | 2026-09-21 |
| 1주 밴드 예측 | 186,408 | 444 | 2024-09-20 | 2026-09-18 |
| 제품 예측 이력 | 812,557 | 474 | 2024-11-12 | 2026-09-21 |

## 테스트 기준

- 명령: `python -m pytest backend/tests -q -p no:cacheprovider`
- 결과: 196개 통과, 7개 건너뜀, 3개 경고
- 응답 스냅샷: 9개 통과

첫 실행은 격리 작업 트리의 테스트용 캐시 디렉터리 쓰기 권한이 없어 수집 단계에서 실패했다.
권한을 허용한 동일 명령 재실행은 정상 통과했다. 이는 코드 실패가 아니라 작업 트리 쓰기 권한 문제였다.
