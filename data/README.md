# data 디렉터리 규칙

`data/`는 실행 코드가 아니라 데이터만 보관하는 디렉터리다.

## 포함 대상

- `universe/`
  - 고정 유니버스 CSV
- `parquet/`
  - parquet 스냅샷
- `cache/`
  - 수집 캐시

## 제외 대상

- 파이썬 실행 스크립트
- DB 적재 유틸
- 피처 계산 코드
- API 코드

실행 코드는 `collector/`, `backend/`, `ai/` 아래에 둔다.
