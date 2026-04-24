# 데이터 디렉터리 규칙

`backend/data/`는 실행 코드가 아니라 데이터 파일만 보관하는 디렉터리다.

## 포함 대상
- `universe/`
- `parquet/`
- `cache/`

## 제외 대상
- 실행 스크립트
- DB 적재 로직
- 피처 계산 코드
- API 코드

실행 코드는 `backend/collector/`, `backend/app/`, `ai/` 아래에 둔다.
