# GitHub 방식의 코덱스 주간 품질 감사 기준

Lens 워크스페이스에서 매주 일요일 읽기 전용 통합 리스크·데이터 품질 감사를 수행한다. 토요일 10시 백필과 주간(1W) 적재 이후의 로컬 수집·계산 품질과 실제 배포 서빙 상태를 각각 판단한다.

코드·테스트·설정·데이터·파일을 수정하지 않는다. DB 쓰기, 백필·동기화·발행 실행, 학습·보정·추론 결과 저장, 빌드·재시작, 커밋·푸시를 하지 않는다. Slack이나 다른 곳에 메시지를 보내지 않는다. 기존 진단 스크립트와 가벼운 읽기 전용 조회만 사용하고 비밀키·토큰·웹훅 값은 출력하지 않는다. 인증을 위해 토큰 발행·권한 확대·비밀키 파일 작성을 하지 않는다.

먼저 운영 전환 상태를 확인한다. 메인 작업 폴더의 일일 파이프라인 코드와 최신 실행 보고, Render의 /api/v1/health/ready 응답을 대조한다. 비공개 GitHub 릴리스 방식은 별도 브랜치에서 구현되거나 최초 발행이 끝났더라도 메인 병합·Render 읽기 권한 연결·실제 배포 전에는 운영 완료가 아니다. 검사 스크립트가 없거나 Render가 Supabase 또는 기존 출처를 보고하면 'GitHub 릴리스 전환 미완료'로 표시하고 현재 운영 경로에 맞춰 검사한다. 전환 전의 정상 데이터 커밋이나 Supabase 동작을 장애로 오판하지 않는다. R2는 이전 설계이며 현재 목표에 R2 계정이나 과금 설정은 필요 없다.

전환 후의 구조는 로컬 PC에 원본·백필·계산 데이터, backend/data/v1에 매일 생성하는 서빙 결과, 비공개 vmgfh878-art/lens-serving 저장소의 고정 serving-snapshots 릴리스에 첨부파일, Render에 검증된 활성 캐시, Git의 backend/data/bootstrap/v1에 고정 폴백 한 벌이다. 릴리스 본문의 latest·previous 포인터로 최신·이전 스냅샷 두 벌만 보관하며 데이터 커밋과 일일 태그를 만들지 않는다. 업로드 중에는 세 벌이나 불완전 파일이 잠시 존재할 수 있다. Vercel은 화면 배포이며 데이터를 직접 수집하지 않는다. Git 데이터 커밋 날짜나 Supabase 최신성은 전환 후 서빙 정상 기준이 아니다. PC가 꺼져 있으면 마지막 발행본을 계속 제공하지만 새 데이터 수집·계산은 진행되지 않는다.

수집·계산 상태는 토요일 백필과 1W 적재의 마지막 실행 시각, run 상태, sync cursor, selected/success/failed/appended 수, fetch_failed·429·source_empty·adjusted_ohlc_contract_failed를 확인한다. 미완료 또는 실행 중이면 이후 수치를 완료된 정상 상태로 단정하지 않는다. 일일 통합 보고의 수집 단계 결과, github_publish_status, deployed_verify_status, serving_verification을 따로 읽는다. 발행 실패와 서비스 응답 불가는 다른 상태이며 고정 폴백 응답을 새 데이터 발행 성공으로 처리하지 않는다.

검사 스크립트가 메인 작업 폴더에 있으면 .venv의 Python에 -B를 붙여 backend/scripts/audit_serving_snapshot.py --verify-remote-files를 실행한다. 이 스크립트는 로컬 산출 파일 10개의 지문, 고정 폴백의 저장 매니페스트·해시, GitHub 릴리스의 latest 및 원격 파일 해시, Render 활성 스냅샷 ID와 실제 가격·예측 응답을 검사한다. 원격 검사는 GET만 사용하며 로컬에서 기존 GitHub CLI 인증을 메모리로 읽을 수 있다. 인증 또는 네트워크 접근이 없으면 '미검증'으로 표시한다. 폴백을 최신 로컬 데이터로 대신 넣어 검사하지 않는다. 검사 범위는 서빙 계약이며 전체 모델·원본 데이터 품질의 대체물이 아니다.

Render의 source=github_release와 릴리스 latest의 snapshot_id가 일치하고 데이터 최신성도 통과하면 서빙 PASS다. source=bootstrap이면 검증된 고정 폴백인지, 필수 파일과 실제 가격·예측 응답이 있는지 확인하고 최신성도 충족할 때 WARN으로 보고한다. source=github_release라도 상태가 degraded이면 동기화 실패 후 이전 버전 유지인지 확인한다. 데이터가 오래됐거나 비었거나 원격 latest와 활성 버전이 다르거나 폴백 해시 검증 실패 또는 응답 불가이면 FAIL로 보고한다. last_remote_success_at은 현재 백엔드 프로세스가 기억하는 성공 시각이므로 재시작 후 값이 없다는 이유만으로 과거 발행이 전혀 없었다고 판단하지 않는다. 로컬 지문과 원격 지문이 다르면 생성·발행 실행 중 여부와 최근 발행 실패를 확인한다. 첨부파일 크기·SHA-256, 고정 릴리스 하나, 최신·이전 보관 정책을 확인하되 파일 삭제는 하지 않는다.

데이터 품질은 provider/source/timeframe별로 분리한다. 로컬 yfinance 1D/1W/1M의 최신일자, 전체 티커 수, 최신일자 coverage, all-present latest를 확인하고 일부 제품 티커만 최신인지 전체 universe가 최신인지 구분한다. 1M이 학습·full run에 부족하면 차단을 유지한다. 전환 전 또는 명시적으로 여전히 사용하는 경우에만 local parquet와 Supabase의 행 수·최신일자·티커·지표 coverage 차이를 별도로 검사한다. 고정 Git 폴백의 데이터 날짜는 로컬 원본이나 원격 최신 데이터 날짜와 혼합하지 않는다.

가격·지표 품질은 adjusted/raw OHLC ratio sanity, open_ratio/high_ratio/low_ratio p99/max, adjusted OHLC violation, duplicate ticker/date/source, NULL·음수 volume, feature NaN/Inf, target 분포·non-finite, atr_ratio의 존재·non-null coverage·p99/max를 확인한다. 1D/1W/1M resample의 미완성 기간 포함 여부를 점검한다.

feature·학습 계약은 FEATURE_CONTRACT_VERSION, MODEL_N_FEATURES, source feature 수, 모델 컬럼 목록, atr_ratio의 피처 포함 여부, cache manifest schema/version, column mismatch, source hash·stale cache, ticker registry mapping/hash, train/val/test 날짜 overlap·purge gap, target/raw_future_returns 분리를 확인한다. fundamentals·macro·breadth·sector_returns의 결측률·0-fill과 실제 값 coverage도 분리한다.

시스템 리스크는 수집·동기화 계약, feature cache fingerprint, checkpoint 호환성, line_gate/band_gate/combined_gate fallback, composite inference meta, predictions/evaluations/backtests의 run_id/timeframe/ticker/horizon/layer/model 계약, band calibration 저장·재사용, line_inside_band, completed/failed_nan/failed_quality_gate 필터링, latest run 선택, 저장형 inference/backtest 경로를 확인한다. stock search 503, CORS/env/demo 안정성, 인증 없는 진단 엔드포인트, 키 노출, 대용량 cache/checkpoint/gitignore, Windows/CUDA/torch requirements, 프론트와 ai_runs_mock의 run_id 불일치를 점검한다. 새로 바뀐 composite_inference, band_calibration, ticker_registry, storage 경로도 포함한다.

결과는 전체 PASS/WARN/FAIL과 모델 full run 가능 여부를 먼저 설명한다. 운영 전환 상태, 수집·계산 품질, 발행 상태, 실제 서빙 출처·최신성, 폴백 작동을 각각 보고한다. 발견 사항은 P0/P1/P2/P3 순으로 파일·함수·라인 또는 데이터 근거를 붙이고, 이전 실행 대비 신규·해결·유지 문제를 구분한다. 공통 계약은 한 번만 검사한다. '즉시 막아야 할 것', '다음 모델 실험 전 확인', '데이터 재건 이후 확인', '사람이 판단할 항목', '수정하지 않았다는 확인', '읽기 전용으로 실행한 명령'을 간결하게 정리한다. 모르는 것은 미검증으로, 추측은 추측으로 표시한다.
