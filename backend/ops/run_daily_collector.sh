#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/daily-collector.log"

mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # 로컬 실행과 cron 실행 모두 같은 환경 변수를 사용한다.
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

python -m backend.collector.pipelines.daily_sync "$@" >> "${LOG_FILE}" 2>&1
