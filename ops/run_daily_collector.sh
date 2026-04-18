#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/daily-collector.log"

mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # 현재 셸에 환경 변수를 주입해 cron에서도 같은 설정으로 실행한다.
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

python collector/pipelines/daily_sync.py "$@" >> "${LOG_FILE}" 2>&1
