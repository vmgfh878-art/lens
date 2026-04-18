#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/daily-sync.log"

mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # 로컬 또는 서버 환경에서 .env 기반 실행이 가능하도록 현재 셸에 주입한다.
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

python db/07_sync_source_to_lens.py "$@" >> "${LOG_FILE}" 2>&1
