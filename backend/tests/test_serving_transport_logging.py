"""비공개 릴리스 다운로드의 서명 URL을 기본 로그에서 차단한다."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_debug_logging_does_not_print_signed_download_urls() -> None:
    root = Path(__file__).resolve().parents[2]
    code = """
import logging
from app.core.logging import configure_logging
configure_logging()
logging.getLogger('httpx').info('https://release-assets.githubusercontent.com/file?sig=private-test')
logging.getLogger('httpcore').debug('private-test-authorization')
logging.getLogger('lens.api').warning('serving-test-warning')
"""
    env = {**os.environ, "PYTHONPATH": str(root / "backend"), "LENS_LOG_LEVEL": "DEBUG"}
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert "private-test" not in result.stdout + result.stderr
    assert "serving-test-warning" in result.stdout
