"""원격 서빙 저장소를 선택한다. 새 운영 경로는 비공개 GitHub 릴리스다."""

from __future__ import annotations

import os
from typing import Any

from app.config import GitHubReleaseConfig, R2Config, get_github_release_config, get_r2_config

RemoteServingConfig = GitHubReleaseConfig | R2Config


def get_remote_serving_config() -> RemoteServingConfig:
    github = get_github_release_config()
    selected = os.environ.get("LENS_SERVING_STORAGE", "").strip().lower()
    if selected not in {"", "github_release", "r2"}:
        raise ValueError("LENS_SERVING_STORAGE 설정이 올바르지 않습니다.")
    if selected == "github_release" or (not selected and github.configured):
        return github
    r2 = get_r2_config()
    return r2 if selected == "r2" or r2.configured else github


def remote_serving_source(config: RemoteServingConfig) -> str:
    return "github_release" if isinstance(config, GitHubReleaseConfig) else "r2"


def create_serving_storage_client(config: RemoteServingConfig) -> Any:
    if isinstance(config, GitHubReleaseConfig):
        from app.services.github_release_storage import GitHubReleaseClient

        return GitHubReleaseClient(config)
    from app.services.r2_storage import create_r2_client

    return create_r2_client(config)
