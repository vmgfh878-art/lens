from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from app.config import GitHubReleaseConfig
from app.services.github_release_storage import GitHubReleaseClient, GitHubReleaseError
from app.services.r2_storage import R2PublishError, publish_snapshot
from app.services.serving_paths import get_serving_data_dir, reset_active_serving_data_dir
from app.services.serving_snapshot import REQUIRED_SERVING_FILES, build_snapshot_manifest
from app.services.serving_sync import (
    R2SyncError,
    get_serving_sync_state,
    inspect_latest_serving_snapshot,
    reset_serving_sync_state,
    sync_latest_serving_snapshot,
)


class FakeGitHub:
    def __init__(self, *, private: bool = True):
        self.private = private
        self.release: dict[str, Any] | None = None
        self.assets: dict[int, dict[str, Any]] = {}
        self.files: dict[int, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_upload: str | None = None
        self.redirect: str | None = None
        self.external_authorization: list[str | None] = []
        self.http = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, str(request.url)))
        if request.url.host != "api.github.com" and request.url.host != "uploads.github.com":
            self.external_authorization.append(request.headers.get("Authorization"))
            asset_id = int(path.rsplit("/", 1)[-1])
            return httpx.Response(200, content=self.files[asset_id])
        if path == "/repos/vmgfh878-art/lens-serving":
            return httpx.Response(
                200,
                json={
                    "private": self.private,
                    "full_name": "vmgfh878-art/lens-serving",
                    "default_branch": "main",
                },
            )
        if "/releases/tags/" in path:
            return httpx.Response(404 if self.release is None else 200, json=self.release or {})
        if path.endswith("/releases") and request.method == "POST":
            self.release = {**json.loads(request.content), "id": 1, "immutable": False}
            return httpx.Response(201, json=self.release)
        if path.endswith("/releases/1") and request.method == "PATCH":
            self.release.update(json.loads(request.content))
            return httpx.Response(200, json=self.release)
        if path.endswith("/releases/1/assets"):
            if request.method == "GET":
                return httpx.Response(200, json=[dict(item) for item in self.assets.values()])
            name = parse_qs(request.url.query.decode())["name"][0]
            if self.fail_upload and self.fail_upload in name:
                return httpx.Response(500)
            payload = request.read()
            asset_id = max(self.assets, default=0) + 1
            asset = {
                "id": asset_id,
                "name": name,
                "size": len(payload),
                "state": "uploaded",
                "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            }
            self.assets[asset_id] = asset
            self.files[asset_id] = payload
            return httpx.Response(201, json=asset)
        if "/releases/assets/" in path:
            asset_id = int(path.rsplit("/", 1)[-1])
            if request.method == "DELETE":
                del self.assets[asset_id]
                del self.files[asset_id]
                return httpx.Response(204)
            if self.redirect:
                return httpx.Response(
                    302, headers={"Location": f"https://{self.redirect}/{asset_id}"}
                )
            return httpx.Response(200, content=self.files[asset_id])
        raise AssertionError(f"예상하지 못한 요청: {request.method} {path}")

    def channel(self) -> dict[str, Any]:
        return json.loads(self.release["body"])


def config() -> GitHubReleaseConfig:
    return GitHubReleaseConfig(LENS_GITHUB_TOKEN="test-token")


def fixture_snapshot(tmp_path: Path, snapshot_id: str) -> tuple[Path, dict[str, Any]]:
    source = tmp_path / snapshot_id
    source.mkdir()
    for name in REQUIRED_SERVING_FILES:
        (source / name).write_bytes(f"{snapshot_id}:{name}".encode())
    return source, build_snapshot_manifest(source, snapshot_id=snapshot_id)


def publisher(fake: FakeGitHub) -> GitHubReleaseClient:
    return GitHubReleaseClient(config(), allow_create=True, http_client=fake.http)


@pytest.fixture(autouse=True)
def reset_runtime():
    reset_active_serving_data_dir()
    reset_serving_sync_state()
    yield
    reset_active_serving_data_dir()
    reset_serving_sync_state()


def test_public_repository_is_rejected_before_any_write() -> None:
    fake = FakeGitHub(private=False)
    with pytest.raises(GitHubReleaseError, match="공개 저장소"):
        publisher(fake)
    assert all(method == "GET" for method, _ in fake.calls)
    assert fake.release is None


def test_publish_then_sync_and_audit_use_only_one_fixed_release(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    source, manifest = fixture_snapshot(tmp_path, "daily-one")
    result = publish_snapshot(source, manifest, config=config(), client=client)
    assert result["uploaded_object_count"] == 11
    assert fake.channel()["latest"]["snapshot_id"] == "daily-one"
    assert fake.channel()["previous"] is None
    assert fake.release["tag_name"] == "serving-snapshots"
    assert not any(
        "/commits" in url or "/contents/" in url or "/git/" in url for _, url in fake.calls
    )
    create_release_count = sum(
        method == "POST" and url.endswith("/releases") for method, url in fake.calls
    )
    assert create_release_count == 1

    synced = sync_latest_serving_snapshot(
        config=config(), client=client, cache_dir=tmp_path / "cache"
    )
    assert synced["source"] == "github_release"
    assert synced["snapshot_id"] == "daily-one"
    assert (get_serving_data_dir() / REQUIRED_SERVING_FILES[0]).read_bytes() == (
        source / REQUIRED_SERVING_FILES[0]
    ).read_bytes()
    before = list(fake.calls)
    audited = inspect_latest_serving_snapshot(config=config(), client=client, verify_files=True)
    assert audited["files_verified"] is True
    assert all(method == "GET" for method, _ in fake.calls[len(before) :])


def test_interrupted_upload_does_not_move_latest_or_delete_old_files(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    first_dir, first = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(first_dir, first, config=config(), client=client)
    channel_before = fake.channel()
    old_ids = set(fake.assets)
    second_dir, second = fixture_snapshot(tmp_path, "daily-two")
    fake.fail_upload = "predictions_band_1d.parquet"
    with pytest.raises(R2PublishError):
        publish_snapshot(second_dir, second, config=config(), client=client)
    assert fake.channel() == channel_before
    assert old_ids.issubset(fake.assets)
    assert not any(method == "DELETE" for method, _ in fake.calls)


def test_three_publications_keep_latest_previous_and_unrelated_assets(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    for snapshot_id in ("daily-one", "daily-two", "daily-three"):
        source, manifest = fixture_snapshot(tmp_path, snapshot_id)
        publish_snapshot(source, manifest, config=config(), client=client)
        retention = client.prune_old_snapshots()
    assert retention["retained_snapshot_ids"] == ["daily-three", "daily-two"]
    assert len(fake.assets) == 22
    assert not any("daily-one" in item["name"] for item in fake.assets.values())
    assert fake.channel()["latest"]["snapshot_id"] == "daily-three"
    assert fake.channel()["previous"]["snapshot_id"] == "daily-two"
    fake.assets[1000] = {"id": 1000, "name": "사용자 별도 파일", "size": 1, "state": "uploaded"}
    fake.files[1000] = b"0"
    client.prune_old_snapshots()
    assert 1000 in fake.assets


def test_republish_same_snapshot_is_idempotent_and_preserves_previous(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    first_dir, first = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(first_dir, first, config=config(), client=client)
    second_dir, second = fixture_snapshot(tmp_path, "daily-two")
    publish_snapshot(second_dir, second, config=config(), client=client)
    result = publish_snapshot(second_dir, second, config=config(), client=client)
    assert result["uploaded_object_count"] == 0
    assert result["skipped_object_count"] == 11
    assert len(fake.assets) == 22
    assert fake.channel()["previous"]["snapshot_id"] == "daily-one"


def test_corrupt_download_does_not_replace_active_snapshot(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    first_dir, first = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(first_dir, first, config=config(), client=client)
    sync_latest_serving_snapshot(config=config(), client=client, cache_dir=tmp_path / "cache")
    active_before = get_serving_data_dir()
    second_dir, second = fixture_snapshot(tmp_path, "daily-two")
    publish_snapshot(second_dir, second, config=config(), client=client)
    asset = next(
        item
        for item in fake.assets.values()
        if item["name"] == "snapshot--daily-two--market_prices_1d.parquet"
    )
    fake.files[asset["id"]] = b"x" * asset["size"]
    with pytest.raises(R2SyncError):
        sync_latest_serving_snapshot(config=config(), client=client, cache_dir=tmp_path / "cache")
    assert get_serving_data_dir() == active_before
    assert get_serving_sync_state()["source"] == "github_release"
    assert get_serving_sync_state()["status"] == "error"
    assert not any(path.name.endswith(".tmp") for path in (tmp_path / "cache").iterdir())


def test_reader_cannot_upload_change_pointer_or_delete_assets(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    source, manifest = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(source, manifest, config=config(), client=client)
    reader = GitHubReleaseClient(config(), http_client=fake.http)
    with pytest.raises(GitHubReleaseError, match="읽기 전용"):
        reader.prune_old_snapshots()
    with pytest.raises(GitHubReleaseError, match="읽기 전용"):
        reader.upload_file(
            str(source / REQUIRED_SERVING_FILES[0]),
            config().bucket,
            "serving/v1/releases/new/ai_runs_mock.json",
            ExtraArgs={"ContentType": "application/json"},
        )
    with pytest.raises(GitHubReleaseError, match="저장소 밖"):
        reader.get_object(Bucket="다른 저장소", Key="serving/v1/latest.json")


@pytest.mark.parametrize(
    "hostname", ["release-assets.githubusercontent.com", "objects.githubusercontent.com"]
)
def test_signed_download_redirect_does_not_forward_github_token(
    tmp_path: Path, hostname: str
) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    source, manifest = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(source, manifest, config=config(), client=client)
    fake.redirect = hostname
    inspected = inspect_latest_serving_snapshot(config=config(), client=client, verify_files=True)
    assert inspected["files_verified"] is True
    assert fake.external_authorization
    assert all(value is None for value in fake.external_authorization)


def test_unknown_redirect_host_is_rejected(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    source, manifest = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(source, manifest, config=config(), client=client)
    fake.redirect = "unexpected.invalid"
    with pytest.raises(GitHubReleaseError, match="리디렉션"):
        inspect_latest_serving_snapshot(config=config(), client=client, verify_files=True)
    assert not fake.external_authorization


def test_channel_identity_and_immutable_release_are_rejected(tmp_path: Path) -> None:
    fake = FakeGitHub()
    client = publisher(fake)
    source, manifest = fixture_snapshot(tmp_path, "daily-one")
    publish_snapshot(source, manifest, config=config(), client=client)
    fake.release["body"] = json.dumps({"protocol": "다른 채널"})
    with pytest.raises(GitHubReleaseError, match="전용 채널"):
        client.prune_old_snapshots()
    fake.release["immutable"] = True
    with pytest.raises(GitHubReleaseError, match="변경 불가"):
        client._load_release(refresh=True)
