"""고정된 비공개 릴리스에서 검증 스냅샷 두 벌만 관리한다.

파일은 릴리스 첨부파일로 저장하고, 최신 포인터는 릴리스 본문을 한 번에
갱신한다. 데이터 커밋이나 일일 태그를 만들지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

import httpx
from app.config import GitHubReleaseConfig
from app.services.r2_storage import (
    serialize_latest_pointer,
    snapshot_object_keys,
    validate_latest_pointer,
)
from app.services.serving_snapshot import (
    REQUIRED_SERVING_FILES,
    SNAPSHOT_MANIFEST_FILENAME,
    parse_snapshot_manifest,
)

_CHANNEL_PROTOCOL = "lens-serving-snapshots"
_MAX_ASSET_BYTES = 128 * 1024 * 1024
_MAX_JSON_BYTES = 1024 * 1024
_FILE_NAMES = (*REQUIRED_SERVING_FILES, SNAPSHOT_MANIFEST_FILENAME)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOWNLOAD_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}


class GitHubReleaseError(RuntimeError):
    """비공개 릴리스 접근이나 발행을 안전하게 완료하지 못했다."""


class _ObjectNotFound(GitHubReleaseError):
    response = {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}


class _DownloadBody:
    """다운로드 응답을 제한된 크기의 스트림으로 노출한다."""

    def __init__(self, response: httpx.Response, expected_size: int):
        self.response = response
        self.iterator = response.iter_bytes(1024 * 1024)
        self.buffer = bytearray()
        self.total = 0
        self.expected_size = expected_size
        self.finished = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        while not self.finished and (size < 0 or len(self.buffer) < size):
            try:
                chunk = next(self.iterator)
            except StopIteration:
                self.finished = True
                if self.total != self.expected_size:
                    self.close()
                    raise GitHubReleaseError("릴리스 파일의 다운로드 크기가 다릅니다.") from None
                break
            self.total += len(chunk)
            if self.total > self.expected_size:
                self.close()
                raise GitHubReleaseError("릴리스 파일이 선언된 크기를 초과합니다.")
            self.buffer.extend(chunk)
        length = len(self.buffer) if size < 0 else min(size, len(self.buffer))
        result = bytes(self.buffer[:length])
        del self.buffer[:length]
        return result

    def close(self) -> None:
        self.response.close()


def resolve_github_config(
    config: GitHubReleaseConfig, *, allow_cli_token: bool = False
) -> GitHubReleaseConfig:
    """로컬 발행기만 기존 CLI 인증을 사용할 수 있다. 토큰은 출력하지 않는다."""
    if config.token:
        return config
    if allow_cli_token:
        try:
            result = subprocess.run(
                ["gh", "auth", "token", "--hostname", "github.com"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitHubReleaseError("로컬 GitHub CLI 인증을 읽지 못했습니다.") from exc
        if result.returncode == 0 and result.stdout.strip():
            return config.model_copy(update={"token": result.stdout.strip()})
    raise GitHubReleaseError("LENS_GITHUB_TOKEN 또는 로컬 GitHub CLI 인증이 필요합니다.")


class GitHubReleaseClient:
    """기존 파일 해시 검증 계약에 맞춘 릴리스 첨부파일 접근기."""

    def __init__(
        self,
        config: GitHubReleaseConfig,
        *,
        allow_cli_token: bool = False,
        allow_create: bool = False,
        http_client: httpx.Client | None = None,
    ):
        self.config = resolve_github_config(config, allow_cli_token=allow_cli_token)
        self.allow_create = allow_create
        self.owned_client = http_client is None
        self.http = http_client or httpx.Client(timeout=120, follow_redirects=False)
        self.release: dict[str, Any] | None = None
        self.assets: list[dict[str, Any]] | None = None
        try:
            repository = self._json("GET", f"/repos/{self.config.repository}")
            if (
                repository.get("private") is not True
                or str(repository.get("full_name", "")).lower() != self.config.repository.lower()
            ):
                raise GitHubReleaseError("공개 저장소에는 서빙 파일을 발행하거나 읽지 않습니다.")
            self.default_branch = repository.get("default_branch")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.owned_client:
            self.http.close()

    def _headers(self, *, binary: bool = False) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.token}",
            "User-Agent": "Lens-Serving-Snapshots",
            "X-GitHub-Api-Version": "2026-03-10",
            "Accept": "application/octet-stream" if binary else "application/vnd.github+json",
        }

    def _json(self, method: str, path: str, *, payload: Any = None) -> Any:
        try:
            response = self.http.request(
                method,
                f"https://api.github.com{path}",
                headers=self._headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise GitHubReleaseError("GitHub API에 연결하지 못했습니다.") from exc
        if response.status_code == 404:
            raise _ObjectNotFound("요청한 비공개 릴리스 또는 첨부파일이 없습니다.")
        if not 200 <= response.status_code < 300:
            raise GitHubReleaseError(f"GitHub API 요청 실패: HTTP {response.status_code}")
        if response.status_code == 204:
            return None
        if len(response.content) > _MAX_JSON_BYTES:
            raise GitHubReleaseError("GitHub 메타데이터 응답이 너무 큽니다.")
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubReleaseError("GitHub 메타데이터 JSON이 손상됐습니다.") from exc

    @staticmethod
    def _empty_channel() -> dict[str, Any]:
        return {
            "protocol": _CHANNEL_PROTOCOL,
            "schema_version": 1,
            "latest": None,
            "previous": None,
        }

    def _load_release(self, *, refresh: bool = False) -> dict[str, Any]:
        if self.release is not None and not refresh:
            return self.release
        path = f"/repos/{self.config.repository}/releases/tags/{quote(self.config.release_tag)}"
        try:
            release = self._json("GET", path)
        except _ObjectNotFound:
            if not self.allow_create:
                raise
            if not self.default_branch:
                raise GitHubReleaseError("초기 안내 문서 커밋이 먼저 필요합니다.") from None
            release = self._json(
                "POST",
                f"/repos/{self.config.repository}/releases",
                payload={
                    "tag_name": self.config.release_tag,
                    "target_commitish": self.default_branch,
                    "name": "렌즈 서빙 스냅샷",
                    "body": json.dumps(self._empty_channel(), ensure_ascii=False),
                    "draft": False,
                    "prerelease": True,
                    "make_latest": "false",
                },
            )
        if (
            not isinstance(release, dict)
            or type(release.get("id")) is not int
            or release.get("id", 0) <= 0
            or release.get("tag_name") != self.config.release_tag
            or release.get("draft") is not False
        ):
            raise GitHubReleaseError("고정 릴리스의 식별 정보가 올바르지 않습니다.")
        if self.allow_create and release.get("immutable") is True:
            raise GitHubReleaseError("변경 불가 릴리스에는 일일 스냅샷을 갱신할 수 없습니다.")
        self.release = release
        self.assets = None
        return release

    def _channel(self, *, refresh: bool = False) -> dict[str, Any]:
        release = self._load_release(refresh=refresh)
        try:
            channel = json.loads(release.get("body") or "")
        except (TypeError, ValueError) as exc:
            raise GitHubReleaseError("고정 릴리스의 채널 본문이 손상됐습니다.") from exc
        if (
            not isinstance(channel, dict)
            or channel.get("protocol") != _CHANNEL_PROTOCOL
            or channel.get("schema_version") != 1
        ):
            raise GitHubReleaseError("렌즈 전용 채널이 아닌 릴리스는 변경하지 않습니다.")
        for name in ("latest", "previous"):
            pointer = channel.get(name)
            if pointer is not None:
                validate_latest_pointer(pointer)
                keys = snapshot_object_keys(self.config, pointer["snapshot_id"])
                if pointer["manifest_key"] != keys["manifest_key"]:
                    raise GitHubReleaseError("채널 포인터가 다른 스냅샷 경로를 가리킵니다.")
        return channel

    def _list_assets(self) -> list[dict[str, Any]]:
        if self.assets is not None:
            return self.assets
        release = self._load_release()
        assets = []
        for page in range(1, 11):
            batch = self._json(
                "GET",
                f"/repos/{self.config.repository}/releases/{release['id']}/assets"
                f"?per_page=100&page={page}",
            )
            if not isinstance(batch, list) or not all(isinstance(item, dict) for item in batch):
                raise GitHubReleaseError("릴리스 첨부파일 목록이 올바르지 않습니다.")
            assets.extend(batch)
            if len(batch) < 100:
                self.assets = assets
                return assets
        raise GitHubReleaseError("릴리스 첨부파일이 예상한 최대 개수를 넘었습니다.")

    def _asset_name(self, key: str) -> str:
        prefix = f"{self.config.object_prefix}/releases/"
        parts = key.removeprefix(prefix).split("/")
        if (
            not key.startswith(prefix)
            or len(parts) != 2
            or not _ID_PATTERN.fullmatch(parts[0])
            or parts[1] not in _FILE_NAMES
        ):
            raise GitHubReleaseError("서빙 스냅샷 계약 밖의 객체 경로입니다.")
        return f"snapshot--{parts[0]}--{parts[1]}"

    def _asset(self, key: str) -> dict[str, Any]:
        name = self._asset_name(key)
        matching = [item for item in self._list_assets() if item.get("name") == name]
        if not matching:
            raise _ObjectNotFound(f"릴리스 첨부파일이 없습니다: {name}")
        if len(matching) != 1:
            raise GitHubReleaseError("동일한 릴리스 첨부파일 이름이 중복됐습니다.")
        asset = matching[0]
        if (
            type(asset.get("id")) is not int
            or asset["id"] <= 0
            or type(asset.get("size")) is not int
            or not 0 < asset["size"] <= _MAX_ASSET_BYTES
            or asset.get("state") != "uploaded"
        ):
            raise GitHubReleaseError("완성되지 않았거나 너무 큰 릴리스 파일입니다.")
        return asset

    def _check_bucket(self, bucket: str) -> None:
        if bucket != self.config.repository:
            raise GitHubReleaseError("설정한 비공개 저장소 밖에는 접근하지 않습니다.")

    def _is_latest(self, key: str) -> bool:
        return key == f"{self.config.object_prefix}/latest.json"

    def _download_body(self, asset: Mapping[str, Any]) -> _DownloadBody:
        url = f"https://api.github.com/repos/{self.config.repository}/releases/assets/{asset['id']}"
        try:
            request = self.http.build_request("GET", url, headers=self._headers(binary=True))
            response = self.http.send(request, stream=True, follow_redirects=False)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location", "")
                response.close()
                target = urlparse(location)
                if (
                    target.scheme != "https"
                    or target.hostname not in _DOWNLOAD_HOSTS
                    or target.username is not None
                    or target.password is not None
                ):
                    raise GitHubReleaseError("허용하지 않은 다운로드 리디렉션입니다.")
                # 서명된 다운로드 URL에는 GitHub 토큰을 전달하지 않는다.
                request = self.http.build_request("GET", location)
                response = self.http.send(request, stream=True, follow_redirects=False)
            if response.status_code != 200:
                status_code = response.status_code
                response.close()
                raise GitHubReleaseError(f"릴리스 다운로드 실패: HTTP {status_code}")
            return _DownloadBody(response, int(asset["size"]))
        except httpx.HTTPError as exc:
            raise GitHubReleaseError("릴리스 파일을 다운로드하지 못했습니다.") from exc

    def _asset_hash(self, asset: dict[str, Any]) -> str:
        digest = str(asset.get("digest", "")).removeprefix("sha256:")
        if _HASH_PATTERN.fullmatch(digest):
            return digest
        # 서버가 지문을 제공하지 않는 경우 실제 파일 바이트로 검사한다.
        body = self._download_body(asset)
        calculated = hashlib.sha256()
        try:
            while chunk := body.read(1024 * 1024):
                calculated.update(chunk)
        finally:
            body.close()
        asset["digest"] = f"sha256:{calculated.hexdigest()}"
        return calculated.hexdigest()

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._check_bucket(Bucket)
        if self._is_latest(Key):
            pointer = self._channel(refresh=True).get("latest")
            if pointer is None:
                raise _ObjectNotFound("최신 스냅샷이 아직 없습니다.")
            payload = serialize_latest_pointer(pointer)
            return {
                "ContentLength": len(payload),
                "Metadata": {"sha256": hashlib.sha256(payload).hexdigest()},
            }
        asset = self._asset(Key)
        return {"ContentLength": asset["size"], "Metadata": {"sha256": self._asset_hash(asset)}}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._check_bucket(Bucket)
        if self._is_latest(Key):
            pointer = self._channel(refresh=True).get("latest")
            if pointer is None:
                raise _ObjectNotFound("최신 스냅샷이 아직 없습니다.")
            payload = serialize_latest_pointer(pointer)
            return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}
        asset = self._asset(Key)
        return {"ContentLength": asset["size"], "Body": self._download_body(asset)}

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        response = self.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            with Path(destination).open("wb") as output:
                while chunk := body.read(1024 * 1024):
                    output.write(chunk)
        finally:
            body.close()

    def _upload(self, key: str, content: Any, size: int, content_type: str) -> None:
        if not self.allow_create:
            raise GitHubReleaseError("읽기 전용 클라이언트로는 파일을 발행할 수 없습니다.")
        if not 0 < size <= _MAX_ASSET_BYTES:
            raise GitHubReleaseError("업로드 파일 크기가 허용 범위를 벗어납니다.")
        self._channel()
        release = self._load_release()
        name = self._asset_name(key)
        url = (
            f"https://uploads.github.com/repos/{self.config.repository}/releases/{release['id']}/assets?"
            + urlencode({"name": name})
        )
        headers = {**self._headers(), "Content-Type": content_type, "Content-Length": str(size)}
        try:
            response = self.http.post(url, headers=headers, content=content)
        except httpx.HTTPError as exc:
            raise GitHubReleaseError("릴리스 업로드에 실패했습니다.") from exc
        if response.status_code != 201:
            raise GitHubReleaseError(f"릴리스 업로드 실패: HTTP {response.status_code}")
        try:
            asset = response.json()
        except ValueError as exc:
            raise GitHubReleaseError("업로드 응답 JSON이 올바르지 않습니다.") from exc
        if not isinstance(asset, dict) or asset.get("name") != name or asset.get("size") != size:
            raise GitHubReleaseError("업로드 응답 파일 정보가 다릅니다.")
        if self.assets is not None:
            self.assets.append(asset)

    def upload_file(
        self, source: str, bucket: str, key: str, *, ExtraArgs: Mapping[str, Any]
    ) -> None:
        self._check_bucket(bucket)
        path = Path(source)
        with path.open("rb") as content:
            self._upload(key, content, path.stat().st_size, str(ExtraArgs["ContentType"]))

    def _verify_pointer_files(self, pointer: Mapping[str, Any]) -> None:
        keys = snapshot_object_keys(self.config, str(pointer["snapshot_id"]))
        if pointer["manifest_key"] != keys["manifest_key"]:
            raise GitHubReleaseError("최신 포인터의 매니페스트 경로가 다릅니다.")
        response = self.get_object(Bucket=self.config.bucket, Key=keys["manifest_key"])
        body = response["Body"]
        try:
            payload = body.read(_MAX_JSON_BYTES + 1)
        finally:
            body.close()
        if (
            len(payload) > _MAX_JSON_BYTES
            or hashlib.sha256(payload).hexdigest() != pointer["manifest_sha256"]
        ):
            raise GitHubReleaseError("매니페스트의 실제 해시가 포인터와 다릅니다.")
        manifest = parse_snapshot_manifest(payload)
        if (
            manifest["snapshot_id"] != pointer["snapshot_id"]
            or manifest["content_sha256"] != pointer["content_sha256"]
        ):
            raise GitHubReleaseError("매니페스트와 포인터의 버전 정보가 다릅니다.")
        for item in manifest["files"]:
            head = self.head_object(
                Bucket=self.config.bucket, Key=f"{keys['release_prefix']}/{item['name']}"
            )
            if (
                head["ContentLength"] != item["size_bytes"]
                or head["Metadata"]["sha256"] != item["sha256"]
            ):
                raise GitHubReleaseError("완성되지 않은 스냅샷은 최신 버전으로 지정할 수 없습니다.")

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **options: Any) -> None:
        self._check_bucket(Bucket)
        if not self._is_latest(Key):
            self._upload(Key, Body, len(Body), str(options["ContentType"]))
            return
        if not self.allow_create:
            raise GitHubReleaseError("읽기 전용 클라이언트로 채널을 갱신할 수 없습니다.")
        pointer = json.loads(Body.decode("utf-8"))
        validate_latest_pointer(pointer)
        self._verify_pointer_files(pointer)
        channel = self._channel(refresh=True)
        old_latest = channel.get("latest")
        if old_latest and old_latest["snapshot_id"] != pointer["snapshot_id"]:
            channel["previous"] = old_latest
        channel["latest"] = pointer
        release = self._load_release()
        updated = self._json(
            "PATCH",
            f"/repos/{self.config.repository}/releases/{release['id']}",
            payload={"body": json.dumps(channel, ensure_ascii=False, sort_keys=True)},
        )
        if not isinstance(updated, dict) or updated.get("id") != release["id"]:
            raise GitHubReleaseError("최신 채널 갱신 결과가 올바르지 않습니다.")
        self.release = updated

    def prune_old_snapshots(self) -> dict[str, Any]:
        """최신·이전 스냅샷을 보호하고 이 채널 소유의 나머지 첨부파일만 지운다."""
        if not self.allow_create:
            raise GitHubReleaseError("읽기 전용 클라이언트로 첨부파일을 정리할 수 없습니다.")
        channel = self._channel(refresh=True)
        if channel.get("latest") is None:
            raise GitHubReleaseError("완성된 최신 스냅샷이 없어 정리를 중단합니다.")
        self._verify_pointer_files(channel["latest"])
        retained = {
            item["snapshot_id"] for item in (channel.get("latest"), channel.get("previous")) if item
        }
        deleted = 0
        for asset in list(self._list_assets()):
            name = str(asset.get("name", ""))
            snapshot_id = None
            for file_name in _FILE_NAMES:
                suffix = f"--{file_name}"
                if name.startswith("snapshot--") and name.endswith(suffix):
                    candidate = name[len("snapshot--") : -len(suffix)]
                    if _ID_PATTERN.fullmatch(candidate):
                        snapshot_id = candidate
                    break
            if snapshot_id is None or snapshot_id in retained:
                continue
            if type(asset.get("id")) is not int or asset["id"] <= 0:
                raise GitHubReleaseError("삭제 대상 첨부파일의 식별 정보가 올바르지 않습니다.")
            self._json("DELETE", f"/repos/{self.config.repository}/releases/assets/{asset['id']}")
            self.assets.remove(asset)
            deleted += 1
        return {
            "status": "ready",
            "retained_snapshot_ids": sorted(retained),
            "deleted_asset_count": deleted,
        }
