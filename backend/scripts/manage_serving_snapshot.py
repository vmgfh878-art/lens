"""서빙 스냅샷 매니페스트를 생성하거나 검증한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.services.serving_paths import get_serving_data_dir  # noqa: E402
from app.services.serving_snapshot import (  # noqa: E402
    SNAPSHOT_MANIFEST_FILENAME,
    SnapshotValidationError,
    load_snapshot_manifest,
    validate_snapshot_directory,
    write_snapshot_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="서빙 스냅샷 매니페스트 생성·검증")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="필수 파일의 크기와 해시로 매니페스트 생성")
    build.add_argument("--data-dir", default=str(get_serving_data_dir()))
    build.add_argument("--output")
    build.add_argument("--snapshot-id")

    validate = subparsers.add_parser("validate", help="매니페스트와 로컬 파일 무결성 검증")
    validate.add_argument("--data-dir", default=str(get_serving_data_dir()))
    validate.add_argument("--manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    try:
        if args.command == "build":
            output = Path(args.output).resolve() if args.output else None
            path, manifest = write_snapshot_manifest(
                data_dir,
                output_path=output,
                snapshot_id=args.snapshot_id,
            )
            summary = validate_snapshot_directory(manifest, data_dir)
            summary["manifest_path"] = str(path)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        manifest_path = (
            Path(args.manifest).resolve()
            if args.manifest
            else data_dir / SNAPSHOT_MANIFEST_FILENAME
        )
        manifest = load_snapshot_manifest(manifest_path)
        summary = validate_snapshot_directory(manifest, data_dir)
        summary["manifest_path"] = str(manifest_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except SnapshotValidationError as exc:
        print(f"스냅샷 검증 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
