"""원본 SISC 크롤러를 실행하고, 필요하면 parquet export까지 이어서 수행한다."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본 크롤러 실행 래퍼")
    parser.add_argument(
        "--crawler-script",
        default=os.environ.get("SOURCE_CRAWLER_PATH", ""),
        help="실제 원본 crawler_bot.py 경로",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="크롤러 실행에 사용할 Python 경로",
    )
    parser.add_argument("--db", default="db", help="원본 크롤러의 DB prefix 이름")
    parser.add_argument("--tickers", nargs="*", help="테스트용 ticker 목록")
    parser.add_argument("--repair", action="store_true", help="repair 모드 실행")
    parser.add_argument("--skip-price", action="store_true")
    parser.add_argument("--skip-info", action="store_true")
    parser.add_argument("--skip-fund", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-crypto", action="store_true")
    parser.add_argument("--skip-event", action="store_true")
    parser.add_argument("--skip-breadth", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--export-parquet", action="store_true", help="수집 후 DB 내용을 parquet로 저장")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "parquet"),
        help="parquet 저장 폴더",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("실행 명령:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True)


def build_crawler_command(args: argparse.Namespace) -> list[str]:
    if not args.crawler_script:
        raise SystemExit("[Error] --crawler-script 또는 SOURCE_CRAWLER_PATH가 필요합니다.")

    command = [args.python_bin, args.crawler_script, "--db", args.db]
    if args.tickers:
        command.extend(["--tickers", *args.tickers])
    if args.repair:
        command.append("--repair")

    for flag in [
        "skip_price",
        "skip_info",
        "skip_fund",
        "skip_macro",
        "skip_crypto",
        "skip_event",
        "skip_breadth",
        "skip_stats",
    ]:
        if getattr(args, flag):
            command.append(f"--{flag.replace('_', '-')}")

    return command


def build_export_command(args: argparse.Namespace) -> list[str]:
    export_script = Path(__file__).resolve().parent / "05_export_parquet.py"
    return [args.python_bin, str(export_script), "--output-dir", args.output_dir]


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" 원본 크롤러 실행")
    print("=" * 60)
    run_command(build_crawler_command(args))

    if args.export_parquet:
        print("\n" + "=" * 60)
        print(" DB -> parquet export")
        print("=" * 60)
        run_command(build_export_command(args))


if __name__ == "__main__":
    main()
