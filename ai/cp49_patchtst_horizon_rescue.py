from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    horizon: int
    patch_len: int
    patch_stride: int
    seq_len: int


def default_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    geometry = (
        ("baseline", 16, 8),
        ("longer_context", 32, 16),
        ("dense_overlap", 16, 4),
    )
    for horizon in (5, 10, 20):
        for geometry_name, patch_len, patch_stride in geometry:
            candidates.append(
                Candidate(
                    name=f"h{horizon}_{geometry_name}_seq252_p{patch_len}_s{patch_stride}",
                    horizon=horizon,
                    patch_len=patch_len,
                    patch_stride=patch_stride,
                    seq_len=252,
                )
            )
    candidates.append(
        Candidate(
            name="h20_baseline_seq504_p16_s8",
            horizon=20,
            patch_len=16,
            patch_stride=8,
            seq_len=504,
        )
    )
    return candidates


def parse_candidate(raw: str) -> Candidate:
    values: dict[str, str] = {}
    for part in raw.split(","):
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return Candidate(
        name=values["name"],
        horizon=int(values["horizon"]),
        patch_len=int(values.get("patch_len", 16)),
        patch_stride=int(values.get("patch_stride", 8)),
        seq_len=int(values.get("seq_len", 252)),
    )


def _read_stream(stream, log_file, chunks: list[str]) -> None:
    for line in stream:
        chunks.append(line)
        log_file.write(line)
        log_file.flush()
        print(line, end="")


def run_command(command: list[str], *, log_path: Path, env: dict[str, str]) -> tuple[int, str, float]:
    started = time.perf_counter()
    chunks: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# command: {' '.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        reader = threading.Thread(target=_read_stream, args=(process.stdout, log_file, chunks), daemon=True)
        reader.start()
        return_code = process.wait()
        reader.join()
        elapsed = time.perf_counter() - started
        log_file.write(f"\n# exit_code: {return_code}\n# elapsed_seconds: {elapsed:.4f}\n")
    return return_code, "".join(chunks), elapsed


def extract_train_json(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if '"step": "before_result_json"' in line:
            start_idx = idx + 1
            break
    if start_idx is None:
        raise ValueError("before_result_json marker를 찾지 못했습니다.")
    payload_lines: list[str] = []
    for line in lines[start_idx:]:
        if line.startswith("[EXIT-MARKER"):
            break
        payload_lines.append(line)
    return json.loads("\n".join(payload_lines))


def build_train_command(candidate: Candidate, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        "patchtst",
        "--timeframe",
        "1D",
        "--horizon",
        str(candidate.horizon),
        "--seq-len",
        str(candidate.seq_len),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--no-compile",
        "--ci-aggregate",
        "target",
        "--line-target-type",
        "raw_future_return",
        "--band-target-type",
        "raw_future_return",
        "--limit-tickers",
        str(args.limit_tickers),
        "--q-low",
        "0.25",
        "--q-high",
        "0.75",
        "--lambda-band",
        "2.0",
        "--checkpoint-selection",
        "line_gate",
        "--patch-len",
        str(candidate.patch_len),
        "--patch-stride",
        str(candidate.patch_stride),
        "--amp-dtype",
        args.amp_dtype,
        "--explicit-cuda-cleanup",
        "--wandb-project",
        args.wandb_project,
    ]
    command.append("--wandb" if args.wandb_train else "--no-wandb")
    return command


def selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "spearman_ic",
        "long_short_spread",
        "mae",
        "smape",
        "overprediction_rate",
        "mean_overprediction",
        "underprediction_rate",
        "mean_underprediction",
        "downside_capture_rate",
        "severe_downside_recall",
        "false_safe_rate",
        "conservative_bias",
        "upside_sacrifice",
        "all_horizon_spearman_ic",
        "all_horizon_long_short_spread",
        "all_horizon_mae",
        "all_horizon_smape",
        "all_horizon_false_safe_rate",
        "all_horizon_downside_capture_rate",
        "all_horizon_severe_downside_recall",
        "h1_h5_spearman_ic",
        "h1_h5_long_short_spread",
        "h1_h5_mae",
        "h1_h5_smape",
        "h1_h5_false_safe_rate",
        "h1_h5_downside_capture_rate",
        "h1_h5_severe_downside_recall",
        "h6_h10_spearman_ic",
        "h6_h10_long_short_spread",
        "h6_h10_mae",
        "h6_h10_smape",
        "h6_h10_false_safe_rate",
        "h6_h10_downside_capture_rate",
        "h6_h10_severe_downside_recall",
        "h11_h20_spearman_ic",
        "h11_h20_long_short_spread",
        "h11_h20_mae",
        "h11_h20_smape",
        "h11_h20_false_safe_rate",
        "h11_h20_downside_capture_rate",
        "h11_h20_severe_downside_recall",
        "coverage",
        "avg_band_width",
        "selected_reason",
        "line_gate_pass",
        "gate_failed",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def build_record(candidate: Candidate, train_result: dict[str, Any], *, exit_code: int, elapsed_seconds: float, log_path: Path) -> dict[str, Any]:
    best_metrics = train_result.get("best_metrics", {})
    test_metrics = train_result.get("test_metrics", {})
    return {
        "candidate": asdict(candidate),
        "exit_code": exit_code,
        "elapsed_seconds": elapsed_seconds,
        "log_path": str(log_path),
        "run_id": train_result.get("run_id"),
        "checkpoint_path": train_result.get("checkpoint_path"),
        "dataset_plan": train_result.get("dataset_plan"),
        "validation": selected_metrics(best_metrics),
        "test": selected_metrics(test_metrics),
        "epoch_seconds": best_metrics.get("epoch_seconds"),
        "vram_peak_allocated_mb": best_metrics.get("vram_peak_allocated_mb"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP49 PatchTST horizon rescue runner")
    parser.add_argument("--phase", choices=["plan", "smoke", "matrix"], default="plan")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--limit-tickers", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--wandb-train", action="store_true")
    parser.add_argument("--wandb-project", default="lens-cp49")
    parser.add_argument("--output-json", default="docs/cp49_patchtst_horizon_rescue_metrics.json")
    parser.add_argument("--log-dir", default="logs/cp49")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [parse_candidate(raw) for raw in args.candidate] if args.candidate else default_candidates()
    if args.phase == "smoke":
        candidates = candidates[:1]

    planned = [
        {
            "candidate": asdict(candidate),
            "command": build_train_command(candidate, args),
        }
        for candidate in candidates
    ]
    payload: dict[str, Any] = {
        "cp": "CP49-M",
        "phase": args.phase,
        "scope": {
            "model": "patchtst",
            "timeframe": "1D",
            "line_target_type": "raw_future_return",
            "band_target_type": "raw_future_return",
            "checkpoint_selection": "line_gate",
            "limit_tickers": args.limit_tickers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "save_run": False,
            "wandb_train": bool(args.wandb_train),
        },
        "planned": planned,
        "records": [],
    }

    if args.phase == "plan":
        write_json(Path(args.output_json), payload)
        print(json.dumps({"output_json": args.output_json, "candidate_count": len(candidates), "executed": False}, ensure_ascii=False))
        return

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if not args.wandb_train:
        env["WANDB_MODE"] = "disabled"

    for candidate in candidates:
        log_path = Path(args.log_dir) / f"{candidate.name}.log"
        command = build_train_command(candidate, args)
        exit_code, stdout, elapsed = run_command(command, log_path=log_path, env=env)
        if exit_code != 0:
            payload["records"].append(
                {
                    "candidate": asdict(candidate),
                    "exit_code": exit_code,
                    "elapsed_seconds": elapsed,
                    "log_path": str(log_path),
                    "error": "train command failed",
                }
            )
            write_json(Path(args.output_json), payload)
            raise SystemExit(exit_code)
        train_result = extract_train_json(stdout)
        payload["records"].append(build_record(candidate, train_result, exit_code=exit_code, elapsed_seconds=elapsed, log_path=log_path))
        write_json(Path(args.output_json), payload)

    print(json.dumps({"output_json": args.output_json, "candidate_count": len(candidates), "executed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
