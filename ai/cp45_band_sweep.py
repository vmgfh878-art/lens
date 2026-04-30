from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LINE_CHECKPOINT = Path("ai/artifacts/checkpoints/patchtst_1D_patchtst-1D-41d584bcb3cb.pt")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


@dataclass(frozen=True)
class Candidate:
    name: str
    seq_len: int
    q_low: float
    q_high: float
    lambda_band: float
    band_mode: str = "direct"


SMOKE_CANDIDATES = (
    Candidate("s60_q20_b2_direct", 60, 0.20, 0.80, 2.0, "direct"),
    Candidate("s45_q20_b2_direct", 45, 0.20, 0.80, 2.0, "direct"),
    Candidate("s90_q15_b2_direct", 90, 0.15, 0.85, 2.0, "direct"),
)


def _direct_sweep_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for seq_len in (45, 60, 90, 120):
        for q_low, q_high in ((0.10, 0.90), (0.15, 0.85), (0.20, 0.80)):
            for lambda_band in (1.0, 2.0, 3.0):
                name = f"s{seq_len}_q{int(q_low * 100):02d}{int(q_high * 100):02d}_b{lambda_band:g}_direct"
                candidates.append(Candidate(name, seq_len, q_low, q_high, lambda_band, "direct"))
    return candidates


def _parse_candidate(raw: str) -> Candidate:
    parts = raw.split(",")
    values: dict[str, str] = {}
    for part in parts:
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    seq_len = int(values["seq_len"])
    q_low = float(values["q_low"])
    q_high = float(values["q_high"])
    lambda_band = float(values["lambda_band"])
    band_mode = values.get("band_mode", "direct")
    name = values.get("name") or f"s{seq_len}_q{int(q_low * 100):02d}{int(q_high * 100):02d}_b{lambda_band:g}_{band_mode}"
    return Candidate(name, seq_len, q_low, q_high, lambda_band, band_mode)


def _extract_train_json(stdout: str) -> dict[str, Any]:
    marker = "[EXIT-MARKER {\"step\": \"before_result_json\""
    marker_index = stdout.rfind(marker)
    search_start = stdout.find("\n", marker_index) + 1 if marker_index >= 0 else 0
    decoder = json.JSONDecoder()
    index = search_start
    while index < len(stdout):
        if stdout[index] != "{":
            index += 1
            continue
        try:
            parsed, end_index = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(parsed, dict) and "checkpoint_path" in parsed and "run_id" in parsed:
            return parsed
        index += max(end_index, 1)
    raise ValueError("train stdout에서 결과 JSON을 찾지 못했습니다.")


def _epoch_summary_from_log(stdout: str) -> dict[str, Any]:
    epoch_seconds: list[float] = []
    vram_peak_mb: list[float] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch_seconds" in payload:
            epoch_seconds.append(float(payload.get("epoch_seconds", 0.0)))
            vram_peak_mb.append(float(payload.get("vram_peak_allocated_mb", 0.0)))
    return {
        "epoch_seconds": epoch_seconds,
        "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "vram_peak_mb": max(vram_peak_mb) if vram_peak_mb else None,
    }


def _read_stream(stream, sink, chunks: list[str]) -> None:
    for line in iter(stream.readline, ""):
        chunks.append(line)
        sink.write(line)
        sink.flush()
    stream.close()


def _run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# command: {' '.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(target=_read_stream, args=(process.stdout, log_file, stdout_chunks), daemon=True)
        stderr_thread = threading.Thread(target=_read_stream, args=(process.stderr, log_file, stderr_chunks), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        elapsed = time.perf_counter() - started
        log_file.write(f"\n# exit_code: {return_code}\n")
        log_file.write(f"# elapsed_seconds: {elapsed:.4f}\n")
        log_file.flush()
    return subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def _train_candidate(
    candidate: Candidate,
    *,
    args: argparse.Namespace,
    log_dir: Path,
    batch_size: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "ai.train",
        "--model",
        "cnn_lstm",
        "--timeframe",
        "1D",
        "--horizon",
        "5",
        "--seq-len",
        str(candidate.seq_len),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        args.device,
        "--no-compile",
        "--line-target-type",
        "raw_future_return",
        "--band-target-type",
        "raw_future_return",
        "--limit-tickers",
        str(args.limit_tickers),
        "--q-low",
        str(candidate.q_low),
        "--q-high",
        str(candidate.q_high),
        "--lambda-band",
        str(candidate.lambda_band),
        "--band-mode",
        candidate.band_mode,
        "--checkpoint-selection",
        "band_gate",
        "--fp32-modules",
        "lstm,heads",
        "--amp-dtype",
        args.amp_dtype,
        "--explicit-cuda-cleanup",
        "--wandb-project",
        args.wandb_project,
    ]
    if args.wandb_train:
        command.append("--wandb")
    else:
        command.append("--no-wandb")
    if args.hard_exit_after_result:
        command.append("--hard-exit-after-result")

    log_path = log_dir / f"{candidate.name}_train_bs{batch_size}.log"
    result = _run_command(command, cwd=PROJECT_ROOT, log_path=log_path)
    if result.returncode != 0:
        return None, {
            "exit_code": result.returncode,
            "log_path": str(log_path),
            "oom": "out of memory" in (result.stdout + result.stderr).lower(),
        }
    train_result = _extract_train_json(result.stdout)
    train_meta = _epoch_summary_from_log(result.stdout)
    train_meta.update({"exit_code": result.returncode, "log_path": str(log_path)})
    return train_result, train_meta


def _survives_calibrated(metrics: dict[str, Any]) -> bool:
    return (
        0.75 <= float(metrics.get("coverage", 0.0)) <= 0.90
        and float(metrics.get("lower_breach_rate", 1.0)) <= 0.12
        and float(metrics.get("upper_breach_rate", 1.0)) <= 0.15
    )


def _flat_for_wandb(candidate: Candidate, record: dict[str, Any]) -> dict[str, Any]:
    calibration = record.get("calibration", {})
    composite = record.get("composite_probe", {}).get("risk_first_lower_preserve", {})
    raw = record.get("band_calibration", {}).get("original", {}).get("test", {})
    scalar = record.get("band_calibration", {}).get("scalar_width", {}).get("test", {})
    train = record.get("train", {})
    return {
        "seq_len": candidate.seq_len,
        "q_low": candidate.q_low,
        "q_high": candidate.q_high,
        "lambda_band": candidate.lambda_band,
        "band_mode": candidate.band_mode,
        "raw_coverage": raw.get("coverage"),
        "raw_lower_breach_rate": raw.get("lower_breach_rate"),
        "raw_upper_breach_rate": raw.get("upper_breach_rate"),
        "raw_avg_band_width": raw.get("avg_band_width"),
        "calibrated_coverage": scalar.get("coverage"),
        "calibrated_lower_breach_rate": scalar.get("lower_breach_rate"),
        "calibrated_upper_breach_rate": scalar.get("upper_breach_rate"),
        "calibrated_avg_band_width": scalar.get("avg_band_width"),
        "lower_scale": calibration.get("lower_scale"),
        "upper_scale": calibration.get("upper_scale"),
        "composite_coverage": composite.get("coverage"),
        "composite_lower_breach_rate": composite.get("lower_breach_rate"),
        "composite_upper_breach_rate": composite.get("upper_breach_rate"),
        "composite_avg_band_width": composite.get("avg_band_width"),
        "line_inside_band_ratio": composite.get("line_inside_band_ratio"),
        "band_width_increase_ratio": composite.get("width_ratio_vs_raw"),
        "epoch_seconds": train.get("epoch_seconds_mean"),
        "vram_peak_mb": train.get("vram_peak_mb"),
    }


def _log_wandb_summary(candidate: Candidate, record: dict[str, Any], args: argparse.Namespace) -> str | None:
    if not args.wandb_summary:
        return None
    try:
        import wandb
    except ImportError:
        return "wandb 패키지가 없어 summary logging을 건너뜀"
    name = (
        f"model=cnn_lstm_seq={candidate.seq_len}_q={candidate.q_low:.2f}-{candidate.q_high:.2f}_"
        f"lb={candidate.lambda_band:g}_mode={candidate.band_mode}_cal=scalar_width"
    )
    try:
        run = wandb.init(
            project=args.wandb_project,
            group=f"cp45_{args.phase}",
            name=name,
            config=asdict(candidate) | {"limit_tickers": args.limit_tickers, "epochs": args.epochs},
        )
        wandb.log(_flat_for_wandb(candidate, record))
        run.finish()
        return name
    except Exception as exc:  # pragma: no cover - 외부 W&B 상태 의존
        return f"W&B logging 실패: {exc}"


def _evaluate_candidate(candidate: Candidate, train_result: dict[str, Any], train_meta: dict[str, Any], args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    from ai.band_calibration import evaluate_candidate
    from ai.composite_policy_eval import evaluate_composite_policies

    checkpoint_path = Path(train_result["checkpoint_path"])
    band_result = evaluate_candidate(
        name=candidate.name,
        checkpoint_path=checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers="auto",
        amp_dtype=args.amp_dtype,
        target_coverage=args.target_coverage,
    )
    calibration = band_result["scalar_width"]["calibration"]
    composite = None
    if args.composite_mode == "all" or (
        args.composite_mode == "survivors" and _survives_calibrated(band_result["scalar_width"]["test"])
    ):
        composite_path = log_dir / f"{candidate.name}_composite_policy.json"
        composite = evaluate_composite_policies(
            line_checkpoint=Path(args.line_checkpoint),
            band_checkpoint=checkpoint_path,
            split="test",
            tickers=None,
            limit_tickers=args.limit_tickers,
            max_rows=args.limit_tickers,
            device_name=args.device,
            batch_size=args.batch_size,
            amp_dtype=args.amp_dtype,
            lower_scale=float(calibration["lower_scale"]),
            upper_scale=float(calibration["upper_scale"]),
            output_json=composite_path,
        )
    record = {
        "candidate": asdict(candidate),
        "train": {
            "run_id": train_result.get("run_id"),
            "checkpoint_path": train_result.get("checkpoint_path"),
            "best_metrics": train_result.get("best_metrics"),
            "test_metrics": train_result.get("test_metrics"),
            **train_meta,
        },
        "calibration": calibration,
        "band_calibration": {
            "original": band_result["original"],
            "scalar_width": band_result["scalar_width"],
        },
        "composite_probe": composite["policy_results"] if composite else None,
        "calibrated_survives": _survives_calibrated(band_result["scalar_width"]["test"]),
    }
    record["wandb_run_name"] = _log_wandb_summary(candidate, record, args)
    return record


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.candidate:
        return [_parse_candidate(raw) for raw in args.candidate]
    if args.phase == "smoke":
        return list(SMOKE_CANDIDATES)
    candidates = _direct_sweep_candidates()
    if args.include_param:
        candidates.extend(
            [
                Candidate("s60_q20_b2_param", 60, 0.20, 0.80, 2.0, "param"),
                Candidate("s90_q15_b2_param", 90, 0.15, 0.85, 2.0, "param"),
            ]
        )
    return candidates[: args.max_candidates] if args.max_candidates else candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP45 CNN-LSTM band sweep 오케스트레이터")
    parser.add_argument("--phase", choices=["smoke", "sweep"], default="smoke")
    parser.add_argument("--candidate", action="append", default=None, help="name=...,seq_len=60,q_low=0.2,q_high=0.8,lambda_band=2,band_mode=direct")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--include-param", action="store_true")
    parser.add_argument("--line-checkpoint", default=str(DEFAULT_LINE_CHECKPOINT))
    parser.add_argument("--limit-tickers", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retry-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--target-coverage", type=float, default=0.85)
    parser.add_argument("--composite-mode", choices=["none", "survivors", "all"], default="survivors")
    parser.add_argument("--wandb-train", action="store_true")
    parser.add_argument("--wandb-summary", action="store_true")
    parser.add_argument("--wandb-project", default="lens-cp45")
    parser.add_argument("--hard-exit-after-result", action="store_true")
    parser.add_argument("--output-json", default="docs/cp45_cnn_lstm_band_sweep_metrics.json")
    parser.add_argument("--log-dir", default="logs/cp45")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_path = Path(args.output_json)
    candidates = build_candidates(args)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        train_result, train_meta = _train_candidate(candidate, args=args, log_dir=log_dir, batch_size=args.batch_size)
        if train_result is None and train_meta.get("oom") and args.retry_batch_size > 0:
            train_result, train_meta = _train_candidate(candidate, args=args, log_dir=log_dir, batch_size=args.retry_batch_size)
            train_meta["retried_after_oom"] = True
        if train_result is None:
            records.append({"candidate": asdict(candidate), "failed": True, "train": train_meta})
            continue
        records.append(_evaluate_candidate(candidate, train_result, train_meta, args, log_dir))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_json_safe({"phase": args.phase, "records": records}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    result = {
        "phase": args.phase,
        "limit_tickers": args.limit_tickers,
        "epochs": args.epochs,
        "candidate_count": len(candidates),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe({"output_json": str(output_path), "candidate_count": len(candidates)}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
