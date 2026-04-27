from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import statistics
from pathlib import Path
import time

import torch

from ai.models.patchtst import PatchTST
from ai.train import should_use_cuda_optimizations

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "cp8_gpu_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST GPU 벤치마크")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--wandb", dest="use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="lens-ai")
    return parser.parse_args()


def _build_loader(samples: int, batch_size: int, seq_len: int, horizon: int):
    features = torch.randn(samples, seq_len, 36)
    line_targets = torch.randn(samples, horizon)
    band_targets = torch.randn(samples, horizon)
    ticker_ids = torch.randint(0, 473, (samples,), dtype=torch.long)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, line_targets, band_targets, ticker_ids),
        batch_size=batch_size,
        shuffle=False,
    )


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def benchmark_case(*, use_bf16: bool, use_compile: bool, repeats: int, samples: int, batch_size: int, seq_len: int, horizon: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"status": "skipped"}

    device = torch.device("cuda")
    durations: list[float] = []
    for _ in range(repeats):
        loader = _build_loader(samples, batch_size, seq_len, horizon)
        model = PatchTST(
            n_features=36,
            seq_len=seq_len,
            horizon=horizon,
            ticker_emb_dim=32,
            num_tickers=473,
            ci_aggregate="target",
            target_channel_idx=0,
        ).to(device)
        if use_compile and should_use_cuda_optimizations(device) and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        start = time.perf_counter()
        for features, line_targets, band_targets, ticker_ids in loader:
            features = features.to(device, non_blocking=True)
            line_targets = line_targets.to(device, non_blocking=True)
            band_targets = band_targets.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_bf16):
                output = model(features, ticker_id=ticker_ids)
                loss = (
                    (output.line - line_targets).pow(2).mean()
                    + (output.lower_band - band_targets).pow(2).mean()
                    + (output.upper_band - band_targets).pow(2).mean()
                )
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    return {
        "status": "ok",
        "durations": durations,
        "median_seconds": statistics.median(durations),
    }


def maybe_log_to_wandb(results: dict[str, object], args: argparse.Namespace) -> None:
    if not args.use_wandb or wandb is None:
        return
    run = wandb.init(
        project=args.wandb_project,
        group="gpu-benchmark",
        name="gpu_benchmark",
        config=results,
        reinit=True,
    )
    wandb.log(results)
    run.finish()


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    cases = {
        "fp32_no_compile": {"use_bf16": False, "use_compile": False},
        "bf16_no_compile": {"use_bf16": True, "use_compile": False},
        "fp32_compile": {"use_bf16": False, "use_compile": True},
        "bf16_compile": {"use_bf16": True, "use_compile": True},
    }
    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repeats": args.repeats,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "cases": {},
    }
    for name, options in cases.items():
        results["cases"][name] = benchmark_case(
            repeats=args.repeats,
            samples=args.samples,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            horizon=args.horizon,
            **options,
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_log_to_wandb(results, args)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    run_benchmark(parse_args())
