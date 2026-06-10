"""CP252 — v2 전략 발굴 드라이버 (멀티프로세싱).

dev:    python run_v2_discovery.py --tickers dev --window val  --workers 6
heldout:python run_v2_discovery.py --tickers heldout --window test --workers 6 \
            --candidates backend/data/ablation/v2_candidates.json --with-ci

dev = 전 config(~1352) × dev200 val 탐색(CI 없이 빠르게) → agg+3트랙+inflation 저장.
heldout = 후보 config 만 × heldout271 test 1회(CI/Wilcoxon 포함) → OOS 통계.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _grouped_for(config_ids: set[str] | None):
    from app.services.strategy_ablation_v2 import generate_grid, group_by_base

    grid = generate_grid()
    grouped = group_by_base(grid)
    if config_ids is None:
        return grouped, len([c for c in grid if c.band or c.line])
    # 후보가 속한 base 만, variants 는 후보로 제한 (none 은 델타 기준으로 유지).
    out = {}
    for bk, g in grouped.items():
        cands = [c for c in g["variants"] if c.id in config_ids]
        if cands:
            out[bk] = {"none": g["none"], "variants": cands}
    return out, len([c for c in grid if c.band or c.line])


def _worker(payload):
    tickers, window, config_ids = payload
    from app.services.strategy_ablation_split import (
        iter_ticker_slices,
        load_ablation_frame_v2,
    )
    from app.services.strategy_ablation_v2 import v2_ticker_rows

    frame = load_ablation_frame_v2()
    grouped, _ = _grouped_for(set(config_ids) if config_ids else None)
    slices = iter_ticker_slices(frame, tickers, window)
    rows = []
    for tk, sl in slices:
        rows.extend(v2_ticker_rows(tk, sl, grouped))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--window", choices=["val", "test", "full"], default="val")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--candidates", default="")
    ap.add_argument("--with-ci", action="store_true")
    ap.add_argument("--out", default="backend/data/ablation/v2_dev_discovery")
    args = ap.parse_args()

    from app.services.strategy_ablation_split import load_split
    from app.services.strategy_ablation_v2_select import (
        aggregate_rows,
        select_tracks,
        selection_inflation,
    )

    split = load_split()
    tickers = split[args.tickers]
    config_ids = None
    if args.candidates:
        cj = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        config_ids = [c["config_id"] if isinstance(c, dict) else c for c in cj["candidates"]]

    _, n_tested = _grouped_for(set(config_ids) if config_ids else None)
    t0 = time.time()
    chunks = [list(c) for c in np.array_split(tickers, args.workers) if len(c)]
    payloads = [(ch, args.window, config_ids) for ch in chunks]
    cfg_desc = f"candidates:{len(config_ids)}" if config_ids else "ALL"
    print(
        f"[run] {args.tickers}({len(tickers)}) window={args.window} workers={args.workers} "
        f"configs={cfg_desc} n_tested={n_tested}"
    )

    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, payloads)
    rows = pd.DataFrame([r for chunk in results for r in chunk])
    elapsed = round(time.time() - t0, 1)
    print(f"[done] rows={len(rows)} in {elapsed}s")

    ci_metrics = ("d_maxDrawdownPct", "d_lla", "d_calmar", "d_sortino", "d_excessReturnPct")
    agg = aggregate_rows(rows, n_tested, with_ci=args.with_ci, ci_metrics=ci_metrics)
    tracks = select_tracks(agg, top_k=15)
    variants = agg[agg["toggle"] != "none"]
    infl = {
        "defensive_dMaxDD": selection_inflation(variants, "d_maxDrawdownPct_mean"),
        "offensive_excess": selection_inflation(agg, "excessReturnPct"),
        "balanced_dCalmar": selection_inflation(variants, "d_calmar_mean"),
    }

    out = ROOT.parent / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out.with_suffix(".rows.parquet"), index=False)
    agg.reset_index().to_parquet(out.with_suffix(".agg.parquet"), index=False)
    payload = {
        "meta": {
            "tickers_set": args.tickers,
            "window": args.window,
            "n_tickers": len(tickers),
            "n_configs": int(agg.shape[0]),
            "n_tested": n_tested,
            "with_ci": args.with_ci,
            "elapsed_sec": elapsed,
            "bonferroni_alpha_fullgrid": ALPHA / max(n_tested, 1),
        },
        "selection_inflation": infl,
        "tracks": {k: _track_json(v) for k, v in tracks.items()},
    }
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_safe), encoding="utf-8"
    )
    print(f"[out] {out.with_suffix('.json')}")
    for tk, df in tracks.items():
        print(f"\n=== {tk.upper()} top5 ===")
        print(_track_preview(tk, df))


ALPHA = 0.05


def _safe(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(str(type(o)))


def _track_json(df: pd.DataFrame) -> list[dict]:
    return df.reset_index().replace({np.nan: None}).to_dict("records")


def _track_preview(track: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "  (none)"
    cols = {
        "defensive": ["d_maxDrawdownPct_mean", "d_lla_mean", "d_totalReturnPct_mean", "n_tickers"],
        "offensive": ["excessReturnPct", "totalReturnPct", "maxDrawdownPct"],
        "balanced": ["d_calmar_mean", "d_sortino_mean", "d_maxDrawdownPct_mean", "n_tickers"],
    }[track]
    lines = []
    for cid, r in df.head(5).iterrows():
        parts = []
        for c in cols:
            label = c.split("_")[0] if "_" not in c else c[:10]
            parts.append(f"{label}={r[c]:+.2f}" if pd.notna(r[c]) else f"{label}=na")
        vals = " ".join(parts)
        lines.append(f"  {str(cid)[:46]:46s} {vals}")
    return "\n".join(lines)


if __name__ == "__main__":
    mp.freeze_support()
    main()
