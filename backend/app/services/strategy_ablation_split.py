"""CP248 — ablation OOS 분할 인프라 (과적합 구조 차단, 두 축).

티커축 (CP248 §P4): dev 200 (개발·선정) / held-out 나머지 (확정 검증, 겹침 0).
  - 섹터 × 시총 tercile stratified (편향 최소), seed 고정, json 으로 박제.
  - **데이터 현실**: market_stock_info 는 100/471 티커만 sector·market_cap 보유.
    → 알려진 섹터는 stratify, 미상은 'Unknown' 단일 버킷. 시총 tercile 도 known 만.
    한계를 정직히 기록 (REFORMS). 분할은 결정적·재현 가능.
시간축: 각 티커 AI창(~251일) 앞 70% val(임계 탐색) / 뒤 30% test(보고 1회, peeking 금지).
per-ticker long/cash 단일 종목 계약.

분할 파일은 CP249/250 이 **그대로** 읽어 누수 방지 (한 번 고정 후 불변).
운영 parquet read-only — 분할 산출물은 data/v1 밖(data/ablation/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.services.strategy_scan import _data_dir, _load_frame, _sector_map

ELIGIBLE_MIN_DAYS = 150  # AI창 내 최소 거래일 (val/test 둘 다 의미있게)
DEV_TARGET = 200
SPLIT_SEED = 248
VAL_FRAC = 0.70
# relative band 토글 컷 분위 (CP248 사용자 확정: decile, dev/val 에서만 산출 → 누수 0).
BAND_LOWER_Q = 0.10  # band_lower_return 하위 10분위 = 깊은-하단(꼬리위험)
BAND_WIDTH_Q = 0.90  # band_width_expansion 상위 10분위 = 넓은-폭(고변동 국면)


def _artifact_dir() -> Path:
    """ablation 산출물 디렉토리 (data/v1 가드 밖)."""
    path = Path(__file__).resolve().parents[2] / "data" / "ablation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_path() -> Path:
    return _artifact_dir() / "ticker_split.json"


def load_ablation_frame() -> pd.DataFrame:
    """AI 가용 구간(line & band 둘 다 존재)만 남긴 전 유니버스 프레임.

    4변형을 **동일 ticker·동일 날짜**에서 돌리기 위해 AI창으로 고정 (paired delta
    청결성). none 팔은 AI 컬럼을 무시하므로 동일창에서 공정 비교.
    """
    frame = _load_frame(True, True)
    mask = (
        frame["line_score"].notna()
        & frame["band_lower_return"].notna()
        & frame["band_width_return"].notna()
    )
    frame = frame[mask].copy()
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def eligible_tickers(frame: pd.DataFrame) -> list[str]:
    counts = frame.groupby("ticker")["date"].count()
    return sorted(counts[counts >= ELIGIBLE_MIN_DAYS].index.astype(str).tolist())


def load_ablation_frame_v2() -> pd.DataFrame:
    """CP252 — load_ablation_frame + 신규 archetype 용 원천/파생 피처.

    기존 컬럼은 불변(ai_band_defense_v2·CP248 경로 영향 0). 추가:
      원천: volume·high(market_prices) · ma_5_ratio·vol_change(market_indicators)
      파생(티커별, 미래 누수 없게 당일까지만): roc_20 · new_high_20 · close_z_20 ·
            vol_surge · macd_accel · line_mom
    """
    frame = load_ablation_frame()
    base_dir = _data_dir()
    px = pd.read_parquet(
        base_dir / "market_prices_1d.parquet", columns=["ticker", "date", "high", "volume"]
    )
    px["ticker"] = px["ticker"].astype(str).str.upper()
    px["date"] = pd.to_datetime(px["date"])
    px["high"] = pd.to_numeric(px["high"], errors="coerce")
    px["volume"] = pd.to_numeric(px["volume"], errors="coerce")
    px = px.drop_duplicates(["ticker", "date"], keep="last")
    ind = pd.read_parquet(
        base_dir / "market_indicators_1d.parquet",
        columns=["ticker", "date", "ma_5_ratio", "vol_change"],
    )
    ind["ticker"] = ind["ticker"].astype(str).str.upper()
    ind["date"] = pd.to_datetime(ind["date"])
    for c in ["ma_5_ratio", "vol_change"]:
        ind[c] = pd.to_numeric(ind[c], errors="coerce")
    ind = ind.drop_duplicates(["ticker", "date"], keep="last")

    frame = frame.merge(px, on=["ticker", "date"], how="left").merge(
        ind, on=["ticker", "date"], how="left"
    )
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = frame.groupby("ticker", sort=False)
    close = frame["close"]
    frame["roc_20"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    # 신고가: 당일 종가가 직전 20일 고가 최대를 돌파 (shift(1) 로 당일 제외 → 누수 방지)
    frame["high_20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max().shift(1))
    frame["new_high_20"] = (close >= frame["high_20"]).astype(float)
    roll_mean = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    roll_std = g["close"].transform(lambda s: s.rolling(20, min_periods=10).std())
    frame["close_z_20"] = ((close - roll_mean) / roll_std.replace(0.0, np.nan)).fillna(0.0)
    vol_ref = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    frame["vol_surge"] = (frame["volume"] / vol_ref).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    frame["macd_accel"] = g["macd_ratio"].transform(lambda s: s - s.shift(3))
    frame["line_mom"] = g["line_score"].transform(lambda s: s - s.shift(5))
    return frame


def _market_cap_map() -> dict[str, float]:
    try:
        info = pd.read_parquet(_data_dir() / "market_stock_info.parquet")
    except Exception:
        return {}
    if "market_cap" not in info.columns:
        return {}
    info["ticker"] = info["ticker"].astype(str).str.upper()
    out: dict[str, float] = {}
    for _, row in info.iterrows():
        cap = row.get("market_cap")
        if pd.notna(cap):
            out[str(row["ticker"]).upper()] = float(cap)
    return out


def _size_label_map(tickers: list[str]) -> dict[str, str]:
    """known market_cap 으로 tercile(S/M/L), 미상은 'NA'."""
    caps = _market_cap_map()
    known = {t: caps[t] for t in tickers if t in caps}
    labels: dict[str, str] = {t: "NA" for t in tickers}
    if len(known) >= 3:
        values = pd.Series(known)
        try:
            terciles = pd.qcut(values, 3, labels=["S", "M", "L"])
            for t, lab in terciles.items():
                labels[t] = str(lab)
        except ValueError:  # 동률로 qcut 실패 시 rank 기반 fallback
            ranks = values.rank(method="first")
            cut = pd.cut(ranks, 3, labels=["S", "M", "L"])
            for t, lab in cut.items():
                labels[t] = str(lab)
    return labels


def _stratum_keys(tickers: list[str]) -> dict[str, str]:
    sector_map = _sector_map()
    size_map = _size_label_map(tickers)
    return {t: f"{sector_map.get(t, 'Unknown')}|{size_map.get(t, 'NA')}" for t in tickers}


def build_ticker_split(
    frame: pd.DataFrame | None = None, dev_target: int = DEV_TARGET, seed: int = SPLIT_SEED
) -> dict[str, Any]:
    """섹터×시총 stratified, seed 고정, 결정적 dev/held-out 분할.

    largest-remainder 로 dev_target 정확히 맞춤. 각 stratum 내 seeded shuffle 후 dev 할당.
    """
    if frame is None:
        frame = load_ablation_frame()
    universe = eligible_tickers(frame)
    n_total = len(universe)
    dev_target = min(dev_target, n_total)
    strata_keys = _stratum_keys(universe)

    # stratum -> tickers (정렬 후 seeded shuffle 으로 결정적 순서)
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[str]] = {}
    for t in sorted(universe):
        by_stratum.setdefault(strata_keys[t], []).append(t)
    for key in sorted(by_stratum):
        arr = by_stratum[key]
        order = rng.permutation(len(arr))
        by_stratum[key] = [arr[i] for i in order]

    # largest-remainder: stratum 별 dev quota
    raw = {k: dev_target * len(v) / n_total for k, v in by_stratum.items()}
    floor_q = {k: int(np.floor(x)) for k, x in raw.items()}
    remainder = dev_target - sum(floor_q.values())
    frac_order = sorted(by_stratum, key=lambda k: (raw[k] - floor_q[k], k), reverse=True)
    quota = dict(floor_q)
    for k in frac_order[:remainder]:
        quota[k] += 1

    dev: list[str] = []
    heldout: list[str] = []
    strata_report: dict[str, dict[str, int]] = {}
    for key in sorted(by_stratum):
        members = by_stratum[key]
        q = min(quota.get(key, 0), len(members))
        dev.extend(members[:q])
        heldout.extend(members[q:])
        strata_report[key] = {"dev": q, "heldout": len(members) - q, "total": len(members)}

    dev = sorted(dev)
    heldout = sorted(heldout)
    band_cuts = compute_band_cuts(frame, dev)
    return {
        "seed": seed,
        "eligible_min_days": ELIGIBLE_MIN_DAYS,
        "n_universe": n_total,
        "n_dev": len(dev),
        "n_heldout": len(heldout),
        "window_start": str(frame["date"].min().date()),
        "window_end": str(frame["date"].max().date()),
        "n_strata": len(by_stratum),
        "n_with_sector": sum(1 for t in universe if t in _sector_map()),
        "n_with_market_cap": sum(1 for t in universe if t in _market_cap_map()),
        "band_cuts": band_cuts,
        "dev": dev,
        "heldout": heldout,
        "strata": strata_report,
    }


def compute_band_cuts(
    frame: pd.DataFrame, dev_tickers: list[str], val_frac: float = VAL_FRAC
) -> dict[str, Any]:
    """relative band 토글 컷을 **dev 티커의 val 구간에서만** 산출 (test/held-out 누수 0).

    lower_cut = band_lower_return 의 dev/val pooled p10 (깊은-하단).
    width_cut = band_width_expansion 의 dev/val pooled p90 (넓은-폭).
    결과 보기 전 사전등록된 decile. 한 번 박제 후 CP249/250 이 그대로 사용.
    """
    slices = iter_ticker_slices(frame, dev_tickers, "val", val_frac)
    blr = pd.concat(
        [pd.to_numeric(sl["band_lower_return"], errors="coerce") for _t, sl in slices]
    ).dropna()
    bwe = pd.concat(
        [pd.to_numeric(sl["band_width_expansion"], errors="coerce") for _t, sl in slices]
    ).dropna()
    return {
        "lower_q": BAND_LOWER_Q,
        "width_q": BAND_WIDTH_Q,
        "lower_cut": float(blr.quantile(BAND_LOWER_Q)),
        "width_cut": float(bwe.quantile(BAND_WIDTH_Q)),
        "n_rows": int(len(blr)),
    }


def save_split(split: dict[str, Any]) -> Path:
    path = split_path()
    path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_split() -> dict[str, Any]:
    path = split_path()
    if not path.exists():
        raise FileNotFoundError(
            f"ticker_split.json 없음 — build_ticker_split + save_split 먼저: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def time_split(
    ticker_slice: pd.DataFrame, val_frac: float = VAL_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 티커 AI창을 시간순 앞 val_frac / 뒤 (1-val_frac) 로 분할 (미래 누수 방지)."""
    ordered = ticker_slice.sort_values("date").reset_index(drop=True)
    n = len(ordered)
    cut = int(np.floor(n * val_frac))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def iter_ticker_slices(
    frame: pd.DataFrame, tickers: list[str], window: str, val_frac: float = VAL_FRAC
) -> list[tuple[str, pd.DataFrame]]:
    """(ticker, slice) 리스트. window ∈ {'val','test','full'}.

    val/test 는 각 티커 내 시간 분할. full 은 전 AI창. MIN_EVAL 미만 슬라이스는 스킵.
    """
    if window not in {"val", "test", "full"}:
        raise ValueError(f"window must be val/test/full, got {window}")
    want = set(t.upper() for t in tickers)
    out: list[tuple[str, pd.DataFrame]] = []
    for ticker, grp in frame[frame["ticker"].isin(want)].groupby("ticker", sort=True):
        ordered = grp.sort_values("date").reset_index(drop=True)
        if window == "full":
            sl = ordered
        else:
            val_df, test_df = time_split(ordered, val_frac)
            sl = val_df if window == "val" else test_df
        if len(sl) < 30:  # 너무 짧은 슬라이스는 메트릭 무의미
            continue
        out.append((str(ticker), sl))
    return out


def load_regime_map() -> pd.DataFrame:
    """regime_label(calm/neutral/stress) per (ticker, date).

    _load_base_frame 은 regime_label 을 drop 하므로(메모리) 별도 로드. regime 분해 전용.
    proxy 주의: regime_label 은 평가의 VIX/DD 와 다른 산출(서로 1:1 아님).
    """
    ind = pd.read_parquet(
        _data_dir() / "market_indicators_1d.parquet", columns=["ticker", "date", "regime_label"]
    )
    ind["ticker"] = ind["ticker"].astype(str).str.upper()
    ind["date"] = pd.to_datetime(ind["date"])
    ind = ind.dropna(subset=["regime_label"]).drop_duplicates(["ticker", "date"], keep="last")
    return ind


def attach_regime(slice_df: pd.DataFrame, regime_map: pd.DataFrame) -> pd.DataFrame:
    out = slice_df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"])
    return out.merge(regime_map, on=["ticker", "date"], how="left")


__all__ = [
    "ELIGIBLE_MIN_DAYS",
    "DEV_TARGET",
    "SPLIT_SEED",
    "VAL_FRAC",
    "split_path",
    "load_ablation_frame",
    "load_ablation_frame_v2",
    "eligible_tickers",
    "compute_band_cuts",
    "build_ticker_split",
    "save_split",
    "load_split",
    "time_split",
    "iter_ticker_slices",
    "load_regime_map",
    "attach_regime",
]
