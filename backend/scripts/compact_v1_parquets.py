"""CP246 — v1 serving parquet 재인코딩(compaction). 메모리 OOM 근본 대응.

문제: serving parquet 들이 문자열/날짜를 plain object string 으로 저장한다.
대표적으로 model_id/source_cp/timeframe/source/created_at 은 **고유값 1~2개**인데
수십만~78만 행에 통째 복제돼, read 시 디스크 15MB 파일이 200~390MB 로 펼쳐진다.
그 transient 가 allocator 에 박혀 RSS 가 안 빠지고, cold load peak 이 누적돼
512MB Render 무료 인스턴스에서 간헐 OOM(주식보기 503 / AI 백테스트 502).

해결: 문자열을 category(parquet dictionary), 날짜를 datetime64 로 재저장한다.
압축해제 크기가 급감해 read 스파이크가 사라진다.
  · 예측 3종(line/band1d/band1w): float 은 float64 그대로 → **값 1바이트 불변**.
  · product_history: 서빙이 이미 float32 로 다이어트(_compress_history_dtypes)
    하므로 파일도 float32 로 둬도 **서빙 출력 불변**(같은 float32 값).

안전장치(필수):
  · 재인코딩 프레임을 **실제 서빙 로드 경로**로 다시 읽어 원본의 같은 경로
    결과와 값 동일성(canonical 비교)을 확인한 뒤에만 atomic replace. 불일치면
    그 파일은 건너뛴다.
  · 멱등 — 이미 compact 면 재실행해도 같은 결과.
  · 운영 parquet 은 git tracked. 문제 시 `git checkout -- <파일>` 로 복원.

용도: 현재 운영 파일 1회 재인코딩 + refresh 파이프라인 마지막 단계.
실행: python backend/scripts/compact_v1_parquets.py [--apply]  (없으면 dry-run)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services import local_market_svc  # noqa: E402
from app.services.parquet_store import _compress_strings  # noqa: E402
from app.services.product_prediction_history_svc import (  # noqa: E402
    _load_history_frame_cached,
)

V1_DIR = ROOT / "backend" / "data" / "v1"


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    """값 비교용 정규화 — category→str, datetime→'YYYY-MM-DD', 그 외 그대로.

    dtype / category-순서 차이를 흡수하고 **값만** 비교한다.
    """
    out = pd.DataFrame(index=range(len(df)))
    for col in sorted(df.columns):
        s = df[col].reset_index(drop=True)
        dt = s.dtype
        if isinstance(dt, pd.CategoricalDtype):
            out[col] = s.astype(str)
        elif pdt.is_datetime64_any_dtype(dt):
            out[col] = s.dt.strftime("%Y-%m-%d")
        else:
            out[col] = s
    return out


# ---- 예측 parquet (line/band1d/band1w) ------------------------------------
_PRED_CAT = ["ticker", "model_id", "source_cp"]
_PRED_DATE = ["asof_date", "forecast_date"]


def _reencode_pred(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _PRED_CAT:
        if col in out.columns:
            out[col] = out[col].astype("category")
    for col in _PRED_DATE:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _load_pred(path: Path) -> pd.DataFrame:
    # parquet_store._load 의 정규화 단계와 동일.
    return _compress_strings(pd.read_parquet(path))


# ---- product_prediction_history_1D ----------------------------------------
_PH_FLOAT = ["line_value", "lower_value", "upper_value"]


def _reencode_product(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in ("asof_date",):
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif col in _PH_FLOAT:
            out[col] = pd.to_numeric(out[col], downcast="float")  # 서빙과 동일 float32
        elif col == "display_horizon":
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pdt.is_object_dtype(out[col].dtype):
            out[col] = out[col].astype("category")
    return out


def _load_product(path: Path) -> pd.DataFrame:
    # 실제 서빙 로드 경로(read columns + 정규화 + dtype 다이어트).
    return _load_history_frame_cached(str(path), os.path.getmtime(path))


# ---- market parquet (prices/indicators) -----------------------------------
# 문자열만 category(=parquet dictionary), 날짜는 datetime. float 은 float64 그대로
# 둬 지표 값 정밀도(서빙 출력)를 1바이트도 안 바꾼다. svc 가 date 를 다시
# 'YYYY-MM-DD' 문자열로 strftime 하므로 출력은 기존과 동일.
def _reencode_market(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in out.columns:
        if col != "date" and pdt.is_object_dtype(out[col].dtype):
            out[col] = out[col].astype("category")
    return out


def _load_market(path: Path) -> pd.DataFrame:
    return local_market_svc._load(path)


# ---- 파일 핸들러 레지스트리 -------------------------------------------------
HANDLERS: dict[str, tuple[Callable, Callable]] = {
    "predictions_line_1d.parquet": (_reencode_pred, _load_pred),
    "predictions_band_1d.parquet": (_reencode_pred, _load_pred),
    "predictions_band_1w.parquet": (_reencode_pred, _load_pred),
    "product_prediction_history_1D.parquet": (_reencode_product, _load_product),
    "market_prices_1d.parquet": (_reencode_market, _load_market),
    "market_indicators_1d.parquet": (_reencode_market, _load_market),
}


def compact_file(path: Path, reencode: Callable, load: Callable, *, apply: bool) -> dict:
    orig = pd.read_parquet(path)
    disk_before = path.stat().st_size / 1024 / 1024

    reenc = reencode(orig)
    tmp = path.with_suffix(".compact.tmp.parquet")
    reenc.to_parquet(tmp, index=False, compression="zstd")
    disk_after = tmp.stat().st_size / 1024 / 1024

    # 안전 게이트: 실제 서빙 로드 경로 결과 값 동일성.
    try:
        old_loaded = load(path)
        new_loaded = load(tmp)
        pd.testing.assert_frame_equal(_canon(old_loaded), _canon(new_loaded), check_like=False)
        value_equal, err = True, ""
    except AssertionError as exc:  # noqa: BLE001
        value_equal = False
        err = (str(exc).splitlines() or ["mismatch"])[0]

    status = "DRY-RUN"
    if value_equal and apply:
        os.replace(tmp, path)
        status = "REPLACED"
    else:
        tmp.unlink(missing_ok=True)
        if not value_equal:
            status = "SKIPPED(value-mismatch)"

    return {
        "file": path.name,
        "rows": len(orig),
        "disk_before_mb": round(disk_before, 1),
        "disk_after_mb": round(disk_after, 1),
        "value_equal": value_equal,
        "status": status,
        "error": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="원본 교체 실행(없으면 dry-run)")
    args = ap.parse_args()

    print(f"v1 dir: {V1_DIR}")
    print(f"mode  : {'APPLY' if args.apply else 'DRY-RUN'}\n")
    any_fail = False
    for name, (reencode, load) in HANDLERS.items():
        path = V1_DIR / name
        if not path.exists():
            print(f"  - {name}: MISSING, skip")
            continue
        r = compact_file(path, reencode, load, apply=args.apply)
        flag = "OK " if r["value_equal"] else "FAIL"
        print(
            f"  [{flag}] {r['file']:38s} rows={r['rows']:>7} "
            f"{r['disk_before_mb']:>5.1f}MB -> {r['disk_after_mb']:>5.1f}MB  {r['status']}"
        )
        if r["error"]:
            print(f"         err: {r['error']}")
        any_fail = any_fail or not r["value_equal"]

    if any_fail:
        print("\n값 불일치 파일 존재 — 해당 파일은 교체하지 않음. 조사 필요.")
        return 1
    print("\n전부 값 동일성 통과." + ("" if args.apply else " (--apply 로 실제 교체)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
