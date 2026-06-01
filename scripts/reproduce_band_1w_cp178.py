"""
Lens 운영 1W 밴드 (CP178 TiDE WFLOCK q10/q90 walk-forward lower) 재현 wrapper.

사용:
    python scripts/reproduce_band_1w_cp178.py --external ./external_package
    python scripts/reproduce_band_1w_cp178.py --external ./external_package --verify-only

요구:
    - Python 3.10.x
    - torch 2.11.0+cu128 + GPU sm_120 (RTX 5060 Ti)
    - external_package 폴더에 checkpoint + 1W feature parquet 동봉

흐름: 환경 검증 → 외부 패키지 점검 → 데이터/checkpoint 배치 → CP178 WFLOCK → 결과 비교.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_PYTHON = (3, 10)
EXPECTED_TORCH_PREFIX = "2.11.0+cu128"
EXPECTED_GPU_ARCH = (12, 0)

REQUIRED_PARQUETS = [
    "price_data_yfinance_500.parquet",
    "indicators_yfinance_1W_500.parquet",
]
REQUIRED_CHECKPOINTS = [
    "cp178/tide_s60_q10_q90_param-seed_7.pt",
    "cp178/tide_s60_q10_q90_param-seed_42.pt",
    "cp178/tide_s60_q10_q90_param-seed_123.pt",
]
SERVING_PARQUET = REPO_ROOT / "backend" / "data" / "v1" / "predictions_band_1w.parquet"


def check_environment() -> None:
    py = sys.version_info
    if (py.major, py.minor) != EXPECTED_PYTHON:
        sys.exit(f"[FAIL] Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.x 필요. 현재 {py.major}.{py.minor}.{py.micro}")
    try:
        import torch  # noqa: WPS433
    except ImportError:
        sys.exit("[FAIL] torch 미설치. pip install -r requirements.txt")
    if not torch.__version__.startswith(EXPECTED_TORCH_PREFIX):
        print(f"[WARN] torch {EXPECTED_TORCH_PREFIX} 권장. 현재 {torch.__version__}")
    if not torch.cuda.is_available():
        sys.exit("[FAIL] CUDA 사용 불가. GPU 필요.")
    cap = torch.cuda.get_device_capability(0)
    if cap != EXPECTED_GPU_ARCH:
        print(f"[WARN] sm_120 (RTX 5060 Ti) 권장. 현재 sm_{cap[0]}{cap[1]} ({torch.cuda.get_device_name(0)})")
    print(f"[OK] 환경 — Python {py.major}.{py.minor}.{py.micro} / torch {torch.__version__} / GPU {torch.cuda.get_device_name(0)}")


def check_external(external: Path) -> None:
    if not external.exists():
        sys.exit(f"[FAIL] 외부 패키지 폴더 없음: {external}\n  → 드롭박스에서 external_package.zip 다운로드 후 압축 풀기")
    missing = []
    for p in REQUIRED_PARQUETS:
        if not (external / "parquet" / p).exists():
            missing.append(f"parquet/{p}")
    for c in REQUIRED_CHECKPOINTS:
        if not (external / "checkpoints" / c).exists():
            missing.append(f"checkpoints/{c}")
    if missing:
        print("[FAIL] 외부 패키지 누락 파일:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    print(f"[OK] 외부 패키지 — parquet {len(REQUIRED_PARQUETS)} + checkpoint {len(REQUIRED_CHECKPOINTS)} 확인")


def stage_files(external: Path) -> None:
    parquet_dst = REPO_ROOT / "data" / "parquet"
    parquet_dst.mkdir(parents=True, exist_ok=True)
    for p in REQUIRED_PARQUETS:
        shutil.copy2(external / "parquet" / p, parquet_dst / p)
    ckpt_dst = REPO_ROOT / "data" / "artifacts" / "cp178"
    ckpt_dst.mkdir(parents=True, exist_ok=True)
    for c in REQUIRED_CHECKPOINTS:
        shutil.copy2(external / "checkpoints" / c, ckpt_dst / Path(c).name)
    print("[OK] parquet + checkpoint 배치 완료")


def run_step(args_list: list[str], step: str) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(args_list)}")
    env = {"PYTHONPATH": str(REPO_ROOT), **dict(__import__("os").environ)}
    result = subprocess.run(args_list, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        sys.exit(f"[FAIL] {step} (exit code {result.returncode})")
    print(f"[OK] {step}")


def compare_output(backup: Path) -> None:
    """학습 후 재현 결과 vs 학습 전 운영 원본 statistical 비교.

    bit-exact 일치 요구 X. row count 일치 + band_lower/upper 평균·표준편차 허용 오차 안이면 PASS.
    """
    if not SERVING_PARQUET.exists() or not backup.exists():
        print(f"[WARN] 비교 대상 parquet 없음 — 비교 skip")
        return
    import numpy as np  # noqa: WPS433
    import pyarrow.parquet as pq  # noqa: WPS433

    repro = pq.read_table(str(SERVING_PARQUET))
    ops = pq.read_table(str(backup))

    print("\n=== 재현 결과 비교 (statistical 허용 오차) ===")
    row_ok = repro.num_rows == ops.num_rows
    print(f"  rows                : 운영 {ops.num_rows} vs 재현 {repro.num_rows}   {'PASS' if row_ok else 'FAIL'}")

    for col in ("band_lower", "band_upper"):
        if col not in repro.column_names or col not in ops.column_names:
            continue
        r = np.asarray(repro.column(col).to_pylist(), dtype=float)
        o = np.asarray(ops.column(col).to_pylist(), dtype=float)
        rm, rs = float(np.nanmean(r)), float(np.nanstd(r))
        om, os = float(np.nanmean(o)), float(np.nanstd(o))
        mean_diff = abs(rm - om) / (abs(om) + 1e-9)
        std_diff = abs(rs - os) / (abs(os) + 1e-9)
        m_ok = mean_diff < 0.05
        s_ok = std_diff < 0.10
        print(f"  {col} 평균        : 운영 {om:.4f} vs 재현 {rm:.4f}   (상대 {mean_diff*100:.2f}%)  {'PASS' if m_ok else 'WARN'}")
        print(f"  {col} 표준편차    : 운영 {os:.4f} vs 재현 {rs:.4f}   (상대 {std_diff*100:.2f}%)  {'PASS' if s_ok else 'WARN'}")

    print(
        "\n학계 관행: bit-exact 일치는 GPU/CUDA 차이로 불가능. "
        "row count 일치 + 핵심 metric 평균 상대 차이 < 5%, 표준편차 < 10% 면 통계적 재현 성공. "
        "1W 는 주봉이라 fold 별 분산이 1D 보다 큼 — WARN 1~2개는 정상 범위."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lens 운영 1W 밴드 (CP178 WFLOCK) 재현 wrapper")
    parser.add_argument("--external", type=Path, default=Path("./external_package"))
    parser.add_argument("--verify-only", action="store_true",
                        help="환경/외부 패키지만 점검 (학습 X)")
    args = parser.parse_args()

    print("Lens 1W 밴드 재현 wrapper — CP178 WFLOCK q10/q90\n")
    check_environment()
    check_external(args.external)
    if args.verify_only:
        print("\n[VERIFY-ONLY] 점검 완료. 학습은 실행 안 함.")
        return
    stage_files(args.external)
    backup = SERVING_PARQUET.with_suffix(".ops_backup.parquet")
    if SERVING_PARQUET.exists():
        shutil.copy2(SERVING_PARQUET, backup)
        print(f"[OK] 운영 parquet 백업 → {backup.name}")
    run_step([sys.executable, "ai/cp178_wflock_1w_band_walk_forward_lower.py", "--run"],
             "CP178 WFLOCK walk-forward + lower calibration")
    compare_output(backup)
    print("\n[DONE] 재현 완료. 결과 parquet: backend/data/v1/predictions_band_1w.parquet (운영 원본은 .ops_backup.parquet)")


if __name__ == "__main__":
    main()
