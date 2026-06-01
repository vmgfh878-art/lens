/**
 * CP220 — 운영 3 모델 재현성 매니페스트 (env, fold, seeds, artifacts, repro steps, TODO).
 *
 * 단일 진리: `docs/v1_operating_models_reproducibility.md` §0 (환경 공통) + §1~3 (모델별).
 * v1 동안 정적. v2 manifest 자동 생성으로 transition 시 fetcher 로 교체.
 */
import type { ProductSlotId } from "@/lib/productSlots";

export interface FoldWindow {
  /** fold 식별자 (예: "W1", "fold_1"). */
  foldId: string;
  /** 학습 시작 (미수록 시 생략). */
  trainStart?: string;
  trainEnd: string;
  valStart?: string;
  valEnd?: string;
  testStart: string;
  testEnd: string;
}

export interface PackageVersion {
  name: string;
  version: string;
}

export interface ReproducibilityBlock {
  /** parquet model_id (예: "cp210_F4_b4_ensemble_mean"). */
  runId: string;
  /** 원본 CP 식별자 (예: "CP208Z_CP209_F4B4"). */
  sourceCp: string;
  /** 백본 (예: "PatchTST p32/s16", "TiDE"). */
  backbone: string;
  /** 출력 contract 한 줄. */
  outputContract: string;
  /** calibration 정의 (라인은 없음 → 생략). */
  calibration?: string;
  seeds: number[];
  folds: FoldWindow[];
  pythonVersion: string;
  torchVersion: string;
  keyPackages: PackageVersion[];
  gpuName: string;
  gpuArch: string;
  cudaRuntime: string;
  gpuEnv: string[];
  /** 깃 안 산출물 (외부에서 클론 후 바로 볼 수 있음). */
  artifactsInGit: string[];
  /** 외부 패키지 산출물 (드롭박스 등 — 학습 재현 위해 다운로드 필요). */
  artifactsExternal: string[];
  servingParquetPath: string;
  /** 학습 재현 명령 — 외부 패키지 다운로드 후 실행. GPU sm_120 필수. */
  trainingSteps: string[];
}

// 공통 환경 — `docs/v1_operating_models_reproducibility.md` §0
const COMMON_PYTHON = "3.10.0";
const COMMON_TORCH = "2.11.0+cu128";
const COMMON_PACKAGES: PackageVersion[] = [
  { name: "numpy", version: "1.26.4" },
  { name: "pandas", version: "2.2.2" },
  { name: "pyarrow", version: "16.1.0" },
  { name: "scipy", version: "1.15.3" },
  { name: "fastapi", version: "0.111.0" },
  { name: "uvicorn", version: "0.30.1" },
  { name: "optuna", version: "4.8.0" },
];
const COMMON_GPU_NAME = "NVIDIA GeForce RTX 5060 Ti";
const COMMON_GPU_ARCH = "sm_120 (compute capability 12.0)";
const COMMON_CUDA_RUNTIME = "12.8";
const COMMON_GPU_ENV = [
  "KMP_DUPLICATE_LIB_OK=TRUE",
  "TORCHDYNAMO_DISABLE=1",
  "DataLoader num_workers=0 (Windows + sm_120 폴백)",
];

const LINE_REPRO: ReproducibilityBlock = {
  runId: "cp210_F4_b4_ensemble_mean",
  sourceCp: "CP208Z_CP209_F4B4",
  backbone: "PatchTST p32/s16",
  outputContract: "score (line_score, safe_line_score; 수익률 단위, 화면 환산). 손실 Asymmetric MSE α=1 β=4. 5-seed mean ensemble.",
  seeds: [7, 13, 23, 42, 71],
  folds: [
    { foldId: "W1", trainEnd: "2024-10-29", testStart: "2024-10-30", testEnd: "2025-02-28" },
    { foldId: "W2", trainEnd: "2025-02-28", testStart: "2025-03-01", testEnd: "2025-06-30" },
    { foldId: "W3", trainEnd: "2025-06-30", testStart: "2025-07-01", testEnd: "2025-10-31" },
    { foldId: "W4", trainEnd: "2025-10-31", testStart: "2025-11-01", testEnd: "2026-05-01" },
  ],
  pythonVersion: COMMON_PYTHON,
  torchVersion: COMMON_TORCH,
  keyPackages: COMMON_PACKAGES,
  gpuName: COMMON_GPU_NAME,
  gpuArch: COMMON_GPU_ARCH,
  cudaRuntime: COMMON_CUDA_RUNTIME,
  gpuEnv: COMMON_GPU_ENV,
  artifactsInGit: [
    "scripts/reproduce_line_cp210.py (한 줄 재현 wrapper — 환경/패키지 점검 + 학습 실행)",
    "docs/cp210_ensemble_report.md",
    "docs/cp210_progress_latest.md",
    "ai/cp209_lm_f4_f6_pre_ship_verification.py + ai/cp210_lm_ensemble_ship_verification.py (학습 entry)",
    "ai/cp208z, cp164, cp175, cp160, cp171 (학습 의존 cascade — 자동 import 됨)",
  ],
  artifactsExternal: [
    "external_package/checkpoints/cp209/seed_{7,13,23,42,71}.pt (5 seed checkpoint, ~수 GB)",
    "external_package/parquet/price_data_yfinance_500.parquet (학습 시점 스냅샷, source_data_hash 11bf3a4831d54815)",
    "external_package/parquet/indicators_yfinance_1D_500.parquet (학습 시점 스냅샷)",
  ],
  servingParquetPath: "backend/data/v1/predictions_line_1d.parquet",
  trainingSteps: [
    "드롭박스에서 external_package_line.zip 다운로드 후 압축 풀기",
    ".venv\\Scripts\\Activate.ps1; pip install -r requirements.txt",
    "python scripts\\reproduce_line_cp210.py --external .\\external_package  (~12h on RTX 5060 Ti)",
    "wrapper 가 환경/패키지/CP209 학습/CP210 ensemble forward 자동 수행 + 결과 검증",
  ],
};

const BAND_1D_REPRO: ReproducibilityBlock = {
  runId: "tide-1D-ea54dcae654d",
  sourceCp: "CP153",
  backbone: "TiDE (Time-series Dense Encoder, Google 2023)",
  outputContract: "quantile pair (q_low 0.15 / q_high 0.85, target coverage 70%) → conformal 보정",
  calibration: "lower_focused (validation-only fit, test 고정 적용)",
  seeds: [7, 42, 123],
  folds: [
    { foldId: "fold_1", trainStart: "2019-05-01", trainEnd: "2024-05-01", valStart: "2024-05-01", valEnd: "2024-11-01", testStart: "2024-11-01", testEnd: "2025-05-01" },
    { foldId: "fold_2", trainStart: "2019-11-01", trainEnd: "2024-11-01", valStart: "2024-11-01", valEnd: "2025-05-01", testStart: "2025-05-01", testEnd: "2025-11-01" },
    { foldId: "fold_3", trainStart: "2020-05-01", trainEnd: "2025-05-01", valStart: "2025-05-01", valEnd: "2025-11-01", testStart: "2025-11-01", testEnd: "2026-05-09" },
  ],
  pythonVersion: COMMON_PYTHON,
  torchVersion: COMMON_TORCH,
  keyPackages: COMMON_PACKAGES,
  gpuName: COMMON_GPU_NAME,
  gpuArch: COMMON_GPU_ARCH,
  cudaRuntime: COMMON_CUDA_RUNTIME,
  gpuEnv: COMMON_GPU_ENV,
  artifactsInGit: [
    "scripts/reproduce_band_1d_cp153.py (한 줄 재현 wrapper)",
    "docs/cp153_bm_1d_band_primary_product_candidate_save_run_report.md",
    "ai/cp153_bm_1d_band_primary_save_run.py (학습 entry)",
    "ai/cp153_bm_1d_band_500_stage{0_1, 2, 2_5_to_5, 4r_5t} (학습 의존 cascade — 자동 import 됨)",
  ],
  artifactsExternal: [
    "external_package/checkpoints/cp153/tide-1D-ea54dcae654d-seed_{7,42,123}.pt (3 seed checkpoint, ~수 GB)",
    "external_package/parquet/price_data_yfinance_500.parquet (학습 시점 스냅샷, source_data_hash 90666b44cbfb8e5c)",
    "external_package/parquet/indicators_yfinance_1D_500.parquet",
    "Stage별 보고서 묶음 (cp153_*_stage{0_1,2,2_5,3,4,5}_*.md) — 운영 핵심만 깃, 나머지는 외부",
  ],
  servingParquetPath: "backend/data/v1/predictions_band_1d.parquet",
  trainingSteps: [
    "드롭박스에서 external_package_band_1d.zip 다운로드 후 압축 풀기",
    ".venv\\Scripts\\Activate.ps1; pip install -r requirements.txt",
    "python scripts\\reproduce_band_1d_cp153.py --external .\\external_package",
    "wrapper 가 환경/패키지/CP153 primary save-run 자동 수행 + 결과 검증",
  ],
};

const BAND_1W_REPRO: ReproducibilityBlock = {
  runId: "tide_s60_q10_q90_param",
  sourceCp: "CP178-WFLOCK",
  backbone: "TiDE (1D 와 동일 백본)",
  outputContract: "quantile pair (q_low 0.10 / q_high 0.90, target coverage 80%)",
  calibration: "walk-forward lower calibration (fold 별 별도 fit, WFLOCK)",
  seeds: [7, 42, 123],
  folds: [
    { foldId: "fold_1", trainEnd: "2024-05-01", testStart: "2024-11-01", testEnd: "2025-05-01" },
    { foldId: "fold_2", trainEnd: "2024-11-01", testStart: "2025-05-01", testEnd: "2025-11-01" },
    { foldId: "fold_3", trainEnd: "2025-05-01", testStart: "2025-11-01", testEnd: "2026-05-09" },
  ],
  pythonVersion: COMMON_PYTHON,
  torchVersion: COMMON_TORCH,
  keyPackages: COMMON_PACKAGES,
  gpuName: COMMON_GPU_NAME,
  gpuArch: COMMON_GPU_ARCH,
  cudaRuntime: COMMON_CUDA_RUNTIME,
  gpuEnv: COMMON_GPU_ENV,
  artifactsInGit: [
    "scripts/reproduce_band_1w_cp178.py (한 줄 재현 wrapper)",
    "docs/cp178_wflock_1w_band_walk_forward_lower_report.md",
    "ai/cp178_wflock_1w_band_walk_forward_lower.py (WFLOCK 학습 entry — cascade 없음, 단독)",
  ],
  artifactsExternal: [
    "external_package/checkpoints/cp178/tide_s60_q10_q90_param-seed_{7,42,123}.pt (3 seed checkpoint, ~수 GB)",
    "external_package/parquet/price_data_yfinance_500.parquet (1D 가격, weekly resample 입력)",
    "external_package/parquet/indicators_yfinance_1W_500.parquet (1W 가공 feature)",
    "Stage별 보고서 묶음 (cp178_*_stage{0,1,2,3,4,5}_*.md, cp178_cal_*, cp178_alt_*) — 운영 핵심만 깃, 나머지는 외부",
  ],
  servingParquetPath: "backend/data/v1/predictions_band_1w.parquet",
  trainingSteps: [
    "드롭박스에서 external_package_band_1w.zip 다운로드 후 압축 풀기",
    ".venv\\Scripts\\Activate.ps1; pip install -r requirements.txt",
    "python scripts\\reproduce_band_1w_cp178.py --external .\\external_package",
    "wrapper 가 환경/패키지/CP178 WFLOCK walk-forward + lower calibration 자동 수행 + 결과 검증",
  ],
};

export const REPRODUCIBILITY: Record<string, ReproducibilityBlock> = {
  "line-1d": LINE_REPRO,
  "band-1d": BAND_1D_REPRO,
  "band-1w": BAND_1W_REPRO,
};

export function getReproducibility(slotId: ProductSlotId | string | null | undefined): ReproducibilityBlock | null {
  if (!slotId) {
    return null;
  }
  return REPRODUCIBILITY[slotId] ?? null;
}
