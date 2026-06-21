/**
 * CP220 — 운영 3 모델 재현성 매니페스트 (env, fold, seeds, artifacts, 재현 경로 A/B).
 *
 * 단일 진리: `docs/v1_operating_models_reproducibility.md` 섹션0 (환경 공통) + 섹션1~3 (모델별)
 * + Google Drive 재현 패키지의 모델별 `재현절차.md`.
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
  /** 깃 클론에 바로 포함된 산출물. */
  artifactsInGit: string[];
  /** Google Drive 재현 패키지 동봉물 — 체크포인트·학습 데이터. */
  drivePackage: string[];
  /** Drive 재현 패키지 폴더 링크. */
  driveUrl: string;
  servingParquetPath: string;
  /** 재현 경로 A — 체크포인트 추론. 빠르고 권장. */
  pathASteps: string[];
  /** 재현 경로 B — 재학습. statistical 재현, GPU sm_120 권장. */
  pathBSteps: string[];
}

// 공통 환경 — `docs/v1_operating_models_reproducibility.md` 섹션0
const COMMON_PYTHON = "3.10.0";
const COMMON_TORCH = "2.11.0+cu128";
const COMMON_PACKAGES: PackageVersion[] = [
  { name: "numpy", version: "1.26.4" },
  { name: "pandas", version: "2.2.2" },
  { name: "pyarrow", version: "17.0.0" },
  { name: "scipy", version: "1.15.3" },
  { name: "fastapi", version: "0.115.6" },
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

// 공통 — Google Drive 재현 패키지 폴더
const DRIVE_URL = "https://drive.google.com/drive/folders/15Y_wLokJP_Y8uOK6WXYgXX-3JqAnw1Q9";
const INSTALL_STEP =
  "GitHub clone, Python 3.10 .venv 준비, pip install -r requirements.txt -r backend/requirements.txt -r backend/collector/requirements.txt";

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
    "ai/cp209_lm_f4_f6_pre_ship_verification.py + ai/cp210_lm_ensemble_ship_verification.py (학습·앙상블 entry)",
    "ai/cp208z, cp164, cp175, cp160, cp171 + line cascade (자동 import)",
    "scripts/reproduce_line_cp210.py (한 줄 재현 wrapper)",
    "docs/cp210_ensemble_report.md · docs/cp210_progress_latest.md · docs/cp63_bm_feature_set_plan.json",
  ],
  drivePackage: [
    "20_per_model/line_CP210/checkpoints/ — cp209/seed_stability seed 7·13·23·71 + cp208z seed42 (운영 F4_b4 5-seed)",
    "10_training_data/latest_full/1D/ — price_data_yfinance_500.parquet · indicators_yfinance_1D_500.parquet (+ .manifest.json)",
    "00_serving_latest/predictions_line_1d.parquet — 대조용 운영 출력",
  ],
  driveUrl: DRIVE_URL,
  servingParquetPath: "backend/data/v1/predictions_line_1d.parquet",
  pathASteps: [
    INSTALL_STEP,
    "Drive 20_per_model/line_CP210/checkpoints/ → ai/artifacts/checkpoints/ 같은 구조로 복사",
    "Drive 10_training_data/latest_full/1D/ parquet → data/parquet/ 배치",
    "python ai/cp210_lm_ensemble_ship_verification.py — 기본값이 운영 F4_b4 한 개라 추론만 실행 (F6 체크포인트 불필요)",
    "출력이 00_serving_latest/predictions_line_1d.parquet 과 통계적으로 일치하는지 확인",
  ],
  pathBSteps: [
    "training_cutoff/1D/ parquet 을 data/parquet/ 에 두면 학습 시간 윈도우",
    "python ai/cp209_lm_f4_f6_pre_ship_verification.py — 5-seed 학습",
    "python ai/cp210_lm_ensemble_ship_verification.py — 앙상블 forward",
    "한 줄 wrapper: python scripts/reproduce_line_cp210.py --external <Drive 다운로드 폴더>  (~12h, RTX 5060 Ti)",
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
    "ai/cp153_bm_1d_band_primary_save_run.py + cp153 stage cascade (자동 import)",
    "backend/scripts/cp210_band_forward_refresh.py (체크포인트 추론 entry)",
    "scripts/reproduce_band_1d_cp153.py (한 줄 재현 wrapper)",
    "docs/cp153_bm_1d_band_primary_product_candidate_run_meta.json (체크포인트·registry 경로 자동 참조) · save_run_report.md",
  ],
  drivePackage: [
    "20_per_model/band1d_CP153/checkpoints/tide_1D-ea54dcae654d.pt — 운영 체크포인트",
    "10_training_data/latest_full/{1D,1W}/ parquet (+ .manifest.json) — refresh 가 1D·1W 함께 읽음",
    "00_serving_latest/predictions_band_1d.parquet — 대조용 운영 출력",
  ],
  driveUrl: DRIVE_URL,
  servingParquetPath: "backend/data/v1/predictions_band_1d.parquet",
  pathASteps: [
    INSTALL_STEP,
    "Drive 20_per_model/band1d_CP153/checkpoints/tide_1D-ea54dcae654d.pt → ai/artifacts/checkpoints/ 배치",
    "Drive 10_training_data/latest_full/{1D,1W}/ parquet → data/parquet/ 배치",
    "python backend/scripts/cp210_band_forward_refresh.py --apply — 1D·1W 밴드 함께 생성",
    "출력이 00_serving_latest/predictions_band_1d.parquet 과 일치하는지 확인",
  ],
  pathBSteps: [
    "training_cutoff/1D/ parquet 을 data/parquet/ 에 두면 학습 시간 윈도우",
    "python ai/cp153_bm_1d_band_primary_save_run.py — 기본 3 epoch, 에폭 오버라이드 인자 없음",
    "한 줄 wrapper: python scripts/reproduce_band_1d_cp153.py --external <Drive 다운로드 폴더>",
  ],
};

const BAND_1W_REPRO: ReproducibilityBlock = {
  runId: "tide_s104_q10q90_param",
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
    "ai/cp178_wflock_1w_band_walk_forward_lower.py + cp178 cascade (cal/alt/diag/stage 자동 import)",
    "backend/scripts/cp210_band_forward_refresh.py (체크포인트 추론 entry, 1D·1W 공용)",
    "scripts/reproduce_band_1w_cp178.py (한 줄 재현 wrapper)",
    "docs/cp178_bm_1w_band_500_stage5_true_walk_forward_summary.csv (운영 9개 선택·calibration 자동 참조) · cp178_wflock report.md",
  ],
  drivePackage: [
    "20_per_model/band1w_CP178/checkpoints/ — tide_1W-*.pt 9개 (3-fold × 3-seed, 운영 tide_s104_q10q90_param)",
    "10_training_data/latest_full/{1D,1W}/ parquet (+ .manifest.json)",
    "00_serving_latest/predictions_band_1w.parquet — 대조용 운영 출력",
  ],
  driveUrl: DRIVE_URL,
  servingParquetPath: "backend/data/v1/predictions_band_1w.parquet",
  pathASteps: [
    INSTALL_STEP,
    "Drive 20_per_model/band1w_CP178/checkpoints/ 의 tide_1W-*.pt 9개 → ai/artifacts/checkpoints/ 배치",
    "Drive 10_training_data/latest_full/{1D,1W}/ parquet → data/parquet/ 배치",
    "python backend/scripts/cp210_band_forward_refresh.py --apply — summary.csv 가 운영 s104 9개 체크포인트·calibration 선택",
    "출력이 00_serving_latest/predictions_band_1w.parquet 과 일치하는지 확인",
  ],
  pathBSteps: [
    "운영 1W 밴드 tide_s104 재현은 경로 A 로 충분하다. 운영 체크포인트 9개가 패키지에 동봉돼 추론만으로 재현된다.",
    "ai/cp178_wflock_1w_band_walk_forward_lower.py 는 운영 모델 재학습이 아니라 비운영 후보 tide_s60 의 walk-forward 하단 보정 분석이다. 그 후보 체크포인트는 용량상 미동봉.",
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
