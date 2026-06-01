"""CP216 — 경로 / 검정 상수."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

OPS_LINE_1D = ROOT / "backend" / "data" / "v1" / "predictions_line_1d.parquet"
OPS_BAND_1D = ROOT / "backend" / "data" / "v1" / "predictions_band_1d.parquet"
OPS_BAND_1W = ROOT / "backend" / "data" / "v1" / "predictions_band_1w.parquet"
CP175_FROZEN = ROOT / "backend" / "data" / "v1" / "predictions_line_1d_cp175_frozen_backup.parquet"

PRICES_1D = ROOT / "data" / "parquet" / "price_data_yfinance_500.parquet"
PRICES_1W = ROOT / "data" / "parquet" / "price_data_yfinance_1W.parquet"
INDICATORS_1D = ROOT / "data" / "parquet" / "indicators_yfinance_1D_500.parquet"

BASELINE_DIR = ROOT / "docs" / "cp216_significance_baselines"
SIG_DIR = ROOT / "docs" / "cp216_significance"
RUNS_DIR = ROOT / "docs" / "cp216_significance_runs"

# CP216.2 fix: 운영 분위수는 timeframe 별 다름.
#   CP153 1D : q15/q85 (target coverage 70%) — reproducibility.md §2
#   CP178 1W : q10/q90 (target coverage 80%) — reproducibility.md §3
# CP216 은 둘 다 q10/q90 으로 잘못 가정 → 1D pinball 재계산 필요.
QUANTILES_BY_TIMEFRAME = {
    "1D": (0.15, 0.85),
    "1W": (0.10, 0.90),
}
# 호환: CP216 mode 에서 legacy 사용 (둘 다 q10/q90)
Q_LOW = 0.10
Q_HIGH = 0.90
Q_LOW_1D_CP216_2 = 0.15
Q_HIGH_1D_CP216_2 = 0.85
Q_LOW_1W_CP216_2 = 0.10
Q_HIGH_1W_CP216_2 = 0.90

# CP216.2 출력 경로
SIG_DIR_CP216_2 = ROOT / "docs" / "cp216_2_significance"
BASELINE_DIR_CP216_2 = ROOT / "docs" / "cp216_2_significance_baselines"
RUNS_DIR_CP216_2 = ROOT / "docs" / "cp216_2_significance_runs"
PRICES_1W_500 = ROOT / "data" / "parquet" / "price_data_yfinance_1W_500.parquet"

# 베이스라인 추정 윈도우
HISTORICAL_WINDOW = 200  # historical_mean / historical_quantile 학습 윈도우 (거래일)
BOLLINGER_WINDOW = 20    # 이동평균 + 2σ
BOLLINGER_SIGMA = 2.0

# Block bootstrap 길이 두 가지 (사용자 lock)
# √T 와 22 (월 단위) 둘 다 돌림. block_sqrt_t 는 시계열 길이마다 동적 계산.
BLOCK_SIZES = {
    "block_sqrt_t": "sqrt_t",  # 동적 — len(series) ** 0.5
    "block_22": 22,
}

# Regime
VIX_THRESHOLD = 30.0
DD_WINDOW = 200
DD_THRESHOLD = -0.10  # 200d rolling max 대비 -10% 이상 drawdown

# 검정 상수
HAC_LAGS_AUTO_FACTOR = 4.0  # Newey-West 자동 lag = floor(4 * (T/100)^(2/9))
N_BOOTSTRAP_DEFAULT = 1000
CI_LEVEL = 0.95
