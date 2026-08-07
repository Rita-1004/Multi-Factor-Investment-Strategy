"""Central configuration for the multi-factor investment project."""

from pathlib import Path


# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
TEST_DIR = PROJECT_ROOT / "tests"


# ---------------------------------------------------------------------
# Research sample
# ---------------------------------------------------------------------

FULL_SAMPLE_START = "2005-01-31"
FULL_SAMPLE_END = "2025-12-31"

BACKTEST_START = "2015-01-31"
BACKTEST_END = "2025-12-31"

MONTHS_PER_YEAR = 12
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------

RAW_FACTOR_COLUMNS = [
    "book_to_market",
    "earnings_to_price",
    "momentum_12_1",
    "profitability_raw",
    "investment_growth_raw",
    "log_market_cap",
    "volatility_12m",
]

FACTOR_SCORE_COLUMNS = [
    "factor_value",
    "factor_momentum",
    "factor_quality",
    "factor_investment",
    "factor_size",
    "factor_low_volatility",
]

PRIMARY_SIGNAL_COLUMNS = [
    "factor_quality",
    "multi_factor_score",
]


# ---------------------------------------------------------------------
# Portfolio settings
# ---------------------------------------------------------------------

DEFAULT_SELECTION_PERCENTAGE = 0.20
DEFAULT_TRANSACTION_COST_BPS = 10
DEFAULT_TRANSACTION_COST = DEFAULT_TRANSACTION_COST_BPS / 10_000

ROBUSTNESS_SELECTION_PERCENTAGES = [
    0.10,
    0.20,
    0.30,
]

ROBUSTNESS_TRANSACTION_COSTS_BPS = [
    0,
    10,
    25,
    50,
]

WEIGHTING_METHODS = [
    "Equal Weight",
    "Inverse Volatility",
    "Score Rank Weight",
]


# ---------------------------------------------------------------------
# Statistical settings
# ---------------------------------------------------------------------

DEFAULT_WINSORIZATION_LOWER = 0.01
DEFAULT_WINSORIZATION_UPPER = 0.99
DEFAULT_NEWEY_WEST_LAGS = 6
DEFAULT_VAR_CONFIDENCE_LEVEL = 0.95
DEFAULT_ROLLING_WINDOW = 36


# ---------------------------------------------------------------------
# Machine-learning settings
# ---------------------------------------------------------------------

ML_FEATURE_COLUMNS = [
    "factor_value",
    "factor_momentum",
    "factor_quality",
    "factor_investment",
    "factor_size",
    "factor_low_volatility",
]

ML_TARGET_COLUMN = "future_return_1m"

RIDGE_ALPHA = 1.0

XGBOOST_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def create_project_directories():
    """Create the standard project directories when they do not exist."""

    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        TEST_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)