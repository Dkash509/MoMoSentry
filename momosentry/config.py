"""Central configuration. Every number that affects results lives here so it
can be quoted in the report and changed in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

RAW_CSV = DATA_DIR / "PS_20174392719_1491204439457_log.csv"  # Kaggle file name
PARQUET = DATA_DIR / "paysim.parquet"
SAMPLE_PARQUET = DATA_DIR / "paysim_sample.parquet"  # synthetic, for dev/CI

RANDOM_STATE = 42

# --- Split -------------------------------------------------------------------
# PaySim has 743 hourly steps. We split by TIME, not randomly: the model is
# trained on the past and tested on the future, which is how it would be used.
TRAIN_FRACTION_OF_STEPS = 0.70

# --- Isolation Forest ---------------------------------------------------------
IF_PARAMS = dict(n_estimators=200, max_samples=256, n_jobs=-1)

# --- Local Outlier Factor -----------------------------------------------------
# LOF is O(n^2)-ish, so it is fitted on a sample of the TRAIN period only.
# The sample is stratified by transaction TYPE (never by the fraud label).
LOF_FIT_SAMPLE = 100_000
LOF_PARAMS = dict(n_neighbors=35, novelty=True, n_jobs=-1)
# Max holdout rows LOF scores during evaluation (all holdout fraud is always
# kept; the rest is a random sample). None = score everything.
LOF_EVAL_MAX = 300_000

# --- Alerting -----------------------------------------------------------------
# The analyst can review 0.5% of traffic. The threshold is set on TRAIN scores
# (no labels), then applied unchanged to the holdout period.
ALERT_RATE = 0.005

# Budgets for the "how do precision and recall change as the budget grows"
# analysis (RQ3). Each one is a share of traffic the analyst can review; the
# threshold for each is again taken from TRAIN scores only. ALERT_RATE must be
# one of them so the headline result appears in the sweep.
ALERT_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02)

# --- Dashboard export -----------------------------------------------------------
# The app ships a small, precomputed slice of holdout scores instead of 6.3M rows.
EXPORT_NORMAL_ROWS = 40_000

TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
