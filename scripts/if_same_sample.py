"""Control experiment: fit Isolation Forest on the SAME type-stratified sample LOF uses.

The main pipeline fits IF on all training rows and LOF on a 100k sample. This
script removes that difference: IF is fitted on exactly the rows LOF is fitted
on, calibrated on different training rows, and scored on the full holdout.
It does not touch artifacts/.

    python scripts/if_same_sample.py --data data/paysim.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from momosentry import config, data  # noqa: E402
from momosentry.evaluate import evaluate_scores  # noqa: E402
from momosentry.features import build_features  # noqa: E402
from momosentry.models import Calibration, anomaly_score, make_isolation_forest  # noqa: E402

SEEDS = (42, 1, 2)
KEYS = ("roc_auc", "pr_auc", "pr_auc_adj", "precision_at_threshold", "recall_at_threshold",
        "alert_rate", "precision_at_top_1000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.PARQUET)
    args = ap.parse_args()

    rng = np.random.default_rng(config.RANDOM_STATE)
    train, holdout, _ = data.time_split(data.load(args.data))
    X_train, X_hold = build_features(train), build_features(holdout)

    fit_rows = data.type_stratified_sample(
        train.assign(_i=np.arange(len(train))), config.LOF_FIT_SAMPLE)["_i"].to_numpy()
    remaining = np.setdiff1d(np.arange(len(train)), fit_rows)
    cal_rows = rng.choice(remaining, size=50_000, replace=False)

    y = holdout["isFraud"].to_numpy()
    ref = float(train["isFraud"].mean())
    for seed in SEEDS:
        iso = make_isolation_forest().set_params(model__random_state=seed)
        iso.fit(X_train.iloc[fit_rows])
        cal = Calibration.from_train_scores(anomaly_score(iso, X_train.iloc[cal_rows]))
        m = evaluate_scores(y, anomaly_score(iso, X_hold), cal.threshold, reference_rate=ref)
        print(f"seed {seed}: " + ", ".join(f"{k}={m[k]:.4f}" for k in KEYS), flush=True)


if __name__ == "__main__":
    main()
