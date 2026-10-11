"""Rule-based baselines.

A rule is a fixed yes/no test written by hand, with no learning. The project
compares the two unsupervised detectors against two of them:

1. PaySim's own `isFlaggedFraud` column (transfers above 200,000). That one is
   already in the data, so there is nothing to compute here.
2. A balance-drain rule, defined below.

Like features.py, a rule may only read the transaction's own fields. It never
reads `isFraud`, and it is never tuned on the holdout period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE_NAMES = {"drain": "Balance-drain rule", "flagged": "isFlaggedFraud rule"}


def balance_drain(df: pd.DataFrame) -> np.ndarray:
    """1.0 if an outgoing transaction leaves the sender with exactly 0, else 0.0.

    "Outgoing" means every type except CASH_IN, because CASH_IN puts money INTO
    the sender's account and so can never empty it.

    The rule deliberately does NOT restrict itself to TRANSFER and CASH_OUT,
    even though those are the only PaySim types that contain fraud. Knowing that
    requires looking at the fraud label, and a baseline built from label
    knowledge would be leakage. The rule encodes only the generic idea that
    "an account suddenly emptied is suspicious".
    """
    old_balance = df["oldbalanceOrg"].to_numpy(dtype=float)
    new_balance = df["newbalanceOrig"].to_numpy(dtype=float)
    outgoing = df["type"].astype(str).to_numpy() != "CASH_IN"
    drained = (old_balance > 0) & (new_balance == 0) & outgoing
    return drained.astype(float)
