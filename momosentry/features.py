"""Feature engineering.

Rule: every feature is computed from ONE transaction's own fields. No feature
uses `isFraud`, `isFlaggedFraud`, or statistics computed on the holdout period.
That is what keeps the evaluation honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TX_TYPES

TYPE_COLS = [f"type_{t}" for t in TX_TYPES]
NUMERIC_COLS = [
    "log_amount",          # size of the transaction
    "hour_sin", "hour_cos",  # time of day, encoded as a circle so 23h is next to 0h
    "log_old_org",         # how much money the sender had
    "amount_to_balance",   # share of the sender's balance moved (log-scaled)
    "orig_emptied",        # sender ends at exactly 0 after having money
    "err_orig",            # sender's balance doesn't add up (signed, scaled)
    "err_dest",            # receiver's balance doesn't add up (signed, scaled)
    "dest_zero_both",      # receiver shows 0 before AND after receiving money
    "dest_is_merchant",    # receiver id starts with 'M'
]
FEATURE_NAMES = TYPE_COLS + NUMERIC_COLS

# Groups used by the perturbation explainer (explain.py). Type / merchant are
# context, not "reasons", so they are not perturbed.
EXPLAIN_GROUPS = {
    "Transaction amount": ["log_amount"],
    "Share of sender balance moved": ["amount_to_balance", "orig_emptied"],
    "Sender balance does not add up": ["err_orig"],
    "Sender balance size": ["log_old_org"],
    "Receiver balance does not add up": ["err_dest", "dest_zero_both"],
    "Time of day": ["hour_sin", "hour_cos"],
}

RAW_INPUTS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
              "nameDest", "oldbalanceDest", "newbalanceDest"]


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _per_row(col: pd.Series, test) -> np.ndarray:
    """True/False for every row, where `test` is a check on the text value.

    PaySim's `type` and `nameDest` columns are categorical: millions of rows
    but only a handful of distinct values. So the check runs once per distinct
    value and is then looked up for each row, instead of turning 6 million
    values into Python strings first. Plain (non-categorical) input, such as
    the single row typed into the scoring form, uses the ordinary slow path.
    """
    if isinstance(col.dtype, pd.CategoricalDtype):
        per_value = np.array([test(str(v)) for v in col.cat.categories], dtype=bool)
        return per_value[col.cat.codes.to_numpy()]
    return col.astype(str).map(test).to_numpy(dtype=bool)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw PaySim rows into the 15 features, one column at a time.

    Memory: the result is written straight into one pre-allocated float32
    matrix. Each feature is still calculated in float64 exactly as before and
    only rounded to float32 when it is stored, which is the same rounding the
    old `.astype(np.float32)` did, so the values are bit-for-bit identical.
    """
    amount = df["amount"].to_numpy(dtype=float)
    old_o = df["oldbalanceOrg"].to_numpy(dtype=float)
    new_o = df["newbalanceOrig"].to_numpy(dtype=float)
    old_d = df["oldbalanceDest"].to_numpy(dtype=float)
    new_d = df["newbalanceDest"].to_numpy(dtype=float)
    merchant = _per_row(df["nameDest"], lambda v: v.startswith("M"))
    hour = df["step"].to_numpy() % 24

    X = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    position = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def put(name, values):
        X[:, position[name]] = values  # numpy rounds float64 -> float32 here

    for col_name, t in zip(TYPE_COLS, TX_TYPES):
        put(col_name, _per_row(df["type"], lambda v, t=t: v == t))

    is_cash_in = _per_row(df["type"], lambda v: v == "CASH_IN")
    # Expected balances if the ledger were consistent
    exp_new_o = np.where(is_cash_in, old_o + amount, old_o - amount)
    exp_new_d = np.where(is_cash_in, old_d - amount, old_d + amount)

    put("log_amount", np.log1p(amount))
    put("hour_sin", np.sin(2 * np.pi * hour / 24))
    put("hour_cos", np.cos(2 * np.pi * hour / 24))
    put("log_old_org", np.log1p(old_o))
    put("amount_to_balance", np.log1p(np.clip(amount / (old_o + 1.0), 0, 1e3)))
    put("orig_emptied", (old_o > 0) & (new_o == 0) & ~is_cash_in)
    put("err_orig", _signed_log((new_o - exp_new_o) / (amount + 1.0) * 100))
    del exp_new_o
    # Merchants in PaySim never have balances, so their dest error carries no information.
    put("err_dest", np.where(merchant, 0.0, _signed_log((new_d - exp_new_d) / (amount + 1.0) * 100)))
    del exp_new_d
    put("dest_zero_both", (old_d == 0) & (new_d == 0) & (amount > 0) & ~merchant)
    put("dest_is_merchant", merchant)

    return pd.DataFrame(X, index=df.index, columns=FEATURE_NAMES)
