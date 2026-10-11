"""The balance-drain rule and the overlap analysis, on hand-made rows."""
import numpy as np
import pandas as pd

from momosentry.evaluate import overlap
from momosentry.rules import balance_drain


def _tx(type_, old, new):
    return {"type": type_, "oldbalanceOrg": old, "newbalanceOrig": new}


def test_balance_drain_cases():
    df = pd.DataFrame([
        _tx("TRANSFER", 5000.0, 0.0),   # emptied an account with money  -> flag
        _tx("PAYMENT", 300.0, 0.0),     # any outgoing type counts         -> flag
        _tx("TRANSFER", 5000.0, 10.0),  # money left over                  -> no
        _tx("CASH_OUT", 0.0, 0.0),      # had nothing to begin with        -> no
        _tx("CASH_IN", 100.0, 0.0),     # CASH_IN cannot empty the sender  -> no
    ])
    assert balance_drain(df).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_overlap_counts():
    y = np.array([1, 1, 1, 0, 0, 0], bool)
    a = np.array([1, 1, 0, 1, 0, 0], bool)  # catches fraud 0 and 1, one false alarm
    b = np.array([0, 1, 1, 1, 1, 0], bool)  # catches fraud 1 and 2, two false alarms
    out = overlap({"A": a, "B": b}, y)
    p = out["pairs"][0]
    assert (p["both"], p["fraud_both"], p["fraud_only_a"], p["fraud_only_b"]) == (2, 1, 1, 1)
    assert abs(p["jaccard"] - 2 / 5) < 1e-12          # 2 shared out of 5 flagged by either
    assert out["fraud_caught_by_any"] == 3 and out["fraud_caught_by_none"] == 0
