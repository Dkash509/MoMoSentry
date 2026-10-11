"""Perturbation-based explanations ("what if this part were normal?").

For one transaction, we take each group of features in turn, reset it to the
typical value for that transaction type (median of the TRAINING period), and
re-score. The bigger the drop in anomaly score, the more that group is to blame.

Chosen over SHAP because it works the same way for both models, needs no extra
library, and is easy to explain to a non-technical reviewer.
"""
from __future__ import annotations

import pandas as pd

from .features import EXPLAIN_GROUPS, build_features
from .models import Bundle, anomaly_score


def explain(bundle: Bundle, raw_row: pd.DataFrame, model_key: str = "if") -> pd.DataFrame:
    assert len(raw_row) == 1, "explain one transaction at a time"
    model = bundle.models[model_key]
    X = build_features(raw_row)
    base = float(anomaly_score(model, X)[0])
    tx_type = str(raw_row["type"].iloc[0])
    typical = bundle.typical_by_type.loc[tx_type]

    rows = []
    for reason, cols in EXPLAIN_GROUPS.items():
        X_pert = X.copy()
        X_pert.loc[:, cols] = typical[cols].to_numpy()
        new = float(anomaly_score(model, X_pert)[0])
        rows.append({"reason": reason, "score_drop": base - new})

    out = pd.DataFrame(rows).sort_values("score_drop", ascending=False).reset_index(drop=True)
    total = out["score_drop"].clip(lower=0).sum()
    out["share"] = out["score_drop"].clip(lower=0) / total if total > 0 else 0.0
    return out
