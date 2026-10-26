"""The perturbation explainer, tested against stand-in detectors whose behaviour is known in advance."""
import numpy as np
import pandas as pd

from momosentry.config import TX_TYPES
from momosentry.explain import explain
from momosentry.features import EXPLAIN_GROUPS, FEATURE_NAMES, TYPE_COLS
from momosentry.models import Bundle


class _ReadsOneColumn:
    """Stand-in detector whose score depends on ONE feature column, so the correct
    explanation is known. sklearn convention: score_samples is lower = more anomalous,
    and anomaly_score() negates it, so the anomaly score here is |feature|."""

    def __init__(self, column):
        self.column = column

    def score_samples(self, X):
        return -np.abs(X[self.column].to_numpy(dtype=float))


def _bundle(model, typical_log_amount=0.0):
    typical = pd.DataFrame(0.0, index=pd.Index(TX_TYPES, name="_type"), columns=FEATURE_NAMES)
    typical["log_amount"] = typical_log_amount
    return Bundle(models={"if": model}, calibration={}, typical_by_type=typical)


def _tx(**kw):
    base = dict(step=13, type="CASH_OUT", amount=1_000.0, oldbalanceOrg=5_000.0, newbalanceOrig=4_000.0,
                nameDest="C", oldbalanceDest=0.0, newbalanceDest=1_000.0)
    return pd.DataFrame([base | kw])


def test_groups_and_context_cover_all_15_features_without_overlap():
    perturbed = [c for cols in EXPLAIN_GROUPS.values() for c in cols]
    context = TYPE_COLS + ["dest_is_merchant"]
    assert len(perturbed) == len(set(perturbed))                 # no feature sits in two groups
    assert not set(perturbed) & set(context)                     # context features are never reset
    assert sorted(perturbed + context) == sorted(FEATURE_NAMES)  # nothing is left out


def test_planted_extreme_group_ranks_first_and_takes_the_whole_share():
    # The stand-in reads only log_amount, so only the amount group can explain the score.
    ex = explain(_bundle(_ReadsOneColumn("log_amount"), typical_log_amount=8.0), _tx(amount=1e9))
    assert len(ex) == 6
    assert ex.reason.iloc[0] == "Transaction amount"
    assert abs(ex.share.iloc[0] - 1.0) < 1e-9 and (ex.share.iloc[1:] == 0).all()
    assert abs(ex.score_drop.iloc[0] - (np.log1p(1e9) - 8.0)) < 1e-3


def test_transaction_type_is_context_and_is_never_reset():
    # A detector that reads only the type indicator: nothing may be blamed on any group.
    ex = explain(_bundle(_ReadsOneColumn("type_CASH_OUT")), _tx())
    assert (ex.score_drop == 0).all() and (ex.share == 0).all()


def test_shares_add_up_to_one_when_something_is_to_blame():
    ex = explain(_bundle(_ReadsOneColumn("log_amount"), typical_log_amount=2.0), _tx(amount=50_000.0))
    assert abs(ex.share.sum() - 1.0) < 1e-9
