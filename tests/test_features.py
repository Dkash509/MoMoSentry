import numpy as np
import pandas as pd

from momosentry.features import FEATURE_NAMES, build_features


def row(**kw):
    base = dict(step=13, type="PAYMENT", amount=100.0, oldbalanceOrg=1000.0, newbalanceOrig=900.0,
                nameDest="M1", oldbalanceDest=0.0, newbalanceDest=0.0)
    return pd.DataFrame([base | kw])


def test_shape_and_no_nans():
    X = build_features(row())
    assert list(X.columns) == FEATURE_NAMES
    assert not X.isna().any().any()


def test_consistent_ledger_has_zero_error():
    X = build_features(row(type="TRANSFER", nameDest="C1", oldbalanceDest=50.0, newbalanceDest=150.0))
    assert X.err_orig.iloc[0] == 0 and X.err_dest.iloc[0] == 0


def test_cash_in_direction():
    X = build_features(row(type="CASH_IN", nameDest="C1", oldbalanceOrg=1000, newbalanceOrig=1100,
                           oldbalanceDest=500, newbalanceDest=400))
    assert X.err_orig.iloc[0] == 0 and X.err_dest.iloc[0] == 0
    assert X.orig_emptied.iloc[0] == 0


def test_drained_account_flags():
    X = build_features(row(type="TRANSFER", amount=1000, oldbalanceOrg=1000, newbalanceOrig=0,
                           nameDest="C1", oldbalanceDest=0, newbalanceDest=0))
    assert X.orig_emptied.iloc[0] == 1 and X.dest_zero_both.iloc[0] == 1
    assert X.type_TRANSFER.iloc[0] == 1


def test_hour_is_circular():
    a = build_features(row(step=23))[["hour_sin", "hour_cos"]].to_numpy()
    b = build_features(row(step=24))[["hour_sin", "hour_cos"]].to_numpy()  # hour 0
    c = build_features(row(step=12))[["hour_sin", "hour_cos"]].to_numpy()
    assert np.linalg.norm(a - b) < np.linalg.norm(a - c)


def test_label_columns_ignored():
    r = row()
    with_labels = r.assign(isFraud=1, isFlaggedFraud=1)
    pd.testing.assert_frame_equal(build_features(r), build_features(with_labels))
