"""Guards on the evaluation protocol."""
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
from make_sample_data import generate

from momosentry import data, features, models, rules

# A quoted column name or attribute access is a read; a mention in a docstring is not.
LABEL_READ = re.compile(r"""["']isFraud["']|\.isFraud\b""")
FLAG_READ = re.compile(r"""["']isFlaggedFraud["']|\.isFlaggedFraud\b""")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_and_evaluate.py"
EVAL_MARKER = "# ---------------- Evaluation (labels used from here on)"


def test_time_split_is_strictly_ordered():
    df = generate(20_000)
    train, hold, cut = data.time_split(df)
    assert train.step.max() <= cut < hold.step.min()
    assert len(train) + len(hold) == len(df)


def test_type_sample_never_uses_label():
    src = inspect.getsource(data.type_stratified_sample)
    assert "isFraud" not in src


def test_feature_and_model_code_never_reads_either_label_column():
    for mod in (features, models):
        src = inspect.getsource(mod)
        assert not LABEL_READ.search(src), mod.__name__
        assert not FLAG_READ.search(src), mod.__name__


def test_rules_module_never_reads_the_fraud_label():
    assert not LABEL_READ.search(inspect.getsource(rules))


def test_only_the_supplied_flag_rule_reads_isflaggedfraud():
    # isFlaggedFraud IS baseline 1, so reading it in rules.py is intended, but only
    # inside supplied_flag. The balance-drain rule uses the transaction's own balances.
    assert FLAG_READ.search(inspect.getsource(rules.supplied_flag))
    assert not FLAG_READ.search(inspect.getsource(rules.balance_drain))
    assert not LABEL_READ.search(inspect.getsource(rules.balance_drain))


def test_training_script_reads_no_label_before_the_evaluation_stage():
    src = SCRIPT.read_text(encoding="utf-8")
    assert EVAL_MARKER in src
    fitting_stage = src.split("def main():")[1].split(EVAL_MARKER)[0]
    assert "isFraud" not in fitting_stage and "isFlaggedFraud" not in fitting_stage


def test_rules_ignore_the_label_column():
    # Behavioural check: scramble the label and the rule outputs must not move.
    df = generate(5_000)
    scrambled = df.assign(isFraud=1 - df["isFraud"])
    np.testing.assert_array_equal(rules.balance_drain(df), rules.balance_drain(scrambled))
    np.testing.assert_array_equal(rules.supplied_flag(df), rules.supplied_flag(scrambled))


def test_model_scores_do_not_change_when_labels_are_scrambled():
    # Behavioural check that catches indirect use, which the text checks cannot:
    # flipping both label columns must leave every score, risk and alert unchanged.
    df = generate(6_000)
    X = features.build_features(df)
    iso = models.make_isolation_forest().fit(X)
    cal = models.Calibration.from_train_scores(models.anomaly_score(iso, X))
    bundle = models.Bundle(models={"if": iso}, calibration={"if": cal}, typical_by_type=None)
    scrambled = df.assign(isFraud=1 - df["isFraud"], isFlaggedFraud=1 - df["isFlaggedFraud"])
    pd.testing.assert_frame_equal(bundle.score_raw(df), bundle.score_raw(scrambled))


def test_calibration_threshold_matches_alert_rate():
    rng = np.random.default_rng(0)
    cal = models.Calibration.from_train_scores(rng.normal(size=100_000), alert_rate=0.01)
    new = rng.normal(size=100_000)
    assert abs((new > cal.threshold).mean() - 0.01) < 0.002
    assert 0 <= cal.percentile(new).min() and cal.percentile(new).max() <= 100
