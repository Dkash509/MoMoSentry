"""Guards on the evaluation protocol."""
import inspect
import re

import numpy as np
from make_sample_data import generate

from momosentry import data, features, models, rules


def test_time_split_is_strictly_ordered():
    df = generate(20_000)
    train, hold, cut = data.time_split(df)
    assert train.step.max() <= cut < hold.step.min()
    assert len(train) + len(hold) == len(df)


def test_type_sample_never_uses_label():
    src = inspect.getsource(data.type_stratified_sample)
    assert "isFraud" not in src


def test_feature_and_model_code_never_reads_label():
    # Mentions in docstrings are fine; a quoted column name or attribute access is not.
    pattern = re.compile(r"""["']isFraud["']|\.isFraud\b|["']isFlaggedFraud["']""")
    for mod in (features, models, rules):
        assert not pattern.search(inspect.getsource(mod)), mod.__name__


def test_calibration_threshold_matches_alert_rate():
    rng = np.random.default_rng(0)
    cal = models.Calibration.from_train_scores(rng.normal(size=100_000), alert_rate=0.01)
    new = rng.normal(size=100_000)
    assert abs((new > cal.threshold).mean() - 0.01) < 0.002
    assert 0 <= cal.percentile(new).min() and cal.percentile(new).max() <= 100
