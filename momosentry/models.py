"""The two unsupervised detectors, score calibration, and saving/loading.

Neither model ever sees `isFraud`. They learn what 'normal' looks like in the
training period and score how far a transaction is from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_NAMES, build_features

MODEL_NAMES = {"if": "Isolation Forest", "lof": "Local Outlier Factor"}


def make_isolation_forest() -> Pipeline:
    # Scaling does not change IF results; kept so both models share one input contract.
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", IsolationForest(contamination="auto", random_state=config.RANDOM_STATE, **config.IF_PARAMS)),
    ])


def make_lof() -> Pipeline:
    # LOF is distance-based, so scaling matters a lot here.
    return Pipeline([("scale", StandardScaler()), ("model", LocalOutlierFactor(**config.LOF_PARAMS))])


def anomaly_score(model: Pipeline, X: pd.DataFrame, chunk: int = 200_000) -> np.ndarray:
    """Higher = more anomalous. sklearn's score_samples is the opposite, so we negate."""
    parts = [-model.score_samples(X.iloc[i:i + chunk]) for i in range(0, len(X), chunk)]
    return np.concatenate(parts) if parts else np.array([])


@dataclass
class Calibration:
    """Maps raw scores to a 0-100 'risk percentile' using TRAIN-period scores only."""
    sorted_scores: np.ndarray
    threshold: float  # score above which we raise an alert

    @classmethod
    def from_train_scores(cls, scores: np.ndarray, alert_rate: float = config.ALERT_RATE):
        s = np.sort(scores)
        return cls(sorted_scores=s, threshold=float(np.quantile(s, 1 - alert_rate)))

    def percentile(self, scores: np.ndarray) -> np.ndarray:
        return 100.0 * np.searchsorted(self.sorted_scores, scores, side="right") / len(self.sorted_scores)


@dataclass
class Bundle:
    models: dict          # {"if": Pipeline, "lof": Pipeline}
    calibration: dict     # {"if": Calibration, "lof": Calibration}
    typical_by_type: pd.DataFrame  # train-period median feature vector per tx type (for explanations)
    meta: dict = field(default_factory=dict)

    def score_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score raw PaySim-shaped rows with both models."""
        X = build_features(raw)
        out = pd.DataFrame(index=raw.index)
        for key, model in self.models.items():
            s = anomaly_score(model, X)
            cal = self.calibration[key]
            out[f"{key}_score"] = s
            out[f"{key}_pct"] = cal.percentile(s)
            out[f"{key}_alert"] = s > cal.threshold
        return out

    def save(self, path=config.ARTIFACTS_DIR / "model_bundle.joblib"):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=config.ARTIFACTS_DIR / "model_bundle.joblib") -> "Bundle":
        return joblib.load(path)


def typical_feature_vectors(X_train: pd.DataFrame, types: pd.Series) -> pd.DataFrame:
    """The median training-period feature vector for each transaction type.

    The explainer (explain.py) uses these as the "typical" value a feature is
    reset to. Computed one type at a time so that only that type's rows are
    copied, instead of copying the whole 6-million-row table. The median is
    taken in float64 and stored as float32, which reproduces the values of the
    earlier pandas groupby-median exactly.
    """
    values = X_train[FEATURE_NAMES].to_numpy()
    labels = types.astype(str).to_numpy()
    names = sorted(set(labels))
    medians = [np.median(values[labels == t].astype(np.float64), axis=0).astype(np.float32) for t in names]
    return pd.DataFrame(np.array(medians), index=pd.Index(names, name="_type"), columns=FEATURE_NAMES)
