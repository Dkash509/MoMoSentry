"""Evaluation. This is the ONLY module that touches the `isFraud` label."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_scores(y: np.ndarray, score: np.ndarray, threshold: float, top_k: int = 1000,
                    reference_rate: float | None = None) -> dict:
    """Threshold-free metrics (ROC-AUC, PR-AUC) plus metrics at the alert threshold.

    PR-AUC matters more than ROC-AUC here: with ~0.1% fraud, a model can have a
    high ROC-AUC while most of its alerts are still false alarms.

    `reference_rate`: precision and PR-AUC depend on how common fraud is. In real
    PaySim the late (holdout) hours contain far fewer normal transactions but about
    the same amount of fraud, so the holdout fraud rate is ~9x the training rate.
    If given, normal rows are re-weighted so fraud makes up `reference_rate` of
    the data, and prevalence-adjusted precision / PR-AUC are reported as well.
    Recall, ROC-AUC and FPR do not depend on prevalence, so they need no adjustment.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    alert = score > threshold
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    k = min(top_k, len(score))
    top_idx = np.argsort(-score, kind="stable")[:k]

    out = {
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None,
        "pr_auc": float(average_precision_score(y, score)) if y.sum() else None,
        "alert_rate": float(alert.mean()),
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f1_at_threshold": f1,
        "tp": tp, "fp": fp, "fn": fn,
        f"precision_at_top_{k}": float(y[top_idx].mean()),
        "fpr_at_threshold": fp / max(int((y == 0).sum()), 1),
    }
    if reference_rate is not None and 0 < y.sum() < len(y):
        w_normal = y.sum() * (1 - reference_rate) / (reference_rate * (y == 0).sum())
        weights = np.where(y == 1, 1.0, w_normal)
        out["reference_rate"] = float(reference_rate)
        out["pr_auc_adj"] = float(average_precision_score(y, score, sample_weight=weights))
        out["precision_at_threshold_adj"] = tp / (tp + w_normal * fp) if tp + fp else 0.0
    return out


def overlap(flags: dict, y: np.ndarray) -> dict:
    """Do the methods flag the SAME transactions, and do they catch the SAME fraud?

    `flags` maps a method name to a boolean array (True = this method raised an
    alert on that row). All arrays cover the same rows in the same order.

    For every pair of methods A and B this reports:
      both        rows flagged by A and B
      jaccard     both / rows flagged by A or B  (1 = identical, 0 = no overlap)
      fraud_both  fraud caught by both
      fraud_only_a / fraud_only_b   fraud caught by one but missed by the other
    The fraud_only numbers are the interesting ones: if they are large, the
    methods are complementary and could be combined.
    """
    y = np.asarray(y).astype(bool)
    names = list(flags)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = np.asarray(flags[a], bool), np.asarray(flags[b], bool)
            union = int((fa | fb).sum())
            both = int((fa & fb).sum())
            pairs.append({
                "a": a, "b": b,
                "flagged_a": int(fa.sum()), "flagged_b": int(fb.sum()),
                "both": both,
                "jaccard": both / union if union else 0.0,
                "fraud_both": int((fa & fb & y).sum()),
                "fraud_only_a": int((fa & ~fb & y).sum()),
                "fraud_only_b": int((fb & ~fa & y).sum()),
            })
    any_flag = np.zeros(len(y), bool)
    for f in flags.values():
        any_flag |= np.asarray(f, bool)
    return {
        "pairs": pairs,
        "n_fraud": int(y.sum()),
        "fraud_caught_by_any": int((any_flag & y).sum()),
        "fraud_caught_by_none": int((~any_flag & y).sum()),
    }


def curves(y: np.ndarray, score: np.ndarray, max_points: int = 300) -> dict:
    """Down-sampled PR and ROC curves for plotting in the dashboard."""
    p, r, _ = precision_recall_curve(y, score)
    fpr, tpr, _ = roc_curve(y, score)

    def thin(*arrs):
        idx = np.unique(np.linspace(0, len(arrs[0]) - 1, min(max_points, len(arrs[0]))).astype(int))
        return [a[idx].round(5).tolist() for a in arrs]

    p, r = thin(p, r)
    fpr, tpr = thin(fpr, tpr)
    return {"precision": p, "recall": r, "fpr": fpr, "tpr": tpr}
