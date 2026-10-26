"""End-to-end pipeline: split -> features -> fit -> calibrate -> evaluate -> export.

    python scripts/train_and_evaluate.py            # uses data/paysim.parquet if present,
                                                    # else the synthetic sample
    python scripts/train_and_evaluate.py --data data/paysim.parquet

Leakage protocol (what the report must state):
  1. Time split on `step`: train = first 70% of hours, holdout = last 30%.
  2. Models are fitted on TRAIN features only. `isFraud` is never an input.
  3. contamination='auto' - we do NOT feed the known fraud rate to the model.
  4. Alert thresholds come from TRAIN score quantiles (ALERT_RATE), not holdout.
  5. No hyperparameter is tuned on the holdout. Labels are used once, at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from momosentry import config, data, rules  # noqa: E402
from momosentry.evaluate import curves, evaluate_scores, overlap  # noqa: E402
from momosentry.features import build_features  # noqa: E402
from momosentry.models import (  # noqa: E402
    Bundle,
    Calibration,
    anomaly_score,
    make_isolation_forest,
    make_lof,
    typical_feature_vectors,
)

CALIBRATION_SAMPLE = 200_000


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def budget_sweep(y, score, cal, ref):
    """Precision and recall at each alert budget in config.ALERT_BUDGETS.

    The threshold for budget b is the (1 - b) quantile of the model's TRAIN
    scores, which `cal.sorted_scores` already holds. So, exactly like the
    headline threshold, no budget threshold ever looks at the holdout period.
    """
    rows = []
    for b in config.ALERT_BUDGETS:
        thr = float(np.quantile(cal.sorted_scores, 1 - b))
        m = evaluate_scores(y, score, thr, reference_rate=ref)
        rows.append({"budget": b, "threshold": thr, "alert_rate": m["alert_rate"],
                     "precision": m["precision_at_threshold"],
                     "precision_adj": m.get("precision_at_threshold_adj"),
                     "recall": m["recall_at_threshold"], "tp": m["tp"], "fp": m["fp"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_STATE)

    path = args.data or (config.PARQUET if config.PARQUET.exists() else config.SAMPLE_PARQUET)
    df = data.load(path)
    source = "synthetic_sample" if path.resolve() == config.SAMPLE_PARQUET.resolve() else "paysim"
    log(f"loaded {len(df):,} rows ({source})")  # no label read before the evaluation stage

    train, holdout, cut = data.time_split(df)
    del df  # the split made copies; free the full table so it is not held twice
    log(f"time split at step {cut}: train {len(train):,} rows, holdout {len(holdout):,} rows")

    X_train = build_features(train)
    X_hold = build_features(holdout)

    # ---------------- Isolation Forest ----------------
    iso = make_isolation_forest().fit(X_train)
    log("fitted Isolation Forest on all train rows")
    cal_idx = rng.choice(len(train), size=min(CALIBRATION_SAMPLE, len(train)), replace=False)
    cal_if = Calibration.from_train_scores(anomaly_score(iso, X_train.iloc[cal_idx]))

    # ---------------- LOF ----------------
    # Fit on a type-stratified train sample; calibrate on DIFFERENT train rows,
    # because novelty-LOF scores on its own fit data are biased (sklearn docs).
    lof_fit_rows = data.type_stratified_sample(train.assign(_i=np.arange(len(train))), config.LOF_FIT_SAMPLE)["_i"].to_numpy()
    lof = make_lof().fit(X_train.iloc[lof_fit_rows])
    log(f"fitted LOF on {len(lof_fit_rows):,} type-stratified train rows")
    remaining = np.setdiff1d(np.arange(len(train)), lof_fit_rows)
    lof_cal_rows = rng.choice(remaining, size=min(50_000, len(remaining)), replace=False)
    assert np.intersect1d(lof_fit_rows, lof_cal_rows).size == 0  # fit and threshold rows never overlap
    cal_lof = Calibration.from_train_scores(anomaly_score(lof, X_train.iloc[lof_cal_rows]))
    log("calibrated both models on train-period scores")

    bundle = Bundle(
        models={"if": iso, "lof": lof},
        calibration={"if": cal_if, "lof": cal_lof},
        typical_by_type=typical_feature_vectors(X_train, train["type"]),
        meta={"source": source, "split_step": cut, "alert_rate": config.ALERT_RATE,
              "lof_fit_rows": int(len(lof_fit_rows)), "train_rows": int(len(train))},
    )
    bundle.save(out / "model_bundle.joblib")

    # ---------------- Evaluation (labels used from here on) ----------------
    y_hold = holdout["isFraud"].to_numpy()
    if_hold = anomaly_score(iso, X_hold)
    log("scored full holdout with Isolation Forest")

    # Comparison set: a UNIFORM random sample of holdout so the fraud base rate is
    # unchanged (keeping 'all fraud + some normals' would inflate PR-AUC).
    n_cmp = len(holdout) if config.LOF_EVAL_MAX is None else min(config.LOF_EVAL_MAX, len(holdout))
    cmp_idx = np.sort(rng.choice(len(holdout), size=n_cmp, replace=False))
    lof_cmp = anomaly_score(lof, X_hold.iloc[cmp_idx])
    log(f"scored {n_cmp:,} holdout rows with LOF")
    # LOF alert flag for every holdout row. cmp_idx holds the positions LOF scored, so the
    # flags land on the right rows; rows LOF did not score stay False. The alert list is
    # only complete when LOF scored the whole holdout (LOF_EVAL_MAX >= holdout size).
    lof_alert_hold = np.zeros(len(holdout), dtype=bool)
    lof_alert_hold[cmp_idx] = lof_cmp > cal_lof.threshold
    lof_alerts_complete = bool(n_cmp == len(holdout))

    y_cmp = y_hold[cmp_idx]
    ref = float(train["isFraud"].mean())  # prevalence the detector faces in normal operation
    log(f"fraud rate: train {ref:.4%}, holdout {y_hold.mean():.4%}")
    flagged = rules.supplied_flag(holdout)  # baseline 1: PaySim's own indicator
    drain = rules.balance_drain(holdout)  # 1.0 where an outgoing tx empties the sender
    metrics = {
        "meta": bundle.meta | {"holdout_rows": int(len(holdout)), "comparison_rows": int(n_cmp),
                               "train_fraud_rate": ref, "holdout_fraud_rate": float(y_hold.mean())},
        "full_holdout": {
            "Isolation Forest": evaluate_scores(y_hold, if_hold, cal_if.threshold, reference_rate=ref),
            "isFlaggedFraud rule": evaluate_scores(y_hold, flagged, 0.5, reference_rate=ref),
            "Balance-drain rule": evaluate_scores(y_hold, drain, 0.5, reference_rate=ref),
        },
        "comparison_set": {
            "Isolation Forest": evaluate_scores(y_cmp, if_hold[cmp_idx], cal_if.threshold, reference_rate=ref),
            "Local Outlier Factor": evaluate_scores(y_cmp, lof_cmp, cal_lof.threshold, reference_rate=ref),
            "isFlaggedFraud rule": evaluate_scores(y_cmp, flagged[cmp_idx], 0.5, reference_rate=ref),
            "Balance-drain rule": evaluate_scores(y_cmp, drain[cmp_idx], 0.5, reference_rate=ref),
        },
        # RQ3: how precision and recall move as the analyst's budget grows.
        # Only the two detectors have a budget; a rule is fixed yes/no.
        "budgets": {
            "Isolation Forest": budget_sweep(y_cmp, if_hold[cmp_idx], cal_if, ref),
            "Local Outlier Factor": budget_sweep(y_cmp, lof_cmp, cal_lof, ref),
        },
        # RQ3: do the four methods flag the same transactions? Detectors use
        # their headline (ALERT_RATE) threshold, rules use their own yes/no.
        "overlap": overlap({
            "Isolation Forest": if_hold[cmp_idx] > cal_if.threshold,
            "Local Outlier Factor": lof_cmp > cal_lof.threshold,
            "isFlaggedFraud rule": flagged[cmp_idx] == 1,
            "Balance-drain rule": drain[cmp_idx] == 1,
        }, y_cmp),
    }
    # Run facts shown by the dashboard (Method panel, model cards). Writing them here
    # means the screen shows what the pipeline actually did, not what the report claims.
    train_mix = train["type"].value_counts(normalize=True)
    lof_mix = train["type"].iloc[lof_fit_rows].value_counts(normalize=True)
    metrics["run"] = {
        "train_steps": [int(train["step"].min()), int(train["step"].max())],
        "holdout_steps": [int(holdout["step"].min()), int(holdout["step"].max())],
        "feature_names": list(X_train.columns),
        "label_first_used": "evaluation stage",
        "alert_budgets": list(config.ALERT_BUDGETS),
        "isolation_forest": {
            "params": dict(config.IF_PARAMS) | {"random_state": config.RANDOM_STATE},
            "fit_rows": int(len(train)),
            # each tree sees max_samples rows, so the forest sees at most this many distinct rows
            "max_rows_seen": int(min(len(train), config.IF_PARAMS["n_estimators"] * config.IF_PARAMS["max_samples"])),
            "calibration_rows": int(len(cal_idx)),
            "threshold": float(cal_if.threshold),
        },
        "local_outlier_factor": {
            "params": dict(config.LOF_PARAMS),
            "fit_rows": int(len(lof_fit_rows)),
            "fit_sampled_by": "transaction type (never the fraud label)",
            "calibration_rows": int(len(lof_cal_rows)),
            "fit_calibration_overlap": int(np.intersect1d(lof_fit_rows, lof_cal_rows).size),
            "threshold": float(cal_lof.threshold),
            "scored_rows": int(n_cmp),
            "alerts_complete": lof_alerts_complete,
            "type_mix": {str(t): {"train_share": float(train_mix.get(t, 0.0)),
                                  "lof_sample_share": float(lof_mix.get(t, 0.0))} for t in train_mix.index},
        },
    }
    # Per-type breakdown for IF (where does it work, where does it fail?)
    metrics["if_by_type"] = {}
    for t in holdout["type"].astype(str).unique():
        m = (holdout["type"].astype(str) == t).to_numpy()
        if y_hold[m].sum() > 0:
            metrics["if_by_type"][t] = evaluate_scores(y_hold[m], if_hold[m], cal_if.threshold)
        else:
            metrics["if_by_type"][t] = {"n": int(m.sum()), "n_fraud": 0, "alert_rate": float((if_hold[m] > cal_if.threshold).mean())}

    curve_data = {
        "Isolation Forest": curves(y_cmp, if_hold[cmp_idx]),
        "Local Outlier Factor": curves(y_cmp, lof_cmp),
    }
    (out / "curves.json").write_text(json.dumps(curve_data))

    # ---------------- Dashboard export ----------------
    hourly = pd.DataFrame({"step": holdout["step"], "alert": if_hold > cal_if.threshold, "fraud": y_hold})
    hourly = hourly.groupby("step").agg(transactions=("alert", "size"), alerts=("alert", "sum"),
                                        fraud=("fraud", "sum")).reset_index()
    hourly.to_parquet(out / "hourly.parquet", index=False)

    # Small slice for the tables: every alert from BOTH detectors + every fraud + every
    # isFlaggedFraud row + random normal rows. Fraud rows are included on purpose so the
    # table can show misses, which means this file is NOT a representative sample: every
    # count on the dashboard comes from metrics.json, never from counting rows here.
    keep = (if_hold > cal_if.threshold) | lof_alert_hold | (y_hold == 1) | (flagged == 1)
    normal_pool = np.flatnonzero(~keep)
    extra = rng.choice(normal_pool, size=min(config.EXPORT_NORMAL_ROWS, len(normal_pool)), replace=False)
    rows = np.sort(np.concatenate([np.flatnonzero(keep), extra]))
    # data.load() no longer reads nameOrig; errors="ignore" keeps this safe either way.
    export = holdout.iloc[rows].drop(columns=["nameOrig"], errors="ignore").reset_index(drop=True)
    export["nameDest"] = export["nameDest"].astype(str).str[:1]  # keep only C/M prefix, drop ids
    export = pd.concat([export, bundle.score_raw(holdout.iloc[rows].reset_index(drop=True))], axis=1)
    export.to_parquet(out / "holdout_scored.parquet", index=False)
    log(f"exported {len(export):,} scored rows for the dashboard")
    metrics["run"]["export_rows"] = int(len(export))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for name, m in metrics["comparison_set"].items():
        log(f"{name:22s} ROC-AUC {m['roc_auc'] or 0:.3f}  PR-AUC {m['pr_auc'] or 0:.3f}  "
            f"PR-AUC(adj) {m.get('pr_auc_adj') or 0:.3f}  P@thr {m['precision_at_threshold']:.3f}  "
            f"P@thr(adj) {m.get('precision_at_threshold_adj', 0):.3f}  R@thr {m['recall_at_threshold']:.3f}")


if __name__ == "__main__":
    main()
