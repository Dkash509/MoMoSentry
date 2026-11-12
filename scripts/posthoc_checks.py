"""Post-hoc checks reported in Section 4.4.7 of the report.

    python scripts/posthoc_checks.py      # needs data/paysim.parquet and artifacts/

These checks were run after the main results, because the results raised
questions a short run could answer. They change no model, threshold or
dashboard file; they only write artifacts/posthoc_checks.json.

Fitting and thresholds still use no label. Labels are read only after the
evaluation marker, as in train_and_evaluate.py.
The receiver-balance pattern was found by looking at holdout labels, so every
check here is post-hoc and is kept out of the headline results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from momosentry import config, data  # noqa: E402
from momosentry.evaluate import evaluate_scores  # noqa: E402
from momosentry.features import FEATURE_NAMES, build_features  # noqa: E402
from momosentry.models import Bundle, Calibration, anomaly_score, make_isolation_forest, make_lof  # noqa: E402

RECEIVER_FEATURES = ("err_dest", "dest_zero_both")


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def summary(y, score, thr, r, types):
    m = evaluate_scores(y, score, thr, reference_rate=r)
    out = {k: m.get(k) for k in ("roc_auc", "pr_auc", "pr_auc_adj", "precision_at_threshold",
                                  "precision_at_threshold_adj", "recall_at_threshold", "tp", "fp")}
    out["alerts"] = m["tp"] + m["fp"]
    out["alert_rate"] = m["alert_rate"]
    alert = score > thr
    out["by_type"] = {t: {"alerts": int(alert[types == t].sum()),
                          "fraud": int(y[types == t].sum()),
                          "tp": int((alert & (y == 1))[types == t].sum())}
                      for t in config.TX_TYPES}
    return out


def main():
    train, hold, cut = data.time_split(data.load(config.PARQUET))
    log(f"time split at step {cut}: train {len(train):,}, holdout {len(hold):,}")
    X_tr = build_features(train)
    X_ho = build_features(hold)
    t_tr = train["type"].astype(str).to_numpy()
    t_ho = hold["type"].astype(str).to_numpy()

    # 1. Committed Isolation Forest, then one threshold per type: the 99.5th
    #    percentile of that type's scores on 500,000 random fitting rows.
    rng = np.random.default_rng(config.RANDOM_STATE)
    cal_idx = rng.choice(len(train), size=500_000, replace=False)
    bundle = Bundle.load()
    iso = bundle.models["if"]
    s_ho = anomaly_score(iso, X_ho)
    s_cal = anomaly_score(iso, X_tr.iloc[cal_idx])
    thr_t = {t: float(np.quantile(s_cal[t_tr[cal_idx] == t], 1 - config.ALERT_RATE)) for t in config.TX_TYPES}
    thr_row = np.array([thr_t[t] for t in t_ho])
    log("per-type thresholds set")

    # 2. Ablation: Isolation Forest refitted without the receiver-balance
    #    features, same settings, threshold by the same 99.5th-percentile rule.
    keep = [c for c in FEATURE_NAMES if c not in RECEIVER_FEATURES]
    iso_ab = make_isolation_forest().fit(X_tr[keep])
    cal_ab = Calibration.from_train_scores(anomaly_score(iso_ab, X_tr[keep].iloc[cal_idx[:200_000]]))
    s_ho_ab = anomaly_score(iso_ab, X_ho[keep])
    log("ablation fitted")

    # 3. Label-free rule written after the pattern was seen: a TRANSFER whose
    #    receiver shows a zero balance before and after.
    rule = ((t_ho == "TRANSFER") & (X_ho["dest_zero_both"].to_numpy() == 1)).astype(float)

    # 4. Local Outlier Factor fitted on 300,000 rows instead of 100,000, with
    #    the same type-stratified sampling and 50,000 separate calibration rows.
    rng_lof = np.random.default_rng(config.RANDOM_STATE)
    fit_rows = data.type_stratified_sample(train.assign(_i=np.arange(len(train))), 300_000)["_i"].to_numpy()
    lof = make_lof().fit(X_tr.iloc[fit_rows])
    remaining = np.setdiff1d(np.arange(len(train)), fit_rows)
    cal_rows = rng_lof.choice(remaining, size=50_000, replace=False)
    cal_lof = Calibration.from_train_scores(anomaly_score(lof, X_tr.iloc[cal_rows]))
    s_ho_lof = anomaly_score(lof, X_ho)
    log("LOF 300k fitted")

    # ---------------- Evaluation (labels used from here on) ----------------
    y_tr = train["isFraud"].to_numpy()
    y = hold["isFraud"].to_numpy()
    r = float(y_tr.mean())
    res = {"meta": {"split_step": int(cut), "reference_rate": r}}

    # Type mix of normal transactions in each period: the assumption behind
    # the single prevalence weight.
    mix_tr = pd.Series(t_tr[y_tr == 0]).value_counts(normalize=True)
    mix_ho = pd.Series(t_ho[y == 0]).value_counts(normalize=True)
    res["normal_type_mix"] = {t: {"fit": float(mix_tr.get(t, 0)), "holdout": float(mix_ho.get(t, 0))}
                              for t in config.TX_TYPES}

    res["if_global"] = summary(y, s_ho, bundle.calibration["if"].threshold, r, t_ho)
    res["if_per_type_thresholds"] = thr_t
    res["if_per_type"] = summary(y, s_ho - thr_row, 0.0, r, t_ho)
    res["if_ablated"] = summary(y, s_ho_ab, cal_ab.threshold, r, t_ho)
    res["if_ablated"]["threshold"] = cal_ab.threshold
    res["rule_transfer_dest_zero"] = summary(y, rule, 0.5, r, t_ho)
    res["lof_300k"] = summary(y, s_ho_lof, cal_lof.threshold, r, t_ho)
    res["lof_300k"]["threshold"] = cal_lof.threshold

    out = config.ARTIFACTS_DIR / "posthoc_checks.json"
    out.write_text(json.dumps(res, indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
