# MoMoSentry

**Unsupervised anomaly detection for mobile money fraud risk monitoring.**
Isolation Forest vs Local Outlier Factor on the PaySim mobile money dataset, evaluated under a strict no-leakage protocol, with a Streamlit dashboard for monitoring, scoring and explaining alerts.

> Final Year Project, BSc Computer Science, ZCAS University (2026), by Dalitso Kashole.

![tests](https://github.com/Dkash509/MoMoSentry/actions/workflows/ci.yml/badge.svg)

---

## Why

Mobile money is the main way many people in Zambia move money, and most published work on mobile money fraud detection comes from Ghana, Kenya and Tanzania. Real operator data isn't public, so this project uses **PaySim**, a synthetic dataset modelled on an African mobile money service. It asks one question:

> **Can unsupervised anomaly detection, trained without any fraud labels, surface fraudulent mobile money transactions at a review rate an analyst can actually handle, and which of Isolation Forest or Local Outlier Factor does it better?**

It's unsupervised on purpose. Operators rarely have clean, up-to-date fraud labels, and new fraud patterns don't look like old labelled ones.

## What's in it

| Part | What it does |
|---|---|
| `momosentry/features.py` | 15 label-free features per transaction: type, amount, time of day (circular), share of balance moved, account drained, sender/receiver ledger inconsistencies, merchant flag |
| `momosentry/models.py` | Isolation Forest and novelty-mode LOF pipelines, percentile risk calibration, model bundle |
| `momosentry/evaluate.py` | ROC-AUC, **PR-AUC**, precision/recall/F1 at the alert threshold, precision@top-k. This is the only module that reads the label |
| `momosentry/explain.py` | Perturbation explanations: "how much does the score drop if this part of the transaction were typical?" |
| `scripts/train_and_evaluate.py` | End-to-end pipeline that writes everything in `artifacts/` |
| `streamlit_app.py` + `views/` | Dashboard, in functional-requirement order: **Data and models** (FR1, FR2), **Score a transaction** (FR3, FR4), **Flagged transactions** (FR5), **Model comparison** (FR6, FR7), **Overview** (FR8) |
| `tests/` | Feature unit tests, leakage guards, an end-to-end pipeline test, and a render test for every dashboard page |

## Evaluation protocol (no label leakage)

1. **Time split** on `step` (hours): train on the first 70% of hours and test on the last 30%. A random split would leak future behaviour into training.
2. **No labels in training.** `isFraud` and `isFlaggedFraud` are never features, and a test enforces this.
3. **`contamination="auto"`.** The models aren't told the fraud rate.
4. **Thresholds come from train scores only.** The alert threshold is the 99.5th percentile of *training-period* scores (an analyst budget of 0.5% of traffic), then applied unchanged to the holdout.
5. **LOF scalability.** LOF is fitted on a 100k-row sample of the training period, stratified by **transaction type** (never by label), and calibrated on *different* training rows because novelty-LOF scores are biased on its own fit data.
6. **Fair comparison.** Every method is scored on the same uniform random holdout sample, which keeps the fraud base rate intact. Keeping "all fraud + some normals" would inflate PR-AUC.
7. **Baseline.** PaySim's built-in `isFlaggedFraud` rule.
8. **Nothing is tuned on the holdout.** All settings live in `momosentry/config.py`.

PR-AUC is the headline metric. With roughly 0.13% fraud, ROC-AUC can look excellent while almost every alert is a false alarm.

## Quick start

```bash
git clone https://github.com/Dkash509/MoMoSentry.git && cd MoMoSentry
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Option A: synthetic sample (runs anywhere in about 1 minute)
python scripts/make_sample_data.py
python scripts/train_and_evaluate.py

# Option B: real PaySim (do this for the actual results)
#   1. Download ealaxi/paysim1 from Kaggle and put the CSV in data/
#   2. Convert it and train:
python scripts/prepare_data.py
python scripts/train_and_evaluate.py --data data/paysim.parquet

streamlit run streamlit_app.py
python -m pytest -q
```

## Deployment (Streamlit Community Cloud)

The raw CSV (about 470 MB) is **never committed**. The app only reads `artifacts/`: the model bundle, metrics, curves, and a small precomputed slice of scored holdout rows. So:

1. Run the pipeline on the real data locally.
2. Commit `artifacts/`.
3. Deploy `streamlit_app.py` from the repo on share.streamlit.io.

## Results (real PaySim, 6,362,620 transactions)

Train: steps 1–520 (6,082,007 rows, 0.095% fraud). Holdout: steps 521–743 (280,613 rows, 0.867% fraud).
All three methods are scored on the full holdout. The alert threshold is the top 0.5% of training-period scores.

| Method | ROC-AUC | PR-AUC | PR-AUC (adj.) | Precision @ alert | Precision @ alert (adj.) | Recall @ alert | Precision @ top 1000 |
|---|---|---|---|---|---|---|---|
| **Isolation Forest** | **0.941** | **0.538** | **0.486** | 0.458 | 0.084 | **0.501** | **0.994** |
| Local Outlier Factor | 0.654 | 0.081 | 0.010 | 0.196 | 0.026 | 0.208 | 0.142 |
| `isFlaggedFraud` rule | 0.502 | 0.013 | 0.005 | 1.000 | 1.000 | 0.004 | 0.024 |

**(adj.)** In PaySim, normal traffic collapses after about step 400 while fraud continues at the same pace, so the holdout fraud rate is about 9× the training rate. That inflates raw precision and PR-AUC. The adjusted columns re-weight normal rows back to the training-period fraud rate (0.095%), and those are the figures to quote. Recall and ROC-AUC don't depend on prevalence.

**Findings**

1. **Isolation Forest clearly beats LOF**, with roughly 48× the adjusted PR-AUC. LOF's local-density view struggles with PaySim's heavy-tailed amounts and mixed transaction types, even with scaling.
2. **IF catches one leg of the fraud, not the other.** On holdout TRANSFERs it flags **100% of fraud at 98.9% precision**. On the matching CASH_OUTs it catches **0.16%**. The cash-out leg moves money out of a mule account with consistent balances, so on its own it looks normal. Catching it needs account-level context (e.g. "this account just received a drained-account transfer"), which a per-transaction detector cannot see. That is the main direction for future work.
3. **False alarms come from types that never contain fraud.** DEBIT (12.9% of them alerted) and CASH_IN (1.6%) cause most of them. A production system would route these through type-specific thresholds.
4. **PaySim's built-in rule is useless for detection.** It is perfectly precise but catches 0.4% of fraud.

Reproduce with `python scripts/prepare_data.py --csv <your csv>` then `python scripts/train_and_evaluate.py --data data/paysim.parquet` (about 90 s on 2 CPU cores).

## Limitations

- **PaySim is synthetic.** Results show what the method can do, not how it performs on Zamtel, MTN or Airtel data. The currency unit is unspecified, so amounts are not Kwacha.
- **Fraud is per-transaction here, but in reality it is per-account.** See finding 2.
- **The features encode ledger consistency.** PaySim fraud has a known signature (accounts drained to zero, receiver balances that don't update). The ledger-mismatch features are generic checks a real operator would run, but they make the task easier on PaySim than it would be on messier real data.
- **Unsupervised detectors flag "unusual", not "fraudulent".** Large legitimate transfers will also be flagged. The per-type breakdown on the Overview page shows where the alert budget is wasted.
- **LOF is evaluated on a sample** for computational reasons. This is documented and applied equally to all methods.

## Related work

- MoMoGuard: a mobile money analytics dashboard (Ghana context)
- "AI-Powered Fraud Detection System" (P. Nandre): RF + XGBoost + Isolation Forest on PaySim with Streamlit
- Lopez-Rojas, Elmir & Axelsson (2016), *PaySim: A financial mobile money simulator for fraud detection*

The PaySim dataset (Kaggle: ealaxi/paysim1) is by E. A. Lopez-Rojas and is licensed CC BY-SA 4.0. `artifacts/holdout_scored.parquet` contains a derived sample of it, with customer ids removed.

## License

MIT
