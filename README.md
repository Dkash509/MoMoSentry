# MoMoSentry

**Unsupervised anomaly detection for mobile money fraud risk monitoring.**
Isolation Forest vs Local Outlier Factor on the PaySim mobile money dataset, evaluated under a strict no-leakage protocol, with a Streamlit dashboard for monitoring, scoring and explaining alerts.

> Final Year Project, BSc Computer Science, ZCAS University (2026), by Dalitso Kashole.

![tests](https://github.com/Dkash509/MoMoSentry/actions/workflows/ci.yml/badge.svg)

---

## Why

Mobile money is widely used in Zambia (one 2026 study reports a national adoption rate of 57%, Chanda et al.) and is linked to better informal business performance (Hassan, 2023). Examples of fraud detection work on African mobile money data, using supervised classifiers on a one-month operator log or on simulated data, are Mubalaike & Adalı (2018) and Azamuke et al. (2024). Real operator data isn't public, so this project uses **PaySim**, a synthetic dataset modelled on an African mobile money service. It asks one question:

> **Can unsupervised anomaly detection, trained without any fraud labels, surface fraudulent mobile money transactions at a review rate an analyst can actually handle, and which of Isolation Forest or Local Outlier Factor does it better?**

It's unsupervised on purpose. Label scarcity and evolving fraud patterns are widely reported challenges in financial fraud detection (Thiong'o et al., 2026), and historical fraud data can be outdated for studying new patterns (Azamuke et al., 2024).

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
4. **Thresholds come from train scores only.** The alert threshold is the 99.5th percentile of *training-period* scores (an analyst budget of 0.5% of traffic), then applied unchanged to the holdout. The 0.5% figure is an assumed budget set in `momosentry/config.py`, not a sourced review capacity. On the holdout Isolation Forest exceeded it, alerting on 0.95% of traffic (2,659 alerts). The main cause is the higher fraud share: fraud rises from 0.095% to 0.867% of traffic, and 1,218 of the alerts (0.43% of all holdout traffic) are true fraud. The other 1,441 are false alarms, 0.51% of traffic, close to the 0.5% the threshold was set for.
5. **LOF scalability.** LOF is fitted on a 100k-row sample (of 6,082,007 training rows) of the training period, stratified by **transaction type** (never by label), and calibrated on *different* training rows because scikit-learn's novelty-mode LOF is not meant to score its own training data ([scikit-learn `LocalOutlierFactor` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)).
6. **Same evaluation set.** Every method is scored on the same full holdout (280,613 rows, nothing subsampled), so the fraud base rate is intact. Keeping "all fraud + some normals" would inflate PR-AUC. Training data is not equal across models, so a control was run (see Limitations).
7. **Baselines.** Two rule-based baselines: PaySim's built-in `isFlaggedFraud` rule, and a balance-drain rule (flags transactions that drain the sender's account).
8. **Nothing is tuned on the holdout.** All settings live in `momosentry/config.py`.

PR-AUC is the headline metric. With fraud at 0.095% of the training period and 0.867% of the holdout, ROC-AUC can look excellent while almost every alert is a false alarm.

## Quick start

```bash
git clone https://github.com/Dkash509/MoMoSentry.git && cd MoMoSentry
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Option A: synthetic sample (runs anywhere in about 1 minute)
python scripts/make_sample_data.py
python scripts/train_and_evaluate.py

# Option B: real PaySim (do this for the actual results)
#   1. Download ealaxi/paysim1 from Kaggle and save the CSV as data/paysim.csv (the filename the scripts expect)
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

`requirements.txt` is pinned on purpose: the model bundle was pickled under those library versions and may not load under others.

## Results (real PaySim, 6,362,620 transactions)

Train: steps 1–520 (6,082,007 rows, 0.095% fraud). Holdout: steps 521–743 (280,613 rows, 0.867% fraud).
All four methods are scored on the full holdout. The alert threshold is the top 0.5% of training-period scores.

| Method | ROC-AUC | PR-AUC | PR-AUC (adj.) | Precision @ alert | Precision @ alert (adj.) | Recall @ alert | Precision @ top 1000 |
|---|---|---|---|---|---|---|---|
| **Isolation Forest** | **0.941** | **0.538** | **0.486** | 0.458 | 0.084 | **0.501** | **0.994** |
| Local Outlier Factor | 0.654 | 0.081 | 0.010 | 0.196 | 0.026 | 0.208 | 0.142 |
| `isFlaggedFraud` rule | 0.502 | 0.013 | 0.005 | 1.000 | 1.000 | 0.004 | 0.024 |
| Balance-drain rule | 0.857 | 0.031 | 0.004 | 0.032 | 0.004 | 0.970 | 0.016 |

The balance-drain rule alerts on 26.2% of all holdout traffic; Isolation Forest alerts on 0.95%.

**(adj.)** In PaySim, normal traffic collapses after about step 400 while fraud continues at the same pace, so the holdout fraud rate is about 9× the training rate. That inflates raw precision and PR-AUC. The adjusted columns re-weight normal rows back to the training-period fraud rate (0.095%), and those are the figures to quote. Recall and ROC-AUC don't depend on prevalence.

**Findings**

1. **Isolation Forest clearly beats LOF**, with about 47× the adjusted PR-AUC (0.486 vs 0.010). Unequal training data does not explain the gap: IF fitted on the same 100k sample as LOF still reaches an adjusted PR-AUC of 0.47–0.51 (see Limitations). A possible cause is that LOF's local-density view struggles with PaySim's heavy-tailed amounts and mixed transaction types, but this was not tested.
2. **IF catches one leg of the fraud, not the other.** On holdout TRANSFERs it flags **100% of fraud at 98.9% precision**. On the matching CASH_OUTs it catches **0.16%**. Our interpretation is that the cash-out leg moves money out of an account whose balances are consistent, so on its own it looks normal. Catching it needs account-level context (e.g. "this account just received a drained-account transfer"), which a per-transaction detector cannot see. That is the main direction for future work.
3. **False alarms come from types that never contain fraud.** DEBIT (12.9% of them alerted) and CASH_IN (1.6%) cause most of them. A production system would route these through type-specific thresholds.
4. **PaySim's built-in rule is useless for detection.** It raised only 10 alerts on the holdout, all correct (precision 1.0), but that is just 0.4% of the fraud.
5. **The balance-drain rule shows the opposite trade-off.** It catches 97% of fraud (recall 0.970) but flags 26.2% of all traffic, with precision 0.032. Isolation Forest catches about half the fraud while alerting on 0.95% of traffic. That is above the assumed 0.5% budget (protocol point 4) but about 28 times fewer alerts than the balance-drain rule.

Reproduce with `python scripts/prepare_data.py --csv <your csv>` then `python scripts/train_and_evaluate.py --data data/paysim.parquet`.

## Limitations

- **PaySim is synthetic.** Results show what the method can do, not how it performs on Zamtel, MTN or Airtel data. The currency unit is unspecified, so amounts are not Kwacha.
- **Fraud is per-transaction here, but in reality it is per-account.** See finding 2.
- **The features encode ledger consistency.** PaySim fraud has a known signature (accounts drained to zero, receiver balances that don't update). The ledger-mismatch features are generic checks a real operator would run, but they make the task easier on PaySim than it would be on messier real data.
- **Unsupervised detectors flag "unusual", not "fraudulent".** Large legitimate transfers will also be flagged. The per-type breakdown on the Overview page shows where the alert budget is wasted.
- **LOF is fitted on a sample** (100k of 6,082,007 training rows) for computational reasons, while Isolation Forest sees all training rows. All methods are scored on the same full holdout, but the main-run models do not get equal training data. A control (`scripts/if_same_sample.py`) fits IF on the same 100k type-stratified rows as LOF, calibrates it on 50k different training rows and scores the full holdout. Over three random seeds the adjusted PR-AUC is 0.468, 0.512 and 0.483 (ROC-AUC 0.942–0.961), against 0.486 for IF on all rows and 0.010 for LOF. So the IF-over-LOF gap is not caused by the sample size. Only the 100k setting was tested, and LOF's `n_neighbors` was not tuned.

## Related work

- Lopez-Rojas, Elmir & Axelsson (2016), *PaySim: A financial mobile money simulator for fraud detection*
- Mubalaike, A. M. & Adalı, E. (2018). Deep learning approach for intelligent financial fraud detection system. *2018 3rd International Conference on Computer Science and Engineering (UBMK)*. https://doi.org/10.1109/ubmk.2018.8566574
- Azamuke, D., Katarahweire, M., et al. (2024). Refining detection mechanism of mobile money fraud using MoMTSim platform. https://doi.org/10.1007/978-3-031-57639-3_3
- Thiong'o, F., Nderu, L., et al. (2026). HADA: A hybrid anomaly detection approach using unsupervised machine learning. *Engineering Reports*. https://doi.org/10.1002/eng2.70696
- Ounacer, S., Ait El Bour, H., et al. (2018). Using isolation forest in anomaly detection: the case of credit card transactions. *Periodicals of Engineering and Natural Sciences*, 6(2). https://doi.org/10.21533/pen.v6i2.533 (compares LOF and Isolation Forest, among others)
- Hassan, R. (2023). Does mobile money adoption increase informal business performance in Zambia? *Journal of the Knowledge Economy*. https://doi.org/10.1007/s13132-023-01205-z
- Chanda, J., Mwange, A. & Matandiko, G. (2026). Beyond basic transactions: small-scale farmers' perspectives on mobile financial technology adoption in Zambia. *Journal of Arts, Humanities and Social Science*, 3(1). https://doi.org/10.69739/jahss.v3i1.1657

The PaySim dataset (Kaggle: ealaxi/paysim1) is by E. A. Lopez-Rojas and is licensed CC BY-SA 4.0. `artifacts/holdout_scored.parquet` contains a derived sample of it, with customer ids removed.

## License

MIT
