import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    COLORS,
    PLOTLY_CONFIG,
    load_json,
    source_banner,
    style,
)


RERUN_HINT = "Re-run `python scripts/train_and_evaluate.py` to produce this analysis."


def budget_section(m: dict):
    """RQ3, part 1: precision and recall at several alert budgets."""
    st.subheader("Precision and recall as the alert budget grows")
    budgets = m.get("budgets")
    if budgets is None:  # metrics.json from before this analysis existed
        st.info(RERUN_HINT)
        return
    rows = [{"Method": name, "Budget": f"{r['budget']:.2%}", "Realised alert rate": r["alert_rate"],
             "Precision (adj.)": r["precision_adj"], "Recall": r["recall"]}
            for name, sweep in budgets.items() for r in sweep]
    df = pd.DataFrame(rows)
    left, right = st.columns(2)
    for col, metric in zip((left, right), ("Recall", "Precision (adj.)")):
        fig = go.Figure()
        for name, part in df.groupby("Method", sort=False):
            fig.add_scatter(x=part["Budget"], y=part[metric], name=name, mode="lines+markers",
                            line=dict(color=COLORS[name], width=2))
        col.plotly_chart(style(fig, title=f"{metric} by budget", xaxis_title="Share of traffic reviewed",
                               yaxis_title=metric), width="stretch", config=PLOTLY_CONFIG)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("Realised alert rate", "Precision (adj.)", "Recall")})
    st.caption("Each budget's threshold is taken from training-period scores, then applied unchanged to "
               "the holdout. The realised alert rate is the share of holdout traffic that actually crossed it, "
               "so it can differ from the budget. A larger budget can only raise recall, so the comparison of "
               "interest is how quickly precision falls.")


def overlap_section(m: dict):
    """RQ3, part 2: do the methods flag the same transactions and catch the same fraud?"""
    st.subheader("Overlap between the methods' alerts")
    ov = m.get("overlap")
    if ov is None:
        st.info(RERUN_HINT)
        return
    table = pd.DataFrame([{"Method A": p["a"], "Method B": p["b"], "Flagged by A": p["flagged_a"],
                           "Flagged by B": p["flagged_b"], "Flagged by both": p["both"],
                           "Overlap (Jaccard)": p["jaccard"], "Fraud caught by both": p["fraud_both"],
                           "Fraud only A caught": p["fraud_only_a"], "Fraud only B caught": p["fraud_only_b"]}
                          for p in ov["pairs"]])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Overlap (Jaccard)": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"Of {ov['n_fraud']:,} fraudulent holdout transactions, {ov['fraud_caught_by_any']:,} were "
               f"flagged by at least one method and {ov['fraud_caught_by_none']:,} by none. Jaccard = rows "
               "flagged by both ÷ rows flagged by either (1 = identical alerts, 0 = no overlap).")


def model_cards(m: dict):
    """How each detector was fitted and calibrated (from metrics.json)."""
    st.subheader("How each detector was fitted")
    run = m.get("run")
    if run is None:
        st.info(RERUN_HINT)
        return
    cmp = m["comparison_set"]
    iso, lof = run["isolation_forest"], run["local_outlier_factor"]
    p = m["meta"]["alert_rate"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Isolation Forest**")
        st.write(f"Fitted without labels on {iso['fit_rows']:,} fitting rows. "
                 f"{iso['params']['n_estimators']} trees, {iso['params']['max_samples']} rows per tree, "
                 f"so at most {iso['max_rows_seen']:,} distinct rows are seen.")
        st.write(f"Alert threshold {iso['threshold']:.4f}: the {100 * (1 - p):.1f}th percentile of "
                 f"{iso['calibration_rows']:,} fitting scores.")
        n = cmp["Isolation Forest"]
        st.metric("Alerts on the holdout", f"{n['tp'] + n['fp']:,}", help=f"{n['alert_rate']:.2%} of holdout traffic")
    with right:
        st.markdown("**Local Outlier Factor**")
        st.write(f"Fitted without labels on a sample of {lof['fit_rows']:,} fitting rows, drawn in proportion to "
                 f"transaction type. {lof['params']['n_neighbors']} neighbours, novelty mode.")
        st.write(f"Alert threshold {lof['threshold']:.4f}: the {100 * (1 - p):.1f}th percentile of "
                 f"{lof['calibration_rows']:,} other fitting rows (no overlap with the fitting sample).")
        n = cmp["Local Outlier Factor"]
        st.metric("Alerts on the holdout", f"{n['tp'] + n['fp']:,}", help=f"{n['alert_rate']:.2%} of holdout traffic")
    st.warning(f"Limitation: Local Outlier Factor is fitted on {lof['fit_rows']:,} of the {iso['fit_rows']:,} "
               "fitting rows because its nearest-neighbour search is too costly for the full set. "
               "Isolation Forest is fitted on the full set, so the two are not fitted on the same amount of data.")
    mix = pd.DataFrame([{"Type": t, "Fitting share": v["train_share"], "LOF sample share": v["lof_sample_share"]}
                        for t, v in lof["type_mix"].items()])
    st.dataframe(mix, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("Fitting share", "LOF sample share")})


def page_compare():
    st.title("Model comparison")
    m = load_json("metrics.json")
    source_banner(m)
    st.caption(f"All four methods are scored on the same {m['meta']['comparison_rows']:,} holdout rows "
               "(a uniform random sample, so the fraud base rate is unchanged).")
    model_cards(m)
    cmp = m["comparison_set"]
    k = next(key for key in next(iter(cmp.values())) if key.startswith("precision_at_top_"))
    table = pd.DataFrame([{"Method": name, "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"],
                           "PR-AUC (adj.)": v.get("pr_auc_adj"),
                           "Precision @ alert": v["precision_at_threshold"],
                           "Precision @ alert (adj.)": v.get("precision_at_threshold_adj"),
                           "Recall @ alert": v["recall_at_threshold"],
                           f"Precision @ top {k.split('_')[-1]}": v[k],
                           "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]} for name, v in cmp.items()])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns[1:8]})
    meta = m["meta"]
    if "train_fraud_rate" in meta:
        st.info(f"**Base-rate shift.** Fraud is {meta['holdout_fraud_rate']:.2%} of the holdout period but only "
                f"{meta['train_fraud_rate']:.3%} of the training period: PaySim's normal traffic collapses in the late "
                "hours while fraud keeps the same pace. Raw precision and PR-AUC are therefore flattering. "
                "The **(adj.)** columns re-weight normal transactions so fraud is back at the training-period rate, "
                "which is the realistic figure to quote. Recall and ROC-AUC are unaffected.")
    base = next(iter(cmp.values()))["base_rate"]

    curves = load_json("curves.json")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, cv in curves.items():
            fig.add_scatter(x=cv["recall"], y=cv["precision"], name=name, mode="lines",
                            line=dict(color=COLORS[name], width=2))
        fig.add_hline(y=base, line_dash="dot", line_color=COLORS["isFlaggedFraud rule"],
                      annotation_text=f"random = {base:.3%}", annotation_position="top right")
        st.plotly_chart(style(fig, title="Precision-recall curve", xaxis_title="Recall",
                              yaxis_title="Precision", hovermode="closest"), width="stretch", config=PLOTLY_CONFIG)
    with right:
        fig = go.Figure()
        for name, cv in curves.items():
            fig.add_scatter(x=cv["fpr"], y=cv["tpr"], name=name, mode="lines", line=dict(color=COLORS[name], width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines",
                        line=dict(color=COLORS["isFlaggedFraud rule"], width=1, dash="dot"))
        st.plotly_chart(style(fig, title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate",
                              hovermode="closest"), width="stretch", config=PLOTLY_CONFIG)

    budget_section(m)
    overlap_section(m)

    with st.expander("How to read these results"):
        st.markdown(
            f"- Fraud is only **{base:.3%}** of transactions. A model can rank fraud above most normal rows "
            "(high ROC-AUC) and still bury every real fraud under thousands of false alarms (low PR-AUC).\n"
            "- **@ alert** metrics use a threshold set on training-period scores so that ~"
            f"{m['meta']['alert_rate']:.1%} of traffic is flagged, which represents the analyst's review budget.\n"
            "- `isFlaggedFraud` is PaySim's built-in rule (large transfers). It is precise but catches almost nothing.\n"
            "- The balance-drain rule flags any outgoing transaction that leaves the sender at exactly 0.\n"
            "- No model saw `isFraud` during training, and no setting was tuned on the holdout.")


page_compare()
