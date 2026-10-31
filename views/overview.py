import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    COLORS,
    MUTED,
    PLOTLY_CONFIG,
    load_json,
    load_parquet,
    pct,
    source_banner,
    style,
)


def page_overview():
    st.title("MoMoSentry: fraud risk monitoring")
    st.caption("Unsupervised anomaly detection on mobile money transactions (PaySim). "
               "Models learn 'normal' from the first 70% of hours and are tested on the last 30%.")
    m = load_json("metrics.json")
    source_banner(m)
    iso = m["full_holdout"]["Isolation Forest"]

    c = st.columns(5)
    c[0].metric("Holdout transactions", f"{iso['n']:,}")
    c[1].metric("Alerts raised (IF)", f"{iso['tp'] + iso['fp']:,}", help=f"{iso['alert_rate']:.2%} of holdout traffic")
    c[2].metric("Fraud in holdout", f"{iso['n_fraud']:,}", help=f"{iso['base_rate']:.3%} of holdout traffic")
    c[3].metric("Fraud caught (recall)", pct(iso["recall_at_threshold"]))
    c[4].metric("Precision", pct(iso["precision_at_threshold"]),
                help="Raw holdout precision. See Model comparison for the base-rate-adjusted figure.")

    hourly = load_parquet("hourly.parquet")
    hourly["hour_of_day"] = hourly["step"] % 24

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=hourly["step"], y=hourly["transactions"], marker_color=MUTED,
                               name="Transactions", hovertemplate="Step %{x}: %{y:,} tx<extra></extra>"))
        st.plotly_chart(style(fig, title="Transactions per hour (holdout)", xaxis_title="Step (hour)"), width="stretch", config=PLOTLY_CONFIG)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=hourly["step"], y=hourly["alerts"], name="IF alerts", mode="lines",
                        line=dict(color=COLORS["Isolation Forest"], width=2))
        fig.add_scatter(x=hourly["step"], y=hourly["fraud"], name="Actual fraud", mode="lines",
                        line=dict(color=COLORS["Actual fraud"], width=2, dash="dot"))
        st.plotly_chart(style(fig, title="Alerts vs actual fraud per hour", xaxis_title="Step (hour)"), width="stretch", config=PLOTLY_CONFIG)

    st.subheader("Isolation Forest performance by transaction type")
    # Types with no fraud have no recall, precision or PR-AUC (dividing by zero
    # fraud cases), so those cells read "n/a" instead of an empty "None".
    def fmt(x, spec):
        return "n/a" if x is None else format(x, spec)

    rows = []
    for t, v in m["if_by_type"].items():
        rows.append({"Type": t, "Transactions": f"{v['n']:,}", "Fraud": f"{v['n_fraud']:,}",
                     "Alert rate": fmt(v["alert_rate"], ".2%"),
                     "Recall": fmt(v.get("recall_at_threshold"), ".2%"),
                     "Precision": fmt(v.get("precision_at_threshold"), ".2%"),
                     "PR-AUC": fmt(v.get("pr_auc"), ".3f"), "_n": v["n"]})
    table = pd.DataFrame(rows).sort_values("_n", ascending=False).drop(columns="_n")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Alerts on types that contain no fraud add review workload without catching fraud. In PaySim each "
               "fraud is a TRANSFER followed by a CASH_OUT, so performance is shown for each leg separately.")


page_overview()
