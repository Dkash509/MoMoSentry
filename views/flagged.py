import streamlit as st

from app_common import load_bundle, load_parquet, pct
from momosentry.config import TX_TYPES
from momosentry.explain import explain


def page_flagged():
    st.title("Flagged transactions")
    st.caption("Holdout-period transactions that crossed an alert threshold. The isFraud column is the "
               "ground truth, shown for evaluation only; an analyst would not have it in production.")
    df = load_parquet("holdout_scored.parquet")
    bundle = load_bundle()

    f1, f2, f3 = st.columns(3)
    model = f1.selectbox("Alerts from", ["Either model", "Isolation Forest", "Local Outlier Factor", "Both models"])
    types = f2.multiselect("Type", TX_TYPES, default=TX_TYPES)
    min_pct = f3.slider("Minimum IF risk", 0.0, 100.0, 0.0, 0.5)

    mask = {"Either model": df.if_alert | df.lof_alert, "Isolation Forest": df.if_alert,
            "Local Outlier Factor": df.lof_alert, "Both models": df.if_alert & df.lof_alert}[model]
    view = df[mask & df["type"].astype(str).isin(types) & (df.if_pct >= min_pct)].sort_values("if_score", ascending=False)

    c = st.columns(3)
    c[0].metric("Flagged", f"{len(view):,}")
    c[1].metric("Actually fraud", f"{int(view.isFraud.sum()):,}")
    c[2].metric("Hit rate", pct(view.isFraud.mean() if len(view) else None))

    show = view[["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                 "newbalanceDest", "if_pct", "lof_pct", "isFlaggedFraud", "isFraud"]]
    st.dataframe(show, width="stretch", height=420,
                 column_config={"amount": st.column_config.NumberColumn(format="%.2f"),
                                "if_pct": st.column_config.ProgressColumn("IF risk", min_value=0, max_value=100, format="%.1f"),
                                "lof_pct": st.column_config.ProgressColumn("LOF risk", min_value=0, max_value=100, format="%.1f"),
                                "isFraud": st.column_config.CheckboxColumn("isFraud (eval only)")})
    st.download_button("Download CSV", show.to_csv(index=False), "momosentry_flagged.csv", "text/csv")

    if len(view):
        st.subheader("Explanation for a selected transaction")
        idx = st.selectbox("Row", view.index[:200], format_func=lambda i: f"#{i} | {view.at[i, 'type']} | "
                           f"{view.at[i, 'amount']:,.0f} | IF {view.at[i, 'if_pct']:.1f}")
        ex = explain(bundle, df.loc[[idx]], "if")
        st.dataframe(ex.assign(share=ex["share"]), hide_index=True, width="stretch",
                     column_config={"share": st.column_config.ProgressColumn("Share of score drop", min_value=0, max_value=1, format="percent"),
                                    "score_drop": st.column_config.NumberColumn("Score drop", format="%.3f")})


page_flagged()
