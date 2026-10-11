import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    COLORS,
    PLOTLY_CONFIG,
    load_bundle,
    style,
)
from momosentry.config import TX_TYPES
from momosentry.explain import explain
from momosentry.models import MODEL_NAMES

PRESETS = {
    "Ordinary payment": dict(type="PAYMENT", amount=8_500.0, hour=13, old_o=42_000.0, new_o=33_500.0,
                             merchant=True, old_d=0.0, new_d=0.0),
    "Ordinary cash-out": dict(type="CASH_OUT", amount=150_000.0, hour=11, old_o=300_000.0, new_o=150_000.0,
                              merchant=False, old_d=500_000.0, new_d=650_000.0),
    "Account drained at 3am": dict(type="TRANSFER", amount=1_250_000.0, hour=3, old_o=1_250_000.0, new_o=0.0,
                                   merchant=False, old_d=0.0, new_d=0.0),
}


def page_score():
    st.title("Score a transaction")
    st.caption("Enter one transaction. Both models score it live, and the explanation shows which part "
               "of it looks unusual compared with typical training-period transactions of the same type.")
    bundle = load_bundle()
    preset = PRESETS[st.radio("Start from", list(PRESETS), horizontal=True)]

    with st.form("score"):
        a, b, c = st.columns(3)
        tx_type = a.selectbox("Type", TX_TYPES, index=TX_TYPES.index(preset["type"]))
        amount = a.number_input("Amount", min_value=0.0, value=preset["amount"], step=1000.0)
        hour = a.slider("Hour of day", 0, 23, preset["hour"])
        old_o = b.number_input("Sender balance before", min_value=0.0, value=preset["old_o"], step=1000.0)
        new_o = b.number_input("Sender balance after", min_value=0.0, value=preset["new_o"], step=1000.0)
        merchant = b.checkbox("Receiver is a merchant", value=preset["merchant"])
        old_d = c.number_input("Receiver balance before", min_value=0.0, value=preset["old_d"], step=1000.0)
        new_d = c.number_input("Receiver balance after", min_value=0.0, value=preset["new_d"], step=1000.0)
        submitted = st.form_submit_button("Score", type="primary")

    if submitted:
        st.session_state["scored_tx"] = pd.DataFrame([{"step": hour, "type": tx_type, "amount": amount, "oldbalanceOrg": old_o,
                         "newbalanceOrig": new_o, "nameDest": "M" if merchant else "C",
                         "oldbalanceDest": old_d, "newbalanceDest": new_d}])
    raw = st.session_state.get("scored_tx")
    if raw is None:
        return
    s = bundle.score_raw(raw).iloc[0]

    cols = st.columns(2)
    for col, key in zip(cols, ["if", "lof"]):
        alert = bool(s[f"{key}_alert"])
        col.metric(MODEL_NAMES[key], f"{s[f'{key}_pct']:.1f} / 100")
        if alert:
            col.error("Alert raised: score is above this model's threshold")
        else:
            col.success("No alert")
    st.caption(f"Risk = percentile of the score among training-period transactions. An alert fires above the "
               f"{100 * (1 - bundle.meta['alert_rate']):.1f}th percentile.")

    key = st.radio("Explain using", ["if", "lof"], format_func=MODEL_NAMES.get, horizontal=True)
    ex = explain(bundle, raw, key)
    fig = go.Figure(go.Bar(x=ex["score_drop"], y=ex["reason"], orientation="h", marker_color=COLORS[MODEL_NAMES[key]],
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, height=300, title="Score drop when each feature group is set to its typical value",
                          xaxis_title="Score drop", hovermode="closest"),
                    width="stretch", config=PLOTLY_CONFIG)


page_score()
