"""MoMoSentry dashboard.   Run:  streamlit run streamlit_app.py

Everything here reads PRECOMPUTED artifacts from artifacts/ (produced by
scripts/train_and_evaluate.py), so the deployed app never loads the 6.3M-row
dataset. Only the manual scoring form runs the models live.
"""
import streamlit as st

from app_common import artifacts_ready

st.set_page_config(page_title="MoMoSentry", layout="wide")

if artifacts_ready():
    st.navigation([
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/score.py", title="Score a transaction"),
        st.Page("views/flagged.py", title="Flagged transactions"),
        st.Page("views/compare.py", title="Model comparison"),
    ]).run()
