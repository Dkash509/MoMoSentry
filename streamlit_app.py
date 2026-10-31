"""MoMoSentry dashboard.   Run:  streamlit run streamlit_app.py

Everything here reads PRECOMPUTED artifacts from artifacts/ (produced by
scripts/train_and_evaluate.py), so the deployed app never loads the 6.3M-row
dataset. Only the manual scoring form runs the models live.
"""
import streamlit as st

from app_common import artifacts_ready

st.set_page_config(page_title="MoMoSentry", layout="wide")

# Pages appear in functional-requirement order, so the sidebar reads FR1 to FR8 top to bottom:
# Data and models (FR1, FR2) -> Score (FR3, FR4) -> Flagged (FR5) -> Comparison (FR6, FR7) -> Overview (FR8).
# The first page is the entry point the app opens on.
PAGES = [
    st.Page("views/data_models.py", title="Data and models", default=True),
    st.Page("views/score.py", title="Score a transaction"),
    st.Page("views/flagged.py", title="Flagged transactions"),
    st.Page("views/compare.py", title="Model comparison"),
    st.Page("views/overview.py", title="Overview"),
]

if artifacts_ready():
    st.navigation(PAGES).run()
