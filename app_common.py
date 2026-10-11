"""Shared loaders, colors and chart styling for the dashboard pages."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from momosentry.config import ARTIFACTS_DIR
from momosentry.models import Bundle

# One color per model, fixed everywhere (validated categorical slots 1 and 2).
COLORS = {"Isolation Forest": "#2a78d6", "Local Outlier Factor": "#eb6834",
          "isFlaggedFraud rule": "#898781", "Balance-drain rule": "#5e5c57", "Actual fraud": "#898781"}
MUTED = "#b7d3f6"


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_bundle() -> Bundle:
    return Bundle.load(ARTIFACTS_DIR / "model_bundle.joblib")


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ARTIFACTS_DIR / name).read_text())


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / name)


def artifacts_ready() -> bool:
    need = ["model_bundle.joblib", "metrics.json", "curves.json", "holdout_scored.parquet", "hourly.parquet"]
    missing = [n for n in need if not (ARTIFACTS_DIR / n).exists()]
    if missing:
        st.error("Missing artifacts: " + ", ".join(missing))
        st.code("python scripts/make_sample_data.py\npython scripts/train_and_evaluate.py", language="bash")
        return False
    return True


def source_banner(metrics: dict):
    if metrics["meta"]["source"] != "paysim":
        st.warning("Running on the **synthetic sample**, not real PaySim. Numbers here demonstrate the "
                   "pipeline only and must not be reported as results.")


def style(fig: go.Figure, height=340, **kw) -> go.Figure:
    kw.setdefault("hovermode", "x unified")
    title = kw.pop("title", None)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=80, b=10),
                      title=dict(text=title, y=0.97, yanchor="top", x=0, xanchor="left"),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.2)", zeroline=False)
    return fig


def pct(x):
    return "n/a" if x is None else f"{x:.1%}"




PLOTLY_CONFIG = {"displayModeBar": False}
