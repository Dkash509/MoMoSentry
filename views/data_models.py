"""Data and models page (FR1 and FR2): how PaySim was split in time, which features
the detectors see, and how each detector was fitted. Every number is read from
the artifacts written by the pipeline; nothing here touches the full dataset."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import PLOTLY_CONFIG, load_json, load_parquet, source_banner, style
from momosentry.config import ARTIFACTS_DIR
from momosentry.features import EXPLAIN_GROUPS

# Categorical slots 3 and 4 of the validated palette. Slots 1 and 2 already mean
# Isolation Forest and LOF on every page, so the two periods get their own hues.
PERIOD_COLORS = {"Fitting": "#1baf7a", "Holdout": "#eda100"}
CHOICES = {"Both periods": None, "Fitting period": "Fitting", "Holdout period": "Holdout"}


def period_figures(m: dict, period: str | None) -> list[tuple[str, str, str]]:
    """The four (label, value, help) metric tiles for the selected period.

    "Both periods" shows the original four facts FR1 lists, so the default view
    (and its screenshot) is unchanged; the other two narrow the same facts to one side.
    """
    meta, run = m["meta"], m["run"]
    total = meta["train_rows"] + meta["holdout_rows"]
    if period is None:
        a, b = run["train_steps"], run["holdout_steps"]
        return [("Fitting rows", f"{meta['train_rows']:,}", f"Steps {a[0]} to {a[1]}"),
                ("Holdout rows", f"{meta['holdout_rows']:,}", f"Steps {b[0]} to {b[1]}"),
                ("Features", str(len(run["feature_names"])), "Per transaction; listed in the table below"),
                ("Fraud rate: fitting / holdout",
                 f"{meta['train_fraud_rate']:.3%} / {meta['holdout_fraud_rate']:.3%}", None)]
    rows, steps, rate = {
        "Fitting": (meta["train_rows"], run["train_steps"], meta["train_fraud_rate"]),
        "Holdout": (meta["holdout_rows"], run["holdout_steps"], meta["holdout_fraud_rate"]),
    }[period]
    return [("Rows", f"{rows:,}", None),
            ("Steps (hours)", f"{steps[0]} to {steps[1]}", f"{steps[1] - steps[0] + 1} hours"),
            ("Share of all rows", f"{rows / total:.1%}", f"of {total:,} transactions"),
            ("Fraud rate", f"{rate:.3%}", "Computed at the evaluation stage, after fitting")]


def timeline_chart(timeline: pd.DataFrame, cut: int, period: str | None) -> go.Figure:
    """Transactions per step, coloured by period, with the cut drawn between steps."""
    fig = go.Figure()
    for name, part in timeline.groupby("period", sort=True):
        selected = period is None or period == name
        fig.add_bar(x=part["step"], y=part["transactions"], name=f"{name} period",
                    marker_color=PERIOD_COLORS[name], marker_line_width=0,
                    opacity=1.0 if selected else 0.2,  # dim, don't recolour: colour stays with the period
                    hovertemplate="Step %{x}: %{y:,} transactions<extra>" + name + "</extra>")
    # The cut sits between step 520 (last fitting hour) and 521 (first holdout hour).
    fig.add_vline(x=cut + 0.5, line_width=2, line_dash="dash", line_color="#5e5c57")
    top = timeline["transactions"].max()
    # Direct text labels, because these two colours are below 3:1 contrast on white.
    fig.add_annotation(x=cut / 2, y=top, text="<b>Fitting period</b>", showarrow=False, yanchor="bottom")
    fig.add_annotation(x=(cut + timeline["step"].max()) / 2, y=top, text="<b>Holdout period</b>",
                       showarrow=False, yanchor="bottom")
    fig.add_annotation(x=cut + 0.5, y=top * 0.8, text=f"Split: step {cut} | {cut + 1}", showarrow=False,
                       xanchor="left", xshift=6)
    fig.update_layout(bargap=0, barmode="overlay")
    return style(fig, height=360, title="Transactions per hour across the whole dataset",
                 xaxis_title="Step (hour)", yaxis_title="Transactions", hovermode="closest")


def split_section(m: dict):
    """FR1: the time split and the features the models see."""
    st.subheader("Data split and features")
    run = m.get("run")
    if run is None:  # metrics.json written before the run facts existed
        st.info("Re-run `python scripts/train_and_evaluate.py` to record the split and feature details.")
        return
    a, b = run["train_steps"], run["holdout_steps"]

    choice = st.segmented_control("Show", list(CHOICES), default="Both periods", key="period")
    period = CHOICES.get(choice)  # clicking the selected option again clears it; that falls back to both

    # Wider last column: "0.095% / 0.867%" is the longest value and was cut off at equal widths.
    for col, (label, value, help_text) in zip(st.columns([1, 1, 0.7, 1.7]), period_figures(m, period)):
        col.metric(label, value, help=help_text)

    if (ARTIFACTS_DIR / "timeline.parquet").exists():
        timeline = load_parquet("timeline.parquet")
        st.plotly_chart(timeline_chart(timeline, a[1], period), width="stretch", config=PLOTLY_CONFIG)
    else:
        st.info("Run `python scripts/make_timeline.py` to draw the split timeline. It reads only the `step` "
                "column and retrains nothing.")

    st.caption(f"The split is by time (fitting steps {a[0]}-{a[1]}, holdout steps {b[0]}-{b[1]}) and is made "
               "before any scaling or model fitting. The fraud label is first used in the " + run["label_first_used"] + ".")

    group_of = {col: group for group, cols in EXPLAIN_GROUPS.items() for col in cols}
    table = pd.DataFrame({"Feature": run["feature_names"],
                          "Explanation group": [group_of.get(f, "Context (never reset)") for f in run["feature_names"]]})
    st.dataframe(table, hide_index=True, width="stretch", height=300)
    st.caption("`isFraud` and `isFlaggedFraud` are not features. The explanation resets one group at a time; "
               "transaction type and the merchant flag are context and are never reset.")


def model_cards(m: dict):
    """FR2: how each detector was fitted and calibrated (from metrics.json)."""
    st.subheader("How each detector was fitted")
    run = m.get("run")
    if run is None:
        st.info("Re-run `python scripts/train_and_evaluate.py` to produce this analysis.")
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


def page_data_models():
    st.title("Data and models")
    st.caption("How the PaySim transactions were split in time, which features the detectors see, and how "
               "each detector was fitted. The detectors never see the fraud label; the fraud rates shown here "
               "were computed afterwards, at the evaluation stage.")
    m = load_json("metrics.json")
    source_banner(m)
    split_section(m)
    model_cards(m)


page_data_models()
