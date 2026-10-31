"""Every dashboard page renders without exceptions (uses the committed artifacts/)."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "streamlit_app.py")
# Sidebar order = FR order: FR1-2, FR3-4, FR5, FR6-7, FR8. The first page is the entry point.
FR_ORDER = ["data_models", "score", "flagged", "compare", "overview"]


def test_sidebar_follows_fr_order():
    src = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert re.findall(r'st\.Page\("views/(\w+)\.py"', src) == FR_ORDER


@pytest.mark.parametrize("page", FR_ORDER)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page != FR_ORDER[0]:  # the entry page is what the app opens on
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, at.exception
    if page == "score":
        at.radio[0].set_value("Account drained at 3am").run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        assert any("Alert raised" in e.value for e in at.error)
        at.radio[1].set_value("lof").run()
        assert not at.exception and len(at.error) + len(at.success) == 2  # result survives the rerun


def test_period_control_switches_the_split_figures():
    """FR1: picking a period changes the tiles to that side of the split, using metrics.json values."""
    meta = json.loads((ROOT / "artifacts" / "metrics.json").read_text())["meta"]
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Fitting rows"] == f"{meta['train_rows']:,}"      # default view: both periods
    assert tiles["Holdout rows"] == f"{meta['holdout_rows']:,}"

    for choice, rows in [("Fitting period", meta["train_rows"]), ("Holdout period", meta["holdout_rows"])]:
        at.segmented_control(key="period").set_value(choice).run()
        assert not at.exception, at.exception
        assert {m.label: m.value for m in at.metric}["Rows"] == f"{rows:,}"


def test_timeline_matches_the_split_it_draws():
    """The timeline is drawn from timeline.parquet, so its totals must equal the split in metrics.json.
    This fails if the timeline was built from different data, or with a different cut."""
    meta = json.loads((ROOT / "artifacts" / "metrics.json").read_text())["meta"]
    tl = pd.read_parquet(ROOT / "artifacts" / "timeline.parquet")
    per_period = tl.groupby("period")["transactions"].sum()
    assert per_period["Fitting"] == meta["train_rows"]
    assert per_period["Holdout"] == meta["holdout_rows"]
    assert tl.loc[tl.period == "Fitting", "step"].max() == meta["split_step"]
    assert tl.loc[tl.period == "Holdout", "step"].min() == meta["split_step"] + 1
    # The holdout part must agree hour by hour with hourly.parquet, which the Overview page draws.
    hourly = pd.read_parquet(ROOT / "artifacts" / "hourly.parquet")
    merged = tl[tl.period == "Holdout"].merge(hourly, on="step", suffixes=("", "_overview"))
    assert len(merged) == len(hourly)
    assert (merged["transactions"] == merged["transactions_overview"]).all()
