"""Every dashboard page renders without exceptions (uses artifacts/ from the bundled sample)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.mark.parametrize("page", ["overview", "score", "flagged", "compare"])
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, at.exception
    if page == "score":
        at.radio[0].set_value("Account drained at 3am").run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        assert any("Alert raised" in e.value for e in at.error)
        at.radio[1].set_value("lof").run()
        assert not at.exception and len(at.error) + len(at.success) == 2  # result survives the rerun
