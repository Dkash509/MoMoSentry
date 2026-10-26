"""Small end-to-end run of the real training script on synthetic data."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_end_to_end(tmp_path, monkeypatch):
    from make_sample_data import generate


    data_path = tmp_path / "sample.parquet"
    generate(30_000).to_parquet(data_path)
    out = tmp_path / "artifacts"
    env_code = (
        "import sys; sys.argv=['x','--data',r'%s','--out',r'%s'];"
        "from momosentry import config; config.LOF_FIT_SAMPLE=5000; config.EXPORT_NORMAL_ROWS=500;"
        "import runpy; runpy.run_path(r'%s', run_name='__main__')"
    ) % (data_path, out, ROOT / "scripts" / "train_and_evaluate.py")
    subprocess.run([sys.executable, "-c", env_code], cwd=ROOT, check=True)

    m = json.loads((out / "metrics.json").read_text())
    assert m["meta"]["source"] == "paysim"  # custom path, not the bundled sample
    for v in m["comparison_set"].values():
        assert 0 <= v["roc_auc"] <= 1
    # Synthetic fraud is designed to be anomalous: IF must beat random clearly.
    assert m["comparison_set"]["Isolation Forest"]["roc_auc"] > 0.7

    # Second rule baseline, budget sweep and overlap analysis are all produced.
    from momosentry import config
    assert "Balance-drain rule" in m["comparison_set"]
    for sweep in m["budgets"].values():
        assert [r["budget"] for r in sweep] == list(config.ALERT_BUDGETS)
        recalls = [r["recall"] for r in sweep]
        assert recalls == sorted(recalls)  # a bigger budget can never lower recall
    assert len(m["overlap"]["pairs"]) == 6  # 4 methods -> 6 pairs

    from momosentry.explain import explain
    from momosentry.models import Bundle
    bundle = Bundle.load(out / "model_bundle.joblib")
    exp = pd.read_parquet(out / "holdout_scored.parquet")
    assert {"if_pct", "lof_pct", "if_alert"} <= set(exp.columns)

    # Run facts that the dashboard shows must exist and be consistent.
    run = m["run"]
    assert run["train_steps"][1] < run["holdout_steps"][0]
    assert len(run["feature_names"]) == 15
    assert not {"isFraud", "isFlaggedFraud"} & set(run["feature_names"])
    assert run["local_outlier_factor"]["fit_calibration_overlap"] == 0

    # The export must hold EVERY alert of both detectors, so the flagged page is complete.
    # The totals come from metrics.json, which was computed on the full holdout.
    iso = m["full_holdout"]["Isolation Forest"]
    assert int(exp["if_alert"].sum()) == iso["tp"] + iso["fp"]
    assert run["local_outlier_factor"]["alerts_complete"]
    lof = m["comparison_set"]["Local Outlier Factor"]
    assert int(exp["lof_alert"].sum()) == lof["tp"] + lof["fp"]
    assert run["export_rows"] == len(exp)
    ex = explain(bundle, exp.iloc[[0]], "if")
    assert len(ex) == 6 and abs(ex.share.sum() - 1) < 1e-6 or ex.share.sum() == 0
