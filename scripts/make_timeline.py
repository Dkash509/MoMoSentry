"""Write artifacts/timeline.parquet WITHOUT retraining anything.

    python scripts/make_timeline.py                    # uses data/paysim.parquet
    python scripts/make_timeline.py --data <file.parquet>

Only the `step` column is read from the Parquet file, so no other column (and
never the label) is loaded. The models, thresholds and metrics are untouched,
which means no reported number can change. train_and_evaluate.py writes the same
file, so a full re-run regenerates it too.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from momosentry import config, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.PARQUET)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR)
    args = ap.parse_args()

    # columns=["step"] makes pyarrow read that one column from disk and skip the rest.
    steps = pq.read_table(args.data, columns=["step"]).column("step").to_pandas()
    cut = data.split_step(int(steps.max()))  # the same cut time_split uses
    timeline = data.step_timeline(steps, cut)

    args.out.mkdir(parents=True, exist_ok=True)
    timeline.to_parquet(args.out / "timeline.parquet", index=False)
    per_period = timeline.groupby("period")["transactions"].sum()
    print(f"{len(timeline)} steps, cut at step {cut}: "
          f"fitting {per_period.get('Fitting', 0):,} rows, holdout {per_period.get('Holdout', 0):,} rows")


if __name__ == "__main__":
    main()
