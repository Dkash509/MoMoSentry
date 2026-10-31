"""Loading PaySim and splitting it in time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config

PAYSIM_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}
LABEL = "isFraud"


def csv_to_parquet(csv_path=config.RAW_CSV, out_path=config.PARQUET) -> pd.DataFrame:
    """Convert the 470 MB Kaggle CSV to a typed Parquet file (~5x smaller, ~10x faster)."""
    df = pd.read_csv(csv_path, dtype=PAYSIM_DTYPES)
    df.to_parquet(out_path, index=False)
    return df


# nameOrig (the sender's account id) is never used by any feature, rule or page,
# so it is not loaded at all. On real PaySim it is 6.3M Python strings, which on
# its own costs well over a gigabyte of RAM.
LOAD_COLUMNS = [c for c in PAYSIM_DTYPES if c != "nameOrig"]


def load(path=None) -> pd.DataFrame:
    """Load the real dataset if prepared, otherwise the synthetic sample.

    Memory-saving step: nameDest (the receiver's account id) is only ever used
    for its first letter, M = merchant or C = customer (see features.py). So the
    id is cut to that one letter while still inside pyarrow, before pandas turns
    it into Python strings, and stored as a category (2 possible values). The
    models see exactly the same information, and the results do not change.
    """
    if path is None:
        path = config.PARQUET if config.PARQUET.exists() else config.SAMPLE_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_data.py (real data) or "
            "scripts/make_sample_data.py (synthetic sample) first."
        )
    table = pq.read_table(path, columns=LOAD_COLUMNS)
    i = table.schema.get_field_index("nameDest")
    first_letter = pc.utf8_slice_codeunits(table["nameDest"], 0, 1)  # "M1979787155" -> "M"
    table = table.set_column(i, "nameDest", pc.dictionary_encode(first_letter))  # -> pandas category
    # self_destruct frees each arrow column as soon as pandas has its copy, so
    # the data is never held twice in full.
    return table.to_pandas(self_destruct=True, split_blocks=True)


def split_step(max_step: int, train_frac=config.TRAIN_FRACTION_OF_STEPS) -> int:
    """The last step of the fitting period. On PaySim: floor(743 * 0.70) = 520."""
    return int(np.floor(max_step * train_frac))


def time_split(df: pd.DataFrame, train_frac=config.TRAIN_FRACTION_OF_STEPS):
    """Split by `step` (hour). Everything before the cut is train, after is holdout.

    A random split would let the model 'see the future' (same accounts, same
    hours on both sides), which inflates results.
    """
    cut = split_step(df["step"].max(), train_frac)
    train = df[df["step"] <= cut].reset_index(drop=True)
    holdout = df[df["step"] > cut].reset_index(drop=True)
    return train, holdout, cut


def step_timeline(steps: pd.Series, cut: int) -> pd.DataFrame:
    """Transactions per hourly step, and which side of the split each step is on.

    Built from the `step` column alone, so it carries no label information.
    The dashboard draws the split from this (FR1).
    """
    counts = steps.value_counts().sort_index()  # index = step, value = number of transactions
    step = counts.index.to_numpy(dtype=int)
    return pd.DataFrame({
        "step": step,
        "transactions": counts.to_numpy(dtype=int),
        "period": np.where(step <= cut, "Fitting", "Holdout"),  # same <= rule as time_split
    })


def type_stratified_sample(df: pd.DataFrame, n: int, seed=config.RANDOM_STATE) -> pd.DataFrame:
    """Sample n rows keeping the transaction-type mix. Uses `type` only, never the label."""
    if n >= len(df):
        return df
    frac = n / len(df)
    return (
        df.groupby("type", observed=True, group_keys=False)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )
