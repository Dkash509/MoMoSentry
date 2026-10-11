"""Generate a SYNTHETIC dataset with PaySim's exact schema.

Why: the real PaySim CSV is 470 MB and needs a Kaggle login, so it cannot live
in the repo or in CI. This sample lets the whole pipeline, the tests and the
dashboard run anywhere. It imitates PaySim's broad behaviour (type mix,
log-normal amounts, daily cycle, account-emptying fraud split into
TRANSFER -> CASH_OUT), but it is NOT the real data. Never report results
from it as findings.

Usage:  python scripts/make_sample_data.py --rows 300000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from momosentry import config  # noqa: E402

TYPE_MIX = {"CASH_OUT": 0.352, "PAYMENT": 0.338, "CASH_IN": 0.220, "TRANSFER": 0.084, "DEBIT": 0.006}
# (log-mean, log-sd) of amount per type, roughly PaySim-like
AMOUNT_LOGN = {"CASH_OUT": (11.9, 1.0), "PAYMENT": (8.9, 1.1), "CASH_IN": (11.8, 1.0),
               "TRANSFER": (12.9, 1.1), "DEBIT": (8.2, 1.0)}
N_STEPS = 743


def _hourly_weights() -> np.ndarray:
    hours = np.arange(N_STEPS) % 24
    # quiet at night, busy in the day
    w = 0.15 + np.exp(-((hours - 14) ** 2) / 30.0)
    return w / w.sum()


def generate(n_rows: int, fraud_rate: float = 0.0013, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud_pairs = max(2, int(n_rows * fraud_rate / 2))
    n_normal = n_rows - 2 * n_fraud_pairs

    # ---------------- normal traffic ----------------
    types = rng.choice(list(TYPE_MIX), size=n_normal, p=list(TYPE_MIX.values()))
    step = rng.choice(np.arange(1, N_STEPS + 1), size=n_normal, p=_hourly_weights())
    mu = np.array([AMOUNT_LOGN[t][0] for t in types])
    sd = np.array([AMOUNT_LOGN[t][1] for t in types])
    amount = np.round(np.exp(rng.normal(mu, sd)), 2)

    has_balance = rng.random(n_normal) > 0.33  # PaySim: ~1/3 of senders show 0 balance
    old_org = np.where(has_balance, np.round(np.exp(rng.normal(11.0, 1.8, n_normal)), 2), 0.0)
    is_cash_in = types == "CASH_IN"
    new_org = np.where(is_cash_in, old_org + amount, np.maximum(old_org - amount, 0.0))
    # PaySim is messy: some normal rows have balances that don't add up
    messy = rng.random(n_normal) < 0.08
    new_org = np.where(messy, np.round(new_org * rng.uniform(0, 1.5, n_normal), 2), new_org)

    is_payment = types == "PAYMENT"
    name_dest = np.where(
        is_payment,
        np.char.add("M", rng.integers(10**8, 10**9, n_normal).astype(str)),
        np.char.add("C", rng.integers(10**8, 10**9, n_normal).astype(str)),
    )
    old_dest = np.where(is_payment, 0.0, np.round(np.exp(rng.normal(12.5, 2.0, n_normal)), 2))
    old_dest = np.where(rng.random(n_normal) < 0.25, 0.0, old_dest)
    new_dest = np.where(is_payment, 0.0, np.where(is_cash_in, np.maximum(old_dest - amount, 0), old_dest + amount))
    dest_messy = (~is_payment) & (rng.random(n_normal) < 0.15)
    new_dest = np.where(dest_messy, np.round(old_dest * rng.uniform(0.5, 2.0, n_normal), 2), new_dest)

    normal = pd.DataFrame({
        "step": step, "type": types, "amount": amount,
        "nameOrig": np.char.add("C", rng.integers(10**8, 10**9, n_normal).astype(str)),
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "nameDest": name_dest, "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "isFraud": 0, "isFlaggedFraud": 0,
    })

    # ---------------- fraud: empty the account, then cash out ----------------
    f_step = rng.integers(1, N_STEPS + 1, n_fraud_pairs)  # fraud ignores the daily cycle
    f_bal = np.round(np.exp(rng.normal(12.5, 1.6, n_fraud_pairs)), 2)
    f_bal = np.minimum(f_bal, 10_000_000)  # PaySim caps a single fraud at 10M
    mule = np.char.add("C", rng.integers(10**8, 10**9, n_fraud_pairs).astype(str))
    zero_dest = rng.random(n_fraud_pairs) < 0.5
    t_old_dest = np.where(zero_dest, 0.0, np.round(np.exp(rng.normal(12.0, 2.0, n_fraud_pairs)), 2))
    t_new_dest = np.where(zero_dest, 0.0, t_old_dest + f_bal)

    transfer = pd.DataFrame({
        "step": f_step, "type": "TRANSFER", "amount": f_bal,
        "nameOrig": np.char.add("C", rng.integers(10**8, 10**9, n_fraud_pairs).astype(str)),
        "oldbalanceOrg": f_bal, "newbalanceOrig": 0.0,
        "nameDest": mule, "oldbalanceDest": t_old_dest, "newbalanceDest": t_new_dest,
        "isFraud": 1,
        # PaySim's built-in rule fires on very few frauds
        "isFlaggedFraud": ((f_bal > 200_000) & (rng.random(n_fraud_pairs) < 0.01)).astype(int),
    })
    c_old_dest = np.round(np.exp(rng.normal(12.5, 2.0, n_fraud_pairs)), 2)
    cashout = pd.DataFrame({
        "step": np.minimum(f_step + rng.integers(0, 2, n_fraud_pairs), N_STEPS), "type": "CASH_OUT",
        "amount": f_bal, "nameOrig": mule, "oldbalanceOrg": f_bal, "newbalanceOrig": 0.0,
        "nameDest": np.char.add("C", rng.integers(10**8, 10**9, n_fraud_pairs).astype(str)),
        "oldbalanceDest": c_old_dest, "newbalanceDest": c_old_dest + f_bal,
        "isFraud": 1, "isFlaggedFraud": 0,
    })

    df = pd.concat([normal, transfer, cashout], ignore_index=True)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    return df.astype({"step": "int16", "type": "category", "isFraud": "int8", "isFlaggedFraud": "int8",
                      "nameOrig": "string", "nameDest": "string"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--out", type=Path, default=config.SAMPLE_PARQUET)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df):,} synthetic rows ({df.isFraud.mean():.3%} fraud) -> {args.out}")


if __name__ == "__main__":
    main()
