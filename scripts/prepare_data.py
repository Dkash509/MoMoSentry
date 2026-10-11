"""Convert the real PaySim CSV (from Kaggle) into data/paysim.parquet.

1. Download "Synthetic Financial Datasets For Fraud Detection" (ealaxi/paysim1)
   from Kaggle and put the CSV in data/.
2. python scripts/prepare_data.py                      # default Kaggle file name
   python scripts/prepare_data.py --csv data/paysim.csv  # any other name
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from momosentry import config, data  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=config.RAW_CSV)
    args = ap.parse_args()
    if not args.csv.exists():
        sys.exit(f"CSV not found at {args.csv}. Download it from Kaggle first.")
    df = data.csv_to_parquet(args.csv)
    print(f"{len(df):,} rows, fraud rate {df.isFraud.mean():.4%} -> {config.PARQUET}")
