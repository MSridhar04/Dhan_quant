"""Check: the refactored dataset (shared quant/features.py) equals the one built before the refactor."""
import sys

import numpy as np
import pandas as pd

if __name__ == "__main__":
    for n in ("NIFTY", "SENSEX"):
        a = pd.read_parquet(f"data/check/minutes_{n}_before.parquet")
        b = pd.read_parquet(f"data/minutes_{n}.parquet")
        cols_a, cols_b = set(a.columns), set(b.columns)
        diff_cols = [c for c in a.columns if c in cols_b and not np.allclose(
            pd.to_numeric(a[c], errors="coerce").fillna(-9e9), pd.to_numeric(b[c], errors="coerce").fillna(-9e9),
            equal_nan=True) and a[c].dtype.kind in "fiu"]
        print(n, "rows", len(a), len(b), "| only before:", cols_a - cols_b, "| only after:", cols_b - cols_a,
              "| numeric columns that differ:", diff_cols)
    sys.exit(0)
