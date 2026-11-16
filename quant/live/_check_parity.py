"""Parity check: features built by the live engine (replay) == research dataset features, same minutes."""
import sys
from datetime import date
import numpy as np
import pandas as pd
from quant.live import engine as E
from quant.step1 import breadth_features

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    day = date(2026, 11, 12)
    times = [pd.Timestamp(f"2026-11-12 {h}") for h in ("10:00", "12:30", "14:45")]
    import quant.live.engine as eng_mod
    # run replay but stop early: reuse replay() internals by monkeypatching compute to record
    rec = {}
    orig = E.QuantEngine.compute
    def compute(self, t):
        out = orig(self, t)
        if t in times:
            rec[t] = {n: x.copy() for n, x in self.last_X.items()}
        return out
    E.QuantEngine.compute = compute
    E.replay(day)
    br = breadth_features()
    for n in ("NIFTY", "SENSEX"):
        d = pd.read_parquet(f"data/minutes_{n}.parquet").merge(br, on="ts", how="left").set_index("ts")
        for t in times:
            live = rec[t][n].iloc[0]
            res = d.loc[t, live.index].astype(float)
            diff = (live - res).abs()
            bad = diff[(diff > 1e-6 * (1 + res.abs())) & ~(live.isna() & res.isna())]
            both_nan_mismatch = live.isna() != res.isna()
            print(f"{n} {t:%H:%M}: {len(live)} features, {len(bad)} differ, NaN mismatch {int(both_nan_mismatch.sum())}")
            if len(bad):
                print("   ", {k: (round(float(live[k]), 4), round(float(res[k]), 4)) for k in bad.index[:12]})
            if both_nan_mismatch.any():
                print("    nan:", {k: (live[k], res[k]) for k in both_nan_mismatch[both_nan_mismatch].index[:8]})
