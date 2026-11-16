"""Train the models the live engine uses (plain English: docs/03_live_dashboard.md).

    python -m quant.live.train      -> models/live/<IDX>_<model>.txt + models/live/meta.json

Models per index (NIFTY, SENSEX):
  big30  P(the next 30 minutes are BUSY = range in the top 25% for that volatility)      direction-free
  tb30   P(bullish / neither / bearish first-touch within 30 minutes)                     direction
Same features as research minus Dhan's historical IV (not in the live feed). Trees are chosen by early stopping on
the last 12 months, then the model is refit on ALL data (Nov 2021 -> latest) - the 2026 out-of-sample results are in
reports/step2_tests.txt; this is the production refit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from quant.features import IV_FEATURES  # noqa: E402
from quant.model import PARAMS  # noqa: E402
from quant.step1 import _feat_list, add_labels, breadth_features  # noqa: E402

OUT = ROOT / "models" / "live"
H = 30


def fit(d: pd.DataFrame, feats: list[str], y: str, params: dict):
    d = d[d[y].notna()]
    cut = d.ts.max() - pd.DateOffset(months=12)
    a, v = d[d.ts < cut], d[d.ts >= cut]
    m0 = lgb.train(params, lgb.Dataset(a[feats], a[y]), 3000, valid_sets=[lgb.Dataset(v[feats], v[y])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    n = max(int(m0.best_iteration * len(d) / len(a)), 50)
    return lgb.train(params, lgb.Dataset(d[feats], d[y]), n), n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    br = breadth_features()
    meta = {}
    for name in ("NIFTY", "SENSEX"):
        df = add_labels(pd.read_parquet(ROOT / "data" / f"minutes_{name}.parquet").merge(br, on="ts", how="left"))
        feats = [f for f in _feat_list(df) if f not in IV_FEATURES]
        q75 = float(df[f"rng{H}"].quantile(0.75))
        df[f"big{H}"] = np.where(df[f"rng{H}"].notna(), (df[f"rng{H}"] > q75).astype(float), np.nan)
        mb, nb = fit(df, feats, f"big{H}", {**PARAMS, "objective": "binary", "num_class": 1, "num_threads": 10})
        mt, nt = fit(df, feats, f"tb{H}", {**PARAMS, "num_threads": 10})
        mb.save_model(str(OUT / f"{name}_big{H}.txt"))
        mt.save_model(str(OUT / f"{name}_tb{H}.txt"))
        base_tb = df[f"tb{H}"].value_counts(normalize=True).reindex([0.0, 1.0, 2.0]).round(3).tolist()
        pb = mb.predict(df[feats])
        meta[name] = {"features": feats, "horizon_min": H, "big_cut_ratio": q75, "big_base": 0.25,
                      "tb_base": base_tb, "trees": {"big": nb, "tb": nt},
                      "big_quiet_below": float(np.nanquantile(pb, 0.10)), "big_busy_above": float(np.nanquantile(pb, 0.90)),
                      "trained_to": str(df.ts.max())}
        print(f"{name}: {len(feats)} features, trees big {nb} / tb {nt}, quiet<{meta[name]['big_quiet_below']:.2f} "
              f"busy>{meta[name]['big_busy_above']:.2f}", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
