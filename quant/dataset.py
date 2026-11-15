"""Minute-by-minute dataset for NIFTY and SENSEX: features known at the close of each 1-min candle + what happened
over the next 1 / 5 / 15 minutes. Reads dhan_algo's cached data READ-ONLY (never writes there).

    python -m quant.dataset            -> data/minutes_NIFTY.parquet, data/minutes_SENSEX.parquet

No look-ahead: every feature at minute t uses candles up to and including t only (the candle that just closed).
Labels (per horizon h): move = close(t+h) / close(t) - 1, threshold = K * sigma60 * sqrt(h) where sigma60 is the
standard deviation of 1-min returns over the last 60 minutes (a quiet day needs a smaller move than a wild day):
  bullish (2) if move > +threshold, bearish (0) if move < -threshold, else sideways (1).
"""
from __future__ import annotations

import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT.parent / "dhan_algo"
SRC = ALGO / "data"                                    # read-only
OUT = ROOT / "data"
sys.path.insert(0, str(ALGO))
from algo.data.expiry_calendar import build_expiries, full_trading_days  # noqa: E402
sys.path.insert(0, str(ROOT))
from quant import features as F  # noqa: E402

IDS = {"NIFTY": "13", "SENSEX": "51"}
START = "2021-11-01"
HORIZONS = (1, 5, 15)
K = 0.5                                                # label threshold in units of sigma60 * sqrt(h)
OI_STRIKES = 3                                         # ATM +-3 for open-interest totals


def load_index(name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(SRC / "bars" / f"IDX_I_{IDS[name]}" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["ts", "open", "high", "low", "close"]) for f in files
                    if Path(f).stem[:7] >= START[:7]], ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts")
    t = df.ts.dt.hour * 60 + df.ts.dt.minute
    df = df[(t >= 555) & (t <= 929) & (df.ts.dt.weekday < 5)]          # 09:15 .. 15:29
    days = set(full_trading_days(df))
    df = df[df.ts.dt.date.isin(days)].reset_index(drop=True)
    return df


def _opt_month(args) -> pd.DataFrame:
    """Per-minute option features for one month: ATM CE/PE premium + IV, OI totals ATM+-3, straddle/spot."""
    uid, ym = args
    parts = {}
    for side in ("CE", "PE"):
        d = SRC / "rolling" / f"{uid}_WEEK0_{side}"
        f = d / f"{ym}.parquet"
        f = f if f.exists() else d / f"{ym}.partial.parquet"
        if not f.exists():
            return pd.DataFrame()
        o = pd.read_parquet(f)                       # empty months (no data at Dhan) lack some columns
        if o.empty or not {"ts", "close", "oi", "iv", "offset"} <= set(o.columns):
            return pd.DataFrame()
        atm = o[o.offset == 0].drop_duplicates("ts", keep="last").set_index("ts")
        near = o[o.offset.abs() <= OI_STRIKES].groupby("ts").oi.sum()
        parts[side] = pd.DataFrame({f"{side}_px": atm.close, f"{side}_iv": atm.iv, f"{side}_oi": near,
                                    f"{side}_k": atm.strike if "strike" in atm else np.nan})
    out = parts["CE"].join(parts["PE"], how="outer")
    out.index.name = "ts"
    return out.reset_index()


def option_features(name: str, months: list[str]) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=10) as ex:
        frames = [f for f in ex.map(_opt_month, [(IDS[name], m) for m in months]) if len(f)]
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts") if frames else pd.DataFrame(columns=["ts"])


def load_vix() -> pd.DataFrame:
    vf = sorted(glob.glob(str(SRC / "bars" / "IDX_I_21" / "*.parquet")))
    if not vf:
        return pd.DataFrame(columns=["ts", "vix"])
    v = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in vf], ignore_index=True)
    return v.drop_duplicates("ts").sort_values("ts").rename(columns={"close": "vix"})


def build(name: str, other: pd.DataFrame | None = None) -> pd.DataFrame:
    """Research dataset = shared features (quant/features.py) on the cached history + labels."""
    df = F.index_features(load_index(name))
    df = F.add_expiry(df, build_expiries(name, sorted(df.day.dt.date.unique())))
    df = F.add_options(df, option_features(name, sorted(df.ts.dt.strftime("%Y-%m").unique())))
    df = F.add_vix(df, load_vix())
    df = F.add_cross(df, other)
    g = df.groupby("day", sort=False)
    for h in HORIZONS:
        fwd = (g.close.shift(-h) / df.close - 1) * 1e4
        thr = K * df.sigma60 * np.sqrt(h)
        df[f"fwd{h}"] = fwd
        lab = np.where(fwd > thr, 2, np.where(fwd < -thr, 0, 1)).astype(float)
        lab[fwd.isna() | thr.isna()] = np.nan
        df[f"y{h}"] = lab
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nifty = build("NIFTY")
    sensex = build("SENSEX", other=nifty)
    nifty = build("NIFTY", other=sensex)
    for name, d in (("NIFTY", nifty), ("SENSEX", sensex)):
        d.to_parquet(OUT / f"minutes_{name}.parquet", index=False)
        print(f"{name}: {len(d):,} minutes, {d.day.nunique()} days, {d.ts.min()} .. {d.ts.max()}, "
              f"{d.shape[1]} columns; y5 share bear/side/bull "
              f"{(d.y5 == 0).mean():.2f}/{(d.y5 == 1).mean():.2f}/{(d.y5 == 2).mean():.2f}")


if __name__ == "__main__":
    main()
