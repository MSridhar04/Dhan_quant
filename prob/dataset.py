"""Minute-by-minute dataset for NIFTY and SENSEX: features known at the close of each 1-min candle + what happened
over the next 1 / 5 / 15 minutes. Reads dhan_algo's cached data READ-ONLY (never writes there).

    python -m prob.dataset            -> data/minutes_NIFTY.parquet, data/minutes_SENSEX.parquet

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


def build(name: str, other: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_index(name)
    df["day"] = df.ts.dt.normalize()
    g = df.groupby("day", sort=False)
    c = df.close
    ret1 = g.close.pct_change()
    df["mod"] = df.ts.dt.hour * 60 + df.ts.dt.minute - 555            # minute of the session, 0 = 09:15
    df["dow"] = df.ts.dt.weekday
    # ---- recent moves (within the day; early minutes fall back to "since open")
    day_open = g.open.transform("first")
    for k in (1, 3, 5, 15, 30, 60):
        prev = g.close.shift(k)
        df[f"r{k}"] = (c / prev.fillna(day_open) - 1) * 1e4                # basis points
    df["body"] = (df.close - df.open) / df.open * 1e4
    rng = (df.high - df.low).replace(0, np.nan)
    df["upper_wick"] = (df.high - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df.low) / rng
    df["range_bp"] = (df.high - df.low) / df.close * 1e4
    sq = ret1.pow(2)
    df["sigma15"] = np.sqrt(sq.groupby(df.day).transform(lambda s: s.rolling(15, min_periods=5).mean())) * 1e4
    df["sigma60"] = np.sqrt(sq.groupby(df.day).transform(lambda s: s.rolling(60, min_periods=10).mean())) * 1e4
    df["vol_ratio"] = df.sigma15 / df.sigma60
    up = ret1.clip(lower=0)
    dn = (-ret1).clip(lower=0)
    rs = up.groupby(df.day).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean()) / \
        dn.groupby(df.day).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean()).replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    # ---- position in the day
    hi, lo = g.high.cummax(), g.low.cummin()
    df["from_open"] = (c / day_open - 1) * 1e4
    df["from_high"] = (c / hi - 1) * 1e4
    df["from_low"] = (c / lo - 1) * 1e4
    df["day_pos"] = ((c - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)
    tp = (df.high + df.low + df.close) / 3
    twap = tp.groupby(df.day).cumsum() / g.cumcount().add(1)
    df["from_twap"] = (c / twap - 1) * 1e4
    for span in (9, 21):
        ema = g.close.transform(lambda s, n=span: s.ewm(span=n, adjust=False).mean())
        df[f"from_ema{span}"] = (c / ema - 1) * 1e4
    # ---- yesterday / daily context
    daily = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), cl=("close", "last"))
    pc = daily.cl.shift()
    tr = pd.concat([daily.h - daily.l, (daily.h - pc).abs(), (daily.l - pc).abs()], axis=1).max(axis=1)
    ctx = pd.DataFrame({"prev_close": pc, "prev_high": daily.h.shift(), "prev_low": daily.l.shift(),
                        "gap": (daily.o / pc - 1) * 1e4, "prev_ret": (daily.cl / daily.o - 1).shift() * 1e4,
                        "prev_range": ((daily.h - daily.l) / daily.cl).shift() * 1e4,
                        "quiet": (tr / tr.rolling(14).mean()).shift(),
                        "atr14_bp": (tr.rolling(14).mean() / daily.cl).shift() * 1e4})
    pp = (ctx.prev_high + ctx.prev_low + ctx.prev_close) / 3
    ctx["pivot"] = pp
    df = df.join(ctx, on="day")
    df["from_prev_close"] = (c / df.prev_close - 1) * 1e4
    df["from_prev_high"] = (c / df.prev_high - 1) * 1e4
    df["from_prev_low"] = (c / df.prev_low - 1) * 1e4
    df["from_pivot"] = (c / df["pivot"] - 1) * 1e4
    df = df.drop(columns=["prev_close", "prev_high", "prev_low", "pivot"])
    # ---- expiry
    tdays = sorted(df.day.dt.date.unique())
    exps = build_expiries(name, tdays)
    ex = pd.Series(pd.to_datetime(exps)).sort_values().to_numpy()
    pos = np.searchsorted(ex, df.day.to_numpy())
    nxt = pd.to_datetime(ex[np.minimum(pos, len(ex) - 1)])
    tidx = {d: i for i, d in enumerate(pd.to_datetime(tdays))}
    df["dte"] = [tidx.get(n, np.nan) - tidx[d] if n in tidx else np.nan for d, n in zip(df.day, nxt)]
    df["expiry_day"] = (df.dte == 0).astype(int)
    # ---- options
    months = sorted(df.ts.dt.strftime("%Y-%m").unique())
    o = option_features(name, months)
    if len(o):
        df = df.merge(o, on="ts", how="left")
        straddle = df.CE_px + df.PE_px
        df["straddle_bp"] = straddle / df.close * 1e4
        df["iv_atm"] = (df.CE_iv + df.PE_iv) / 2
        oi_ok = (df.CE_oi > 0) & (df.PE_oi > 0)
        df["pcr"] = np.where(oi_ok, df.PE_oi / df.CE_oi.replace(0, np.nan), np.nan)
        gg = df.groupby("day")
        for k in (5, 15):
            for side in ("CE", "PE"):
                prev = gg[f"{side}_oi"].shift(k)
                df[f"{side}_oi_chg{k}"] = np.where(oi_ok, (df[f"{side}_oi"] / prev - 1) * 100, np.nan)
            df[f"pcr_chg{k}"] = df.pcr - gg.pcr.shift(k) if "pcr" in df else np.nan
            df[f"ce_px_r{k}"] = (df.CE_px / gg.CE_px.shift(k) - 1) * 100
            df[f"pe_px_r{k}"] = (df.PE_px / gg.PE_px.shift(k) - 1) * 100
            df[f"iv_chg{k}"] = df.iv_atm - gg.iv_atm.shift(k)
        # synthetic future (put-call parity at the ATM strike): F = K + C - P  (Step 2, 2026-11-15)
        same_k = df.CE_k == df.PE_k
        syn = np.where(same_k, df.CE_k + df.CE_px - df.PE_px, np.nan)
        df["syn_basis_bp"] = (syn / df.close - 1) * 1e4                  # futures premium over the index
        df["_syn"] = syn
        gg = df.groupby("day")
        for k in (1, 5, 15):
            df[f"syn_r{k}"] = (df._syn / gg._syn.shift(k) - 1) * 1e4
            df[f"syn_lead{k}"] = df[f"syn_r{k}"] - df[f"r{k}"]           # future moved more than the index
        df["basis_chg15"] = df.syn_basis_bp - gg.syn_basis_bp.shift(15)
        df = df.drop(columns=["CE_px", "PE_px", "CE_iv", "PE_iv", "CE_oi", "PE_oi", "CE_k", "PE_k", "_syn"])
    # ---- India VIX (Step 2): level, changes, vs day open / yesterday
    vf = sorted(glob.glob(str(SRC / "bars" / "IDX_I_21" / "*.parquet")))
    if vf:
        v = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in vf], ignore_index=True)
        v = v.drop_duplicates("ts").sort_values("ts").rename(columns={"close": "vix"})
        df = df.merge(v, on="ts", how="left")
        df["vix"] = df.groupby("day").vix.ffill()
        gv = df.groupby("day").vix
        for k in (5, 15, 60):
            df[f"vix_chg{k}"] = df.vix - gv.shift(k)
        df["vix_from_open"] = df.vix - gv.transform("first")
        vd = df.groupby("day").vix.last()
        df = df.join(pd.DataFrame({"vix_prev": vd.shift(), "vix_20d": vd.rolling(20).mean().shift()}), on="day")
        df["vix_vs_prev"] = df.vix - df.vix_prev
        df["vix_vs_20d"] = df.vix / df.vix_20d - 1
        df = df.drop(columns=["vix_prev", "vix_20d"])
    # ---- the other index (same minute)
    if other is not None:
        oth = other[["ts", "r1", "r5", "r15", "from_open"]].rename(
            columns={"r1": "x_r1", "r5": "x_r5", "r15": "x_r15", "from_open": "x_from_open"})
        df = df.merge(oth, on="ts", how="left")
        df["x_gap5"] = df.r5 - df.x_r5
    # ---- labels
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
