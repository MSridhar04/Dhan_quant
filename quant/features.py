"""Shared feature code: the SAME functions build the research dataset (quant/dataset.py) and the live minute
features (quant/live/engine.py). One definition = no train/live mismatch.

All functions work on in-memory frames:
  index bars  : ts, open, high, low, close   (1-min, several days so yesterday's context exists)
  option frame: ts, CE_px, PE_px, CE_oi, PE_oi, CE_k, PE_k [, CE_iv, PE_iv]   (ATM premiums, OI ATM+-3, ATM strikes)
  vix frame   : ts, vix
  other index : the other index's feature frame (NIFTY <-> SENSEX)
Every feature at minute t uses only candles up to and including t (no look-ahead).
Units: bp = basis points (1 bp = 0.01%).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

IV_FEATURES = ("iv_atm", "iv_chg5", "iv_chg15")      # Dhan's historical IV: not available live -> live models skip


def index_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price, volatility, position-in-day and yesterday features from 1-min index bars."""
    df = df.copy()
    df["day"] = df.ts.dt.normalize()
    g = df.groupby("day", sort=False)
    c = df.close
    ret1 = g.close.pct_change()
    df["mod"] = df.ts.dt.hour * 60 + df.ts.dt.minute - 555            # minute of the session, 0 = 09:15
    df["dow"] = df.ts.dt.weekday
    day_open = g.open.transform("first")
    for k in (1, 3, 5, 15, 30, 60):
        prev = g.close.shift(k)
        df[f"r{k}"] = (c / prev.fillna(day_open) - 1) * 1e4
    df["body"] = (df.close - df.open) / df.open * 1e4
    rng = (df.high - df.low).replace(0, np.nan)
    df["upper_wick"] = (df.high - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df.low) / rng
    df["range_bp"] = (df.high - df.low) / df.close * 1e4
    sq = ret1.pow(2)
    df["sigma15"] = np.sqrt(sq.groupby(df.day).transform(lambda s: s.rolling(15, min_periods=5).mean())) * 1e4
    df["sigma60"] = np.sqrt(sq.groupby(df.day).transform(lambda s: s.rolling(60, min_periods=10).mean())) * 1e4
    df["vol_ratio"] = df.sigma15 / df.sigma60
    up, dn = ret1.clip(lower=0), (-ret1).clip(lower=0)
    rs = up.groupby(df.day).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean()) / \
        dn.groupby(df.day).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean()).replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
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
    daily = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), cl=("close", "last"))
    pc = daily.cl.shift()
    tr = pd.concat([daily.h - daily.l, (daily.h - pc).abs(), (daily.l - pc).abs()], axis=1).max(axis=1)
    ctx = pd.DataFrame({"prev_close": pc, "prev_high": daily.h.shift(), "prev_low": daily.l.shift(),
                        "gap": (daily.o / pc - 1) * 1e4, "prev_ret": (daily.cl / daily.o - 1).shift() * 1e4,
                        "prev_range": ((daily.h - daily.l) / daily.cl).shift() * 1e4,
                        "quiet": (tr / tr.rolling(14).mean()).shift(),
                        "atr14_bp": (tr.rolling(14).mean() / daily.cl).shift() * 1e4})
    ctx["pivot"] = (ctx.prev_high + ctx.prev_low + ctx.prev_close) / 3
    df = df.join(ctx, on="day")
    df["from_prev_close"] = (c / df.prev_close - 1) * 1e4
    df["from_prev_high"] = (c / df.prev_high - 1) * 1e4
    df["from_prev_low"] = (c / df.prev_low - 1) * 1e4
    df["from_pivot"] = (c / df["pivot"] - 1) * 1e4
    return df.drop(columns=["prev_close", "prev_high", "prev_low", "pivot"])


def add_expiry(df: pd.DataFrame, expiries: list[date], holidays: set[date] | None = None) -> pd.DataFrame:
    """dte = trading days until the nearest weekly expiry (0 on expiry day). Days in the data are counted directly;
    an expiry beyond the data (live) is counted with the business-day calendar minus market holidays."""
    tdays = sorted(df.day.dt.date.unique())
    ex = pd.Series(pd.to_datetime(sorted(expiries))).to_numpy()
    pos = np.searchsorted(ex, df.day.to_numpy())
    nxt = pd.to_datetime(ex[np.minimum(pos, len(ex) - 1)])
    tidx = {d: i for i, d in enumerate(pd.to_datetime(tdays))}
    hol = np.array(sorted(holidays or []), dtype="datetime64[D]")

    def dte(d, n):
        if n in tidx:
            return tidx[n] - tidx[d]
        if n < d:
            return np.nan
        return float(np.busday_count(d.date(), n.date(), holidays=hol))
    df["dte"] = [dte(d, n) for d, n in zip(df.day, nxt)]
    df["expiry_day"] = (df.dte == 0).astype(int)
    return df


def add_options(df: pd.DataFrame, o: pd.DataFrame, with_iv: bool = True) -> pd.DataFrame:
    """ATM straddle, put/call OI, option premium moves, synthetic future (F = K + C - P)."""
    if o is None or not len(o):
        return df
    df = df.merge(o, on="ts", how="left")
    df["straddle_bp"] = (df.CE_px + df.PE_px) / df.close * 1e4
    if with_iv and "CE_iv" in df:
        df["iv_atm"] = (df.CE_iv + df.PE_iv) / 2
    oi_ok = (df.CE_oi > 0) & (df.PE_oi > 0)
    df["pcr"] = np.where(oi_ok, df.PE_oi / df.CE_oi.replace(0, np.nan), np.nan)
    gg = df.groupby("day")
    for k in (5, 15):
        for side in ("CE", "PE"):
            prev = gg[f"{side}_oi"].shift(k)
            df[f"{side}_oi_chg{k}"] = np.where(oi_ok, (df[f"{side}_oi"] / prev - 1) * 100, np.nan)
        df[f"pcr_chg{k}"] = df.pcr - gg.pcr.shift(k)
        df[f"ce_px_r{k}"] = (df.CE_px / gg.CE_px.shift(k) - 1) * 100
        df[f"pe_px_r{k}"] = (df.PE_px / gg.PE_px.shift(k) - 1) * 100
        if with_iv and "iv_atm" in df:
            df[f"iv_chg{k}"] = df.iv_atm - gg.iv_atm.shift(k)
    same_k = df.CE_k == df.PE_k
    df["_syn"] = np.where(same_k, df.CE_k + df.CE_px - df.PE_px, np.nan)
    df["syn_basis_bp"] = (df._syn / df.close - 1) * 1e4
    gg = df.groupby("day")
    for k in (1, 5, 15):
        df[f"syn_r{k}"] = (df._syn / gg._syn.shift(k) - 1) * 1e4
        df[f"syn_lead{k}"] = df[f"syn_r{k}"] - df[f"r{k}"]
    df["basis_chg15"] = df.syn_basis_bp - gg.syn_basis_bp.shift(15)
    drop = [c for c in ("CE_px", "PE_px", "CE_iv", "PE_iv", "CE_oi", "PE_oi", "CE_k", "PE_k", "_syn") if c in df]
    return df.drop(columns=drop)


def add_vix(df: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """India VIX level, its changes, vs today's open / yesterday's close / 20-day average."""
    if v is None or not len(v):
        return df
    df = df.merge(v[["ts", "vix"]], on="ts", how="left")
    df["vix"] = df.groupby("day").vix.ffill()
    gv = df.groupby("day").vix
    for k in (5, 15, 60):
        df[f"vix_chg{k}"] = df.vix - gv.shift(k)
    df["vix_from_open"] = df.vix - gv.transform("first")
    vd = df.groupby("day").vix.last()
    df = df.join(pd.DataFrame({"vix_prev": vd.shift(), "vix_20d": vd.rolling(20).mean().shift()}), on="day")
    df["vix_vs_prev"] = df.vix - df.vix_prev
    df["vix_vs_20d"] = df.vix / df.vix_20d - 1
    return df.drop(columns=["vix_prev", "vix_20d"])


def add_cross(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """The other index at the same minute (NIFTY <-> SENSEX)."""
    if other is None:
        return df
    oth = other[["ts", "r1", "r5", "r15", "from_open"]].rename(
        columns={"r1": "x_r1", "r5": "x_r5", "r15": "x_r15", "from_open": "x_from_open"})
    df = df.merge(oth, on="ts", how="left")
    df["x_gap5"] = df.r5 - df.x_r5
    return df


# ------------------------------------------------------------------ 50-stock breadth + heavyweights
HEAVY = {"1333": "hdfcbank", "2885": "reliance", "4963": "icici", "1594": "infy", "10604": "bharti", "11536": "tcs"}


def stock_minute(s: pd.DataFrame, sid: str) -> pd.DataFrame:
    """One stock's per-minute flags/moves (input: ts, open, close)."""
    s = s.drop_duplicates("ts").sort_values("ts").copy()
    s["day"] = s.ts.dt.normalize()
    g = s.groupby("day")
    return pd.DataFrame({"ts": s.ts, "up1": (g.close.diff() > 0).astype(float),
                         "up5": (s.close / g.close.shift(5) - 1 > 0).astype(float),
                         "above_open": (s.close > g.open.transform("first")).astype(float),
                         "r1": (s.close / g.close.shift(1) - 1) * 1e4,
                         "r5": (s.close / g.close.shift(5) - 1) * 1e4, "sid": sid})


def breadth(frames: list[pd.DataFrame], min_stocks: int = 30) -> pd.DataFrame:
    """Share of the 50 stocks rising / above their open, and the heavyweights' last moves, per minute."""
    if not frames:
        return pd.DataFrame(columns=["ts"])
    allst = pd.concat(frames, ignore_index=True)
    b = allst.groupby("ts").agg(br_up1=("up1", "mean"), br_up5=("up5", "mean"), br_above_open=("above_open", "mean"),
                                br_n=("up1", "size"))
    b = b[b.br_n >= min_stocks].drop(columns="br_n")
    for c in ("br_up1", "br_up5"):
        b[f"{c}_chg5"] = b[c] - b[c].shift(5)
    hv = allst[allst.sid.isin(HEAVY)].pivot_table(index="ts", columns="sid", values=["r1", "r5"])
    hv.columns = [f"hv_{HEAVY[sid]}_{k}" for k, sid in hv.columns]
    r1c = [c for c in hv.columns if c.endswith("_r1")]
    r5c = [c for c in hv.columns if c.endswith("_r5")]
    hv["hv_avg_r1"] = hv[r1c].mean(axis=1) if r1c else np.nan
    hv["hv_avg_r5"] = hv[r5c].mean(axis=1) if r5c else np.nan
    return b.join(hv, how="outer").reset_index()
