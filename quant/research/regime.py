"""Research #2: regime detection (plain English: docs/07_regimes.md).

    python -m quant.research.run_regime     (unattended; Telegram)

1. DAILY FEATURES (NIFTY + India VIX), each known BEFORE the day opens (computed from yesterday and earlier):
   er20     efficiency ratio over 20 days = |net move| / sum of |daily moves|  (1 = straight trend, ~0 = chop)
   ret20    20-day return; dist50 = distance from the 50-day average
   rv10     10-day realised volatility; vix, vix_vs_250 = VIX vs its 1-year median
   day_eff  average of |close - open| / (high - low) over 10 days (are days trending or whipsawing?)
   rng5_20  5-day range / 20-day range (compression)
2. REGIME LABELS, three ways (all refit using training years only):
   rule     trend (er20 above the training median) x volatility (VIX above its 1-year median) -> 4 regimes
   kmeans   4 clusters of the standardised features
   hmm      3-state hidden Markov model on daily return + range; the state used for day t is the FORWARD-filtered
            one from data up to t-1 (no look-ahead)
3. TEST, walk-forward by year: for test year Y, learn on all years before Y which regimes each strategy LOST in
   (average per trade < 0, at least 15 trades), then in Y skip that strategy on those regimes' days.
   Compare with trading every day. Test years 2023, 2024, 2025, 2026 (Jan-Oct).
Strategies: the 8 paper strategies' 5-year backtest trades + the weekly iron-fly sale from research #1.
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ALGO = ROOT.parent / "dhan_algo"
sys.path.insert(0, str(ROOT))

TEST_YEARS = (2023, 2024, 2025, 2026)
MIN_TRADES = 15
FEATS = ["er20", "ret20", "dist50", "rv10", "vix", "vix_vs_250", "day_eff", "rng5_20"]
LABELS = {2: "trend", 0: "range"}


# ------------------------------------------------------------------ 1. daily features
def daily_features() -> pd.DataFrame:
    m = pd.read_parquet(ROOT / "data" / "minutes_NIFTY.parquet", columns=["ts", "day", "open", "high", "low", "close", "vix"])
    d = m.groupby("day").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
                             vix=("vix", "last"))
    r = d.c.pct_change()
    f = pd.DataFrame(index=d.index)
    net = (d.c - d.c.shift(20)).abs()
    path = d.c.diff().abs().rolling(20).sum()
    f["er20"] = net / path
    f["ret20"] = (d.c / d.c.shift(20) - 1) * 100
    f["dist50"] = (d.c / d.c.rolling(50).mean() - 1) * 100
    f["rv10"] = r.rolling(10).std() * math.sqrt(250) * 100
    f["vix"] = d.vix
    f["vix_vs_250"] = d.vix / d.vix.rolling(250, min_periods=60).median() - 1
    f["day_eff"] = ((d.c - d.o).abs() / (d.h - d.l).replace(0, np.nan)).rolling(10).mean()
    f["rng5_20"] = (d.h.rolling(5).max() - d.l.rolling(5).min()) / (d.h.rolling(20).max() - d.l.rolling(20).min())
    f = f.shift(1)                                   # everything known before today's open
    f["ret"] = r                                     # today's return (for the HMM fit only, never as a feature)
    f["lrange"] = np.log((d.h - d.l) / d.c)
    return f


# ------------------------------------------------------------------ 2. regime labellers (fit on training rows only)
def label_rule(f: pd.DataFrame, train: pd.Series) -> pd.Series:
    er_cut = f.er20[train].median()
    trend = np.where(f.er20 >= er_cut, "trend", "range")
    vol = np.where(f.vix_vs_250 > 0, "volatile", "calm")
    out = pd.Series([f"{a}-{b}" for a, b in zip(trend, vol)], index=f.index)
    return out.where(f.er20.notna() & f.vix_vs_250.notna())


def label_kmeans(f: pd.DataFrame, train: pd.Series, k: int = 4) -> pd.Series:
    from sklearn.cluster import KMeans
    ok = f[FEATS].notna().all(axis=1)
    X = f.loc[ok, FEATS]
    mu, sd = X[train[ok]].mean(), X[train[ok]].std()
    Z = (X - mu) / sd
    km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(Z[train[ok]])
    lab = pd.Series(np.nan, index=f.index, dtype=object)
    lab[ok] = [f"cluster{c}" for c in km.predict(Z)]
    return lab


def label_hmm(f: pd.DataFrame, train: pd.Series, n: int = 3) -> pd.Series:
    """Gaussian HMM on (return, log range); day t gets the state most likely given data up to t-1 (forward filter)."""
    from hmmlearn.hmm import GaussianHMM
    from scipy.stats import multivariate_normal
    ok = f[["ret", "lrange"]].notna().all(axis=1)
    X = f.loc[ok, ["ret", "lrange"]].to_numpy()
    tr = train[ok].to_numpy()
    model = GaussianHMM(n_components=n, covariance_type="full", n_iter=200, random_state=7).fit(X[tr])
    # order states by volatility so names are stable: calm < normal < stormy
    order = np.argsort(model.means_[:, 1])
    names = {order[0]: "calm", order[1]: "normal", order[2]: "stormy"} if n == 3 else {i: f"s{i}" for i in range(n)}
    em = np.column_stack([multivariate_normal(model.means_[i], model.covars_[i]).pdf(X) for i in range(n)])
    alpha = model.startprob_.copy()
    prior_next = []                                  # P(state today | data up to yesterday)
    for t in range(len(X)):
        pred = alpha @ model.transmat_
        prior_next.append(pred)
        post = pred * em[t]
        alpha = post / post.sum() if post.sum() > 0 else pred
    states = np.argmax(np.array(prior_next), axis=1)
    lab = pd.Series(np.nan, index=f.index, dtype=object)
    lab[ok] = [names[s] for s in states]
    return lab


LABELLERS = {"rule": label_rule, "kmeans": label_kmeans, "hmm": label_hmm}


# ------------------------------------------------------------------ strategies' daily results
def strategy_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(ALGO / "reports" / "bt_*_2021-11-01_2026-10-30_20261115_*_trades.csv")))
    t = pd.concat([pd.read_csv(f, parse_dates=["entry_ts"]).assign(strategy=Path(f).name[3:].split("_2021-11-01")[0])
                   for f in files], ignore_index=True)
    t["day"] = t.entry_ts.dt.normalize()
    t = t[["day", "strategy", "net_pnl"]]
    # research #1 weekly iron fly (NIFTY, 1 day before expiry at 15:10, wings +-400, hold) - the selling side
    vf = ROOT / "reports" / "vrp_trades_NIFTY.csv"
    if vf.exists():
        v = pd.read_csv(vf, parse_dates=["day"])
        v = v[(v.entry == "15:10") & (v.dte == 1) & (v.w == 8) & (v.stop == 0)]
        t = pd.concat([t, pd.DataFrame({"day": v.day, "strategy": "iron_fly_weekly", "net_pnl": v.net})], ignore_index=True)
    return t


# ------------------------------------------------------------------ 3. walk-forward test
def walk_forward(f: pd.DataFrame, t: pd.DataFrame, labeller) -> tuple[pd.DataFrame, dict]:
    rows, skip_log = [], {}
    for Y in TEST_YEARS:
        train = pd.Series(f.index.year < Y, index=f.index)
        lab = labeller(f, train)
        tt = t.merge(lab.rename("regime"), left_on="day", right_index=True, how="left")
        hist, test = tt[tt.day.dt.year < Y], tt[tt.day.dt.year == Y]
        stats = hist.groupby(["strategy", "regime"]).net_pnl.agg(["mean", "size"])
        bad = set(stats[(stats["mean"] < 0) & (stats["size"] >= MIN_TRADES)].index)
        skip_log[Y] = sorted(f"{s}:{r}" for s, r in bad)
        keep = ~test.apply(lambda r: (r.strategy, r.regime) in bad, axis=1)
        for s, g in test.assign(keep=keep).groupby("strategy"):
            rows.append({"year": Y, "strategy": s, "all": g.net_pnl.sum(), "kept": g.net_pnl[g.keep].sum(),
                         "n": len(g), "skipped": int((~g.keep).sum())})
    return pd.DataFrame(rows), skip_log


def regime_table(f: pd.DataFrame, t: pd.DataFrame) -> str:
    """Whole-period view (descriptive only): average result per trade by rule-regime, per strategy."""
    lab = label_rule(f, pd.Series(True, index=f.index))
    tt = t.merge(lab.rename("regime"), left_on="day", right_index=True, how="left")
    tab = tt.pivot_table(index="strategy", columns="regime", values="net_pnl", aggfunc="mean").round(0)
    days = lab.value_counts()
    return tab.to_string() + "\n\nshare of days: " + ", ".join(f"{k} {v / days.sum():.0%}" for k, v in days.items())


def run():
    REP = ROOT / "reports"
    f = daily_features()
    t = strategy_trades()
    L = ["RESEARCH #2: REGIME DETECTION", "Regime of each day known before it opens; walk-forward by year "
         "(learn on earlier years, test the next year).", "",
         "DESCRIPTIVE (all years, rule regimes): average Rs per trade",
         regime_table(f, t), ""]
    S, best = [], None
    for name, fn in LABELLERS.items():
        res, skips = walk_forward(f, t, fn)
        port = res.groupby("year")[["all", "kept"]].sum()
        tot_all, tot_kept = port["all"].sum(), port["kept"].sum()
        L += [f"===== {name}", "portfolio by test year (all trades -> with regime filter):",
              port.round(0).to_string(), f"TOTAL test years: {tot_all:,.0f} -> {tot_kept:,.0f}",
              "per strategy (all test years):",
              res.groupby("strategy")[["all", "kept", "n", "skipped"]].sum().round(0).to_string(),
              "skipped strategy:regime pairs by test year: " + "; ".join(f"{y}: {', '.join(v) or 'none'}" for y, v in skips.items()), ""]
        yrs = " · ".join(f"{y}: {a / 1000:+.0f}k→{k / 1000:+.0f}k" for y, (a, k) in port.iterrows())
        S.append(f"{'🧭' if name == 'rule' else '🔢' if name == 'kmeans' else '🎲'} {name}: total {tot_all / 1000:+.0f}k → "
                 f"{tot_kept / 1000:+.0f}k")
        S.append(f"   {yrs}")
        better_years = int((port["kept"] > port["all"]).sum())
        S.append(f"   filter helped in {better_years} of {len(port)} test years")
        if best is None or tot_kept - tot_all > best[1]:
            best = (name, tot_kept - tot_all, res)
    (REP / "regime.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L, best
