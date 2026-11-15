"""Research #3: position sizing by risk (plain English: docs/08_sizing.md).

    python -m quant.research.run_sizing      (unattended; Telegram)

Input: daily Rs results of the 8 paper strategies at 1 lot (5-year dhan_algo backtests). Every trade is kept; only
the SIZE changes. Sizes are recalculated at the start of each month from PAST days only (trailing 250 trading days,
at least 60 days of a strategy's own history, else 1 lot).

METHODS
  equal      1 lot each (today)
  inv_vol    lots proportional to 1 / (recent daily volatility of that strategy)
  risk_par   equal risk contribution using the recent covariance (allows for strategies moving together)
  vol_target equal lots, all scaled by (target / recent 20-day portfolio volatility), capped 0.5x..2x
  dd_brake   equal lots, a strategy is halved while it is more than 50% of its past worst drop below its peak,
             back to full at a new high
Fair comparison: fractional ("ideal") sizes are normalised so the AVERAGE number of lots stays 1 per strategy;
results are compared on Sharpe and return / worst drop (both independent of overall size), and also rescaled to
the same risk as 'equal'. A practical version rounds to whole lots 0..3.
Test period 2023-01 .. 2026-10 (all methods only use data before each month).
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
TEST_FROM = "2023-01-01"
LOOKBACK, MIN_HIST = 250, 60


def daily_pnl() -> pd.DataFrame:
    files = sorted(glob.glob(str(ALGO / "reports" / "bt_*_2021-11-01_2026-10-30_20261115_*_trades.csv")))
    t = pd.concat([pd.read_csv(f, parse_dates=["entry_ts"]).assign(strategy=Path(f).name[3:].split("_2021-11-01")[0])
                   for f in files], ignore_index=True)
    t["day"] = t.entry_ts.dt.normalize()
    m = pd.read_parquet(ROOT / "data" / "minutes_NIFTY.parquet", columns=["day"])
    cal = pd.DatetimeIndex(sorted(m.day.unique()))
    cal = cal[(cal >= "2021-11-01") & (cal <= "2026-10-30")]
    P = t.pivot_table(index="day", columns="strategy", values="net_pnl", aggfunc="sum").reindex(cal).fillna(0.0)
    first = t.groupby("strategy").day.min()                       # a strategy "exists" from its first trade
    for s in P.columns:
        P.loc[P.index < first[s], s] = np.nan
    return P


def weights(P: pd.DataFrame, method: str) -> pd.DataFrame:
    """Lots per strategy per day (fractional, average 1 across available strategies), set monthly from the past."""
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    months = P.index.to_period("M")
    for mth in months.unique():
        idx = P.index[months == mth]
        past = P[P.index < idx[0]].tail(LOOKBACK)
        avail = [s for s in P.columns if past[s].notna().sum() >= MIN_HIST]
        live = [s for s in P.columns if P.loc[idx, s].notna().any()]
        w = pd.Series(1.0, index=live)
        if method in ("inv_vol", "risk_par") and len(avail) >= 2:
            sd = past[avail].std().replace(0, np.nan)
            if method == "inv_vol":
                raw = 1 / sd
            else:
                raw = _risk_parity(past[avail].fillna(0.0).cov().to_numpy())
                raw = pd.Series(raw, index=avail)
            raw = raw / raw.mean()                               # average 1 lot among strategies with history
            for s in avail:
                if s in w.index and np.isfinite(raw[s]):
                    w[s] = raw[s]
        W.loc[idx, w.index] = w.values
    if method == "vol_target":
        port = P.fillna(0.0).sum(axis=1)
        vol20 = port.rolling(20).std().shift(1)
        target = port[port.index < TEST_FROM].std()             # long-run level from the pre-test years
        scale = (target / vol20).clip(0.5, 2.0).fillna(1.0)
        W = pd.DataFrame(1.0, index=P.index, columns=P.columns).where(P.notna()).mul(scale, axis=0)
    if method == "dd_brake":
        W = pd.DataFrame(1.0, index=P.index, columns=P.columns).where(P.notna())
        for s in P.columns:
            x = P[s].fillna(0.0).to_numpy()
            eq = peak = worst = 0.0
            for i in range(len(x)):
                W.iat[i, W.columns.get_loc(s)] = 0.5 if (worst < 0 and (eq - peak) < 0.5 * worst) else (
                    W.iat[i, W.columns.get_loc(s)])
                eq += x[i] * (W.iat[i, W.columns.get_loc(s)] if np.isfinite(W.iat[i, W.columns.get_loc(s)]) else 0)
                peak = max(peak, eq)
                worst = min(worst, eq - peak)
    return W


def _risk_parity(cov: np.ndarray, iters: int = 500) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        rc = w * (cov @ w)
        target = rc.sum() / n
        w = w * np.sqrt(target / np.maximum(rc, 1e-12))
        w = np.maximum(w, 1e-6)
        w /= w.sum()
    return w


def stats(d: pd.Series) -> dict:
    eq = d.cumsum()
    dd = float((eq - eq.cummax()).min())
    mon = d.groupby(d.index.to_period("M")).sum()
    sharpe = float(d.mean() / d.std() * math.sqrt(250)) if d.std() > 0 else 0.0
    return {"net": float(d.sum()), "sharpe": sharpe, "worst_drop": dd, "ret_dd": float(d.sum() / -dd) if dd < 0 else float("inf"),
            "worst_month": float(mon.min()), "losing_months": float((mon < 0).mean())}


def run():
    REP = ROOT / "reports"
    P = daily_pnl()
    test = P.index >= TEST_FROM
    base = (P.fillna(0.0) * 1.0).sum(axis=1)[test]
    base_sd = base.std()
    L = ["RESEARCH #3: POSITION SIZING BY RISK (8 paper strategies, every trade kept, only size changes)",
         f"Sizes set monthly from past data only; test {TEST_FROM[:7]} .. 2026-10.", ""]
    rows, S, ideal = [], [], {}
    for method in ("equal", "inv_vol", "risk_par", "vol_target", "dd_brake"):
        W = weights(P, method)
        for variant in ("ideal", "whole lots"):
            Wv = W if variant == "ideal" else W.round().clip(0, 3)
            d = (P.fillna(0.0) * Wv.fillna(0.0)).sum(axis=1)[test]
            st = stats(d)
            same_risk = d * (base_sd / d.std()) if d.std() > 0 else d       # rescaled to equal's risk
            st2 = stats(same_risk)
            avg_lots = float(Wv[test].mean(axis=1).mean()) if variant == "whole lots" else 1.0
            rows.append({"method": method, "variant": variant, **{k: round(v, 2) for k, v in st.items()},
                         "net_same_risk": round(st2["net"]), "drop_same_risk": round(st2["worst_drop"]),
                         "avg_lots": round(avg_lots, 2)})
            if variant == "ideal":
                ideal[method] = (st, st2, W)
    R = pd.DataFrame(rows)
    last = {m: w[test].iloc[-1].round(2).to_dict() for m, (_, _, w) in ideal.items() if m in ("inv_vol", "risk_par")}
    L += [R.to_string(index=False), "", "latest suggested lots (ideal, average 1):"]
    L += [f"  {m}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if pd.notna(v)) for m, d in last.items()]
    (REP / "sizing.txt").write_text("\n".join(L), encoding="utf-8")
    e = ideal["equal"][0]
    S.append(f"Baseline 1 lot each: {e['net'] / 1000:+.0f}k · worst drop {e['worst_drop'] / 1000:+.0f}k · "
             f"return/drop {e['ret_dd']:.1f} · Sharpe {e['sharpe']:.2f}")
    names = {"inv_vol": "Inverse volatility", "risk_par": "Risk parity", "vol_target": "Volatility targeting",
             "dd_brake": "Drawdown brake"}
    for m in ("inv_vol", "risk_par", "vol_target", "dd_brake"):
        st, st2, _ = ideal[m]
        S.append(f"{names[m]}: return/drop {st['ret_dd']:.1f} (vs {e['ret_dd']:.1f}) · Sharpe {st['sharpe']:.2f} · "
                 f"at same risk {st2['net'] / 1000:+.0f}k, drop {st2['worst_drop'] / 1000:+.0f}k")
    return S, L, R, last
