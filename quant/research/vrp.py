"""Research #1: volatility risk premium (VRP) - plain English: docs/06_vrp.md.

    python -m quant.research.run_vrp      (unattended; sends results to Telegram)

TRADE (1 lot): sell the ATM call + put of the nearest weekly expiry, buy protection w strikes away on both sides
(iron fly = capped loss). Enter k trading days before expiry (k = 1/2/3) at 15:10 (or 09:30), hold to expiry-day
15:15, optional stop when the loss reaches m x the credit. Real charges on all 8 fills + 1 tick slippage per fill.
Legs are priced from the 1-min option cache; when a leg leaves the cached strike window its price is never
allowed below its intrinsic value (otherwise a deep in-the-money short leg would "freeze" and hide the loss).

SIGNALS known at entry:
  implied move  = ATM straddle / index                     (what options charge for the move to expiry)
  realized move = recent daily volatility x sqrt(days left) (what the market has actually been doing)
  vrp_ratio     = implied / realized   (> 1 = premium is rich)
  India VIX vs its 20-day average, VIX change since yesterday
FILTERS tried: none (sell every week) / vrp_ratio in the top 50% or 30% of training readings / + VIX not rising.
Choose on training (NIFTY Nov 2021 - Dec 2024, SENSEX May 2023 - Dec 2024), judge on 2025 - Oct 2026 (unseen),
deflated Sharpe for the number of variants tried, Monte Carlo of the chosen one.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ALGO = ROOT.parent / "dhan_algo"
sys.path.insert(0, str(ROOT))
from quant.harness import Costs, deflated_sharpe  # noqa: E402

IDX = {"NIFTY": {"sid": "13", "step": 50, "qty": 65, "start": "2021-11-01"},
       "SENSEX": {"sid": "51", "step": 100, "qty": 20, "start": "2023-05-01"}}
SPLIT = "2025-01-01"
END = "2026-10-31"
TICK = 0.05
WINGS = (4, 6, 8)
STOPS = (0.0, 1.0, 2.0)            # 0 = hold to expiry
DTES = (1, 2, 3)
ENTRIES = ("15:10", "09:30")
EXIT_HM = "15:15"


# ------------------------------------------------------------------ data
class OptionCache:
    """Weekly-option minutes for one index, loaded month by month when needed (keeps memory small)."""

    def __init__(self, sid: str, keep: int = 4):
        self.sid, self.keep, self.months = sid, keep, {}

    def _month(self, side: str, ym: str) -> dict:
        key = (side, ym)
        if key not in self.months:
            d = ALGO / "data" / "rolling" / f"{self.sid}_WEEK0_{side}"
            f = d / f"{ym}.parquet"
            f = f if f.exists() else d / f"{ym}.partial.parquet"
            out = {}
            if f.exists():
                o = pd.read_parquet(f)
                if not o.empty and {"ts", "close", "strike"} <= set(o.columns):
                    o = o[["ts", "strike", "close"]].drop_duplicates(["ts", "strike"], keep="last")
                    out = {float(k): g.set_index("ts").close.sort_index() for k, g in o.groupby("strike")}
            self.months[key] = out
            while len(self.months) > self.keep * 2:
                self.months.pop(next(iter(self.months)))
        return self.months[key]

    def get(self, side: str, k: float, start, end):
        """Close series of one strike between start and end (may span two months)."""
        yms = pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M")
        parts = [self._month(side, str(p)).get(float(k)) for p in yms]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        s = pd.concat(parts)
        return s[~s.index.duplicated(keep="last")].sort_index()


def index_minutes(name: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / f"minutes_{name}.parquet",
                        columns=["ts", "day", "mod", "close", "dte", "straddle_bp", "vix", "vix_vs_20d", "vix_vs_prev"])
    return d[(d.ts >= IDX[name]["start"]) & (d.ts <= END)].reset_index(drop=True)


def realized_daily_bp(m: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Std of daily close-to-close moves over the previous `lookback` days, in bp (known before today)."""
    daily = m.groupby("day").close.last()
    r = (daily / daily.shift() - 1) * 1e4
    return r.rolling(lookback).std().shift()                  # shift: yesterday's value is known today


# ------------------------------------------------------------------ one trade
def fly_path(name, cache, spot: pd.Series, entry_ts, exit_ts, k, w):
    """Minute path of [short CE, short PE, long CE, long PE] prices from entry to exit, intrinsic-floored."""
    step = IDX[name]["step"]
    legs = [("CE", k), ("PE", k), ("CE", k + w * step), ("PE", k - w * step)]
    s = spot[(spot.index >= entry_ts) & (spot.index <= exit_ts)]
    if len(s) < 5 or s.index[0] != entry_ts:
        return None
    cols = []
    for side, kk in legs:
        ser = cache.get(side, kk, entry_ts, exit_ts)
        if ser is None:
            return None
        ser = ser[(ser.index >= entry_ts) & (ser.index <= exit_ts)]
        if ser.empty or ser.index[0] != entry_ts:
            return None                                       # need a real price at entry
        x = ser.reindex(s.index).ffill()
        intrinsic = (s - kk).clip(lower=0) if side == "CE" else (kk - s).clip(lower=0)
        cols.append(np.maximum(x.to_numpy(), intrinsic.to_numpy()))
    P = np.column_stack(cols)
    # a call (or put) spread is never worth more than its width: if a protective leg left the cached window its
    # stale price must not make the short leg look worse than that limit (found on the 4 Jun 2024 gap)
    width = w * step
    P[:, 2] = np.maximum(P[:, 2], P[:, 0] - width)
    P[:, 3] = np.maximum(P[:, 3], P[:, 1] - width)
    return s.index, P


def run_trade(name, cache, costs, spot, entry_ts, exit_ts, w, stop):
    step, q = IDX[name]["step"], IDX[name]["qty"]
    s0 = float(spot.get(entry_ts, np.nan))
    if not np.isfinite(s0):
        return None
    k = round(s0 / step) * step
    pp = fly_path(name, cache, spot, entry_ts, exit_ts, k, w)
    if pp is None:
        return None
    idx, P = pp
    sell = P[0, :2] - TICK
    buy = P[0, 2:] + TICK
    credit = sell.sum() - buy.sum()
    if credit <= 0:
        return None
    close_cost = (P[:, 0] + TICK) + (P[:, 1] + TICK) - (P[:, 2] - TICK) - (P[:, 3] - TICK)
    ex = len(P) - 1
    if stop:
        hit = np.where(close_cost[1:] - credit >= stop * credit)[0]
        if len(hit):
            ex = hit[0] + 1
    a = P[ex]
    pnl = ((sell[0] - (a[0] + TICK)) + (sell[1] - (a[1] + TICK)) + ((a[2] - TICK) - buy[0]) + ((a[3] - TICK) - buy[1])) * q
    ch = (costs.fill(name, "SELL", q, sell[0]) + costs.fill(name, "SELL", q, sell[1]) + costs.fill(name, "BUY", q, buy[0])
          + costs.fill(name, "BUY", q, buy[1]) + costs.fill(name, "BUY", q, a[0] + TICK) + costs.fill(name, "BUY", q, a[1] + TICK)
          + costs.fill(name, "SELL", q, max(a[2] - TICK, TICK)) + costs.fill(name, "SELL", q, max(a[3] - TICK, TICK)))
    return {"net": pnl - ch, "credit": credit * q, "max_loss": w * step * q - credit * q,
            "exit": idx[ex], "stopped": ex < len(P) - 1}


# ------------------------------------------------------------------ all trades for all variants
def build_trades(name: str) -> pd.DataFrame:
    m = index_minutes(name)
    cache = OptionCache(IDX[name]["sid"])
    costs = Costs()
    spot = m.set_index("ts").close
    rv = realized_daily_bp(m)
    days = m.groupby("day").agg(dte=("dte", "first")).reset_index()
    rows = []
    for d, dte in zip(days.day, days.dte):
        if dte not in DTES:
            continue
        exp_day = days.day[(days.day > d) & (days.dte == 0)]
        if exp_day.empty:
            continue
        exit_ts = pd.Timestamp(f"{exp_day.iloc[0]:%Y-%m-%d} {EXIT_HM}")
        for hm in ENTRIES:
            entry_ts = pd.Timestamp(f"{d:%Y-%m-%d} {hm}")
            r = m[m.ts == entry_ts]
            if r.empty:
                continue
            r = r.iloc[0]
            left = dte + (0.3 if hm == "15:10" else 1.0)            # rough trading days of movement left
            rv_move = rv.get(d, np.nan) * math.sqrt(left)
            sig = {"day": d, "entry": hm, "dte": int(dte), "implied_bp": r.straddle_bp, "rv_move_bp": rv_move,
                   "vrp_ratio": r.straddle_bp / rv_move if rv_move and np.isfinite(rv_move) else np.nan,
                   "vix": r.vix, "vix_vs_20d": r.vix_vs_20d, "vix_vs_prev": r.vix_vs_prev}
            for w in WINGS:
                for stop in STOPS:
                    t = run_trade(name, cache, costs, spot, entry_ts, exit_ts, w, stop)
                    if t:
                        rows.append({**sig, "w": w, "stop": stop, **t})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ choose on training, judge on unseen
def evaluate(name: str, t: pd.DataFrame) -> tuple[list[str], list[str], dict]:
    t = t.copy()
    t["tune"] = t.day < SPLIT
    L, S = [f"========== {name} ({IDX[name]['start'][:7]} .. {END[:7]}): {t.day.nunique()} entry days"], []
    base = t[(t.stop == 0)]
    L.append("BASELINE sell every week, hold to expiry (net Rs by year):")
    L.append(base.groupby(["entry", "dte", "w", t.day.dt.year]).net.sum().unstack().round(0).to_string())
    res = []
    for (hm, dte, w, stop), g in t.groupby(["entry", "dte", "w", "stop"]):
        tr = g[g.tune]
        for filt in ("none", "vrp50", "vrp30", "vrp50+vix"):
            if filt == "none":
                keep = np.ones(len(g), bool)
            else:
                q = 0.5 if filt.startswith("vrp50") else 0.7
                cut = tr.vrp_ratio.quantile(q)
                keep = (g.vrp_ratio >= cut).to_numpy()
                if filt.endswith("+vix"):
                    keep = keep & (g.vix_vs_prev <= 0).to_numpy()     # (pandas arrays are read-only)
            k = g[keep]
            res.append({"entry": hm, "dte": dte, "w": w, "stop": stop, "filter": filt,
                        "tune_n": int(k.tune.sum()), "tune_net": k.net[k.tune].sum(),
                        "test_n": int((~k.tune).sum()), "test_net": k.net[~k.tune].sum(),
                        "max_loss": float(g.max_loss.median()), "_keep": keep, "_g": g})
    R = pd.DataFrame(res)
    R = R[R.tune_n >= 20]
    both = ((R.tune_net > 0) & (R.test_net > 0)).mean() * 100
    best = R.sort_values("tune_net", ascending=False).iloc[0]
    g, keep = best["_g"], best["_keep"]
    k = g[keep].sort_values("day")
    daily = k.groupby("day").net.sum()
    cal = pd.Series(0.0, index=pd.Index(sorted(t.day.unique())))
    cal.loc[daily.index] = daily
    test = cal[cal.index >= SPLIT]
    sharpe = float(test.mean() / test.std() * math.sqrt(250)) if test.std() > 0 else 0.0
    ds = deflated_sharpe(sharpe, len(R), len(test), skew=float(test.skew()), kurt=float(test.kurt() + 3)) if len(test) > 10 else float("nan")
    tk = k[~k.tune]
    eq = tk.net.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    every = R[(R.entry == best.entry) & (R.dte == best.dte) & (R.w == best.w) & (R.stop == best.stop) & (R["filter"] == "none")].iloc[0]
    mc = monte_carlo(cal[cal.index >= SPLIT].to_numpy())
    stop_txt = "hold to expiry" if not best.stop else f"stop at {best.stop:g}× credit"
    L += ["", f"variants tried {len(R)}; {both:.0f}% made money in BOTH periods",
          f"CHOSEN on training: enter {best.entry} with {best.dte} day(s) to expiry, wings ±{best.w * IDX[name]['step']} pts, "
          f"{stop_txt}, filter {best['filter']}",
          f"  training: {best.tune_n} trades, net {best.tune_net:,.0f}",
          f"  UNSEEN 2025-26: {best.test_n} trades, net {best.test_net:,.0f}, win {(tk.net > 0).mean() * 100:.0f}%, "
          f"worst drop {dd:,.0f}, max loss per trade ~{best.max_loss:,.0f}",
          f"  same setup selling EVERY week: unseen net {every.test_net:,.0f} ({every.test_n} trades)",
          f"  luck check: Sharpe {sharpe:.2f}, probability real after {len(R)} tries = {ds:.0%}",
          f"  Monte Carlo (unseen days reshuffled): middle year {mc['median']:,.0f}, bad year {mc['p5']:,.0f}, "
          f"P(losing year) {mc['p_loss']:.0%}", "", "TOP 8 by training net:",
          R.drop(columns=["_keep", "_g"]).sort_values("tune_net", ascending=False).head(8).round(0).to_string(index=False), ""]
    flt = {"none": "every week", "vrp50": "premium rich (top 50%)", "vrp30": "premium very rich (top 30%)",
           "vrp50+vix": "rich + VIX not rising"}[best["filter"]]
    S += [f"📉 {name}", f"Chosen: {best.dte}d before expiry at {best.entry}, wings ±{best.w * IDX[name]['step']}, {stop_txt}, {flt}",
          f"Training {best.tune_n} tr {best.tune_net / 1000:+.1f}k → unseen {best.test_n} tr {best.test_net / 1000:+.1f}k "
          f"(win {(tk.net > 0).mean() * 100:.0f}%, drop {dd / 1000:+.1f}k)",
          f"Every week instead: unseen {every.test_net / 1000:+.1f}k",
          f"{both:.0f}% of {len(R)} variants won both periods · luck-adjusted chance real {ds:.0%}",
          f"Max loss per trade ~₹{best.max_loss / 1000:.0f}k (broker margin higher)", ""]
    return L, S, {"best": best.drop(["_keep", "_g"]).to_dict(), "both_pct": both, "ds": ds}


def monte_carlo(daily: np.ndarray, n: int = 10_000, year: int = 250, seed: int = 7) -> dict:
    """Reshuffle daily results into n simulated years (whole-week blocks keep shocks together)."""
    if len(daily) < 20:
        return {"median": float("nan"), "p5": float("nan"), "p_loss": float("nan")}
    rng = np.random.default_rng(seed)
    blk = 5
    starts = rng.integers(0, len(daily) - blk, size=(n, year // blk + 1))
    idx = (starts[:, :, None] + np.arange(blk)).reshape(n, -1)[:, :year]
    tot = daily[idx].sum(axis=1)
    return {"median": float(np.median(tot)), "p5": float(np.percentile(tot, 5)), "p_loss": float((tot < 0).mean())}


def run():
    REP = ROOT / "reports"
    L = ["RESEARCH #1: VOLATILITY RISK PREMIUM (sell a hedged ATM straddle before weekly expiry, hold to expiry)",
         "Chosen on data up to Dec 2024; judged on 2025 - Oct 2026 (unseen). 1 lot, real costs, intrinsic-floored legs.", ""]
    S = []
    import os
    for name in ("NIFTY", "SENSEX"):
        f = REP / f"vrp_trades_{name}.csv"
        if os.environ.get("VRP_REUSE") == "1" and f.exists():     # retry after an evaluation-only fix
            t = pd.read_csv(f, parse_dates=["day", "exit"])
        else:
            t = build_trades(name)
            t.to_csv(f, index=False)
        l, s, _ = evaluate(name, t)
        L += l
        S += s
    (REP / "vrp.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L
