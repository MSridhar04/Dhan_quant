"""Research #6: pairs trading / statistical arbitrage on the 50 NIFTY stocks (plain English: docs/11_pairs.md).

    python -m quant.research.run_pairs      (unattended; Telegram)

Data: 1-min prices of the 50 NIFTY stocks, Oct 2023 - Oct 2026 (dhan_algo cache, read-only), Rs 1 lakh per leg,
dollar-neutral (buy Rs 1L of the laggard, short Rs 1L of the leader). Real charges + 2 bp slippage per fill.

A. MULTI-DAY (shorting overnight needs stock futures -> futures charges):
   each month: look back F trading days, keep pairs whose daily returns correlate >= 0.6, rank by how strongly
   the price gap mean-reverts (Engle-Granger: log A = a + b log B; ADF-style t-stat of the gap), trade the best N
   next month: z = (gap - formation mean) / formation std; enter at |z| >= z_in (next day 09:20 price), exit when z
   crosses 0, stop at |z| >= z_stop, or at month end.
B. INTRADAY (cash shorting allowed intraday -> equity-intraday charges):
   pairs = the N most correlated on 15-min returns over the last 20 days; each day from 10:00 the gap is measured
   from its 09:30 value; enter at |gap| >= z_in x (typical gap move), exit when it closes or at 15:15.
Settings chosen on 2024 test months; judged on 2025 - Oct 2026 (unseen); deflated Sharpe for the variants tried.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ALGO = ROOT.parent / "dhan_algo"
sys.path.insert(0, str(ROOT))
from quant.harness import Costs, deflated_sharpe  # noqa: E402

LEG = 100_000.0
SLIP = 0.0002
UNSEEN = "2025-01-01"
SYMS = json.loads((ROOT / "data" / "stock_symbols.json").read_text())


def _load(sid: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ALGO / "data" / "bars" / f"NSE_EQ_{sid}" / "*.parquet")))
    s = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in fs], ignore_index=True)
    s = s.drop_duplicates("ts").sort_values("ts")
    t = s.ts.dt.hour * 60 + s.ts.dt.minute
    s = s[(t >= 555) & (t <= 929)]
    return s.assign(sid=sid)


def prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(daily close, daily 09:20 price, 15-min close) wide frames, columns = symbols."""
    with ProcessPoolExecutor(max_workers=6) as ex:
        frames = list(ex.map(_load, list(SYMS)))
    a = pd.concat(frames, ignore_index=True)
    a["sym"] = a.sid.map(SYMS)
    a["day"] = a.ts.dt.normalize()
    close = a.groupby(["day", "sym"]).close.last().unstack()
    a920 = a[a.ts.dt.strftime("%H:%M") == "09:20"].groupby(["day", "sym"]).close.last().unstack()
    m15 = a[((a.ts.dt.hour * 60 + a.ts.dt.minute - 555 + 1) % 15) == 0]
    bars15 = m15.pivot_table(index="ts", columns="sym", values="close", aggfunc="last")
    days_ok = close.notna().sum(axis=1) >= 40
    return close[days_ok], a920.reindex(close.index[days_ok]), bars15


def _cost(costs: Costs, kind: str, px_buy_in, px_sell_in, px_sell_out, px_buy_out) -> float:
    """Charges for: buy leg in/out (long), sell leg in/out (short), Rs 1L each, equity-intraday or futures."""
    k = "FUTSTK" if kind == "futures" else "EQUITY"
    tot = 0.0
    for side, px in (("BUY", px_buy_in), ("SELL", px_sell_in), ("SELL", px_sell_out), ("BUY", px_buy_out)):
        q = max(int(LEG / px), 1)
        tot += _fill(costs, side, q, px, k)
    return tot + 4 * SLIP * LEG


def _fill(costs: Costs, side, qty, px, kind):
    inst = costs._I("x", "NSE_FNO" if kind == "FUTSTK" else "NSE_EQ", "x", kind, lot_size=1)
    return costs._cm.compute(inst, costs._S.BUY if side == "BUY" else costs._S.SELL, qty, px)


# ------------------------------------------------------------------ A. multi-day
def _adf_t(e: np.ndarray) -> float:
    de, lag = np.diff(e), e[:-1] - e[:-1].mean()
    g = (lag @ de) / (lag @ lag)
    resid = de - g * lag
    se = math.sqrt((resid @ resid) / max(len(de) - 2, 1) / (lag @ lag))
    return g / se if se > 0 else 0.0


def select_pairs(lp: pd.DataFrame, n: int) -> list[tuple]:
    r = lp.diff().dropna(how="all")
    corr = r.corr()
    cand = []
    cols = [c for c in lp.columns if lp[c].notna().all()]
    for a, b in itertools.combinations(cols, 2):
        if corr.loc[a, b] < 0.6:
            continue
        y, x = lp[a].to_numpy(), lp[b].to_numpy()
        beta = np.polyfit(x, y, 1)
        e = y - (beta[0] * x + beta[1])
        cand.append((_adf_t(e), a, b, beta[0], beta[1], e.mean(), e.std()))
    cand.sort()                                       # most negative t = strongest mean reversion
    return cand[:n]


def multi_day(close, open920, costs, F, N, z_in, z_stop) -> pd.DataFrame:
    lp = np.log(close)
    months = close.index.to_period("M").unique()
    trades = []
    for mth in months:
        days = close.index[close.index.to_period("M") == mth]
        hist = lp[lp.index < days[0]].tail(F)
        if len(hist) < F:
            continue
        for _, a, b, beta, alpha, mu, sd in select_pairs(hist, N):
            pos, entry = 0, None
            seq = list(days)
            for i, d in enumerate(seq):
                if d not in lp.index or pd.isna(lp.at[d, a]) or pd.isna(lp.at[d, b]):
                    continue
                z = (lp.at[d, a] - (beta * lp.at[d, b] + alpha) - mu) / sd
                last = i == len(seq) - 1
                nxt = seq[i + 1] if not last else None
                if pos == 0 and abs(z) >= z_in and nxt is not None:
                    pos = -1 if z > 0 else 1          # z>0: A rich -> short A, long B
                    entry = (nxt, open920.at[nxt, a], open920.at[nxt, b])
                elif pos != 0 and ((pos == -1 and z <= 0) or (pos == 1 and z >= 0) or (z_stop and abs(z) >= z_stop) or last):
                    ex_d = nxt if nxt is not None else d
                    pa = open920.at[ex_d, a] if nxt is not None else close.at[d, a]
                    pb = open920.at[ex_d, b] if nxt is not None else close.at[d, b]
                    ea, eb = entry[1], entry[2]
                    if any(pd.isna(v) for v in (pa, pb, ea, eb)):
                        pos = 0
                        continue
                    ra, rb = pa / ea - 1, pb / eb - 1
                    gross = LEG * (rb - ra) if pos == -1 else LEG * (ra - rb)
                    long_in, short_in, long_out, short_out = (eb, ea, pb, pa) if pos == -1 else (ea, eb, pa, pb)
                    net = gross - _cost(costs, "futures", long_in, short_in, short_out, long_out)
                    trades.append({"entry": entry[0], "exit": ex_d, "pair": f"{a}/{b}", "net": net, "gross": gross})
                    pos = 0
    return pd.DataFrame(trades)


# ------------------------------------------------------------------ B. intraday
def intraday(bars15, costs, N, z_in) -> pd.DataFrame:
    lp = np.log(bars15)
    day = lp.index.normalize()
    days = sorted(day.unique())
    trades = []
    for i in range(20, len(days)):
        d = days[i]
        hist = lp[(day >= days[i - 20]) & (day < d)]
        r = hist.diff()
        r = r[r.index.normalize() == r.index.normalize()]                # 15-min returns (overnight rows included, small)
        cols = [c for c in hist.columns if hist[c].notna().mean() > 0.95]
        if len(cols) < 20:
            continue
        corr = r[cols].corr()
        pairs = sorted(((corr.loc[a, b], a, b) for a, b in itertools.combinations(cols, 2)), reverse=True)[:N]
        today = lp[day == d]
        if len(today) < 20:
            continue
        for _, a, b in pairs:
            g_hist = (hist[a] - hist[b])
            g_day = g_hist.groupby(g_hist.index.normalize()).transform(lambda s: s - s.iloc[1] if len(s) > 1 else s * np.nan)
            sd = g_day.abs().quantile(0.68)
            if not np.isfinite(sd) or sd <= 0:
                continue
            g = today[a] - today[b]
            g = g - g.iloc[1]                                          # gap measured from 09:30
            pos, ent = 0, None
            for ts, v in g.items():
                hm = ts.strftime("%H:%M")
                if hm < "10:00":
                    continue
                if pos == 0 and hm < "14:30" and abs(v) >= z_in * sd:
                    pos, ent = (-1 if v > 0 else 1), ts
                elif pos != 0 and ((pos == -1 and v <= 0) or (pos == 1 and v >= 0) or hm >= "15:14"):
                    ea, eb = math.exp(today.at[ent, a]), math.exp(today.at[ent, b])
                    pa, pb = math.exp(today.at[ts, a]), math.exp(today.at[ts, b])
                    ra, rb = pa / ea - 1, pb / eb - 1
                    gross = LEG * (rb - ra) if pos == -1 else LEG * (ra - rb)
                    long_in, short_in, long_out, short_out = (eb, ea, pb, pa) if pos == -1 else (ea, eb, pa, pb)
                    trades.append({"entry": ent, "exit": ts, "pair": f"{a}/{b}", "gross": gross,
                                   "net": gross - _cost(costs, "equity", long_in, short_in, short_out, long_out)})
                    pos = 0
                    break                                              # one trade per pair per day
    return pd.DataFrame(trades)


# ------------------------------------------------------------------ evaluation
def _score(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n_pick": 0, "net_pick": 0.0, "n_unseen": 0, "net_unseen": 0.0}
    d = pd.to_datetime(t.entry)
    pick, uns = t[d < UNSEEN], t[d >= UNSEEN]
    return {"n_pick": len(pick), "net_pick": pick.net.sum(), "gross_pick": pick.gross.sum(),
            "n_unseen": len(uns), "net_unseen": uns.net.sum(), "gross_unseen": uns.gross.sum(),
            "win_unseen": float((uns.net > 0).mean() * 100) if len(uns) else float("nan")}


def run():
    REP = ROOT / "reports"
    costs = Costs()
    close, o920, b15 = prices()
    L = ["RESEARCH #6: PAIRS TRADING on the 50 NIFTY stocks (Rs 1L per leg, dollar-neutral, real costs + 2bp slippage)",
         f"Data {close.index.min():%b %Y} .. {close.index.max():%b %Y}. Settings picked on 2024, judged on 2025-Oct 2026.", ""]
    S = []
    for label, grid, fn in (
            ("A. MULTI-DAY (futures costs)",
             [dict(F=F, N=N, z_in=z, z_stop=s) for F in (60, 120) for N in (5, 10) for z in (1.5, 2.0, 2.5) for s in (3.5, 0)],
             lambda p: multi_day(close, o920, costs, **p)),
            ("B. INTRADAY (equity-intraday costs)",
             [dict(N=N, z_in=z) for N in (5, 10) for z in (1.5, 2.0, 3.0)],
             lambda p: intraday(b15, costs, **p))):
        rows = []
        for p in grid:
            t = fn(p)
            rows.append({**p, **_score(t), "_t": t})
        R = pd.DataFrame(rows)
        R = R[R.n_pick >= 15] if (R.n_pick >= 15).any() else R
        best = R.sort_values("net_pick", ascending=False).iloc[0]
        both = float(((R.net_pick > 0) & (R.net_unseen > 0)).mean() * 100)
        t = best["_t"]
        uns = t[pd.to_datetime(t.entry) >= UNSEEN] if len(t) else t
        daily = uns.groupby(pd.to_datetime(uns.exit).dt.normalize()).net.sum() if len(uns) else pd.Series(dtype=float)
        cal = daily.reindex(close.index[close.index >= UNSEEN], fill_value=0.0)
        sh = float(cal.mean() / cal.std() * math.sqrt(250)) if cal.std() > 0 else 0.0
        ds = deflated_sharpe(sh, len(R), len(cal)) if len(cal) > 20 else float("nan")
        eq = uns.net.cumsum() if len(uns) else pd.Series([0.0])
        dd = float((eq - eq.cummax()).min())
        params = {k: best[k] for k in grid[0]}
        top = uns.groupby("pair").net.agg(["size", "sum"]).sort_values("sum").round(0) if len(uns) else pd.DataFrame()
        L += [f"===== {label}: {len(R)} variants, {both:.0f}% made money in both periods",
              R.drop(columns=["_t"]).round(0).sort_values("net_pick", ascending=False).to_string(index=False), "",
              f"CHOSEN on 2024: {params}",
              f"  2024: {best.n_pick} trades, gross {best.get('gross_pick', 0):,.0f}, net {best.net_pick:,.0f}",
              f"  UNSEEN: {best.n_unseen} trades, gross {best.get('gross_unseen', 0):,.0f}, net {best.net_unseen:,.0f}, "
              f"win {best.get('win_unseen', float('nan')):.0f}%, worst drop {dd:,.0f}, Sharpe {sh:.2f}, luck-adjusted {ds:.0%}",
              "  unseen by pair (worst / best):", (pd.concat([top.head(4), top.tail(4)]).to_string() if len(top) else "none"), ""]
        S += [f"{'📆' if label.startswith('A') else '⏱️'} {label.split(' (')[0][3:].title()}",
              f"Chosen on 2024: " + ", ".join(f"{k} {v:g}" for k, v in params.items()),
              f"2024 {best.n_pick} tr net {best.net_pick / 1000:+.1f}k → unseen {best.n_unseen} tr "
              f"gross {best.get('gross_unseen', 0) / 1000:+.1f}k, net {best.net_unseen / 1000:+.1f}k (drop {dd / 1000:+.1f}k)",
              f"{both:.0f}% of {len(R)} variants won both periods · luck-adjusted {ds:.0%}", ""]
    (REP / "pairs.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L
