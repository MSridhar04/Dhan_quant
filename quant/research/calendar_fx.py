"""Research #4: calendar and event effects (plain English: docs/09_calendar.md).

    python -m quant.research.run_calendar     (unattended; Telegram)

For the 8 paper strategies (5-year backtest trades) and the index itself:
  weekday            Mon .. Fri
  expiry             days to the strategy's index weekly expiry: 0 (expiry day), 1, 2, 3+
  month              first 3 / last 3 trading days of the month, other days
  event              RBI policy decision day, Union Budget day, 2024 election result day, the Indian day after a
                     US Fed (FOMC) decision, normal day
1. Descriptive: average Rs per trade by bucket, picking years (<= 2024) vs unseen 2025-26 - an effect counts only if
   it has the same sign in both, with at least 10 trades in each.
2. Walk-forward by year (2023, 2024, 2025, 2026): skip a strategy on a bucket where it lost on average in EARLIER
   years (>= 15 trades); compare with trading every day.
Event dates: RBI 2021-2025 and FOMC 2021-2026 from the public schedules; RBI 2026 left out (dates not confirmed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from quant.research.regime import strategy_trades  # noqa: E402

SPLIT = "2025-01-01"
TEST_YEARS = (2023, 2024, 2025, 2026)
MIN_WF, MIN_DESC = 15, 10

RBI = ["2021-12-08", "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
       "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
       "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
       "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-01", "2025-12-05"]
BUDGET = ["2022-02-01", "2023-02-01", "2024-02-01", "2024-07-23"]          # weekday sessions only
ELECTION = ["2024-06-04"]
FOMC = ["2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02",
        "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01",
        "2023-12-13", "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
        "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29",
        "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28"]


def day_table() -> pd.DataFrame:
    """One row per trading day: weekday, month position, events, NIFTY/SENSEX days to expiry, NIFTY daily range."""
    rows = {}
    for name in ("NIFTY", "SENSEX"):
        m = pd.read_parquet(ROOT / "data" / f"minutes_{name}.parquet", columns=["day", "dte", "high", "low", "close", "open"])
        g = m.groupby("day")
        rows[f"dte_{name}"] = g.dte.first()
        if name == "NIFTY":
            rows["range_bp"] = (g.high.max() - g.low.min()) / g.close.last() * 1e4
            rows["move_bp"] = (g.close.last() / g.open.first() - 1).abs() * 1e4
    d = pd.DataFrame(rows)
    d["weekday"] = d.index.day_name().str[:3]
    ym = d.index.to_period("M")
    pos = d.groupby(ym).cumcount()
    left = d.groupby(ym).cumcount(ascending=False)
    d["month"] = np.where(pos < 3, "first 3 days", np.where(left < 3, "last 3 days", "mid-month"))
    days = list(d.index)
    nxt = {}
    for f in pd.to_datetime(FOMC):                    # Indian trading day after the US decision (announced at night IST)
        later = [x for x in days if x > f]
        if later:
            nxt[later[0]] = True
    ev = pd.Series("normal day", index=d.index)
    ev[d.index.isin(pd.to_datetime(list(nxt)))] = "day after Fed"
    ev[d.index.isin(pd.to_datetime(RBI))] = "RBI policy"
    ev[d.index.isin(pd.to_datetime(BUDGET))] = "Budget"
    ev[d.index.isin(pd.to_datetime(ELECTION))] = "Election result"
    d["event"] = ev
    return d


def expiry_bucket(x) -> str:
    return "unknown" if pd.isna(x) else ("expiry day" if x == 0 else f"{int(x)} day before" if x <= 2 else "3+ days before")


def tagged_trades(days: pd.DataFrame) -> pd.DataFrame:
    t = strategy_trades()
    t = t[t.strategy != "iron_fly_weekly"].merge(days, left_on="day", right_index=True, how="left")
    sx = t.strategy.str.contains("sensex")
    t["expiry"] = np.where(sx, t.dte_SENSEX, t.dte_NIFTY)
    t["expiry"] = t.expiry.map(expiry_bucket)
    return t


def descriptive(t: pd.DataFrame, family: str) -> tuple[str, list[str]]:
    t = t.assign(period=np.where(t.day < SPLIT, "pick", "unseen"))
    g = t.groupby(["strategy", family, "period"]).net_pnl.agg(["mean", "size"]).unstack("period")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.dropna()
    keep = (g.size_pick >= MIN_DESC) & (g.size_unseen >= MIN_DESC)
    same = keep & (np.sign(g.mean_pick) == np.sign(g.mean_unseen))
    found = []
    for (s, b), r in g[same].iterrows():
        if abs(r.mean_pick) >= 150 and abs(r.mean_unseen) >= 150:   # meaningful in Rs, both periods
            found.append(f"{s} on {b}: {r.mean_pick:+.0f} → {r.mean_unseen:+.0f} per trade "
                         f"({int(r.size_pick)}/{int(r.size_unseen)} trades)")
    return g.round(0).to_string(), found


def walk_forward(t: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = []
    for Y in TEST_YEARS:
        hist, test = t[t.day.dt.year < Y], t[t.day.dt.year == Y]
        st = hist.groupby(["strategy", family]).net_pnl.agg(["mean", "size"])
        bad = set(st[(st["mean"] < 0) & (st["size"] >= MIN_WF)].index)
        keep = ~pd.Series([(s, b) in bad for s, b in zip(test.strategy, test[family])], index=test.index)
        rows.append({"year": Y, "all": test.net_pnl.sum(), "kept": test.net_pnl[keep].sum(), "skipped": int((~keep).sum())})
    return pd.DataFrame(rows).set_index("year")


def run():
    REP = ROOT / "reports"
    days = day_table()
    t = tagged_trades(days)
    L = ["RESEARCH #4: CALENDAR AND EVENT EFFECTS (8 paper strategies, 5-year backtest trades)",
         "Picking years <= 2024, unseen 2025-26; walk-forward skip test by year 2023-2026.",
         "Event dates: RBI 2021-2025, Budget (weekday sessions), 2024 election result, day after US Fed decision 2021-2026.", ""]
    mk = days.groupby("event")[["range_bp", "move_bp"]].mean().round(0)
    mk["days"] = days.event.value_counts()
    L += ["INDEX CONTEXT: NIFTY average day range / open-to-close move (bp) by event", mk.to_string(), ""]
    for fam in ("weekday", "month"):
        L += [f"by {fam}:", days.groupby(fam)[["range_bp", "move_bp"]].mean().round(0).to_string(), ""]
    S, consistent, wf_lines = [], [], []
    for fam in ("weekday", "expiry", "month", "event"):
        tab, found = descriptive(t, fam)
        wf = walk_forward(t, fam)
        L += [f"===== {fam}", "average Rs per trade (pick = <=2024, unseen = 2025-26):", tab, "",
              "walk-forward (skip buckets that lost in earlier years):", wf.round(0).to_string(),
              f"TOTAL {wf['all'].sum():,.0f} -> {wf['kept'].sum():,.0f}", "consistent effects: " + ("; ".join(found) or "none"), ""]
        consistent += [f"[{fam}] {f}" for f in found]
        wf_lines.append(f"{fam}: {wf['all'].sum() / 1000:+.0f}k → {wf['kept'].sum() / 1000:+.0f}k "
                        f"(helped {int((wf['kept'] > wf['all']).sum())} of 4 years)")
    ev = mk["range_bp"]
    S += ["📅 Index moves (NIFTY average day range):",
          " · ".join(f"{k} {v:.0f}bp" for k, v in ev.sort_values(ascending=False).items()), "",
          "⏭️ Skip-on-bad-days rule (walk-forward 2023–26):"] + [f"• {x}" for x in wf_lines]
    S += ["", f"🔎 Effects that held in BOTH periods (≥₹150/trade, ≥10 trades each): {len(consistent)}"]
    S += [f"• {c}" for c in consistent[:12]]
    (REP / "calendar.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L
