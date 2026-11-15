"""Stage 4: what the minutes look like, Nov 2021 - Dec 2025 only (2026 is kept for the test).

    python -m quant.analysis       -> reports/analysis.txt (+ phone summary returned by summary())
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, REP = ROOT / "data", ROOT / "reports"
END = "2026-01-01"


def odds(d: pd.DataFrame, by, y: str = "y5") -> pd.DataFrame:
    t = pd.crosstab(by, d[y], normalize="index").reindex(columns=[0.0, 1.0, 2.0], fill_value=0) * 100
    t.columns = ["bear%", "side%", "bull%"]
    t["minutes"] = by.value_counts().reindex(t.index)
    return t.round(1)


def one(name: str) -> tuple[list[str], dict]:
    d = pd.read_parquet(DATA / f"minutes_{name}.parquet")
    d = d[(d.ts < END) & d.y5.notna()]
    L = [f"===== {name}: {len(d):,} minutes, {d.day.nunique()} days (Nov 2021 - Dec 2025)"]
    base = d.y5.value_counts(normalize=True).reindex([0.0, 1.0, 2.0]) * 100
    L.append(f"Next 5 min, normal odds: bearish {base.iloc[0]:.1f}% / sideways {base.iloc[1]:.1f}% / bullish {base.iloc[2]:.1f}%")
    absmove = d.fwd5.abs()
    L.append(f"Typical 5-min move: median {absmove.median():.1f} bp, 90th pct {absmove.quantile(.9):.1f} bp "
             f"(1 bp = 0.01%; NIFTY 24,000 -> 1 bp = 2.4 pts)")
    slot = pd.cut(d["mod"], [-1, 15, 45, 105, 195, 285, 330, 375],
                  labels=["09:15-09:30", "09:30-10:00", "10:00-11:00", "11:00-12:30", "12:30-14:00", "14:00-14:45",
                          "14:45-15:30"])
    L += ["", "BY TIME OF DAY (next 5 min)", odds(d, slot).to_string(),
          "", "Average 5-min move size by time (bp):",
          d.assign(slot=slot).groupby("slot", observed=True).fwd5.apply(lambda s: s.abs().mean()).round(1).to_string()]
    L += ["", "EXPIRY DAY vs OTHER DAYS", odds(d, d.expiry_day.map({1: "expiry day", 0: "other day"})).to_string()]
    gap = pd.cut(d.gap, [-1e9, -50, -15, 15, 50, 1e9], labels=["gap down >0.5%", "gap down", "flat open",
                                                             "gap up", "gap up >0.5%"])
    L += ["", "BY OPENING GAP (first hour only)", odds(d[d["mod"] < 60], gap[d["mod"] < 60]).to_string()]
    mom = pd.qcut(d.r5, 5, labels=["fell most", "fell", "flat", "rose", "rose most"])
    L += ["", "AFTER THE LAST 5 MIN (momentum or snap-back?)", odds(d, mom).to_string()]
    pos = pd.cut(d.day_pos, [-0.01, 0.1, 0.3, 0.7, 0.9, 1.01], labels=["at day low", "low", "middle", "high", "at day high"])
    L += ["", "WHERE IN THE DAY'S RANGE", odds(d, pos).to_string()]
    by_year = odds(d, d.ts.dt.year)
    L += ["", "BY YEAR (stability)", by_year.to_string()]
    m = odds(d, mom)
    facts = {"base": base.round(0).tolist(),
             "after_rise_bull": float(m.loc["rose most", "bull%"]), "after_fall_bear": float(m.loc["fell most", "bear%"]),
             "open_move": float(d.assign(slot=slot).groupby("slot", observed=True).fwd5.apply(lambda s: s.abs().mean()).iloc[0]),
             "mid_move": float(d.assign(slot=slot).groupby("slot", observed=True).fwd5.apply(lambda s: s.abs().mean()).iloc[3]),
             "exp_side": float(odds(d, d.expiry_day.map({1: "e", 0: "o"})).loc["e", "side%"]),
             "oth_side": float(odds(d, d.expiry_day.map({1: "e", 0: "o"})).loc["o", "side%"])}
    return L + [""], facts


def run() -> tuple[str, str]:
    REP.mkdir(exist_ok=True)
    full, S = [], ["🔎 dhan_quant · Stage 4: how the minutes behave", "Nov 2021 – Dec 2025 (2026 kept for the test)", ""]
    for name in ("NIFTY", "SENSEX"):
        L, f = one(name)
        full += L
        b = f["base"]
        S += [f"{name}", f"Next 5 min normal odds: bear {b[0]:.0f}% · side {b[1]:.0f}% · bull {b[2]:.0f}%",
              f"After the biggest 5-min rises: bull {f['after_rise_bull']:.0f}%",
              f"After the biggest 5-min falls: bear {f['after_fall_bear']:.0f}%",
              f"Moves at 09:15-09:30 are {f['open_move'] / max(f['mid_move'], 1e-9):.1f}× midday size",
              f"Sideways on expiry days {f['exp_side']:.0f}% vs {f['oth_side']:.0f}% other days", ""]
    (REP / "analysis.txt").write_text("\n".join(full), encoding="utf-8")
    S.append("Next: model trained on ≤2025, tested on 2026.")
    return "\n".join(S), "\n".join(full)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    s, f = run()
    print(s, "\n\n", f)
