"""Quiet-DAY test (2026-11-15): at 09:30 decide whether to sell ONE iron fly and hold it to 15:15.

    python -m prob.run_quietday    -> reports/quietday.txt + Telegram

Trade: at 09:30 sell the ATM CE + PE and buy CE / PE w strikes away (w = 2/4/6), 1 lot, 1 tick slippage per leg,
dhan_algo charges on all 8 fills. Exit 15:15, or (variant) when the loss reaches 1x the credit received.
Baseline: sell it EVERY day. Model: one row per day with the 09:30 minute features (gap, first-15-min range, IV,
straddle price, days to expiry, breadth, yesterday ...) -> P(the fly makes money). Trade only on the top-rated days.
Settings (wing width, stop, how many days to trade) chosen on 2025 with a model trained Nov 2021 - Dec 2024;
the 2026 test uses a model trained up to Dec 2025. SENSEX options start May 2023, so its training is shorter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prob.step1 import ALGO, DATA, REP, TICK, _feat_list, breadth_features  # noqa: E402
from prob.step1b import QTY, STEP, _cost  # noqa: E402

ENTRY_MOD = 15                                       # the 09:30 candle (minute 15 of the session)
EXIT_HM = "15:15"
WINGS = (2, 4, 6)
PQ = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 7, "min_data_in_leaf": 25,
      "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 20.0,
      "num_threads": 10, "verbose": -1, "seed": 7}


def _month_series(uid: str, side: str, ym: str) -> dict:
    d = ALGO / "data" / "rolling" / f"{uid}_WEEK0_{side}"
    f = d / f"{ym}.parquet"
    f = f if f.exists() else d / f"{ym}.partial.parquet"
    if not f.exists():
        return {}
    o = pd.read_parquet(f)
    if o.empty or "strike" not in o:
        return {}
    o = o.drop_duplicates(["ts", "strike"], keep="last")
    return {k: g.set_index("ts").close.sort_index() for k, g in o.groupby("strike")}


def fly_days(name: str) -> pd.DataFrame:
    """P&L of the 09:30 iron fly on every day, for each wing width, with and without the 1x-credit stop."""
    uid, step, q = ("13" if name == "NIFTY" else "51"), STEP[name], QTY[name]
    leg = _cost()
    idx = pd.read_parquet(DATA / f"minutes_{name}.parquet", columns=["ts", "day", "mod"])
    entries = idx[idx["mod"] == ENTRY_MOD]
    rows, cache = [], {}
    for ts, day in zip(entries.ts, entries.day):
        ym = ts.strftime("%Y-%m")
        for side in ("CE", "PE"):
            if (side, ym) not in cache:
                cache[(side, ym)] = _month_series(uid, side, ym)
        ce_m, pe_m = cache[("CE", ym)], cache[("PE", ym)]
        if not ce_m or not pe_m:
            continue
        # ATM strike at 09:30 = the CE strike with offset 0 -> nearest strike to the spot
        spot_ce = [k for k, s in ce_m.items() if ts in s.index]
        if not spot_ce:
            continue
        atm = min(spot_ce, key=lambda k: abs(k - _spot(name, ts)))
        end = pd.Timestamp(f"{day:%Y-%m-%d} {EXIT_HM}")
        for w in WINGS:
            legs = [ce_m.get(atm), pe_m.get(atm), ce_m.get(atm + w * step), pe_m.get(atm - w * step)]
            if any(s is None for s in legs):
                continue
            path = pd.concat([s[(s.index >= ts) & (s.index <= end)] for s in legs], axis=1).ffill().dropna()
            if len(path) < 30 or path.index[0] != ts:
                continue
            s_ce, s_pe, b_ce, b_pe = path.iloc[0, 0] - TICK, path.iloc[0, 1] - TICK, path.iloc[0, 2] + TICK, path.iloc[0, 3] + TICK
            credit = s_ce + s_pe - b_ce - b_pe
            if credit <= 0:
                continue
            close_cost = (path.iloc[:, 0] + TICK) + (path.iloc[:, 1] + TICK) - (path.iloc[:, 2] - TICK) - (path.iloc[:, 3] - TICK)
            for stop in (False, True):
                ex = len(path) - 1
                if stop:
                    hit = np.where((close_cost.to_numpy()[1:] - credit) >= credit)[0]
                    ex = hit[0] + 1 if len(hit) else ex
                a = path.iloc[ex]
                pnl = ((s_ce - (a.iat[0] + TICK)) + (s_pe - (a.iat[1] + TICK)) + ((a.iat[2] - TICK) - b_ce)
                       + ((a.iat[3] - TICK) - b_pe)) * q
                ch = (leg(name, "SELL", q, s_ce) + leg(name, "SELL", q, s_pe) + leg(name, "BUY", q, b_ce)
                      + leg(name, "BUY", q, b_pe) + leg(name, "BUY", q, a.iat[0] + TICK) + leg(name, "BUY", q, a.iat[1] + TICK)
                      + leg(name, "SELL", q, a.iat[2] - TICK) + leg(name, "SELL", q, a.iat[3] - TICK))
                rows.append({"day": day, "ts": ts, "w": w, "stop": stop, "net": pnl - ch, "credit": credit * q,
                             "max_loss": (w * step - credit) * q})
    return pd.DataFrame(rows)


_spot_cache: dict = {}


def _spot(name: str, ts) -> float:
    if name not in _spot_cache:
        _spot_cache[name] = pd.read_parquet(DATA / f"minutes_{name}.parquet", columns=["ts", "close"]).set_index("ts").close
    return float(_spot_cache[name].get(ts, np.nan))


def day_features(name: str, br: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"minutes_{name}.parquet")
    df = df[df["mod"] == ENTRY_MOD].merge(br, on="ts", how="left")
    return df


def _fit(x, y, rounds=300):
    return lgb.train(PQ, lgb.Dataset(x, y), rounds)


def evaluate(name: str, br: pd.DataFrame, L: list, S: list) -> None:
    fl = fly_days(name)
    if fl.empty:
        L.append(f"{name}: no fly data"); return
    feats_df = day_features(name, br)
    feats = [c for c in _feat_list(feats_df) if c not in ("mod",)]
    yr = lambda d: pd.to_datetime(d).dt.year                                   # noqa: E731
    L.append(f"========== {name}: {fl.day.nunique()} days with option data ({fl.day.min():%b %Y} .. {fl.day.max():%b %Y})")
    L.append("BASELINE: sell the 09:30 iron fly EVERY day (net per year, Rs):")
    base = fl.groupby(["w", "stop", yr(fl.day)]).net.sum().unstack().round(0)
    L.append(base.to_string())
    best = None
    for (w, stop), g in fl.groupby(["w", "stop"]):
        d = g.merge(feats_df[["day"] + feats], on="day", how="left")
        d["win"] = (d.net > 0).astype(int)
        tr24, v25 = d[d.day < "2025-01-01"], d[(d.day >= "2025-01-01") & (d.day < "2026-01-01")]
        if len(tr24) < 80 or len(v25) < 50:
            continue
        mA = _fit(tr24[feats], tr24.win)
        p25 = mA.predict(v25[feats])
        for keep in (0.3, 0.5, 0.7):
            cut = np.quantile(p25, 1 - keep)
            sel = v25.net[p25 >= cut].sum()
            cand = (sel, w, stop, keep, cut)
            L.append(f"  2025 w{w} stop{int(stop)} trade top {int(keep * 100)}% days: {sel:,.0f} "
                     f"(every day {v25.net.sum():,.0f})")
            if best is None or sel > best[0]:
                best = cand
    if best is None:
        L.append("  not enough days to choose settings"); return
    sel25, w, stop, keep, _ = best
    g = fl[(fl.w == w) & (fl.stop == stop)].merge(feats_df[["day"] + feats], on="day", how="left")
    g["win"] = (g.net > 0).astype(int)
    tr25, te26 = g[g.day < "2026-01-01"], g[g.day >= "2026-01-01"]
    v25 = g[(g.day >= "2025-01-01") & (g.day < "2026-01-01")]
    mB = _fit(tr25[feats], tr25.win)
    cut = np.quantile(mB.predict(v25[feats]), 1 - keep)                      # level set on 2025 readings
    p26 = mB.predict(te26[feats])
    k = p26 >= cut
    eq = te26.net[k].cumsum()
    dd = float((eq - eq.cummax()).min()) if k.any() else 0.0
    every26 = te26.net.sum()
    L += ["", f"  CHOSEN on 2025: wings ±{w * STEP[name]} pts, {'stop at 1x credit' if stop else 'hold to 15:15'}, "
              f"trade top {int(keep * 100)}% days (2025 {sel25:,.0f})",
          f"  2026 TEST model days: {k.sum()} trades, net {te26.net[k].sum():,.0f}, win {(te26.net[k] > 0).mean() * 100:.0f}%, "
          f"drop {dd:,.0f} | every day: {len(te26)} trades, net {every26:,.0f} | skipped days net {te26.net[~k].sum():,.0f}",
          f"  max loss per fly ~Rs {te26.max_loss.median():,.0f}, credit ~Rs {te26.credit.median():,.0f}", ""]
    S += [f"📅 {name} (options from {fl.day.min():%b %Y})",
          f"Sell every day, 2026: {every26 / 1000:+.1f}k ({len(te26)} days)",
          f"Model picks (top {int(keep * 100)}%, wings ±{w * STEP[name]}{', stop 1×' if stop else ''}): "
          f"{te26.net[k].sum() / 1000:+.1f}k ({k.sum()} days, win {(te26.net[k] > 0).mean() * 100:.0f}%, drop {dd / 1000:+.1f}k)",
          f"Days it skipped made {te26.net[~k].sum() / 1000:+.1f}k",
          f"Every-day selling by year (wings ±{w * STEP[name]}): " + " · ".join(
              f"{c}: {v / 1000:+.0f}k" for c, v in base.loc[(w, stop)].items() if pd.notna(v)), ""]


def run():
    REP.mkdir(exist_ok=True)
    br = breadth_features()
    L = ["QUIET-DAY TEST: sell one iron fly at 09:30, hold to 15:15 (or stop at 1x credit)",
         "Settings chosen on 2025 (model trained <= 2024); test 2026 (model trained <= 2025)", ""]
    S = []
    for name in ("NIFTY", "SENSEX"):
        evaluate(name, br, L, S)
    (REP / "quietday.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L
