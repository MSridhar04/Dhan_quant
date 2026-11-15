"""Research #5: dealer positioning / gamma exposure (plain English: docs/10_gamma.md).

    python -m quant.research.run_gex      (unattended; Telegram)

Per minute, from the cached weekly options (ATM +-10 strikes: open interest, implied volatility):
  gamma of every option (Black-Scholes, its own IV, time left to expiry in trading time)
  gamma_ratio   = (call gamma x OI - put gamma x OI) / (call + put)     -1..+1   (standard "dealer" sign: hedgers are
                  assumed long the calls customers sold and short the puts customers bought; >0 = moves dampened)
  gamma_total_z = log(total gamma x OI) vs its own 60-day median         how heavy the positioning is
  pin_dist_bp   = index minus the strike with the most open interest (calls + puts), in bp
  call_wall_bp / put_wall_bp = distance to the biggest call OI above / put OI below
Tests (pick on <= 2024, check on 2025-26 unless stated):
  A  next-60-minute range (vs its normal size) and "busy" rate by gamma_ratio fifth  (do hedgers dampen moves?)
  B  expiry days: does the close end nearer the max-OI strike than the ATM strike seen at 12:00?  (pinning)
  C  does adding these features raise the busy-model AUC on 2026?
  D  paper-strategy trades by gamma_ratio at entry; walk-forward skip rule 2023-26
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ALGO = ROOT.parent / "dhan_algo"
sys.path.insert(0, str(ROOT))
IDS = {"NIFTY": "13", "SENSEX": "51"}
SPLIT = "2025-01-01"
GEX_FEATS = ["gamma_ratio", "gamma_total_z", "pin_dist_bp", "call_wall_bp", "put_wall_bp", "oi_skew"]


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _month(args) -> pd.DataFrame:
    uid, ym, dte_map = args
    parts = {}
    for side in ("CE", "PE"):
        f = ALGO / "data" / "rolling" / f"{uid}_WEEK0_{side}" / f"{ym}.parquet"
        f = f if f.exists() else f.with_name(f"{ym}.partial.parquet")
        if not f.exists():
            return pd.DataFrame()
        o = pd.read_parquet(f)
        if o.empty or not {"ts", "strike", "oi", "iv", "spot"} <= set(o.columns):
            return pd.DataFrame()
        parts[side] = o[["ts", "strike", "oi", "iv", "spot"]].drop_duplicates(["ts", "strike"], keep="last")
    m = parts["CE"].merge(parts["PE"], on=["ts", "strike"], suffixes=("_c", "_p"), how="outer")
    m["spot"] = m.spot_c.fillna(m.spot_p)
    m = m[m.spot > 0]
    day = m.ts.dt.normalize()
    dte = day.map(dte_map)
    mins_left = (15 * 60 + 30) - (m.ts.dt.hour * 60 + m.ts.dt.minute)
    T = ((dte.fillna(0) + mins_left.clip(lower=5) / 375) / 252).clip(lower=1e-5).to_numpy()
    S, K = m.spot.to_numpy(), m.strike.to_numpy()
    out = {}
    for side in ("c", "p"):
        iv = (m[f"iv_{side}"] / 100).to_numpy()
        iv = np.where((iv > 0.03) & (iv < 1.5), iv, np.nan)
        d1 = (np.log(S / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
        g = _norm_pdf(d1) / (S * iv * np.sqrt(T))
        out[side] = np.nan_to_num(g * m[f"oi_{side}"].fillna(0).to_numpy() * S * S * 0.01)
    m["gc"], m["gp"] = out["c"], out["p"]
    m["oi_c"], m["oi_p"] = m.oi_c.fillna(0), m.oi_p.fillna(0)
    m["oi_t"] = m.oi_c + m.oi_p
    g = m.groupby("ts")
    agg = pd.DataFrame({"gc": g.gc.sum(), "gp": g.gp.sum(), "spot": g.spot.first(),
                        "oic": g.oi_c.sum(), "oip": g.oi_p.sum()})
    agg["gamma_ratio"] = (agg.gc - agg.gp) / (agg.gc + agg.gp).replace(0, np.nan)
    agg["gamma_total"] = agg.gc + agg.gp
    agg["oi_skew"] = agg.oip / agg.oic.replace(0, np.nan)
    pin = m.loc[m.groupby("ts").oi_t.idxmax(), ["ts", "strike"]].set_index("ts").strike
    above = m[m.strike > m.spot]
    below = m[m.strike < m.spot]
    cw = above.loc[above.groupby("ts").oi_c.idxmax(), ["ts", "strike"]].set_index("ts").strike
    pw = below.loc[below.groupby("ts").oi_p.idxmax(), ["ts", "strike"]].set_index("ts").strike
    agg["pin_strike"] = pin
    agg["pin_dist_bp"] = (agg.spot / pin - 1) * 1e4
    agg["call_wall_bp"] = (cw / agg.spot - 1) * 1e4
    agg["put_wall_bp"] = (agg.spot / pw - 1) * 1e4
    return agg.reset_index()


def build_gex(name: str) -> pd.DataFrame:
    m = pd.read_parquet(ROOT / "data" / f"minutes_{name}.parquet", columns=["ts", "day", "dte"])
    dte_map = m.groupby("day").dte.first().to_dict()
    months = sorted(m.ts.dt.strftime("%Y-%m").unique())
    with ProcessPoolExecutor(max_workers=6) as ex:
        frames = [f for f in ex.map(_month, [(IDS[name], ym, dte_map) for ym in months]) if len(f)]
    g = pd.concat(frames, ignore_index=True).sort_values("ts")
    g["day"] = g.ts.dt.normalize()
    lt = np.log(g.gamma_total.replace(0, np.nan))
    daily = lt.groupby(g.day).median()
    med60 = daily.rolling(60, min_periods=20).median().shift(1)        # past days only
    g["gamma_total_z"] = lt - g.day.map(med60)
    g.to_parquet(ROOT / "data" / f"gex_{name}.parquet", index=False)
    return g


# ------------------------------------------------------------------ tests
def test_a(name: str, g: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    d = labels.merge(g[["ts"] + GEX_FEATS], on="ts", how="inner")
    d = d[d["mod"].isin([45, 165, 285]) & d.rng60.notna() & d.gamma_ratio.notna()]       # 10:00, 12:00, 14:00
    cuts = d.loc[d.ts < SPLIT, "gamma_ratio"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    d["fifth"] = np.digitize(d.gamma_ratio, cuts)
    q75 = d.loc[d.ts < SPLIT, "rng60"].quantile(0.75)
    d["busy"] = (d.rng60 > q75).astype(float)
    d["period"] = np.where(d.ts < SPLIT, "pick", "unseen")
    t = d.groupby(["fifth", "period"]).agg(range_vs_normal=("rng60", "mean"), busy=("busy", "mean"), n=("busy", "size")).unstack("period")
    return [f"A. {name}: next-60-min range (x normal) and busy rate by gamma_ratio fifth (0 = most negative/amplifying, "
            f"4 = most positive/dampening)", t.round(3).to_string(), ""]


def test_b(name: str, g: pd.DataFrame, minutes: pd.DataFrame) -> tuple[list[str], dict]:
    exp_days = minutes.groupby("day").dte.first()
    exp_days = exp_days[exp_days == 0].index
    step = 50 if name == "NIFTY" else 100
    rows = []
    for d in exp_days:
        noon = g[(g.day == d) & (g.ts.dt.strftime("%H:%M") == "12:00")]
        close = minutes[(minutes.day == d)].close
        if noon.empty or close.empty or pd.isna(noon.pin_strike.iat[0]):
            continue
        s12, pin = float(noon.spot.iat[0]), float(noon.pin_strike.iat[0])
        atm = round(s12 / step) * step
        c = float(close.iat[-1])
        rows.append({"day": d, "pin_err": abs(c - pin) / c * 1e4, "atm_err": abs(c - atm) / c * 1e4,
                     "pin_is_atm": pin == atm, "move": abs(c - s12) / s12 * 1e4})
    r = pd.DataFrame(rows)
    r["period"] = np.where(r.day < SPLIT, "pick", "unseen")
    diff = r[~r.pin_is_atm]
    t = diff.groupby("period").agg(days=("pin_err", "size"), close_to_pin_bp=("pin_err", "median"),
                                   close_to_atm_bp=("atm_err", "median"),
                                   pin_closer_pct=("pin_err", lambda s: float((s < diff.loc[s.index, "atm_err"]).mean() * 100)))
    lines = [f"B. {name}: expiry days where the max-OI strike at 12:00 differs from ATM ({len(diff)} of {len(r)} days)",
             t.round(1).to_string(), ""]
    return lines, {"n": len(diff), "pin_closer": float((diff.pin_err < diff.atm_err).mean() * 100) if len(diff) else float("nan")}


def test_c(name: str, g: pd.DataFrame) -> tuple[list[str], dict]:
    from sklearn.metrics import roc_auc_score
    from quant.model import PARAMS
    from quant.step1 import _feat_list, _fit, add_labels, breadth_features
    df = add_labels(pd.read_parquet(ROOT / "data" / f"minutes_{name}.parquet").merge(breadth_features(), on="ts", how="left"))
    df = df.merge(g[["ts"] + GEX_FEATS], on="ts", how="left")
    q75 = df.loc[df.ts < "2026-01-01", "rng30"].quantile(0.75)
    df["big30"] = np.where(df.rng30.notna(), (df.rng30 > q75).astype(float), np.nan)
    base_feats = [f for f in _feat_list(df) if f not in GEX_FEATS]
    out = {}
    for label, feats in (("without", base_feats), ("with gamma", base_feats + GEX_FEATS)):
        mdl, tr, te = _fit(df, feats, "big30", {**PARAMS, "objective": "binary", "num_class": 1, "num_threads": 10})
        out[label] = roc_auc_score(te.big30, mdl.predict(te[feats]))
    return [f"C. {name}: busy-30-min model AUC on 2026 - without gamma {out['without']:.4f} | with gamma {out['with gamma']:.4f}", ""], out


def test_d(gexs: dict) -> tuple[list[str], dict]:
    from quant.research.regime import strategy_trades
    t = strategy_trades()
    t = t[t.strategy != "iron_fly_weekly"].copy()
    files = sorted((ALGO / "reports").glob("bt_*_2021-11-01_2026-10-30_20261115_*_trades.csv"))
    ent = pd.concat([pd.read_csv(f, parse_dates=["entry_ts"]).assign(strategy=f.name[3:].split("_2021-11-01")[0]) for f in files])
    ent["idx"] = np.where(ent.symbol.str.startswith("SENSEX"), "SENSEX", "NIFTY")
    ent["feat_ts"] = ent.entry_ts.dt.floor("min") - pd.Timedelta(minutes=1)
    parts = []
    for n, e in ent.groupby("idx"):
        gg = gexs[n].set_index("ts")
        e = e.assign(gamma_ratio=gg.gamma_ratio.reindex(e.feat_ts).to_numpy())
        parts.append(e)
    e = pd.concat(parts).dropna(subset=["gamma_ratio"])
    e["sign"] = np.where(e.gamma_ratio > 0, "dampening (+)", "amplifying (-)")
    e["period"] = np.where(e.entry_ts < SPLIT, "pick", "unseen")
    tab = e.groupby(["strategy", "sign", "period"]).net_pnl.agg(["mean", "size"]).unstack(["period"]).round(0)
    rows = []
    for Y in (2023, 2024, 2025, 2026):
        hist, test = e[e.entry_ts.dt.year < Y], e[e.entry_ts.dt.year == Y]
        st = hist.groupby(["strategy", "sign"]).net_pnl.agg(["mean", "size"])
        bad = set(st[(st["mean"] < 0) & (st["size"] >= 15)].index)
        keep = ~pd.Series([(s, b) in bad for s, b in zip(test.strategy, test.sign)], index=test.index)
        rows.append({"year": Y, "all": test.net_pnl.sum(), "kept": test.net_pnl[keep].sum()})
    wf = pd.DataFrame(rows).set_index("year")
    return ["D. paper strategies by gamma sign at entry (avg Rs per trade, trades):", tab.to_string(), "",
            "walk-forward skip rule:", wf.round(0).to_string(), f"TOTAL {wf['all'].sum():,.0f} -> {wf['kept'].sum():,.0f}", ""], \
        {"all": float(wf["all"].sum()), "kept": float(wf["kept"].sum()), "helped": int((wf.kept > wf["all"]).sum())}


def run():
    from quant.step1 import add_labels
    REP = ROOT / "reports"
    L = ["RESEARCH #5: GAMMA EXPOSURE (dealer positioning) from cached weekly-option OI and IV", ""]
    S, gexs, res = [], {}, {}
    for name in ("NIFTY", "SENSEX"):
        g = build_gex(name)
        gexs[name] = g
        minutes = pd.read_parquet(ROOT / "data" / f"minutes_{name}.parquet",
                                  columns=["ts", "day", "mod", "close", "high", "low", "sigma60", "dte"])
        labels = add_labels(minutes.copy())
        L += test_a(name, g, labels)
        lb, rb = test_b(name, g, minutes)
        L += lb
        lc, rc = test_c(name, g)
        L += lc
        res[name] = (rb, rc)
    ld, rd = test_d(gexs)
    L += ld
    (REP / "gex.txt").write_text("\n".join(L), encoding="utf-8")
    for name, (rb, rc) in res.items():
        S += [f"📈 {name}", f"Expiry pinning: close nearer the max-OI strike than ATM on {rb['pin_closer']:.0f}% of "
                            f"{rb['n']} expiry days (50% = no effect)",
              f"Busy model AUC 2026: {rc['without']:.3f} → {rc['with gamma']:.3f} with gamma features", ""]
    S += [f"🧰 Strategies by gamma sign, skip rule 2023–26: {rd['all'] / 1000:+.0f}k → {rd['kept'] / 1000:+.0f}k "
          f"(helped {rd['helped']} of 4 years)"]
    return S, L
