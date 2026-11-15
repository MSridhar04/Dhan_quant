"""Step 1 improvements (2026-11-15): better questions, more features, a filter for the 7 paper strategies.
Train/tune on data up to 2025-12-31, test once on 2026 (same rule as prob/model.py).

    python -m prob.step1            -> reports/step1.txt + reports/step1_phone.txt

A. New features: market breadth from the 50 NIFTY stocks (share rising this minute / over 5 min / above their open)
   and the heavyweights' last moves (HDFCBANK, RELIANCE, ICICIBANK, INFY, BHARTIARTL, TCS). Stock data starts
   Oct 2023, so these are blank before that.
B. New questions, for H = 15 / 30 / 60 minutes:
   target-first  barrier = 1.0 x sigma60 x sqrt(H): does the index touch +barrier before -barrier (bullish), the
                 other way (bearish), or neither within H minutes?  -> 3 odds
   big move      is the index's range over the next H minutes in the top 25% (relative to its recent volatility,
                 cut-off learned on the training years)?  -> 1 probability (direction-free)
   Profit checks on 2026 with the real option prices: target-first strong readings -> buy ATM CE/PE, exit when the
   index touches either barrier or at H; big-move readings -> buy the ATM straddle (CE + PE), exit at H.
C. Filter for the 7 paper strategies (meta-labelling): at each 5-year backtest trade's entry, the minute features
   (direction-aligned for CE/PE) + strategy -> P(trade wins). Trained on trades before 2026; on 2026 compare
   "take every trade" with "skip trades the model rates poorly" (skip level chosen on the training trades).
"""
from __future__ import annotations

import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT.parent / "dhan_algo"
DATA, REP = ROOT / "data", ROOT / "reports"
sys.path.insert(0, str(ROOT))
from prob.model import DROP, PARAMS, TRAIN_END, TUNE_END, _opt, logloss  # noqa: E402

HEAVY = {"1333": "hdfcbank", "2885": "reliance", "4963": "icici", "1594": "infy", "10604": "bharti", "11536": "tcs"}
HORIZONS = (15, 30, 60)
BARRIER_K = 1.0
TICK = 0.05
P1 = {**PARAMS, "num_threads": 10}


# ------------------------------------------------------------------ A. breadth + heavyweights
def _stock(sid: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ALGO / "data" / "bars" / f"NSE_EQ_{sid}" / "*.parquet")))
    s = pd.concat([pd.read_parquet(f, columns=["ts", "open", "close"]) for f in files], ignore_index=True)
    s = s.drop_duplicates("ts").sort_values("ts")
    s["day"] = s.ts.dt.normalize()
    g = s.groupby("day")
    out = pd.DataFrame({"ts": s.ts,
                        "up1": (g.close.diff() > 0).astype(float),
                        "up5": (s.close / g.close.shift(5) - 1 > 0).astype(float),
                        "above_open": (s.close > g.open.transform("first")).astype(float),
                        "r1": (s.close / g.close.shift(1) - 1) * 1e4,
                        "r5": (s.close / g.close.shift(5) - 1) * 1e4})
    out["sid"] = sid
    return out


def breadth_features() -> pd.DataFrame:
    sids = [Path(p).name.split("_")[-1] for p in glob.glob(str(ALGO / "data" / "bars" / "NSE_EQ_*"))]
    with ProcessPoolExecutor(max_workers=6) as ex:
        frames = list(ex.map(_stock, sids))
    allst = pd.concat(frames, ignore_index=True)
    b = allst.groupby("ts").agg(br_up1=("up1", "mean"), br_up5=("up5", "mean"), br_above_open=("above_open", "mean"),
                                br_n=("up1", "size"))
    b = b[b.br_n >= 30].drop(columns="br_n")
    for c in ("br_up1", "br_up5"):
        b[f"{c}_chg5"] = b[c] - b[c].shift(5)
    hv = allst[allst.sid.isin(HEAVY)].pivot_table(index="ts", columns="sid", values=["r1", "r5"])
    hv.columns = [f"hv_{HEAVY[sid]}_{k}" for k, sid in hv.columns]
    hv["hv_avg_r1"] = hv[[c for c in hv.columns if c.endswith("_r1")]].mean(axis=1)
    hv["hv_avg_r5"] = hv[[c for c in hv.columns if c.endswith("_r5")]].mean(axis=1)
    return b.join(hv, how="outer").reset_index()


# ------------------------------------------------------------------ B. labels
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    day = df.day.to_numpy()
    c, hi, lo = df.close.to_numpy(), df.high.to_numpy(), df.low.to_numpy()
    sig = df.sigma60.to_numpy()
    n = len(df)
    for H in HORIZONS:
        bar = BARRIER_K * sig * np.sqrt(H)                       # basis points
        first_up = np.full(n, np.inf)
        first_dn = np.full(n, np.inf)
        fmax, fmin = np.full(n, -np.inf), np.full(n, np.inf)
        complete = np.zeros(n, dtype=bool)
        for j in range(1, H + 1):
            same = np.zeros(n, dtype=bool)
            same[:-j] = day[j:] == day[:-j]
            h_j = np.full(n, np.nan); h_j[:-j] = hi[j:]
            l_j = np.full(n, np.nan); l_j[:-j] = lo[j:]
            up = same & ((h_j / c - 1) * 1e4 >= bar)
            dn = same & ((l_j / c - 1) * 1e4 <= -bar)
            first_up = np.where(up & np.isinf(first_up), j, first_up)
            first_dn = np.where(dn & np.isinf(first_dn), j, first_dn)
            fmax = np.where(same, np.fmax(fmax, h_j), fmax)
            fmin = np.where(same, np.fmin(fmin, l_j), fmin)
            if j == H:
                complete = same
        lab = np.where(first_up < first_dn, 2.0, np.where(first_dn < first_up, 0.0, 1.0))
        ok = complete & np.isfinite(bar) & (bar > 0)
        lab[~ok] = np.nan
        df[f"tb{H}"] = lab
        df[f"tb{H}_hit"] = np.where(np.isfinite(np.minimum(first_up, first_dn)), np.minimum(first_up, first_dn), np.nan)
        ratio = ((fmax - fmin) / c * 1e4) / (sig * np.sqrt(H))
        ratio[~ok] = np.nan
        df[f"rng{H}"] = ratio
    return df


# ------------------------------------------------------------------ training helpers
def _fit(df, feats, ycol, params, extra_mask=None):
    m = df[ycol].notna() if extra_mask is None else (df[ycol].notna() & extra_mask)
    d = df[m]
    a, v = d[d.ts < TUNE_END], d[(d.ts >= TUNE_END) & (d.ts < TRAIN_END)]
    tr, te = d[d.ts < TRAIN_END], d[d.ts >= TRAIN_END]
    m0 = lgb.train(params, lgb.Dataset(a[feats], a[ycol]), 3000,
                   valid_sets=[lgb.Dataset(v[feats], v[ycol])], callbacks=[lgb.early_stopping(100, verbose=False)])
    n = max(int(m0.best_iteration * len(tr) / max(len(a), 1)), 50)
    model = lgb.train(params, lgb.Dataset(tr[feats], tr[ycol]), n)
    return model, tr, te


def _feat_list(df):
    skip = DROP | {c for c in df.columns if c.startswith(("tb", "rng", "big"))}
    return [c for c in df.columns if c not in skip]


def opt_px(name, side, ts, strike=None):
    uid = "13" if name == "NIFTY" else "51"
    o = _opt(uid, side, ts.strftime("%Y-%m"))
    if not len(o):
        return None, None
    try:
        row = o.loc[ts]
    except KeyError:
        return None, None
    if strike is None:
        atm = row[row.offset == 0]
        if atm.empty:
            return None, None
        return float(atm.close.iat[0]), float(atm.index[0])
    if strike in row.index:
        return float(row.loc[[strike], "close"].iat[-1]), strike     # duplicates in the cache: take the last
    return None, None


def charges():
    sys.path.insert(0, str(ALGO))
    from algo.config import load_settings
    from algo.core.events import Instrument, Side
    from algo.portfolio.charges import ChargeModel
    cm = ChargeModel(load_settings()["charges"])

    def rt(name, px_in, px_out, qty):
        inst = Instrument("x", "NSE_FNO" if name == "NIFTY" else "BSE_FNO", "x", "OPTIDX", lot_size=qty)
        return cm.compute(inst, Side.BUY, qty, px_in) + cm.compute(inst, Side.SELL, qty, px_out)
    return rt


def profit_tb(name, te, p, H, thr, rt):
    qty = 65 if name == "NIFTY" else 20
    out, busy = [], pd.Timestamp(0)
    hit = te[f"tb{H}_hit"].to_numpy()
    for i in np.where((p[:, 2] >= thr) | (p[:, 0] >= thr))[0]:
        ts = te.ts.iat[i]
        if ts < busy or te["mod"].iat[i] > 360 - H:
            continue
        side = "CE" if p[i, 2] >= thr else "PE"
        px_in, k = opt_px(name, side, ts)
        if px_in is None:
            continue
        hold = int(hit[i]) if np.isfinite(hit[i]) else H
        ex_ts = ts + pd.Timedelta(minutes=hold)
        px_out, _ = opt_px(name, side, ex_ts, k)
        if px_out is None:
            continue
        a, b = px_in + TICK, max(px_out - TICK, TICK)
        out.append((b - a) * qty - rt(name, a, b, qty))
        busy = ex_ts
    t = np.array(out)
    return {"trades": len(t), "net": float(t.sum()) if len(t) else 0.0, "win": float((t > 0).mean() * 100) if len(t) else np.nan}


def profit_straddle(name, te, pbig, H, cut, rt):
    qty = 65 if name == "NIFTY" else 20
    out, busy = [], pd.Timestamp(0)
    for i in np.where(pbig >= cut)[0]:
        ts = te.ts.iat[i]
        if ts < busy or te["mod"].iat[i] > 360 - H:
            continue
        ce, k = opt_px(name, "CE", ts)
        if ce is None:
            continue
        pe, _ = opt_px(name, "PE", ts, k)
        ex_ts = ts + pd.Timedelta(minutes=H)
        ce2, _ = opt_px(name, "CE", ex_ts, k)
        pe2, _ = opt_px(name, "PE", ex_ts, k)
        if None in (pe, ce2, pe2):
            continue
        net = 0.0
        for a, b in ((ce, ce2), (pe, pe2)):
            a, b = a + TICK, max(b - TICK, TICK)
            net += (b - a) * qty - rt(name, a, b, qty)
        out.append(net)
        busy = ex_ts
    t = np.array(out)
    return {"trades": len(t), "net": float(t.sum()) if len(t) else 0.0, "win": float((t > 0).mean() * 100) if len(t) else np.nan}


# ------------------------------------------------------------------ C. filter for the paper strategies
DIRECTIONAL = ["r1", "r3", "r5", "r15", "r30", "r60", "body", "from_open", "from_high", "from_low", "from_twap",
               "from_ema9", "from_ema21", "from_prev_close", "from_prev_high", "from_prev_low", "from_pivot", "gap",
               "prev_ret", "x_r1", "x_r5", "x_r15", "x_from_open", "x_gap5", "hv_avg_r1", "hv_avg_r5"]


def strategy_filter(minutes: dict[str, pd.DataFrame]) -> tuple[list[str], dict]:
    files = sorted(glob.glob(str(ALGO / "reports" / "bt_*_2021-11-01_2026-10-30_20261115_*_trades.csv")))
    rows = []
    for f in files:
        strat = Path(f).name[3:].split("_2021-11-01")[0]
        t = pd.read_csv(f, parse_dates=["entry_ts"])
        t["strategy"] = strat
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["idx"] = np.where(t.symbol.str.startswith("SENSEX"), "SENSEX", "NIFTY")
    t["sign"] = np.where(t.symbol.str.contains(" CE"), 1.0, -1.0)
    t["feat_ts"] = t.entry_ts.dt.floor("min") - pd.Timedelta(minutes=1)       # the candle that just closed
    parts = []
    for name, g in t.groupby("idx"):
        m = minutes[name].set_index("ts")
        cols = [c for c in _feat_list(minutes[name]) if c in m.columns]
        x = m.reindex(g.feat_ts)[cols].reset_index(drop=True)
        x.index = g.index
        parts.append(pd.concat([g, x], axis=1))
    t = pd.concat(parts).sort_values("entry_ts").reset_index(drop=True)
    for c in DIRECTIONAL:
        if c in t:
            t[c] = t[c] * t.sign                                      # "move in the trade's favour"
    t["strat_code"] = t.strategy.astype("category").cat.codes
    t["is_ce"] = (t.sign > 0).astype(int)
    feats = [c for c in _feat_list(minutes["NIFTY"]) if c in t.columns] + ["strat_code", "is_ce"]
    t["win"] = (t.net_pnl > 0).astype(int)
    tr, te = t[t.entry_ts < TRAIN_END], t[t.entry_ts >= TRAIN_END]
    pm = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 7, "min_data_in_leaf": 60,
          "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 20.0,
          "num_threads": 10, "verbose": -1, "seed": 7}
    # out-of-fold scores on the training trades (5 time blocks) to choose the skip level honestly
    oof = np.full(len(tr), np.nan)
    blocks = np.array_split(np.arange(len(tr)), 5)
    for b in blocks[1:]:
        fit_idx = np.arange(b[0])
        mdl = lgb.train(pm, lgb.Dataset(tr.iloc[fit_idx][feats], tr.iloc[fit_idx].win), 300)
        oof[b] = mdl.predict(tr.iloc[b][feats])
    ok = ~np.isnan(oof)
    best_q, best_gain = 0.0, 0.0
    for q in (0.1, 0.2, 0.3, 0.4):
        cut = np.nanquantile(oof[ok], q)
        kept = tr.net_pnl.to_numpy()[ok][oof[ok] >= cut].sum()
        gain = kept - tr.net_pnl.to_numpy()[ok].sum()
        if gain > best_gain:
            best_q, best_gain = q, gain
    model = lgb.train(pm, lgb.Dataset(tr[feats], tr.win), 300)
    p_tr = model.predict(tr[feats])
    p = model.predict(te[feats])
    L = ["C. FILTER FOR THE 7 PAPER STRATEGIES (5-year backtest trades)",
         f"training trades {len(tr)} (before 2026), test trades {len(te)} (2026)",
         f"skip level chosen on training (out-of-fold): skip the worst {int(best_q * 100)}% "
         f"(training gain {best_gain:,.0f})"]
    res = {"q": best_q}
    if best_q == 0:
        L.append("No skip level helped on the training trades: the filter is not used.")
        res.update(all=float(te.net_pnl.sum()), kept=float(te.net_pnl.sum()), auc=np.nan)
    else:
        cut = np.quantile(p_tr, best_q)
        keep = p >= cut
        auc = roc_auc_score(te.win, p) if te.win.nunique() > 1 else np.nan
        L += [f"2026: all trades {te.net_pnl.sum():,.0f} ({len(te)}) | kept {te.net_pnl[keep].sum():,.0f} ({keep.sum()}) | "
              f"skipped {te.net_pnl[~keep].sum():,.0f} ({(~keep).sum()})  AUC {auc:.3f}",
              "", "per strategy 2026 (all -> kept):"]
        for s, g in te.assign(keep=keep).groupby("strategy"):
            L.append(f"  {s:22s} {g.net_pnl.sum():>10,.0f} -> {g.net_pnl[g.keep].sum():>10,.0f}  "
                     f"(skipped {(~g.keep).sum()} of {len(g)})")
        res.update(all=float(te.net_pnl.sum()), kept=float(te.net_pnl[keep].sum()), auc=float(auc),
                   skipped=int((~keep).sum()), n=len(te))
    return L, res


# ------------------------------------------------------------------ run
def run():
    REP.mkdir(exist_ok=True)
    br = breadth_features()
    rt = charges()
    minutes, L, S = {}, ["STEP 1 RESULTS (train <= 2025, test 2026)", ""], []
    for name in ("NIFTY", "SENSEX"):
        df = pd.read_parquet(DATA / f"minutes_{name}.parquet").merge(br, on="ts", how="left")
        df = add_labels(df)
        minutes[name] = df
        feats = _feat_list(df)
        L += [f"========== {name} ({len(feats)} features, breadth/heavyweights from Oct 2023)"]
        S.append(f"📈 {name}")
        for H in HORIZONS:
            # target-first
            mdl, tr, te = _fit(df, feats, f"tb{H}", {**P1})
            p = mdl.predict(te[feats])
            y = te[f"tb{H}"].to_numpy().astype(int)
            prior = tr[f"tb{H}"].value_counts(normalize=True).reindex([0.0, 1.0, 2.0]).to_numpy()
            gain = (1 - logloss(p, y) / logloss(np.tile(prior, (len(y), 1)), y)) * 100
            strong = {thr: ((p[:, 2] >= thr) | (p[:, 0] >= thr)) for thr in (0.5, 0.55, 0.6)}
            sline = []
            for thr, m in strong.items():
                if m.sum():
                    pick = np.where(p[m, 2] >= thr, 2, 0)
                    sline.append(f">={int(thr * 100)}%: {m.sum()} ({(y[m] == pick).mean() * 100:.0f}% right, "
                                 f"{(y[m] == 2 - pick).mean() * 100:.0f}% wrong way)")
            prof = {thr: profit_tb(name, te.reset_index(drop=True), p, H, thr, rt) for thr in (0.5, 0.55, 0.6)}
            L += [f"-- target-first {H} min: normal odds bear {prior[0]:.0%} / neither {prior[1]:.0%} / bull {prior[2]:.0%}; "
                  f"model beats normal odds by {gain:.2f}%",
                  "   strong readings 2026: " + (" · ".join(sline) or "none"),
                  "   option profit 2026: " + " · ".join(f">={int(k * 100)}%: {v['trades']} tr {v['net']:,.0f}"
                                                         for k, v in prof.items())]
            best_tb = max(prof.items(), key=lambda kv: kv[1]["net"])
            # big move
            q75 = tr[f"rng{H}"].quantile(0.75)
            df[f"big{H}"] = np.where(df[f"rng{H}"].notna(), (df[f"rng{H}"] > q75).astype(float), np.nan)
            mdl2, tr2, te2 = _fit(df, feats, f"big{H}", {**P1, "objective": "binary", "num_class": 1})
            pb, pb_tr = mdl2.predict(te2[feats]), mdl2.predict(tr2[feats])
            yb = te2[f"big{H}"].to_numpy()
            auc = roc_auc_score(yb, pb)
            dec = pd.qcut(pb, 10, labels=False, duplicates="drop")
            top = yb[dec == dec.max()].mean() * 100
            bot = yb[dec == dec.min()].mean() * 100
            cuts = {f"top {int((1 - q) * 100)}%": np.quantile(pb_tr, q) for q in (0.9, 0.95)}
            sprof = {k: profit_straddle(name, te2.reset_index(drop=True), pb, H, c, rt) for k, c in cuts.items()}
            L += [f"-- big move {H} min: base {yb.mean() * 100:.0f}% · AUC {auc:.3f} · top-10% predicted -> {top:.0f}% "
                  f"big · bottom-10% -> {bot:.0f}% big",
                  "   straddle profit 2026: " + " · ".join(f"{k}: {v['trades']} tr {v['net']:,.0f}" for k, v in sprof.items())]
            best_st = max(sprof.items(), key=lambda kv: kv[1]["net"])
            S.append(f"{H}m target-first: +{gain:.1f}% vs normal · best opt {best_tb[1]['net'] / 1000:+.1f}k "
                     f"({best_tb[1]['trades']} tr)")
            S.append(f"{H}m big move: AUC {auc:.2f} · top10% {top:.0f}% vs base {yb.mean() * 100:.0f}% · "
                     f"straddle {best_st[1]['net'] / 1000:+.1f}k ({best_st[1]['trades']} tr)")
        L.append("")
        S.append("")
    fl, fres = strategy_filter(minutes)
    L += fl
    S.append("🧰 Filter for the 7 strategies (2026)")
    if fres["q"] == 0:
        S.append("No skip level helped on training data → not useful")
    else:
        S.append(f"All trades {fres['all'] / 1000:+.1f}k → kept {fres['kept'] / 1000:+.1f}k "
                 f"(skipped {fres['skipped']} of {fres['n']}, AUC {fres['auc']:.2f})")
    (REP / "step1.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    S, L = run()
    head = ["🧪 dhan_prob Step 1 · test 2026 (trained ≤2025)", "AUC: 0.50 = coin toss, 0.60+ = useful", ""]
    (REP / "step1_phone.txt").write_text("\n".join(head + S), encoding="utf-8")
    print("\n".join(head + S))
    print("\n".join(L))
