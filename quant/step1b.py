"""Step 1b (2026-11-15): two uses of the big-move model.

    python -m quant.step1b        -> reports/step1b.txt, reports/step1b_phone.txt (run via quant.run_step1b)

Models (no peeking): model A trained Nov 2021 - Dec 2024 -> predicts 2025 (used ONLY to choose settings);
model B trained up to Dec 2025 -> predicts 2026 (the test). Cut-offs are quantiles of each model's 2025 predictions.

1. SELL OPTIONS IN PREDICTED-QUIET WINDOWS: when P(big move in next H min) is in the bottom q of readings, sell an
   ATM iron fly (sell ATM CE + PE, buy CE w strikes up + PE w strikes down), exit after H minutes or when the loss
   reaches 1x the credit (variant), one position at a time, 09:30 .. 15:15-H. Baseline: the SAME trade at fixed
   times regardless of the model (every 30 min), to see whether "quiet" adds anything.
2. FILTER FOR THE 7 BUYING STRATEGIES: take a trade only when P(big move in next 30 min) at its entry minute is in
   the top (1 - q) of readings. q chosen on 2025 trades with model A, tested on 2026 trades with model B.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quant.model import PARAMS  # noqa: E402
from quant.step1 import ALGO, DATA, REP, TICK, _feat_list, add_labels, breadth_features, opt_px  # noqa: E402

STEP = {"NIFTY": 50, "SENSEX": 100}
QTY = {"NIFTY": 65, "SENSEX": 20}
PB = {**PARAMS, "objective": "binary", "num_class": 1, "num_threads": 10}


def _cost():
    sys.path.insert(0, str(ALGO))
    from algo.config import load_settings
    from algo.core.events import Instrument, Side
    from algo.portfolio.charges import ChargeModel
    cm = ChargeModel(load_settings()["charges"])

    def leg(name, side, qty, px):
        inst = Instrument("x", "NSE_FNO" if name == "NIFTY" else "BSE_FNO", "x", "OPTIDX", lot_size=qty)
        return cm.compute(inst, Side.BUY if side == "BUY" else Side.SELL, qty, px)
    return leg


def big_models(df: pd.DataFrame, feats: list[str], H: int):
    """-> (pred for 2025 from model A, pred for 2026 from model B), indexed like df."""
    q75 = df.loc[df.ts < "2025-01-01", f"rng{H}"].quantile(0.75)
    y = np.where(df[f"rng{H}"].notna(), (df[f"rng{H}"] > q75).astype(float), np.nan)
    ok = ~np.isnan(y)
    out = pd.Series(np.nan, index=df.index)
    for train_end, pred_from, pred_to in (("2025-01-01", "2025-01-01", "2026-01-01"),
                                          ("2026-01-01", "2026-01-01", "2099-01-01")):
        tr = ok & (df.ts < train_end).to_numpy()
        a = tr & (df.ts < pd.Timestamp(train_end) - pd.DateOffset(years=1)).to_numpy()
        v = tr & ~a
        m0 = lgb.train(PB, lgb.Dataset(df.loc[a, feats], y[a]), 2000,
                       valid_sets=[lgb.Dataset(df.loc[v, feats], y[v])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        n = max(int(m0.best_iteration * tr.sum() / a.sum()), 50)
        mdl = lgb.train(PB, lgb.Dataset(df.loc[tr, feats], y[tr]), n)
        sel = ((df.ts >= pred_from) & (df.ts < pred_to)).to_numpy()
        out[sel] = mdl.predict(df.loc[sel, feats])
    return out


# ------------------------------------------------------------------ 1. iron fly in quiet windows
def iron_fly(name, ts, H, w, stop, leg):
    ce, k = opt_px(name, "CE", ts)
    if ce is None:
        return None
    s = STEP[name]
    pe, _ = opt_px(name, "PE", ts, k)
    ceh, _ = opt_px(name, "CE", ts, k + w * s)
    peh, _ = opt_px(name, "PE", ts, k - w * s)
    if None in (pe, ceh, peh):
        return None
    q = QTY[name]
    s_ce, s_pe, b_ce, b_pe = ce - TICK, pe - TICK, ceh + TICK, peh + TICK
    credit = s_ce + s_pe - b_ce - b_pe
    if credit <= 0:
        return None
    exit_px = None
    for j in range(1, H + 1):
        t = ts + pd.Timedelta(minutes=j)
        px = [opt_px(name, "CE", t, k)[0], opt_px(name, "PE", t, k)[0],
              opt_px(name, "CE", t, k + w * s)[0], opt_px(name, "PE", t, k - w * s)[0]]
        if None in px:
            continue
        cost_close = (px[0] + TICK) + (px[1] + TICK) - (px[2] - TICK) - (px[3] - TICK)
        exit_px = px
        if (stop and cost_close - credit >= credit) or j == H:
            break
    if exit_px is None:
        return None
    a = [px for px in exit_px]
    pnl = ((s_ce - (a[0] + TICK)) + (s_pe - (a[1] + TICK)) + ((a[2] - TICK) - b_ce) + ((a[3] - TICK) - b_pe)) * q
    ch = (leg(name, "SELL", q, s_ce) + leg(name, "SELL", q, s_pe) + leg(name, "BUY", q, b_ce) + leg(name, "BUY", q, b_pe)
          + leg(name, "BUY", q, a[0] + TICK) + leg(name, "BUY", q, a[1] + TICK)
          + leg(name, "SELL", q, a[2] - TICK) + leg(name, "SELL", q, a[3] - TICK))
    return pnl - ch, w * s * q - credit * q


def run_fly(name, df, pred, H, w, stop, mask, leg):
    nets, margins, busy = [], [], pd.Timestamp(0)
    for i in np.where(mask)[0]:
        ts = df.ts.iat[i]
        if ts < busy or df["mod"].iat[i] < 15 or df["mod"].iat[i] > 360 - H:
            continue
        r = iron_fly(name, ts, H, w, stop, leg)
        if r is None:
            continue
        nets.append(r[0]); margins.append(r[1])
        busy = ts + pd.Timedelta(minutes=H)
    n = np.array(nets)
    if not len(n):
        return {"trades": 0, "net": 0.0, "avg": 0.0, "win": np.nan, "dd": 0.0, "margin": 0.0}
    eq = np.cumsum(n)
    return {"trades": len(n), "net": float(n.sum()), "avg": float(n.mean()), "win": float((n > 0).mean() * 100),
            "dd": float((eq - np.maximum.accumulate(np.concatenate([[0], eq]))[1:]).min()),
            "margin": float(np.median(margins))}


# ------------------------------------------------------------------ 2. filter for the buying strategies
def strategy_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(ALGO / "reports" / "bt_*_2021-11-01_2026-10-30_20261115_*_trades.csv")))
    t = pd.concat([pd.read_csv(f, parse_dates=["entry_ts"]).assign(strategy=Path(f).name[3:].split("_2021-11-01")[0])
                   for f in files], ignore_index=True)
    t["idx"] = np.where(t.symbol.str.startswith("SENSEX"), "SENSEX", "NIFTY")
    t["feat_ts"] = t.entry_ts.dt.floor("min") - pd.Timedelta(minutes=1)
    return t


def run():
    leg = _cost()
    br = breadth_features()
    trades = strategy_trades()
    trades["pbig"] = np.nan
    L = ["STEP 1b: big-move model -> (1) option selling in quiet windows, (2) filter for the 7 buying strategies",
         "Settings chosen on 2025 (model trained <= 2024), tested on 2026 (model trained <= 2025)", ""]
    S = []
    for name in ("NIFTY", "SENSEX"):
        df = add_labels(pd.read_parquet(DATA / f"minutes_{name}.parquet").merge(br, on="ts", how="left"))
        feats = _feat_list(df)
        L.append(f"========== {name}")
        S.append(f"📉 {name} · sell iron fly when 'quiet'")
        best, preds = None, {}
        for H in (30, 60):
            pred = preds[H] = big_models(df, feats, H)
            y25, y26 = (df.ts >= "2025-01-01") & (df.ts < "2026-01-01"), df.ts >= "2026-01-01"
            if H == 30:                                   # filter uses the 30-min model
                m = trades.idx == name
                p = pred.set_axis(df.ts)
                trades.loc[m, "pbig"] = p.reindex(trades.loc[m, "feat_ts"]).to_numpy()
                trades.loc[m, "pbig_cut_src"] = 0
                trades.attrs[f"{name}_p25"] = pred[y25].to_numpy()
                trades.attrs[f"{name}_p26"] = pred[y26].to_numpy()
            p25 = pred[y25].to_numpy()
            for q in (0.1, 0.2):
                cut25 = np.nanquantile(p25, q)
                cut26 = np.nanquantile(p25, q)            # same absolute level, set on 2025 readings
                for w in (2, 4):
                    for stop in (False, True):
                        r25 = run_fly(name, df[y25].reset_index(drop=True), None, H, w, stop,
                                      (pred[y25] <= cut25).to_numpy(), leg)
                        if best is None or r25["net"] > best[0]["net"]:
                            best = (r25, H, q, w, stop, cut26)
                        L.append(f"  2025 quiet{int(q * 100)} H{H} w{w} stop{int(stop)}: {r25['trades']} tr net "
                                 f"{r25['net']:,.0f} avg {r25['avg']:,.0f}")
        r25, H, q, w, stop, cut = best
        pred = preds[H]
        m26 = df.ts >= "2026-01-01"
        d26 = df[m26].reset_index(drop=True)
        test = run_fly(name, d26, None, H, w, stop, (pred[m26] <= cut).to_numpy(), leg)
        every = np.zeros(len(d26), dtype=bool)
        every[(d26["mod"].to_numpy() % 30) == 15] = True  # baseline: fixed times, ignore the model
        base = run_fly(name, d26, None, H, w, stop, every, leg)
        L += ["", f"  CHOSEN on 2025: H{H} quiet{int(q * 100)} w{w} stop{int(stop)} (2025 net {r25['net']:,.0f})",
              f"  2026 TEST quiet-only: {test['trades']} tr, net {test['net']:,.0f}, avg {test['avg']:,.0f}, win "
              f"{test['win']:.0f}%, drop {test['dd']:,.0f}, max-loss/trade ~{test['margin']:,.0f}",
              f"  2026 baseline fixed times: {base['trades']} tr, net {base['net']:,.0f}, avg {base['avg']:,.0f}, "
              f"win {base['win']:.0f}%", ""]
        S += [f"Chosen on 2025: hold {H}m, wings ±{w * STEP[name]} pts{', stop 1× credit' if stop else ''}",
              f"2026 quiet-only: {test['trades']} tr · {test['net'] / 1000:+.1f}k · avg ₹{test['avg']:,.0f} · "
              f"win {test['win']:.0f}% · drop {test['dd'] / 1000:+.1f}k",
              f"Same trade at fixed times: avg ₹{base['avg']:,.0f}/trade ({base['trades']} tr)",
              f"Max loss per fly ~₹{test['margin'] / 1000:.0f}k (broker margin higher)", ""]
    # ---------------- 2. filter
    t = trades.dropna(subset=["pbig"])
    t25 = t[(t.entry_ts >= "2025-01-01") & (t.entry_ts < "2026-01-01")]
    t26 = t[t.entry_ts >= "2026-01-01"]
    bestq, bestgain = 0.0, 0.0
    for q in (0.25, 0.5):
        cut = t25.groupby("idx").pbig.transform(lambda s: np.nanquantile(trades.attrs[f"{s.name}_p25"], q))
        gain = t25.net_pnl[t25.pbig >= cut].sum() - t25.net_pnl.sum()
        L.append(f"  filter 2025: keep top {int((1 - q) * 100)}% big-move odds -> gain {gain:,.0f}")
        if gain > bestgain:
            bestq, bestgain = q, gain
    S.append("🧰 Filter: buying strategies only when big-move odds are high")
    if bestq == 0:
        allp = t26.net_pnl.sum()
        L.append("  no cut helped on 2025: filter not used")
        S.append(f"No cut-off helped on 2025 → not useful (2026 all trades {allp / 1000:+.1f}k)")
    else:
        cut = t26.groupby("idx").pbig.transform(lambda s: np.nanquantile(trades.attrs[f"{s.name}_p25"], bestq))
        keep = t26.pbig >= cut
        L += [f"  CHOSEN on 2025: keep top {int((1 - bestq) * 100)}% (2025 gain {bestgain:,.0f})",
              f"  2026: all {t26.net_pnl.sum():,.0f} ({len(t26)}) -> kept {t26.net_pnl[keep].sum():,.0f} ({keep.sum()}) | "
              f"skipped {t26.net_pnl[~keep].sum():,.0f} ({(~keep).sum()})", "  per strategy 2026:"]
        for s, g in t26.assign(keep=keep).groupby("strategy"):
            L.append(f"    {s:22s} {g.net_pnl.sum():>10,.0f} -> {g.net_pnl[g.keep].sum():>10,.0f} (skipped {(~g.keep).sum()}/{len(g)})")
        S.append(f"2026: all {t26.net_pnl.sum() / 1000:+.1f}k → kept {t26.net_pnl[keep].sum() / 1000:+.1f}k "
                 f"(skipped {(~keep).sum()} of {len(t26)}, their P&L {t26.net_pnl[~keep].sum() / 1000:+.1f}k)")
    (REP / "step1b.txt").write_text("\n".join(L), encoding="utf-8")
    return S, L
