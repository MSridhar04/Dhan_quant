"""Train on data up to 2025-12-31, freeze, then test on 2026 (never seen). LightGBM multiclass -> P(bearish),
P(sideways), P(bullish) for the next h minutes (h = 1, 5, 15; 5 is the main one).

    python -m prob.model            -> reports/verdict.txt, reports/calibration_<IDX>_h5.png, models/*.txt

Steps per index and horizon:
  1. choose the number of trees: train Nov 2021 - Dec 2024, early-stop on 2025 (2026 untouched)
  2. final model: same settings, trained on everything up to 2025-12-31, then frozen
  3. 2026 test: log-loss vs two baselines (overall odds; odds for that time of day), calibration by 10% bins,
     strong readings (>= 55/60/65/70%) and how often they came true, month by month stability
  4. profit check (h = 5): buy the ATM option on each strong reading at that minute's close (+1 tick),
     sell the same strike 5 minutes later (-1 tick), dhan_algo's charge model, one trade at a time
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT.parent / "dhan_algo"
DATA, REP, MOD = ROOT / "data", ROOT / "reports", ROOT / "models"
TUNE_END, TRAIN_END = "2025-01-01", "2026-01-01"
DROP = {"ts", "open", "high", "low", "close", "day", "fwd1", "fwd5", "fwd15", "y1", "y5", "y15"}
PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.03, "num_leaves": 31,
          "min_data_in_leaf": 800, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
          "lambda_l2": 10.0, "num_threads": 12, "verbose": -1, "seed": 7}
CLASSES = ["bearish", "sideways", "bullish"]


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y.astype(int)], 1e-9, 1))))


def tod_baseline(train: pd.DataFrame, test: pd.DataFrame, ycol: str) -> np.ndarray:
    """Odds for that 15-minute slot of the day, learned on the training years: the honest 'no model' answer."""
    slot = lambda d: (d["mod"] // 15).astype(int)                         # noqa: E731
    tab = pd.crosstab(slot(train), train[ycol], normalize="index").reindex(columns=[0.0, 1.0, 2.0], fill_value=0)
    return tab.reindex(slot(test)).fillna(tab.mean()).to_numpy()


def fit(name: str, h: int):
    df = pd.read_parquet(DATA / f"minutes_{name}.parquet")
    y = f"y{h}"
    df = df[df[y].notna()].reset_index(drop=True)
    feats = [c for c in df.columns if c not in DROP]
    a, v = df[df.ts < TUNE_END], df[(df.ts >= TUNE_END) & (df.ts < TRAIN_END)]
    tr, te = df[df.ts < TRAIN_END], df[df.ts >= TRAIN_END].reset_index(drop=True)
    d_a = lgb.Dataset(a[feats], a[y]); d_v = lgb.Dataset(v[feats], v[y], reference=d_a)
    m0 = lgb.train(PARAMS, d_a, 3000, valid_sets=[d_v], callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = max(m0.best_iteration, 50)
    p_val = m0.predict(v[feats], num_iteration=m0.best_iteration)
    model = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr[y]), int(n_trees * len(tr) / len(a)))
    MOD.mkdir(exist_ok=True)
    model.save_model(str(MOD / f"{name}_h{h}.txt"))
    p = model.predict(te[feats])
    return df, feats, tr, v, te, p, p_val, model


def calib_table(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for k, cname in ((2, "bullish"), (0, "bearish")):
        bins = np.clip((p[:, k] * 10).astype(int), 0, 9)
        for b in range(10):
            m = bins == b
            if m.sum() >= 200:
                rows.append({"class": cname, "said": f"{b * 10}-{b * 10 + 10}%", "n": int(m.sum()),
                             "said_avg": p[m, k].mean() * 100, "happened": (y[m] == k).mean() * 100})
    return pd.DataFrame(rows)


def strong(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in (0.5, 0.55, 0.6, 0.65, 0.7):
        for k, cname in ((2, "bullish"), (0, "bearish")):
            m = p[:, k] >= thr
            rows.append({"reading": f"{cname} >= {int(thr * 100)}%", "count": int(m.sum()),
                         "per_day": m.sum() / max(1, len(y) / 370),
                         "came_true_%": (y[m] == k).mean() * 100 if m.sum() else np.nan,
                         "opposite_%": (y[m] == 2 - k).mean() * 100 if m.sum() else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ profit check (options)
_opt_cache: dict = {}


def _opt(uid: str, side: str, ym: str) -> pd.DataFrame:
    k = (uid, side, ym)
    if k not in _opt_cache:
        d = ALGO / "data" / "rolling" / f"{uid}_WEEK0_{side}"
        f = d / f"{ym}.parquet"
        f = f if f.exists() else d / f"{ym}.partial.parquet"
        o = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        _opt_cache[k] = o.set_index(["ts", "strike"]).sort_index() if len(o) and "strike" in o else o
    return _opt_cache[k]


def profit_check(name: str, te: pd.DataFrame, p: np.ndarray, thr: float, hold: int = 5) -> dict:
    sys.path.insert(0, str(ALGO))
    from algo.config import load_settings
    from algo.core.events import Instrument, Side
    from algo.portfolio.charges import ChargeModel
    cm = ChargeModel(load_settings()["charges"])
    uid, qty = ("13", 65) if name == "NIFTY" else ("51", 20)
    seg = "NSE_FNO" if name == "NIFTY" else "BSE_FNO"
    tick = 0.05
    trades, busy_until = [], pd.Timestamp(0)
    for i in np.where((p[:, 2] >= thr) | (p[:, 0] >= thr))[0]:
        ts = te.ts.iat[i]
        if ts < busy_until or te["mod"].iat[i] > 360 - hold:               # one at a time; no entries after 15:15-hold
            continue
        side = "CE" if p[i, 2] >= thr else "PE"
        o = _opt(uid, side, ts.strftime("%Y-%m"))
        if not len(o):
            continue
        try:
            row = o.loc[ts]
        except KeyError:
            continue
        atm = row[row.offset == 0]
        if atm.empty:
            continue
        strike = atm.index[0]
        exit_ts = ts + pd.Timedelta(minutes=hold)
        try:
            px_in = float(atm.close.iat[0]) + tick
            px_out = float(o.loc[(exit_ts, strike), "close"]) - tick
        except KeyError:
            continue
        inst = Instrument(f"{name} {strike:.0f} {side}", seg, "x", "OPTIDX", lot_size=qty)
        ch = cm.compute(inst, Side.BUY, qty, px_in) + cm.compute(inst, Side.SELL, qty, px_out)
        trades.append({"ts": ts, "side": side, "net": (px_out - px_in) * qty - ch, "gross": (px_out - px_in) * qty})
        busy_until = exit_ts
    t = pd.DataFrame(trades)
    if t.empty:
        return {"thr": thr, "trades": 0, "net": 0.0, "gross": 0.0, "win_%": np.nan}
    return {"thr": thr, "trades": len(t), "net": t.net.sum(), "gross": t.gross.sum(), "win_%": (t.net > 0).mean() * 100}


def run() -> str:
    REP.mkdir(exist_ok=True)
    L = ["dhan_prob: probability of the next move after each 1-min candle",
         "Trained on Nov 2021 - Dec 2025 only. TEST = 1 Jan 2026 onwards (never seen, model frozen before).", ""]
    summary = {}
    for name in ("NIFTY", "SENSEX"):
        for h in (5, 1, 15):
            df, feats, tr, v, te, p, p_val, model = fit(name, h)
            y = te[f"y{h}"].to_numpy().astype(int)
            prior = tr[f"y{h}"].value_counts(normalize=True).reindex([0.0, 1.0, 2.0]).to_numpy()
            ll_m, ll_0 = logloss(p, y), logloss(np.tile(prior, (len(y), 1)), y)
            ll_t = logloss(tod_baseline(tr, te, f"y{h}"), y)
            gain = (1 - ll_m / ll_t) * 100
            L += [f"===== {name}  next {h} min  (test {te.ts.min():%d %b %Y} .. {te.ts.max():%d %b %Y}, "
                  f"{len(te):,} minutes)",
                  f"normal odds (training): bear {prior[0]:.0%} / side {prior[1]:.0%} / bull {prior[2]:.0%}",
                  f"log-loss: model {ll_m:.4f} | overall odds {ll_0:.4f} | time-of-day odds {ll_t:.4f}  "
                  f"-> model better than time-of-day odds by {gain:.2f}%"]
            st = strong(p, y)
            summary[(name, h)] = {"gain": gain, "strong": st, "ll": (ll_m, ll_t)}
            if h == 5:
                cal = calib_table(p, y)
                L += ["", "CALIBRATION on 2026 (said X% -> it happened Y% of the time):",
                      cal.round(1).to_string(index=False), "",
                      "STRONG READINGS on 2026:", st.round(1).to_string(index=False), ""]
                mon = te.assign(ok=(p.argmax(1) == y), conf=p.max(1)).groupby(te.ts.dt.strftime("%Y-%m")).agg(
                    minutes=("ok", "size"), right_pct=("ok", "mean"))
                L += ["MONTH BY MONTH (top pick right %):", (mon.assign(right_pct=mon.right_pct * 100)).round(1).to_string(), ""]
                imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
                L += ["WHAT MATTERS MOST (top 12 features):", ", ".join(imp.index[:12]), ""]
                pc = [profit_check(name, te, p, t) for t in (0.5, 0.55, 0.6, 0.65)]
                summary[(name, h)]["profit"] = pc
                L += ["PROFIT CHECK 2026 (buy ATM option on the reading, sell 5 min later, after charges, 1 lot):"]
                L += [f"  >= {int(x['thr'] * 100)}%: {x['trades']} trades, gross {x['gross']:,.0f}, net {x['net']:,.0f}, "
                      f"win {x['win_%']:.0f}%" for x in pc]
                L.append("")
                _plot(name, h, cal)
            else:
                L += [st[st["reading"].str.contains("60%")].round(1).to_string(index=False), ""]
    (REP / "verdict.txt").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L), summary


def _plot(name: str, h: int, cal: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 100], [0, 100], color="#8c98aa", lw=1, ls="--", label="perfect")
        for cname, col in (("bullish", "#199e70"), ("bearish", "#d95926")):
            c = cal[cal["class"] == cname]
            ax.plot(c.said_avg, c.happened, marker="o", color=col, label=cname)
        ax.set_xlabel("model said (%)"); ax.set_ylabel("actually happened (%)")
        ax.set_title(f"{name} next {h} min, 2026 test"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(REP / f"calibration_{name}_h{h}.png", dpi=110); plt.close(fig)
    except Exception as e:                                                # noqa: BLE001
        print("plot failed:", e)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    text, _ = run()
    print(text)
