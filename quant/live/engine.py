"""Live quant engine (plain English: docs/03_live_dashboard.md).

    python -m quant.live.engine                    # live: waits for 09:14, runs to 15:31 (scheduled task)
    python -m quant.live.engine --replay 2026-11-13 [--speed 0]   # test: replay a past day from the cache

Every minute, for NIFTY and SENSEX:
  1. 1-min candles: index, India VIX, ATM +-8 weekly options (price + OI), the 50 NIFTY stocks
     (own Dhan websocket - the paper session's feed is never touched; startup fetches today's candles so far)
  2. the SAME feature code as research (quant/features.py)
  3. models (quant/live/train.py): P(busy next 30 min) and P(bullish / neither / bearish first, next 30 min)
  4. writes live/state.json (the Control Center Quant tab reads it) and live/predictions/<date>.csv
  5. 30 minutes later, checks what really happened -> live scorecard (today and since launch)
Nothing here places orders.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ALGO = ROOT.parent / "dhan_algo"
LIVE = ROOT / "live"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ALGO))
from quant import features as F  # noqa: E402

IDX = {"NIFTY": {"sid": "13", "opt_seg": "NSE_FNO", "step": 50},
       "SENSEX": {"sid": "51", "opt_seg": "BSE_FNO", "step": 100}}
VIX_SID = "21"
OPT_BAND = 8                    # subscribe ATM +- 8 strikes (re-centred when the index moves)
OI_STRIKES = 3                  # open interest summed over ATM +- 3 (as in research)
H = 30
LOG = ROOT / "reports" / "live_engine.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ candle store
class Store:
    """Per-instrument 1-min candles keyed by bar start time; later updates of the same minute overwrite."""

    def __init__(self):
        self.bars: dict[str, dict] = {}

    def add(self, key: str, ts, o, h, l, c, v=0, oi=0):
        self.bars.setdefault(key, {})[pd.Timestamp(ts)] = (o, h, l, c, v, oi)

    def add_frame(self, key: str, df: pd.DataFrame):
        for r in df.itertuples(index=False):
            self.add(key, r.ts, r.open, r.high, r.low, r.close, getattr(r, "volume", 0), getattr(r, "oi", 0))

    def frame(self, key: str, upto=None) -> pd.DataFrame:
        b = self.bars.get(key, {})
        if not b:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "oi"])
        df = pd.DataFrame([(t, *v) for t, v in b.items()],
                          columns=["ts", "open", "high", "low", "close", "volume", "oi"]).sort_values("ts")
        return df[df.ts <= upto] if upto is not None else df

    def last_ts(self, key: str):
        b = self.bars.get(key)
        return max(b) if b else None


# ------------------------------------------------------------------ the engine
class QuantEngine:
    def __init__(self, day: date, mode: str = "live"):
        self.day, self.mode = day, mode
        # replay output is kept apart so test days never enter the live scorecard
        self.pred_dir = LIVE / ("predictions" if mode == "live" else "replay_predictions")
        meta = json.loads((ROOT / "models" / "live" / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.models = {n: {"big": lgb.Booster(model_file=str(ROOT / "models" / "live" / f"{n}_big{H}.txt")),
                           "tb": lgb.Booster(model_file=str(ROOT / "models" / "live" / f"{n}_tb{H}.txt"))}
                       for n in IDX}
        self.store = Store()
        self.hist: dict[str, pd.DataFrame] = {}          # previous days' 1-min candles: NIFTY, SENSEX, VIX
        self.opt_keys: dict[str, dict] = {n: {} for n in IDX}   # name -> {(side, strike): key}
        self.stock_ids: list[str] = sorted(p.name.split("_")[-1] for p in (ALGO / "data" / "bars").glob("NSE_EQ_*"))
        from algo.api.watchdog import load_holidays
        self.holidays = load_holidays(ALGO / "config" / "market_holidays.txt")
        self.expiries: dict[str, list[date]] = {}
        self.preds: list[dict] = []
        self.last_done = None
        self.history_rows: dict[str, list] = {n: [] for n in IDX}
        self.last_X: dict[str, pd.DataFrame] = {}

    # ---------------- context
    def load_history(self, fetch=None):
        """Previous ~45 calendar days of index + VIX candles. Live: Dhan intraday API (in memory, nothing cached);
        replay: dhan_algo's cache (read-only)."""
        start = self.day - timedelta(days=45)
        for key, sid in (("NIFTY", "13"), ("SENSEX", "51"), ("VIX", VIX_SID)):
            df = fetch(sid, "IDX_I", start, self.day) if fetch else _cache_bars(f"IDX_I_{sid}", start, self.day)
            df = df[(df.ts.dt.hour * 60 + df.ts.dt.minute).between(555, 929)]
            prev = df[df.ts.dt.date < self.day]
            n_bars = prev.groupby(prev.ts.dt.date).ts.transform("size")
            # as research (dhan_algo full_trading_days): weekdays with a normal session only (no Muhurat etc.)
            prev = prev[(prev.ts.dt.weekday < 5) & (n_bars >= 200)]
            self.hist[key] = prev.reset_index(drop=True)
            today = df[df.ts.dt.date == self.day]
            if len(today):
                self.store.add_frame(key, today)
        from algo.data.expiry_calendar import build_expiries
        for n in IDX:
            tdays = sorted(set(self.hist[n].ts.dt.date)) + [self.day]
            exps = build_expiries(n, tdays)
            self.expiries[n] = sorted(set(exps) | set(self.future_expiries(n)))

    def future_expiries(self, name: str) -> list[date]:
        try:
            return self.master.expiries(name, on_or_after=self.day)[:3]
        except Exception:                                      # noqa: BLE001
            return []

    # ---------------- features + prediction for one minute
    def compute(self, t: pd.Timestamp) -> dict:
        idx_feats = {}
        for n in IDX:
            today = self.store.frame(n, t)[["ts", "open", "high", "low", "close"]]
            bars = pd.concat([self.hist[n][["ts", "open", "high", "low", "close"]], today], ignore_index=True)
            idx_feats[n] = F.index_features(bars)
        vix = pd.concat([self.hist["VIX"][["ts", "close"]], self.store.frame("VIX", t)[["ts", "close"]]],
                        ignore_index=True).rename(columns={"close": "vix"})
        br = F.breadth([F.stock_minute(self.store.frame(f"EQ:{s}", t)[["ts", "open", "close"]], s)
                        for s in self.stock_ids if self.store.last_ts(f"EQ:{s}") is not None], min_stocks=30)
        out = {}
        for n in IDX:
            df = F.add_expiry(idx_feats[n].copy(), self.expiries[n], self.holidays)
            df = F.add_options(df, self.option_frame(n, t), with_iv=False)
            df = F.add_vix(df, vix)
            df = F.add_cross(df, idx_feats["SENSEX" if n == "NIFTY" else "NIFTY"])
            if len(br) and "ts" in br:
                df = df.merge(br, on="ts", how="left")
            row = df[df.ts == t]
            if row.empty:
                continue
            feats = self.meta[n]["features"]
            X = row.reindex(columns=feats).astype(float)
            self.last_X[n] = X
            pb = float(self.models[n]["big"].predict(X)[0])
            pt = self.models[n]["tb"].predict(X)[0]
            r = row.iloc[0]
            m = self.meta[n]
            label = "quiet" if pb <= m["big_quiet_below"] else "busy" if pb >= m["big_busy_above"] else "normal"
            out[n] = {"spot": round(float(r.close), 2), "big": round(pb, 3), "big_label": label,
                      "tb": {"bear": round(float(pt[0]), 3), "neither": round(float(pt[1]), 3), "bull": round(float(pt[2]), 3)},
                      "base": {"big": m["big_base"], "bear": m["tb_base"][0], "neither": m["tb_base"][1], "bull": m["tb_base"][2]},
                      "market": {k: (None if pd.isna(r.get(k)) else round(float(r.get(k)), 3)) for k in
                                 ("vix", "vix_chg15", "syn_basis_bp", "br_up5", "br_above_open", "pcr", "hv_avg_r5",
                                  "straddle_bp", "sigma60", "from_open", "dte")},
                      "sigma60": None if pd.isna(r.sigma60) else float(r.sigma60),
                      # what the reading means in index points, and when it can be checked (for the owner)
                      "busy_pts": None if pd.isna(r.sigma60) else round(m["big_cut_ratio"] * float(r.sigma60) * np.sqrt(H) / 1e4 * float(r.close)),
                      "move_pts": None if pd.isna(r.sigma60) else round(1.0 * float(r.sigma60) * np.sqrt(H) / 1e4 * float(r.close)),
                      "check_at": (t + pd.Timedelta(minutes=H)).strftime("%H:%M")}
            self.preds.append({"ts": t, "index": n, "spot": float(r.close), "p_big": pb, "p_bear": float(pt[0]),
                               "p_neither": float(pt[1]), "p_bull": float(pt[2]),
                               "sigma60": None if pd.isna(r.sigma60) else float(r.sigma60)})
            self.history_rows[n].append([t.strftime("%H:%M"), round(pb, 3), round(float(pt[2]), 3), round(float(pt[0]), 3)])
        return out

    def option_frame(self, name: str, t) -> pd.DataFrame:
        """Per minute today: ATM strike (nearest to the index), ATM CE / PE close, OI summed over ATM +- 3."""
        step = IDX[name]["step"]
        spot = self.store.frame(name, t)[["ts", "close"]]
        keys = self.opt_keys[name]
        if spot.empty or not keys:
            return pd.DataFrame()
        px, oi = {}, {}
        for (side, k), key in keys.items():
            f = self.store.frame(key, t)
            if f.empty:
                continue
            s = f.set_index("ts")
            px[(side, k)] = s.close
            oi[(side, k)] = s.oi.replace(0, np.nan)
        rows = []
        for ts, c in zip(spot.ts, spot.close):
            atm = round(c / step) * step
            ce, pe = px.get(("CE", atm)), px.get(("PE", atm))
            ce_v = ce[ce.index <= ts] if ce is not None else pd.Series(dtype=float)
            pe_v = pe[pe.index <= ts] if pe is not None else pd.Series(dtype=float)
            if ce_v.empty and pe_v.empty:
                continue                                          # keep a side even if the other is missing
            oic = sum(float(oi[("CE", atm + i * step)][oi[("CE", atm + i * step)].index <= ts].ffill().iloc[-1])
                      for i in range(-OI_STRIKES, OI_STRIKES + 1)
                      if ("CE", atm + i * step) in oi and len(oi[("CE", atm + i * step)][oi[("CE", atm + i * step)].index <= ts].dropna()))
            oip = sum(float(oi[("PE", atm + i * step)][oi[("PE", atm + i * step)].index <= ts].ffill().iloc[-1])
                      for i in range(-OI_STRIKES, OI_STRIKES + 1)
                      if ("PE", atm + i * step) in oi and len(oi[("PE", atm + i * step)][oi[("PE", atm + i * step)].index <= ts].dropna()))
            rows.append({"ts": ts, "CE_px": float(ce_v.iloc[-1]) if len(ce_v) else np.nan,
                         "PE_px": float(pe_v.iloc[-1]) if len(pe_v) else np.nan,
                         "CE_oi": oic or np.nan, "PE_oi": oip or np.nan, "CE_k": float(atm), "PE_k": float(atm)})
        return pd.DataFrame(rows)

    # ---------------- outcomes + scorecard
    def resolve(self, now: pd.Timestamp) -> dict:
        """For predictions at least 30 min old: did a busy 30 min follow, and which barrier was touched first."""
        if not self.preds:
            return {}
        p = pd.DataFrame(self.preds)
        rows = []
        for n in IDX:
            bars = self.store.frame(n).set_index("ts")
            cut = self.meta[n]["big_cut_ratio"]
            for r in p[(p["index"] == n) & (p.ts <= now - pd.Timedelta(minutes=H))].itertuples(index=False):
                fut = bars[(bars.index > r.ts) & (bars.index <= r.ts + pd.Timedelta(minutes=H))]
                if len(fut) < H - 2 or not r.sigma60:
                    continue
                bar = 1.0 * r.sigma60 * np.sqrt(H)
                up = ((fut.high / r.spot - 1) * 1e4 >= bar).to_numpy()
                dn = ((fut.low / r.spot - 1) * 1e4 <= -bar).to_numpy()
                fu = np.argmax(up) if up.any() else 10**6
                fd = np.argmax(dn) if dn.any() else 10**6
                outcome = "bull" if fu < fd else "bear" if fd < fu else "neither"
                ratio = (fut.high.max() - fut.low.min()) / r.spot * 1e4 / (r.sigma60 * np.sqrt(H))
                m = self.meta[n]
                label = "quiet" if r.p_big <= m["big_quiet_below"] else "busy" if r.p_big >= m["big_busy_above"] else "normal"
                rows.append({**r._asdict(), "busy": int(ratio > cut), "first": outcome, "label": label,
                             "range_pts": round(float(fut.high.max() - fut.low.min()), 1),
                             "busy_pts": round(cut * r.sigma60 * np.sqrt(H) / 1e4 * r.spot, 1)})
        res = pd.DataFrame(rows)
        self.recent = res
        if len(res):
            res.to_csv(self.pred_dir / f"{self.day}_resolved.csv", index=False)
        return scorecard(res, self.pred_dir)

    def recent_checks(self) -> dict:
        r = getattr(self, "recent", None)
        if r is None or not len(r):
            return {}
        r = r[pd.to_datetime(r.ts).dt.minute % 15 == 0]
        out = {}
        for n, g in r.groupby("index"):
            g = g.tail(12)
            out[n] = [{"at": pd.Timestamp(x.ts).strftime("%H:%M"), "label": x.label, "p_big": round(float(x.p_big), 2),
                       "busy_pts": x.busy_pts, "range_pts": x.range_pts, "busy": int(x.busy),
                       "lean": "bull" if x.p_bull >= x.p_bear else "bear",
                       "lean_p": round(float(max(x.p_bull, x.p_bear)), 2), "first": x["first"]}
                      for _, x in g.iterrows()]
        return out

    def write_state(self, out: dict, score: dict, status: str = "running", msg: str = ""):
        LIVE.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(exist_ok=True)
        state = {"updated": datetime.now().isoformat(timespec="seconds"), "mode": self.mode, "day": str(self.day),
                 "status": status, "message": msg, "minute": self.last_done.strftime("%H:%M") if self.last_done else None,
                 "horizon_min": H, "indexes": out,
                 "history": {n: v[-150:] for n, v in self.history_rows.items()}, "scorecard": score,
                 "recent": self.recent_checks()}
        tmp = LIVE / "state.tmp"
        # JSON has no NaN: blank values (e.g. the first minutes, before volatility is known) become null, otherwise the
        # Control Center's /api/quant answers 500 (found live 16 Nov, 09:15 rows)
        tmp.write_text(json.dumps(_clean(state), default=str, allow_nan=False), encoding="utf-8")
        tmp.replace(LIVE / "state.json")
        if self.preds:
            pd.DataFrame(self.preds).to_csv(self.pred_dir / f"{self.day}.csv", index=False)


def _clean(o):
    """Replace NaN / inf anywhere in a nested structure with None (JSON-safe)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def scorecard(res: pd.DataFrame, pred_dir: Path = LIVE / "predictions") -> dict:
    """Today + since launch: busy-meter accuracy by label and first-touch hit rate on strong readings."""
    files = sorted(pred_dir.glob("*_resolved.csv")) if pred_dir.exists() else []
    allres = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else res

    def card(d: pd.DataFrame) -> dict:
        if d is None or not len(d):
            return {"n": 0}
        out = {"n": int(len(d)), "busy_rate": round(float(d.busy.mean()), 3)}
        labels = d["label"] if "label" in d else pd.Series("", index=d.index)   # files from before 15 Nov lack it
        for lab in ("quiet", "normal", "busy"):
            m = labels == lab
            out[f"when_{lab}_n"] = int(m.sum())
            out[f"when_{lab}_busy"] = round(float(d.busy[m].mean()), 3) if m.any() else None
        strong = (d.p_bull >= 0.5) | (d.p_bear >= 0.5)
        pick = np.where(d.p_bull >= d.p_bear, "bull", "bear")
        out["strong_n"] = int(strong.sum())
        out["strong_right"] = round(float((d["first"][strong] == pick[strong]).mean()), 3) if strong.any() else None
        out["brier_model"] = round(float(((d.p_big - d.busy) ** 2).mean()), 4)
        out["brier_base"] = round(float(((0.25 - d.busy) ** 2).mean()), 4)
        return out
    return {"today": card(res), "all": card(allres), "days": len(files)}


# ------------------------------------------------------------------ data sources
def _cache_bars(folder: str, start: date, end: date) -> pd.DataFrame:
    fs = [f for f in sorted((ALGO / "data" / "bars" / folder).glob("*.parquet"))
          if f"{start:%Y-%m}" <= f.stem[:7] <= f"{end:%Y-%m}"]
    if not fs:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "oi"])
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates("ts").sort_values("ts")
    for c in ("volume", "oi"):
        if c not in df:
            df[c] = 0
    return df[(df.ts.dt.date >= start) & (df.ts.dt.date <= end)].reset_index(drop=True)


def replay(day: date, speed: float = 0.0):
    """Feed a past day minute by minute from dhan_algo's cache (read-only) through the same engine."""
    eng = QuantEngine(day, mode="replay")
    eng.master = None
    eng.load_history()
    # today's bars for replay come from the cache: take them out of the store and feed them minute by minute
    day_bars = {k: eng.store.frame(k) for k in ("NIFTY", "SENSEX", "VIX")}
    eng.store = Store()
    for s in eng.stock_ids:
        d = _cache_bars(f"NSE_EQ_{s}", day, day)
        if len(d):
            day_bars[f"EQ:{s}"] = d
    for n, cfg in IDX.items():
        for side in ("CE", "PE"):
            fs = ALGO / "data" / "rolling" / f"{cfg['sid']}_WEEK0_{side}"
            f = fs / f"{day:%Y-%m}.parquet"
            f = f if f.exists() else fs / f"{day:%Y-%m}.partial.parquet"
            if not f.exists():
                continue
            o = pd.read_parquet(f)
            o = o[o.ts.dt.date == day].drop_duplicates(["ts", "strike"], keep="last")
            for k, g in o.groupby("strike"):
                key = f"OPT:{n}:{side}:{k:g}"
                eng.opt_keys[n][(side, float(k))] = key
                day_bars[key] = g[["ts", "open", "high", "low", "close", "volume", "oi"]]
    minutes = sorted(day_bars["NIFTY"].ts)
    log(f"replay {day}: {len(minutes)} minutes, {len(day_bars)} instruments")
    for t in minutes:
        for key, d in day_bars.items():
            row = d[d.ts == t]
            if len(row):
                r = row.iloc[0]
                eng.store.add(key, t, r.open, r.high, r.low, r.close, r.get("volume", 0), r.get("oi", 0))
        out = eng.compute(t)
        eng.last_done = t
        score = eng.resolve(t)
        eng.write_state(out, score, status="replay")
        if speed:
            time.sleep(speed)
    log(f"replay done: {len(eng.preds)} predictions")
    return eng


def live():
    """Real market: own websocket, startup back-fill of today's candles, compute every minute."""
    from algo.api.watchdog import load_holidays
    from algo.config import load_settings
    from algo.core.events import Instrument
    from algo.core.timeutil import now_ist
    from algo.data.dhan_client import DhanClient
    from algo.data.dhan_feed import DhanTickFeed
    from algo.data.historical import HistoricalStore
    from algo.data.instruments import InstrumentMaster
    today = now_ist().date()
    if today.weekday() >= 5 or today in load_holidays(ALGO / "config" / "market_holidays.txt"):
        log("market closed today")
        return
    while now_ist().strftime("%H:%M") < "09:14":
        time.sleep(20)
    s = load_settings()
    client = DhanClient(s.client_id, s.access_token, data_rate=3.0)
    hs = HistoricalStore(s.path("data_dir"), client)

    def fetch(sid, seg, start, end, kind="INDEX"):
        try:
            return hs._fetch_intraday(Instrument(sid, seg, sid, kind), start, end)
        except Exception as e:                                 # noqa: BLE001
            log(f"history fetch {seg}:{sid} failed: {e}")
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "oi"])

    eng = QuantEngine(today, mode="live")
    eng.master = InstrumentMaster(s.path("data_dir")).load()
    eng.load_history(fetch=lambda sid, seg, a, b: fetch(sid, seg, a, b))
    feed = DhanTickFeed(s.client_id, s.access_token, build_bars=True, record_bars=False)
    key_of = {}

    def sub(inst, key):
        key_of[inst.key] = key
        feed.subs[inst.key] = inst

    for n, sid in (("NIFTY", "13"), ("SENSEX", "51"), ("VIX", VIX_SID)):
        sub(Instrument(sid, "IDX_I", n, "INDEX"), n)
    for sid in eng.stock_ids:
        inst = Instrument(sid, "NSE_EQ", sid, "EQUITY")
        sub(inst, f"EQ:{sid}")
        d = fetch(sid, "NSE_EQ", today, today, "EQUITY")
        if len(d):
            eng.store.add_frame(f"EQ:{sid}", d)

    def subscribe_options(n):
        last = eng.store.frame(n)
        if last.empty:
            return []
        spot, step = float(last.close.iloc[-1]), IDX[n]["step"]
        atm = round(spot / step) * step
        exp = eng.master.expiries(n, on_or_after=today)[0]
        new = []
        for i in range(-OPT_BAND, OPT_BAND + 1):
            for side in ("CE", "PE"):
                k = float(atm + i * step)
                if (side, k) in eng.opt_keys[n]:
                    continue
                try:
                    inst = eng.master.option(n, exp, k, side)
                except Exception:                              # noqa: BLE001
                    continue
                key = f"OPT:{n}:{side}:{k:g}"
                eng.opt_keys[n][(side, k)] = key
                sub(inst, key)
                d = fetch(inst.security_id, inst.exchange_segment, today, today, "OPTIDX")
                if len(d):
                    eng.store.add_frame(key, d)
                new.append(inst)
        return new

    for n in IDX:
        subscribe_options(n)
    feed.start()
    log(f"live: subscribed {len(feed.subs)} instruments")
    eng.write_state({}, {}, status="starting", msg="waiting for the first minute")
    pending = None
    for kind, payload in feed.events():
        now = pd.Timestamp(now_ist().replace(tzinfo=None))
        if kind == "bar":
            key = key_of.get(payload.instrument.key)
            if key:
                b = payload
                eng.store.add(key, b.ts, b.open, b.high, b.low, b.close, getattr(b, "volume", 0), getattr(b, "oi", 0))
                if key == "NIFTY":
                    pending = pd.Timestamp(b.ts)
        elif kind == "error":
            log(f"feed error: {payload}")
            eng.write_state({}, {}, status="error", msg=str(payload))
        if pending is not None and pending != eng.last_done and now >= pending + pd.Timedelta(minutes=1, seconds=4):
            try:
                out = eng.compute(pending)
                eng.last_done = pending
                eng.write_state(out, eng.resolve(pending))
            except Exception as e:                             # noqa: BLE001
                log(f"compute failed at {pending}: {type(e).__name__}: {e}")
            for n in IDX:                                      # re-centre option strikes if the index moved far
                new = subscribe_options(n)
                for inst in new:
                    feed.subscribe(inst)
            pending = None
        if now.strftime("%H:%M") >= "15:31":
            break
    feed.stop()
    eng.write_state({}, eng.resolve(pd.Timestamp(now_ist().replace(tzinfo=None))), status="closed",
                    msg="market closed - next run 09:14 on the next trading day")
    log("live: done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="")
    ap.add_argument("--speed", type=float, default=0.0)
    a = ap.parse_args()
    if a.replay:
        replay(date.fromisoformat(a.replay), a.speed)
    else:
        live()
