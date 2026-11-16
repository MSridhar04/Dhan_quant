"""Startup check for live mode outside market hours: Dhan history fetch, option contract lookup, feed connect."""
import sys
import time
from datetime import timedelta

import pandas as pd

from quant.live.engine import ALGO, IDX, OPT_BAND, VIX_SID, QuantEngine

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(ALGO))
    from algo.config import load_settings
    from algo.core.events import Instrument
    from algo.core.timeutil import now_ist
    from algo.data.dhan_client import DhanClient
    from algo.data.dhan_feed import DhanTickFeed
    from algo.data.historical import HistoricalStore
    from algo.data.instruments import InstrumentMaster
    s = load_settings()
    hs = HistoricalStore(s.path("data_dir"), DhanClient(s.client_id, s.access_token, data_rate=3.0))
    today = now_ist().date()
    eng = QuantEngine(today, mode="live")
    eng.master = InstrumentMaster(s.path("data_dir")).load()
    eng.load_history(fetch=lambda sid, seg, a, b: hs._fetch_intraday(Instrument(sid, seg, sid, "INDEX"), a, b))
    for k, h in eng.hist.items():
        print(f"history {k}: {h.ts.dt.date.nunique()} days, last {h.ts.max()}")
    print("expiries", {n: [str(d) for d in eng.expiries[n][-3:]] for n in IDX})
    insts = []
    for n in IDX:
        spot = float(eng.hist[n].close.iloc[-1])
        step = IDX[n]["step"]
        atm = round(spot / step) * step
        exp = eng.master.expiries(n, on_or_after=today)[0]
        got = 0
        for i in range(-OPT_BAND, OPT_BAND + 1):
            for side in ("CE", "PE"):
                try:
                    insts.append(eng.master.option(n, exp, float(atm + i * step), side)); got += 1
                except Exception as e:                       # noqa: BLE001
                    print("  missing", n, atm + i * step, side, e)
        print(f"{n}: spot {spot:,.1f}, ATM {atm}, expiry {exp}, {got} option contracts found")
    feed = DhanTickFeed(s.client_id, s.access_token, build_bars=True, record_bars=False)
    for inst in insts[:4] + [Instrument("13", "IDX_I", "NIFTY", "INDEX"), Instrument(VIX_SID, "IDX_I", "VIX", "INDEX")]:
        feed.subs[inst.key] = inst
    feed.start()
    ok = feed.connected.wait(15)
    time.sleep(3)
    print("feed connected:", ok, "| packets seen:", feed.last_tick_at > 0)
    feed.stop()
