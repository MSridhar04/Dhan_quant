# 03 · The live engine and the Quant tab

## What runs, and when
- A Windows scheduled task **"DhanQuant Live"** starts the engine on weekdays at 09:05. It waits for 09:14 and runs
  to 15:31, and skips market holidays (`dhan_algo\config\market_holidays.txt`).
- It opens **its own** Dhan market-data connection. Paper trading and the depth recorder have their own, and Dhan
  allows up to 5.
- At start it fetches the last ~45 days of NIFTY / SENSEX / VIX minutes (for yesterday's context) and today's
  candles so far. A late start still sees the whole day.
- It follows: NIFTY, SENSEX, India VIX, 17 strikes of calls and puts around ATM for each index (re-centred if the
  index moves far), and the 50 NIFTY stocks.
- **Every minute**, about 4 seconds after the minute closes:
  1. builds the 85 features with the same code as research;
  2. runs the two models per index;
  3. writes `live/state.json` (the tab reads it every 15 s) and `live/predictions/<date>.csv`.
- **30 minutes later** it checks what really happened and updates the scorecard.
- It never places orders.

## How to read the tab (example)
```
NIFTY                                   25,144
Big move in the next 30 min          [ BUSY ]
52%   normal 25%          ███████████████░░░░░|░░░░░░░
Which way first in the next 30 min
███████████ ░░░░░░ ████████
Bearish 42%    Neither 24%    Bullish 33%
normal 27%     normal 47%     normal 26%
Busy odds today  ~~~/\~~~___~~  (dashed line = normal 25%)
India VIX 16.59 (+0.40 in 15m) · Futures premium 9.5 bp · Stocks rising 62% · Put/call 1.12 ...
```
How to read that example:
- **Busy 52% vs normal 25%** → the next half hour is about twice as likely as usual to see a big move, in either
  direction. Option buyers like busy markets; sellers prefer quiet ones.
- **Direction** → bearish 42% vs normal 27% leans down, but direction is the model's weak side: in the 2026 test it
  was only about 3% better than the normal odds. Treat it as a mild tilt, not a signal.
- **Labels:** QUIET means the bottom 10% of busy readings, BUSY the top 10%, NORMAL everything in between.

## The scorecard (is it still working?)
For every prediction, 30 minutes later:
- *When it said QUIET, how often was it actually busy?* Good: far below 25%.
  On the 13 Nov replay: **0%** after QUIET, 18.5% after NORMAL, 24% after BUSY.
- *Error score* (Brier): the average squared gap between the odds and what happened. Compared with always
  saying 25%. Lower is better.
- *Strong direction readings* (≥50%): how often they came true.

"Since launch" adds up every live day. Replay tests are stored separately (`live/replay_predictions/`) and never
count.

## If something looks wrong
| Tab shows | Meaning | Fix |
|---|---|---|
| CLOSED | outside 09:14–15:31 or a holiday | nothing |
| STARTING | connected, waiting for the first full minute | wait a minute |
| NO UPDATE >3 MIN | engine stopped or token expired | refresh `DHAN_ACCESS_TOKEN` in `dhan_algo\.env`, then run the task again |
| ERROR | the message explains it (e.g. token expired) | as above |
Log: `dhan_quant\reports\live_engine.log`.

## Test it any time (replay)
`.venv\Scripts\python.exe -m quant.live.engine --replay 2026-11-13` plays a past day through the same engine
(about 5 minutes) and the tab shows it as REPLAY.
