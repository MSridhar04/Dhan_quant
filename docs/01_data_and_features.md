# 01 · Data and features

## The data (all already on this computer, read from `dhan_algo\data`)
| Data | Period | Notes |
|---|---|---|
| NIFTY, SENSEX index, 1-minute | Nov 2021 → today | 1,239 trading days, about 4.6 lakh minutes each |
| Weekly options ATM ±10 strikes, 1-minute (price, open interest, IV) | NIFTY Nov 2021 →, SENSEX May 2023 → | SENSEX weekly options restarted in May 2023 |
| 50 NIFTY stocks, 1-minute | Oct 2023 → Oct 2026 | for breadth and heavyweight features |
| India VIX, 1-minute | Nov 2021 → today | downloaded 15 Nov 2026 |
| Order-book depth (5 levels) of NIFTY / SENSEX futures | from 16 Nov 2026 | recorded live every trading day; can't be downloaded |

Futures history can't be downloaded from Dhan (only current contracts), so we build a **synthetic future** from
the options: *future = strike + call price − put price*.
Example: strike 25,150, call 120, put 95 → synthetic future 25,175, i.e. 25 points (about 10 bp) above the index.

## What a "feature" is
A feature is one number describing the market **at the minute that just closed**, using nothing from the future.
The model sees 85 of them every minute. The main groups, with examples:

| Group | Feature (code name) | Example reading |
|---|---|---|
| Recent moves | `r5` = move over the last 5 minutes, in basis points | r5 = +12 → NIFTY rose 0.12% (about 30 points) in 5 min |
| Jumpiness | `sigma60` = typical 1-minute move over the last hour | 3 bp on a calm day, 8 bp on a wild day |
| Position in day | `day_pos` = where the price is in today's range (0 = day's low, 1 = high) | 0.95 → near today's high |
| Yesterday | `gap`, `prev_range`, `quiet` (yesterday's range ÷ 14-day average) | quiet 0.6 → yesterday was unusually calm |
| Calendar | `mod` minute of the day, `dte` days to expiry | dte 0 → expiry day |
| Options | `straddle_bp` = ATM call + put price as % of the index | 120 bp → the market is pricing a ±1.2% move by expiry |
| Options | `pcr` = put open interest ÷ call open interest (ATM ±3) | 1.3 → more puts than calls open |
| Synthetic future | `syn_basis_bp` = future's premium over the index; `syn_lead1` = future moved more than the index | |
| VIX | `vix`, `vix_chg15`, `vix_vs_20d` | VIX 16.6, up 0.4 in 15 min, 8% above its 20-day average |
| Breadth | `br_up5` = share of the 50 stocks up over 5 min | 0.80 → 40 of 50 stocks rising |
| Heavyweights | `hv_avg_r5` = HDFC Bank, Reliance, ICICI, Infosys, Bharti, TCS average 5-min move | |
| Other index | `x_r1` = SENSEX's last-minute move when predicting NIFTY (and vice versa) | |

## What the models predict ("labels")
- **Busy** (30 min): was the index's high-to-low range over the next 30 minutes in the **top 25%** for the current
  jumpiness? Example: on a calm day a 0.4% range can count as busy; on a wild day it needs much more.
- **First touch** (30 min): from now, does the index first rise by about one "typical move" (bullish), first fall by
  that much (bearish), or do neither within 30 minutes?

These questions were chosen because they match how trades work (a target vs a stop) and because big moves are
what option buyers need.
