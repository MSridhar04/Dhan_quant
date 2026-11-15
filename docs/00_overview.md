# 00 · What we're building and why

## What "quant" means here
A quant strategy is a trading rule that comes **from measured statistics**, not from a chart pattern that "looks
right". Every rule must answer three questions with numbers:

1. **What is the edge?** Example: "after the 15 minutes the model rates QUIET, a big move followed only 5% of the
   time, against a normal 25%."
2. **Does it survive costs?** Example: an iron fly sold for one hour earns about ₹230 of time decay, but 8 orders
   cost about ₹230. Edge minus costs = zero, so no strategy.
3. **Was it real or luck?** Example: if we try 2,772 versions of an idea, the best one will look good even if
   none has an edge. We discount for that (see `02_how_we_test.md`).

## Why not HFT (high-frequency trading)
HFT firms put their computers inside the exchange building and react in **microseconds** (millionths of a
second). We receive prices through Dhan about **once a second**, and an order takes **100+ milliseconds**.
Racing them is impossible, so we do what most quant funds do: **mid-frequency** strategies that hold for
minutes to days, where speed matters less than being statistically right.

## The roadmap
| Phase | What | Status |
|---|---|---|
| 0 | Rename to dhan_quant, one shared feature module, honest test harness | ✅ 15 Nov 2026 |
| 1 | Live **Quant** tab in the Control Center + live scorecard | ✅ live from 16 Nov 2026 |
| 2 | Quant strategy research (volatility risk premium, regimes, lead-lag, order flow, ...) | next - see `05_research_plan.md` |
| 3 | Paper trading of anything that passes, measuring real slippage | later |

## How the pieces fit
```
dhan_algo/data (5 years of minute data, read-only) ──► quant/dataset.py ──► research ──► docs/04 results
                                                         │
                                         quant/features.py (one definition)
                                                         │
Dhan live feed (own connection) ──► quant/live/engine.py ─┴─► live/state.json ──► Control Center "Quant" tab
```
The shared `features.py` matters: if research and live computed a feature even slightly differently, the
model would see something it was never trained on. On 15 Nov we checked all 85 features, live vs research,
at several minutes of a real day: **85 of 85 identical**.
