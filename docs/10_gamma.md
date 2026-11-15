# 10 · Research #5: gamma exposure (dealer positioning)

Code: `quant/research/gex.py` · per-minute figures: `data/gex_<IDX>.parquet` · results: `reports/gex.txt`

## The idea
People who sell options usually hedge by buying or selling the index. How they hedge depends on which options are
open:
- **positive gamma** (they own options): they sell into rallies and buy dips, which **dampens** moves;
- **negative gamma** (they're short options): they buy rallies and sell dips, which **amplifies** moves.

On expiry day the strike with the most open interest is often said to act like a magnet: the popular "max pain" or
"pinning" idea.

## What we computed, every minute (ATM ±10 strikes, from cached open interest and IV)
| Figure | Meaning | Example |
|---|---|---|
| gamma ratio | (call gamma × OI − put gamma × OI) ÷ total, from −1 to +1. Above 0 = hedging dampens (standard convention) | +0.10 typical |
| total gamma | how heavy the hedging is, compared with its 60-day median | |
| max-OI strike | the strike with the most calls + puts open (the "pin") | NIFTY 25,113 → pin 25,100 |
| call wall / put wall | the biggest call OI above spot / put OI below | 10 Jun 2025: call wall 25,500, put wall 25,000 |

## The four tests and results (15 Nov 2026)
| Test | Result |
|---|---|
| A. Do moves shrink when hedging dampens? (next-60-min range, 10:00 / 12:00 / 14:00) | No clear pattern. NIFTY busy rate 23–26% in every gamma group; SENSEX random |
| B. Expiry pinning: does the close end nearer the max-OI strike than the ATM strike seen at noon? | **No: the opposite.** Nearer the pin only 35% of expiry days (NIFTY) and 22% (SENSEX), where 50% would mean no effect. The index mostly stays near where it was at noon |
| C. Does gamma improve the live busy model? | No: AUC 0.694 → 0.693 (NIFTY), 0.685 → 0.684 (SENSEX) |
| D. Skip strategies by gamma sign (walk-forward 2023–26) | Hurts: +₹6.83L → +₹5.67L (helped 2 of 4 years, cost ₹1.2L in 2024) |

Small side note: Pivot NIFTY does slightly worse when hedging amplifies (−₹160 per trade picking years, −₹101 unseen),
but it's too small and too few trades to act on.

## Caveats
- We only see ±10 strikes around the money; walls further out are invisible.
- The dampen/amplify sign rests on an assumption about **who** holds which options. In India retail buys a lot of
  both calls and puts, so the standard US convention may not fit.

The flat results in A–C suggest a full option chain wouldn't change the picture much.

**Verdict:** gamma exposure gives no usable edge with our data, and "max-OI pinning" is not supported.
