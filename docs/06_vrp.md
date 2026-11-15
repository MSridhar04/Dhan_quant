# 06 · Research #1: volatility risk premium (VRP)

Code: `quant/research/vrp.py` · results: `reports/vrp.txt`, trades `reports/vrp_trades_<IDX>.csv`

## The idea in one paragraph
An option's price contains the market's guess of how much the index will move before expiry. Option sellers
take on the risk of a crash, so they charge a bit **more** than the move that usually happens. That extra is the
*volatility risk premium*. Selling it wins most weeks; it loses badly in shocks. Professional volatility sellers
make it work by (a) **capping** the shock loss and (b) **selling only when the premium is rich**.

**Example:** NIFTY at 25,150, two days to expiry. ATM call 110 + put 100 = straddle **210 points**, so options
charge for a move of about ±0.8%. Over the last 10 days NIFTY has typically moved about 0.45% a day, roughly
0.6% over two days. Premium charged 0.8% vs 0.6% expected → *rich* (ratio 1.3).

## The exact trade we test
- **Iron fly, 1 lot:** sell the ATM call and put, buy a call and a put *w* strikes further out as protection, so
  the worst loss is capped.
  Example (NIFTY, wings ±300): sell 25,150 CE + PE for 210, buy 25,450 CE + 24,850 PE for 40 → credit 170 points
  (₹11,050). Worst case: 300 − 170 = 130 points (₹8,450) plus charges.
- **Entry:** 1, 2 or 3 trading days before the weekly expiry, at 15:10 (to also collect the overnight decay) or
  09:30.
- **Exit:** expiry day 15:15, or (variant) a stop when the loss reaches 1× or 2× the credit.
- **Costs:** real charges on all 8 fills + 1 tick slippage each.

## The signals (all known at entry)
| Signal | Meaning |
|---|---|
| implied move | ATM straddle ÷ index: what options charge |
| realized move | daily volatility of the last 10 days × √(days left): what the market has been doing |
| **VRP ratio** | implied ÷ realized. Above 1 = rich premium |
| VIX not rising | today's VIX ≤ yesterday's: avoid selling into a building panic |

## Making the backtest honest about shocks
The option cache only holds strikes near the money. On a crash a short option can drift out of that window, and
its price would "freeze", hiding the loss on the worst day. Two guards:
1. A leg is never priced below its **intrinsic value** (what it's worth if exercised now).
2. A call or put spread is never worth more than its **width** (a hard market limit).

Both guards were found and fixed on a real day: 4 June 2024, election results, NIFTY crashed about 6%.

## How it was judged
Every combination (entry time × days before expiry × wing width × stop × filter) was scored on **Nov 2021 –
Dec 2024** (SENSEX May 2023 – Dec 2024). The best one was then run **once** on **Jan 2025 – Oct 2026**. We also report:
- what selling **every** week would have done;
- what share of all combinations made money in **both** periods (a hill or a spike?);
- the luck-adjusted probability (deflated Sharpe);
- a Monte Carlo of bad years.

## Results
**Verdict (15 Nov 2026): no reliable edge after costs.**

| | NIFTY | SENSEX |
|---|---|---|
| Chosen on training | 2 days before expiry, 09:30, wings ±400, hold, only when premium rich and VIX not rising | 3 days before, 09:30, wings ±600, hold, every week |
| Training | 25 trades, **+₹54.8k** | 79 trades, **+₹54.1k** |
| Unseen 2025–Oct 2026 | 22 trades, **+₹9.3k**, worst drop −₹35.4k | 92 trades, **+₹1.9k**, worst drop −₹29.6k |
| Selling every week (same setup), unseen | +₹53.7k | +₹1.9k |
| Variants that won in both periods | 13% of 207 | 7% of 180 |
| Luck-adjusted chance it's real | 1% | 0% |

**What it means in plain words:**
- Selling weekly iron flies **lost in most years 2021–24**. Example: NIFTY, wings ±400, 1 day before expiry: −₹16k to
  −₹28k a year. It turned positive in 2025–26 for wide wings (+₹30k, +₹33k). The premium was not consistently
  "rich" enough to pay 8 orders of costs and the occasional shock.
- The good training results came from picking the best of about 200 combinations. Unseen data shrank them to
  about break-even, which is exactly the trap the luck check warns about.
- 2025–26 looking different may be linked to SEBI's November 2024 changes (one weekly expiry per exchange, bigger
  lots). That's a hypothesis for later; there isn't enough post-change data yet to rely on.

**Takeaway:** in Indian weekly index options, prices are mostly fair; costs and rare shocks eat the premium.
Next research: **regime detection**, i.e. can we recognise 2022–23-type markets in advance?
