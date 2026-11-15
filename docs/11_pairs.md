# 11 · Research #6: pairs trading (statistical arbitrage)

Code: `quant/research/pairs.py` · results: `reports/pairs.txt`

## The idea
Two related stocks (two IT companies, two private banks) usually move together. When the gap between them gets
unusually wide, **buy the one that fell behind and short the one that ran ahead**, and wait for the gap to close.
The trade doesn't care whether the market goes up or down.

**Example:** INFY and TCS normally move within about 0.3% of each other in a day. Today INFY is up 1.2% and TCS up
0.2% (gap 1.0%, about 3× normal) → short ₹1 lakh of INFY, buy ₹1 lakh of TCS. If the gap closes to 0.2% you
make about 0.8% of ₹1 lakh = ₹800, before costs.

## Two versions (shorting rules decide what's possible)
| | Multi-day | Intraday |
|---|---|---|
| Holding | days, up to month end | same day, closed by 15:15 |
| How to short | stock futures (overnight) → futures charges | cash shorting allowed intraday → equity-intraday charges |
| Pair choice | each month, from the previous 60 or 120 days: returns correlated ≥ 0.6 and the strongest pull back together (cointegration) | each day, the most correlated pairs on 15-minute moves over the last 20 days |
| Entry / exit | gap ≥ 1.5 / 2 / 2.5 × its normal size; exit when it closes; optional stop | gap since 09:30 ≥ 1.5 / 2 / 3 × normal, from 10:00; exit when it closes or at 15:15 |

Both: ₹1 lakh per side, real charges + 2 bp slippage per order (4 orders per trade). Settings picked on 2024,
judged on Jan 2025 – Oct 2026.

## Results (15 Nov 2026)
| | Multi-day | Intraday |
|---|---|---|
| Chosen on 2024 | 120-day look-back, 10 pairs, enter at 2.5× | 10 pairs, enter at 3× |
| 2024 (picking year) | 33 trades, **+₹39.7k** | 246 trades, −₹18.7k (+₹41.5k before costs) |
| Unseen 2025–26, before costs | **−₹93.6k** | **+₹17.6k** |
| Unseen 2025–26, after costs | **−₹1.22L** (worst drop −₹1.44L) | **−₹73.3k** |
| Settings that won in both periods | 0 of 24 | 0 of 6 |

**What it means in plain words:**
- **Multi-day:** pairs that belonged together in the past drifted apart for weeks at a time, especially IT and bank
  pairs. No edge even before costs.
- **Intraday:** every setting made money **before costs** in both periods, so related stocks really do snap back
  within the day. But the snap-back is only ₹50–110 per trade, against about ₹250 of costs (4 orders, STT,
  slippage). After costs, every setting lost.
- Bigger positions don't fix it: STT and slippage grow with size. Intraday pairs would need about 3× lower costs.

**Takeaway:** the same lesson as the option tests. The market is efficient enough that small statistical edges
don't survive retail costs.
