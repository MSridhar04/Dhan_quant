# 08 · Research #3: position sizing by risk

Code: `quant/research/sizing.py` · results: `reports/sizing.txt`

## The question
Today every paper strategy trades **1 lot**. But they swing very differently: EMA SENSEX's daily ups and downs are
roughly 2–3× Judas's. On a bad day the big swingers dominate the total. Can we size each strategy so the
portfolio earns the same but falls less, **without skipping any trade**?

## Methods (sizes reset at the start of each month, using only past days)
| Method | How lots are set | Example |
|---|---|---|
| Equal | 1 lot each (today) | |
| Inverse volatility | lots ∝ 1 ÷ how much the strategy's daily result has swung over the last ~250 days | EMA SENSEX swings 2× the average → about half the lots |
| Risk parity | each strategy contributes the same share of total risk, allowing for strategies that move together | |
| Volatility targeting | everything is scaled down when the whole portfolio has been jumpy for the last 20 days (0.5×–2×) | |
| Drawdown brake | a strategy is halved after it falls more than half its past worst drop; back to full at a new high | |

**Fair comparison:** the average size stays 1 lot per strategy, and we compare *profit per unit of risk*
(return ÷ worst drop, Sharpe). "Better" means more profit for the same risk, not just bigger bets.

## Results, test Jan 2023 – Oct 2026
| Method (ideal fractional sizes) | Return ÷ worst drop | Same risk as equal: profit / worst drop |
|---|---|---|
| Equal (1 lot each) | 7.8 | +₹6.56L / −₹84k |
| **Inverse volatility** | **9.2** | **+₹6.50L / −₹70k** |
| Risk parity | 7.9 | +₹6.45L / −₹82k |
| Volatility targeting | 5.4 | +₹5.76L / −₹1.07L |
| Drawdown brake | 9.5 | +₹6.19L / −₹65k (less profit) |

**With whole lots** (what you can actually trade), at an average of 2 lots per strategy:

| | Profit | Worst drop | Worst month |
|---|---|---|---|
| Equal, 2 lots each | +₹13.12L | −₹1.68L | −₹46k |
| **Inverse volatility, 2-lot base** | **+₹13.13L** | **−₹1.38L** | −₹45k |

Same profit, about **18% smaller worst drop**.

Suggested lots at a 2-lot level (latest month): EMA SENSEX 1 · PSAR SENSEX 1 · Judas 3 · ORB 2 · Pivot NIFTY 2 ·
Pivot SENSEX 2 · Prev-zone 2 · Failbreak 2.

## What to do with it
- **At today's 1 lot each: nothing.** Lots can't be split; rounding 0.6 → 1 and 1.5 → 2 is too coarse to help.
- **When you scale up to 2+ lots, use inverse-volatility sizing,** recalculated monthly. It's the one method that
  cut the worst drop without giving up profit.
- **Avoid volatility targeting and pausing strategies.** Shrinking in jumpy periods cut profit more than risk, and
  pausing at 1 lot threw away more than half the profit.
- Caution: the Sharpe ratios are nearly equal (2.05 vs 2.07). The gain is in **smaller drops**, not more profit.
