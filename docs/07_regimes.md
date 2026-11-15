# 07 · Research #2: regime detection

Code: `quant/research/regime.py` · results: `reports/regime.txt`

## The question
Our paper strategies made about ₹0 in Nov 2021 – Oct 2023 but +₹6.2 lakh in Nov 2023 – Oct 2026. If the lean
years were a recognisable *type* of market, we could switch strategies off (or size down) **before** such days.

## How each day was labelled (only with information known before it opened)
1. **Rule (transparent):** two questions about the days up to yesterday.
   - *Trend or range?* Efficiency ratio over 20 days = net move ÷ total distance travelled.
     Example: NIFTY rose 600 points net but travelled 3,000 points up and down → 0.2 = choppy range.
   - *Calm or volatile?* India VIX vs its 1-year median.
   → 4 regimes: trend-calm, trend-volatile, range-calm, range-volatile (about 25% of days each).
2. **Clustering (k-means):** groups days with similar trend, volatility, compression and VIX into 4 clusters.
3. **Hidden Markov model:** a statistical model that assumes the market switches between hidden "moods" (calm /
   stormy) and estimates today's mood from yesterday and before.

## How it was tested (walk-forward by year)
For each test year (2023, 2024, 2025, 2026):
- learn, on **earlier years only**, which regimes each strategy lost money in (average per trade below 0, at least
  15 trades);
- in the test year, skip that strategy on those regimes' days;
- compare with simply trading every day.

## Results (15 Nov 2026): the filter hurts
| Method | All trades, test years 2023–26 | With regime filter | Years it helped |
|---|---|---|---|
| Rule | +₹6.78L | +₹3.83L | 1 of 4 |
| Clustering | +₹6.78L | +₹4.16L | 1 of 4 |
| Hidden Markov | +₹6.78L | +₹5.06L | 0 of 4 |

It only helped in 2023, the lean year (−₹9k → +₹10k). In 2024 it cost more than ₹1.5 lakh.

**Why:** a regime where a strategy lost money in the past was usually *not* bad for it the next year, so skipping
removed many good trades.

**The useful finding:** the strategies earn in most regimes. Average ₹ per trade over 5 years:

| Strategy | range-calm | range-volatile | trend-calm | trend-volatile |
|---|---|---|---|---|
| PSAR SENSEX | +162 | +566 | +517 | +73 |
| ORB NIFTY | +220 | +575 | +569 | −124 |
| Pivot NIFTY | +36 | +432 | +236 | +31 |
| Judas NIFTY | +135 | +571 | +430 | +1 |

The 2022–23 weakness was spread across all regimes; it isn't a type of day we can avoid in advance.

**Decision:** keep all 8 strategies on in every regime. Next steps with better odds:
1. position sizing by risk, to cut drawdowns without skipping trades;
2. the live Quant tab's track record;
3. order-flow research once the depth data has built up (about Feb 2027).
