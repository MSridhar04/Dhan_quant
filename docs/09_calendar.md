# 09 · Research #4: calendar and event effects

Code: `quant/research/calendar_fx.py` · results: `reports/calendar.txt`

## The question
Do our 8 paper strategies do systematically better or worse on certain kinds of days? For example on Fridays, on
expiry day, at the end of the month, or on RBI policy days.

## Kinds of day tested
| Family | Buckets |
|---|---|
| Weekday | Mon … Fri |
| Expiry | expiry day, 1 day before, 2 days before, 3+ days before (each strategy's own index) |
| Month | first 3 trading days, last 3, mid-month |
| Events | RBI policy decision, Union Budget, 2024 election result, the Indian day after a US Fed decision, normal day |

Event dates: RBI 2021–2025 and Fed 2021–2026 from the public schedules. Budget days only when they were weekday
sessions. RBI 2026 is left out because the exact dates weren't confirmed; better to leave them out than guess.

## How it was tested
1. **Descriptive:** average ₹ per trade in each bucket, for the picking years (≤2024) and unseen 2025–26. An effect
   only counts if it has the same direction in both, with at least 10 trades each.
2. **Walk-forward by year:** skip a strategy on kinds of day it lost on in **earlier** years; compare with trading
   every day.

## Results (15 Nov 2026)
**How much the index moves (NIFTY average day range):** election result 863 bp · Budget 213 bp · RBI 110 bp · day
after Fed 102 bp · normal day 98 bp. Budget days move about twice a normal day; RBI and Fed days only slightly more.

**Skipping on "bad" days hurts, in every family:**

| Family | All trades 2023–26 | With skip rule | Years it helped |
|---|---|---|---|
| Weekday | +₹6.54L | +₹4.22L | 0 of 4 |
| Expiry | +₹6.54L | +₹4.88L | 1 of 4 |
| Month | +₹6.54L | +₹4.01L | 1 of 4 |
| Events | +₹6.54L | +₹4.46L | 1 of 4 |

**Kinds of day that lost money in both periods:** only 5, and 4 of them are tiny (−₹16 to −₹66 per trade in
2025–26). The one standout:
- **EMA SENSEX on Fridays:** −₹243 per trade (57 trades) in the picking years, then −₹410 per trade (64 trades)
  unseen. By year: 2023 −₹16k, 2024 +₹3k, 2025 −₹4k, 2026 −₹22k.

**Why not act on it yet:** there are about 140 strategy × kind-of-day combinations. A handful staying negative is
roughly what luck alone produces, and the walk-forward test shows that acting on past "bad days" cost money overall.

**Decision:** no calendar rules for now. **Watch EMA SENSEX on Fridays in paper trading.** If it keeps losing over
the next few months, skipping Fridays for that one strategy is a simple, low-risk rule to add.
