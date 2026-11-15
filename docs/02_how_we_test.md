# 02 · How we test (so we don't fool ourselves)

Most strategies that look great in a backtest fail live. These rules, built into `quant/harness.py`, catch the
usual reasons.

## 1. Never judge on the data you used to choose
**Walk-forward:** choose settings on older data, then test on the next period the model has never seen, and roll
forward.
Example: train Nov 2021 – Dec 2024, choose settings on 2025, then test once on 2026.

Why it matters, from our own results: the 7 paper strategies made +₹6.2 lakh on Nov 2023 – Oct 2026 (the years
they were tuned on), but about **₹0** on Nov 2021 – Oct 2023, which they had never seen.

## 2. Always include real costs
Every result uses dhan_algo's charge model (brokerage, STT, exchange fees, GST, stamp duty) plus 1 tick of
slippage per fill.
Example: buy 65 NIFTY calls at ₹120 and sell at ₹130. Gross +₹650, after slippage and charges **+₹577**.
For strategies with many small trades, costs decide everything. Selling iron flies every hour earned about zero
before costs and lost about ₹230 per trade after them.

## 3. Check the percentages are honest (calibration)
If the model says "40% bullish" a thousand times, the market should go bullish about 400 times.
Example from the 2026 test (SENSEX): said 43% → happened 41%; said 26% → happened 26%. Good calibration means
you can take the number at face value. The live scorecard keeps checking this every day.

## 4. Discount for luck (multiple testing)
If you try many versions, the best one looks good **by chance**.
Example: flip 2,772 coins 100 times each; the luckiest coin may land heads 65 times, yet it's still a fair coin.

`deflated_sharpe()` turns this into a probability: "how sure are we the real edge is above zero, given how many
versions we tried?"

| Strategy | Sharpe (return ÷ risk, per year) | Versions tried | Probability it's real |
|---|---|---|---|
| a strategy tried once | 1.4 | 1 | 99% |
| the same result after 48 tries | 1.4 | 48 | 57% |
| SAR SENSEX (5-year engine trades) | 1.15 | 2,772 | **5%** |

So SAR SENSEX's backtest is **not proof**; paper trading is its real test. We rule nothing in on backtest
alone. We want an idea to work on unseen data **and** in many nearby settings, not just one lucky one.

## 5. A real edge looks like a hill, not a spike
If stop 50 works but stops 45 and 55 lose, it's a spike (luck). If a whole region of settings works, it's a hill.
We report the share of nearby settings that won in both periods.

## 6. Monte Carlo for risk
We reshuffle the real daily results 10,000 times to see bad years that could have happened.
Example: all 7 paper strategies over 5 years. The actual worst drop was −₹86k; in 1 of 100 simulated years it
reaches about −₹83k to −₹90k, so plan capital for that, not for the best-case backtest.
