# Glossary

| Term | Meaning | Example |
|---|---|---|
| Basis point (bp) | 0.01% | NIFTY 25,000 → 1 bp = 2.5 points |
| ATM option | strike closest to the current index | NIFTY 25,140 → ATM strike 25,150 |
| Straddle | buy (or sell) the ATM call and put together | call 120 + put 95 = straddle 215 |
| Iron fly | sell the ATM straddle and buy cheaper options further out as protection | max loss is capped by the far options |
| Implied volatility (IV) | the move the option price "expects", per year | IV 14% ≈ ±0.9% a day expected |
| India VIX | NSE's index of NIFTY implied volatility | VIX 16 = market expects about ±1% a day |
| Open interest (OI) | number of option contracts still open | rising put OI = more puts being written |
| PCR | put OI ÷ call OI | 1.3 = 30% more puts than calls |
| Synthetic future | strike + call − put (put-call parity) | 25,150 + 120 − 95 = 25,175 |
| Feature | one number describing the market right now, given to a model | "VIX rose 0.4 in 15 minutes" |
| Label | what actually happened next, used to train and grade | "the next 30 minutes were busy" |
| Calibration | predicted % matches actual frequency | says 40% → happens about 40% of the time |
| AUC | how well a model ranks: 0.5 = coin toss, 1.0 = perfect | our busy model: 0.69 |
| Brier score | average squared gap between predicted odds and what happened; lower is better | |
| Walk-forward | train on the past, test on the next unseen period, roll forward | train ≤2024, test 2025; train ≤2025, test 2026 |
| Overfitting | settings that fit past noise; they look great in the past and fail later | picking the best of 2,772 versions |
| Sharpe ratio | yearly return ÷ yearly ups-and-downs; above 1 is good | |
| Deflated Sharpe | the chance the Sharpe is real after allowing for how many versions were tried | SAR SENSEX: 5% |
| Monte Carlo | reshuffle real results thousands of times to see possible bad years | 1-in-100 year drop −₹83k |
| Drawdown | fall from a previous peak in total profit | worst drop −₹43k |
| Regime | the market's current state: trend, range, volatile | |
| Lead-lag | one price moves just before another | SENSEX follows NIFTY's last minute |
| Order-flow imbalance | buy-side vs sell-side queue sizes in the order book | 3× more bids than offers |
| Gamma exposure | how option sellers' hedging dampens or amplifies moves | pinning at a big strike on expiry |
| Cointegration | two prices that wander but stay tied together over time | two private banks |
| HFT | high-frequency trading; reacting in microseconds from inside the exchange | not possible for us |
