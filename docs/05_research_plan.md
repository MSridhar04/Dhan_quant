# 05 · Research plan: quant strategies, explained simply

Order = most promising first. Each one goes through `02_how_we_test.md` (unseen data, real costs, luck discount).

## 1. Volatility risk premium (VRP) - next
**Idea:** option prices imply how much the market *expects* to move (implied volatility). On average the market
*actually* moves less, because sellers demand a premium for taking the risk. Selling that premium pays most of the
time, but loses big in shocks.
**Example:** the ATM straddle prices a ±1.2% move by expiry; the index ends up moving 0.8% → straddle sellers keep
the difference.
**Our twist:** the busy/quiet model and VIX as a *risk switch*. Sell only when implied volatility is rich and the
model says calm; stay out before busy periods. Hold to expiry or for several days, not one hour: our tests showed
intraday selling dies to costs.

## 2. Regime detection - next
**Idea:** markets alternate between trending, ranging and volatile states. A statistical model (e.g. a "hidden
Markov model") labels each day from returns, ranges and VIX.
**Why for us:** our paper strategies earned in 2024–26 but not in 2022–23. If they only work in certain regimes,
switching them off (or sizing down) in the wrong regime could cut the lean years.
**Example output:** "Today: RANGE regime (70% sure) → breakout strategies historically −₹40 per trade in this
regime → half size."

## 3. Lead-lag
**Idea:** some prices move a moment before others. We already measured NIFTY → SENSEX (62% next-minute accuracy).
Heavyweights → NIFTY is the next to test.
**Catch:** the move is small, so options are too expensive. It needs futures (lower costs, about ₹1.5–2L margin
per lot).

## 4. Order-flow imbalance (from about Feb 2027)
**Idea:** if buyers queue much more than sellers in the order book, price tends to tick up over the next 1–5
minutes. It's one of the strongest short-term signals in research.
**Status:** depth is being recorded since 16 Nov 2026; we need 2–3 months of it.

## 5. Dealer positioning (gamma exposure)
**Idea:** option sellers hedge. When they are "long gamma" their hedging dampens moves (price gets pinned near big
strikes); when "short gamma" it amplifies moves. We can estimate this from open interest per strike.
**Use:** explain expiry-day pinning; another input for the regime and busy models.

## 6. Statistical arbitrage (pairs)
**Idea:** two related stocks (e.g. two private banks) usually move together. When they drift apart unusually far,
bet on them coming back ("cointegration").
**Catch:** needs shorting and has costs on two legs; intraday only for most retail accounts.

## 7. Calendar and event effects
Expiry days, RBI policy days, budget, month-end flows. Cheap to test; usually small but steady effects.
Example we already saw: Tuesday (NIFTY expiry) was the only weekday where selling iron flies wasn't negative.

## 8. Portfolio construction
**Idea:** instead of 1 lot of everything, size each strategy by its risk so no single one dominates bad days
("volatility targeting", "risk parity").
**Example:** EMA SENSEX swings twice as much as Judas → half its size. This can cut drawdowns without any new
signal.
