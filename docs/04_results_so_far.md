# 04 · Results so far (all on data the model never saw unless stated)

| Date | Test | Result | Lesson |
|---|---|---|---|
| 15 Nov | Next-5-min direction (NIFTY, SENSEX) | Honest odds, but they rarely move from normal (no ≥65% readings in 2026); only 0.5–1.2% better than time-of-day odds | 5-minute direction is mostly noise |
| 15 Nov | SENSEX next 1 minute | Real lead-lag from NIFTY: ≥60% readings 3 a day, right 62%. Buying options on them: +₹28.9k before costs, **−₹31.4k after** (2026) | A real edge, smaller than option costs |
| 15 Nov | Busy/quiet in the next 15/30/60 min | **AUC 0.69**: top-10% readings → about 47% busy (normal 25%); bottom-10% → about 5% | The model really knows busy vs quiet |
| 15 Nov | Buying straddles on busy readings | NIFTY 15 min −₹80k (2026) | Option prices already include it (IV) |
| 15 Nov | Selling iron flies in quiet windows (hold 30–60 min) | NIFTY −₹68k, SENSEX −₹94k (2026) | 8 orders ≈ ₹230 per trade kills it |
| 15 Nov | Sell one iron fly at 09:30 on quiet-rated days | Model halves losses (NIFTY −₹17k vs −₹35k every day) but still negative | Can't fix a losing trade by selecting days |
| 15 Nov | Filter for the paper strategies (skip low-odds trades) | No skill (AUC 0.51–0.53); skipping would have cut profit | Leave the strategies alone |
| 15 Nov | Adding India VIX + synthetic futures | Almost no change | Option prices already contain VIX's information |
| 15 Nov | **Research #1 VRP**: sell hedged ATM straddle 1–3 days before weekly expiry, hold to expiry | Best on training +₹55k → unseen NIFTY +₹9k, SENSEX +₹2k; luck-adjusted 0–1% (see `06_vrp.md`) | Weekly option premium isn't reliably rich after costs; results depend on the period |
| 15 Nov | **Research #2 regimes**: skip strategies in regimes they lost in before (rule / clustering / hidden Markov), walk-forward 2023–26 | Hurts: +₹6.78L → +₹3.83L to +₹5.06L; helped 0–1 of 4 years (see `07_regimes.md`) | Strategies earn in most regimes; keep them all on |

Related dhan_algo results that shape the plan:
- **5 years:** the 7 paper strategies ≈ ₹0 on Nov 2021 – Oct 2023 (never tuned on) vs +₹6.2L on the tuned years.
  Edges depend on the market regime → research item "regime detection".
- **Mean reversion:** only "failed breakout of yesterday's high/low" survived (NIFTY, +₹29k unseen); added to
  paper 15 Nov.
- **SAR SENSEX:** passes unseen data, but a 5% luck-adjusted probability (see `02_how_we_test.md`). Paper decides.

Files: `reports/verdict.txt`, `step1.txt`, `step1b.txt`, `quietday.txt`, `step2_tests.txt`, `step2_quietday.txt`.
