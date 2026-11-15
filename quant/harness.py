"""Research harness: the same honest rules for every quant idea (plain-English guide: docs/02_how_we_test.md).

1. walk_forward()  train on the past, test on the next period, roll forward - never test on data used to choose.
2. Costs()         dhan_algo's real charge model + slippage, per option / futures fill.
3. deflated_sharpe() / haircut()  if we tried N versions, the best one looks better than it really is
                   (luck). These shrink the best result for the number of tries before we believe it.
4. summary()       one line per strategy: net, trades, win %, worst drop, Sharpe, by year.

Example
    from quant.harness import walk_forward, deflated_sharpe
    for train, test in walk_forward(days, train_years=2, test_months=6):
        model = fit(df[df.day.isin(train)]); evaluate(model, df[df.day.isin(test)])
    deflated_sharpe(sharpe=1.4, n_trials=48, n_obs=750)   # -> 0.62: only 62% sure it beats zero after 48 tries
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT.parent / "dhan_algo"


# ------------------------------------------------------------------ 1. walk-forward
def walk_forward(days, train_years: float = 2, test_months: int = 6, expanding: bool = True):
    """Yield (train_days, test_days). expanding=True keeps all history in train; False uses a rolling window.
    Example with days Nov 2021 - Oct 2026, 2y train, 6m test:
      train Nov21-Oct23 -> test Nov23-Apr24; train Nov21-Apr24 -> test May24-Oct24; ... (6 tests)."""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(days).unique())))
    start = days[0] + pd.DateOffset(months=int(train_years * 12))
    while start <= days[-1]:
        end = start + pd.DateOffset(months=test_months)
        lo = days[0] if expanding else start - pd.DateOffset(months=int(train_years * 12))
        train = days[(days >= lo) & (days < start)]
        test = days[(days >= start) & (days < end)]
        if len(test):
            yield train, test
        start = end


# ------------------------------------------------------------------ 2. costs
@dataclass
class Costs:
    """Real charges (brokerage, STT, exchange, GST, stamp) from dhan_algo + slippage in ticks per fill."""
    slip_ticks: int = 1
    tick: float = 0.05

    def __post_init__(self):
        sys.path.insert(0, str(ALGO))
        from algo.config import load_settings
        from algo.core.events import Instrument, Side
        from algo.portfolio.charges import ChargeModel
        self._cm, self._I, self._S = ChargeModel(load_settings()["charges"]), Instrument, Side

    def fill(self, index: str, side: str, qty: int, price: float, kind: str = "OPTIDX") -> float:
        """Charges of one fill in Rs. side = 'BUY' / 'SELL'."""
        inst = self._I("x", "NSE_FNO" if index == "NIFTY" else "BSE_FNO", "x", kind, lot_size=qty)
        return self._cm.compute(inst, self._S.BUY if side == "BUY" else self._S.SELL, qty, price)

    def round_trip(self, index: str, qty: int, entry: float, exit_: float, long: bool = True, kind: str = "OPTIDX"):
        """Net Rs of one position: slippage against us on both fills + charges on both fills.
        Example: buy 65 NIFTY CE at 120, sell at 130 -> gross 650, minus ~2 ticks x 65 slippage and ~Rs 60 charges."""
        s = self.slip_ticks * self.tick
        e, x = (entry + s, exit_ - s) if long else (entry - s, exit_ + s)
        gross = (x - e) * qty if long else (e - x) * qty
        ch = self.fill(index, "BUY" if long else "SELL", qty, e, kind) + self.fill(index, "SELL" if long else "BUY", qty, x, kind)
        return gross - ch


# ------------------------------------------------------------------ 3. multiple testing
def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Best Sharpe you'd expect from n_trials strategies with NO real edge (per-observation units)."""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    z = lambda p: math.sqrt(2) * _inv_erf(2 * p - 1)                        # noqa: E731
    return ((1 - g) * z(1 - 1 / n_trials) + g * z(1 - 1 / (n_trials * math.e))) / math.sqrt(n_obs)


def _inv_erf(y: float) -> float:
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                    periods_per_year: int = 250) -> float:
    """Probability (0-1) that the true Sharpe is above zero AFTER allowing for n_trials tries (Bailey & Lopez de
    Prado). sharpe = annualised; n_obs = number of daily returns. Rule of thumb: believe it only above ~0.95."""
    sr = sharpe / math.sqrt(periods_per_year)
    sr0 = expected_max_sharpe(n_trials, n_obs)
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return _norm_cdf((sr - sr0) * math.sqrt(n_obs - 1) / den)


def haircut(best_net: float, all_nets: list[float]) -> float:
    """Simple 'luck discount': best result minus how far the typical try sits above zero by chance.
    Example: best +Rs 74k among tries whose median is -Rs 10k and spread Rs 30k -> treat it as roughly +Rs 44k."""
    a = np.asarray(all_nets, dtype=float)
    if len(a) < 5:
        return best_net
    return float(best_net - max(np.std(a), 0.0))


# ------------------------------------------------------------------ 4. summary
def summary(daily_net: pd.Series, n_trials: int = 1) -> dict:
    """daily_net: Rs per trading day (0 on days without trades), DatetimeIndex."""
    d = daily_net.astype(float)
    eq = d.cumsum()
    sharpe = float(d.mean() / d.std() * math.sqrt(250)) if d.std() > 0 else 0.0
    out = {"net": float(d.sum()), "days": len(d), "worst_drop": float((eq - eq.cummax()).min()),
           "sharpe": round(sharpe, 2),
           "by_year": d.groupby(d.index.year).sum().round(0).to_dict()}
    if n_trials > 1:
        out["deflated_sharpe_prob"] = round(deflated_sharpe(sharpe, n_trials, len(d),
                                                             skew=float(d.skew()), kurt=float(d.kurt() + 3)), 3)
    return out
