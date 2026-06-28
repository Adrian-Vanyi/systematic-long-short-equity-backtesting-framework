
"""Margin requirement computation and uniform-shrinkage cure.

Two distinct margin requirements:

- Maintenance MR: ongoing requirement against existing positions;
  a haircut-weighted sum of long and short market values.

- Rebalance MR: amount of equity needed to execute a portfolio
  transition, accounting for the maintenance margin released by
  reductions/closures and the initial margin required by openings/
  increases, plus the trading fees incurred along the way.

When the rebalance MR (or maintenance MR) exceeds equity, the
portfolio can be uniformly shrunk via a scaling factor `alpha` (value in [0, 1])
applied to all positions. Function `compute_shrinking_factor` finds the
largest feasible `alpha` via bisection (the largest feasible factor shrinks the least, as it is closer to 1 than any other feasible factor).

Public API
----------
* class `MRHaircuts`
        per (date, ticker) panel of initial/maintenance haircuts.

* function `compute_maintenance_MR(...)`
        margin requirement of the current portfolio (haircut-weighted long + short market values).

* function `compute_rebalance_MR(...)`
        margin requirement to execute portfolio rebalance (current -> intended): maintenance released + initial required + fees.

* function `compute_shrinking_factor(...)`
        largest `alpha` in [0, 1] such that the rebalance MR for the alpha-shrunk intended portfolio is complied with by current equity

"""

from __future__ import annotations
import logging
import pandas as pd

from modules import trading_utils as tu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Haircut panel 
# ---------------------------------------------------------------------------

class MRHaircuts:
    """Per (date, ticker) panel of initial/maintenance haircuts.

    Stored as scalar defaults; per-(date, ticker)  overrides may be set via method `set_overrides`. 
    Consumers query via method `at_date` rather than reaching into internal storage.

    Defaults:
      * initial:     50% / 50% (Reg-T baseline)
      * maintenance: 25% / 30% (FINRA floors)
    """
    def __init__(
        self,
        backtest_dates: pd.DatetimeIndex,
        tickers: list[str],
        initial_long_haircut: float = 0.5,
        initial_short_haircut: float = 0.5,
        maintenance_long_haircut: float = 0.25,
        maintenance_short_haircut: float = 0.30,
        overrides: pd.DataFrame | None = None,
    ):
        self.backtest_dates = backtest_dates
        self.tickers = tickers
        self.initial_long_haircut = float(initial_long_haircut)
        self.initial_short_haircut = float(initial_short_haircut)
        self.maintenance_long_haircut = float(maintenance_long_haircut)
        self.maintenance_short_haircut = float(maintenance_short_haircut)
        self._overrides = overrides

    # accessor
    def at_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """Return a per-ticker DataFrame of haircuts at one date.
        Columns: ``initial_long``, ``initial_short``,
        ``maintenance_long``, ``maintenance_short``. Indexed by ticker.
        """
        df = pd.DataFrame(
            {
                "initial_long": self.initial_long_haircut,
                "initial_short": self.initial_short_haircut,
                "maintenance_long": self.maintenance_long_haircut,
                "maintenance_short": self.maintenance_short_haircut,
            },
            index=pd.Index(self.tickers, name="ticker"),
        )
        if self._overrides is not None:
            try:
                day = self._overrides.xs(date, level=0)
                df.update(day)
            except KeyError:
                pass
        return df

    def set_overrides(self, overrides: pd.DataFrame) -> None:
        """Set per-(date, ticker) haircut overrides."""
        if not isinstance(overrides.index, pd.MultiIndex):
            raise ValueError("overrides must be indexed by (date, ticker)")
        self._overrides = overrides


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prices_at_date(prices: pd.Series, date: pd.Timestamp) -> pd.Series:
    return prices.xs(date, level=0).astype(float)


def _validate_int_shares(shares: pd.Series, name: str) -> pd.Series:
    """Return the input cast to int"""
    if shares.isna().any():
        raise ValueError(
            f"{name} contains NaN; share counts values must be of type integer"
        )
    return shares.astype(int)


# ---------------------------------------------------------------------------
# Maintenance MR
# ---------------------------------------------------------------------------

def compute_maintenance_MR(
    date: pd.Timestamp,
    mr_haircuts: MRHaircuts,
    prices: pd.Series,
    current_shares: pd.Series,
) -> float:
    """Compute margin requirement of the current portfolio."""
    q = _validate_int_shares(current_shares, "current_shares")
    tickers = q.index
    prices = _prices_at_date(prices, date).reindex(tickers, fill_value=0.0)
    haircuts = mr_haircuts.at_date(date).reindex(tickers, fill_value=0.0)

    mr_longs = (haircuts["maintenance_long"] * prices * q.clip(lower=0)).sum()
    mr_shorts = (haircuts["maintenance_short"] * prices * (-q.clip(upper=0))).sum()
    return float(mr_longs + mr_shorts)


# ---------------------------------------------------------------------------
# Rebalance MR (see §3.4 of the documentation )
# ---------------------------------------------------------------------------

def compute_rebalance_MR(
    date: pd.Timestamp,
    mr_haircuts: MRHaircuts,
    prices: pd.Series,
    current_shares: pd.Series,
    intended_new_shares: pd.Series,
    trading_costs: tu.TradingCosts,
) -> float:
    """Rebalance margin requirement to transition from `current_shares` to `intended_new_shares`."""
    q = _validate_int_shares(current_shares, "current_shares")
    qp = _validate_int_shares(intended_new_shares, "intended_new_shares")

    all_tickers = q.index.union(qp.index)
    q = q.reindex(all_tickers, fill_value=0)
    qp = qp.reindex(all_tickers, fill_value=0)

    px = _prices_at_date(prices, date).reindex(all_tickers, fill_value=0.0)
    haircuts = mr_haircuts.at_date(date).reindex(all_tickers, fill_value=0.0)
    h_lm = haircuts["maintenance_long"]
    h_sm = haircuts["maintenance_short"]
    h_li = haircuts["initial_long"]
    h_si = haircuts["initial_short"]

    costs_today = trading_costs.at_date(date).reindex(all_tickers, fill_value=0.0)
    fees = costs_today["commission_per_share"]

    # Maintenance MR of the current portfolio
    mr_maint = float(
        (h_lm * px * q.clip(lower=0)).sum()
        + (h_sm * px * (-q.clip(upper=0))).sum()
    )

    # Trade-case partition (mutually exclusive)
    unchanged = q == qp
    open_long = (q == 0) & (qp > 0)
    open_short = (q == 0) & (qp < 0)
    close_long = (qp == 0) & (q > 0)
    close_short = (qp == 0) & (q < 0)
    reduce_long = (q > 0) & (qp > 0) & (qp < q)
    reduce_short = (q < 0) & (qp < 0) & (qp.abs() < q.abs())
    increase_long = (q > 0) & (qp > 0) & (qp > q)
    increase_short = (q < 0) & (qp < 0) & (qp.abs() > q.abs())
    switch_to_long = (q < 0) & (qp > 0)
    switch_to_short = (q > 0) & (qp < 0)

    masks = [
        unchanged, open_long, open_short, close_long, close_short,
        reduce_long, reduce_short, increase_long, increase_short,
        switch_to_long, switch_to_short,
    ]
    mask_sum = sum(m.astype(int) for m in masks)
    if not (mask_sum == 1).all():
        raise AssertionError(
            "trade-case partition is not a true partition; coverage gap or "
            "overlap detected"
        )

    delta = 0.0

    # Close existing long / short  (fee minus maintenance released)
    delta += ((fees - h_lm * px) * q)[close_long].sum()
    delta += ((fees - h_sm * px) * (-q))[close_short].sum()

    # Open new long / short
    delta += ((fees + h_li * px) * qp)[open_long].sum()
    delta += ((fees + h_si * px) * (-qp))[open_short].sum()

    # Reduce long / short
    delta += ((fees - h_lm * px) * (q - qp))[reduce_long].sum()
    delta += ((fees - h_sm * px) * (q.abs() - qp.abs()))[reduce_short].sum()

    # Increase long / short
    delta += ((fees + h_li * px) * (qp - q))[increase_long].sum()
    delta += ((fees + h_si * px) * (qp.abs() - q.abs()))[increase_short].sum()

    # Switch from short to long: close short leg, then open long leg
    delta += ((fees - h_sm * px) * (-q))[switch_to_long].sum()
    delta += ((fees + h_li * px) * qp)[switch_to_long].sum()

    # Switch from long to short: close long leg, then open short leg
    delta += ((fees - h_lm * px) * q)[switch_to_short].sum()
    delta += ((fees + h_si * px) * (-qp))[switch_to_short].sum()

    return float(mr_maint + delta)


# ---------------------------------------------------------------------------
# Optimal shrinkage factor estimation (using bisection) (see §3.4.4 of the documentation)
# ---------------------------------------------------------------------------

def compute_shrinking_factor(
    equity: float,
    date: pd.Timestamp,
    mr_haircuts: MRHaircuts,
    prices: pd.Series,
    trading_costs: tu.TradingCosts,
    current_shares: pd.Series,
    intended_new_shares: pd.Series,
    precision: float = 1e-6,
) -> float:
    all_tickers = intended_new_shares.index.union(current_shares.index)
    intended = intended_new_shares.reindex(all_tickers, fill_value=0)
    current = current_shares.reindex(all_tickers, fill_value=0)

    def f(alpha: float) -> float:
        shrunk = tu.truncate_shares_to_int(alpha * intended)
        return compute_rebalance_MR(
            date = date,
            mr_haircuts = mr_haircuts,
            prices = prices,
            current_shares = current,
            intended_new_shares = shrunk,
            trading_costs = trading_costs
        )

    if f(1.0) <= equity:
        return 1.0  # no shrinkage needed

    low, hi = 0.0, 1.0
    while hi - low > precision:
        mid = (low + hi) / 2.0
        if f(mid) <= equity:
            low = mid
        else:
            hi = mid
    return low