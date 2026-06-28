"""
Utilities for aggregating period returns.

- compounds daily returns into inter-rebalance period returns
- computes per-ticker and market period excess returns,
- computes risk-free period returns from daily EFFR fixings.


Naming convention
-----------------
- Period return at rebalance date t_k means the return realised over  the inter-rebalance period (t_{k-1}, t_k]
(the first rebalance date has NaN).

-Daily return at a trading day means the return realised over the calendar days from the previous trading day to the current one.


Public API
----------
* function `compound_daily_returns_into_periods(...)`
        compound a daily-return series into period returns

* function `compute_rf_period_returns(...)`
        convenience wrapper around the above, for risk-free returns

* function `compute_period_returns(...)`
        per-ticker period returns computed from a (date, ticker) price panel

* function `compute_period_excess_returns(...)`
        per-ticker period returns minus the risk-free period return, indexed by (date, ticker)

"""

from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def compound_daily_returns_into_periods(
    period_end_dates: pd.DatetimeIndex,
    daily_returns: pd.Series,
) -> pd.Series:
    """Compound daily returns into per-period returns ending at each date.

    For each i >= 1, the value at `period_end_dates[i]` of the returned series is `prod(1 + r_d) - 1`
    over all daily returns `r_d` with index in `(period_end_dates[i-1], period_end_dates[i]]`.
    (`period_end_dates[0]` gets NaN since it has no preceding period). 
    A period with no daily observations also gets NaN, and a warning is logged via the
    module-level logger.

    Parameters
    ----------
    period_end_dates
        End dates of the periods, strictly increasing. Duplicates are not
        allowed (the interval for a duplicate would be empty).
    daily_returns
        Return series indexed by trading date, where the value at date t is
        the return from the previous trading date to t.

    Returns
    -------
    pd.Series
        Indexed by `period_end_dates`, named "period_returns".
    """
    daily_returns = (
        daily_returns.sort_index()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    out = pd.Series(index=period_end_dates, dtype=float, name="period_returns")
    for i in range(1, len(period_end_dates)):
        t_prev, t = period_end_dates[i - 1], period_end_dates[i]
        segment = daily_returns.loc[
            (daily_returns.index > t_prev) & (daily_returns.index <= t)
        ]
        if len(segment):
            out.iloc[i] = float(np.prod(1.0 + segment.to_numpy()) - 1.0)
        else:
            logger.warning(
                f"no daily return for {daily_returns.name} between {t_prev} and {t} (exclusive of "
                f"{t_prev}, inclusive of {t}); period return set to NaN",
                stacklevel=2,
            )         
    return out


def compute_rf_period_returns(
    daily_rf_returns: pd.Series,
    period_end_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Compound daily risk-free returns into per-period
    returns ending at `period_end_dates`.

    This function is a wrapper; it is equivalent to calling 
    `compound_daily_returns_into_periods(period_end_dates, daily_rf_returns)` 
    and is provided so that the risk-free aggregation has its own function.
    """
    return compound_daily_returns_into_periods(period_end_dates, daily_rf_returns)


def _wide_period_returns(
    prices: pd.Series,
    period_end_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build Wide DataFrame of inter-period returns, one column per ticker."""
    return (
        prices.unstack("ticker")
        .sort_index()
        .reindex(period_end_dates)
        .pct_change(fill_method=None)
    )


def compute_period_returns(
    prices: pd.Series,
    period_end_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Compute inter-rebalance period returns, indexed by (date, ticker).
    The first rebalance date has NaN for every ticker (no prior price).
    """
    result = (
        _wide_period_returns(prices, period_end_dates)
        .stack(future_stack=True)
        .rename("period_returns")
    )
    result.index.set_names(["date", "ticker"], inplace=True)
    return result


def compute_period_excess_returns(
    prices: pd.Series,
    period_end_dates: pd.DatetimeIndex,
    rf_period_returns: pd.Series,
) -> pd.Series:
    """Per-ticker inter-rebalance period excess returns
    (return - risk-free return), indexed by (date, ticker).
    """
    returns_wide = _wide_period_returns(prices, period_end_dates)
    excess_wide = returns_wide.sub(rf_period_returns, axis="index")
    result = (
        excess_wide.stack(future_stack=True).rename("period_excess_returns")
    )
    result.index.set_names(["date", "ticker"], inplace=True)
    return result