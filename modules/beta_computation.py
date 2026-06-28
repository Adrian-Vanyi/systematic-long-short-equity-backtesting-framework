"""Rolling estimation of CAPM beta of each ticker on the rebalance dates.

At each rebalance date, beta is the slope coefficient from a rolling
OLS regression of inter-rebalance excess returns on market excess
returns over the previous periods. The intercept (alpha) is not
extracted, because only the slope is needed for portfolio-beta computation.

The market returns proxy is passed in as a Series of daily simple returns
(the backtest uses SPY adjusted-close returns), where daily means between consecutive trading days.

Public API
----------
* function `compute_capm_betas(...)`
        estimate CAPM betas for every ticker at every rebalance date.

"""

from __future__ import annotations
import logging
import pandas as pd

from modules import period_returns as pr

logger = logging.getLogger(__name__)


def _beta_regression_using_rebalance_periods(
    rebalance_dates: pd.DatetimeIndex,
    tickers_period_excess_returns: pd.Series,
    market_period_excess_returns: pd.Series,
    rolling_periods_for_beta_regression: int
) -> pd.Series:
    """Rolling-window estimation of CAPM beta of each ticker at each rebalance date.
    Tickers with insufficient non-NaN history period returns are dropped.
    """
    if rolling_periods_for_beta_regression < 2:
        raise ValueError("rolling_periods_for_beta_regression must be >= 2")

    n = rolling_periods_for_beta_regression

    excess_wide = (
        tickers_period_excess_returns.unstack("ticker").sort_index()
    )
    market_excess = market_period_excess_returns.sort_index()

    covs = excess_wide.rolling(n, min_periods=n).cov(market_excess)
    var_market = market_excess.rolling(n, min_periods=n).var()
    betas_wide = covs.div(var_market, axis=0).reindex(rebalance_dates)

    betas = betas_wide.stack(future_stack=True).rename("beta").dropna()
    betas.index.set_names(["date", "ticker"], inplace=True)
    return betas


def compute_capm_betas(
    rf_daily_returns: pd.Series,
    market_daily_returns: pd.Series,
    training_rebalance_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    prices: pd.Series,
    rolling_periods_for_beta_regression: int
) -> pd.Series:
    """Estimate CAPM betas for every ticker at every rebalance date.

    `training_rebalance_dates` must contain `rebalance_dates` plus
    at least `rolling_periods_for_beta_regression` preceding rebalance dates (in order for a full period window 
    of `rolling_periods_for_beta_regression` to be available).

    Parameters
    ----------
    rf_daily_returns
        Risk-free returns from one trading date to the next, indexed
        by trading date.
    market_daily_returns
        Market-proxy daily simple returns (e.g. SPY adjusted-close
        returns), indexed by trading date.
    training_rebalance_dates, rebalance_dates, prices
        Standard rebalance grid + per-ticker price data.
    rolling_periods_for_beta_regression
        Must be greater or equal to 2.

    Returns
    -------
    pd.Series
        Indexed by (date, ticker), named "beta". 
        Ticker-date pairs where tickers have insufficient price history to estimate their beta are absent.
    """
    if not set(rebalance_dates).issubset(set(training_rebalance_dates)):
        raise ValueError(
            "rebalance_dates must be a subset of training_rebalance_dates"
        )

    rf_period_returns = pr.compute_rf_period_returns(
        rf_daily_returns, training_rebalance_dates
    )

    market_period_returns = pr.compound_daily_returns_into_periods(
        training_rebalance_dates, market_daily_returns,
    )
    market_period_excess = (
        market_period_returns - rf_period_returns
    ).rename("market_period_excess_returns")

    tickers_period_excess = pr.compute_period_excess_returns(
        prices, training_rebalance_dates, rf_period_returns,
    )

    return _beta_regression_using_rebalance_periods(
        rebalance_dates,
        tickers_period_excess,
        market_period_excess,
        rolling_periods_for_beta_regression=rolling_periods_for_beta_regression
    )