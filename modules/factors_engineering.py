"""Cross-sectional factor engineering.

Two factors are computed at each rebalance date:

- Momentum: log(P_{t-b} / P_{t-(b+K)}) where `b` is a buffer (e.g. 1 trading month) 
  intended to skip the most recent month's noise,  and `K` is the estimation window (e.g. 4 trading months).

- Volatility: annualized standard deviation of daily log-returns (i.e. between consecutive trading days)
  over a rolling window: sqrt(252) x std(log_returns, ddof=1).

(Both these factors are computed from log returns).

Both factors are winsorized, and then z-scored. (see §8.3 documentation)


Public API
----------
* function `compute_momentum_at_training_rebalance_dates(...)`
        log-momentum panel across rebalance dates; restricted to per-date universes if given.

* function `compute_momentum_at_date_for_tickers(...)`
        single-date version of the above, for a ticker subset.

* function `compute_volatility_at_training_rebalance_dates(...)`
        annualized log-return volatility panel across rebalance dates; restricted to per-date universes if given.

* function `compute_volatility_at_date_for_tickers(...)`
        single-date version of the above, for a ticker subset.

* function `compute_factors_at_training_rebalance_dates(...)`
        combined momentum + volatility panel; inner-joined on (date, ticker), with optional winsorize/z-score preprocessing.

* function `compute_factors_at_date_for_tickers(...)`
        single-date version of the above, for a ticker subset.

* function `winsorize_per_date(...)` 
        clips the cross-section at the 1st/99th percentile per date.

* function `cross_sectional_zscore_per_date(...)` 
        normalizes to mean 0, std 1 per date. 

"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-sectional preprocessing
# ---------------------------------------------------------------------------

def _per_date_transform(data: pd.Series | pd.DataFrame, transform_fn):
    """Apply a per-date cross-sectional transform.

    Polymorphic over Series/DataFrame, with or without a (date, ticker)
    MultiIndex. For non-MultiIndex inputs, the entire input is treated
    as one cross-section.
    """
    has_date_level = (
        isinstance(data.index, pd.MultiIndex) and "date" in data.index.names
    )
    if has_date_level:
        return data.groupby(level="date", observed=False).transform(transform_fn)
    if isinstance(data, pd.DataFrame):
        return data.apply(transform_fn)
    return transform_fn(data)


def cross_sectional_zscore_per_date(data: pd.Series | pd.DataFrame):
    """Cross-sectional z-score per date: (x - mean) / std`"""
    return _per_date_transform(data, lambda x: (x - x.mean()) / x.std())


def winsorize_per_date(
    data: pd.Series | pd.DataFrame,
    limits: tuple[float, float] = (0.01, 0.99)
):
    """Cross-sectional winsorization per date: clip to the given quantile bounds 
    within each cross-section.
    """
    def _clip(x):
        return x.clip(x.quantile(limits[0]), x.quantile(limits[1]))
    
    return _per_date_transform(data, _clip)


# ---------------------------------------------------------------------------
# Restricting wide frames to per-date universes
# ---------------------------------------------------------------------------

def restrict_data_to_rebalance_dates_and_universes(
    factor_wide: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    factor_name: str,
    reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]] | None
) -> pd.Series:
    """Reindex a wide factor DataFrame (i.e, indexed by date with columns the tickers) to a (date, ticker)-indexed series from
    per-date universes.

    Per-ticker missing values are dropped (and affected (date, ticker) pairs are absent from the result).
    Callers that need to log such drops should compare the result's multi-index with the requested universes.
    """
    if reb_date_to_tickers_universe_dict is not None:
        missing_keys = [d for d in rebalance_dates if d not in reb_date_to_tickers_universe_dict]
        if missing_keys:
            raise KeyError(
                f"reb_date_to_tickers_universe_dict missing keys: "
                f"{[d.strftime('%Y-%m-%d') for d in missing_keys[:5]]}"
                f"{'...' if len(missing_keys) > 5 else ''}"
            )
        
    factor_wide = factor_wide.reindex(rebalance_dates)

    if not reb_date_to_tickers_universe_dict:
        return (
            factor_wide.stack(future_stack=True)
            .dropna()
            .rename(factor_name)
            .sort_index()
        )
    idx = pd.MultiIndex.from_tuples(
        [
            (date, ticker)
            for date in rebalance_dates
            for ticker in reb_date_to_tickers_universe_dict[date]
        ],
        names=["date", "ticker"],
    )
    return (
        factor_wide.stack(future_stack=True)
        .reindex(idx)
        .dropna()
        .rename(factor_name)
        .sort_index()
    )


# ---------------------------------------------------------------------------
# Wide factor computations (used by bulk and single-date variants)
# ---------------------------------------------------------------------------

def _momentum_wide(
    prices_wide: pd.DataFrame,
    rolling_window_trading_days: int,
    buffer_trading_days: int
) -> pd.DataFrame:
    """Wide DataFrame (i.e. indexed by date and columns are the tickers) of log-momentum at every available date,
      for every ticker in `prices_wide`.
    """
    return np.log(prices_wide.shift(buffer_trading_days)) - np.log(
        prices_wide.shift(buffer_trading_days + rolling_window_trading_days)
    )


def _annualized_volatility_wide(
    prices_wide: pd.DataFrame, rolling_window_trading_days: int
) -> pd.DataFrame:
    """Wide DataFrame (i.e. indexed by date and columns are the tickers) of annualized log-return volatility."""
    daily_log_returns = np.log(prices_wide) - np.log(prices_wide.shift(1))
    rolling_std = daily_log_returns.rolling(
        rolling_window_trading_days, min_periods=rolling_window_trading_days
    ).std(ddof=1)
    return np.sqrt(252) * rolling_std


def _slice_wide_at_date_for_tickers(
    wide: pd.DataFrame,
    date: pd.Timestamp,
    tickers: list[str],
    metric_name: str
) -> pd.Series:
    """Slice a wide factor DataFrame at one date for a subset of tickers,
    drop NaNs, return a Series indexed by ticker (sorted).
    """
    if date not in wide.index:
        raise ValueError(
            f"date {date.strftime('%Y-%m-%d')} not present in {metric_name} data"
        )
    available = [t for t in tickers if t in wide.columns]
    series = wide.loc[date, available].dropna()
    return series.sort_index().rename(metric_name)


# ---------------------------------------------------------------------------
# Public: momentum
# ---------------------------------------------------------------------------

def compute_momentum_at_training_rebalance_dates(
    prices: pd.Series,
    training_rebalance_dates: pd.DatetimeIndex,
    rolling_window_trading_days: int,
    buffer_trading_days: int,
    reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]] | None
) -> pd.Series:
    """Compute momentum at every training rebalance date.

    Restricted to per-date universes if provided; otherwise returns all
    (date, ticker) pairs with non-NaN momentum, for all tickers provided in `prices`.
    """
    prices_wide = prices.unstack()

    missing = [
        d.strftime("%Y-%m-%d")
        for d in training_rebalance_dates
        if d not in prices_wide.index
    ]
    if missing:
        raise ValueError(
            f"price data not available for training rebalance date(s): {missing}"
        )

    momentums_wide = _momentum_wide(
        prices_wide, rolling_window_trading_days, buffer_trading_days
    )
    return restrict_data_to_rebalance_dates_and_universes(
        momentums_wide,
        training_rebalance_dates,
        "momentum",
        reb_date_to_tickers_universe_dict
    )


def compute_momentum_at_date_for_tickers(
    date: pd.Timestamp,
    tickers: list[str],
    prices: pd.Series,
    rolling_window_trading_days: int,
    buffer_trading_days: int
) -> pd.Series:
    """Compute momentum at date `date` for the subset of tickers `tickers.`"""
    prices_wide = prices.unstack()
    momentums_wide = _momentum_wide(
        prices_wide, rolling_window_trading_days, buffer_trading_days
    )
    return _slice_wide_at_date_for_tickers(momentums_wide, date, tickers, "momentum")


# ---------------------------------------------------------------------------
# Public: volatility
# ---------------------------------------------------------------------------

def compute_volatility_at_training_rebalance_dates(
    prices: pd.Series,
    training_rebalance_dates: pd.DatetimeIndex,
    rolling_window_trading_days: int,
    reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]] | None
) -> pd.Series:
    """Computes the annualized log-return volatility at every training rebalance date.
    
    Restricted to per-date universes if provided; otherwise returns all
    (date, ticker) pairs with non-NaN momentum, for all tickers provided in `prices`.
    """
    prices_wide = prices.unstack()

    missing = [
        d.strftime("%Y-%m-%d")
        for d in training_rebalance_dates
        if d not in prices_wide.index
    ]
    if missing:
        raise ValueError(
            f"price data not available for training rebalance date(s): {missing}"
        )

    vols_wide = _annualized_volatility_wide(
        prices_wide, rolling_window_trading_days
    )
    return restrict_data_to_rebalance_dates_and_universes(
        vols_wide,
        training_rebalance_dates,
        "volatility",
        reb_date_to_tickers_universe_dict
    )


def compute_volatility_at_date_for_tickers(
    date: pd.Timestamp,
    tickers: list[str],
    prices: pd.Series,
    rolling_window_trading_days: int
) -> pd.Series:
    """Computes the annualized log-return volatility at a single date `date` for the subset of tickers `tickers`."""
    prices_wide = prices.unstack()
    vols_wide = _annualized_volatility_wide(
        prices_wide, rolling_window_trading_days
    )
    return _slice_wide_at_date_for_tickers(vols_wide, date, tickers, "volatility")


# ---------------------------------------------------------------------------
# Public: combined factor panel
# ---------------------------------------------------------------------------

def _process_factors(
    factors: pd.DataFrame,
    winsorize_factors_per_date: bool,
    z_score_factors_per_date: bool
) -> pd.DataFrame:
    if winsorize_factors_per_date:
        factors = winsorize_per_date(factors)
    if z_score_factors_per_date:
        factors = cross_sectional_zscore_per_date(factors)
    return factors


def compute_factors_at_training_rebalance_dates(
    prices: pd.Series,
    training_rebalance_dates: pd.DatetimeIndex,
    rolling_window_trading_days_for_momentum: int,
    buffer_trading_days_for_momentum: int,
    rolling_window_trading_days_for_volatility: int,
    reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]] | None,
    winsorize_factors_per_date: bool = True,
    z_score_factors_per_date: bool = True
) -> pd.DataFrame:
    """Combined momentum and volatility factor panel at every training rebalance date. 
    Inner-join on (date, ticker) so only pairs with both factors are kept.

    Optional preprocessing: winsorize at (1%, 99%) per date, then z-score per date.
    """
    momentums = compute_momentum_at_training_rebalance_dates(
        prices,
        training_rebalance_dates,
        rolling_window_trading_days_for_momentum,
        buffer_trading_days_for_momentum,
        reb_date_to_tickers_universe_dict
    )
    vols = compute_volatility_at_training_rebalance_dates(
        prices,
        training_rebalance_dates,
        rolling_window_trading_days_for_volatility,
        reb_date_to_tickers_universe_dict
    )
    factors = pd.concat([momentums, vols], axis=1, join="inner")

    return _process_factors(
        factors, winsorize_factors_per_date, z_score_factors_per_date
    )


def compute_factors_at_date_for_tickers(
    date: pd.Timestamp,
    tickers: list[str],
    prices: pd.Series,
    rolling_window_trading_days_for_momentum: int,
    buffer_trading_days_for_momentum: int,
    rolling_window_trading_days_for_volatility: int,
    winsorize_factors_per_date: bool = True,
    z_score_factors_per_date: bool = True
) -> pd.DataFrame:
    """Single-date (`date`) combined factor panel (momentum + volatility) for the subset of tickers `tickers`."""
    momentums = compute_momentum_at_date_for_tickers(
        date,
        tickers,
        prices,
        rolling_window_trading_days_for_momentum,
        buffer_trading_days_for_momentum
    )
    vols = compute_volatility_at_date_for_tickers(
        date,
        tickers,
        prices,
        rolling_window_trading_days_for_volatility
    )
    factors = pd.concat([momentums, vols], axis=1, join="inner")
    factors.index.name = "ticker"
    return _process_factors(
        factors, winsorize_factors_per_date, z_score_factors_per_date
    )