"""Compute rolling Ledoit-Wolf shrunk covariance matrices at each rebalance date.

At each rebalance date, we estimate the covariance matrix of inter-rebalance period returns over the previous 
W periods (each of same length), applying the Ledoit-Wolf shrinkage procedure using a scaled-identity target. 
Tickers without a complete W-periods history are excluded from that date's covariance matrix.

Public API
----------
* class `CovEstimationResults`
        output container: dict of per-date covariance matrices and a Series of Ledoit-Wolf shrinkage intensities.

* function `compute_rolling_cov_matrices(...)`
        at each rebalance date, estimate the next-period return covariance from the W previous realized period returns, and then apply
        Ledoit-Wolf shrinkage toward a scaled identity target.

"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.covariance import LedoitWolf


logger = logging.getLogger(__name__)


@dataclass
class CovEstimationResults:
    """Output of function `compute_rolling_cov_matrices`.

    Attributes
    ----------
    matrices
        Mapping from rebalance date to covariance matrix (a `tickers x tickers` dataframe) . Dates with fewer than 2 tickers
        having complete price history for the estimation of their covariance are absent.
    shrinkage_intensities
        Series indexed by date; values = the Ledoit-Wolf shrinkage intensity (value in [0, 1]) used at each date.
    """
    matrices: dict[pd.Timestamp, pd.DataFrame]
    shrinkage_intensities: pd.Series


def compute_rolling_cov_matrices(
    prices: pd.Series,
    training_rebalance_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    rolling_periods_for_estimation: int
) -> CovEstimationResults:
    """At each rebalance date, estimate the covariance matrix of next-period returns
      using the returns for each of the previous `rolling_periods_for_estimation` inter-rebalance periods (referenced by `W` in this docstring).

    Window convention: the returns used are [r_{k-W}, r_{k-W+1}, ..., r_{k-1}],
    the W most recent realised returns, where r_j := r(t_j, t_{j+1}) denotes the forward realized return, 
    between rebalance dates t_j and t_{j+1}

    Parameters
    ----------
    prices
        Series indexed by (date, ticker)
    training_rebalance_dates
        Must contain every date in `rebalance_dates` plus at least
        W+1 preceding dates (so that `W` non-NaN returns are
        available at the first rebalance date).
    rebalance_dates
        Dates at which the covariance matrix is computed.
    rolling_periods_for_estimation
        Window length `W`; must be greater or equal to 2

    Returns
    -------
    CovEstimationResults
        Per-date covariance matrices & per-date shrinkage intensities.
    """
    if rolling_periods_for_estimation < 2:
        raise ValueError("rolling_periods_for_estimation must be >= 2")

    W = rolling_periods_for_estimation

    prices_wide = (
        prices.unstack("ticker")
        .sort_index()
        .reindex(training_rebalance_dates)
    )
    returns_wide = prices_wide.pct_change(fill_method = None)

    matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    intensities: dict[pd.Timestamp, float] = {}
    dates_index = returns_wide.index

    for date in rebalance_dates:
        if date not in dates_index:
            raise KeyError(
                f"rebalance date {date} is not in training_rebalance_dates"
            )
        idx = dates_index.get_loc(date)
        # Need W valid returns ending at idx; the first row of returns_wide
        # is NaN (no prior price), so the earliest feasible idx is W.
        if idx < W:
            raise ValueError(
                f"insufficient history at {date}: need {W} prior periods of "
                f"non-NaN returns in training_rebalance_dates, have {idx}"
            )

        window_returns = (
            returns_wide.iloc[idx - (W - 1) : idx + 1]  # last W rows including date
            .dropna(axis=1, how="any")
        )

        if window_returns.shape[1] < 2:
            logger.warning(
                "skipping cov estimation at %s: fewer than 2 tickers with a "
                "complete %d-period return history",
                date.strftime("%Y-%m-%d"), W,
            )
            continue

        tickers = window_returns.columns.tolist()
        lw = LedoitWolf().fit(window_returns.to_numpy())
        matrices[date] = pd.DataFrame(lw.covariance_, index = tickers, columns = tickers)
        intensities[date] = float(lw.shrinkage_)

    return CovEstimationResults(
        matrices = matrices,
        shrinkage_intensities = pd.Series(intensities, name = "shrinkage_intensity")
    )