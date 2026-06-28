"""Utilities for the trading simulation: execution-price model, per-share commission,
share-delta computation, integer-share truncation, and a per-(date,ticker) trading-cost panel.

Public API
----------  
* function `compute_execution_price(...)`
        apply half-spread + slippage friction to a mid-price

* function `compute_trading_fee(...)`
        per-share commission; polymorphic over scalar/Series inputs

* function `compute_shares_to_trade(...)`
        signed share delta between current and target book

* function `truncate_shares_to_int(...)`
        round toward zero for fractional share amounts

* class `TradingCosts`
        per-(date, ticker) cost panel with possible override 

"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution-price model
# ---------------------------------------------------------------------------

def compute_execution_price(
    shares_traded_for_ticker: int,
    full_spread_bps_ticker: float,
    slippage_bps_ticker: float,
    price_pre_trade: float,
) -> float:
    """Apply the half-spread + slippage friction in the adverse direction. (see §2.5.2 of the documentation)

    Parameters
    ----------
    shares_traded_for_ticker 
        Signed share count for this trade. Sign determines the adverse direction.
    full_spread_bps_ticker
        Full bid-ask spread in basis points.
    slippage_bps_ticker
        Simulated slippage in basis points.
    price_pre_trade
        Mid-price proxy
    """
    if full_spread_bps_ticker < 0 or slippage_bps_ticker < 0 or price_pre_trade < 0:
        raise ValueError(
            "full_spread_bps, slippage_bps, and price_pre_trade must be non-negative"
        )
    if shares_traded_for_ticker == 0:
        return price_pre_trade
    trade_sign = int(np.sign(shares_traded_for_ticker))
    half_spread_bps_ticker = full_spread_bps_ticker / 2
    return price_pre_trade * (
        1 + trade_sign * ((half_spread_bps_ticker + slippage_bps_ticker) / 1e4)
    )


def compute_trading_fee(
    shares_traded: int | pd.Series,
    trading_commission_per_share: float | pd.Series,
) -> float:
    """Computes per-share trading fee (see §2.6.4 of the documentation) 

    Polymorphic over scalar/Series inputs:
      - (scalar shares, scalar fee) -> fee x |shares|
      - (Series shares, Series fee)  -> Σ fee_i x |shares_i| over aligned tickers
      - (Series shares, scalar fee) ->  fee x Σ |shares_i|

    Returns
    -------
    float
        Exact (unrounded) fee.
    """
    if isinstance(shares_traded, pd.Series):
        if isinstance(trading_commission_per_share, pd.Series):
            tickers = shares_traded.index.intersection(
                trading_commission_per_share.index
            )
            return float(
                (
                    shares_traded.loc[tickers].abs()
                    * trading_commission_per_share.loc[tickers]
                ).sum()
            )
        return float(shares_traded.abs().sum() * trading_commission_per_share)

    if isinstance(shares_traded, (int, np.integer)):
        if isinstance(trading_commission_per_share, (int, float, np.floating)):
            return float(abs(shares_traded) * trading_commission_per_share)
        
    raise TypeError(
        "compute_trading_fee: expected (Series, Series), (Series, scalar), "
        f"or (int, scalar); got "
        f"({type(shares_traded).__name__}, "
        f"{type(trading_commission_per_share).__name__})"
    )


# ---------------------------------------------------------------------------
# Trade derivation
# ---------------------------------------------------------------------------

def compute_shares_to_trade(
    current_shares_in_book: pd.Series, new_shares_in_book: pd.Series
) -> pd.Series:
    """Computes per-ticker share delta as a Series of int values, to get
     from `current_shares_in_book` to `new_shares_in_book`.

    Tickers absent from one side are treated as zero on that side.
    Zero deltas are dropped.
    """
    tickers = current_shares_in_book.index.union(new_shares_in_book.index)
    current = current_shares_in_book.reindex(tickers, fill_value=0)
    target = new_shares_in_book.reindex(tickers, fill_value=0)
    delta = (target - current).rename("shares_to_trade")
    return delta[delta != 0].astype(int)


def truncate_shares_to_int(raw_shares: pd.Series) -> pd.Series:
    """Truncate share amounts toward zero.

    Examples: -3.25  -> -3
               7.58  -> 7
               4     -> 4
    """
    rounded = np.trunc(raw_shares)
    return pd.Series(rounded, index=raw_shares.index, name="shares").astype(int)


# ---------------------------------------------------------------------------
# Trading-cost panel
# ---------------------------------------------------------------------------

class TradingCosts:
    """Per-(date, ticker) trading-cost panel with an API for overrides.

    Stored as scalar defaults; per-(date, ticker) overrides may be added via method `set_overrides`. 
    Consumers query via method `at` rather than reaching into internal storage.

    Default values match the documentation's (§2.5.2, §2.6.4)
    """

    def __init__(
        self,
        backtest_dates: pd.DatetimeIndex,
        tickers: list[str],
        default_slippage_bps: float = 3.0,
        default_full_spread_bps: float = 2.0,
        default_commission_per_share: float = 0.001,
        overrides: pd.DataFrame | None = None,
    ):
        self.backtest_dates = backtest_dates
        self.tickers = tickers
        self.default_slippage_bps = float(default_slippage_bps)
        self.default_full_spread_bps = float(default_full_spread_bps)
        self.default_commission_per_share = float(default_commission_per_share)
        self._overrides = overrides

    # ---- accessors:

    def at(self, date: pd.Timestamp, ticker: str) -> dict[str, float]:
        """Return the trading-cost dict for one (date, ticker)."""
        defaults = {
            "slippage_bps": self.default_slippage_bps,
            "full_spread_bps": self.default_full_spread_bps,
            "commission_per_share": self.default_commission_per_share,
        }
        if self._overrides is None:
            return defaults
        try:
            row = self._overrides.loc[(date, ticker)]
            for k in defaults:
                if k in row and not pd.isna(row[k]):
                    defaults[k] = float(row[k])
        except KeyError:
            pass
        return defaults

    def at_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """Return a per-ticker DataFrame of costs at one date.

        Columns: "slippage_bps", "full_spread_bps", "commission_per_share"
        Indexed by ticker.
        """
        df = pd.DataFrame(
            {
                "slippage_bps": self.default_slippage_bps,
                "full_spread_bps": self.default_full_spread_bps,
                "commission_per_share": self.default_commission_per_share,
            },
            index=pd.Index(self.tickers, name="ticker"),
        )
        if self._overrides is not None:
            try:
                day_overrides = self._overrides.xs(date, level=0)
                df.update(day_overrides)
            except KeyError:
                pass
        return df

    # ---- overrides API:

    def set_overrides(self, overrides: pd.DataFrame) -> None:
        """Set overrides indexed by (date, ticker).
        Columns: any subset of {"slippage_bps", "full_spread_bps", "commission_per_share"}. 
        """
        if not isinstance(overrides.index, pd.MultiIndex):
            raise ValueError("overrides must be indexed by (date, ticker)")
        self._overrides = overrides
