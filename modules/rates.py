"""Daily financing-rate panel.

The overnight financing accruals are computed from daily rates fixed at
the previous day's close. Rates may be:

- Constants (e.g. `annual_borrow_fee` = 25 bps, constant across the backtest).
- Date-indexed Series (e.g. EFFR-derived cash and debit rates).
- (date, ticker) MultiIndex Series (e.g. per-ticker borrow fees for hard-to-borrow names, varying in time).

The `*_at(date)` accessors return a scalar for account rates or a per-ticker Series for position-level rates.

Day-count convention used
-------------------------
- `ACT_360` (money-market convention)


Public API
----------
* class `DailyFinancingRates`
        exposes `*_at(date)` accessors.

* function `compute_daily_rate_from_annual_rate(...)`
        self-explanatory.

* constant `ACT_360`
        see above.


"""
from __future__ import annotations
import logging
import pandas as pd


logger = logging.getLogger(__name__)


# Day-count constants
ACT_360 = 360


def compute_daily_rate_from_annual_rate(
    annual_rate: pd.Series | float,
    day_count_basis: int,
) -> pd.Series | float:
    """Linear daily accrual: annual_rate / day_count_basis"""
    return annual_rate / day_count_basis


class DailyFinancingRates:
    """Daily financing rates used by the broker.

    Account-level rates (must be scalar or date-indexed Series):
      - `annual_cash_rate`
      - `annual_debit_rate`
      - `annual_margin_collateral_rate`

    Position-level rates (scalar, date-indexed, or (date, ticker)-indexed):
      - `annual_borrow_fee`
      - `annual_rebate_rate`

    The cash rate is floored at zero before conversion 
    (a cash deposit interest rate is non-negative).
    """

    def __init__(
        self,
        backtest_dates: pd.DatetimeIndex,
        annual_cash_rate: pd.Series | float,
        annual_debit_rate: pd.Series | float,
        annual_margin_collateral_rate: pd.Series | float,
        annual_borrow_fee: pd.Series | float,
        annual_rebate_rate: pd.Series | float,
        *,
        day_count_basis: int = ACT_360
    ):
        if not backtest_dates.is_monotonic_increasing:
            raise ValueError("backtest_dates must be monotonic increasing")
        for name, rate in [
            ("annual_cash_rate", annual_cash_rate),
            ("annual_debit_rate", annual_debit_rate),
            ("annual_margin_collateral_rate", annual_margin_collateral_rate),
        ]:
            if isinstance(rate, pd.Series) and isinstance(rate.index, pd.MultiIndex):
                raise ValueError(
                    f"{name} must be scalar or date-indexed; "
                    f"(date, ticker) MultiIndex is not allowed"
                )
        self.backtest_dates = backtest_dates
        self.day_count_basis = day_count_basis

        # floor cash rate at zero
        cash_rate = annual_cash_rate
        if isinstance(cash_rate, pd.Series):
            cash_rate = cash_rate.clip(lower=0.0)
        else:
            cash_rate = max(cash_rate, 0.0)

        self.r_cash_d = self._prepare_account_rate(
            compute_daily_rate_from_annual_rate(cash_rate, day_count_basis),
            "cash_rate",
        )
        self.r_debit_d = self._prepare_account_rate(
            compute_daily_rate_from_annual_rate(annual_debit_rate, day_count_basis),
            "debit_rate",
        )
        self.r_margin_collateral_d = self._prepare_account_rate(
            compute_daily_rate_from_annual_rate(
                annual_margin_collateral_rate, day_count_basis
            ),
            "margin_collateral_rate",
        )
        self.f_borrow_d = self._prepare_position_rate(
            compute_daily_rate_from_annual_rate(annual_borrow_fee, day_count_basis),
            "borrow_fee",
        )
        self.r_rebate_d = self._prepare_position_rate(
            compute_daily_rate_from_annual_rate(annual_rebate_rate, day_count_basis),
            "rebate_rate",
        )


    # --- Internal helpers :
    
    def _prepare_account_rate(
        self, daily_rate: pd.Series | float, name: str
    ) -> pd.Series | float:
        """A scalar passes through; a date-indexed Series is
        reindexed onto `backtest_dates` with forward-fill.
        """
        if isinstance(daily_rate, pd.Series):
            reindexed = daily_rate.reindex(self.backtest_dates).ffill()
            if reindexed.isna().any():
                first_valid = reindexed.first_valid_index()
                raise ValueError(
                    f"{name} series has no observation on or before "
                    f"{self.backtest_dates[0]}; first observation is "
                    f"at {first_valid}, so forward-fill cannot cover the prefix"
                )
            return reindexed
        return float(daily_rate)


    def _prepare_position_rate(
        self, daily_rate: pd.Series | float, name: str
    ) -> pd.Series | float:
        """A scalar passes through; a date-indexed Series is
        reindexed onto `backtest_dates` with forward-fill; a (date, ticker)-MultiIndex is reindexed onto the full
        `backtest_dates x tickers` grid and forward-filled per ticker.
        """
        if not isinstance(daily_rate, pd.Series):
            return float(daily_rate)
        
        if isinstance(daily_rate.index, pd.MultiIndex):
            tickers = daily_rate.index.get_level_values("ticker").unique()
            full_idx = pd.MultiIndex.from_product(
                [self.backtest_dates, tickers], names=["date", "ticker"],
            )
            return (
                daily_rate.reindex(full_idx)
                .groupby(level="ticker", observed=True)
                .ffill()
            )
        
        reindexed = daily_rate.reindex(self.backtest_dates).ffill()
        if reindexed.isna().any():
            first_valid = reindexed.first_valid_index()
            raise ValueError(
                f"{name} series has no observation on or before "
                f"{self.backtest_dates[0]}; first observation is "
                f"at {first_valid}, so forward-fill cannot cover the prefix"
            )
        return reindexed


    # --- accessors at a date 
    @staticmethod
    def _at_date(rate: pd.Series | float, date: pd.Timestamp):
        if not isinstance(rate, pd.Series):
            return rate
        if isinstance(rate.index, pd.MultiIndex):
            return rate.xs(date, level=0)
        return rate.loc[date]

    def cash_rate_at(self, date: pd.Timestamp) -> float:
        return float(self._at_date(self.r_cash_d, date))

    def debit_rate_at(self, date: pd.Timestamp) -> float:
        return float(self._at_date(self.r_debit_d, date))

    def margin_collateral_rate_at(self, date: pd.Timestamp) -> float:
        return float(self._at_date(self.r_margin_collateral_d, date))

    def borrow_fee_at(self, date: pd.Timestamp) -> float | pd.Series:
        """Return scalar (if borrow fee is constant across tickers) or per-ticker Series."""
        return self._at_date(self.f_borrow_d, date)

    def rebate_rate_at(self, date: pd.Timestamp) -> float | pd.Series:
        """Return scalar (if rebate rate is constant across tickers) or per-ticker Series."""
        return self._at_date(self.r_rebate_d, date)
