"""Build rebalance and training calendars for US equity backtests, aligned to
the NYSE trading calendar.

Public API
----------
* function `get_all_trading_days(start, end)`
        NYSE trading days in [start, end]

* function `align_to_trading_day(date)` 
        date snapped backward to a NYSE trading day (inclusive)

* function `next_trading_day_on_or_after(date)`
        date snapped forward to a NYSE trading day (inclusive)

* function `add_trading_days(date, n)`
        n-th NYSE trading day strictly after `date`

* function `subtract_trading_days(date, n)`
        n-th NYSE trading day strictly before `date`

* function `first_trading_day_of_month(year, month)`
        first NYSE trading day of the given calendar month

* function `last_trading_day_of_month(year, month)`
        last NYSE trading day of the given calendar month

* function `first_trading_day_of_every_month(start_date, end_date)`
        first NYSE trading day of each month in [start_date, end_date]; returns a list

* function `build_rebalance_dates_monthly(t0, n_periods, direction)`
        `n_periods` rebalance dates from `t0` (excluded) by first-trading-day-of-month (NYSE), forward or backward

* function `build_rebalance_dates_ndays(t0, n_periods, freq, direction)`
        `n_periods` rebalance dates from `t0` (excluded) stepping `freq` NYSE trading days, forward or backward

* class `BacktestCalendar`
        dataclass returned by `build_calendar`: rebalance, training-rebalance, and daily backtest calendars

* function `build_calendar(...)`
        main entry point: assemble the rebalance, training-rebalance, and daily backtest calendars
 
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
import pandas_market_calendars as mcal


logger = logging.getLogger(__name__)

_NYSE = mcal.get_calendar("NYSE")


def get_all_trading_days(
    start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """All NYSE trading days in [start, end] (timezone-naive).

    Returns an empty index when start > end or the range contains no
    trading days.
    """
    if pd.Timestamp(start) > pd.Timestamp(end):
        return pd.DatetimeIndex([])
    schedule = _NYSE.schedule(start_date=start, end_date=end)
    if schedule.empty:
        return pd.DatetimeIndex([])
    return mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)


def align_to_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Snap `date` backward to the closest NYSE trading day on or before it.

    Looks back up to 14 calendar days, which certainly covers the gap between any calendar day and
      the previous NYSE trading day since 1950 (longest closure: post-9/11, four trading days).
    """
    date = pd.Timestamp(date).normalize()
    days = get_all_trading_days(date - pd.Timedelta(days=14), date)
    if len(days) == 0:
        raise ValueError(f"no NYSE trading day on or before {date}")
    return days[-1]


def next_trading_day_on_or_after(date: pd.Timestamp) -> pd.Timestamp:
    """Snap `date` forward:
             returns `date` if it's a trading day,
             else: the next trading day.
    """
    date = pd.Timestamp(date).normalize()
    days = get_all_trading_days(date, date + pd.Timedelta(days=21))
    if len(days) == 0:
        raise ValueError(f"no NYSE trading day on or after {date}")
    return days[0]


def add_trading_days(date: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th NYSE trading day strictly after `date` (n >= 1).

    For the trading day on or after `date`", use function `next_trading_day_on_or_after`.
    """
    if n < 1:
        raise ValueError(
            f"n must be >= 1; got {n}. Use next_trading_day_on_or_after for n=0."
        )
    fetch_end = date + pd.Timedelta(days=n * 21 + 7)
    days = get_all_trading_days(date + pd.Timedelta(days=1), fetch_end)
    if len(days) < n:
        raise ValueError(f"could not find the {n}-th trading day after {date}")
    return days[n - 1]


def subtract_trading_days(date: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th NYSE trading day strictly before `date` (n >= 1)."""
    if n < 1:
        raise ValueError(
            f"n must be >= 1; got {n}. Use align_to_trading_day for n=0."
        )
    fetch_start = date - pd.Timedelta(days=n * 21 + 7)
    days = get_all_trading_days(fetch_start, date - pd.Timedelta(days=1))
    if len(days) < n:
        raise ValueError(f"could not find the {n}-th trading day before {date}")
    return days[-n]


def first_trading_day_of_month(year: int, month: int) -> pd.Timestamp:
    """First NYSE trading day on or after the calendar 1st of the given month."""
    return next_trading_day_on_or_after(pd.Timestamp(year=year, month=month, day=1))


def last_trading_day_of_month(year: int, month: int) -> pd.Timestamp:
    """Last NYSE trading day on or before the calendar end-of-month."""
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return align_to_trading_day(end)


def first_trading_day_of_every_month(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> list[pd.Timestamp]:
    month_starts = pd.date_range(start_date, end_date, freq="MS")
    return [first_trading_day_of_month(d.year, d.month) for d in month_starts]


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


# ---------------------------------------------------------------------------
# Rebalance-date builders:
# ---------------------------------------------------------------------------

def build_rebalance_dates_monthly(
    t0: pd.Timestamp, n_periods: int, direction: str
) -> list[pd.Timestamp]:
    """`n_periods` rebalance dates from `t0` going forward
    (`direction='fwd'`) or backward (`direction='bwd'`), using
    first-trading-day-of-month rule. `t0` is NOT included in the
    returned list.

    Note: when `direction='bwd'`, the first rebalance date starts in the calendar
    month before `t0`'s month, so the training period never overlaps
    `t0`'s month.
    """
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"direction must be 'fwd' or 'bwd', got {direction!r}")
    dates: list[pd.Timestamp] = []
    year, month = t0.year, t0.month
    for _ in range(n_periods):
        year, month = (
            _next_month(year, month) if direction == "fwd" else _prev_month(year, month)
        )
        dates.append(first_trading_day_of_month(year, month))
    return dates


def build_rebalance_dates_ndays(
    t0: pd.Timestamp, n_periods: int, freq: int, direction: str
) -> list[pd.Timestamp]:
    """`n_periods` rebalance dates from `t0` stepping by `freq`
    trading days, forward or backward. `t0` is NOT included in the
    returned list.
    """
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"direction must be 'fwd' or 'bwd', got {direction!r}")
    if freq < 1:
        raise ValueError(f"freq must be >= 1, got {freq}")

    # Calendar-day buffer to cover all trading days: n_periods × freq trading days, with calendar slack (3 calendar days to cover one trading day, and an additional 90 calendar days)
    buffer_days = int(n_periods * freq * 3) + 90
    if direction == "fwd":
        all_td = get_all_trading_days(t0, t0 + pd.Timedelta(days=buffer_days))
    else:
        all_td = get_all_trading_days(t0 - pd.Timedelta(days=buffer_days), t0)

    if t0 not in all_td:
        raise ValueError(
            f"t0={t0} is not a NYSE trading day; align it first via "
            f"align_to_trading_day"
        )
    idx0 = all_td.get_loc(t0)

    dates: list[pd.Timestamp] = []
    for k in range(1, n_periods + 1):
        step = k * freq if direction == "fwd" else -(k * freq)
        target_idx = idx0 + step
        if target_idx < 0 or target_idx >= len(all_td):
            raise ValueError(
                f"calendar buffer too small to build {n_periods} periods of "
                f"{freq} trading days {direction} from {t0}"
            )
        dates.append(all_td[target_idx])
    return dates


# ---------------------------------------------------------------------------
# Build backtest calendar:
# ---------------------------------------------------------------------------

@dataclass
class BacktestCalendar:
    """All calendar outputs for one backtest configuration.

    Attributes (and properties)
    ----------
    rebalance_dates
        [t_0, t_1, ..., t_P] (P+1 dates, P inter-rebalance periods)

    training_rebalance_dates
        [t_{-P_train}, ..., t_0, ..., t_P] (full rebalance grid, including the
        training prefix)

    backtest_dates
        All NYSE trading days in [t_0, t_{P+1}], where
        `t_{P+1}` is one rebalance period after `t_P` (using the
        same frequency rule), and equals `backtest_dates[-1]` 
    
    t0
        First backtest date (which is also the first rebalance date)

    last_backtest_date 
    """
    rebalance_dates: pd.DatetimeIndex
    training_rebalance_dates: pd.DatetimeIndex
    backtest_dates: pd.DatetimeIndex

    @property
    def t0(self) -> pd.Timestamp:
        return self.rebalance_dates[0]

    @property
    def last_backtest_date(self) -> pd.Timestamp:
        return self.backtest_dates[-1]


def build_calendar(
    start: str | pd.Timestamp,
    freq_type: str,
    P: int,
    P_train: int,
    freq: int | None = None
) -> BacktestCalendar:
    """Build the backtest and training rebalance calendars.

    Parameters
    ----------
    start
        Desired start of the backtest. Snapped backward to the closest
        NYSE trading day if not already one.
    freq_type
        if `"monthly"`, rebalance on the first trading day of each month.
        if `"ndays"`,  rebalance every `freq` trading days.
    P
        Number of backtest inter-rebalance periods. `rebalance_dates`
        will have P+1 entries.
    P_train
        Number of training inter-rebalance periods prepended before t_0.
    freq
        Trading days per period. Required when `freq_type="ndays"`.
    """
    if freq_type not in ("monthly", "ndays"):
        raise ValueError(f"freq_type must be 'monthly' or 'ndays', got {freq_type!r}")
    if freq_type == "ndays" and (freq is None or freq < 1):
        raise ValueError("freq must be a positive integer when freq_type='ndays'")
    if P < 0:
        raise ValueError(f"P must be >= 0, got {P}")
    if P_train < 0:
        raise ValueError(f"P_train must be >= 0, got {P_train}")

    t0 = align_to_trading_day(pd.Timestamp(start))

    if freq_type == "monthly":
        fwd_dates = build_rebalance_dates_monthly(t0, P, direction="fwd")
        bwd_dates = build_rebalance_dates_monthly(t0, P_train, direction="bwd")
        last_backtest_date = build_rebalance_dates_monthly(
            fwd_dates[-1] if fwd_dates else t0, 1, direction="fwd"
        )[0]
    else:
        fwd_dates = build_rebalance_dates_ndays(t0, P, freq, "fwd")
        bwd_dates = build_rebalance_dates_ndays(t0, P_train, freq, "bwd")
        last_backtest_date = build_rebalance_dates_ndays(
            fwd_dates[-1] if fwd_dates else t0, 1, freq, "fwd"
        )[0]

    rebalance_dates = pd.DatetimeIndex([t0] + fwd_dates)
    training_rebalance_dates = pd.DatetimeIndex(
        list(reversed(bwd_dates)) + [t0] + fwd_dates
    )
    backtest_dates = get_all_trading_days(t0, last_backtest_date)

    return BacktestCalendar(
            rebalance_dates = rebalance_dates,
            training_rebalance_dates = training_rebalance_dates,
            backtest_dates = backtest_dates
    )