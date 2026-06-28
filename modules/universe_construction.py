"""Construction of the investable universe of tickers at each backtest rebalance date (see §6 of the documentation).

Design
------
At each rebalance date, candidates are the historical S&P 500 constituents on the first trading day of that month.
They pass these filters:

1. Forward data available. Ticker has non-NaN close and open prices for every NYSE trading day in the upcoming inter-rebalance period 
(including the rebalance dates bounding the period).

2. ADV available (optional, used only when capping the universe to a limit number of tickers). In this case, we select the tickers
with the larger ADV values (as a measure of liquidity).

After applying filter 1., if filter 2. is also applied, then the universe is restricted by keeping only the top 
`capping_num_tickers_per_universe` tickers by trailing average daily dollar volume (ADV) traded.

Bad-data removal
----------------
function `remove_bad_tickers` applies the cleaning rules from Appendix A of the documentation on the price data, 
dropping tickers with any single-day zero-volume print, or any single-day adjusted return outside [-90%, +500%].

Public API
----------
* function `compute_adv(...)`
        add "dollar_volume" and trailing "adv" columns to a (date, ticker) price panel.

* function `build_pit_universe_at_reb_date(...)`
        point-in-time investable universe at one rebalance date; returns the tuple (universe, eligibility frame).

* function `build_pit_universes(...)`
        bulk version across all training rebalance dates; returns the tuple  (date -> universe mapping, concatenated eligibility frame).

* function `remove_bad_tickers(...)`
           drop tickers failing the documentation's Appendix-A data-quality rules; returns the tuple (cleaned price panel, bad-ticker report).

"""

from __future__ import annotations
import logging
import pandas as pd
from modules import backtest_calendar as bcal


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_period_end(
    date: pd.Timestamp,
    schedule: pd.DatetimeIndex,
    sentinel_for_last: pd.Timestamp,
) -> pd.Timestamp:
    """End of the period starting at `date`: the next entry in
    `schedule`, or `sentinel_for_last` when `date` is the last entry."""
    if date == schedule[-1]:
        return sentinel_for_last
    return schedule[schedule.get_loc(date) + 1]


def _eligibility_at_date(
    date: pd.Timestamp,
    members: list[str],
    close_prices_wide: pd.DataFrame,
    open_prices_wide: pd.DataFrame,
    adv_wide: pd.DataFrame | None,
    next_period_end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """Build the per-ticker eligibility DataFrame at one date.
    Returns a DataFrame indexed by ticker, with one boolean column per filter. 
    """
    if len(members) != len(set(members)):
        duplicates = [m for m in set(members) if members.count(m) > 1]
        raise ValueError(
            f"duplicate tickers in S&P 500 members at {date}: "
            f"{duplicates[:5]}"  + (" ..." if len(duplicates)>5 else "")
        )

    elig = pd.DataFrame(index=pd.Index(members, name="ticker"))

    members_idx = pd.Index(members)
    in_close_data = members_idx.isin(close_prices_wide.columns)
    in_open_data = members_idx.isin(open_prices_wide.columns)

    present = [t for t, ok in zip(members, in_close_data & in_open_data) if ok]

    if next_period_end_date is None:
        elig["forward_close_prices_available"] = True
        elig["forward_open_prices_available"] = True
    else:
        inter_dates = bcal.get_all_trading_days(date, next_period_end_date)

        close_block = close_prices_wide.reindex(index=inter_dates, columns=present)
        elig["forward_close_prices_available"] = (
            close_block.notna().all(axis=0).reindex(elig.index, fill_value=False)
        )

        open_block = open_prices_wide.reindex(index=inter_dates, columns=present)
        elig["forward_open_prices_available"] = (
            open_block.notna().all(axis=0).reindex(elig.index, fill_value=False)
        )

    if adv_wide is not None:
        if date in adv_wide.index:
            elig["adv_available"] = adv_wide.loc[date].reindex(elig.index).notna()
        else:
            elig["adv_available"] = False

    return elig


def _resolve_universe_from_eligibility(
    elig: pd.DataFrame,
    date: pd.Timestamp,
    adv_wide: pd.DataFrame | None,
    cap: int | None,
) -> list[str]:
    """Combine eligibility flags into a final universe; optionally cap by ADV."""
    eligible_mask = elig.all(axis=1)
    universe = elig.index[eligible_mask].tolist()

    if not universe:
        raise ValueError(
            f"no ticker is eligible for universe construction at rebalance "
            f"date {date}"
        )

    if cap is not None and adv_wide is not None and len(universe) > cap:
        advs = adv_wide.loc[date, universe]
        universe = advs.nlargest(cap).index.tolist()

    return universe


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_adv(prices: pd.DataFrame, adv_window: int = 60) -> pd.DataFrame:
    """Add "dollar_volume" and "adv" columns to a (date, ticker)-indexed price panel. """
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.names != [
        "date", "ticker"
    ]:
        raise ValueError("prices must be indexed by (date, ticker)")
    out = prices.sort_index().copy()
    out["dollar_volume"] = out["close"].astype(float) * out["volume"].astype(float)
    out["adv"] = (
        out.groupby(level="ticker", observed=True)["dollar_volume"]
        .transform(
            lambda s: s.rolling(adv_window, min_periods=adv_window).mean()
        )
    )
    return out


def build_pit_universe_at_reb_date(
    reb_date: pd.Timestamp,
    rebalance_dates: pd.DatetimeIndex,
    final_period_end_date: pd.Timestamp,
    sp500_members_at_first_date_of_current_month: list[str],
    price_data: pd.DataFrame,
    require_prices_for_next_period: bool = True,
    capping_num_tickers_per_universe: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Build the PIT (point-in-time) universe  of investable tickers at a single rebalance date.

    Returns tuple `(universe_list, eligibility_dataframe)`.
    """
    close_prices_wide = price_data["close"].unstack()
    open_prices_wide = price_data["open"].unstack()
    adv_wide = (
        price_data["adv"].unstack()
        if capping_num_tickers_per_universe is not None
        else None
    )

    if not require_prices_for_next_period:
        next_end = None
    else:
        next_end = _next_period_end(reb_date, rebalance_dates, final_period_end_date)

    elig = _eligibility_at_date(
            date = reb_date,
            members = sp500_members_at_first_date_of_current_month,
            close_prices_wide = close_prices_wide,
            open_prices_wide = open_prices_wide,
            adv_wide = adv_wide,
            next_period_end_date = next_end
    )
    universe = _resolve_universe_from_eligibility(
        elig, reb_date, adv_wide, capping_num_tickers_per_universe
    )
    return universe, elig


def build_pit_universes(
    training_rebalance_dates: pd.DatetimeIndex,
    final_period_end_date: pd.Timestamp,
    first_trading_day_of_month_to_sp500_members_dict: dict,
    price_data: pd.DataFrame,
    require_prices_for_next_period: bool = True,
    capping_num_tickers_per_universe: int | None = None,
) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame]:
    
    """Build PIT universes at all training rebalance dates.

    Returns a tuple (mapping_date_to_universe, concatenated_eligibility_dataframe)
    The eligibility DataFrame has a (training_rebalance_date, ticker)
    MultiIndex, with one row per (date, candidate ticker) pair.
    """
    close_prices_wide = price_data["close"].unstack()
    open_prices_wide = price_data["open"].unstack()
    adv_wide = (
        price_data["adv"].unstack()
        if capping_num_tickers_per_universe is not None
        else None
    )

    mapping: dict[pd.Timestamp, list[str]] = {}
    eligibility_per_date: dict[pd.Timestamp, pd.DataFrame] = {}

    for reb_date in training_rebalance_dates:
        first_td_of_month = bcal.first_trading_day_of_month(
            reb_date.year, reb_date.month
        )
        members = first_trading_day_of_month_to_sp500_members_dict[first_td_of_month]

        if not require_prices_for_next_period:
            next_end = None
        else:
            next_end = _next_period_end(reb_date, training_rebalance_dates, final_period_end_date)

        elig = _eligibility_at_date(
                date = reb_date,
                members = members,
                close_prices_wide = close_prices_wide,
                open_prices_wide = open_prices_wide,
                adv_wide = adv_wide,
                next_period_end_date = next_end
        )
        universe = _resolve_universe_from_eligibility(elig, reb_date, adv_wide, capping_num_tickers_per_universe)

        eligibility_per_date[reb_date] = elig
        mapping[reb_date] = universe

    eligibility_concat = pd.concat(
        eligibility_per_date.values(), keys=eligibility_per_date.keys()
    )
    eligibility_concat.index.set_names(["training_rebalance_date", "ticker"], inplace=True)
    return mapping, eligibility_concat


def remove_bad_tickers(
    price_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop tickers exhibiting price data corruption.

    We drop tickers (accross all dates) if:
    - (Rule 1) any zero-volume print 
    - (Rule 2) any single-day return of adjusted-close prices falls outside [-90%, +500%]

    (See the rationale begind these two rules in Appendix A of the documentation)

    Returns a tuple  (cleaned_price_data, bad_ticker_report) where:
        * `cleaned_price_data` is the same DataFrame (with MultiIndex (date, ticker)) with rows for bad
        tickers removed.
        *  `bad_ticker_report` is a DataFrame indexed by ticker, with columns "zero_volume" (bool) and
        "extreme_return" (bool), flagging which rule(s) triggered.
    """
    def _classify(df: pd.DataFrame) -> pd.Series:
        vol_bad = (df["volume"] == 0).any()
        daily_rets = df["adj_close"].pct_change(fill_method=None)
        daily_ret_bad = ((daily_rets > 5.0) | (daily_rets < -0.9)).any()
        return pd.Series(
            {"zero_volume": bool(vol_bad), "extreme_return": bool(daily_ret_bad)}
        )
    classification = (
        price_data.groupby(level="ticker", observed=True).apply(_classify)
    )
    bad_mask = classification["zero_volume"] | classification["extreme_return"]
    bad_report = classification.loc[bad_mask]

    if not bad_report.empty:
        logger.info(
            "removing %d ticker(s) with bad data (zero_volume=%d, "
            "extreme_return=%d)",
            len(bad_report),
            int(bad_report["zero_volume"].sum()),
            int(bad_report["extreme_return"].sum()),
        )
    ticker_idx = price_data.index.get_level_values("ticker")
    cleaned = price_data.loc[~ticker_idx.isin(bad_report.index)]
    return cleaned, bad_report