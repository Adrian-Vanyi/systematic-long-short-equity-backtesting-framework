"""Computation of the KPIs of one backtest run

Public API
----------
* class `BacktestKPIs`
        dataclass holding all KPIs sections and the returns time series
        (its method `plot_rolling_beta_vs_market` plots the rolling market beta)

* function `compute_backtest_kpis(...)` 
        build a `BacktestKPIs` from a `BacktestResults` object

* function `print_kpi_summary(...)`
        print kpis report to stdout

* function `save_kpi_summary(...)`
        save kpis report to disk

* function `style_returns_df(...)`
        formatted DataFrames for notebook display (color-mapping of the returns columns)

* function `compute_drawdowns(...)`
        equity drawdown time-series (used by plotting)

* function `compute_pnl_decomposition(...)`
        daily equity P&L decomposition (§2.6.6), asserting the identity closes

* function `compute_market_exposure(...)`
        ex-post regression of equity excess returns on market excess returns

* function `build_book_ledger_across_dates(...)`
        build the tabular view of the book values across all backtest dates

* function `show_book(...)`
        print formatted tabular view of book across all backtest dates in a notebook cell
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path

from modules import period_returns as pr
from modules.backtest import (
    BacktestResults,
    EVENT_MTM_MR_CURE,
    EVENT_REBALANCE_MR_CURE_SHRINK,
    EVENT_REBALANCE_MR_CURE_COLLATERAL
)
from modules.book_management import Book
from modules.strategies import BaseStrategy


logger = logging.getLogger(__name__)


# Minimum number of aligned observations required to run the market-exposure
# regression. Used in `compute_market_exposure`
_MIN_OBS_FOR_MARKET_EXPOSURE = 30

# Per-date tolerance for the §2.6.6 daily P&L identity, in dollars. Used in `compute_pnl_decomposition`
_PNL_IDENTITY_TOLERANCE_PER_DAY = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_round(val, decimals: int):
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if val is None:
        return None
    return round(float(val), decimals)


def _summary_stats(values, decimals: int = 3) -> dict:
    """Min/max/avg/std for a list of floats."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"min": None, "max": None, "avg": None, "std": None}
    std = float(np.std(valid)) if len(valid) > 1 else None
    return {
        "min": round(float(np.min(valid)), decimals),
        "max": round(float(np.max(valid)), decimals),
        "avg": round(float(np.mean(valid)), decimals),
        "std": round(std, decimals) if std is not None else None,
    }


def _align_index(obj: pd.Series) -> pd.Series:
    """Coerce to a tz-naive, nanosecond-resolution, normalised DatetimeIndex.

    Series reaching the market-exposure regression come from different
    sources (the backtest book, a parquet price panel, a downloaded ETF
    series) and may carry different datetime resolutions or a time
    component. Normalising all of them here makes the subsequent inner
    join independent of cross-resolution alignment behaviour.
    """
    out = obj.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.as_unit("ns").normalize()
    return out


# ---------------------------------------------------------------------------
# MR violation statistics
# ---------------------------------------------------------------------------

def _mr_violation_stats(
    backtest_dates: list[pd.Timestamp],
    date_events: dict[pd.Timestamp, tuple],
    rebalance_dates: pd.DatetimeIndex,
) -> tuple[float, int, float, int]:
    maint_violations = sum(
        1 for date in backtest_dates if EVENT_MTM_MR_CURE in date_events[date]
    )
    rebalance_cures = sum(
        1 for date in rebalance_dates
        if (EVENT_REBALANCE_MR_CURE_SHRINK in date_events[date]
            or EVENT_REBALANCE_MR_CURE_COLLATERAL in date_events[date])
    )
    n_dates = len(backtest_dates)
    n_rebs = len(rebalance_dates)
    return (
        maint_violations / n_dates if n_dates else 0.0,
        maint_violations,
        rebalance_cures / n_rebs if n_rebs else 0.0,
        rebalance_cures,
    )


# ---------------------------------------------------------------------------
# Per-section KPI builders
# ---------------------------------------------------------------------------

def _strategy_section(
    strategy: BaseStrategy,
    inter_rebalance_periods_of_same_duration: bool,
    return_target_for_strategy: float,
    return_target_per_period: float,
) -> dict:
    """Strategy section. Uses the strategy's own `kpi_parameters()`method."""
    return {
        "strategy_name": strategy.strategy_name,
        "return_target_for_strategy": return_target_for_strategy,
        "inter_rebalance_periods_of_same_duration":
            "yes" if inter_rebalance_periods_of_same_duration else "no",
        "return_target_for_inter_rebalance_period": (
            return_target_per_period
            if not inter_rebalance_periods_of_same_duration
            else "N/A"
        ),
        "strategy_parameters": strategy.kpi_parameters(),
    }


def _duration_section(
    backtest_dates: list[pd.Timestamp], date_events: dict
) -> dict:
    last = backtest_dates[-1]
    last_events = date_events[last]
    if "hit_return_target_for_strategy" in last_events:
        cause = "hit_return_target_for_strategy"
    elif "stop_loss_termination" in last_events:
        cause = "stop_loss_termination"
    elif "last_scheduled_backtest_date" in last_events:
        cause = "reached_last_scheduled_backtest_date"
    else:
        cause = "unknown. Check code."
    return {
        "number_of_backtest_days": len(backtest_dates),
        "first_backtest_date": backtest_dates[0].strftime("%Y-%m-%d"),
        "last_backtest_date": last.strftime("%Y-%m-%d"),
        "cause_of_backtest_termination": cause,
    }


def _pnl_section(
    book_at_date: dict[pd.Timestamp, Book],
    backtest_dates: list[pd.Timestamp]
) -> dict:
    first, last = backtest_dates[0], backtest_dates[-1]
    initial_eq = book_at_date[first].open.equity_excluding_margin_collateral
    final_eq = book_at_date[last].close.equity_excluding_margin_collateral
    eq_return = final_eq / initial_eq - 1
    return {
        "initial_equity": initial_eq,
        "final_equity": final_eq,
        "unrealized_return_of_equity": eq_return
    }


def _performance_between_backtest_dates_section(
    book_at_date: dict[pd.Timestamp, Book],
    backtest_dates: list[pd.Timestamp],
    rf_daily_returns: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """Return statistics between consecutive backtest dates.
    """
    eq_at_backtest_dates = pd.Series(
        {
            date: book.close.equity_excluding_margin_collateral
            for date, book in book_at_date.items()
        }
    ).sort_index()
    return_since_prev = eq_at_backtest_dates.pct_change(fill_method=None)
    rf_aligned = rf_daily_returns.reindex(backtest_dates)
    excess_return = (return_since_prev - rf_aligned).dropna()
    df = pd.DataFrame(
        {
            "equity_return_since_previous_backtest_date": return_since_prev,
            "rf_return_since_previous_backtest_date": rf_aligned,
            "equity_excess_return_since_previous_backtest_date": excess_return,
        }
    )

    df.index.name = "backtest_dates"

    avg = excess_return.mean() if not excess_return.empty else float("nan")
    std = excess_return.std(ddof=1) if len(excess_return) > 1 else float("nan")

    section = {
        "number_of_backtest_dates": len(backtest_dates),
        "avg_excess_return_between_backtest_dates": _safe_round(avg, 4),
        "std_of_excess_returns_between_backtest_dates": _safe_round(std, 4),
        "min_excess_return_between_backtest_dates": _safe_round(excess_return.min(), 4) if not excess_return.empty else None,
        "max_excess_return_between_backtest_dates": _safe_round(excess_return.max(), 4) if not excess_return.empty else None
    }
    return section, df


def _period_performance_section(
    book_at_date: dict[pd.Timestamp, Book],
    rebalance_dates: pd.DatetimeIndex,
    last_backtest_date: pd.Timestamp,
    rf_daily_returns: pd.Series,
    inter_rebalance_periods_of_same_duration: bool
) -> tuple[dict, pd.DataFrame]:
    """Returns over inter-rebalance periods."""
    eq_at_period_ends = pd.Series(
        {
            d: book_at_date[d].close.equity_excluding_margin_collateral
            for d in rebalance_dates
        }
    )
    if last_backtest_date != rebalance_dates[-1]:
        eq_at_period_ends[last_backtest_date] = book_at_date[last_backtest_date].close.equity_excluding_margin_collateral

    eq_at_period_ends = eq_at_period_ends.sort_index()
    period_eq_return = eq_at_period_ends.pct_change(fill_method=None)

    end_dates = period_eq_return.index
    period_rf_return = pr.compound_daily_returns_into_periods(
        end_dates, rf_daily_returns)
    period_excess = period_eq_return - period_rf_return

    df = pd.DataFrame(
        {
            "previous_period_equity_returns": period_eq_return,
            "previous_period_rf_returns": period_rf_return,
            "previous_period_equity_excess_returns": period_excess,
        }
    )
    df.index.name = "rebalance dates (except last date)"

    valid_excess = period_excess.dropna()
    n = len(valid_excess)
    avg = valid_excess.mean() if n > 0 else float("nan")
    std = valid_excess.std(ddof=1) if n > 1 else float("nan")

    section = {
        "number_of_periods": n,
        "periods_of_same_duration":
            "yes" if inter_rebalance_periods_of_same_duration else "no",
        "avg_period_excess_return": _safe_round(avg, 4),
        "std_of_period_excess_return": _safe_round(std, 4),
        "min_period_excess_return": _safe_round(valid_excess.min(), 4) if n > 0 else None,
        "max_period_excess_return": _safe_round(valid_excess.max(), 4) if n > 0 else None,
    }
    return section, df


def _trading_section(trades_log: dict) -> dict:
    max_pct_volume: float = 0.
    ticker_with_max_pct_volume: str = ""
    date_of_max_pct_volume: pd.Timestamp = list(
        trades_log.keys())[0]  # fist trading date
    for date, log in trades_log.items():
        pct_of_market_volume_traded = log['pct_of_market_volume_traded']
        ticker_max = pct_of_market_volume_traded.idxmax()
        max_pct = pct_of_market_volume_traded[ticker_max]
        date_of_max = date
        if max_pct > max_pct_volume:
            max_pct_volume = max_pct
            date_of_max_pct_volume = date_of_max
            ticker_with_max_pct_volume = ticker_max
    return {
        "total_trading_fees_paid_in_backtest": round(sum(v["trading_fee"] for v in trades_log.values()), 2),
        "total_execution_cost_paid_in_backtest": round(sum(v["execution_cost"] for v in trades_log.values()), 2),
        "total_gross_notional_traded_in_backtest": round(sum(v["gross_notional_traded"] for v in trades_log.values()), 2),
        "turnover": _summary_stats([v["turnover"] for v in trades_log.values()], decimals=4),
        "max_pct_of_daily_market_volume_traded_for_a_ticker_during_backtest": {'max_pct_of_daily_market_volume': max_pct_volume,
                                                                               'for_ticker': ticker_with_max_pct_volume,
                                                                               'at_backtest_date': date_of_max_pct_volume.strftime("%Y-%m-%d")
                                                                               }
    }


def _margin_section(
    backtest_dates: list[pd.Timestamp],
    date_events: dict,
    rebalance_dates: pd.DatetimeIndex,
    cure_method: str,
    shrink_factors: dict,
    posted_collateral: dict,
) -> dict:
    maint_freq, maint_n, reb_freq, reb_n = _mr_violation_stats(
        backtest_dates, date_events, rebalance_dates)
    return {
        "maint_MR_violation_number_of_backtest_days": maint_n,
        "maint_MR_violation_frequency": round(maint_freq, 4),
        "rebalance_MR_cure_number_of_backtest_days": reb_n,
        "rebalance_MR_cure_frequency": round(reb_freq, 4),
        "cure_method": cure_method,
        "shrink_factor_to_cure_an_MR_violation": _summary_stats(list(shrink_factors.values())) if shrink_factors else None,
        "collateral_per_cure":  _summary_stats(list(posted_collateral.values()), decimals=2) if posted_collateral else None,
        "total_collateral_posted":  round(float(np.sum(list(posted_collateral.values()))), 2) if posted_collateral else None
    }


def _accruals_section(financing_accruals: dict, dividend_accruals: dict) -> dict:
    return {
        "dividends_per_trading_date": _summary_stats(list(dividend_accruals.values()), decimals=2),
        "financing_costs_per_trading_date": _summary_stats(list(financing_accruals.values()), decimals=2),
    }


def _leverage_section(book_at_date: dict[pd.Timestamp, Book]) -> dict:
    gross, long, short = [], [], []
    for date, book in book_at_date.items():
        close = book.close
        if close.equity == 0:
            continue
        gross.append(close.gross_leverage)
        long.append(close.long_leverage)
        short.append(close.short_leverage)
    return {
        "gross_leverage": _summary_stats(gross, decimals=3),
        "long_leverage": _summary_stats(long, decimals=3),
        "short_leverage": _summary_stats(short, decimals=3)
    }


def compute_pnl_decomposition(
    backtest_results: BacktestResults,
    close_prices: pd.Series,
    tolerance_per_day: float = _PNL_IDENTITY_TOLERANCE_PER_DAY,
) -> pd.DataFrame:
    """Daily equity P&L decomposition, per the §2.6.6 identity.

    Parameters
    ----------
    backtest_results
        Output of `run_backtest`.
    close_prices
        Close prices indexed by (date, ticker).
    tolerance_per_day
        Maximum tolerated absolute daily residual, in dollars.

    Returns
    -------
    pd.DataFrame
        Indexed by backtest date, with one column per P&L stream plus
        `equity_change`, `identity_residual`, and `traded`.

    Raises
    ------
    AssertionError
        If any day's residual exceeds `tolerance_per_day`.
    """
    book_at_date = backtest_results.book_at_date
    dates = sorted(book_at_date.keys())
    trades_log = backtest_results.trades_log
    dividends = backtest_results.equity_accruals_from_dividends
    financing = backtest_results.equity_accruals_from_financing_costs
    collateral = backtest_results.posted_collateral_at_MR_violation_cures

    def costs_at(date: pd.Timestamp) -> tuple[float, float]:
        log = trades_log.get(date)
        if log is None:
            return 0.0, 0.0
        return float(log["trading_fee"]), float(log["execution_cost"])

    records = []
    for i, date in enumerate(dates):
        trading_fees, execution_cost = costs_at(date)

        if i == 0:
            # Equity starts at the opening cash and ends the day at the close of the updated book
            price_pnl = 0.0
            equity_change = (
                book_at_date[date].close.equity
                - book_at_date[date].open.equity
            )
        else:
            prev_date = dates[i - 1]
            shares_held = book_at_date[prev_date].close.shares
            prices_prev = close_prices.xs(
                prev_date, level=0).reindex(shares_held.index)
            prices_now = close_prices.xs(
                date, level=0).reindex(shares_held.index)
            price_pnl = float((shares_held * (prices_now - prices_prev)).sum())
            equity_change = (
                book_at_date[date].close.equity
                - book_at_date[prev_date].close.equity
            )

        dividend_pnl = float(dividends.get(date, 0.0))
        financing_pnl = float(financing.get(date, 0.0))
        posted = float(collateral.get(date, 0.0))

        explained = (
            price_pnl + dividend_pnl + financing_pnl
            - trading_fees - execution_cost + posted
        )
        records.append({
            "price_pnl": price_pnl,
            "dividend_pnl": dividend_pnl,
            "financing_pnl": financing_pnl,
            "trading_fees": -trading_fees,
            "execution_costs": -execution_cost,
            "posted_collateral": posted,
            "equity_change": equity_change,
            "identity_residual": equity_change - explained,
            "traded": date in trades_log,
        })

    df = pd.DataFrame(records, index=pd.DatetimeIndex(dates, name="date"))

    breaches = df.index[df["identity_residual"].abs() > tolerance_per_day]
    if len(breaches):
        detail = ", ".join(
            f"{d:%Y-%m-%d}: ${df.at[d, 'identity_residual']:,.4f}"
            for d in breaches[:5]
        )
        raise AssertionError(
            f"daily P&L identity failed on {len(breaches)} of {len(df)} date(s) "
            f"at a tolerance of ${tolerance_per_day:.1e}: {detail}"
            + (" ..." if len(breaches) > 5 else "")
            + ". A residual above the rounding tolerance means a P&L stream is "
            "missing from the identity or is being double-counted."
        )
    return df


def _pnl_decomposition_section(
    decomposition: pd.DataFrame,
    initial_equity: float,
    tolerance_per_day: float,
) -> dict:
    """Cumulative P&L streams, in dollars and as a share of initial equity."""
    streams = [
        "price_pnl", "dividend_pnl", "financing_pnl",
        "trading_fees", "execution_costs", "posted_collateral",
    ]
    totals = {name: float(decomposition[name].sum()) for name in streams}
    residual = decomposition["identity_residual"]

    return {
        "cumulative_pnl_in_dollars": {
            **{name: round(value, 2) for name, value in totals.items()},
            "total_equity_change": round(float(decomposition["equity_change"].sum()), 2),
        },
        "cumulative_pnl_as_pct_of_initial_equity": {
            **{
                name: (value / initial_equity if initial_equity else None)
                for name, value in totals.items()
            },
            "total_equity_change": (
                float(decomposition["equity_change"].sum()) / initial_equity
                if initial_equity else None
            ),
        },
        "identity_check": {
            "number_of_dates_tested": len(decomposition),
            # Not rounded: the residual is float-noise scale and rounding it
            # to a fixed number of decimals would report it as zero.
            "max_absolute_daily_residual": float(residual.abs().max()),
            "tolerance_per_day": tolerance_per_day,
            "number_of_dates_above_tolerance": int(
                (residual.abs() > tolerance_per_day).sum()
            ),
        },
    }


def _compound_annualize(daily_return: float) -> float:
    """Annualise a SIMPLE daily return by compounding."""
    return (1.0 + daily_return) ** 252 - 1.0


def _rolling_beta_with_standard_errors(
    df: pd.DataFrame, window: int, hac_lags: int
) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS beta and its HAC standard error, one fit per window."""
    lags = min(hac_lags, max(1, window // 10))
    betas, errors, dates = [], [], []
    for end in range(window, len(df) + 1):
        block = df.iloc[end - window:end]
        fit = sm.OLS(
            block["equity_excess"], sm.add_constant(block[["market_excess"]])
        ).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        betas.append(fit.params["market_excess"])
        errors.append(fit.bse["market_excess"])
        dates.append(block.index[-1])

    idx = pd.DatetimeIndex(dates, name="date")
    return (
        pd.Series(betas, index=idx, name="rolling_beta_vs_market"),
        pd.Series(errors, index=idx, name="rolling_beta_standard_error"),
    )


def compute_market_exposure(
    equity_excess_returns: pd.Series,
    spy_daily_returns: pd.Series,
    rf_daily_returns: pd.Series,
    book_at_date: dict[pd.Timestamp, Book] | None = None,
    hac_lags: int = 5,
    rolling_window: int = 60,
) -> tuple[dict, pd.Series, pd.Series]:
    """Ex-post market exposure of the realised equity curve.

    OLS of daily equity excess returns on market (SPY) excess returns,
    with Newey-West (HAC) standard errors, plus a rolling-window beta.
    The market proxy must be a total-return series (dividend-adjusted
    close).

    (`book_at_date` is optional so that the regression can be recomputed offline from a persisted equity curve).

    Returns
    -------
    (section_dict, rolling_beta_series)
    """
    if book_at_date is None:
        avg_net = None
    else:
        net_exposure = [
            book.close.long_leverage - book.close.short_leverage
            for book in book_at_date.values()
            if book.close.equity != 0
        ]
        avg_net = float(np.mean(net_exposure)) if net_exposure else None

    df = pd.concat(
        {
            "equity_excess": _align_index(equity_excess_returns),
            "spy": _align_index(spy_daily_returns),
            "rf": _align_index(rf_daily_returns),
        },
        axis=1,
        join="inner",
    ).dropna()

    if len(df) < _MIN_OBS_FOR_MARKET_EXPOSURE:
        logger.warning(
            "market exposure regression skipped: only %d aligned observations "
            "(minimum %d). Check index alignment between the KPI returns frame, "
            "the market proxy series, and the risk-free series.",
            len(df), _MIN_OBS_FOR_MARKET_EXPOSURE,
        )
        section = {
            "n_observations": len(df),
            "null_hypothesis_for_t_and_p": "alpha = 0 and beta = 0",
            "p_value_reference_distribution": "standard normal (asymptotic)",
            "beta": None,
            "hac_standard_error_beta": None,
            "classical_standard_error_beta": None,
            "t_stat_beta_hac": None,
            "p_value_beta_two_sided": None,
            "beta_95pct_confidence_interval_hac": {"lower": None, "upper": None},
            "alpha_daily": None,
            "hac_standard_error_alpha_daily": None,
            "classical_standard_error_alpha_daily": None,
            "t_stat_alpha_hac": None,
            "p_value_alpha_two_sided": None,
            "alpha_annualization_convention": "compounded: (1+alpha_daily)^252 - 1",
            "alpha_annualized": None,
            "alpha_annualized_95pct_confidence_interval_hac": {
                "lower": None, "upper": None,
            },
            "r_squared": None,
            "hac_lags_for_standard_errors": hac_lags,
            "rolling_beta_window_in_trading_days": rolling_window,
            "rolling_beta": _summary_stats([]),
            "avg_net_exposure_pct_of_equity": _safe_round(avg_net, 4),
        }
        return (
            section,
            pd.Series(dtype=float, name="rolling_beta_vs_market"),
            pd.Series(dtype=float, name="rolling_beta_standard_error"),
        )

    df["market_excess"] = df["spy"] - df["rf"]

    model = sm.OLS(df["equity_excess"], sm.add_constant(df[["market_excess"]]))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    # The same model refitted with classical standard errors.
    classical = model.fit()

    rolling_beta, rolling_se = _rolling_beta_with_standard_errors(
        df, rolling_window, hac_lags
    )

    alpha, beta = res.params["const"], res.params["market_excess"]
    ci = res.conf_int(alpha=0.05)

    section = {
        "n_observations": int(res.nobs),
        "null_hypothesis_for_t_and_p": "alpha = 0 and beta = 0",
        "p_value_reference_distribution": "standard normal (asymptotic)",
        "beta": _safe_round(beta, 3),
        "hac_standard_error_beta": _safe_round(res.bse["market_excess"], 4),
        "classical_standard_error_beta": _safe_round(classical.bse["market_excess"], 4),
        "t_stat_beta_hac": _safe_round(res.tvalues["market_excess"], 2),
        "p_value_beta_two_sided": _safe_round(res.pvalues["market_excess"], 3),
        "beta_95pct_confidence_interval_hac": {
            "lower": _safe_round(ci.loc["market_excess", 0], 3),
            "upper": _safe_round(ci.loc["market_excess", 1], 3),
        },
        "alpha_daily": _safe_round(alpha, 6),
        "hac_standard_error_alpha_daily": _safe_round(res.bse["const"], 6),
        "classical_standard_error_alpha_daily": _safe_round(classical.bse["const"], 6),
        "t_stat_alpha_hac": _safe_round(res.tvalues["const"], 2),
        "p_value_alpha_two_sided": _safe_round(res.pvalues["const"], 3),
        "alpha_annualization_convention": "compounded: (1+alpha_daily)^252 - 1",
        "alpha_annualized": _safe_round(_compound_annualize(alpha), 4),
        "alpha_annualized_95pct_confidence_interval_hac": {
            "lower": _safe_round(_compound_annualize(ci.loc["const", 0]), 4),
            "upper": _safe_round(_compound_annualize(ci.loc["const", 1]), 4),
        },
        "r_squared": _safe_round(res.rsquared, 3),
        "hac_lags_for_standard_errors": hac_lags,
        "rolling_beta_window_in_trading_days": rolling_window,
        "rolling_beta": _summary_stats(rolling_beta.values, decimals=3),
        "avg_net_exposure_pct_of_equity": _safe_round(avg_net, 4),
    }
    return section, rolling_beta, rolling_se


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BacktestKPIs:
    strategy: dict
    backtest_duration: dict
    backtest_PnL: dict
    perf_between_backtest_dates: dict
    performance_per_period: dict
    drawdown_metrics: dict
    trading: dict
    margin: dict
    accruals: dict
    leverage: dict
    market_exposure: dict | None
    pnl_decomposition: dict | None

    period_returns: pd.DataFrame
    returns_between_backtest_dates: pd.DataFrame
    rolling_beta_vs_market: pd.Series
    rolling_beta_standard_error: pd.Series
    pnl_decomposition_daily: pd.DataFrame
    inter_rebalance_periods_of_same_duration: bool

    def to_flat_dict(self) -> dict:
        flat = {}
        sections = (
            self.strategy, self.backtest_duration, self.backtest_PnL,
            self.perf_between_backtest_dates, self.performance_per_period,
            self.drawdown_metrics, self.trading, self.margin, self.accruals,
            self.leverage, self.pnl_decomposition, self.market_exposure,
        )
        for section in sections:
            if section is None:
                continue
            for k, v in section.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}_{sub_k}"] = sub_v
                else:
                    flat[k] = v
        return flat

    def summary_series(self) -> pd.Series:
        return pd.Series(self.to_flat_dict())

    def plot_rolling_beta_vs_market(
        self,
        save_path: Path | str | None = None,
        figsize: tuple[float, float] = (12, 4),
    ) -> tuple[plt.Figure, plt.Axes] | None:
        """Plot the rolling-window beta of the equity curve on the market.

        Draws `rolling_beta_vs_market`, a zero line, and the full-sample
        beta as a reference. Returns `(figure, axes)`, or None when the
        market-exposure section was not computed (no market proxy was
        passed to `compute_backtest_kpis`).
        """
        if self.market_exposure is None or self.rolling_beta_vs_market.empty:
            logger.warning(
                "no rolling beta to plot: the market-exposure section was not "
                "computed, or produced too few observations"
            )
            return None

        window = self.market_exposure["rolling_beta_window_in_trading_days"]
        full_sample_beta = self.market_exposure["beta"]

        fig, ax = plt.subplots(figsize=figsize)

        # Pointwise 95% band from the per-window HAC standard errors.
        if not self.rolling_beta_standard_error.empty:
            se = self.rolling_beta_standard_error.reindex(
                self.rolling_beta_vs_market.index
            )
            ax.fill_between(
                self.rolling_beta_vs_market.index,
                self.rolling_beta_vs_market - 1.96 * se,
                self.rolling_beta_vs_market + 1.96 * se,
                color="grey", alpha=0.25, linewidth=0,
                label="pointwise 95% interval (HAC)",
            )
        ax.plot(
            self.rolling_beta_vs_market.index,
            self.rolling_beta_vs_market.values,
            color="black", linewidth=1.5,
            label=f"rolling {window}-day beta",
        )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        if full_sample_beta is not None:
            ax.axhline(
                full_sample_beta, color="blue", linestyle=(0, (5, 5)),
                linewidth=1.5, alpha=0.8,
                label=f"full-sample beta = {full_sample_beta:.3f}",
            )
        ax.set_ylabel("Beta vs market")
        ax.set_xlabel("Date")
        ax.set_title(
            f"Rolling {window}-day beta of equity excess returns "
            f"on market excess returns"
        )
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("rolling-beta plot saved to %s", save_path)
        return fig, ax


def compute_drawdowns(book_at_date: dict[pd.Timestamp, Book]) -> pd.Series:
    """Drawdown series of equity-excluding-margin-collateral."""
    equity = pd.Series(
        {
            date: book.close.equity_excluding_margin_collateral
            for date, book in book_at_date.items()
        }
    ).sort_index()
    peak = equity.cummax()
    return ((equity - peak) / peak).rename("drawdowns")


def compute_backtest_kpis(
    backtest_results: BacktestResults,
    rf_daily_returns: pd.Series,
    market_daily_returns: pd.Series | None = None,
    close_prices: pd.Series | None = None,
) -> BacktestKPIs:
    """Compute all KPI sections from a `BacktestResults` object.

    Parameters
    ----------
    backtest_results
        Output of function `run_backtest` from `modules/backtest.py`
    rf_daily_returns
        Daily risk-free returns indexed by trading date
    market_daily_returns
        Optional daily total returns of a market proxy (e.g. SPY,
        from `download_market_returns`). When provided, the MARKET
        EXPOSURE section is computed; when omitted it is skipped.
    close_prices
        Optional close prices indexed by (date, ticker). When provided, the DAILY P&L
        DECOMPOSITION section is computed and the §2.6.6 identity is
        asserted; when omitted the section is skipped.
    """
    book_at_date = backtest_results.book_at_date
    backtest_dates = sorted(book_at_date.keys())
    rebalance_dates = backtest_results.actual_rebalance_dates
    same_duration = backtest_results.inter_rebalance_periods_of_same_duration

    duration = _duration_section(backtest_dates, backtest_results.date_events)

    strategy = _strategy_section(
        backtest_results.strategy,
        same_duration,
        backtest_results.return_target_for_strategy,
        backtest_results.return_target_for_inter_rebalance_period,
    )

    pnl = _pnl_section(book_at_date, backtest_dates)

    perf_between, returns_between_df = _performance_between_backtest_dates_section(
        book_at_date, backtest_dates, rf_daily_returns)

    period, period_df = _period_performance_section(
        book_at_date, rebalance_dates, backtest_dates[-1],
        rf_daily_returns, same_duration
    )

    drawdown_series = compute_drawdowns(book_at_date)

    drawdown_metrics = {
        "drawdown of equity-excluding-margin-collateral": _summary_stats(drawdown_series.values, decimals=4)
    }

    trading = _trading_section(backtest_results.trades_log)

    margin = _margin_section(
        backtest_dates,
        backtest_results.date_events,
        rebalance_dates,
        backtest_results.cure_method_for_MR_violation,
        backtest_results.shrink_factors_at_MR_violation_cures,
        backtest_results.posted_collateral_at_MR_violation_cures
    )

    accruals = _accruals_section(
        backtest_results.equity_accruals_from_financing_costs,
        backtest_results.equity_accruals_from_dividends
    )

    leverage = _leverage_section(book_at_date)

    if market_daily_returns is not None:
        market_exposure, rolling_beta, rolling_beta_se = compute_market_exposure(
            returns_between_df["equity_excess_return_since_previous_backtest_date"],
            market_daily_returns,
            rf_daily_returns,
            book_at_date,
        )
    else:
        market_exposure = None
        rolling_beta = pd.Series(dtype=float, name="rolling_beta_vs_market")
        rolling_beta_se = pd.Series(dtype=float, name="rolling_beta_standard_error")

    if close_prices is not None:
        pnl_decomposition_daily = compute_pnl_decomposition(
            backtest_results, close_prices
        )
        pnl_decomposition = _pnl_decomposition_section(
            pnl_decomposition_daily,
            initial_equity=pnl["initial_equity"],
            tolerance_per_day=_PNL_IDENTITY_TOLERANCE_PER_DAY,
        )
    else:
        pnl_decomposition = None
        pnl_decomposition_daily = pd.DataFrame()

    return BacktestKPIs(
        strategy=strategy,
        backtest_duration=duration,
        backtest_PnL=pnl,
        perf_between_backtest_dates=perf_between,
        performance_per_period=period,
        drawdown_metrics=drawdown_metrics,
        trading=trading,
        margin=margin,
        accruals=accruals,
        leverage=leverage,
        market_exposure=market_exposure,
        pnl_decomposition=pnl_decomposition,
        period_returns=period_df,
        returns_between_backtest_dates=returns_between_df,
        rolling_beta_vs_market=rolling_beta,
        rolling_beta_standard_error=rolling_beta_se,
        pnl_decomposition_daily=pnl_decomposition_daily,
        inter_rebalance_periods_of_same_duration=same_duration
    )


# ---------------------------------------------------------------------------
# Save & display
# ---------------------------------------------------------------------------

_KEY_FORMATS: dict[str, str] = {
    # Percent values
    "return_target_for_strategy": "{:.2%}",
    "return_target_for_inter_rebalance_period": "{:.2%}",
    "unrealized_return_of_equity": "{:.2%}",
    "avg_excess_return_between_backtest_dates": "{:.2%}",
    "std_of_excess_returns_between_backtest_dates":  "{:.2%}",
    "max_excess_return_between_backtest_dates": "{:.2%}",
    "min_excess_return_between_backtest_dates": "{:.2%}",
    "volatility_of_excess_return_between_backtest_dates": "{:.2%}",
    "avg_period_excess_return": "{:.2%}",
    "std_of_period_excess_return": "{:.2%}",
    "max_period_excess_return": "{:.2%}",
    "min_period_excess_return": "{:.2%}",
    "volatility_of_the_period_excess_return": "{:.2%}",
    "drawdown of equity-excluding-margin-collateral": "{:.2%}",
    "turnover": "{:.2%}",
    "max_pct_of_daily_market_volume": "{:.2}%",
    "maint_MR_violation_frequency": "{:.2%}",
    "rebalance_MR_cure_frequency": "{:.2%}",
    "shrink_factor_to_cure_an_MR_violation": "{:.2%}",
    "gross_leverage": "{:.2%}",
    "long_leverage": "{:.2%}",
    "short_leverage": "{:.2%}",
    "alpha_annualized": "{:.2%}",
    "alpha_annualized_95pct_confidence_interval_hac": "{:.2%}",
    "avg_net_exposure_pct_of_equity": "{:.2%}",
    "cumulative_pnl_as_pct_of_initial_equity": "{:.2%}",
    # Market-exposure regression outputs
    "beta": "{:.3f}",
    "hac_standard_error_beta": "{:.4f}",
    "classical_standard_error_beta": "{:.4f}",
    "t_stat_beta_hac": "{:.2f}",
    "p_value_beta_two_sided": "{:.3f}",
    "beta_95pct_confidence_interval_hac": "{:.3f}",
    "alpha_daily": "{:.6f}",
    "hac_standard_error_alpha_daily": "{:.6f}",
    "classical_standard_error_alpha_daily": "{:.6f}",
    "t_stat_alpha_hac": "{:.2f}",
    "p_value_alpha_two_sided": "{:.3f}",
    "r_squared": "{:.3f}",
    "rolling_beta": "{:.3f}",
    # Dollar amounts
    "initial_equity": "{:,.2f}",
    "final_equity": "{:,.2f}",
    "total_trading_fees_paid_in_backtest": "{:,.2f}",
    "total_execution_cost_paid_in_backtest": "{:,.2f}",
    "total_gross_notional_traded_in_backtest": "{:,.2f}",
    "dividends_per_trading_date": "{:,.2f}",
    "financing_costs_per_trading_date": "{:,.2f}",
    "collateral_per_cure": "{:,.2f}",
    "total_collateral_posted": "{:,.2f}",
    "cumulative_pnl_in_dollars": "{:,.2f}",
    # Residual and tolerance are float-noise scale; 2dp would print "0.00".
    "max_absolute_daily_residual": "{:.2e}",
    "tolerance_per_day": "{:.0e}"
}


def _fmt_value(format_key: str, val) -> str:
    """Format a value using `_KEY_FORMATS[format_key]` if defined,
    else `str(val)`. 
    """
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    if isinstance(val, (int, float)):
        fmt = _KEY_FORMATS.get(format_key)
        if fmt is not None:
            return fmt.format(val)
    return str(val)


def _build_kpi_summary(kpis: BacktestKPIs) -> str:
    """Build the full KPIs report as a single string."""
    LINE_WIDTH = 80
    lines: list[str] = []

    def _row(key: str, val_str: str, indent: int) -> None:
        left = " " * indent + key
        gap = max(LINE_WIDTH - len(left) - len(val_str), 1)
        lines.append(left + " " * gap + val_str)

    def _section(title: str, data: dict) -> None:
        lines.append("")
        lines.append("-" * LINE_WIDTH)
        lines.append(title.center(LINE_WIDTH))
        lines.append("-" * LINE_WIDTH)
        for k, v in data.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sub_k, sub_v in v.items():
                    fmt_key = sub_k if sub_k in _KEY_FORMATS else k
                    _row(sub_k, _fmt_value(fmt_key, sub_v), indent=4)
            else:
                _row(k, _fmt_value(k, v), indent=2)

    _section("STRATEGY", kpis.strategy)
    _section("BACKTEST DURATION", kpis.backtest_duration)
    _section("BACKTEST P&L", kpis.backtest_PnL)
    _section("PERFORMANCE BETWEEN BACKTEST DATES",
             kpis.perf_between_backtest_dates)
    _section("PERFORMANCE PER INTER-REBALANCE PERIOD",
             kpis.performance_per_period)
    _section("DRAWDOWNS", kpis.drawdown_metrics)
    _section("TRADING", kpis.trading)
    _section("MARGIN REQUIREMENTS", kpis.margin)
    _section("ACCRUALS", kpis.accruals)
    _section("LEVERAGE", kpis.leverage)
    if kpis.pnl_decomposition is not None:
        _section("DAILY P&L DECOMPOSITION", kpis.pnl_decomposition)
    if kpis.market_exposure is not None:
        _section("MARKET EXPOSURE", kpis.market_exposure)
    lines.append("")
    lines.append("-" * LINE_WIDTH)
    lines.append("")

    return "\n".join(lines)


def save_kpi_summary(kpis: BacktestKPIs, path: Path) -> Path:
    """Save the KPI report to a .txt file.

    Parameters
    ----------
    kpis : BacktestKPIs
        The KPI object to serialize.
    path : Path
        Output file path. 

    Returns
    -------
    Path 
       returns the path the report was written to.
    """
    path.write_text(_build_kpi_summary(kpis), encoding="utf-8")
    return path


def print_kpi_summary(kpis: BacktestKPIs) -> None:
    """print all KPIs sections to stdout."""
    print(_build_kpi_summary(kpis))


# ---------------------------------------------------------------------------
# Formatted DataFrames
# ---------------------------------------------------------------------------

def _format_date_index(label):
    """Render a Timestamp index label as YYYY-MM-DD for display."""
    if isinstance(label, pd.Timestamp):
        return label.strftime("%Y-%m-%d")
    return str(label)


def _make_returns_styler(df: pd.DataFrame, color_cols: list[str], fmt_dict: dict):
    if not color_cols:
        raise ValueError("no recognized return columns found in dataframe")
    max_abs = df[color_cols].abs().max().max()
    if not max_abs or np.isnan(max_abs):
        max_abs = 1.0  # fallback to avoid divide-by-zero in TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = LinearSegmentedColormap.from_list(
        "rwg",
        ["red", "darkred", "lightcoral", "lightgreen", "darkgreen", "green"],
    )

    def _color(val):
        if pd.isna(val):
            return "background-color: black; color: white"
        rgba = cmap(norm(val))
        r, g, b = (int(x * 255) for x in rgba[:3])
        text = "white" if abs(val) > 0.08 else "black"
        return f"background-color: rgb({r},{g},{b}); color: {text}"

    return (
        df.style
        .format(fmt_dict)
        .format_index(_format_date_index, axis=0)
        .map(_color, subset=color_cols)
    )


def style_returns_df(df: pd.DataFrame):
    """Red/white/green colormap on returns columns (Handles
    both period-returns and consecutive-backtest-dates returns) .

    The date index is formatted as YYYY-MM-DD at display time; the
    underlying DataFrame keeps its DatetimeIndex.
    """
    column_groups = [
        ("previous_period_equity_returns",
         "previous_period_rf_returns",
         "previous_period_equity_excess_returns"),
        ("equity_return_since_previous_backtest_date",
         "rf_return_since_previous_backtest_date",
         "equity_excess_return_since_previous_backtest_date"),
    ]
    color_cols, fmt_pct_2, fmt_pct_3 = [], [], []
    for eq_col, rf_col, ex_col in column_groups:
        if eq_col in df.columns:
            color_cols.append(eq_col)
            fmt_pct_2.append(eq_col)
        if ex_col in df.columns:
            color_cols.append(ex_col)
            fmt_pct_2.append(ex_col)
        if rf_col in df.columns:
            fmt_pct_3.append(rf_col)

    fmt_dict = {
        c: lambda v: "N/A" if pd.isna(v) else f"{v:.2%}" for c in fmt_pct_2}
    fmt_dict.update(
        {c: lambda v: "N/A" if pd.isna(v) else f"{v:.3%}" for c in fmt_pct_3}
    )
    return _make_returns_styler(df, color_cols, fmt_dict)


# ---------------------------------------------------------------------------
# Book-values table
# ---------------------------------------------------------------------------

def build_book_ledger_across_dates(
    backtest_results: BacktestResults,
) -> pd.DataFrame:
    """Build tabular view of book values across all backtest dates.

    Returns a DataFrame indexed by date with a MultiIndex on columns
    (account, moment), where moment sweeps ("open", "close").
    """
    accounts = [
        "equity",
        "equity_excluding_margin_collateral",
        "margin_collateral",
        "cash",
        "debit",
        "LMV",
        "SMV",
        "total_short_proceeds",
    ]
    moments = ["open", "close"]
    records = {}
    for date, book in backtest_results.book_at_date.items():
        row = {}
        for moment_label, snapshot in [
            ("open", book.open), ("close", book.close),
        ]:
            for acct in accounts:
                row[(acct, moment_label)] = getattr(snapshot, acct)
        records[date] = row

    cols_idx = pd.MultiIndex.from_product(
        [accounts, moments], names=["account", "moment"],
    )
    book_df = pd.DataFrame.from_dict(records, orient="index").reindex(
        columns=cols_idx
    )
    book_df.index = pd.DatetimeIndex(book_df.index, name="date")
    return book_df


def show_book(book: pd.DataFrame):
    """Display the book-values DataFrame with separators between account groups.
    """
    n_moments = 2
    n_accounts = len(book.columns) // n_moments

    def border_styles(s):
        styles = []
        for i in range(len(s)):
            if i == 0 or i % n_moments == 0:
                styles.append("border-left: 2px solid #333")
            else:
                styles.append("border-left: 1px dashed #666")
        return styles

    header_styles = [
        {"selector": "thead th.col_heading.level0",
         "props": [("text-align", "center")]},
    ]
    for i in range(n_accounts):
        header_styles.append({
            "selector":
                f"thead tr:nth-child(1) th.col_heading.level0:nth-child({i + 2})",
            "props": [("border-left", "2px solid #333")],
        })
        header_styles.append({
            "selector":
                f"thead tr:nth-child(2) th:nth-child({i * n_moments + 2})",
            "props": [("border-left", "2px solid #333")],
        })
    for i in range(n_accounts):
        for j in range(1, n_moments):
            nth = i * n_moments + j + 2
            header_styles.append({
                "selector": f"thead tr:nth-child(2) th:nth-child({nth})",
                "props": [("border-left", "1px dashed #666")],
            })
    return (
        book.style.format("{:,.2f}")
        .format_index(_format_date_index, axis=0)
        .apply(border_styles, axis=1)
        .set_table_styles(header_styles)
    )
