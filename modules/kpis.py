"""Computation of the KPIs of one backtest run

Public API
----------
* class `BacktestKPIs`
        dataclass holding all KPIs sections and the returns time series

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

* function `build_book_ledger_across_dates(...)`
        build the tabular view of the book values across all backtest dates

* function `show_book(...)`
        print formatted tabular view of book across all backtest dates in a notebook cell

"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """P&L section."""
    first, last = backtest_dates[0], backtest_dates[-1]
    initial_eq = book_at_date[first].close.equity_excluding_margin_collateral
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
            date : book.close.equity_excluding_margin_collateral
            for date, book in book_at_date.items()
        }
    )
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
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "backtest dates"

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
    period_rf_return = pr.compound_daily_returns_into_periods(end_dates, rf_daily_returns)
    period_excess = period_eq_return - period_rf_return

    df = pd.DataFrame(
        {
            "previous_period_equity_returns": period_eq_return,
            "previous_period_rf_returns": period_rf_return,
            "previous_period_equity_excess_returns": period_excess,
        }
    )
    df.index = df.index.strftime("%Y-%m-%d")
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


def _safe_round(val, decimals: int):
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if val is None:
        return None
    return round(float(val), decimals)


def _trading_section(trades_log: dict) -> dict:
    max_pct_volume: float = 0.
    ticker_with_max_pct_volume:str = ""
    date_of_max_pct_volume:pd.Timestamp = list(trades_log.keys())[0] # fist trading date 
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
        "turnover": _summary_stats([v["turnover"] for v in trades_log.values()], decimals=4),
        "max_pct_of_daily_market_volume_traded_for_a_ticker_during_backtest" : {'max_pct_of_daily_market_volume': max_pct_volume,
                                                                 'for_ticker' : ticker_with_max_pct_volume,
                                                                'at_backtest_date' : date_of_max_pct_volume.strftime("%Y-%m-%d")
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
    maint_freq, maint_n, reb_freq, reb_n = _mr_violation_stats(backtest_dates, date_events, rebalance_dates)
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

    period_returns: pd.DataFrame
    returns_between_backtest_dates: pd.DataFrame
    inter_rebalance_periods_of_same_duration: bool

    def to_flat_dict(self) -> dict:
        flat = {}
        sections = (
            self.strategy, self.backtest_duration, self.backtest_PnL,
            self.perf_between_backtest_dates, self.performance_per_period,
            self.drawdown_metrics, self.trading, self.margin,
            self.accruals, self.leverage,
        )
        for section in sections:
            for k, v in section.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}_{sub_k}"] = sub_v
                else:
                    flat[k] = v
        return flat

    def summary_series(self) -> pd.Series:
        return pd.Series(self.to_flat_dict())


def compute_drawdowns(book_at_date: dict[pd.Timestamp, Book]) -> pd.Series:
    """Drawdown series of equity-excluding-margin-collateral."""
    equity = pd.Series(
        {
            date : book.close.equity_excluding_margin_collateral
            for date, book in book_at_date.items()
        }
    )
    peak = equity.cummax()
    return ((equity - peak) / peak).rename("drawdowns")


def compute_backtest_kpis(
    backtest_results: BacktestResults,
    rf_daily_returns: pd.Series
) -> BacktestKPIs:
    """Compute all KPI sections from a `BacktestResults` object.

    Parameters
    ----------
    backtest_results
        Output of function `run_backtest` from `modules/backtest.py`
    rf_daily_returns
        Daily risk-free returns indexed by trading date
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

    perf_between, returns_between_df = _performance_between_backtest_dates_section(book_at_date, backtest_dates, rf_daily_returns)

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
        period_returns=period_df,
        returns_between_backtest_dates=returns_between_df,
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
    # Dollar amounts
    "initial_equity": "{:,.2f}",
    "final_equity": "{:,.2f}",
    "total_trading_fees_paid_in_backtest": "{:,.2f}",
    "dividends_per_trading_date": "{:,.2f}",
    "financing_costs_per_trading_date": "{:,.2f}",
    "collateral_per_cure": "{:,.2f}",
    "total_collateral_posted": "{:,.2f}"
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
    _section("PERFORMANCE BETWEEN BACKTEST DATES", kpis.perf_between_backtest_dates)
    _section("PERFORMANCE PER INTER-REBALANCE PERIOD", kpis.performance_per_period)
    _section("DRAWDOWNS", kpis.drawdown_metrics)
    _section("TRADING", kpis.trading)
    _section("MARGIN REQUIREMENTS", kpis.margin)
    _section("ACCRUALS", kpis.accruals)
    _section("LEVERAGE", kpis.leverage)
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
    
    return df.style.format(fmt_dict).map(_color, subset=color_cols)

def style_returns_df(df: pd.DataFrame):
    """Red/white/green colormap on returns columns (Handles
    both period-returns and consecutive-backtest-dates returns) .
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

    fmt_dict = {c: lambda v: "N/A" if pd.isna(v) else f"{v:.2%}" for c in fmt_pct_2}
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
        records[date.strftime("%Y-%m-%d")] = row

    cols_idx = pd.MultiIndex.from_product(
        [accounts, moments], names=["account", "moment"],
    )
    book_df = pd.DataFrame.from_dict(records, orient="index").reindex(
        columns=cols_idx
    )
    book_df.index.name = "date"
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
        .apply(border_styles, axis=1)
        .set_table_styles(header_styles)
    )

