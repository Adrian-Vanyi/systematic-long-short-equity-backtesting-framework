"""Module used for plotting results of a backtest run.

Design
------
Function `plot_book_values` builds a list of `_PlotSeries` specs from the config dataclass,
then renders each series uniformly using  either an "close-only" or a "two-snapshots" path (open & close). 
Reference lines (rebalance dates, first/last backtest date markers, y=0 / y=1 baselines) are added by internal helpers.

Public API
----------
* class `MainPanelConfig`
        main configuration dataclass for the book-values panel.
    
* class `LeveragePanelConfig`
        configuration dataclass for the leverage panel.

 class `PlotConfig`
        full plot configuration.

* function `plot_book_values(...)`
        displays stacked panels of book-value diagnostics over time.

"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from modules import kpis
from modules.book_management import Book, BookSnapshot
from modules.backtest import (
        BacktestResults,
        EVENT_REBALANCE,
        EVENT_RETURN_TARGET_FOR_STRATEGY,
        EVENT_INTER_REB_RETURN_TARGET,
        EVENT_MTM_MR_CURE, 
        EVENT_REBALANCE_MR_CURE_SHRINK,
        EVENT_REBALANCE_MR_CURE_COLLATERAL, 
        EVENT_STOP_LOSS
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MainPanelConfig:
    """Main panel of book-value plotting."""
    show_equity: bool = True
    show_equity_excluding_margin_collateral: bool = False
    show_margin_requirement: bool = False
    show_LMV: bool = False
    show_SMV: bool = False
    show_total_short_proceeds: bool = False
    show_cash: bool = False
    show_debit: bool = False
    show_collateral: bool = False


@dataclass
class LeveragePanelConfig:
    """Leverage panel of book-value plotting."""
    show_all: bool = False
    show_gross: bool = False
    show_long: bool = False
    show_short: bool = False

    @property
    def any_shown(self) -> bool:
        return self.show_all or self.show_gross or self.show_long or self.show_short


@dataclass
class PlotConfig:
    """Full plot configuration.

    Three optional panels stacked top to bottom:
      1. Main (equity, LMV/SMV, cash/debit, etc.): when `main_panel` has any flags set.
      2. Leverage (gross/long/short ratios): when `leverage_panel.any_shown`.
      3. Drawdowns: when `show_drawdowns=True`.
    """
    plot_start_date: pd.Timestamp
    plot_end_date: pd.Timestamp

    main_panel: MainPanelConfig = field(default_factory=MainPanelConfig)
    leverage_panel: LeveragePanelConfig = field(default_factory=LeveragePanelConfig)
    show_drawdowns: bool = False

    show_markers: bool = True
    close_marker_size: float = 5
    intraday_marker_size: float = 10

    # When True, plots both the open and close book's snapshots per date,
    # with vertical dashes between them. 
    show_two_snapshots: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _PlotSeries:
    """One series to plot.
    `getter` extracts a float from a `BookSnapshot`. The default pulls the attribute named `key`.
    """
    key: str
    color: str
    label: str
    linewidth: float = 1.5
    linestyle: str = "-"
    alpha: float = 1.0
    getter: Callable[[BookSnapshot], float] | None = None

    def get(self, snap: BookSnapshot) -> float:
        if self.getter is not None:
            return self.getter(snap)
        return getattr(snap, self.key)


def _plot_two_snapshots_series(
    ax,
    book_at_date: dict[pd.Timestamp, Book],
    plotted_dates: list,
    spec: _PlotSeries,
    markersize: int,
) -> None:
    """Render a series in two-snapshots mode (open & close)."""
    if len(plotted_dates) < 2:
        return
    dates_num = mdates.date2num([d.to_pydatetime() for d in plotted_dates])
    close = np.array([spec.get(book_at_date[d].close) for d in plotted_dates])
    open = np.array([spec.get(book_at_date[d].open) for d in plotted_dates])

    # Overnight segments: close[i] -> open[i+1]
    starts = np.column_stack([dates_num[:-1], close[:-1]])
    ends = np.column_stack([dates_num[1:], open[1:]])
    segments = np.stack([starts, ends], axis=1)

    ax.add_collection(
        LineCollection(
            segments,
            colors = spec.color,
            linewidth = spec.linewidth,
            linestyle = spec.linestyle,
            label = spec.label
        )
    )
    ax.scatter(
    dates_num, close,
    color="red", zorder=4, s=markersize + 12, marker="D"
    )
    ax.scatter(
        dates_num, open,
        color="blue", zorder=3, s=markersize + 6, marker="D"
    )
    # Vertical dashes only on dates where the open and close values differ
    TOL = 1e-6  # relative tolerance vs rounding
    diff = np.abs(close - open) > TOL
    if diff.any():
        ax.vlines(
            dates_num[diff],
            np.minimum(close[diff], open[diff]),
            np.maximum(close[diff], open[diff]),
            colors = spec.color,
            linestyle = (0, (2, 2)),
            linewidth = 1.5,
            alpha = 0.8,
            zorder = 4
        )



def _plot_close_series(
    ax,
    book_at_date: dict[pd.Timestamp, Book],
    plotted_dates: list,
    spec: _PlotSeries,
    markersize: int,
    marker: str | None
) -> None:
    """Render a series in close-only mode (one snapshot per date)."""
    vals = [spec.get(book_at_date[d].close) for d in plotted_dates]
    ax.plot(
        plotted_dates, vals,
        color=spec.color,
        linestyle=spec.linestyle,
        marker=marker,
        markersize=markersize,
        linewidth=spec.linewidth,
        alpha=spec.alpha,
        label=spec.label
    )


def _plot_series_list(
    ax,
    book_at_date: dict[pd.Timestamp, Book],
    plotted_dates: list,
    specs: list[_PlotSeries],
    show_two_snapshots: bool,
    close_marker: str | None,
    close_marker_size: int,
    intraday_marker_size: int,
) -> None:
    """Render a list of series on `ax` in either two-snapshots or close mode."""
    for spec in specs:
        if show_two_snapshots:
            _plot_two_snapshots_series(
                ax, book_at_date, plotted_dates, spec,
                markersize=intraday_marker_size
            )
        else:
            _plot_close_series(
                ax, book_at_date, plotted_dates, spec,
                markersize = close_marker_size, marker = close_marker
            )
    if show_two_snapshots and specs:
        # Add the three shared marker entries to the legend once
        ax.scatter([], [], color="red", s=intraday_marker_size + 8,
                   marker="D", label = "close")
        ax.scatter([], [], color="blue", s=intraday_marker_size + 6,
                   marker="D", label = "open")



def _add_panel_reference_lines(
    ax,
    rebalance_dates_in_window: list,
    first_backtest_date: pd.Timestamp,
    last_backtest_date: pd.Timestamp,
    date_events: dict,
    horizontal_baselines: tuple[float, ...] = ()
) -> None:
    early_label_added = False
    regular_label_added = False

    for reb_date in rebalance_dates_in_window:
        is_early = EVENT_INTER_REB_RETURN_TARGET in date_events[reb_date]
        if is_early:
            color = "orange"
            label = None if early_label_added else "early rebalance (hit running ret. target from prev. reb.)"
            early_label_added = True
        else:
            color = "black"
            label = None if regular_label_added else "scheduled rebalance"
            regular_label_added = True
        ax.axvline(
            x=reb_date, color=color, linestyle="-", linewidth=1,
            zorder=1, alpha=0.8, label=label,
        )
    ax.axvline(
        x=first_backtest_date, color="brown",
        linestyle=(0, (5, 5)), linewidth=1.5, zorder=2, alpha=0.7,
        label="first backtest date"
    )
    ax.axvline(
        x=last_backtest_date, color="purple",
        linestyle=(0, (5, 5)), linewidth=1.5, zorder=2, alpha=1,
        label="last backtest date"
    )
    for y in horizontal_baselines:
        ax.axhline(y=y, color="black", linestyle="-", linewidth=1,
                   alpha=0.7, zorder=1)


def _add_event_markers(
    ax,
    book_at_date: dict[pd.Timestamp, Book],
    annotation_key: str,
    stop_loss_date: pd.Timestamp | None,
    strategy_target_hit_date: pd.Timestamp | None
) -> None:
    """Mark stop-loss, strategy-return-target, and inter-rebalance-target events
     on the equity series.
    """
    if stop_loss_date is not None:
        ax.scatter(
            [stop_loss_date],
            [getattr(book_at_date[stop_loss_date].close, annotation_key)],
            color="red", marker="x", s=600, zorder=5,
            label="stop-loss termination"
        )
    elif strategy_target_hit_date is not None:
        ax.scatter(
            [strategy_target_hit_date],
            [getattr(book_at_date[strategy_target_hit_date].close, annotation_key)],
            color="orange", marker="x", s=600, zorder=6,
            label="strategy return target hit"
        )


def _add_mr_markers(
    ax,
    plotted_dates: list,
    maintenance_MR_at_close: dict,
    date_events: dict
) -> None:
    """Add markers at dates of MR violations: maintenance-MR and/or at rebalance MR.
    Markers are put on the maintenance MR time series, displaying the values eventually POST-CURE,
    (if there was a cure at the given date)."""
    maint_dates = [d for d in plotted_dates if EVENT_MTM_MR_CURE in date_events[d]]
    reb_cure_dates = [
        d for d in plotted_dates
        if (EVENT_REBALANCE_MR_CURE_SHRINK in date_events[d] or EVENT_REBALANCE_MR_CURE_COLLATERAL in date_events[d])
    ]
    if maint_dates:
        ax.scatter(
            maint_dates,
            [maintenance_MR_at_close[d] for d in maint_dates],
            marker="v", color="darkorange",
            edgecolors="black", linewidths=1,
            zorder=5, s=70, label="mtm MR violation cured"
        )
    if reb_cure_dates:
        ax.scatter(
            reb_cure_dates,
            [maintenance_MR_at_close[d] for d in reb_cure_dates],
            marker="s", color="purple", s=70, zorder=5,
            label="rebalance MR violation cured"
        )


def _build_main_panel_specs(cfg: MainPanelConfig) -> list[_PlotSeries]:
    specs: list[_PlotSeries] = []
    if cfg.show_equity:
        specs.append(_PlotSeries("equity", "black", "equity", linewidth=2))
    if cfg.show_equity_excluding_margin_collateral:
        specs.append(_PlotSeries(
            "equity_excluding_margin_collateral",
            "green", "equity (excl. posted margin collateral)", linewidth=2,
        ))
    if cfg.show_collateral:
        specs.append(_PlotSeries(
            "margin_collateral", "olive", "margin collateral",
            linewidth=1.2,
        ))
    if cfg.show_LMV:
        specs.append(_PlotSeries("LMV", "blue", "LMV", alpha=0.5))
    if cfg.show_SMV:
        specs.append(_PlotSeries("SMV", "red", "SMV", alpha=0.8))
    if cfg.show_cash:
        specs.append(_PlotSeries("cash", "orange", "cash", linewidth=2.5))
    if cfg.show_debit:
        specs.append(_PlotSeries("debit", "brown", "debit"))
    if cfg.show_total_short_proceeds:
        specs.append(_PlotSeries(
            "total_short_proceeds",
            "purple", "total short proceeds",
        ))
    return specs


def _build_leverage_panel_specs(cfg: LeveragePanelConfig) -> list[_PlotSeries]:
    show_gross = cfg.show_all or cfg.show_gross
    show_long = cfg.show_all or cfg.show_long
    show_short = cfg.show_all or cfg.show_short
    specs: list[_PlotSeries] = []
    if show_gross:
        specs.append(_PlotSeries(
            "gross_leverage", "black", "gross leverage",
            linewidth=1.2,
            getter=lambda snap: (
                (snap.LMV + snap.SMV) / snap.equity if snap.equity else 0.0
            ),
        ))
    if show_long:
        specs.append(_PlotSeries(
            "long_leverage", "blue", "long leverage",
            linewidth=1.2,
            getter=lambda snap: snap.LMV / snap.equity if snap.equity else 0.0
        ))
    if show_short:
        specs.append(_PlotSeries(
            "short_leverage", "red", "short leverage",
            linewidth=1.2,
            getter=lambda snap: snap.SMV / snap.equity if snap.equity else 0.0
        ))
    return specs


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def plot_book_values(
    backtest_results: BacktestResults,
    config: PlotConfig,
) -> tuple[plt.Figure, list[plt.Axes]] | None:
    """Stacked panels of book-value diagnostics over time.
    Returns (figure, axes) for further customization, or None if 
    nothing was selected to plot.
    """
    main_specs = _build_main_panel_specs(config.main_panel)
    leverage_specs = _build_leverage_panel_specs(config.leverage_panel)

    show_main = bool(main_specs) or config.main_panel.show_margin_requirement
    show_leverage = bool(leverage_specs)
    show_drawdowns = config.show_drawdowns

    if not (show_main or show_leverage or show_drawdowns):
        logger.warning("nothing selected to plot")
        return None

    # Setup state 
    book_at_date = backtest_results.book_at_date
    maintenance_MR_at_close = backtest_results.maintenance_MR_at_close
    date_events = backtest_results.date_events
    backtest_dates = list(book_at_date.keys())
    first_backtest_date = backtest_dates[0]
    last_backtest_date = backtest_dates[-1]

    plot_start_date = max(config.plot_start_date, first_backtest_date)
    plot_end_date = min(config.plot_end_date, last_backtest_date)
    
    plotted_dates = [
        d for d in backtest_dates if plot_start_date <= d <= plot_end_date
    ]
    rebalance_dates_in_window = [
        d for d in plotted_dates if EVENT_REBALANCE in date_events[d]
    ]
    stop_loss_date = next(
        (d for d, ev in date_events.items() if EVENT_STOP_LOSS in ev),
        None,
    )
    strategy_target_hit_date = next(
        (d for d, ev in date_events.items() if EVENT_RETURN_TARGET_FOR_STRATEGY in ev),
        None,
    )

    marker = "o" if config.show_markers else None
    close_marker_size_eff = int(config.close_marker_size if config.show_markers else 0)
    intra_marker_size_eff = int(
        config.intraday_marker_size if config.show_two_snapshots else 0
    )

    # Figure layout 
    n_subplots = int(show_main) + int(show_leverage) + int(show_drawdowns)
    height_ratios: list[float] = []
    if show_main:
        height_ratios.append(2)
    if show_leverage:
        height_ratios.append(1)
    if show_drawdowns:
        height_ratios.append(1)
    total_height = 6 if n_subplots == 1 else (9 if n_subplots == 2 else 12)
    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize = (12, total_height),
        sharex = (n_subplots > 1),
        gridspec_kw = {"height_ratios": height_ratios},
        squeeze = False
    )
    axes = axes.flatten().tolist()
    role_to_ax: dict[str, plt.Axes] = {}
    i = 0
    if show_main:
        role_to_ax["main"] = axes[i]; i += 1
    if show_leverage:
        role_to_ax["lev"] = axes[i]; i += 1
    if show_drawdowns:
        role_to_ax["dd"] = axes[i]; i += 1

    prefix = "Two-snapshot" if config.show_two_snapshots else "close"

    # Main panel 
    if show_main:
        ax = role_to_ax["main"]
        _plot_series_list(
            ax, book_at_date, plotted_dates, main_specs,
            show_two_snapshots = config.show_two_snapshots,
            close_marker = marker,
            close_marker_size = close_marker_size_eff,
            intraday_marker_size = intra_marker_size_eff
        )

        # Event markers on the equity series, when one is plotted.
        annotation_key = None
        if config.main_panel.show_equity:
            annotation_key = "equity"
        if config.main_panel.show_equity_excluding_margin_collateral:
            annotation_key = "equity_excluding_margin_collateral"
        if annotation_key is not None:
            _add_event_markers(
                ax, book_at_date, annotation_key,
                stop_loss_date, strategy_target_hit_date
            )
        if config.main_panel.show_margin_requirement:
            mr_vals = [maintenance_MR_at_close[d] for d in plotted_dates]
            ax.plot(
                plotted_dates, mr_vals,
                color = "orange", linestyle = "-", marker = marker,
                markersize = close_marker_size_eff, linewidth = 1.5,
                label = "maint_MR"
            )
            _add_mr_markers(ax, plotted_dates, maintenance_MR_at_close, date_events)

        _add_panel_reference_lines(
            ax, rebalance_dates_in_window,
            first_backtest_date, last_backtest_date,
            date_events,
            horizontal_baselines=(0.0,)
        )
        ax.legend(loc="best")
        ax.set_ylabel("USD")
        ax.set_title(
            f"{prefix} book values over time "
            f"(MR violations cured by: "
            f"{backtest_results.cure_method_for_MR_violation})"
        )
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    #  Leverage panel 
    if show_leverage:
        ax = role_to_ax["lev"]
        _plot_series_list(
            ax, book_at_date, plotted_dates, leverage_specs,
            show_two_snapshots = config.show_two_snapshots,
            close_marker = marker,
            close_marker_size = close_marker_size_eff,
            intraday_marker_size = intra_marker_size_eff,
        )
        _add_panel_reference_lines(
            ax, rebalance_dates_in_window,
            first_backtest_date, last_backtest_date,
            date_events,
            horizontal_baselines=(1.0,),
        )
        ax.set_ylabel("Leverage ratio")
        ax.legend(loc="best")
        ax.set_title(f"{prefix} leverage ratios over time")
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    # Drawdowns panel
    if show_drawdowns:
        ax = role_to_ax["dd"]       
        dd = kpis.compute_drawdowns(book_at_date)
        dd_vals = dd.reindex(plotted_dates).values
        
        ax.fill_between(
            plotted_dates, dd_vals, 0,
            color="red", alpha=0.4, label="drawdown"
        )
        ax.plot(
            plotted_dates, dd_vals,
            color = "red", linewidth = 1.0,
            marker = marker, markersize = close_marker_size_eff
        )
        _add_panel_reference_lines(
            ax, rebalance_dates_in_window,
            first_backtest_date, last_backtest_date,
            date_events,
            horizontal_baselines=(0.0,)
        )
        ax.set_ylabel("Drawdown")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)
        ax.set_title("Drawdowns of (close) equity-excluding-margin-collateral")
        ax.set_ylim(top=0.001)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}" if x <= 1e-9 else "")
                                    )

    # X-axis formatting on the bottom-most subplot
    bottom_ax = axes[-1]
   
    span = plot_end_date - plot_start_date
    pad = span * 0.02
    bottom_ax.set_xlim(plot_start_date - pad, plot_end_date + pad)

    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(bottom_ax.get_xticklabels(), rotation=45, ha="right")
    bottom_ax.set_xlabel("Date")

    plt.tight_layout()
    plt.show()
    return fig, axes