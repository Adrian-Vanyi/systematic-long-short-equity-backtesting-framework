"""Backtest orchestrator.

Design
------
The daily loop is: at each backtest date, the orchestrator runs the following 7-step
sequence (skipped at the very first rebalance date, which is a special case):

1. Apply overnight P&L (financing + dividends)
2. Mark to market the book's positions using the open prices
3. Snapshot the book at the open, bifurcate to a working book for the snapshot at the close
4. Maintenance MR check (using close prices) + cure (if violated)
5. Termination tests (stop-loss, strategy return target, last backtest date reached)
6. Follow the dynamic rebalancing rule (if enabled)
7. If the date is on the scheduled rebalance calendar, rebalance the book's positions according to the strategy.

Public API
----------
* class `BacktestSchedule`
        backtest calendar and ticker universe inputs

* class `BacktestData`
        market data inputs for the backtest.

* class  `BacktestRules`
        termination and MR violation cure-method rules.

* class  `BacktestParameters`
        top-level input bundle (groups + strategy).

* class `BacktestResults`
        full output bundle.

* function `run_backtest(...)`
        runs the full backtest and returns BacktestResults.

"""

from __future__ import annotations
import pandas as pd
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from modules import backtest_calendar as bcal
from modules import book_management as bm
from modules import margin_requirement as mr
from modules import trading_utils as tu
from modules import universe_construction as uc
from modules.book_management import Book, BookSnapshot
from modules.rates import DailyFinancingRates
from modules.strategies import BaseStrategy, RebalanceContext


logger = logging.getLogger(__name__)


# =============================================================================
# Public input groups
# =============================================================================

@dataclass
class BacktestSchedule:
    """Calendar and ticker universe inputs"""
    backtest_dates: pd.DatetimeIndex
    scheduled_rebalance_dates: pd.DatetimeIndex
    scheduled_reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]]
    first_trading_day_of_month_to_sp500_members_dict: dict[pd.Timestamp, list[str]]
    scheduled_rebalance_freq_type: Literal["ndays", "monthly"]
    scheduled_rebalance_frequency_trading_days: int | None
    scheduled_number_of_inter_rebalance_periods_for_backtest: int
    capping_num_tickers_per_universe: int | None = None


@dataclass
class BacktestData:
    """Market data inputs."""
    price_data: pd.DataFrame
    dividend_data: pd.Series
    trading_costs: tu.TradingCosts
    daily_financing_rates: DailyFinancingRates
    mr_haircuts: mr.MRHaircuts


@dataclass
class BacktestRules:
    """Termination and MR violation cure-method rules."""
    initial_cash: float
    use_return_target_hit_rule: bool
    return_target_for_strategy: float | None
    use_dynamic_rebalancing_rule: bool
    return_target_for_inter_rebalance_period: float | None
    cure_method_for_MR_violation: Literal["shrinking_exposures", "posting_collateral"]
    pct_initial_equity_triggering_stop_loss: float


@dataclass
class BacktestParameters:
    """Full backtest input bundle.
    Holds the strategy plus three configuration objects.
    """
    strategy: BaseStrategy
    schedule: BacktestSchedule
    data: BacktestData
    rules: BacktestRules


@dataclass
class BacktestResults:
    """Full backtest output bundle."""
    strategy: BaseStrategy
    return_target_for_strategy: float
    return_target_for_inter_rebalance_period: float
    actual_rebalance_dates: pd.DatetimeIndex
    actual_reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]]
    inter_rebalance_periods_of_same_duration: bool
    book_at_date: dict[pd.Timestamp, Book]
    date_events: dict[pd.Timestamp, tuple[str, ...]]
    trades_log: dict[pd.Timestamp, dict]
    maintenance_MR_at_close: dict[pd.Timestamp, float]
    cure_method_for_MR_violation: str
    shrink_factors_at_MR_violation_cures: dict[pd.Timestamp, float]
    posted_collateral_at_MR_violation_cures: dict[pd.Timestamp, float]
    equity_accruals_from_financing_costs: dict[pd.Timestamp, float]
    equity_accruals_from_dividends: dict[pd.Timestamp, float]
    optimizer_results_at_rebalance_date: dict[pd.Timestamp, Any]


# =============================================================================
# Event flags
# =============================================================================

EVENT_MTM_ONLY = "mtm_only"
EVENT_REBALANCE = "rebalance"
EVENT_MTM_MR_CURE = "mtm_MR_violation_cure_executed"
EVENT_REBALANCE_MR_CURE_SHRINK = "rebalance_MR_cure_shrinking_exposures"
EVENT_REBALANCE_MR_CURE_COLLATERAL = "rebalance_MR_cure_posting_collateral"
EVENT_STOP_LOSS = "stop_loss_termination"
EVENT_RETURN_TARGET_FOR_STRATEGY = "hit_return_target_for_strategy"
EVENT_INTER_REB_RETURN_TARGET = "hit_inter_rebalance_return_target"
EVENT_LAST_DATE = "last_scheduled_backtest_date"


# =============================================================================
# Internal helper types
# =============================================================================

@dataclass
class _BacktestState:
    """Mutable runtime state of the backtest. Phase helpers mutate this."""
    book_working: BookSnapshot
    book_at_date: dict[pd.Timestamp, Book] = field(default_factory=dict)
    actual_rebalance_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    actual_reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]] = field(default_factory=dict)
    new_scheduled_rebalance_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    date_events: dict[pd.Timestamp, tuple[str, ...]] = field(default_factory=dict)
    trades_log: dict[pd.Timestamp, dict] = field(default_factory=dict)
    maintenance_MR_at_close: dict[pd.Timestamp, float] = field(default_factory=dict)
    shrink_factors_at_MR_violation_cures: dict[pd.Timestamp, float] = field(default_factory=dict)
    posted_collateral_at_MR_violation_cures: dict[pd.Timestamp, float] = field(default_factory=dict)
    equity_accruals_from_financing_costs: dict[pd.Timestamp, float] = field(default_factory=dict)
    equity_accruals_from_dividends: dict[pd.Timestamp, float] = field(default_factory=dict)
    optimizer_results_at_rebalance_date: dict[pd.Timestamp, Any] = field(default_factory=dict)
    schedule_diverged_from_scheduled: bool = False


@dataclass
class _TradeSimulationResult:
    snapshot: BookSnapshot
    trades_log: dict
    maintenance_MR_post_trades: float
    flag_added_to_date: tuple[str, ...]
    posted_margin_collateral: float | None
    shrink_factor: float | None


@dataclass
class _MtmCureResult:
    snapshot: BookSnapshot
    trades_log: dict
    maintenance_MR_post_cure: float
    posted_margin_collateral: float | None
    shrink_factor: float | None


# =============================================================================
# Maintenance MR helper 
# =============================================================================

def _maint_mr(
    date: pd.Timestamp, params: BacktestParameters, shares: pd.Series
) -> float:
    return mr.compute_maintenance_MR(
            date = date,
            mr_haircuts = params.data.mr_haircuts,
            prices = params.data.price_data["close"],
            current_shares = shares
    )


# =============================================================================
# Phase helper: overnight P&L
# =============================================================================

def _apply_overnight_pnl(
    state: _BacktestState,
    params: BacktestParameters,
    prev_date: pd.Timestamp,
    date: pd.Timestamp,
    close_prices: pd.Series,
    snapshot_at_open: BookSnapshot
) -> BookSnapshot:
    """Apply overnight financing & dividends; record per-date accruals."""
    eq_before = snapshot_at_open.equity
    snapshot_at_open = bm.apply_financing_and_carry(
        snapshot_at_open, date, prev_date, close_prices,
        params.data.daily_financing_rates,
    )
    state.equity_accruals_from_financing_costs[date] = round(
        snapshot_at_open.equity - eq_before, 2
    )
    eq_before = snapshot_at_open.equity
    snapshot_at_open = bm.account_dividends(date, snapshot_at_open, params.data.dividend_data)
    state.equity_accruals_from_dividends[date] = round(snapshot_at_open.equity - eq_before, 2)
    return snapshot_at_open


# =============================================================================
# Phase helper:  MTM cure
# =============================================================================

def _execute_mtm_cure(
    date: pd.Timestamp,
    snapshot: BookSnapshot,
    trades_log: dict,
    current_shares: pd.Series,
    maintenance_MR: float,
    cure_method: str,
    params: BacktestParameters,
    prices: pd.Series,
    market_volume: pd.Series
) -> _MtmCureResult:
    """Cure a maintenance-MR violation."""
    current_equity = snapshot.equity

    if cure_method == "posting_collateral":
        posted = maintenance_MR - current_equity
        snapshot.margin_collateral += posted
        snapshot.update_equity_fields()
        snapshot.update_leverage_ratios()
        # MR doesn't change because positions are unchanged (only collateral
        # was posted): return the already-computed value for maintenance_MR.
        return _MtmCureResult(
                snapshot = snapshot,
                trades_log = trades_log,
                maintenance_MR_post_cure = maintenance_MR,
                posted_margin_collateral = posted,
                shrink_factor = None
        )

    if cure_method == "shrinking_exposures":
        shrink_factor = mr.compute_shrinking_factor(
                current_equity,
                date,
                params.data.mr_haircuts,
                prices,
                params.data.trading_costs,
                current_shares = current_shares,
                intended_new_shares = current_shares
        )
        shrunk_shares = tu.truncate_shares_to_int(shrink_factor * current_shares)
        shares_to_trade = tu.compute_shares_to_trade(current_shares, shrunk_shares)
        snapshot, trades_log = bm.update_from_all_trades(
                current_snapshot = snapshot,
                trades_log = trades_log,
                market_volume = market_volume,
                date = date,
                shares_to_trade = shares_to_trade,
                trading_costs = params.data.trading_costs,
                prices_pre_trade = prices,
                prices_post_trade = prices
        )
        return _MtmCureResult(
                snapshot = snapshot,
                trades_log = trades_log,
                maintenance_MR_post_cure = _maint_mr(date, params, snapshot.shares),
                posted_margin_collateral = None,
                shrink_factor = shrink_factor
        )

    raise ValueError(f"unknown cure_method_for_MR_violation: {cure_method!r}")


def _cure_mtm_violation(
    state: _BacktestState,
    params: BacktestParameters,
    date: pd.Timestamp,
    prices: pd.Series,
    market_volume: pd.Series,
    events_today: list[str],
) -> None:
    """Check maintenance MR; cure if violated."""
    current_shares = state.book_working.shares
    maint_MR = _maint_mr(date, params, current_shares)
    if state.book_working.equity >= maint_MR:
        state.maintenance_MR_at_close[date] = maint_MR
        return

    events_today.append(EVENT_MTM_MR_CURE)

    cure = _execute_mtm_cure(
        date = date,
        snapshot = state.book_working,
        trades_log = state.trades_log,
        current_shares = current_shares,
        maintenance_MR = maint_MR,
        cure_method = params.rules.cure_method_for_MR_violation,
        params = params,
        prices = prices,
        market_volume = market_volume
    )
    state.book_working = cure.snapshot
    state.trades_log = cure.trades_log
    state.maintenance_MR_at_close[date] = cure.maintenance_MR_post_cure

    if cure.posted_margin_collateral is not None:
        state.posted_collateral_at_MR_violation_cures[date] = (
            cure.posted_margin_collateral
        )
    if cure.shrink_factor is not None:
        state.shrink_factors_at_MR_violation_cures[date] = cure.shrink_factor


# =============================================================================
# Phase helper: terminations
# =============================================================================

def _stop_loss_triggered(
    state: _BacktestState,
    params: BacktestParameters,
    date: pd.Timestamp,
    prices: pd.Series
) -> bool:
    threshold = bm.compute_stop_loss_threshold(
        date,
        state.book_working,
        params.rules.initial_cash,
        params.rules.pct_initial_equity_triggering_stop_loss,
        prices,
        params.data.trading_costs
    )
    return state.book_working.equity_excluding_margin_collateral <= threshold


def _strategy_return_target_hit(
    state: _BacktestState, params: BacktestParameters, first_date: pd.Timestamp
) -> bool:
    if not params.rules.use_return_target_hit_rule:
        return False
    initial_eq = state.book_at_date[first_date].close.equity_excluding_margin_collateral
    current_eq = state.book_working.equity_excluding_margin_collateral
    return (current_eq / initial_eq - 1.0) >= params.rules.return_target_for_strategy


def _inter_rebalance_target_hit(
    state: _BacktestState, params: BacktestParameters
) -> bool:
    if not params.rules.use_dynamic_rebalancing_rule:
        return False
    prev_reb_date = state.actual_rebalance_dates[-1]
    prev_reb_eq = (
        state.book_at_date[prev_reb_date].close.equity_excluding_margin_collateral
    )
    current_eq = state.book_working.equity_excluding_margin_collateral
    return (
        current_eq / prev_reb_eq - 1.0
        >= params.rules.return_target_for_inter_rebalance_period
    )


# =============================================================================
# Phase helper: dynamic universe construction and rebalance schedule update
# =============================================================================

def _build_pit_universe_for_dynamic_rebalance(
    state: _BacktestState, params: BacktestParameters, date: pd.Timestamp
) -> list[str]:
    first_td_of_month = bcal.first_trading_day_of_month(date.year, date.month)
    sp500_members = (
        params.schedule.first_trading_day_of_month_to_sp500_members_dict[
            first_td_of_month
        ]
    )
    universe, _ = uc.build_pit_universe_at_reb_date(
            date,
            state.new_scheduled_rebalance_dates,
            params.schedule.backtest_dates[-1],
            sp500_members,
            params.data.price_data,
            require_prices_for_next_period = True,
            capping_num_tickers_per_universe = params.schedule.capping_num_tickers_per_universe
    )
    return universe


def _update_future_rebalance_schedule(
    params: BacktestParameters,
    actual_rebalance_dates: pd.DatetimeIndex,
    last_backtest_date: pd.Timestamp,
    date: pd.Timestamp
) -> pd.DatetimeIndex:
    """When the inter-rebalance return target fires, rebuild the future
    rebalance schedule from `date` forward, using the same frequency
    rule, capping at `last_backtest_date`.
    """
    n_periods = params.schedule.scheduled_number_of_inter_rebalance_periods_for_backtest

    if params.schedule.scheduled_rebalance_freq_type == "monthly":
        future = bcal.build_rebalance_dates_monthly(
                    date,
                    n_periods=n_periods,
                    direction="fwd"
        )
    else:
        future = bcal.build_rebalance_dates_ndays(
                    date,
                    n_periods=n_periods,
                    freq=params.schedule.scheduled_rebalance_frequency_trading_days,
                    direction="fwd"
        )

    future = pd.DatetimeIndex(future)
    future = future[future <= last_backtest_date]
    return actual_rebalance_dates.append(future)


def _resolve_universe_at_rebalance_date(
    state: _BacktestState, params: BacktestParameters, date: pd.Timestamp
) -> list[str]:
    if state.schedule_diverged_from_scheduled:
        return _build_pit_universe_for_dynamic_rebalance(state, params, date)
    return params.schedule.scheduled_reb_date_to_tickers_universe_dict[date]


# =============================================================================
# Phase helper:  rebalance trade simulation
# =============================================================================

def _execute_rebalance_trades(
    date: pd.Timestamp,
    snapshot: BookSnapshot,
    trades_log: dict,
    intended_new_shares: pd.Series,
    current_shares: pd.Series,
    shares_to_trade: pd.Series,
    cure_method: str,
    params: BacktestParameters,
    prices: pd.Series,
    market_volume: pd.Series
) -> _TradeSimulationResult:
    """Execute the rebalance trades, with MR-cure handling."""
    rebalance_MR = mr.compute_rebalance_MR(
                        date = date,
                        mr_haircuts = params.data.mr_haircuts,
                        prices = prices,
                        current_shares = current_shares,
                        intended_new_shares = intended_new_shares,
                        trading_costs = params.data.trading_costs
    )

    # MR-compliant path
    if snapshot.equity >= rebalance_MR:
        snapshot, trades_log = bm.update_from_all_trades(
                                    current_snapshot = snapshot,
                                    trades_log = trades_log,
                                    market_volume = market_volume,
                                    date = date,
                                    shares_to_trade = shares_to_trade,
                                    trading_costs = params.data.trading_costs,
                                    prices_pre_trade = prices,
                                    prices_post_trade = prices
                                )
        return _TradeSimulationResult(
                    snapshot = snapshot,
                    trades_log = trades_log,
                    maintenance_MR_post_trades = _maint_mr(date, params, snapshot.shares),
                    flag_added_to_date = (EVENT_REBALANCE,),
                    posted_margin_collateral = None,
                    shrink_factor = None
        )

    # MR-violation cure paths
    current_equity = snapshot.equity

    if cure_method == "posting_collateral":
        posted = rebalance_MR - current_equity
        snapshot.margin_collateral += posted
        snapshot.update_equity_fields()
        snapshot.update_leverage_ratios()

        snapshot, trades_log = bm.update_from_all_trades(
                                    current_snapshot = snapshot,
                                    trades_log = trades_log,
                                    market_volume = market_volume,
                                    date = date,
                                    shares_to_trade = shares_to_trade,
                                    trading_costs = params.data.trading_costs,
                                    prices_pre_trade = prices,
                                    prices_post_trade = prices
                                )
        return _TradeSimulationResult(
                    snapshot = snapshot,
                    trades_log = trades_log,
                    maintenance_MR_post_trades = _maint_mr(date, params, snapshot.shares),
                    flag_added_to_date = (EVENT_REBALANCE, EVENT_REBALANCE_MR_CURE_COLLATERAL),
                    posted_margin_collateral = posted,
                    shrink_factor = None
        )

    if cure_method == "shrinking_exposures":
        shrink_factor = mr.compute_shrinking_factor(
                            current_equity,
                            date,
                            params.data.mr_haircuts,
                            prices,
                            params.data.trading_costs,
                            current_shares = current_shares,
                            intended_new_shares = intended_new_shares
        )
        shrunk_intended = tu.truncate_shares_to_int(shrink_factor * intended_new_shares)
        shares_to_trade = tu.compute_shares_to_trade(current_shares, shrunk_intended)

        snapshot, trades_log = bm.update_from_all_trades(
                                    current_snapshot = snapshot,
                                    trades_log = trades_log,
                                    market_volume = market_volume,
                                    date = date,
                                    shares_to_trade = shares_to_trade,
                                    trading_costs = params.data.trading_costs,
                                    prices_pre_trade = prices,
                                    prices_post_trade = prices
        )
        return _TradeSimulationResult(
                    snapshot = snapshot,
                    trades_log = trades_log,
                    maintenance_MR_post_trades = _maint_mr(date, params, snapshot.shares),
                    flag_added_to_date = (EVENT_REBALANCE, EVENT_REBALANCE_MR_CURE_SHRINK),
                    posted_margin_collateral = None,
                    shrink_factor = shrink_factor
        )

    raise ValueError(f"unknown cure_method_for_MR_violation: {cure_method!r}")


def _execute_rebalance(
    state: _BacktestState,
    params: BacktestParameters,
    date: pd.Timestamp,
    tickers: list[str],
    prices: pd.Series,
    market_volume: pd.Series,
    events_today: list[str],
    using_fixed_calendar: bool
) -> None:
    """Run the strategy at `date`, compute desired trades, execute
    them with MR-cure handling, and record events and metrics.
    """
    book_equity = state.book_working.equity
    current_shares = state.book_working.shares

    ctx = RebalanceContext(
        date = date,
        tickers = tickers,
        book_equity = book_equity,
        prices = prices,
        using_fixed_rebalance_calendar = using_fixed_calendar,
        scheduled_rebalance_dates = state.new_scheduled_rebalance_dates
    )
    strategy_output = params.strategy.determine_shares(ctx)
    intended_new_shares = strategy_output.shares

    if strategy_output.optimizer_results is not None:
        state.optimizer_results_at_rebalance_date[date] = strategy_output.optimizer_results
        logger.info(
            "optimizer status at %s: %s",
            date.strftime("%Y-%m-%d"),
            strategy_output.optimizer_results.metrics["status"]
        )

    shares_to_trade = tu.compute_shares_to_trade(current_shares, intended_new_shares)
    if shares_to_trade.empty:
        events_today.append(EVENT_REBALANCE)
        state.maintenance_MR_at_close[date] = _maint_mr(date, params, current_shares)
        return

    result = _execute_rebalance_trades(
                date = date,
                snapshot = state.book_working,
                trades_log = state.trades_log,
                intended_new_shares = intended_new_shares,
                current_shares = current_shares,
                shares_to_trade = shares_to_trade,
                cure_method = params.rules.cure_method_for_MR_violation,
                params = params,
                prices = prices,
                market_volume = market_volume
    )
    state.book_working = result.snapshot
    state.trades_log = result.trades_log
    state.maintenance_MR_at_close[date] = result.maintenance_MR_post_trades
    events_today.extend(result.flag_added_to_date)
    if result.posted_margin_collateral is not None:
        state.posted_collateral_at_MR_violation_cures[date] = result.posted_margin_collateral
    if result.shrink_factor is not None:
        state.shrink_factors_at_MR_violation_cures[date] = result.shrink_factor


# =============================================================================
# Phase helper: recording at close
# =============================================================================

def _record_at_close(
    state: _BacktestState, date: pd.Timestamp, events_today: list[str]
) -> None:
    """Finalize the snapshot at the close of `date` and write event flags."""
    if not events_today:
        events_today = [EVENT_MTM_ONLY]
    state.date_events[date] = tuple(events_today)
    state.book_at_date[date].close = bm.round_snapshot_for_display(state.book_working)


# =============================================================================
# Initialization
# =============================================================================

def _validate_params(params: BacktestParameters) -> None:
    valid_cure_methods = ("shrinking_exposures", "posting_collateral")
    if params.rules.cure_method_for_MR_violation not in valid_cure_methods:
        raise ValueError(
            f"cure_method_for_MR_violation must be one of {valid_cure_methods}, "
            f"got {params.rules.cure_method_for_MR_violation!r}"
        )
    if len(params.schedule.backtest_dates) < 2:
        raise ValueError("backtest_dates must contain at least 2 dates")
    first_date = params.schedule.backtest_dates[0]
    if first_date not in params.schedule.scheduled_reb_date_to_tickers_universe_dict:
        raise ValueError(
            f"first backtest date {first_date} missing from "
            f"scheduled_reb_date_to_tickers_universe_dict"
        )


def _initialize_state(params: BacktestParameters) -> _BacktestState:
    """Set up runtime state at the start of the backtest."""
    first_date = params.schedule.backtest_dates[0]
    tickers = params.schedule.scheduled_reb_date_to_tickers_universe_dict[first_date]

    book = bm.create_book(tickers, params.rules.initial_cash)
    book_working = copy.deepcopy(book.open)

    state = _BacktestState(
                book_working = book_working,
                new_scheduled_rebalance_dates= copy.deepcopy(params.schedule.scheduled_rebalance_dates)
            )
    state.actual_rebalance_dates = state.actual_rebalance_dates.append(pd.DatetimeIndex([first_date]))
    state.actual_reb_date_to_tickers_universe_dict[first_date] = tickers
    state.book_at_date[first_date] = Book(
            open = bm.round_snapshot_for_display(book.open),
            close = bm.round_snapshot_for_display(book.open)  # overwritten before moving to the second backtest date. See thread.
    )
    # First-date accruals are zero by convention (no prior day).
    state.equity_accruals_from_financing_costs[first_date] = 0.0
    state.equity_accruals_from_dividends[first_date] = 0.0
    return state


# =============================================================================
# Orchestrator
# =============================================================================

def run_backtest(params: BacktestParameters) -> BacktestResults:
    """Run the full backtest end-to-end."""
    _validate_params(params)

    backtest_dates = params.schedule.backtest_dates
    close_prices = params.data.price_data["close"]
    open_prices = params.data.price_data["open"]
    market_volume = params.data.price_data["volume"]
    using_fixed_calendar = (
        not params.rules.use_dynamic_rebalancing_rule
    )

    state = _initialize_state(params)
    first_date = backtest_dates[0]
    last_backtest_date = backtest_dates[-1]

    logger.info(
        "backtest started at rebalance date %s",
        first_date.strftime("%Y-%m-%d"),
    )

    # ----- First rebalance date (special-case) ---------------------------
    events_today: list[str] = []
    first_universe = params.schedule.scheduled_reb_date_to_tickers_universe_dict[first_date]
    _execute_rebalance(
            state, params, first_date, first_universe, close_prices, market_volume,
            events_today, using_fixed_calendar
    )
    _record_at_close(state, first_date, events_today)

    # ----- Daily loop ----------------------------------------------------
    for prev_date, date in zip(backtest_dates[:-1], backtest_dates[1:]):
        events_today = []

        # 1. Overnight P&L
        snapshot_at_open= copy.deepcopy(state.book_working)
        snapshot_at_open = _apply_overnight_pnl(
            state, params, prev_date, date, close_prices, snapshot_at_open
        )

        # 2. Mark to market at open prices
        snapshot_at_open = bm.mark_to_market_positions(
            date, snapshot_at_open, open_prices
        )
        
        # 3. Snapshot the book at the open; bifurcate to a working close snapshot
        state.book_at_date[date] = Book(
            open = bm.round_snapshot_for_display(snapshot_at_open),
            close = bm.round_snapshot_for_display(snapshot_at_open)  # overwritten below
        )
        state.book_working = copy.deepcopy(snapshot_at_open)

        # 4. Mark-to-market at close price 
        state.book_working = bm.mark_to_market_positions(
            date, state.book_working, close_prices
        )

        # 5. Maintenance MR check, and cure (if MR violation)
        _cure_mtm_violation(
            state, params, date, close_prices, market_volume, events_today
        )

        # 6. Termination tests 
        if _stop_loss_triggered(state, params, date, close_prices):
            events_today.append(EVENT_STOP_LOSS)
            logger.info("(%s) stop-loss triggered", date.strftime("%Y-%m-%d"))
            _record_at_close(state, date, events_today)
            break

        if _strategy_return_target_hit(state, params, first_date):
            events_today.append(EVENT_RETURN_TARGET_FOR_STRATEGY)
            logger.info(
                "(%s) hit_return_target_for_strategy", date.strftime("%Y-%m-%d"),
            )
            _record_at_close(state, date, events_today)
            break

        if date == last_backtest_date:
            events_today.append(EVENT_LAST_DATE)
            logger.info(
                "(%s) reached last scheduled backtest date",
                date.strftime("%Y-%m-%d"),
            )
            _record_at_close(state, date, events_today)
            break

        # 7. Inter-rebalance return target rule
        if _inter_rebalance_target_hit(state, params):
            events_today.append(EVENT_INTER_REB_RETURN_TARGET)
            state.schedule_diverged_from_scheduled = True
            state.actual_rebalance_dates = state.actual_rebalance_dates.append(pd.DatetimeIndex([date]))
            state.new_scheduled_rebalance_dates = _update_future_rebalance_schedule(
                params, state.actual_rebalance_dates, last_backtest_date, date
            )
            logger.info(
                "(%s) hit inter-rebalance return target; rebalancing today, ahead of schedule",
                date.strftime("%Y-%m-%d"),
            )
            tickers = _build_pit_universe_for_dynamic_rebalance(state, params, date)
            state.actual_reb_date_to_tickers_universe_dict[date] = tickers
            _execute_rebalance(
                state, params, date, tickers, close_prices, market_volume,
                events_today, using_fixed_calendar
            )
            _record_at_close(state, date, events_today)
            continue

        # 8. Scheduled rebalance date?
        if date not in state.new_scheduled_rebalance_dates:
            _record_at_close(state, date, events_today)
            continue

        state.actual_rebalance_dates = state.actual_rebalance_dates.append(pd.DatetimeIndex([date]))
        logger.info("(%s) scheduled rebalance date", date.strftime("%Y-%m-%d"))

        tickers = _resolve_universe_at_rebalance_date(state, params, date)
        state.actual_reb_date_to_tickers_universe_dict[date] = tickers
        
        _execute_rebalance(
            state, params, date, tickers, close_prices, market_volume,
            events_today, using_fixed_calendar
        )
        _record_at_close(state, date, events_today)

    logger.info("backtest finished")

    return BacktestResults(
            strategy = params.strategy,
            return_target_for_strategy = params.rules.return_target_for_strategy,
            return_target_for_inter_rebalance_period = params.rules.return_target_for_inter_rebalance_period,
            actual_rebalance_dates = state.actual_rebalance_dates,
            actual_reb_date_to_tickers_universe_dict = state.actual_reb_date_to_tickers_universe_dict,
            inter_rebalance_periods_of_same_duration = using_fixed_calendar,
            book_at_date = state.book_at_date,
            date_events = state.date_events,
            trades_log = state.trades_log,
            maintenance_MR_at_close = state.maintenance_MR_at_close,
            cure_method_for_MR_violation = params.rules.cure_method_for_MR_violation,
            shrink_factors_at_MR_violation_cures = state.shrink_factors_at_MR_violation_cures,
            posted_collateral_at_MR_violation_cures = state.posted_collateral_at_MR_violation_cures,
            equity_accruals_from_financing_costs = state.equity_accruals_from_financing_costs,
            equity_accruals_from_dividends = state.equity_accruals_from_dividends,
            optimizer_results_at_rebalance_date = state.optimizer_results_at_rebalance_date
    )