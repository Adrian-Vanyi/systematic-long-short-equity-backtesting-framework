"""Utilities used for comparing the backtest results for different strategies.

This module bundles the helpers used by the `strategy_comparison.ipyb` notebook to sweep many backtests 
across grids of start dates, strategy return targets, strategies, and rebalance-rule settings, and then aggregate
 the results into comparison tables (see documentation, §13).

Public API
----------
* class `FixedBacktestConfig`
        settings held constant across every run in a comparison grid (rebalance frequency, rolling windows, 
        initial cash, stop-loss trigger, MR cure method, etc.).

* class `BacktestDataBundle` 
        pre-loaded market data, point-in-time universe mapping, and risk-free rate series shared across runs.

* class `RunResult`  
       record of a single backtest's relevant output.
    
* function `run_one_backtest(...)` 
        end-to-end orchestrator that returns a `RunResult`. 

* function `aggregate_to_table(...)` 
        aggregates a flat `RunResult` DataFrame into  a (return_target x group) comparison table.

* function `style_comparison_table(...)`
        apply per-metric value-formatting and vertical column separators.

* function `format_sharpe_records(...)`
        arrange Sharpe metrics by strategy and historical date window.

* function `style_sharpe_table(...)`
        per-metric value-formatting, vertical column separators, and a red/neutral/green color map applied to selected metrics.        

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from matplotlib.colors import LinearSegmentedColormap
import logging
import numpy as np
import pandas as pd

from modules import backtest as b
from modules import backtest_calendar as bcal
from modules import beta_computation as bc
from modules import covariance_matrix as cm
from modules import factors_engineering as fe
from modules import kpis as kpi_mod
from modules import margin_requirement as mr
from modules import market_data as md
from modules import rates as r
from modules import returns_prediction_model as rpm
from modules import strategies as strat
from modules import trading_utils as tu
from modules import universe_construction as uc


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration classes
# ---------------------------------------------------------------------------

@dataclass
class FixedBacktestConfig:
    """Settings held fixed across every run in a comparison grid.
    Defaults match the example values specified in the documentation (§13).
    """
    # Rebalance frequency and horizon
    rebalance_freq_type: str = "ndays"
    rebalance_frequency_trading_days: int = 23
    number_of_inter_rebalance_periods: int = 12  

    # Estimation windows
    rolling_periods_for_cov_matrix_estimation: int = 18
    rolling_periods_for_beta_regression: int = 12
    rolling_periods_for_avg_return_computation: int = 12
    rolling_periods_for_factors_model_regression: int = 6
    rolling_window_trading_days_for_volatility: int = 60
    rolling_window_trading_days_for_momentum: int = 4 * 23
    buffer_trading_days_for_momentum: int = 23

    initial_cash: float = 100_000.0
    pct_initial_equity_triggering_stop_loss: float = 0.5
    cure_method_for_mr_violation: str = "shrinking_exposures"

    # Inter-rebalance return target used when the rule is enabled
    inter_rebalance_target: float = 0.03/12


@dataclass
class BacktestDataBundle:
    """Pre-loaded market data shared across all comparison runs."""
    price_data: pd.DataFrame
    dividend_data: pd.Series
    daily_spy_returns: pd.Series
    rf_daily_returns: pd.Series
    effr: pd.Series
    first_trading_day_of_month_to_sp500_members_dict: dict
    all_tickers_during_backtest: list[str]


@dataclass
class RunResult:
    """Record of a single backtest's relevant output."""
    start_date: pd.Timestamp
    strategy_name: str
    return_target: float
    use_inter_rebalance_rule: bool
    inter_rebalance_target: float | None
    cause_of_termination: str
    trading_days_run: int
    total_return: float
    target_hit: bool
    error: str | None = None

    @property
    def time_to_target(self) -> float:
        """Trading days to target hit; NaN if target was not hit."""
        return float(self.trading_days_run) if self.target_hit else float("nan")
    

@dataclass
class Results_Sharpe:
    initial_equity: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    rebalance_freq_type: str
    rebalance_frequency_trading_days: int | Literal["nan"]
    use_inter_rebalance_rule: bool  
    strategy_name: str
    daily_sharpe: float | Literal["nan"]
    annualized_sharpe: float | Literal["nan"]
    max_drawdown: float | Literal["nan"]
    n_daily_returns: int | Literal["nan"]
    error: str | None       
         


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_calendar_and_universes(
    start_date: pd.Timestamp,
    config: FixedBacktestConfig,
    data: BacktestDataBundle
) -> tuple[bcal.BacktestCalendar, dict[pd.Timestamp, list[str]]]:
    """Build the backtest calendar and per-rebalance-date universe mapping."""
    n_training = max(
        config.rolling_periods_for_factors_model_regression,
        config.rolling_periods_for_cov_matrix_estimation,
        config.rolling_periods_for_beta_regression,
        config.rolling_periods_for_avg_return_computation
    )
    cal = bcal.build_calendar(
            start = start_date,
            freq_type = config.rebalance_freq_type,
            P = config.number_of_inter_rebalance_periods,
            P_train = n_training,
            freq = config.rebalance_frequency_trading_days
    )
    fetch_start, fetch_end = md.compute_price_data_window(
            cal.backtest_dates,
            cal.training_rebalance_dates,
            config.rolling_window_trading_days_for_momentum,
            config.buffer_trading_days_for_momentum,
            config.rolling_window_trading_days_for_volatility,
            rolling_window_trading_days_for_adv = 60
    )
    if fetch_start < pd.Timestamp("2008-01-01") or fetch_end >pd.Timestamp("2026-03-01"):
        raise ValueError(
            f"start {start_date:%Y-%m-%d} requires data outside supported window"
        )
    reb_to_universe, _ = uc.build_pit_universes(
            cal.training_rebalance_dates,
            final_period_end_date = cal.backtest_dates[-1],
            first_trading_day_of_month_to_sp500_members_dict = (
                data.first_trading_day_of_month_to_sp500_members_dict
        ),
        price_data = data.price_data,
        require_prices_for_next_period=True,
        capping_num_tickers_per_universe = None
    )
    return cal, reb_to_universe


def _build_costs_and_rates(
    backtest_dates: pd.DatetimeIndex,
    data: BacktestDataBundle
) -> tuple[mr.MRHaircuts, r.DailyFinancingRates, tu.TradingCosts]:
    """Build the per-backtest costs and rates."""
    mr_haircuts = mr.MRHaircuts(
            backtest_dates = backtest_dates,
            tickers = data.all_tickers_during_backtest,
            initial_long_haircut = 0.5,
            initial_short_haircut = 0.5,
            maintenance_long_haircut = 0.25,
            maintenance_short_haircut = 0.30
    )
    daily_financing_rates = r.DailyFinancingRates(
            backtest_dates = backtest_dates,
            annual_cash_rate = data.effr - 0.0042,
            annual_debit_rate = data.effr + 0.0300,
            annual_margin_collateral_rate = 0.0,
            annual_borrow_fee = 0.0025,
            annual_rebate_rate = 0.0,
            day_count_basis = r.ACT_360
    )
    trading_costs = tu.TradingCosts(
            backtest_dates = backtest_dates,
            tickers = data.all_tickers_during_backtest,
            default_slippage_bps = 3.0,
            default_full_spread_bps = 2.0,
            default_commission_per_share = 0.001
    )
    return mr_haircuts, daily_financing_rates, trading_costs


def _configure_strategy(
    strategy_name: str,
    cal: bcal.BacktestCalendar,
    reb_to_universe: dict,
    use_inter_rebalance_rule: bool,
    config: FixedBacktestConfig,
    data: BacktestDataBundle
) -> strat.BaseStrategy :
    """Configure the used strategy.

    Returns a `strat.BaseStrategy` object configured for a static rebalance calendar if and only
    if `use_inter_rebalance_rule` is`False; otherwise the strategy is built in dynamic mode (no precomputed inputs).
    """
    using_fixed = not use_inter_rebalance_rule
    last_backtest_date = cal.backtest_dates[-1]

    if strategy_name == "momentum":
        precomputed = None
        if using_fixed:
            precomputed = fe.compute_momentum_at_training_rebalance_dates(
                    prices = data.price_data["close"],
                    training_rebalance_dates = cal.rebalance_dates,
                    rolling_window_trading_days = config.rolling_window_trading_days_for_momentum,
                    buffer_trading_days = config.buffer_trading_days_for_momentum,
                    reb_date_to_tickers_universe_dict = reb_to_universe
            )
        return strat.MomentumStrategy(
                top_k_bottom_k = 10,
                shares_per_ticker = 100,
                rolling_window_trading_days_for_momentum = config.rolling_window_trading_days_for_momentum,
                buffer_trading_days_for_momentum = config.buffer_trading_days_for_momentum,
                precomputed_momentums=precomputed
        )
    
    if strategy_name == "factors_model":
        precomputed_predictions = None
        if using_fixed:
            factors = fe.compute_factors_at_training_rebalance_dates(
                    prices = data.price_data["close"],
                    training_rebalance_dates = cal.training_rebalance_dates,
                    rolling_window_trading_days_for_momentum = config.rolling_window_trading_days_for_momentum,
                    buffer_trading_days_for_momentum = config.buffer_trading_days_for_momentum,
                    rolling_window_trading_days_for_volatility = config.rolling_window_trading_days_for_volatility,
                    reb_date_to_tickers_universe_dict = reb_to_universe,
                    winsorize_factors_per_date = True,
                    z_score_factors_per_date = True
            )
            realized = rpm.compute_realized_period_returns(
                    prices = data.price_data["close"],
                    training_rebalance_dates = cal.training_rebalance_dates,
                    final_period_end_date = last_backtest_date,
                    reb_date_to_tickers_universe_dict = reb_to_universe
            )
            model_input = pd.concat([factors, realized], axis=1, join="inner")
            fm = rpm.fit_predict_factors_model_at_rebalance_dates(
                    rebalance_dates = cal.rebalance_dates,
                    training_rebalance_dates = cal.training_rebalance_dates,
                    model_input = model_input,
                    rolling_periods = config.rolling_periods_for_factors_model_regression,
                    use_ridge=False
            )
            precomputed_predictions = fm.predictions

        return strat.FactorsModelStrategy(
                top_k_bottom_k = 10,
                shares_per_ticker = 100,
                training_rebalance_dates = cal.training_rebalance_dates,
                rebalance_freq_type = config.rebalance_freq_type,
                rebalance_frequency_trading_days = config.rebalance_frequency_trading_days,
                last_backtest_date = last_backtest_date,
                rolling_window_trading_days_for_momentum = config.rolling_window_trading_days_for_momentum,
                buffer_trading_days_for_momentum = config.buffer_trading_days_for_momentum,
                rolling_window_trading_days_for_volatility = config.rolling_window_trading_days_for_volatility,
                rolling_periods_for_factors_model_regression = config.rolling_periods_for_factors_model_regression,
                winsorize_factors_per_date = True,
                z_score_factors_per_date = True,
                precomputed_predictions = precomputed_predictions
        )

    if strategy_name in ("mvo_max_return", "mvo_min_variance"):
        if strategy_name == "mvo_max_return":
            targets = strat.MVOOptimizerTargets(
                    objective = "max_return",
                    ptf_variance_cap = 0.04,
                    ptf_beta_cap = 0.001,
                    ptf_net_exposure_cap = 0.001,
                    ptf_gross_exposure_cap = 2.0,
                    max_shares_per_ticker = 10_000
            )
        else:  # mvo_min_variance
            targets = strat.MVOOptimizerTargets(
                    objective = "min_variance",
                    ptf_return_lower_bound = 0.03,
                    ptf_beta_cap = 0.001,
                    ptf_net_exposure_cap = 0.001,
                    ptf_gross_exposure_cap = 2.0,
                    max_shares_per_ticker = 10_000
            )
        estimation = strat.MVOEstimationSettings(
                next_period_expected_returns_estimation_method = "factors_model",
                rolling_periods_for_factors_model_regression = config.rolling_periods_for_factors_model_regression,
                rolling_window_trading_days_for_momentum = config.rolling_window_trading_days_for_momentum,
                buffer_trading_days_for_momentum = config.buffer_trading_days_for_momentum,
                rolling_window_trading_days_for_volatility = config.rolling_window_trading_days_for_volatility,
                winsorize_factors_per_date = True,
                z_score_factors_per_date = True,
                rolling_periods_for_avg_return_computation = config.rolling_periods_for_avg_return_computation,
                avg_type = "regular",
                halflife_ew = None,
                rolling_periods_for_cov_matrix_estimation = config.rolling_periods_for_cov_matrix_estimation,
                rolling_periods_for_beta_regression = config.rolling_periods_for_beta_regression,
        )

        if using_fixed:
            factors = fe.compute_factors_at_training_rebalance_dates(
                    prices = data.price_data["close"],
                    training_rebalance_dates = cal.training_rebalance_dates,
                    rolling_window_trading_days_for_momentum = config.rolling_window_trading_days_for_momentum,
                    buffer_trading_days_for_momentum = config.buffer_trading_days_for_momentum,
                    rolling_window_trading_days_for_volatility = config.rolling_window_trading_days_for_volatility,
                    reb_date_to_tickers_universe_dict = reb_to_universe,
                    winsorize_factors_per_date = True,
                    z_score_factors_per_date = True
            )
            realized = rpm.compute_realized_period_returns(
                    prices = data.price_data["close"],
                    training_rebalance_dates = cal.training_rebalance_dates,
                    final_period_end_date = last_backtest_date,
                    reb_date_to_tickers_universe_dict = reb_to_universe
            )
            model_input = pd.concat([factors, realized], axis=1, join="inner")

            fm = rpm.fit_predict_factors_model_at_rebalance_dates(
                    rebalance_dates = cal.rebalance_dates,
                    training_rebalance_dates = cal.training_rebalance_dates,
                    model_input = model_input,
                    rolling_periods = config.rolling_periods_for_factors_model_regression,
                    use_ridge=False
            )
            mu = fm.predictions
            sigma_result = cm.compute_rolling_cov_matrices(
                    prices = data.price_data["close"],
                    training_rebalance_dates = cal.training_rebalance_dates,
                    rebalance_dates = cal.rebalance_dates,
                    rolling_periods_for_estimation = config.rolling_periods_for_cov_matrix_estimation
            )
            betas = bc.compute_capm_betas(
                    rf_daily_returns = data.rf_daily_returns,
                    market_daily_returns = data.daily_spy_returns,
                    training_rebalance_dates = cal.training_rebalance_dates,
                    rebalance_dates = cal.rebalance_dates,
                    prices = data.price_data["close"],
                    rolling_periods_for_beta_regression = config.rolling_periods_for_beta_regression
            )
            precomputed = strat.MVOPrecomputedInputs(
                    expected_returns_predictions_at_reb_dates = mu,
                    cov_matrices_at_reb_dates = sigma_result.matrices,
                    tickers_betas_at_reb_dates = betas
            )
        else:
            precomputed = strat.MVOPrecomputedInputs()

        return strat.MeanVarianceOptimizationStrategy(
                targets = targets,
                estimation = estimation,
                precomputed = precomputed,
                rebalance_freq_type = config.rebalance_freq_type,
                rebalance_frequency_trading_days = config.rebalance_frequency_trading_days,
                last_backtest_date = last_backtest_date,
                rf_daily_returns = data.rf_daily_returns,
                market_daily_returns = data.daily_spy_returns
        )

    raise ValueError(f"unknown strategy_name: {strategy_name!r}")


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def run_one_backtest(
    start_date: pd.Timestamp,
    strategy_name: str,
    use_return_target_hit_rule: bool,
    return_target: float | None,
    use_inter_rebalance_rule: bool,
    inter_rebalance_target: float | None,
    config: FixedBacktestConfig,
    data: BacktestDataBundle
) -> RunResult:
    """Run a single backtest end-to-end and return the relevant outputs for the comparison of strategies.

    Errors during a run (e.g. an infeasible MVO target on a specific date) are caught and returned in the
    `error` field of the result (instance of `RunResult`) rather than raised, so the grid completes even if
    an individual backtest fail.

    When `inter_rebalance_target` is None the value falls back to `config.inter_rebalance_target`.
    """
    if inter_rebalance_target is None:
        inter_rebalance_target = config.inter_rebalance_target
    try:
        cal, reb_to_universe = _build_calendar_and_universes(start_date, config, data)
        mr_haircuts, daily_financing_rates, trading_costs = _build_costs_and_rates(cal.backtest_dates,  data)
        strategy = _configure_strategy(
            strategy_name, cal, reb_to_universe, use_inter_rebalance_rule, config, data,
        )
        schedule = b.BacktestSchedule(
                backtest_dates = cal.backtest_dates,
                scheduled_rebalance_dates = cal.rebalance_dates,
                scheduled_reb_date_to_tickers_universe_dict = reb_to_universe,
                first_trading_day_of_month_to_sp500_members_dict = (
                    data.first_trading_day_of_month_to_sp500_members_dict
                ),
                scheduled_rebalance_freq_type = config.rebalance_freq_type,
                scheduled_rebalance_frequency_trading_days = config.rebalance_frequency_trading_days,
                scheduled_number_of_inter_rebalance_periods_for_backtest = len(cal.rebalance_dates) - 1,
                capping_num_tickers_per_universe = None
        )
        bdata = b.BacktestData(
                price_data = data.price_data,
                dividend_data = data.dividend_data,
                trading_costs = trading_costs,
                daily_financing_rates = daily_financing_rates,
                mr_haircuts = mr_haircuts
        )
        rules = b.BacktestRules(
                initial_cash = config.initial_cash,
                use_return_target_hit_rule = use_return_target_hit_rule,
                return_target_for_strategy = return_target,
                use_dynamic_rebalancing_rule = use_inter_rebalance_rule,
                return_target_for_inter_rebalance_period = inter_rebalance_target,
                cure_method_for_MR_violation = config.cure_method_for_mr_violation,
                pct_initial_equity_triggering_stop_loss = config.pct_initial_equity_triggering_stop_loss
        )
        params = b.BacktestParameters(
                strategy = strategy,
                schedule = schedule,
                data = bdata,
                rules = rules
        )

        results = b.run_backtest(params)
        kpis = kpi_mod.compute_backtest_kpis(backtest_results = results, rf_daily_returns = data.rf_daily_returns)

        cause = kpis.backtest_duration["cause_of_backtest_termination"]
        n_days = int(kpis.backtest_duration["number_of_backtest_days"])
        total_return = float(kpis.backtest_PnL["unrealized_return_of_equity"])
        target_hit = cause == "hit_return_target_for_strategy"

        return RunResult(
                start_date = start_date,
                strategy_name = strategy_name,
                return_target = return_target,
                use_inter_rebalance_rule = use_inter_rebalance_rule,
                inter_rebalance_target = (
                    inter_rebalance_target if use_inter_rebalance_rule else None
                ),
                cause_of_termination = cause,
                trading_days_run = n_days,
                total_return = total_return,
                target_hit = target_hit
        )

    except Exception as exc:
        logger.warning(
            "run failed: start=%s strategy=%s target=%s rule=%s err=%s",
            start_date.date(), strategy_name, return_target, use_inter_rebalance_rule, exc,
        )
        return RunResult(
                start_date = start_date,
                strategy_name = strategy_name,
                return_target = return_target,
                use_inter_rebalance_rule = use_inter_rebalance_rule,
                inter_rebalance_target = (
                    inter_rebalance_target if use_inter_rebalance_rule else None
                ),
                cause_of_termination = "error",
                trading_days_run = 0,
                total_return = float("nan"),
                target_hit = False,
                error = str(exc)
        )


def compute_sharpe_ratio(
    start_date: pd.Timestamp,
    strategy_name: str,
    use_inter_rebalance_rule:bool,
    config: FixedBacktestConfig,
    data: BacktestDataBundle
) -> Results_Sharpe:
    """Run a single backtest end-to-end and return the results from the sharpe ratio computation.

    Errors during a run (e.g. an infeasible MVO target on a specific date) are caught and returned in the
    `error` field of the result (instance of `Results_Sharpe`) rather than raised, so the grid completes even if
    an individual backtest fail.
    """
    try:
        cal, reb_to_universe = _build_calendar_and_universes(start_date, config, data)
        mr_haircuts, daily_financing_rates, trading_costs = _build_costs_and_rates(cal.backtest_dates,  data)
        strategy = _configure_strategy(
            strategy_name, cal, reb_to_universe, use_inter_rebalance_rule, config, data,
        )
        schedule = b.BacktestSchedule(
                backtest_dates = cal.backtest_dates,
                scheduled_rebalance_dates = cal.rebalance_dates,
                scheduled_reb_date_to_tickers_universe_dict = reb_to_universe,
                first_trading_day_of_month_to_sp500_members_dict = (
                    data.first_trading_day_of_month_to_sp500_members_dict
                ),
                scheduled_rebalance_freq_type = config.rebalance_freq_type,
                scheduled_rebalance_frequency_trading_days = config.rebalance_frequency_trading_days,
                scheduled_number_of_inter_rebalance_periods_for_backtest = len(cal.rebalance_dates) - 1,
                capping_num_tickers_per_universe = None
        )
        bdata = b.BacktestData(
                price_data = data.price_data,
                dividend_data = data.dividend_data,
                trading_costs = trading_costs,
                daily_financing_rates = daily_financing_rates,
                mr_haircuts = mr_haircuts
        )
        rules = b.BacktestRules(
                initial_cash = config.initial_cash,
                use_return_target_hit_rule = False,
                return_target_for_strategy = None, 
                use_dynamic_rebalancing_rule = use_inter_rebalance_rule,
                return_target_for_inter_rebalance_period = config.inter_rebalance_target,
                cure_method_for_MR_violation = config.cure_method_for_mr_violation,
                pct_initial_equity_triggering_stop_loss = config.pct_initial_equity_triggering_stop_loss
        )
        params = b.BacktestParameters(
                strategy = strategy,
                schedule = schedule,
                data = bdata,
                rules = rules
        )

        results = b.run_backtest(params)


        book_at_date = results.book_at_date
        backtest_dates = sorted(book_at_date.keys())


        eq_at_backtest_dates = pd.Series(
            {
                date : book.close.equity_excluding_margin_collateral
                for date, book in book_at_date.items()
            }
        )
        return_since_prev = eq_at_backtest_dates.pct_change(fill_method=None)


        rf_daily_returns = data.rf_daily_returns
        rf_aligned = rf_daily_returns.reindex(backtest_dates)


        excess_return = (return_since_prev - rf_aligned).dropna()

        avg = excess_return.mean() if not excess_return.empty else float("nan")
        std = excess_return.std(ddof=1) if len(excess_return) > 1 else float("nan")

        daily_sharpe = avg/std

        annualized_sharpe = np.sqrt(252) * daily_sharpe

        drawdowns = kpi_mod.compute_drawdowns(book_at_date)

        max_drawdown = drawdowns.min()

        end_date = backtest_dates[-1]
        rebalance_freq_type = config.rebalance_freq_type
        rebalance_frequency_trading_days = config.rebalance_frequency_trading_days

        n_daily_returns = len(excess_return)

        return Results_Sharpe(
            initial_equity = config.initial_cash,
            start_date = start_date,
            end_date = end_date,
            rebalance_freq_type = rebalance_freq_type,
            rebalance_frequency_trading_days = rebalance_frequency_trading_days if rebalance_freq_type == "ndays" else "nan",
            use_inter_rebalance_rule = use_inter_rebalance_rule,
            strategy_name = strategy_name,
            daily_sharpe = daily_sharpe,
            annualized_sharpe = annualized_sharpe,
            n_daily_returns = n_daily_returns,
            max_drawdown =  max_drawdown,
            error = None
        )
    
    except Exception as exc:
        number_of_inter_rebalance_periods = config.number_of_inter_rebalance_periods
        logger.warning(
            "run failed: start=%s strategy=%s number_of_inter_rebalance_periods= %s  err=%s",
            start_date.date(), strategy_name, number_of_inter_rebalance_periods,  exc,
        )
        return Results_Sharpe(
            initial_equity = config.initial_cash,
            start_date = start_date,
            end_date = "nan",
            rebalance_freq_type = "nan",
            rebalance_frequency_trading_days ="nan",
            use_inter_rebalance_rule = use_inter_rebalance_rule,
            strategy_name = strategy_name,
            daily_sharpe = "nan",
            annualized_sharpe = "nan",
            n_daily_returns = "nan",
            max_drawdown = "nan",
            error = str(exc)        
         )



# ---------------------------------------------------------------------------
# Aggregation and display
# ---------------------------------------------------------------------------

def aggregate_to_table(
    raw: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    return_targets: list[float],
) -> pd.DataFrame:
    """Aggregate raw run records into a (return_target x group) table.

    Output: MultiIndex columns with level 0 = metric name, level 1 = group.
    Rows: return_target (reindexed to `return_targets`).
    """

    def n_start_dates_tested(sub: pd.DataFrame) -> pd.Series:
        return sub.groupby("return_target").size().astype(int)

    def frequency_target_hit(sub: pd.DataFrame) -> pd.Series:
        return sub.groupby("return_target")["target_hit"].mean()

    def avg_time_to_target_if_target_hit(sub: pd.DataFrame) -> pd.Series:
        hit = sub[sub["target_hit"]]
        return hit.groupby("return_target")["time_to_target"].mean()

    metrics = {
        "n_start_dates_tested": n_start_dates_tested,
        "frequency_target_hit": frequency_target_hit,
        "avg_time_to_target_if_target_hit": avg_time_to_target_if_target_hit
    }

    sections: dict[str, pd.DataFrame] = {}
    for metric_name, fn in metrics.items():
        per_group = {
            group_value: fn(raw[raw[group_col] == group_value])
            for group_value in group_order
        }
        sections[metric_name] = pd.DataFrame(per_group)

    out = pd.concat(sections, axis=1)
    out.columns.names = ["metric", group_col]
    out.index.name = "return_target"
    out = out.reindex(return_targets)
    return out


_METRIC_FORMATS = {
    "n_start_dates_tested": "{:d}",
    "frequency_target_hit": "{:.1%}",
    "avg_time_to_target_if_target_hit": "{:.1f}",
    "avg_annualized_daily_sharpe_if_target_hit": "{:.2f}",
    "n_daily_returns": "{:d}",
    "daily_sharpe": "{:.2f}",
    "annualized_sharpe": "{:.2f}",
    "max_drawdown": "{:.2%}"
}

def _format_cell(val, metric: str) -> str:
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    fmt = _METRIC_FORMATS.get(metric)
    if fmt is None:
        return str(val)
    if metric in ("n_start_dates_tested","n_daily_returns"):
        return fmt.format(int(val))
    return fmt.format(float(val))


def style_comparison_table(df: pd.DataFrame):
    """Apply per-metric value-formatting and vertical column separators:
    solid borders between metric groups, dashed borders between sub-columns.
    """
    formatted = df.copy()
    for col in formatted.columns:
        metric = col[0]
        formatted[col] = formatted[col].map(lambda v, m=metric: _format_cell(v, m))
    metrics = list(formatted.columns.get_level_values(0).unique())
    n_groups_per_metric = formatted.columns.get_level_values(0).value_counts().max()
    n_index_cols = formatted.index.nlevels
    solid = ("border-left", "2px solid #333")
    dashed = ("border-left", "1px dashed #666")
    table_styles = []

    for i in range(len(metrics)):
        # --- solid separator before each metric group ---
        row1_pos = n_index_cols + i + 1
        row2_pos = n_index_cols + i * n_groups_per_metric + 1
        body_pos = row2_pos
        table_styles.append({
            "selector": f"thead tr:nth-child(1) th:nth-child({row1_pos})",
            "props": [solid],
        })
        table_styles.append({
            "selector": f"thead tr:nth-child(2) th:nth-child({row2_pos})",
            "props": [solid],
        })
        table_styles.append({
            "selector": f"tbody tr td:nth-child({body_pos})",
            "props": [solid],
        })

        # --- dashed separators between sub-columns within the group ---
        for j in range(1, n_groups_per_metric):
            sub_pos = n_index_cols + i * n_groups_per_metric + j + 1
            table_styles.append({
                "selector": f"thead tr:nth-child(2) th:nth-child({sub_pos})",
                "props": [dashed],
            })
            table_styles.append({
                "selector": f"tbody tr td:nth-child({sub_pos})",
                "props": [dashed],
            })

    return (
        formatted.style
        .format_index(lambda v: f"{v:.1%}", axis=0)
        .set_table_styles(table_styles)
    )


def format_sharpe_records(sharpe_records: pd.DataFrame) -> pd.DataFrame:
    """Return Sharpe metrics grouped by strategy across historical date windows."""
    sharpe_records["historical window"] = (
        pd.to_datetime(sharpe_records["start_date"]).dt.strftime("(%Y-%m-%d)")
        + " - "
        + pd.to_datetime(sharpe_records["end_date"]).dt.strftime("(%Y-%m-%d)")
    )

    metrics = ["n_daily_returns", "daily_sharpe", "annualized_sharpe", "max_drawdown"]

    result = sharpe_records.pivot_table(
        index="strategy_name",
        columns="historical window",
        values=metrics,
        aggfunc="first",
    )

    result = result.reindex(metrics, axis=1, level=0)
    result.columns.names = ["", "historical window"]
    result.index.name = "strategy"
    return result


COLOR_METRICS = {"daily_sharpe", "annualized_sharpe", "max_drawdown"}

SATURATION_AT = {
    "daily_sharpe":      0.06,
    "annualized_sharpe": 1.0,
    "max_drawdown":      0.80,   # saturate at -50%
}


def style_sharpe_table(df: pd.DataFrame):
    """Per-metric value-formatting, vertical column separators, and a diverging
    red/neutral/green color map applied only to selected metrics.

    Coloring rule: the sign of the value picks the hue (negative -> red,
    positive -> green) and the magnitude drives intensity, saturating at a
    per-metric threshold (SATURATION_AT). max_drawdown is one-sided (negative
    only) and uses a sequential red map.
    """
    # --- formatting (display strings)
    formatted = df.copy()
    for col in formatted.columns:
        metric = col[0]
        formatted[col] = formatted[col].map(lambda v, m=metric: _format_cell(v, m))

    metrics = list(formatted.columns.get_level_values(0).unique())
    n_groups_per_metric = formatted.columns.get_level_values(0).value_counts().max()
    n_index_cols = formatted.index.nlevels
    solid = ("border-left", "2px solid #333")
    dashed = ("border-left", "1px dashed #666")
    table_styles = []

    # --- vertical separators between/within metric groups
    for i in range(len(metrics)):
        row1_pos = n_index_cols + i + 1
        row2_pos = n_index_cols + i * n_groups_per_metric + 1
        body_pos = row2_pos
        table_styles.append({
            "selector": f"thead tr:nth-child(1) th:nth-child({row1_pos})",
            "props": [solid],
        })
        table_styles.append({
            "selector": f"thead tr:nth-child(2) th:nth-child({row2_pos})",
            "props": [solid],
        })
        table_styles.append({
            "selector": f"tbody tr td:nth-child({body_pos})",
            "props": [solid],
        })
        for j in range(1, n_groups_per_metric):
            sub_pos = n_index_cols + i * n_groups_per_metric + j + 1
            table_styles.append({
                "selector": f"thead tr:nth-child(2) th:nth-child({sub_pos})",
                "props": [dashed],
            })
            table_styles.append({
                "selector": f"tbody tr td:nth-child({sub_pos})",
                "props": [dashed],
            })

    # --- diverging colormap: 0 -> neutral, sign -> hue, extreme -> saturated ---
    cmap = LinearSegmentedColormap.from_list(
        "sharpe_rwg",
        [
            (0.00, "#8b0000"),  # |v| >= threshold, negative: deep red
            (0.25, "#e57373"),  # mid negative
            (0.48, "#fbe9e9"),  # just below 0: faint red
            (0.50, "#f5f5f5"),  # exactly 0: neutral
            (0.52, "#e7f4e7"),  # just above 0: faint green
            (0.75, "#66bb6a"),  # mid positive
            (1.00, "#1b5e20"),  # |v| >= threshold, positive: deep green
        ],
    )

    # --- sequential red colormap for one-sided (negative-only) drawdown ---
    cmap_dd = LinearSegmentedColormap.from_list(
        "dd_reds",
        [
            (0.00, "#f5f5f5"),  # 0% drawdown: neutral
            (0.50, "#e57373"),  # mid
            (1.00, "#8b0000"),  # |v| >= threshold: deep red
        ],
    )

    def _color(raw_val, m):
        if pd.isna(raw_val):
            return "background-color: black; color: white"
        sat = SATURATION_AT.get(m, 1.0)

        if m == "max_drawdown":
            # negative-only: intensity = magnitude / sat, saturating at -sat
            t = float(np.clip(abs(raw_val) / abs(sat), 0.0, 1.0))
            r, g, b = (int(x * 255) for x in cmap_dd(t)[:3])
            text = "white" if t > 0.6 else "black"
            return f"background-color: rgb({r},{g},{b}); color: {text}"

        t = float(np.clip((raw_val + sat) / (2 * sat), 0.0, 1.0))
        r, g, b = (int(x * 255) for x in cmap(t)[:3])
        text = "white" if (t > 0.78 or t < 0.22) else "black"
        return f"background-color: rgb({r},{g},{b}); color: {text}"

    styler = formatted.style.set_table_styles(table_styles)
    for col in df.columns:
        metric = col[0]
        if metric not in COLOR_METRICS:
            continue
        colors = df[col].map(lambda v, m=metric: _color(v, m))
        styler = styler.apply(lambda s, c=colors: c, subset=[col], axis=0)

    return styler