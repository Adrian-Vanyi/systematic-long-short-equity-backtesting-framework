"""Trading strategies for a long/short equity portfolio

Design
------
Every concrete strategy is a child class of the abstract class `BaseStrategy`, and implements its version of `determine_shares(ctx)`
 (each strategy reads the fields of `ctx` they need and ignore the rest).

Each strategy also implements `kpi_parameters()`, which returns its
parameters as a dict for KPIs reporting; 

Public API
----------
* class `RebalanceContext`
        per-rebalance frozen inputs the backtest orchestrator hands to a strategy.

* class `BaseStrategy`: abstract base; concrete strategies implement `determine_shares(ctx)` and `kpi_parameters()`.

* class `StrategyOutput`
        standardized return type of `BaseStrategy.determine_shares(ctx)`.

* class `MomentumStrategy`
        top-k/bottom-k by momentum signal (see §7 of documentation)

* class `FactorsModelStrategy`
        top-k/bottom-k from the factors model predictions (see §8 of documentation).

* class `MeanVarianceOptimizationStrategy`
        constrained MVO (see §9 of documentation).

* class `MVOOptimizerTargets`
        optimizer constraint targets for the MVO strategy.

* class `MVOEstimationSettings`
        configuration for the estimation of the optimer inputs for the MVO strategy.

* class `MVOPrecomputedInputs`
        optional pre-computed optimizer inputs for the MVO strategy )(hen using a fixed rebalancing schedule).

"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
import pandas as pd

from modules import backtest_calendar as bcal
from modules import beta_computation as bc
from modules import covariance_matrix as cm
from modules import factors_engineering as fe
from modules import returns_prediction_model as rpm
from modules import trading_utils as tu
from modules import weights_optimizer as wo



logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RebalanceContext:
    """Per-rebalance frozen inputs handed by the orchestrator to a strategy.

    Strategies may read only a subset of these fields. 

    Fields
    ------
    date
        The rebalance date.
    tickers
        Investable universe at `date`.
    book_equity
        Current equity of the book at `date`.
    prices
        MultiIndex (date, ticker) Series of prices for the
        entire backtest history (e.g. prices at close of each backtest date); 
        strategies that need to compute their own signals dynamically use these prices.
    using_fixed_rebalance_calendar
        Set to True when the backtest is running on a fixed pre-built rebalance schedule 
        (and pre-computed signals can be reused).
    scheduled_rebalance_dates
        The currently-scheduled rebalance calendar (may have been updated 
        by the dynamic rebalance rule; strategies use this only when `using_fixed_rebalance_calendar` is set to False).
    """
    date: pd.Timestamp
    tickers: list[str]
    book_equity: float
    prices: pd.Series
    using_fixed_rebalance_calendar: bool
    scheduled_rebalance_dates: pd.DatetimeIndex


@dataclass
class StrategyOutput:
    """Standardized return type from method `BaseStrategy.determine_shares`.

    Attributes
    ----------
    shares
        Integer-rounded target portfolio for the date.
    optimizer_results
        Populated only when the strategy ran a constrained optimizer (MVO); 
        otherwise set to None.
    """
    shares: pd.Series
    optimizer_results: wo.OptimizerResult | None = None


class BaseStrategy(ABC):
    """Every concrete strategy inherits this abstract class and implements their version 
     of method `determine_shares(ctx)` and `kpi_parameters()`.
    """
    strategy_name: str  # subclasses set this for KPI reporting

    @abstractmethod
    def determine_shares(self, ctx: RebalanceContext) -> StrategyOutput:
        ...

    @abstractmethod
    def kpi_parameters(self) -> dict[str, Any]:
        """Return the strategy's parameters as a dict for KPI reporting."""
        ...


    # ---- Shared helpers:

    @staticmethod
    def _apply_shares_cap_per_ticker(
        shares: pd.Series, max_shares_per_ticker: int
    ) -> pd.Series:
        """Uniform position-size capping:

        If the largest absolute position exceeds the cap, uniformely scale all
        positions down so the largest equals the cap. This preserves relative proportions 
        (and hence Sharpe ratio of the intended portfolio, before integer rounding).
        """
        largest_abs_position = shares.abs().max()
        if largest_abs_position == 0 or largest_abs_position <= max_shares_per_ticker:
            return shares
        shrinking_factor = max_shares_per_ticker / largest_abs_position
        return shares * shrinking_factor


    @staticmethod
    def _build_top_bottom_k_portfolio(
        signal: pd.Series,
        tickers: list[str],
        top_k_bottom_k: int,
        shares_per_ticker: int,
    ) -> pd.Series:
        """Long top-k tickers with positive signal, short bottom-k with
        negative signal, plus or minus `shares_per_ticker` each.

        For example, used by class `MomentumStrategy` and class `FactorsModelStrategy`.
        """
        selectable = signal.index.intersection(tickers)
        signal = signal.loc[selectable]

        tickers_long = signal[signal > 0].dropna().nlargest(top_k_bottom_k).index
        tickers_short = signal[signal < 0].dropna().nsmallest(top_k_bottom_k).index

        ptf = pd.Series(0, index=tickers, dtype=int)
        ptf.loc[tickers_long] = shares_per_ticker
        ptf.loc[tickers_short] = -shares_per_ticker
        return ptf[ptf != 0]


# ---------------------------------------------------------------------------
# Strategy 1:  Momentum top-k / bottom-k
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """Rank tickers by momentum, and go long Q shares for the the top-k tickers with positive momentum (if any),
     and go short Q shares for the bottom-k tickers with negative momentum (if any).
     """

    strategy_name = "top_bottom_k_from_momentum"

    def __init__(
        self,
        top_k_bottom_k: int,
        shares_per_ticker: int,
        rolling_window_trading_days_for_momentum: int,
        buffer_trading_days_for_momentum: int,
        precomputed_momentums: pd.Series | None = None
    ):
        self.top_k_bottom_k = top_k_bottom_k
        self.shares_per_ticker = shares_per_ticker
        self.rolling_window_trading_days_for_momentum = rolling_window_trading_days_for_momentum
        self.buffer_trading_days_for_momentum = buffer_trading_days_for_momentum
        self.precomputed_momentums = precomputed_momentums


    def determine_shares(self, ctx: RebalanceContext) -> StrategyOutput:
        if ctx.using_fixed_rebalance_calendar:
            if self.precomputed_momentums is None:
                raise RuntimeError(
                    "fixed-calendar mode requires precomputed momentums; "
                    "got None"
                )
            signal = self.precomputed_momentums.xs(ctx.date, level=0).squeeze()
        else:
            signal = fe.compute_momentum_at_date_for_tickers(
                ctx.date,
                ctx.tickers,
                ctx.prices,
                self.rolling_window_trading_days_for_momentum,
                self.buffer_trading_days_for_momentum
            )

        raw_shares = self._build_top_bottom_k_portfolio(
            signal, ctx.tickers, self.top_k_bottom_k, self.shares_per_ticker
        )
        return StrategyOutput(shares=tu.truncate_shares_to_int(raw_shares))


    def kpi_parameters(self) -> dict[str, Any]:
        return {
            "top_k_bottom_k": self.top_k_bottom_k,
            "shares_per_ticker": self.shares_per_ticker,
            "rolling_window_trading_days_for_momentum": self.rolling_window_trading_days_for_momentum,
            "buffer_trading_days_for_momentum": self.buffer_trading_days_for_momentum
        }


# ---------------------------------------------------------------------------
# Strategy 2: Cross-sectional factors model
# ---------------------------------------------------------------------------

class FactorsModelStrategy(BaseStrategy):
    """Rank tickers by predicted next-period returns (from a rolling cross-sectional factors model),
      and go long Q shares for the top-k tickers  with positive predicted return (if any), and
      go short Q shares for the bottom-k tickers with negative predicted return (if any).
    """

    strategy_name = "top_bottom_k_from_factors_model"

    def __init__(
        self,
        top_k_bottom_k: int,
        shares_per_ticker: int,
        training_rebalance_dates: pd.DatetimeIndex,
        rebalance_freq_type: str,
        rebalance_frequency_trading_days: int | None,
        last_backtest_date: pd.Timestamp,
        rolling_window_trading_days_for_momentum: int,
        buffer_trading_days_for_momentum: int,
        rolling_window_trading_days_for_volatility: int,
        rolling_periods_for_factors_model_regression: int,
        winsorize_factors_per_date: bool = True,
        z_score_factors_per_date: bool = True,
        precomputed_predictions: pd.Series | None = None,
    ):
        self.top_k_bottom_k = top_k_bottom_k
        self.shares_per_ticker = shares_per_ticker
        self.training_rebalance_dates = training_rebalance_dates
        self.rebalance_freq_type = rebalance_freq_type
        self.rebalance_frequency_trading_days = rebalance_frequency_trading_days
        self.last_backtest_date = last_backtest_date
        self.rolling_window_trading_days_for_momentum = rolling_window_trading_days_for_momentum
        self.buffer_trading_days_for_momentum = buffer_trading_days_for_momentum
        self.rolling_window_trading_days_for_volatility = rolling_window_trading_days_for_volatility
        self.rolling_periods_for_factors_model_regression = rolling_periods_for_factors_model_regression     
        self.winsorize_factors_per_date = winsorize_factors_per_date
        self.z_score_factors_per_date = z_score_factors_per_date
        self.precomputed_predictions = precomputed_predictions


    def _compute_predictions_dynamic(
        self,
        date: pd.Timestamp,
        prices: pd.Series,
        scheduled_rebalance_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute the factors model predictions on the fly at one date.

        Used in dynamic-calendar mode. Builds a local training calendar
        of `rolling_periods_for_factors_model_regression` preceding
        rebalance dates before `date`, fits factors and forward
        returns, runs the regression for `date`, and returns the
        predicted returns indexed by ticker.
        """
        local_training_dates = bcal.build_calendar(
                date,
                self.rebalance_freq_type,
                P = 0,
                P_train = self.rolling_periods_for_factors_model_regression,
                freq = self.rebalance_frequency_trading_days
        ).training_rebalance_dates

        factors = fe.compute_factors_at_training_rebalance_dates(
            prices,
            local_training_dates,
            self.rolling_window_trading_days_for_momentum,
            self.buffer_trading_days_for_momentum,
            self.rolling_window_trading_days_for_volatility,
            None,
            self.winsorize_factors_per_date,
            self.z_score_factors_per_date,
        )

        if date == scheduled_rebalance_dates[-1]:
            next_end = self.last_backtest_date
        else:
            idx = scheduled_rebalance_dates.get_loc(date)
            next_end = scheduled_rebalance_dates[idx + 1]

        fwd_returns = rpm.compute_realized_period_returns(
            prices, local_training_dates, next_end, None
        )
        model_input = pd.concat([factors, fwd_returns], axis=1, join="inner")

        result = rpm.fit_predict_factors_model_at_date(
            date,
            local_training_dates,
            model_input,
            rolling_periods=self.rolling_periods_for_factors_model_regression,
        )
        return result.predictions


    def determine_shares(self, ctx: RebalanceContext) -> StrategyOutput:
        if ctx.using_fixed_rebalance_calendar:
            if self.precomputed_predictions is None:
                raise RuntimeError(
                    "fixed-calendar mode requires precomputed_predictions; "
                    "got None"
                )
            signal = self.precomputed_predictions.xs(ctx.date, level=0)
        else:
            signal = self._compute_predictions_dynamic(
                ctx.date, ctx.prices, ctx.scheduled_rebalance_dates,
            )

        raw_shares = self._build_top_bottom_k_portfolio(
            signal, ctx.tickers, self.top_k_bottom_k, self.shares_per_ticker
        )
        return StrategyOutput(shares=tu.truncate_shares_to_int(raw_shares))


    def kpi_parameters(self) -> dict[str, Any]:
        return {
            "top_k_bottom_k": self.top_k_bottom_k,
            "shares_per_ticker": self.shares_per_ticker,
            "rolling_window_trading_days_for_momentum":
                self.rolling_window_trading_days_for_momentum,
            "buffer_trading_days_for_momentum":
                self.buffer_trading_days_for_momentum,
            "rolling_window_trading_days_for_volatility":
                self.rolling_window_trading_days_for_volatility,
            "rolling_periods_for_factors_model_regression":
                self.rolling_periods_for_factors_model_regression,
        }


# ---------------------------------------------------------------------------
# Strategy 3: Constrained mean-variance portfolio optimization
# ---------------------------------------------------------------------------

@dataclass
class MVOOptimizerTargets:
    """Optimizer constraint targets for class `MeanVarianceOptimizationStrategy`."""
    objective: Literal["max_return", "min_variance"] = "max_return"

    # Common across both objectives
    ptf_beta_cap: float = 0.001
    ptf_net_exposure_cap: float = 0.001
    ptf_gross_exposure_cap: float = 4.0
    max_shares_per_ticker: int = 10_000

    # Required if `objective` set to "max_return"
    ptf_variance_cap: float | None = 0.04

    # Required if `objective` set to "min_variance"
    ptf_return_lower_bound: float | None = None
    min_return_lower_bound_for_feasibility_recovery: float = 0.0


@dataclass
class MVOEstimationSettings:
    """Settings for estimating the optimizer's inputs at a given rebalance date:

    * next-period expected returns for each ticker in the investable universe,
    * the covariance matrix of next-period returns for those tickers,
    * each ticker's beta (as of the rebalance date).
    """
    # Expected returns method
    next_period_expected_returns_estimation_method: Literal[
        "factors_model", "regular_historical_average", "ew_historical_average"
    ] = "factors_model"

    # Factors model parameters (used when `next_period_expected_returns_estimation_method` set to "factors_model")
    rolling_periods_for_factors_model_regression: int = 6
    rolling_window_trading_days_for_momentum: int = 92
    buffer_trading_days_for_momentum: int = 23
    rolling_window_trading_days_for_volatility: int = 60
    winsorize_factors_per_date: bool = True
    z_score_factors_per_date: bool = True

    # Rolling-mean parameters (used when `next_period_expected_returns_estimation_method` set 
    # to either "regular_historical_average" or "ew_historical_average"
    rolling_periods_for_avg_return_computation: int = 12
    avg_type: Literal["regular", "ew"] = "regular"
    halflife_ew: float | None = None

    # Covariance matrix
    rolling_periods_for_cov_matrix_estimation: int = 18

    # Beta regression
    rolling_periods_for_beta_regression: int = 12


@dataclass
class MVOPrecomputedInputs:
    """Optional precomputed inputs for a fixed rebalance schedule grid.

    All fields are set to None when running in dynamic-calendar mode (the
    strategy estimates inputs on the fly at each rebalance date).
    """
    expected_returns_predictions_at_reb_dates: pd.Series | None = None
    cov_matrices_at_reb_dates: dict[pd.Timestamp, pd.DataFrame] | None = None
    tickers_betas_at_reb_dates: pd.Series | None = None


class MeanVarianceOptimizationStrategy(BaseStrategy):
    """Constrained mean-variance portfolio optimization with two supported objective formulations:

        * Maximize portfolio's expected return subject to a variance cap (default), or
        * Minimize portfolio's variance subject to a return lower bound (with feasibility recovery via halving of the lower bound).

      All variants enforce caps on beta neutrality, net exposure, and gross exposure.
    """

    strategy_name = "return_mean_variance_optimization"

    def __init__(
        self,
        targets: MVOOptimizerTargets,
        estimation: MVOEstimationSettings,
        precomputed: MVOPrecomputedInputs,
        # Calendar / data context (needed for dynamic rebalance mode)
        rebalance_freq_type: str,
        rebalance_frequency_trading_days: int | None,
        last_backtest_date: pd.Timestamp,
        # Inputs needed for dynamic-mode beta estimation
        rf_daily_returns: pd.Series,
        market_daily_returns: pd.Series,
    ):
        self.targets = targets
        self.estimation = estimation
        self.precomputed = precomputed
        self.rebalance_freq_type = rebalance_freq_type
        self.rebalance_frequency_trading_days = rebalance_frequency_trading_days
        self.last_backtest_date = last_backtest_date
        self.rf_daily_returns = rf_daily_returns
        self.market_daily_returns = market_daily_returns

    # ---- Estimation of optimizer's inputs (used only in dynamic-rebalance mode):

    def _local_training_calendar(
        self, date: pd.Timestamp, n_periods: int
    ) -> pd.DatetimeIndex:
        """Build a local rolling training calendar of `n_periods` directly preceeding `date`."""
        return bcal.build_calendar(
            date,
            self.rebalance_freq_type,
            P=0,
            P_train=n_periods,
            freq=self.rebalance_frequency_trading_days,
        ).training_rebalance_dates


    def _estimate_mu(
        self,
        date: pd.Timestamp,
        prices: pd.Series,
        scheduled_rebalance_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        method = self.estimation.next_period_expected_returns_estimation_method
        est = self.estimation

        if method == "factors_model":
            local_training_dates = self._local_training_calendar(
                date, est.rolling_periods_for_factors_model_regression
            )
            factors = fe.compute_factors_at_training_rebalance_dates(
                prices,
                local_training_dates,
                est.rolling_window_trading_days_for_momentum,
                est.buffer_trading_days_for_momentum,
                est.rolling_window_trading_days_for_volatility,
                None,
                est.winsorize_factors_per_date,
                est.z_score_factors_per_date
            )
            if date == scheduled_rebalance_dates[-1]:
                next_end = self.last_backtest_date
            else:
                idx = scheduled_rebalance_dates.get_loc(date)
                next_end = scheduled_rebalance_dates[idx + 1]
            fwd_returns = rpm.compute_realized_period_returns(
                prices, local_training_dates, next_end, None
            )
            model_input = pd.concat([factors, fwd_returns], axis=1, join="inner")
            result = rpm.fit_predict_factors_model_at_date(
                date,
                local_training_dates,
                model_input,
                rolling_periods=est.rolling_periods_for_factors_model_regression
            )
            return result.predictions

        if method in ("regular_historical_average", "ew_historical_average"):
            local_training_dates = self._local_training_calendar(
                date, est.rolling_periods_for_avg_return_computation
            )
            avg_at_date = rpm.rolling_mean_period_returns(
                prices,
                local_training_dates,
                pd.DatetimeIndex([date]),
                rolling_number_of_periods=est.rolling_periods_for_avg_return_computation,
                avg_type=est.avg_type,
                halflife_ew=est.halflife_ew,
            )
            return avg_at_date.droplevel("date")

        raise ValueError(
            f"unsupported expected-returns estimation method: {method!r}"
        )


    def _estimate_sigma(
        self, date: pd.Timestamp, prices: pd.Series
    ) -> pd.DataFrame:
        local_training_dates = self._local_training_calendar(
            date, self.estimation.rolling_periods_for_cov_matrix_estimation
        )
        result = cm.compute_rolling_cov_matrices(
            prices,
            local_training_dates,
            pd.DatetimeIndex([date]),
            rolling_periods_for_estimation = self.estimation.rolling_periods_for_cov_matrix_estimation
        )
        if date not in result.matrices:
            raise RuntimeError(f"covariance matrix could not be estimated at {date}")
        return result.matrices[date]


    def _estimate_betas(
        self, date: pd.Timestamp, prices: pd.Series
    ) -> pd.Series:
        local_training_dates = self._local_training_calendar(
            date, self.estimation.rolling_periods_for_beta_regression
        )
        betas = bc.compute_capm_betas(
            self.rf_daily_returns,
            self.market_daily_returns,
            local_training_dates,
            pd.DatetimeIndex([date]),
            prices,
            rolling_periods_for_beta_regression = self.estimation.rolling_periods_for_beta_regression
        )
        return betas.droplevel("date")

    # ---- Core:

    @staticmethod
    def _shares_from_weights(
        date: pd.Timestamp,
        weights: pd.Series,
        book_equity: float,
        prices: pd.Series,
    ) -> pd.Series:
        prices = prices.xs(date, level=0).loc[weights.index]
        return (book_equity * weights).div(prices).rename("shares")


    def _resolve_optimizer_inputs(
        self, ctx: RebalanceContext
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series | None]:
        """Resolve the optimizer's inputs for the rebalance date in `ctx`.
        Looks up precomputed inputs if the backtest uses a fixed rebalance
        schedule, otherwise estimates them on the fly.

         Returns a tuple `(mu, sigma, betas)`:
        * `mu`: estimated next-period expected returns for each ticker in
        the investable universe.
        * `sigma`: estimated covariance matrix of next-period returns for
        those tickers.
        * `betas`: estimated beta for each ticker, or `None` if insufficient data for the estimation.
        """
        if ctx.using_fixed_rebalance_calendar:
            mu = self.precomputed.expected_returns_predictions_at_reb_dates.xs(
                ctx.date, level=0
            )
            sigma = self.precomputed.cov_matrices_at_reb_dates[ctx.date]
            betas = None
            if self.precomputed.tickers_betas_at_reb_dates is not None:
                try:
                    betas = self.precomputed.tickers_betas_at_reb_dates.xs(
                        ctx.date, level=0
                    )
                except KeyError:
                    logger.warning(
                        "no betas at %s; dropping beta-neutrality constraint",
                        ctx.date.strftime("%Y-%m-%d"),
                    )
            return mu, sigma, betas

        # Dynamic rebalance mode
        mu = self._estimate_mu(ctx.date, ctx.prices, ctx.scheduled_rebalance_dates)
        sigma = self._estimate_sigma(ctx.date, ctx.prices)
        try:
            betas = self._estimate_betas(ctx.date, ctx.prices)
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(
                "could not estimate betas at %s (%s); dropping "
                "beta-neutrality constraint",
                ctx.date.strftime("%Y-%m-%d"), e,
            )
            betas = None
        return mu, sigma, betas


    def _solve_optimizer(
        self,
        tickers: list[str],
        mu: pd.Series,
        sigma: pd.DataFrame,
        betas: pd.Series | None,
    ) -> wo.OptimizerResult:
        t = self.targets
        common_kwargs = dict(
            ptf_beta_cap = t.ptf_beta_cap,
            ptf_net_exposure_cap = t.ptf_net_exposure_cap,
            ptf_gross_exposure_cap = t.ptf_gross_exposure_cap,
        )
        if t.objective == "max_return":
            return wo.optimize_weights(
                tickers, mu, sigma, betas,
                objective = "max_return",
                ptf_variance_cap = t.ptf_variance_cap,
                **common_kwargs,
            )
        # min_variance (with feasibility recovery)
        if t.ptf_return_lower_bound is None:
            raise ValueError(
                "MVOOptimizerTargets.ptf_return_lower_bound is required when "
                "objective='min_variance'"
            )
        result, _used_target = wo.optimize_min_variance_with_feasibility_recovery(
            tickers, mu, sigma, betas,
            initial_return_lower_bound = t.ptf_return_lower_bound,
            min_return_lower_bound = t.min_return_lower_bound_for_feasibility_recovery,
            **common_kwargs,
        )
        return result


    def determine_shares(self, ctx: RebalanceContext) -> StrategyOutput:
        mu, sigma, betas = self._resolve_optimizer_inputs(ctx)
        optimizer_result = self._solve_optimizer(ctx.tickers, mu, sigma, betas)

        raw_shares = self._shares_from_weights(
            ctx.date,
            optimizer_result.optimal_weights,
            ctx.book_equity,
            ctx.prices,
        )
        capped = self._apply_shares_cap_per_ticker(
            raw_shares, self.targets.max_shares_per_ticker
        )
        rounded = tu.truncate_shares_to_int(capped)
        return StrategyOutput(shares=rounded, optimizer_results=optimizer_result)


    def kpi_parameters(self) -> dict[str, Any]:
        t, e = self.targets, self.estimation
        params: dict[str, Any] = {
            "objective": t.objective,
            "max_shares_per_ticker_allowed": t.max_shares_per_ticker,
            "ptf_beta_cap": t.ptf_beta_cap,
            "ptf_net_exposure_cap": t.ptf_net_exposure_cap,
            "ptf_gross_exposure_cap": t.ptf_gross_exposure_cap,
            "next_period_expected_returns_estimation_method":
                e.next_period_expected_returns_estimation_method,
            "rolling_periods_for_cov_matrix_estimation":
                e.rolling_periods_for_cov_matrix_estimation,
            "rolling_periods_for_beta_regression":
                e.rolling_periods_for_beta_regression,
        }
        if t.objective == "max_return":
            params["ptf_variance_cap"] = t.ptf_variance_cap
        else:
            params["ptf_return_lower_bound"] = t.ptf_return_lower_bound
        return params