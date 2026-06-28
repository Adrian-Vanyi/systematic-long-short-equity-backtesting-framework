"""Cross-sectional factor model and rolling-average return predictors.

Three model families for predicting next-period returns at each
rebalance date:

1. Linear factor model (cross-sectional): rolling OLS or Ridge
   regression of next-period returns on factor values. Used by
   Strategy 2 (factor-model long/short) and can be used by Strategy 3 (mean-variance
   optimization with the mean of expected returns estimated by the factor-model).

2. Rolling historical mean: simple average of past inter-rebalance
   period returns. Can be used by Strategy 3 (mean-variance
   optimization with the mean of expected returns estimated by this rolling average).

3. Exponentially weighted moving average: down-weights distant
   observations using a half-life parameter. Can be used by Strategy 3 (mean-variance
   optimization with the mean of expected returns estimated by this rolling average).

Public API
----------
* function `compute_realized_period_returns(...)`
        target variable for regression.

* function `fit_predict_factors_model_at_rebalance_dates(...)`
        bulk regressions across all specified rebalance dates.

* function `fit_predict_factors_model_at_date(...)`
        single-date regression.

* class `FactorsModelResult`
        result type

* function `rolling_mean_period_returns(...)`
        rolling average of past-periods' returns.

* function `evaluate_rolling_periods_by_hit_rate(...)`
        evaluate rolling-window lengths by the corresponding average out-of-sample overall sign hit rate
        across test dates (rebalance dates).

* function `plot_diagnostics(...)`
        plot time-series diagnostics for the factors model:  in-sample and out-of-sample metrics, 
        and the evolution of factors loadings.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from modules.factors_engineering import restrict_data_to_rebalance_dates_and_universes


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sign hit rate (out-of-sample directional accuracy)
# ---------------------------------------------------------------------------

def _sign_hit_rates(
    y_pred: pd.Series,
    y_true: pd.Series,
    selection_k: int | None = None,
) -> dict[str, float]:
    """Directional accuracy of predicted vs realized return signs.

    Returns the overall hit rate and, separately, the long-side and
    short-side hit rates, so that asymmetry between the long and short
    books is visible.

    A name is a "hit" when sign(pred) == sign(realized). A realized
    return of exactly 0 counts as a miss for a nonzero prediction.

    Parameters
    ----------
    y_pred, y_true
        Predicted and realized returns at a single rebalance date,
        indexed by ticker.
    selection_k
        If None (default), rates are computed over the full
        cross-section: the long side is every name with pred > 0, the
        short side every name with pred < 0. If an int k, rates are
        computed only over the traded tickers per §8.9: the top-k
        positive-prediction tickers (longs) and the bottom-k
        negative-prediction tickers (shorts).

    Returns
    -------
    dict with keys:
        "sign_hit_rate"        overall, over longs ∪ shorts
        "sign_hit_rate_long"   over the long side only
        "sign_hit_rate_short"  over the short side only
    Each value is in [0, 1], or NaN if its bucket is empty.
    """
    nan = float("nan")
    empty = {"sign_hit_rate": nan,
             "sign_hit_rate_long": nan,
             "sign_hit_rate_short": nan}

    pred, true = y_pred.align(y_true, join="inner")
    if len(pred) == 0:
        return empty
    if selection_k is not None:
        ranked = pred.sort_values()
        long_idx = ranked[ranked > 0].tail(selection_k).index
        short_idx = ranked[ranked < 0].head(selection_k).index
    else:
        long_idx = pred[pred > 0].index
        short_idx = pred[pred < 0].index

    def _rate(idx) -> float:
        if len(idx) == 0:
            return nan
        hits = np.sign(pred.loc[idx].to_numpy()) == np.sign(true.loc[idx].to_numpy())
        return float(hits.mean())

    traded_idx = long_idx.union(short_idx)
    return {
        "sign_hit_rate": _rate(traded_idx),
        "sign_hit_rate_long": _rate(long_idx),
        "sign_hit_rate_short": _rate(short_idx),
    }


# ---------------------------------------------------------------------------
# Realized period returns (regression target variable)
# ---------------------------------------------------------------------------

def compute_realized_period_returns(
    prices: pd.DataFrame,
    training_rebalance_dates: pd.DatetimeIndex,
    final_period_end_date: pd.Timestamp,
    reb_date_to_tickers_universe_dict: dict[pd.Timestamp, list[str]] | None,
) -> pd.Series:
    """Realized next-period returns at each training rebalance date.

    At training rebalance date `t_k`, the value is the simple return from
    `t_k` to `t_{k+1}` (or to `final_period_end_date` for the very last training rebalance date).
    The series, with MultiIndex (date, ticker), is restricted to the universe of tickers at training rebalance date 
    if a universe dict is provided.
    """
    if len(training_rebalance_dates) < 2:
        raise ValueError(
            "need at least two training rebalance dates to compute a "
            "next-period return"
        )
    prices_wide = prices.unstack()
    extended_dates = training_rebalance_dates.append(
        pd.DatetimeIndex([final_period_end_date])
    )
    missing = [
        d.strftime("%Y-%m-%d") for d in extended_dates
        if d not in prices_wide.index
    ]
    if missing:
        raise ValueError(
            f"price data not available for date(s): {missing}"
        )

    prices_wide = prices_wide.reindex(extended_dates)
    fwd_returns_wide = prices_wide.pct_change(fill_method=None).shift(-1)

    return restrict_data_to_rebalance_dates_and_universes(
        fwd_returns_wide,
        training_rebalance_dates,
        "realized_period_returns",
        reb_date_to_tickers_universe_dict
    )


# ---------------------------------------------------------------------------
# Factor model: result types
# ---------------------------------------------------------------------------

@dataclass
class FactorsModelResult:
    """Output of the factors model fit/predict function.

    Attributes
    ----------
    predictions
        For the bulk variant (fit/predict across all rebalance dates): Series indexed by `(date, ticker)` with
        predicted returns for each rebalance date `date`.
        For the single-date variant: Series indexed by ticker.
    coefs
        DataFrame of regression coefficients (excluding intercept) (i.e., each factor's loading):
        one row per rebalance date, columns named after the factors.
    in_sample_metrics
        DataFrame indexed by date with columns "mse", "r2".
    out_sample_metrics
        DataFrame indexed by date with columns "mse", "r2", "spearman_corr", "pearson_corr",  "sign_hit_rate",
        "sign_hit_rate_long", "sign_hit_rate_short".
    condition_numbers
        Series indexed by date, with values the condition number for each design matrix  (see documentation, §8.10).
    """
    predictions: pd.Series
    coefs: pd.DataFrame
    in_sample_metrics: pd.DataFrame
    out_sample_metrics: pd.DataFrame
    condition_numbers: pd.Series


# ---------------------------------------------------------------------------
# Factors model: fit-predict
# ---------------------------------------------------------------------------

def _fit_predict_one_date(
    date: pd.Timestamp,
    training_rebalance_dates: pd.DatetimeIndex,
    model_input: pd.DataFrame,
    rolling_periods: int,
    use_ridge: bool,
    ridge_penalty: float,
    selection_k: int | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict, dict, float]:
    """Fit on the rolling window preceding `date` and predict at `date`."""
    factors = model_input.columns.drop("realized_period_returns")
    idx_date = training_rebalance_dates.get_loc(date)
    training_dates = training_rebalance_dates[idx_date - rolling_periods : idx_date]

    train_data = model_input.loc[pd.IndexSlice[training_dates, :], :]
    X_train = train_data[factors]
    y_train = train_data["realized_period_returns"]

    condition_number = float(np.linalg.cond(X_train.T @ X_train))

    if use_ridge:
        model = Ridge(alpha=ridge_penalty, fit_intercept=True)
    else:
        model = LinearRegression(fit_intercept=True)

    model.fit(X_train, y_train)

    coefs = pd.DataFrame([model.coef_], index=[date], columns=X_train.columns)

    y_train_pred = model.predict(X_train)
    in_sample = {
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "r2": float(r2_score(y_train, y_train_pred)),
    }

    test_data = model_input.xs(date, level=0)
    X_test = test_data[factors]
    y_test = test_data["realized_period_returns"]
    y_test_pred = pd.Series(
        model.predict(X_test), index=X_test.index, name="prediction"
    )

    if len(y_test) < 2:
            out_sample = {"mse": float("nan"), "r2": float("nan"),
                        "spearman_corr": float("nan"), "pearson_corr": float("nan"),
                        "sign_hit_rate": float("nan"),
                        "sign_hit_rate_long": float("nan"),
                        "sign_hit_rate_short": float("nan")}
    else:
        try:
            spearman = float(spearmanr(y_test_pred, y_test).statistic)
        except Exception:
            spearman = float("nan")
        try:
            pearson = float(pearsonr(y_test_pred, y_test).statistic)
        except Exception:
            pearson = float("nan")
        out_sample = {
            "mse": float(mean_squared_error(y_test, y_test_pred)),
            "r2": float(r2_score(y_test, y_test_pred)),
            "spearman_corr": spearman,
            "pearson_corr": pearson,
            **_sign_hit_rates(y_test_pred, y_test, selection_k),
        }

    return y_test_pred, coefs, in_sample, out_sample, condition_number


def fit_predict_factors_model_at_date(
    date: pd.Timestamp,
    training_rebalance_dates: pd.DatetimeIndex,
    model_input: pd.DataFrame,
    rolling_periods: int = 1,
    use_ridge: bool = False,
    ridge_penalty: float = 0.01,
    selection_k: int | None = None
) -> FactorsModelResult:
    """Fit the cross-sectional factors model at one rebalance date.

    Fits on the `rolling_periods` rebalance dates immediately preceding
    `date` (excluding `date` itself), then predicts at `date`.

    Parameters
    ----------
    date
        Must be in `training_rebalance_dates` and have at least `rolling_periods` predecessors.
    training_rebalance_dates
        Full grid (training + actual rebalance dates).
    model_input
        DataFrame indexed by (date, ticker) with one column per factor plus 
        a "realized_period_returns"  column (the regression target).
    rolling_periods
        Window length for the rolling regression.
    use_ridge
        Use Ridge instead of OLS.
    ridge_penalty
        L2 penalty for Ridge; ignored if `use_ridge=False`.

    Returns
    -------
    FactorsModelResult
        With single-date results: `predictions` is a Series indexed by ticker; 
        `coefs` has one row; metrics are dicts unwrapped into single-row DataFrames.
    """
    if date not in training_rebalance_dates:
        raise ValueError(f"date {date} is not in training_rebalance_dates")

    idx_date = training_rebalance_dates.get_loc(date)
    if idx_date < rolling_periods:
        raise ValueError(
            f"date {date} is at position {idx_date} in training_rebalance_dates; "
            f"need at least {rolling_periods} prior dates"
        )
    
    pred, coefs, in_sample, out_sample, cond = _fit_predict_one_date(
            date, training_rebalance_dates, model_input,
            rolling_periods, use_ridge, ridge_penalty,
            selection_k=selection_k,
    )

    return FactorsModelResult(
        predictions = pred,
        coefs = coefs,
        in_sample_metrics = pd.DataFrame([in_sample], index=[date]),
        out_sample_metrics = pd.DataFrame([out_sample], index=[date]),
        condition_numbers = pd.Series({date: cond}, name="condition_number")
    )


def fit_predict_factors_model_at_rebalance_dates(
    rebalance_dates: pd.DatetimeIndex,
    training_rebalance_dates: pd.DatetimeIndex,
    model_input: pd.DataFrame,
    rolling_periods: int = 1,
    use_ridge: bool = False,
    ridge_penalty: float = 0.01,
    selection_k: int | None = None
) -> FactorsModelResult:
    """Fit the factors model at every rebalance date in `rebalance_dates`.

    See function `fit_predict_factors_model_at_date` for parameters.
    """
    not_in_training = [d for d in rebalance_dates if d not in training_rebalance_dates]
    if not_in_training:
        raise ValueError(
            f"rebalance_dates must be a subset of training_rebalance_dates; "
            f"missing: {not_in_training}"
        )

    earliest = rebalance_dates[0]
    idx_earliest = training_rebalance_dates.get_loc(earliest)
    if idx_earliest < rolling_periods:
        raise ValueError(
            f"earliest rebalance date {earliest} is at position {idx_earliest} "
            f"in training_rebalance_dates; need at least {rolling_periods} prior dates"
        )

    preds, coefs_list = [], []
    in_sample_per_date: dict[pd.Timestamp, dict] = {}
    out_sample_per_date: dict[pd.Timestamp, dict] = {}
    cond_per_date: dict[pd.Timestamp, float] = {}

    for t in rebalance_dates:
        p, c, isamp, osamp, cn = _fit_predict_one_date(
                    t, training_rebalance_dates, model_input,
                    rolling_periods, use_ridge, ridge_penalty,
                    selection_k=selection_k
                )
        preds.append(p)
        coefs_list.append(c)
        in_sample_per_date[t] = isamp
        out_sample_per_date[t] = osamp
        cond_per_date[t] = cn

    predictions = (
        pd.concat(preds, keys=rebalance_dates, names=["date", "ticker"]).sort_index()
    )
    return FactorsModelResult(
        predictions = predictions,
        coefs = pd.concat(coefs_list),
        in_sample_metrics = pd.DataFrame.from_dict(in_sample_per_date, orient="index"),
        out_sample_metrics = pd.DataFrame.from_dict(out_sample_per_date, orient="index"),
        condition_numbers = pd.Series(cond_per_date, name="condition_number")
    )


# ---------------------------------------------------------------------------
# Hyperparameter optimization 
# ---------------------------------------------------------------------------

def evaluate_rolling_periods_by_hit_rate(
    training_rebalance_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    model_input: pd.DataFrame,
    rolling_periods_grid: list[int],
    selection_k: int,
    use_ridge: bool = False,
    ridge_penalty: float = 0.01,
    plot: bool = False,
) -> pd.Series:
    """For each candidate `rolling_periods`, return the average out-of-sample
    overall sign hit rate on the traded tickers across all rebalance dates.

    Used to inform the selection of the rolling window: we seek the window
    maximizing the directional accuracy of the factors model on the tickers it
    actually trades (top-`selection_k` / bottom-`selection_k`), which is the
    primary profitability proxy (see documentation, §8.11).

    Parameters
    ----------
    selection_k
        Number of tickers per side (top-k long, bottom-k short) over which the
        sign hit rate is evaluated, matching the §8.9 selection rule. The
        hit rate is computed on the traded subset, so its no-skill baseline
        is 0.5.
    plot : bool, default False
        If True, plot the average out-of-sample hit rate against the number
        of rolling periods used.
    """
    avg_hit_rate = {}

    for rp in rolling_periods_grid:
        result = fit_predict_factors_model_at_rebalance_dates(
                rebalance_dates,
                training_rebalance_dates,
                model_input,
                rolling_periods = rp,
                selection_k = selection_k,
                use_ridge = use_ridge,
                ridge_penalty = ridge_penalty,
        )

        avg_hit_rate[rp] = float(
            result.out_sample_metrics["sign_hit_rate"].mean()
        )

    hit_rate_by_number_of_rolling_periods_used = pd.Series(
        avg_hit_rate,
        name="avg_sign_hit_rate",
    )

    if plot:
        ax = hit_rate_by_number_of_rolling_periods_used.plot(marker="o")
        ax.axhline(0.5, linestyle="--", color="black")  # no-skill baseline
        ax.set_xlabel("W")
        ax.set_ylabel("avg out-sample sign hit rate (for traded tickers) across reb dates")
        plt.tight_layout()

    return hit_rate_by_number_of_rolling_periods_used


# ---------------------------------------------------------------------------
# Rolling-average return predictors
# ---------------------------------------------------------------------------

def _ew_weights(n: int, alpha: float) -> np.ndarray:
    """Weights for windowed EWMA: most recent period has weight 1
    before normalization; weights then normalize to sum to 1."""
    w = (1 - alpha) ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def rolling_mean_period_returns(
    prices: pd.Series,
    training_rebalance_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    rolling_number_of_periods: int,
    avg_type: Literal["regular", "ew"] = "regular",
    halflife_ew: float | None = None
) -> pd.Series:
    """Rolling average of past inter-rebalance period returns at each
    date in `rebalance_dates`.

    For each rebalance date t_k, the average is over the previous
    n = `rolling_number_of_periods` realized returns ending at t_k
    inclusive. These n returns consume prices at the n+1 dates
    (t_{k-n}, t_{k-n+1}, ..., t_{k-1}, t_k), all of which must be present in
    `training_rebalance_dates`.

    Parameters
    ----------
    prices
        Series of close prices indexed by (date, ticker).
    training_rebalance_dates
        Full grid (must contain enough predecessors of each rebalance
        date to compute `rolling_number_of_periods` returns).
    rebalance_dates
        Dates at which the rolling average is reported.
    rolling_number_of_periods
        Window length.
    avg_type
        "regular` for equally-weighted mean, "ew" for
        exponentially weighted mean over the same finite window.
    halflife_ew
        Half-life in inter-rebalance periods (must be > 0). Required
        when `avg_type`="ew". 

    Returns
    -------
    pd.Series
        Indexed by (`date`, `ticker`). Rows for which any of the n+1 required prices 
        at `date` for `ticker` is missing are dropped.
    """
    if avg_type not in ("regular", "ew"):
        raise ValueError(f"unsupported avg_type: {avg_type!r}")
    if avg_type == "ew" and (halflife_ew is None or halflife_ew <= 0):
        raise ValueError("avg_type='ew' requires halflife_ew > 0")

    n = rolling_number_of_periods
    prices_wide = (
        prices.unstack("ticker")
        .sort_index()
        .reindex(training_rebalance_dates)
    )
    returns_wide = prices_wide.pct_change(fill_method=None)

    rolling = returns_wide.rolling(n, min_periods=n)
    if avg_type == "regular":
        avg_wide = rolling.mean()
    else:
        alpha = 1 - 2 ** (-1 / halflife_ew)
        weights = _ew_weights(n, alpha)
        avg_wide = rolling.apply(lambda x: np.dot(x, weights), raw=True)

    result = (
        avg_wide.reindex(rebalance_dates)
        .stack(future_stack=True)
        .dropna()
    )
    result.name = f"rolling_{avg_type}_mean_period_returns"
    result.index = result.index.set_names("date", level=0)
    return result



# ---------------------------------------------------------------------------
# Plotting diagnostics
# ---------------------------------------------------------------------------

def plot_diagnostics(in_sample_metrics, out_sample_metrics, coefs,
                    plot_coefs=True, start_date=None, end_date=None,
                    show_markers=False):
    def _filter(df):
        if start_date is not None:
            df = df.loc[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df.loc[df.index <= pd.Timestamp(end_date)]
        return df

    in_sample_metrics = _filter(in_sample_metrics)
    out_sample_metrics = _filter(out_sample_metrics)
    coefs = _filter(coefs)

    style = "o-" if show_markers else "-"

    fig, axs = plt.subplots(4, 2, figsize=(14, 17), sharex=True)

    axs[0, 0].plot(in_sample_metrics.index, in_sample_metrics["mse"], style)
    axs[0, 0].set_title("Train MSE")
    axs[0, 1].plot(in_sample_metrics.index, in_sample_metrics["r2"], style)
    axs[0, 1].set_title("Train R²")

    test_metric_map = [
        (1, 0, "mse", "Test MSE"),
        (1, 1, "r2", "Test R²"),
        (2, 0, "spearman_corr", "Test Spearman IC"),
        (2, 1, "pearson_corr", "Test Pearson IC"),
    ]
    for row, col, key, title in test_metric_map:
        ax = axs[row, col]
        ax.plot(out_sample_metrics.index, out_sample_metrics[key], style)
        ax.set_title(title)
        if "IC" in title:
            ax.axhline(0, linestyle="--", color="black")

    ax = axs[3, 0]
    ax.plot(out_sample_metrics.index, out_sample_metrics["sign_hit_rate"], style)
    ax.set_title("Test sign hit rate (overall)")
    ax.axhline(0.5, linestyle="--", color="black")  # no-skill baseline
    ax.set_ylim(0, 1)

    ax = axs[3, 1]
    ax.plot(out_sample_metrics.index, out_sample_metrics["sign_hit_rate_long"],
            style, label="long side")
    ax.plot(out_sample_metrics.index, out_sample_metrics["sign_hit_rate_short"],
            style, label="short side")
    ax.set_title("Test sign hit rate (long vs short)")
    ax.axhline(0.5, linestyle="--", color="black")
    ax.set_ylim(0, 1)
    ax.legend()

    for ax in axs[-1, :]:
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelbottom=True)

    fig.suptitle("Model Fit (Training) and Prediction Accuracy (Out-of-Sample)")
    fig.tight_layout()

    if plot_coefs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(coefs.index, coefs, style, label=coefs.columns)
        ax.set_title("Factors loadings")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(0, linestyle="--", color="black")
        ax.legend()
