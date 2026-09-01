# A Broker-Realistic Backtesting Framework for Systematic Long/Short Equity Strategies

A daily-frequency backtesting framework for systematic long/short equity strategies on the historical S&P 500 (US equities). It is built around a realistic brokerage ledger: financing (debit interest, cash interest, stock-borrow fees, short rebate), dividends on the long and short legs, Reg-T / FINRA margin-requirement mechanics, trading costs, and per-ticker FIFO short-proceeds lots. On top of this I implemented point-in-time construction of the investable universe, a pluggable and extensible portfolio construction strategy interface, and a detailed diagnostics/KPI layer that, among other things, decomposes each day's equity P&L into its price, dividend, financing, commission and execution-cost streams, and checks that decomposition as an identity on every backtest date. 

My goal was the framework itself, a foundation for later work on additional strategies. The three strategy families I currently implement (momentum, a predictive factors model, and mean-variance portfolio optimization in two variants) are well-known baselines; they serve to test the framework and to be measured by it, through three performance comparison methodologies: target-hit statistics, Sharpe ratios, and an ex-post market-exposure regression that separates the part of a strategy's realized return explained by exposure to the market from the part that is not.

## Contents

- [Motivation](#motivation)
- [Data](#data)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Results at a glance](#results-at-a-glance)
- [Limitations and possible extensions](#limitations-and-possible-extensions)

## Motivation

Given a certain amount of capital, how could we invest it to generate excess returns? Concretely: what portfolio construction strategy could we deploy in the equities market, and what would its performance have been had we deployed it over some period in the past?

The second half of that question is where many backtests are unrealistic: a strategy's price-only P&L is not what reaches our account. Financing runs on four streams at once: debit interest paid on the margin loan funding the long leg, interest earned on credit cash, borrow fees paid on the short leg, and the rebate received on the short-sale proceeds. Dividends are credited on longs and debited on shorts. The margin requirement is a binding constraint: Reg-T's initial margin requirement constrains the size of positions the book can open, and FINRA maintenance requirements can trigger a margin deficiency requiring us to post collateral or deleverage our positions. Also, every trade fills at an execution price incorporating a half-spread and slippage difference relative to the pre-trade mid-price. These brokerage mechanics impact a strategy's return, and a backtest that ignores them can show a profit where the account would have shown a loss.

So I built the ledger first and the strategies second. 


## Data

Two data sources feed the framework. Price (OHLCV) and dividend data  come from the `yfinance` package, which pulls from Yahoo Finance. Historical S&P 500 membership is reconstructed point-in-time by parsing Wikipedia's revision history, so the universe on any rebalance date reflects the index as it stood then, not as it stands today. 

## Documentation

**The full design of the framework and the derivations of the strategies are in the accompanying [`backtesting_framework_documentation.pdf`](./backtesting_framework_documentation.pdf).** It covers the modelling of the brokerage-book mechanics and margin requirements; the construction of the backtest calendar; backtest termination rules; the strategies’ portfolio-rebalancing rules; the construction of the investable universe for each rebalance date; the derivation of each strategy; the KPIs and daily P&L decomposition; and a complete single-backtest example. It also presents the cross-strategy performance comparison methodology and results, and an appendix explaining the filter applied to the price feed to flag and remove corrupted ticker data.


## Repository layout

| Path | Purpose |
| --- | --- |
| `backtesting_framework_documentation.pdf` | Documentation |
| `modules/` | Python modules with the core framework implementation |
| `data/` | Pre-downloaded price and index-membership data used to generate the documented results |
| `download_data.py` | Standalone utility to download price and index-membership data up to the present time |
| `outputs/` | Default destination for saved backtest files: audit logs, KPI report, performance comparison tables, persisted curves |
| `notebooks/single_backtest_runner.ipynb` | Runs one configurable backtest end to end |
| `notebooks/strategy_comparison.ipynb` | Compares strategies across parameter settings over many runs |
| `notebooks/mvo_solver.ipynb` | Tests and validates the portfolio optimizer used by the MVO strategy |
| `notebooks/factors_model.ipynb` | Tests and validates the factors model implementation used by the corresponding strategy |
| `notebooks/backtest_calendar.ipynb` | Visualises the construction of the backtest calendar |
| `notebooks/sp500_historical_members.ipynb` | Tests retrieval of historical S&P 500 members from Wikipedia revision history |
| `requirements.txt` | Required packages to run the framework |
| `LICENSE` | MIT license |


## Installation

Requires **Python 3.13**.

```bash
# 1. In a new directory, clone the repo
git clone https://github.com/Adrian-Vanyi/systematic-long-short-equity-backtesting-framework.git

# 2. Navigate to the repo's local directory, and create a virtual environment
python -m venv venv

# 3. Activate it
#   Windows (cmd):   venv\Scripts\activate
#   macOS / Linux:   source venv/bin/activate

# 4. Install dependencies (pinned in requirements.txt; includes JupyterLab)
pip install -r requirements.txt

# 5. Register a Jupyter kernel bound to the virtual environment's interpreter
python -m ipykernel install --user \
  --name=long_short_backtester_kernel \
  --display-name "Python (LongShortEquityBacktester)"
```

Then launch JupyterLab (`jupyter lab`) and select the **Python (LongShortEquityBacktester)** kernel in any notebook.


## Quick start

1. The datasets used for the documented results are already included (`data/`), so you can skip to 2. Run `download_data.py` only if you want to refresh the data up to the present.
2. Open `notebooks/single_backtest_runner.ipynb`, set the configuration (strategy, calendar, margin and cost parameters, termination rules), and run it end to end to produce the diagnostic plots, book-value table, audit logs, and KPI report.
3. Open `notebooks/strategy_comparison.ipynb` to run the cross-strategy comparison sweeps.

A complete single-backtest example, with every output explained, is given in the documentation (§13).


## Results at a glance

Sharpe ratios and market-exposure results are reported across 24 combinations of strategy
variant, historical window, and rebalancing rule, spanning the four strategies:
momentum, a predictive factors model, and two portfolio MVO variants (max-return and min-variance). The three
historical windows tested are 2010-03 to 2020-01, 2010-03 to 2026-02, and 2022-10 to 2026-01, so
the set spans both a long sample and a recent one. The target-hit study (§14.2) varies the return target and start date while holding the rest of the backtest configuration fixed, across 288 backtests in total.


- **The ledger reconciles:** Every backtest decomposes each day's equity P&L into price,
  dividend, financing, trading-fee and execution-cost streams, while separately reconciling posted collateral under the margin-cure variant, and checks the equity P&L decomposition is exact on every date.
  For example, on the documented run (the predictive factors model strategy over 2023, monthly rebalancing,
  $100k initial equity; §13.1), the largest daily residual is 1.5e-10 against a 1e-6
  tolerance, with no date above tolerance.

- **Costs are explicit:** On that run, a price-only P&L of +12.08% of initial equity nets
  to +9.39% once every other stream is accounted for. The breakdown as a percentage of
  initial equity is: execution costs -1.66%, dividends -0.92%, financing -0.08%, trading fees
  -0.03% (§12.8).

- **The margin requirement constrains the portfolio:** On that run, 44.4% of rebalances
  produced a target portfolio that failed the rebalance margin requirement and had to be shrunk, retaining between 22% and 98% of the target position sizes (mean 61.4%). No maintenance-margin violation occurred on any backtest date under the implemented margin model; consequently, no margin cure (deleveraging or posting collateral) was required (§13.7).

- **The diagnostics separate constrained from unconstrained strategies:** The two MVO strategies apply beta and net-exposure caps at each rebalance. Across all runs, their realized rolling $|\beta|$ remains at or below 0.272, with mean $|\beta|$ of 0.15 (for the max-return variant) and 0.09 (for the min-variance variant). Their average net exposure ranges from −11.7% to +3.2%. By contrast, the strategies without explicit beta and net-exposure constraints realized larger beta and net exposures: the predictive factors model reaches a $|\beta|$ of 0.516 (mean 0.25) with average net exposure between 21.0% and 43.5%, and the momentum strategy reaches a $|\beta|$ of 1.590 (mean 0.52) with average net exposure between 18.1% and 81.9%. Importantly, these results show that applying the rebalance constraints results in lower magnitudes of realized rolling beta and net exposure without forcing either to zero (§14.3.7).

- **Sharpe ratios are weak across strategies:** The maximum annualized Sharpe ratio achieved across the 24 combinations is 0.22, and 18 of the 24 are negative. These results are read as a descriptive cross-strategy comparison of historical realized risk-adjusted performance, not as forward-looking estimates of risk-adjusted performance (§14.1).

- **The momentum and max-return MVO strategies hit a specific return target most often:** Across 12 quarterly start dates from 2021-01 to 2023-10, using one-year runs with a 50% stop-loss, momentum and max-return MVO each reached the 7% return target in 10 of 12 start dates (83.3%), compared with 6 of 12 (50.0%) for the predictive factors model. Max-return MVO reached the target faster, averaging 30.5 trading days versus 71.8 for momentum. The min-variance MVO strategy was the only strategy with no stop-loss termination in any of the 288 backtests (§14.2).

* **The target-hit metrics illustrate the methodology rather than establish an edge:** I only tested twelve start dates, which is not a statistically meaningful sample (§14.2.1). The Sharpe-ratio analysis and target-hit analysis answer different questions. For the Sharpe analysis, the stop-loss and return target are disabled so that each strategy can generate an uninterrupted daily-return series over the specified historical window. The resulting Sharpe ratio depends on the historical window used, but, for a fixed window, is invariant to the ordering of those daily returns. The target-hit analysis instead varies the start date. Changing the start date can change the information set used to construct the initial portfolio and therefore its composition; this can in turn change the subsequent sequence of holdings and returns. Under the dynamic-rebalancing configuration, the realized equity path can additionally trigger a rebalance ahead of the scheduled date, further changing the subsequent portfolio composition and corresponding return path. Target-hit statistics therefore reflect aspects of start-date dependence and path dependence that are not reflected by a Sharpe ratio computed over a fixed historical window. Neither metric, however, should be used on its own to justify a deployment decision today; we must also understand the sources of risk in each strategy (§10, §15).


- **The estimated market alphas of the strategies are largely inconclusive:** Using heteroskedasticity-and-autocorrelation-consistent standard errors, 23 of the 24 alpha estimates, measured relative to the market alone rather than to a broader set of established risk factors, are not statistically distinguishable from zero at the 5% significance level. 



## Limitations and possible extensions

The framework is a research tool, and the documentation presents its limitations as well as some of its natural extensions.

Among the limitations (§16.1):

- **Daily frequency only:** trades execute at the close price (used as the pre-trade mid proxy); there is no intraday execution.
- **Static trading-cost assumptions:** half-spread, slippage and the per-share commission are parameter inputs applied uniformly across dates and stocks,
overridable per (date, ticker) for stress-testing. The half-spread and slippage are expressed in basis points of the pre-trade mid and do not scale
with trade size, which becomes a relevant issue for large orders. A size-dependent extension based on the square-root market-impact law is discussed in §2.5.2.
- ****Partial survivorship bias mitigation:**** the backtest uses point-in-time historical S&P 500 constituents. However, the free price feed (`yfinance`) may lack historical price data for stocks that were later delisted, acquired, or merged. Because daily prices are required to mark positions to market throughout their holding period, constituents for which the necessary subsequent price data are unavailable are currently excluded from the investable universe, which introduces a survivorship/selection bias. A more realistic implementation would use a price feed with comprehensive historical coverage and handle a stock's delisting or other terminal corporate actions when they occur.
- **No ADV-based order-size constraints:** the strategies impose position-size constraints (a share-count target for the momentum and predictive factors model strategies and a maximum share cap for the MVO strategies), but trade size is not constrained, rejected, or resized based on average daily volume (ADV). Consequently, a trade can represent a substantial fraction of a stock's ADV while still being executed at the price specified by the execution model in §2.5.2. The KPI report nevertheless measures and flags the maximum percentage of daily market volume represented by any ticker's daily traded volume via `max_pct_of_daily_market_volume_traded_for_a_ticker_during_backtest`.
- **Limited corporate-action handling:** cash dividends are explicitly accounted for, and stock splits are reflected in the historical price data provided by `yfinance`. Other corporate actions (including spin-offs and mergers) are not modelled.
- **No short-availability or recall modelling:** opening a short requires borrowing the underlying shares, which I assume are always available; broker recalls, i.e. forced buy-ins, are not modelled.
- **No tax accounting.**

Among the possible extensions (§16.2):

- **Improving the expected-return input for the MVO strategy:** e.g. a Black–Litterman approach for a Bayesian combination of views (§16.2 explains why improving the expected-return vector may offer greater value than further refinement of the covariance matrix).
- **Multi-factor neutrality for the constrained MVO strategy:** extending the single-factor beta constraint to a set of known risk factors.
- **Walk-forward hyperparameter optimization of a strategy:** estimation is already walk-forward; the configuration is not.
- **Additional strategies:** the strategy interface is a pluggable part.
- **Implementing a size-dependent trading execution cost model, to make a strategy's capacity measurable.**
- **An additional risk-management overlay:** e.g. sector caps; single-name caps; inverse-volatility sizing.
- **Live-trading integration:** developing the connection to a broker's account via API.
