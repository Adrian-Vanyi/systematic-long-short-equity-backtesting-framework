# A Broker-Realistic Backtesting Framework for Systematic Long/Short Equity Strategies

A daily-frequency backtesting framework for systematic long/short equity strategies on the historical S&P 500 (US equities). It is built around a realistic brokerage ledger: financing (debit interest, cash interest, stock-borrow fees, short rebate), dividends on both legs, Reg-T / FINRA margin-requirement mechanics, trading costs, and per-ticker FIFO short-proceeds lots. On top of this I implemented point-in-time construction of the investable universe, a pluggable and extensible portfolio construction strategy interface, and a detailed diagnostics/KPI layer that, among other things, decomposes each day's equity P&L into its price, dividend, financing, commission and execution-cost streams, and checks that decomposition as an identity on every backtest date. 

My goal was the framework itself, a foundation for later work on additional strategies. The three strategies I currently implement (momentum, a factors model, MVO) are well-known baselines; they serve to test the framework and to be measured by it, through three performance comparison methodologies: target-hit statistics, Sharpe ratios, and an ex-post market-exposure regression that separates the part of a strategy's realised return explained by exposure to the market from the part that is not.

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

The second half of that question is where many backtests are unrealistic: A strategy's gross price P&L is not what reaches our account. Financing on the debit balance, borrow fees on the short leg, dividends owed and received, margin requirements that force deleveraging at the worst moment, spread and slippage on every rebalance: these are part of the strategy's return, and a book that ignores them can show a profit where the account would have shown a loss.

So I built the ledger first and the strategies second. 


## Data

Two data sources feed the framework. Price (OHLCV) and dividend data  come from the `yfinance` package, which pulls from Yahoo Finance. Historical S&P 500 membership is reconstructed point-in-time by parsing Wikipedia's revision history, so the universe on any rebalance date reflects the index as it stood then, not as it stands today. 

## Documentation

**The full design of the framework and the derivations of the strategies are in the accompanying [`backtesting_framework_documentation.pdf`](./backtesting_framework_documentation.pdf).** It covers the modelling of the brokerage-book mechanics and margin requirements; the construction of the backtest calendar; backtest termination; the strategies’ portfolio-rebalancing rules; the construction of the investable universe for each rebalance date; the derivation of each strategy; the KPIs and daily P&L decomposition; and a complete single-backtest example. It also presents the cross-strategy performance comparison methodology and results, together with an appendix justifying the filter applied to the price feed to remove bad ticker data.


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

1. The dataset used for the documented results is already included (`data/`), so you can skip to 2. Run `download_data.py` only if you want to refresh the data up to the present.
2. Open `notebooks/single_backtest_runner.ipynb`, set the configuration (strategy, calendar, margin and cost parameters, termination rules), and run it end to end to produce the diagnostic plots, book-value table, audit logs, and KPI report.
3. Open `notebooks/strategy_comparison.ipynb` to run the cross-strategy comparison sweeps.

A complete single-backtest example, with every output explained, is given in the documentation (§13).


## Results at a glance

These are some of the framework's results:

**The ledger reconciles:** Every backtest decomposes each day's equity P&L into
price, dividend, financing, commission and execution-cost streams and checks the
decomposition as an identity on every date. On the documented run (§13.1) the
largest daily residual is 1.5e-10 against a 1e-6 tolerance.

**Costs are explicit:** On that run, a gross price P&L of +12.08% of initial
equity nets to +9.39% after costs, the largest stream being execution costs at
1.66% of initial equity (§12.8).

**The diagnostics separate constrained from unconstrained books:** Across the
twelve strategy and window combinations, the two MVO strategies (using beta and net-exposure caps at rebalance)
realise |beta| <= 0.272 and average net exposure within [-11.7%, +3.2%], while
momentum and the factors model (both unconstrained) reach a maximum beta of +1.590 and average net exposures of
81.9% and 43.5% (§14.3.7). The caveat about the impact that the beta constraint at rebalance has on realized beta is addressed in §14.3.7.

**The strategies mostly do not perform:** Across 24 combinations of strategy/historical-window/
rebalancing-rule, the highest annualized Sharpe is 0.22 and 18 are
negative. Annualized alpha against the market is negative in 19 of 24
runs (median -11.8%); the single run rejecting the hypothesis alpha = 0 at the level 5% is a loss of
-23.1%. However, the Sharpe figures here are cross-strategy comparators, not
forward-looking estimates (§14.1, §14.1.6, §14.3.7).

**Target-hit statistics** (§14.2), over 12 quarterly start dates from 2021-01 to
2023-10, one-year runs, and a 50% stop-loss: at a 7% annual return target, momentum hit
in 10 of 12 start dates (83.3%), averaging 71.8 trading days to target, against
50.0% for the factors model. Min-variance MVO was the only one of the four
strategy variants with no stop-loss termination anywhere in the 288 backtests run.
Twelve start dates is a demonstration of the methodology, not a statistically
meaningful sample (§14.2.1), and strong comparison metrics are "weak
positive evidence": reasonably necessary for deployment, but not sufficient to remove their risk (§15).

## Limitations and possible extensions

The framework is a research tool, and the documentation presents its limitations as well as some of its natural extensions.

Among the limitations (see §16.1):

- **Daily frequency only:** trades execute at the close price (used as the pre-trade mid proxy); there is no intraday execution.
- **Static trading-cost assumptions:** half-spread, slippage and the per-share commission are parameter inputs applied uniformly across dates and stocks,
overridable per (date, ticker) for stress-testing. The half-spread and slippage are expressed in basis points of the pre-trade mid and do not scale
with trade size, which becomes a relevant issue for large orders. A size-dependent extension based on the square-root market-impact law is discussed in §2.5.2.
- **Partial survivorship-bias mitigation:** I require full forward price data over each inter-rebalance window (e.g. one month) for accurate mark-to-market, which can drop stocks that are delisted or acquired during that period.
- **No order-size constraints:** trade size is not checked against average daily volume, so a position representing a few percent of ADV still fills at the execution price modelled in §2.5.2. Nevertheless, the KPI report flags the worst case via `max_pct_of_daily_market_volume_traded_for_a_ticker_during_backtest`.
- **Limited corporate-action handling:** stock dividends are accounted for, but splits, spin-offs, mergers, ticker changes and special dividends are handled only insofar as our price feed series accommodates them.
- **No short-availability or recall modelling:** opening a short requires borrowing the underlying shares, which I assume are always available; broker recalls, i.e. forced buy-ins, are not modelled.
- **No tax accounting.**

Among the possible extensions (see §16.2):

- **Improving the expected-return input for the MVO strategy:** e.g. a Black–Litterman approach for a Bayesian combination of views (§16.2 explains why the expected-return vector, rather than the covariance matrix, is the highest-value target).
- **Multi-factor neutrality for the constrained MVO strategy:** extending the single-factor beta constraint to a set of known risk factors.
- **Walk-forward hyperparameter optimisation:** estimation is already walk-forward; the configuration is not.
- **Additional strategies:** the strategy interface is a pluggable part.
- **An additional risk-management overlay.**
- **Live-trading integration:** developing the connection to a broker's account via API.
