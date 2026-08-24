# A Broker-Realistic Backtesting Framework for Systematic Long/Short Equity Strategies

A daily-frequency backtesting framework for systematic long/short equity strategies on the historical S&P 500 (US equities). It is built around a realistic brokerage ledger: financing (debit interest, cash interest, stock-borrow fees, short rebate), dividends on both legs, Reg-T / FINRA margin-requirement mechanics, trading costs, and per-ticker FIFO short-proceeds lots. On top of this we implemented point-in-time construction of the investable universe, a pluggable and extensible strategy interface, and a detailed diagnostics/KPI layer that, among other things, decomposes each day's equity P&L into its price, dividend, financing, commission and execution-cost streams, and checks that decomposition as an identity on every backtest date.

The goal was the framework itself (a foundation for later work on alpha discovery) not the strategies. The three strategies currently implemented (momentum, a factors model, MVO) are well known; they serve to test the framework and be measured by it, through three performance comparison methodologies: target-hit statistics, Sharpe ratios, and an ex-post market-exposure regression that separates the part of a strategy's realised return explained by exposure to the market from the part that is not.

## Contents

- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Results at a glance](#results-at-a-glance)
- [Limitations and possible extensions](#limitations)



## Documentation

**The full design of the framework and the derivations of the strategies are in the accompanying [`backtesting_framework_documentation.pdf`](./backtesting_framework_documentation.pdf).** It covers brokerage-book mechanics, margin requirements, backtest-calendar construction, backtest termination and the strategies' portfolio rebalancing rules, construction of the universe of investable tickers per rebalance date, the derivation of each strategy, the KPIs and the daily P&L decomposition, a complete single-backtest example, the cross-strategy performance comparison methodology and results, and an appendix justifying the filter we apply to our price feed to remove bad ticker data.


## Repository layout

| Path | Purpose |
| --- | --- |
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

## Quick start

1. The dataset used for the documented results is already included (`data/`), so you can skip to 2. Run `download_data.py` only if you want to refresh the data up to the present.
2. Open `notebooks/single_backtest_runner.ipynb`, set the configuration (strategy, calendar, margin and cost parameters, termination rules), and run it end to end to produce the diagnostic plots, book-value table, audit logs, and KPI report.
3. Open `notebooks/strategy_comparison.ipynb` to run the cross-strategy comparison sweeps.

A complete single-backtest example, with every output explained, is given in the documentation (§13).

## Installation

Requires **Python 3.13**.

```bash
# 1. Clone
git clone https://github.com/Adrian-Vanyi/systematic-long-short-equity-backtesting-framework.git

# 2. Create a virtual environment
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

## Results at a glance

The following are some framework results. The three strategies are
well-known baselines.

**The ledger reconciles:** Every backtest decomposes each day's equity P&L into
price, dividend, financing, commission and execution-cost streams and checks the
decomposition as an identity on every date. On the documented run (§13.1) the
largest daily residual is 1.5e-10 against a 1e-6 tolerance.

**Costs are explicit:** On that run, a gross price P&L of +12.08% of initial
equity nets to +9.39% after costs, the largest stream being execution costs at
1.66% (§12.8).

**The diagnostics separate constrained from unconstrained books:** Across the
twelve strategy and window combinations, the two MVO strategies (using beta and net-exposure caps at rebalance)
realise |beta| <= 0.272 and average net exposure within [-11.7%, +3.2%], while
momentum and the factors model (both unconstrained) reach beta = +1.590 and average net exposures of
81.9% and 43.5% (§14.3.7).

**The strategies mostly do not perform:** Across 24 combinations of strategy/historical-window/
rebalancing-rule, the highest annualized Sharpe is 0.22 and 18 are
negative (§14.1.6). Annualized alpha against the market is negative in 19 of 24
runs (median -11.8%); the single run rejecting alpha = 0 at 5% is a loss of
-23.1% (§14.3.7). However, sharpe figures here are cross-strategy comparators, not
forward-looking estimates (§14.1).

**Target-hit statistics** (§14.2), over 12 quarterly start dates from 2021-01 to
2023-10, one-year runs, 50% stop-loss: at a 7% annual return target, momentum hit
in 10 of 12 start dates (83.3%), averaging 71.8 trading days to target, against
50.0% for the factors model. Min-variance MVO was the only one of the four
strategy variants with no stop-loss termination anywhere in the 288 backtests.
Twelve start dates is a demonstration of the methodology, not a statistically
meaningful sample (§14.2.1), and §15 treats strong comparison metrics as weak
positive evidence: a necessary condition for deployment, not a sufficient one.

## Limitations and possible extensions

The framework is a research tool, and the documentation (§16) presents its limitations as well as natural extensions.

 Among the limitations:

- **Daily frequency only** (trades execute at the close price (used as the pre-trade mid proxy); there is no intraday execution)
- **Static trading-cost assumptions** (half-spread, slippage and the per-share commission are parameter inputs applied uniformly across dates and stocks,
overridable per (date, ticker) for stress-testing. The half-spread and slippage are expressed in basis points of the pre-trade mid and do not scale
with trade size, which becomes a relevant issue for large orders; a size-dependent extension based on the square-root market-impact law is discussed in §2.5.2.)
- **Partial survivorship-bias mitigation** (we require full forward price data over each inter-rebalance window, e.g. one month, for accurate mark-to-market, which can drop stocks that are delisted or acquired during that period).
- **No order-size constraints** (trade size is not checked against average daily volume, so a position representing a few percent of ADV still fills at the §2.5.2 execution price. Nevertheless, the KPI report flags the worst case via `max_pct_of_daily_market_volume_traded_for_a_ticker_during_backtest`)
- **Limited corporate-action handling** (stock dividends are accounted for, but splits, spin-offs, mergers, ticker changes and special dividends are handled only insofar as the price feed series accommodates them)
- **No short-availability or recall modelling** (opening a short requires borrowing the underlying shares, which we assume are always available; broker recalls, i.e. forced buy-ins, are not modelled)
- **No exhaustive risk-management overlay** (see §16.1 for details)
- **No regime-switching or adaptive parameters** (estimation windows and thresholds are fixed for the whole backtest)
- **No tax accounting**

Among the possible extensions:

- **Improving the expected-return input for the MVO strategy** (e.g. a Black–Litterman approach for a Bayesian combination of views; §16.2 explains why the expected-return vector, rather than the covariance matrix, is the highest-value target)
- **Multi-factor neutrality for the constrained MVO strategy** (extending the single-factor beta constraint to a set of known risk factors)
- **Walk-forward hyperparameter optimisation** (estimation is already walk-forward; the configuration is not)
- **Additional strategies** (the strategy interface is a pluggable part)
- **An additional risk-management overlay (§16.2)**
- **Live-trading integration** (developing the connection to a broker's account via API)