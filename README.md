# A Broker-Realistic Backtesting Framework for Systematic Long/Short Equity Strategies

A daily-frequency backtesting framework for systematic long/short equity strategies on the historical S&P 500 (US equities). It is built around a realistic brokerage ledger: financing (debit interest, cash interest, stock-borrow fees, short rebate), dividends on both legs, Reg-T / FINRA margin-requirement mechanics, trading costs, and per-ticker FIFO short-proceeds lots. On top of this we implemented point-in-time construction of the investable universe, a pluggable and extensible strategy interface, and a detailed diagnostics/KPI layer.

The goal was the framework itself (a foundation for later work on alpha discovery) not the strategies. The three strategies currently implemented (momentum, a factors model, MVO) are well known; they serve to test the framework and be measured by it, through two performance comparison methodologies (target-hit statistics and Sharpe ratios).

## Contents

- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Results at a glance](#results-at-a-glance)
- [Limitations and possible extensions](#limitations)



## Documentation

**The full design of the framework and the derivations of the strategies are in the accompanying [`backtesting_framework_documentation.pdf`](./backtesting_framework_documentation.pdf).** It covers brokerage-book mechanics, margin requirements, backtest-calendar construction, backtest termination and the strategies' portfolio rebalancing rules, construction of the universe of investable tickers per rebalance date, the rigorous derivation of each strategy, the KPIs, a complete single-backtest example, the cross-strategy performance comparison methodology and results, and an appendix justifying the filter we apply to our price feed to remove bad ticker data.


## Repository layout

| Path | Purpose |
| --- | --- |
| `modules/` | Python modules with the core framework implementation |
| `data/` | Pre-downloaded price and index-membership data used to generate the documented results |
| `download_data.py` | Standalone utility to download price and index-membership data up to the present time |
| `outputs/` | Default destination for saved backtest files: audit logs, KPI report, and performance comparison tables |
| `single_backtest_runner.ipynb` | Runs one configurable backtest end to end |
| `strategies_comparison.ipynb` | Compares strategies across parameter settings over many runs |
| `mvo_solver.ipynb` | Tests and validates the portfolio optimizer used by the MVO strategy |
| `factors_model.ipynb` | Tests and validates the factors model implementation used by the corresponding strategy |
| `backtest_calendar.ipynb` | Visualises the construction of the backtest calendar |
| `sp500_historical_members.ipynb` | Tests retrieval of historical S&P 500 members from Wikipedia revision history |

## Quick start

1. The dataset used for the documented results is already included (`data/`), so you can skip to 2. Run `download_data.py` only if you want to refresh the data up to the present.
2. Open `single_backtest_runner.ipynb`, set the configuration (strategy, calendar, margin and cost parameters, termination rules), and run it end to end to produce the diagnostic plots, book-value table, audit logs, and KPI report.
3. Open `strategies_comparison.ipynb` to run the cross-strategy comparison sweeps.

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

The comparison methodology (§14) presents some results. For example, sweeping each strategy across **12 start dates** and a range of return targets, the runs produced:

- The **momentum strategy** hit a **7% return target in 83.3% of runs**, with an average time-to-target of **~72 trading days**.
- The **minimum-variance MVO** strategy recorded **zero stop-loss terminations across all 288 runs**.

The framework also reports risk-adjusted performance (Sharpe ratios) across historical market windows; these are meant for comparing the strategies against each other, not for predicting future performance (see §14–§15 of the documentation for the full tables and the discussion of what they do and do not justify).

## Limitations and possible extensions

The framework is a research tool, and the documentation (§16) presents its limitations as well as natural extensions. Among the limitations:

- **Daily frequency only** (trades execute at the close price (used as the pre-trade mid proxy); there is no intraday execution)
- **Static trading-cost assumptions:** (trading costs are modelled as fixed half-spread and slippage terms in basis points relative to the pre-trade mid-price, and do not scale with trade size, which becomes a relevant issue for large orders. A size-dependent extension based on the square-root market-impact law is discussed in §2.5.2.)
- **Partial survivorship-bias mitigation** (we require full forward price data over each inter-rebalance window, e.g. one month, for accurate mark-to-market, which can drop stocks that are delisted or acquired during that period).
- **No short-availability or recall modelling** (opening a short requires borrowing the underlying shares, which we assume are always available; broker recalls, i.e. forced buy-ins, are not modelled)
- **No exhaustive risk-management overlay** (see §16.1 for details)
- **No tax accounting**

Among the possible extensions:

- **Improving the expected-return input for the MVO strategy** (e.g. a Black–Litterman approach for a Bayesian combination of views)
- **An additional risk-management overlay**
- **Live-trading integration** (developing the connection to a broker's account via API)