"""One-time bulk download of S&P 500 historical membership, prices, and dividends.

For each first NYSE trading day of each month in [2008-01-01, present]:
  -  record the S&P 500 constituents at that date.

For the union of all tickers that appear in any of those monthly snapshots:
  - download daily prices (2008-present).
  - download dividend history (2008-present).

Also download daily SPY prices as a market-return proxy.

Outputs (written to ./data/):
  - sp500_members_at_start_of_months.parquet
  - prices_sp500_members.parquet
  - daily_spy_prices.parquet
  - dividends_data.parquet

Run from the project root ONCE; notebooks read from the parquet files rather than re-fetching.
"""

import pandas as pd
import modules.sp500_historical_members as sp500
import modules.backtest_calendar as bcal
import modules.market_data as md


start = pd.Timestamp("2008-01-01")
end = pd.Timestamp.now() 


first_trading_day_of_every_month = bcal.first_trading_day_of_every_month(start, end)

# download sp500 historical members for each first-trading-day of month since start, until end
print(f"\ndownloading historical members of S&P 500 index at each first-trading-day of month "
      f"since {start.strftime("%Y-%m-%d")} until {end.strftime("%Y-%m-%d")}...",
      flush=True
)
first_trading_day_of_month_to_sp500_members_dict = sp500.get_index_members_history(dates = first_trading_day_of_every_month)
sp500_members_at_start_of_months = pd.DataFrame.from_dict(first_trading_day_of_month_to_sp500_members_dict, orient = 'index')
sp500_members_at_start_of_months.index.rename("date", inplace=True)
sp500_members_at_start_of_months.index = pd.to_datetime(sp500_members_at_start_of_months.index)
sp500_members_at_start_of_months.to_parquet("data/sp500_members_at_start_of_months.parquet")

all_sp500_members_during_backtest = list(set().union(*sp500_members_at_start_of_months.values) - {None})
all_sp500_members_during_backtest = [ ticker for ticker in all_sp500_members_during_backtest  if isinstance(ticker, str)]

# download price data for the above sp500 historical members 
print(f"\ndownloading prices for all {len(all_sp500_members_during_backtest)} historical members "
      f"since {start.strftime("%Y-%m-%d")} until {end.strftime("%Y-%m-%d")}...",
      flush=True
)
price_data = md.download_prices_yf(all_sp500_members_during_backtest, start, end)
price_data.to_parquet("data/prices_sp500_members.parquet")  

# download dividend data for the above sp500 historical members 
print(f"\ndownloading dividend events for all {len(all_sp500_members_during_backtest)} historical members "
      f"since {start.strftime("%Y-%m-%d")} until {end.strftime("%Y-%m-%d")}...",
      flush=True
  )
dividend_data = md.download_dividend_data(all_sp500_members_during_backtest, start , end)
dividend_data.to_frame().to_parquet("data/dividends_data.parquet")


# Download daily SPY adjusted returns
print(f"\ndownloading SPY daily adjusted returns since {start.strftime("%Y-%m-%d")} until {end.strftime("%Y-%m-%d")}...",
      flush = True     
)
daily_spy_returns = md.download_market_returns("SPY", start, end)
daily_spy_returns.to_frame().to_parquet("data/daily_spy_returns.parquet")  

