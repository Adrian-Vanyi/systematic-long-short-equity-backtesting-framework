"""Market-data fetching utilities (yfinance + FRED CSVs).

Public API
----------
* function `compute_price_data_window(...)`
        determine window bounds for a backtest

* function `download_prices_yf(...)`
        wide price data panel for a list of tickers (open, close, adj_close, volume).

* function `download_market_returns(...)`
        daily returns of the specified market proxy.

* function `download_dividend_data(...)`
        the dividend-per-share on every ex-dividend date of each ticker, in the specified date window.

* function `download_dividend_data_at_date(...)`
        the dividend-per-share at date (for every ticker), if date is an ex-div date for the ticker.

"""

from __future__ import annotations
import logging
import pandas as pd
import yfinance as yf

from modules import backtest_calendar as bcal
from modules.utils import timeit


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def compute_price_data_window(
    backtest_dates: pd.DatetimeIndex,
    training_rebalance_dates: pd.DatetimeIndex,
    rolling_window_trading_days_for_momentum: int,
    buffer_trading_days_for_momentum: int,
    rolling_window_trading_days_for_volatility: int,
    rolling_window_trading_days_for_adv: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute the (fetch_start, fetch_end) date window required to
    support all model inputs at every training rebalance date. (see documentation, §4.4)

    The start date is set to the oldest trading date that any
    feature-engineering window will reach back to from the first
    training rebalance date. The end date extends one trading day past
    the last backtest date (yfinance excludes the end date when fetching).
    """
    max_offset_trading_days = max(
        buffer_trading_days_for_momentum + rolling_window_trading_days_for_momentum,
        rolling_window_trading_days_for_volatility + 1,
        rolling_window_trading_days_for_adv,
    )

    fetch_start = bcal.subtract_trading_days(
        training_rebalance_dates[0], max_offset_trading_days
    )
    fetch_end = bcal.add_trading_days(backtest_dates[-1], 1)
    return fetch_start, fetch_end


@timeit
def download_prices_yf(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Download daily OHLCV prices via yfinance.

    Returns a long DataFrame indexed by (date, ticker) with columns
    ["close", "adj_close", "volume"]. 

    Tickers missing from the yfinance response are logged at WARNING
    level. The returned DataFrame contains only tickers for which data
    was successfully retrieved.
    """
    fetch_start = start.date().isoformat()
    fetch_end = end.date().isoformat()

    data = yf.download(
        tickers = tickers,
        start = fetch_start,
        end = fetch_end,
        auto_adjust = False,
        group_by = "ticker",
        progress = False,
        threads = True
    )
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for t in tickers:
        if t not in data:
            missing.append(t)
            continue
        df = data[t][["Open","Close", "Adj Close", "Volume"]].rename(
            columns={"Open": "open","Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
        )
        df["ticker"] = t
        frames.append(df.reset_index())

    if missing:
        preview = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        logger.warning(
            "yfinance returned no data for %d/%d tickers: %s",
            len(missing), len(tickers), preview,
        )
    if not frames:
        raise ValueError(
            "yfinance returned no data for any of the requested tickers; "
            "check ticker spelling and date range"
        )
    prices = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date"})
    prices["ticker"] = prices["ticker"].astype("category")
    prices = prices.set_index(["date", "ticker"]).sort_index()
    prices.columns.name = None  # yfinance leaves a leftover axis name
    return prices


# ---------------------------------------------------------------------------
# Market returns
# ---------------------------------------------------------------------------

@timeit
def download_market_returns(
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.Series:
    """Adjusted-close daily simple returns of a market proxy ETF (e.g., the "SPY" ETF)
    where adjusted-close is dividend-adjusted and provides a total-return proxy. 
    """
    df = yf.download(
        symbol, start=start, end=end,
        auto_adjust=False, progress=False, threads=True
    )[["Adj Close"]]
    s = df.iloc[:, 0].astype(float)
    s.name = f"{symbol}_returns"
    s.index.name = "date"
    return s.pct_change()


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

def _ticker_dividend_series(ticker: str) -> pd.Series | None:
    """Fetch a ticker's dividend history from yfinance.
    Returns 'None' if the ticker has no dividend record.
    """
    try:
        s = yf.Ticker(ticker).dividends
    except Exception:
        return None
    if s is None or s.empty:
        return None
    if isinstance(s, pd.DataFrame):
        s = s["Dividends"] if "Dividends" in s.columns else s.squeeze("columns")
    s.name = "dividend per share"
    if s.dtype == object:
        logger.debug("dividend series for %s has object dtype; coercing", ticker)
        s = s.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
        s = pd.to_numeric(s, errors="coerce")
    return s


def _build_long_dividend_series(
    ticker_to_dates_and_amounts: dict[str, tuple[pd.DatetimeIndex, pd.Series]],
) -> pd.Series:
    """Stack a dict mapping a ticker to its {ex_dates : amounts} into a long Series indexed by
    (ticker, ex_dividend_date) and values the amounts (dividend-per-share).
    """
    if not ticker_to_dates_and_amounts:
        return pd.Series(
            index = pd.MultiIndex.from_arrays(
                [[], []], names=["ticker", "ex-dividend date"]
            ),
            name = "dividend per share",
            dtype = float
        )
    rows = []
    for ticker, (dates, amounts) in ticker_to_dates_and_amounts.items():
        rows.append(
            pd.Series(
                amounts,
                index = pd.MultiIndex.from_arrays(
                    [[ticker] * len(dates), dates],
                    names=["ticker", "ex-dividend date"],
                ),
                name = "dividend per share"
            )
        )
    return pd.concat(rows).sort_index()


@timeit
def download_dividend_data(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Fetch all dividend events in [start, end] for a list of tickers.
    Returns a long Series indexed by (ticker, ex_dividend_date), values = dividend per share.

    Note: this issues one yfinance network API call per ticker.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    out: dict = {}

    for ticker in tickers:
        s = _ticker_dividend_series(ticker)
        if s is None:
            continue
        ex_dates = pd.to_datetime(s.index.date)
        mask = (ex_dates >= start) & (ex_dates <= end)
        if mask.any():
            out[ticker] = (ex_dates[mask], s.loc[mask].to_numpy())

    return _build_long_dividend_series(out)


@timeit
def download_dividend_data_at_date(
    date: pd.Timestamp, tickers: list[str]
) -> pd.Series:
    """Fetch dividend events on a specific ex-date for a list of tickers."""
    return download_dividend_data(tickers, date, date)


