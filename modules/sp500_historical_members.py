"""Build the point-in-time S&P 500 membership using the MediaWiki Revisions API.

Wikipedia's "List of S&P 500 companies" page is updated as the index composition changes. 
We scrape the revision history to reconstruct membership at any past date (best-effort, no paid data vendor needed).

Limitations
-----------
- Earliest supported date: 2007-03-06 (when Wikipedia first published the table in its current structured form).
- Wikipedia data may not be perfectly accurate; cross-source validation is recommended.

References
----------
- MediaWiki Revisions API: https://www.mediawiki.org/wiki/API:Revisions
- Wikipedia page: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Public API
----------
* function `get_revisions_metadata(...)`
        raw MediaWiki revisions metadata for a Wikipedia page

* function `get_index_members_at(...)`
        constituents at one date (from the latest revision as of the market open of that date)

* function `get_index_members_history(...)`
        bulk fetch across dates, with optional on-disk JSON cache

"""
from __future__ import annotations
import datetime
import json
import logging
import time
import urllib.parse
from io import StringIO
from pathlib import Path
from typing import Iterable
import pandas as pd
from pandas.tseries.offsets import BDay
import requests

from modules import backtest_calendar as bcal
from modules.utils import timeit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

WIKIPEDIA_PAGES: dict[str, str] = {
    "SPX": "List of S&P 500 companies",
}

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_URL_BASE = "https://en.wikipedia.org/w/index.php"
USER_AGENT = "TradingBacktestingFramework/1.0 (idontwanttodisclosemyemail@gmail.com)"
EARLIEST_SUPPORTED_DATE = pd.Timestamp("2007-03-06")

_HTTP_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 1.5

# Reasonable bounds for the constituent count; warn if outside.
_EXPECTED_CONSTITUENT_RANGE = (400, 600)

# Module-level session (TCP connection pooling for repeated calls).
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT})
    return _session


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------

def _http_get(url: str, params: dict | None = None) -> requests.Response:
    """GET with exponential backoff on transient failures."""
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = _get_session().get(url, params=params, timeout=_HTTP_TIMEOUT_SECONDS)
            r.raise_for_status()
            return r
        except (requests.RequestException, ConnectionError) as e:
            last_err = e
            wait = _BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                "HTTP attempt %d/%d failed for %s: %s. Retrying in %.1fs.",
                attempt + 1, _MAX_RETRIES, url, e, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after {_MAX_RETRIES} attempts: {url}") from last_err


# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------

def _isoformat(date) -> str:
    return pd.to_datetime(date).isoformat()


def get_revisions_metadata(
    page_title: str = WIKIPEDIA_PAGES["SPX"],
    rvstart=None,
    rvend=None,
    rvdir: str = "older",
    rvlimit: int = 1,
    **kwargs,
) -> list[dict]:
    """Fetch revision metadata via the MediaWiki Revisions API.

    Parameters
    ----------
    page_title
        Wikipedia page title.
    rvstart, rvend
        Filter revisions by date. Most date formats accepted; `None` means no bound.
    rvdir
        "older" (default):  results ordered new -> old; suitable for finding the latest revision before a date.
        "newer" → results ordered old -> new.
    rvlimit
        Maximum number of revisions to return.
    **kwargs
        Additional MediaWiki API parameters.
    """
    query_params: dict = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "ids|timestamp|user|comment",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json",
        "rvlimit": rvlimit,
        "rvdir": rvdir,
    }
    if rvstart is not None:
        rvstart += BDay(1)
        query_params["rvstart"] = _isoformat(rvstart)
    if rvend is not None:
        query_params["rvend"] = _isoformat(rvend)
    query_params.update(kwargs)

    r = _http_get(WIKIPEDIA_API_URL, params=query_params)
    data = r.json()

    try:
        revisions = data["query"]["pages"][0]["revisions"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"unexpected MediaWiki API response shape for "
            f"page {page_title!r}: {data}"
        ) from e
    return revisions


# ---------------------------------------------------------------------------
# Members at a single date
# ---------------------------------------------------------------------------

def get_index_members_at(
    date: pd.Timestamp,
    index: str = "SPX",
) -> tuple[list[str], pd.DataFrame]:
    """Index members at a given date, taken from the latest Wikipedia revision 
    strictly before market open on date `date`.

    Returns
    -------
    (members_list, members_info_df)
        `members_list` is the list of tickers (sorted alphabetically).
        `members_info_df` is the full Wikipedia table, indexed by ticker, 
        retained for downstream metadata if needed.
    """
    date = date.normalize()
    if date < EARLIEST_SUPPORTED_DATE:
        raise ValueError(
            f"date {date:%Y-%m-%d} is before the earliest supported date "
            f"({EARLIEST_SUPPORTED_DATE:%Y-%m-%d})"
        )

    page = WIKIPEDIA_PAGES[index]
    revisions = get_revisions_metadata(
        page, rvdir="older", rvlimit=1, rvstart=date,
    )
    if not revisions:
        raise RuntimeError(f"no revisions found before {date} for page {page!r}")
    revision = revisions[0]
    revid = revision["revid"]

    rev_ts = pd.Timestamp(revision["timestamp"])
    gap_days = (date - rev_ts.tz_localize(None)).days
    if gap_days > 60:
        logger.warning(
            "membership for %s comes from a revision %d days old (%s)"
            "(may be stale).", date.date(), gap_days, revision["timestamp"],
        )

    url = f"{WIKIPEDIA_PAGE_URL_BASE}?title={urllib.parse.quote(page)}&oldid={revid}"
    html = _http_get(url).text

    try:
        tables = pd.read_html(StringIO(html))
    except ValueError as e:
        raise RuntimeError(
            f"could not parse any HTML table from {url}"
        ) from e

    for df in tables:
        for col in ("Symbol", "Ticker symbol"):
            if col in df.columns:
                members_info_df = df.set_index(col).sort_index()
                members = list(members_info_df.index)
                lo, hi = _EXPECTED_CONSTITUENT_RANGE
                if not (lo <= len(members) <= hi):
                    logger.warning(
                        "unexpected constituent count at %s: %d (expected %d–%d)",
                        date.date(), len(members), lo, hi,
                    )
                return members, members_info_df

    cols_seen = [list(df.columns) for df in tables]
    raise RuntimeError(
        f"could not find Symbol/Ticker symbol column in any table fetched "
        f"from {url}. Tables had columns: {cols_seen}"
    )


# ---------------------------------------------------------------------------
# Bulk history with caching
# ---------------------------------------------------------------------------

@timeit
def get_index_members_history(
    dates: Iterable[pd.Timestamp] | None = None,
    start_date: pd.Timestamp = EARLIEST_SUPPORTED_DATE,
    end_date: pd.Timestamp | None = None,
    freq: str = "BME",
    index: str = "SPX",
    cache_path: str | Path | None = None,
) -> dict[pd.Timestamp, list[str]]:
    """Fetch historical index members at a sequence of dates.

    Parameters
    ----------
    dates
        Explicit dates at which to fetch members. If provided,
        `start_date`/`end_date`/`freq` are ignored.
    start_date, end_date
        Used to build a date range when `dates` is None. `end_date` defaults to today.
    freq
        Pandas frequency string for the date range (default "BME", business month-end).
    index
        Index identifier (currently only "SPX" supported, i.e., the S&P 500 index).
    cache_path
        If provided, results are cached to this JSON file. On a second call with the same 
        `cache_path`, only dates not already cached are fetched. Wikipedia membership for a 
        past date is immutable, so caching is safe and efficient.

    Returns
    -------
    dict[pd.Timestamp, list[str]]
        Mapping from each requested date to the list of constituent tickers.
    """
    if dates is None:
        if end_date is None:
            end_date = pd.Timestamp(datetime.date.today())
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    dates = [pd.Timestamp(d).normalize() for d in dates]

    cache_path = Path(cache_path) if cache_path is not None else None
    cache: dict[str, list[str]] = {}
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    out: dict[pd.Timestamp, list[str]] = {}
    n_fetched = 0
    for date in dates:
        key = date.strftime("%Y-%m-%d")
        if key in cache:
            out[date] = cache[key]
            continue
        members, _ = get_index_members_at(date=date, index=index)
        out[date] = members
        cache[key] = members
        n_fetched += 1
        logger.info("fetched index members for %s (n=%d)", key, len(members))

    if cache_path is not None and n_fetched > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        logger.info("wrote cache (%d total entries) to %s", len(cache), cache_path)

    return out
