"""Logs of book changes during a backtest.

Three log functions, each writing chronological entries to a text file.

Public API
----------
* function `log_short_proceeds_changes(...)`
        changes to FIFO short-proceeds lots.

* function `log_trades(...)`
        per-date trade simulation records.

* function `log_position_changes(...)`
        changes to the share positions of the book.

"""

from __future__ import annotations
import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator
import pandas as pd

from modules.backtest import BacktestResults


logger = logging.getLogger(__name__)


# Formatting parameters
_HEADER_DASH = "-" * 15
_SECTION_DASH = "-" * 50
_INDENT = " " * 5


def _write_log_header(f, title: str) -> None:
    f.write(f"\n{_HEADER_DASH}{_INDENT}{title}{_INDENT}{_HEADER_DASH}\n\n")


def _write_log_sub_header(f, subtitle: str) -> None:
    f.write(f"{_INDENT} {subtitle} {_INDENT}\n\n")


def _write_log_entry(f, label: str, content: Any) -> None:
    f.write(f"\n{label}\n\n")
    f.write(str(content))
    f.write(f"\n{_SECTION_DASH}\n")


def _write_metadata_header(f, backtest_results: BacktestResults) -> None:
    """Common metadata header for all audit logs."""
    book_at_date = backtest_results.book_at_date
    backtest_dates = list(book_at_date.keys())
    first_date = backtest_dates[0].strftime("%Y-%m-%d")
    last_date = backtest_dates[-1].strftime("%Y-%m-%d")
    f.write(f"Strategy: {backtest_results.strategy.strategy_name}\n")
    f.write(f"MR cure method: {backtest_results.cure_method_for_MR_violation}\n")
    f.write(f"Backtest period: {first_date} to {last_date}\n")
    f.write(f"Number of backtest dates: {len(backtest_dates)}\n\n")


def _walk_snapshot_changes(
    book_at_date: dict,
    field_name: str,
    equality_fn: Callable[[Any, Any], bool]
) -> Iterator[tuple[int, pd.Timestamp, str, Any]]:
    """Yield (index, date, book_snapshot, value)-tuples for each change
    in `book_snapshot.field_name` across the book history.

    The first tuple is the value at the first date's
    snapshot at the open. Subsequent tuples are produced whenever the
    snapshot value differs from the previously-yielded value, scanning in the 
    in order 1. open, 2. close  for each date.
    """
    backtest_dates = list(book_at_date.keys())
    if not backtest_dates:
        return

    first_date = backtest_dates[0]
    current = getattr(book_at_date[first_date].open, field_name)
    yield 1, first_date, "open", current

    close_value = getattr(book_at_date[first_date].close, field_name)
    if not equality_fn(close_value, current):
        current = close_value
        yield 1, first_date, "close", current

    for i, date in enumerate(backtest_dates[1:], start=2):
        post_value = getattr(book_at_date[date].open, field_name)
        if not equality_fn(post_value, current):
            current = post_value
            yield i, date, "open", current
        close_value = getattr(book_at_date[date].close, field_name)
        if not equality_fn(close_value, current):
            current = close_value
            yield i, date, "close", current


def _series_equal(s1: pd.Series, s2: pd.Series) -> bool:
    """Element-wise equality for share Series."""
    if not s1.index.equals(s2.index):
        return False
    return bool((s1.fillna(0) == s2.fillna(0)).all())


def _lots_equal(
    a: dict[str, deque],
    b: dict[str, deque],
    tol: float = 1e-9,
) -> bool:
    """Equality of FIFO short-proceeds lot dicts;
      Tolerant on lot prices small numerical discrepancies (because of floating-point arithmetic), controled with parameter `tol`."""
    if a.keys() != b.keys():
        return False
    for k in a:
        if len(a[k]) != len(b[k]):
            return False
        for (n_a, p_a), (n_b, p_b) in zip(a[k], b[k]):
            if n_a != n_b or abs(p_a - p_b) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_short_proceeds_changes(
    filepath: str | Path,
    backtest_results: BacktestResults
) -> None:
    """Write a chronological log of changes to FIFO short-proceeds lots."""
    with open(filepath, "w") as f:
        _write_metadata_header(f, backtest_results)
        _write_log_header(f, "CHRONOLOGICAL ORDER OF CHANGES IN SHORT-PROCEEDS LOTS DURING BACKTEST")
        _write_log_sub_header(f,"(on any given date, changes occur when opening, increasing, reducing, or closing short positions)")
        for i, date, moment, value in _walk_snapshot_changes(backtest_results.book_at_date, "short_proceeds_lots", _lots_equal):
            label = (
                f"short-proceeds lots on backtest date #{i} "
                f"({date.strftime('%Y-%m-%d')}), at {moment}:"
            )
            _write_log_entry(f, label, value)


def log_trades(
    filepath: str | Path,
    backtest_results: BacktestResults
) -> None:
    """Write a chronological log of all trades."""
    trades_log = backtest_results.trades_log
    backtest_dates = list(backtest_results.book_at_date.keys())
    with open(filepath, "w") as f:
        _write_metadata_header(f, backtest_results)
        _write_log_header(f, "BACKTEST TRADES IN CHRONOLOGICAL ORDER")
        if not trades_log:
            f.write("No trades executed during the backtest.\n")
            return

        first = next(iter(trades_log)).strftime("%Y-%m-%d")
        last = list(trades_log.keys())[-1].strftime("%Y-%m-%d")
        f.write(f"(first trade date: {first}; last trade date: {last})\n\n")

        for date, values in trades_log.items():
            i = backtest_dates.index(date) +1
            label = f"shares traded on backtest date #{i} ({date.strftime('%Y-%m-%d')}):"
            _write_log_entry(f, label, values["shares_traded"])


def log_position_changes(
    filepath: str | Path,
    backtest_results: BacktestResults
) -> None:
    """Write a chronological log of changes to share positions in the book."""
    book_at_date = backtest_results.book_at_date
    with open(filepath, "w") as f:
        _write_metadata_header(f, backtest_results)
        _write_log_header(f, "CHRONOLOGICAL ORDER OF CHANGES IN POSITIONS DURING BACKTEST")
        last_logged_was_at_last_date = False
        backtest_dates = list(book_at_date.keys())
        last_date = backtest_dates[-1]

        for i, date, moment, value in _walk_snapshot_changes(
            book_at_date, "shares", _series_equal
        ):
            label = (
                f"shares in book on backtest date #{i} "
                f"({date.strftime('%Y-%m-%d')}), at {moment}:"
            )
            _write_log_entry(f, label, value)
            if date == last_date:
                last_logged_was_at_last_date = True

        if not last_logged_was_at_last_date:
            label = (
                f"shares in book on the last backtest date "
                f"({last_date.strftime('%Y-%m-%d')}), at the close:"
            )
            _write_log_entry(
                f, label, "(unchanged from the last logged change above)"
            )
            