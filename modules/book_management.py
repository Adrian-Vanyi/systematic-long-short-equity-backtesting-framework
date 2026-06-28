"""Book (ledger) management.

Design
------
Trade simulation uses a `_TradeEffect` dataclass per ticker. The four-trade-cases dispatcher (`_build_trade_effect`)
translates a (pre-trade shares, shares-traded) pair into a `_TradeEffect`; 
the applier (`_apply_trade_effect`) consumes it,  mutating the book snapshot and running the cash/debit settlement.

Public API
----------
* class `BookSnapshot`
        typed dataclass for one moment-in-time book state.

* class `Book`
        container holding the `open` and `close` snapshots.

* function `create_book(tickers, initial_cash)`
        build a fresh book for the start of the backtest.

* function `mark_to_market_positions(date, snapshot, prices)`
        re-mark to market.

* function `apply_financing_and_carry(...)`
        overnight financing accruals

* function `account_dividends(date, snapshot, dividends_data)`
        ex-dividend-date booking of dividends.

* function `update_from_all_trades(...)`
        simulate a batch of trades.

* function `compute_stop_loss_threshold(...)`
        equity threshold.
        
"""

from __future__ import annotations
import copy
import logging
from collections import deque
from dataclasses import dataclass, field
import pandas as pd

from modules.trading_utils import (
    TradingCosts,
    compute_execution_price,
    compute_trading_fee,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Book schema
# =============================================================================

@dataclass
class BookSnapshot:
    """One moment-in-time state of our brokerage book.

    All scalar fields are non-negative by construction. Sign of a
    position is encoded by the integer in `shares` (positive = long,
    negative = short).

    Attributes
    ----------
    cash
        Free, unrestricted cash balance
    debit
        Outstanding margin loan balance
    LMV, SMV
        Long and short market values
    margin_collateral
        External capital posted to cure MR violations (Variant A of the backtest).
    short_proceeds_lots
        Per-ticker FIFO queue of (n_shares, exec_price) short-sale
        records. The total over all lots equals the value of the 
        attribute `total_short_proceeds`
    total_short_proceeds
        Cached sum of n x p over all short-proceeds lots
    shares
        Series of integer share counts (signed), indexed by ticker
        Empty by default for tickers with zero positions
    long_leverage, short_leverage, gross_leverage
        Leverage ratios 
    equity, equity_excluding_margin_collateral
        Total equity (used for MR checks) and equity excluding the value of the eventual margin collateral posted 
        in our account to cure MR violations (used for performance metrics).
    """
    cash: float = 0.0
    debit: float = 0.0
    LMV: float = 0.0
    SMV: float = 0.0
    margin_collateral: float = 0.0
    short_proceeds_lots: dict[str, deque] = field(default_factory=dict)
    total_short_proceeds: float = 0.0
    shares: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=int, name="shares")
    )
    long_leverage: float = 0.0
    short_leverage: float = 0.0
    gross_leverage: float = 0.0
    equity: float = 0.0
    equity_excluding_margin_collateral: float = 0.0

    def compute_equity(self) -> float:
        return (
            self.cash
            - self.debit
            + self.LMV
            - self.SMV
            + self.total_short_proceeds
            + self.margin_collateral
        )

    def compute_total_short_proceeds_from_lots(self) -> float:
        """Recompute total short proceeds by walking every lot.
        Provided for diagnostics onlt. The hot code path uses the cached field instead.
        """
        return float(
            sum(
                n * p
                for lots in self.short_proceeds_lots.values()
                for n, p in lots
            )
        )

    def update_equity_fields(self) -> None:
        """Recompute `equity` and `equity_excluding_margin_collateral`
        from the current scalar fields.
        """
        self.equity = self.compute_equity()
        self.equity_excluding_margin_collateral = (
            self.equity - self.margin_collateral
        )

    def update_leverage_ratios(self) -> None:
        """Recompute `long_leverage`, `short_leverage`, and `gross_leverage` from the current scalar fields."""
        if self.equity != 0:
            self.long_leverage = self.LMV / self.equity
            self.short_leverage = self.SMV / self.equity
            self.gross_leverage = (self.LMV + self.SMV) / self.equity
        else:
            self.long_leverage = self.short_leverage = self.gross_leverage = 0.0
    

@dataclass
class Book:
    """Container holding the two intra-day snapshots of the book.

    * `open` is the snapshot after marking to market at the start
      of day (using open prices), after accounting for overnight financing/dividends; 

    * `close` is the snapshot at end of day (after any trades and/or MR cures occuring since the `open` moment,
      and then marking to market at the close prices of the day). 
    """
    open: BookSnapshot
    close: BookSnapshot

    def deepcopy(self) -> "Book":
        return copy.deepcopy(self)


# =============================================================================
# Construction
# =============================================================================

def create_book(
    tickers: pd.Index | list[str],
    initial_cash: float = 0.0
) -> Book:
    """Build a new book for the start of the backtest with the given initial cash.

    Both snapshots are initialised identically; the orchestrator
    overwrites the snapshot at close of this new book after the rebalance of the first backtest date.
    """
    snap = BookSnapshot(
        cash = float(initial_cash),
        shares = pd.Series(0, index=pd.Index(tickers, name="ticker"), dtype=int)
    )
    snap.update_equity_fields()
    snap.update_leverage_ratios()
    return Book(open = snap, close = copy.deepcopy(snap))


# =============================================================================
# Mark to market
# =============================================================================

def _compute_LMV_and_SMV(
    shares: pd.Series, prices_today: pd.Series
) -> tuple[float, float]:
    signed = shares * prices_today
    LMV = float(signed.clip(lower=0).sum(skipna=True))
    SMV = float((-signed.clip(upper=0)).sum(skipna=True))
    return LMV, SMV


def mark_to_market_positions(
    date: pd.Timestamp,
    snapshot: BookSnapshot,
    prices: pd.Series
) -> BookSnapshot:
    """Re-mark LMV / SMV / equity / leverage at close prices.

    Mutates and returns ``snapshot``.
    """
    shares = snapshot.shares
    prices_today = prices.xs(date, level=0)

    missing = set(shares.index) - set(prices_today.index)
    if missing:
        logger.warning(
            "missing close prices on %s for %d ticker(s): %s",
            date.strftime("%Y-%m-%d"), len(missing), sorted(missing)[:10],
        )

    prices_aligned = prices_today.reindex(shares.index)
    LMV, SMV = _compute_LMV_and_SMV(shares, prices_aligned)
    snapshot.LMV = LMV
    snapshot.SMV = SMV
    snapshot.update_equity_fields()
    snapshot.update_leverage_ratios()
    return snapshot


# =============================================================================
# Cash / debit settlement
# =============================================================================

def _convert_negative_cash_to_positive_debit(
    cash: float, debit: float
) -> tuple[float, float]:
    if cash < 0:
        debit += -cash
        cash = 0.0
    return cash, debit


def _pay_down_debit_using_cash(
    cash: float, debit: float
) -> tuple[float, float]:
    if cash > 0 and debit > 0:
        x = min(cash, debit)
        cash -= x
        debit -= x
    return cash, debit


def _settle_cash_and_debit(cash: float, debit: float) -> tuple[float, float]:
    """Apply both reclassification (mutually exclusive) and sweep; both no-ops when not needed."""
    cash, debit = _convert_negative_cash_to_positive_debit(cash, debit)
    cash, debit = _pay_down_debit_using_cash(cash, debit)
    return cash, debit


# =============================================================================
# Financing & carry
# =============================================================================

def apply_financing_and_carry(
    snapshot_prev_day: BookSnapshot,
    date: pd.Timestamp,
    prev_date: pd.Timestamp,
    close_prices: pd.Series,
    rates
) -> BookSnapshot:
    """Apply overnight financing & carry to a copy of the snapshot.

    Updates cash, debit, margin_collateral, equity, equity_excluding_margin_collateral. 
    Lots and shares are unchanged, so `total_short_proceeds` is preserved.
    """
    delta_dates = (date - prev_date).days # number of calendar days between the two consecuvite trading dates

    # daily rates fixed at the previous date's close.
    r_cash_d = rates.cash_rate_at(prev_date)
    r_debit_d = rates.debit_rate_at(prev_date)
    r_margin_collateral_d = rates.margin_collateral_rate_at(prev_date)
    r_rebate_d = rates.rebate_rate_at(prev_date)
    f_borrow_d = rates.borrow_fee_at(prev_date)

    # bases
    cash_base = snapshot_prev_day.cash
    margin_collateral_base = snapshot_prev_day.margin_collateral
    debit_base = snapshot_prev_day.debit

    shares = snapshot_prev_day.shares
    abs_shares_short = -shares[shares < 0]
    prices_prev = close_prices.xs(prev_date, level=0)
    value_shorts = abs_shares_short * prices_prev

    # cash accruals
    accrual_cash = r_cash_d * cash_base * delta_dates
    accrual_borrow_fee = float(
        (f_borrow_d * value_shorts * delta_dates).sum(skipna=True)
    )

    # rebate accrual on per-ticker short-proceeds bases
    short_proceeds_per_ticker = pd.Series(
        {
            ticker: sum(n * p for n, p in snapshot_prev_day.short_proceeds_lots.get(ticker, []))
            for ticker in abs_shares_short.index
        }
    )
    if isinstance(r_rebate_d, pd.Series):
        r_rebate_aligned = r_rebate_d.reindex(
            short_proceeds_per_ticker.index, fill_value=0.0
        )
    else:
        r_rebate_aligned = r_rebate_d
    accrual_rebate = float(
        (r_rebate_aligned * short_proceeds_per_ticker).sum()
    ) * delta_dates

    cash = snapshot_prev_day.cash + accrual_cash - accrual_borrow_fee + accrual_rebate
    debit = snapshot_prev_day.debit + r_debit_d * debit_base * delta_dates
    cash, debit = _settle_cash_and_debit(cash, debit)

    margin_collateral = (
        snapshot_prev_day.margin_collateral
        + r_margin_collateral_d * margin_collateral_base * delta_dates
    )

    updated = copy.deepcopy(snapshot_prev_day)
    updated.cash = cash
    updated.debit = debit
    updated.margin_collateral = margin_collateral
    updated.update_equity_fields()
    updated.update_leverage_ratios()
    return updated


# =============================================================================
# Dividends accounting
# =============================================================================

def account_dividends(
    date: pd.Timestamp,
    snapshot: BookSnapshot,
    dividends_data: pd.Series
) -> BookSnapshot:
    """Book dividend cash flows at `date`, if date is an ex-dividend date for any of tickers in the book.

    Long positions receive dividends (cash inflow); short positions
    owe dividends (cash outflow). Entitlement is on the start-of-day
    position, so this function must be called before any rebalance trades on the same
    date.

    Updates cash, debit, equity, equity_excluding_margin_collateral.
    Short-proceeds lots and shares are unchanged.

    Parameters
    ----------
    date
        Backtest date (the candidate ex-date)
    snapshot
        Start-of-day snapshot of date `date`. Not mutated; a deep copy is returned.
    dividends_data
        Long Series indexed by (ticker, ex-dividend date), values= dividend per share
    """
    updated = copy.deepcopy(snapshot)
    shares = updated.shares

    ex_dates = dividends_data.index.get_level_values("ex-dividend date")
    if date not in ex_dates:
        return updated

    divs_today = dividends_data.xs(date, level="ex-dividend date")
    aligned = divs_today.reindex(shares.index).dropna()
    if aligned.empty:
        return updated

    net_flow = float((shares.loc[aligned.index] * aligned).sum())

    cash, debit = _settle_cash_and_debit(updated.cash + net_flow, updated.debit)
    updated.cash = cash
    updated.debit = debit
    updated.update_equity_fields()
    updated.update_leverage_ratios()
    return updated


# =============================================================================
# Trade simulation: TradeEffect + applier
# =============================================================================

@dataclass
class _TradeEffect:
    """Elementary effects of one ticker's trade on the snapshot.

    Attributes
    ------
    shares_delta
        Signed number of shares to add to the ticker's position.
    fee
        Trading fee for this trade (non-negative). Subtracted from cash.
    cover_shares
        Number of shares to FIFO-consume from existing short-proceeds lots
        (a non-negative number). When > 0, `cover_exec_price` is used to compute the
        cash flow as `consumed_value - cover_exec_price x cover_shares`.
    cover_exec_price
        Execution price for the cover step.
    new_short_lot
        Tuple (n, exec_price) to append as a new short-proceeds lot, or None.
    cash_change_long_side
        Signed cash change for the long-side leg (excluding fee):
        postive = outflow (from a buy), negative = inflow (from a sell). This
        is substracted from cash by the applier.
    """
    shares_delta: int
    fee: float
    cover_shares: int = 0
    cover_exec_price: float = 0.0
    new_short_lot: tuple[int, float] | None = None
    cash_change_long_side: float = 0.0


def _consume_fifo_lots(lots: deque, shares_to_cover: int) -> float:
    """Consume `shares_to_cover` shares from `lots` (in place, FIFO).

    Returns the sum of `lot_price x n_consumed` over consumed lot
    fragments (this equals the short proceeds released by the cover).
    """
    remaining = shares_to_cover
    consumed_value = 0.0

    while remaining > 0 and lots:
        lot_shares, lot_price = lots.popleft()
        if lot_shares <= remaining:
            consumed_value += lot_price * lot_shares
            remaining -= lot_shares
        else:
            consumed_value += lot_price * remaining
            lots.appendleft((lot_shares - remaining, lot_price))
            remaining = 0
    return consumed_value


def _apply_trade_effect(
    snapshot: BookSnapshot, ticker: str, effect: _TradeEffect
) -> None:
    """Apply a `_TradeEffect` to `snapshot` in place.

    Order of operations:
      1. FIFO-consume `cover_shares`, if any; if so, update short-proceeds cache
      2. Append `new_short_lot` if any; if so, update short-proceeds cache
      3. Apply long-side cash change (signed inflow or outflow)
      4. Subtract trading fee
      5. Apply share delta to the ticker's shares in the book snapshot
      6. Settle cash/debit (reclassification or sweep, whatever applies)
    """
    cash = snapshot.cash
    debit = snapshot.debit

    # 1. Cover (FIFO)
    if effect.cover_shares > 0:
        lots = snapshot.short_proceeds_lots[ticker]
        consumed_value = _consume_fifo_lots(lots, effect.cover_shares)
        cash += consumed_value - effect.cover_exec_price * effect.cover_shares
        snapshot.total_short_proceeds -= consumed_value

    # 2. Open / increase short
    if effect.new_short_lot is not None:
        n_new, p_new = effect.new_short_lot
        snapshot.short_proceeds_lots.setdefault(ticker, deque()).append(
            (n_new, p_new)
        )
        snapshot.total_short_proceeds += n_new * p_new

    # 3. Long-side cash change
    cash -= effect.cash_change_long_side 

    # 4. Trading fee
    cash -= effect.fee

    # 5. Share delta
    shares = snapshot.shares
    if ticker in shares.index:
        shares.loc[ticker] = shares.loc[ticker] + effect.shares_delta
    else:
        shares.loc[ticker] = effect.shares_delta
    snapshot.shares = shares.astype("int64")

    # 6. Settle
    snapshot.cash, snapshot.debit = _settle_cash_and_debit(cash, debit)


# =============================================================================
# Per-ticker trade dispatcher (computes a _TradeEffect)
# =============================================================================

def _build_trade_effect(
    shares_pre_trade: int,
    shares_traded: int,
    full_spread_bps: float,
    slippage_bps: float,
    price_pre_trade: float,
    commission_per_share: float
) -> _TradeEffect:
    """Translate one ticker's trade into a `_TradeEffect`, dispatching
    on the four cases:

    1. Cover short (possibly flipping to long):  q_pre < 0, Δq > 0
    2. Open / increase short:                    q_pre ≤ 0, Δq < 0
    3. Reduce long (possibly flipping to short): q_pre > 0, Δq < 0
    4. Open / increase long :                    q_pre ≥ 0, Δq > 0
    """
    abs_traded = abs(shares_traded)
    exec_price = compute_execution_price(
        shares_traded, full_spread_bps, slippage_bps, price_pre_trade
    )
    fee = compute_trading_fee(abs_traded, commission_per_share)

    # Case 1: cover short, possibly flipping to long
    if shares_pre_trade < 0 and shares_traded > 0:
        cover = int(min(-shares_pre_trade, shares_traded))
        flip = int(shares_traded - cover)
        return _TradeEffect(
            shares_delta = shares_traded,
            fee = fee,
            cover_shares = cover,
            cover_exec_price = exec_price,
            cash_change_long_side = exec_price * flip  # buy on the flip leg
        )

    # Case 2: open / increase short
    if shares_pre_trade <= 0 and shares_traded < 0:
        return _TradeEffect(
            shares_delta=shares_traded,
            fee=fee,
            new_short_lot=(abs_traded, exec_price)
        )

    # Case 3: reduce long, possibly flipping to short
    if shares_pre_trade > 0 and shares_traded < 0:
        sell = int(min(shares_pre_trade, abs_traded))
        short = int(abs_traded - sell)
        new_short_lot = (short, exec_price) if short > 0 else None
        return _TradeEffect(
            shares_delta = shares_traded,
            fee = fee,
            new_short_lot = new_short_lot,
            cash_change_long_side = - exec_price * sell  # sell brings cash in
        )

    # Case 4: open / increase long
    if shares_pre_trade >= 0 and shares_traded > 0:
        return _TradeEffect(
            shares_delta = shares_traded,
            fee = fee,
            cash_change_long_side = exec_price * shares_traded,  # outflow
        )

    raise RuntimeError(
        f"unreachable: shares_pre_trade={shares_pre_trade}, "
        f"shares_traded={shares_traded}"
    )


# =============================================================================
# Batch trade simulation
# =============================================================================

def update_from_all_trades(
    current_snapshot: BookSnapshot,
    trades_log: dict,
    market_volume: pd.Series, 
    date: pd.Timestamp,
    shares_to_trade: pd.Series,
    trading_costs: TradingCosts,
    prices_pre_trade: pd.Series,
    prices_post_trade: pd.Series | None = None
) -> tuple[BookSnapshot, dict]:
    """Execute a batch of per-ticker trades on a copy of the snapshot.

    Builds a `_TradeEffect` per non-zero trade, applies them in order,
    then prunes zero positions and empty short-lot deques and re-marks
    LMV/SMV using post-trade prices (defaulting to pre-trade prices).

    Returns the updated snapshot and an updated `trades_log` (the caller's dict, 
    mutated in place by adding an entry for `date`), which is also returned.
    """
    snap = copy.deepcopy(current_snapshot)

    if shares_to_trade.empty:
        return snap, trades_log

    trades_log[date] = {
        "pct_of_market_volume_traded": pd.Series(
            name = "pct_of_market_volume_traded", dtype = float
        ),
        "trading_fee": 0.0,
        "shares_traded": shares_to_trade.rename("shares_traded")
    }

    prices_today_pre = prices_pre_trade.xs(date, level=0)
    volumes_today = market_volume.xs(date, level=0)
    costs_today = trading_costs.at_date(date)

    gross_exposure_pre_trade = snap.LMV + snap.SMV
    total_trading_fees = 0.0

    for ticker, qty_traded in shares_to_trade.items():
        if qty_traded == 0:
            continue

        if ticker in snap.shares.index:
            shares_pre_trade = int(snap.shares.loc[ticker])
        else:
            shares_pre_trade = 0

        full_spread_bps = float(costs_today.at[ticker, "full_spread_bps"])
        slippage_bps = float(costs_today.at[ticker, "slippage_bps"])
        commission_per_share = float(costs_today.at[ticker, "commission_per_share"])
        price_pre_trade = float(prices_today_pre.at[ticker])

        effect = _build_trade_effect(
            shares_pre_trade = shares_pre_trade,
            shares_traded = int(qty_traded),
            full_spread_bps = full_spread_bps,
            slippage_bps = slippage_bps,
            price_pre_trade = price_pre_trade,
            commission_per_share = commission_per_share
        )
        _apply_trade_effect(snap, ticker, effect)

        total_trading_fees += effect.fee

        volume = float(volumes_today.at[ticker])
        if volume == 0:
            logger.warning(
                "zero market volume for ticker %s at %s  (pct_of_market_volume "
                "set to NaN for this trade)",
                ticker, date.strftime("%Y-%m-%d"),
            )
            trades_log[date]["pct_of_market_volume_traded"].loc[ticker] = float("nan")
        else:
            trades_log[date]["pct_of_market_volume_traded"].loc[ticker] = round(
                (abs(qty_traded) / volume) * 100, 5
            )

    trades_log[date]["trading_fee"] = total_trading_fees

    # compute turnover
    gross_value_traded = (shares_to_trade.abs() * prices_today_pre).sum(skipna=True)
    if gross_exposure_pre_trade > 0:
        trades_log[date]["turnover"] = round(
            float(gross_value_traded / gross_exposure_pre_trade), 4
        )
    else:
        trades_log[date]["turnover"] = float("nan")

    # Prune zero positions and empty short-lot deques
    snap.shares = snap.shares[snap.shares != 0].astype("int64")
    snap.short_proceeds_lots = {
        ticker: lots for ticker, lots in snap.short_proceeds_lots.items() if lots
    }

    # Re-mark to market using post-trade prices (default = pre-trade)
    prices_today_post = (
        prices_post_trade.xs(date, level=0)
        if prices_post_trade is not None
        else prices_today_pre
    )
    LMV, SMV = _compute_LMV_and_SMV(
        snap.shares, prices_today_post.reindex(snap.shares.index)
    )
    snap.LMV = LMV
    snap.SMV = SMV
    snap.update_equity_fields()
    snap.update_leverage_ratios()
 
    return snap, trades_log


# =============================================================================
# Stop-loss threshold
# =============================================================================

def compute_stop_loss_threshold(
    date: pd.Timestamp,
    snapshot: BookSnapshot,
    initial_equity: float,
    pct_initial_equity_triggering_stop_loss: float,
    prices: pd.Series,
    trading_costs: TradingCosts
) -> float:
    """Equity threshold below which the stop-loss fires.

    Threshold = pct x initial_equity + cost to liquidate the current
    book at this date (per the execution-price model + commissions),
    so that even after the liquidation costs the residual equity covers
    pct x initial_equity.
    """
    minimal_equity = pct_initial_equity_triggering_stop_loss * initial_equity

    shares = snapshot.shares
    tickers = shares.index
    prices_today = prices.xs(date, level="date").reindex(tickers)

    costs_today = trading_costs.at_date(date).reindex(tickers)
    slippage = costs_today["slippage_bps"]
    full_spread = costs_today["full_spread_bps"]
    commission = costs_today["commission_per_share"]

    full_liquidation_cost = (
        (commission + prices_today * ((full_spread / 2) + slippage) / 10_000)
        * shares.abs()
    ).sum()

    return float(minimal_equity + full_liquidation_cost)


# =============================================================================
# Display helpers
# =============================================================================

def round_snapshot_for_display(snapshot: BookSnapshot) -> BookSnapshot:
    """Return a deep copy with scalar fields rounded to 2dp.

    Lots are rounded individually; shares are kept as-is (integer by
    construction). For human-readable snapshots only (never use the
    rounded version for further calculation).
    """
    rounded = copy.deepcopy(snapshot)
    rounded.LMV = round(rounded.LMV, 2)
    rounded.SMV = round(rounded.SMV, 2)
    rounded.cash = round(rounded.cash, 2)
    rounded.debit = round(rounded.debit, 2)
    rounded.margin_collateral = round(rounded.margin_collateral, 2)
    rounded.total_short_proceeds = round(rounded.total_short_proceeds, 2)
    rounded.equity = round(rounded.equity, 2)
    rounded.equity_excluding_margin_collateral = round(
        rounded.equity_excluding_margin_collateral, 2
    )
    rounded.long_leverage = round(rounded.long_leverage, 4)
    rounded.short_leverage = round(rounded.short_leverage, 4)
    rounded.gross_leverage = round(rounded.gross_leverage, 4)
    rounded.short_proceeds_lots = {
        ticker: deque((n, round(p, 2)) for n, p in lots)
        for ticker, lots in rounded.short_proceeds_lots.items()
    }
    return rounded