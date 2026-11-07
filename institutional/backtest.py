"""M5 bid/ask execution, structural stops and backtest performance."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import Candle, ZoneState, candles, closed_bars, validate_m5
from .strategy import InstitutionalStrategy, Signal, StrategyConfig

TRADE_COLUMNS = [
    "side", "signal_time", "entry_time", "exit_time", "entry", "exit", "sl", "tp",
    "risk", "R", "reason", "zone_id", "target_id", "invalidated", "created",
]


@dataclass(frozen=True)
class ExecutionConfig:
    pair: str = "EURUSD"
    spread_pips: float = 0.6
    risk_fraction: float = 0.01
    capital: float = 10_000.0
    price_basis: str = "mid"

    def __post_init__(self) -> None:
        if not np.isfinite([self.spread_pips, self.risk_fraction, self.capital]).all():
            raise ValueError("Execution parameters must be finite")
        if self.spread_pips < 0 or not 0 < self.risk_fraction < 1 or self.capital <= 0:
            raise ValueError("Invalid spread, risk fraction or capital")
        if self.price_basis not in ("mid", "bid"):
            raise ValueError("price_basis must be mid or bid")

    @property
    def half_spread(self) -> float:
        pip = 0.01 if self.pair.upper().endswith("JPY") else 0.0001
        return self.spread_pips * pip / 2


@dataclass
class Position:
    signal: Signal
    entry_at: pd.Timestamp
    entry: float
    stop: float
    risk: float
    stake: float


class ExecutionEngine:
    """One position, no forced final close, stop-first when OHLC is ambiguous."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.pending: Signal | None = None
        self.position: Position | None = None
        self.equity = config.capital
        self.equity_curve = [config.capital]
        self.trades: list[dict] = []
        self.audit: list[dict] = []
        self.bankrupt = False

    @property
    def busy(self) -> bool:
        return self.pending is not None or self.position is not None or self.bankrupt

    def cancel(self, at: pd.Timestamp, reason: str) -> None:
        self.audit.append({"time": at, "event": "order_cancelled", "reason": reason})
        self.pending = None

    def on_bar(self, bar: Candle) -> None:
        hs = self.config.half_spread
        filled_at_open = False
        just_entered = False
        signal = self.pending
        if signal is not None:
            if bar.opened_at < signal.at:
                raise ValueError("An order cannot precede its confirmation")
            side = signal.side
            if side * (bar.open - signal.target) >= 0:
                self.cancel(bar.opened_at, "external liquidity reached before entry")
            else:
                filled_at_open = side * (bar.open - signal.entry_level) <= 0
                touched = bar.low <= signal.entry_level if side == 1 else bar.high >= signal.entry_level
                if filled_at_open or touched:
                    mid_fill = bar.open if filled_at_open else signal.entry_level
                    entry = mid_fill + side * hs
                    stop = signal.structural_stop - side * hs
                    planned_entry = signal.entry_level + side * hs
                    risk = side * (planned_entry - stop)
                    if risk <= 0:
                        raise ValueError("Order has a nonpositive planned risk")
                    self.position = Position(
                        signal, bar.opened_at, entry, stop, risk,
                        self.equity * self.config.risk_fraction,
                    )
                    self.pending = None
                    just_entered = True
                    self.audit.append({"time": bar.opened_at, "event": "entry", "side": side})
                elif (bar.high >= signal.target if side == 1 else bar.low <= signal.target):
                    self.cancel(bar.closed_at, "external liquidity reached without retracement")

        position = self.position
        if position is not None:
            side = position.signal.side
            quote_open = bar.open - side * hs
            adverse = bar.low - hs if side == 1 else bar.high + hs
            favorable = bar.high - hs if side == 1 else bar.low + hs
            stop_hit = side * (adverse - position.stop) <= 0
            target_hit = side * (favorable - position.signal.target) >= 0
            exit_price: float | None = None
            reason = ""
            if stop_hit:
                # On an intrabar limit fill the earlier opening price is not an exit.
                exit_price = (
                    min(quote_open, position.stop) if side == 1 else max(quote_open, position.stop)
                ) if not just_entered or filled_at_open else position.stop
                reason = "stop_loss"
            elif target_hit and (not just_entered or filled_at_open):
                exit_price, reason = position.signal.target, "take_profit"
            if exit_price is not None:
                r = side * (exit_price - position.entry) / position.risk
                self.equity += position.stake * r
                s = position.signal
                self.trades.append(dict(
                    side=side, signal_time=s.at, entry_time=position.entry_at,
                    exit_time=bar.closed_at, entry=position.entry, exit=exit_price,
                    sl=position.stop, tp=s.target, risk=position.risk, R=r, reason=reason,
                    zone_id=s.zone_id, target_id=s.target_id,
                    invalidated=len(s.violated_ids), created=len(s.created_ids),
                ))
                self.position = None
                if self.equity <= 0:
                    self.bankrupt = True
                    self.audit.append({"time": bar.closed_at, "event": "bankrupt"})
        marked = self.equity
        if self.position is not None:
            p = self.position
            liquidation = bar.close - p.signal.side * hs
            marked += p.stake * p.signal.side * (liquidation - p.entry) / p.risk
        self.equity_curve.append(marked)


def performance(
    results: np.ndarray, risk_fraction: float = 0.01, capital: float = 10_000,
) -> dict:
    """Closed-trade drawdown includes starting equity, including the first loss."""
    r = np.asarray(results, dtype=float)
    if r.size == 0:
        return dict(trades=0, expectancy_r=None, win_rate_pct=None, profit_factor=None,
                    max_drawdown_pct=0.0, final_equity=capital, total_r=0.0)
    equity = np.concatenate(([capital], capital * np.cumprod(1 + risk_fraction * r)))
    loss = -r[r < 0].sum()
    return dict(
        trades=int(r.size), expectancy_r=float(r.mean()), win_rate_pct=float(100 * (r > 0).mean()),
        profit_factor=float(r[r > 0].sum() / loss) if loss else (
            "Infinity" if (r > 0).any() else None
        ),
        max_drawdown_pct=float(100 * (1 - equity / np.maximum.accumulate(equity)).max()),
        final_equity=float(equity[-1]), total_r=float(r.sum()),
    )


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    audit: pd.DataFrame
    signals: list[Signal]
    metrics: dict


def backtest(
    m5: pd.DataFrame, strategy: StrategyConfig = StrategyConfig(),
    execution: ExecutionConfig = ExecutionConfig(),
) -> BacktestResult:
    m5 = validate_m5(m5)
    if execution.price_basis == "bid":
        m5 = m5.copy()
        m5[["open", "high", "low", "close"]] += execution.half_spread
    engine, broker = InstitutionalStrategy(strategy), ExecutionEngine(execution)
    frames = {tf: closed_bars(m5, tf) for tf in sorted(strategy.timeframes | {"M15"})}
    schedule: dict[pd.Timestamp, list[Candle]] = {}
    for tf in sorted(strategy.timeframes):
        for candle in candles(frames[tf], tf):
            schedule.setdefault(candle.closed_at, []).append(candle)
    batches = iter(sorted(schedule.items()))
    upcoming = next(batches, None)
    for at, row in zip(pd.DatetimeIndex(m5.index), m5.to_numpy()):
        while upcoming is not None and upcoming[0] <= at:
            _, batch = upcoming
            signal = engine.on_close(batch, busy=broker.busy)
            if broker.pending is not None and any(
                engine.zone(identity).state == ZoneState.VIOLATED
                for identity in (broker.pending.zone_id, broker.pending.macro_id)
            ):
                broker.cancel(upcoming[0], "entry or macro zone invalidated on a closed candle")
            if signal is not None:
                broker.pending = signal
            upcoming = next(batches, None)
        bar = Candle("M5", at, at + pd.Timedelta(minutes=5), *map(float, row))
        was_busy = broker.busy
        broker.on_bar(bar)
        engine.observe_price(bar, busy=was_busy or broker.busy)
    # The final M5 close can confirm a setup, but there is no future bar to fill it.
    while upcoming is not None:
        signal = engine.on_close(upcoming[1], busy=broker.busy)
        if broker.pending is not None and any(
            engine.zone(identity).state == ZoneState.VIOLATED
            for identity in (broker.pending.zone_id, broker.pending.macro_id)
        ):
            broker.cancel(upcoming[0], "entry or macro zone invalidated on a closed candle")
        if signal is not None:
            broker.pending = signal
        upcoming = next(batches, None)
    trades = pd.DataFrame(broker.trades, columns=TRADE_COLUMNS)
    metrics = performance(trades.R.to_numpy(), execution.risk_fraction, execution.capital)
    eq = np.asarray(broker.equity_curve)
    metrics.update(
        mtm_max_drawdown_pct=float(100 * (1 - eq / np.maximum.accumulate(eq)).max()),
        open_positions=int(broker.position is not None), pending_orders=int(broker.pending is not None),
        confirmed_signals=len(engine.signals), bankrupt=broker.bankrupt,
        resampled_bars={tf: len(frame) for tf, frame in frames.items()},
    )
    audit = pd.DataFrame(engine.audit + broker.audit)
    if not audit.empty:
        audit = audit.sort_values("time", kind="stable").reset_index(drop=True)
    return BacktestResult(trades, audit, engine.signals, metrics)


def data_fingerprint(m5: pd.DataFrame) -> str:
    """Stable hash of UTC nanosecond timestamps and ordered float64 OHLCV values."""
    data = validate_m5(m5)
    digest = hashlib.sha256()
    digest.update(pd.DatetimeIndex(data.index).as_unit("ns").asi8.astype("<i8").tobytes())
    digest.update(data.to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()
