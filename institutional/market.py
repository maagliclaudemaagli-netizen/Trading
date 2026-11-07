"""Closed candles, confirmed structure and an auditable FVG / OB / BB registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

RULES = {
    "M5": "5min", "M15": "15min", "H1": "1h",
    "H4": "4h", "D1": "1D", "W1": "W-MON", "MN": "MS",
}
OHLCV = ["open", "high", "low", "close", "volume"]
AGG = dict(zip(OHLCV, ["first", "max", "min", "last", "sum"]))


def validate_m5(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous timestamps or malformed prices; never fill missing candles."""
    if not frame.columns.is_unique:
        raise ValueError("M5: duplicate column names")
    missing = set(OHLCV) - set(frame.columns)
    if missing:
        raise ValueError(f"M5: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError("M5: no data")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None or index.hasnans or index.has_duplicates:
        raise ValueError("M5: timestamps must be unique, timezone-aware and non-null")
    index = index.tz_convert("UTC")
    if not index.is_monotonic_increasing or not index.equals(index.floor("5min")):
        raise ValueError("M5: timestamps must be sorted and aligned to five minutes")
    if len(index) > 1 and not (index[1:] - index[:-1] == pd.Timedelta(minutes=5)).any():
        raise ValueError("M5: no five-minute spacing; coarse candles cannot be reconstructed")
    out = frame[OHLCV].astype(float).copy()
    out.index = index
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("M5: all OHLCV values must be finite")
    if (out.high < out[["open", "close", "low"]].max(axis=1)).any() or (
        out.low > out[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("M5: inconsistent OHLC bounds")
    if (out[["open", "high", "low", "close"]] <= 0).any().any() or (out.volume < 0).any():
        raise ValueError("M5: prices must be positive and volume nonnegative")
    return out


def closed_bars(m5: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Index bars by availability, using shift(1) on the COMPLETE calendar grid.

    Dropping empty bins before shifting would move Friday's bar to Monday.
    A final boundary is appended so the last fully closed bar is not lost.
    """
    rule = RULES[timeframe]
    grouped = m5.resample(rule, label="left", closed="left").agg(AGG)
    starts = pd.DatetimeIndex(grouped.index)
    grouped["opened_at"] = starts
    boundary = starts[-1] + pd.tseries.frequencies.to_offset(rule)
    shifted = grouped.reindex(starts.append(pd.DatetimeIndex([boundary]))).shift(1)
    shifted = shifted.dropna(subset=["open", "high", "low", "close"])
    source_index = pd.DatetimeIndex(m5.index)
    horizon = source_index[-1] + pd.Timedelta(minutes=5)
    return shifted.loc[
        (shifted.index <= horizon) & (shifted.opened_at >= source_index[0])
    ]


@dataclass(frozen=True)
class Candle:
    timeframe: str
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def intersects(self, bottom: float, top: float) -> bool:
        return self.low <= top and self.high >= bottom


def candles(frame: pd.DataFrame, timeframe: str) -> list[Candle]:
    return [
        Candle(timeframe, pd.Timestamp(start), at, *map(float, values))
        for at, start, values in zip(
            pd.DatetimeIndex(frame.index), frame.opened_at, frame[OHLCV].to_numpy()
        )
    ]


class ZoneState(str, Enum):
    ACTIVE = "active"
    MITIGATED = "mitigated"
    VIOLATED = "violated"


@dataclass
class Zone:
    id: str
    timeframe: str
    kind: str
    side: int
    bottom: float
    top: float
    origin: pd.Timestamp
    created_at: pd.Timestamp
    state: ZoneState = ZoneState.ACTIVE
    violated_at: pd.Timestamp | None = None
    mitigated_at: pd.Timestamp | None = None
    parent_id: str | None = None


@dataclass
class Swing:
    id: str
    side: int  # +1 = high, -1 = low
    price: float
    pivot_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    broken_at: pd.Timestamp | None = None
    taken_at: pd.Timestamp | None = None


@dataclass(frozen=True)
class StructureBreak:
    side: int
    swing: Swing
    at: pd.Timestamp


class MarketStructure:
    """Strict fractals confirmed only after n right-hand candles have closed."""

    def __init__(self, n: int = 2):
        if n < 1:
            raise ValueError("swing_n must be positive")
        self.n = n
        self.history: list[Candle] = []
        self.swings: list[Swing] = []

    def update(self, candle: Candle) -> list[StructureBreak]:
        breaks: list[StructureBreak] = []
        for side in (1, -1):
            unbroken = [s for s in self.swings if s.side == side and s.broken_at is None]
            latest = unbroken[-1] if unbroken else None
            for swing in unbroken:
                if side * (candle.close - swing.price) > 0:
                    swing.broken_at = candle.closed_at
                    if swing is latest:
                        breaks.append(StructureBreak(side, swing, candle.closed_at))
        for swing in self.swings:
            touched = candle.high >= swing.price if swing.side == 1 else candle.low <= swing.price
            if touched and swing.taken_at is None:
                swing.taken_at = candle.closed_at

        self.history.append(candle)
        if len(self.history) >= 2 * self.n + 1:
            window = self.history[-2 * self.n - 1:]
            pivot = window[self.n]
            others = window[:self.n] + window[self.n + 1:]
            for side, valid, price in (
                (1, all(pivot.high > c.high for c in others), pivot.high),
                (-1, all(pivot.low < c.low for c in others), pivot.low),
            ):
                if valid:
                    self.swings.append(Swing(
                        f"{candle.timeframe}:swing:{side}:{pivot.opened_at.isoformat()}",
                        side, price, pivot.opened_at, candle.closed_at,
                    ))
        return breaks


class FVGDetector:
    @staticmethod
    def detect(history: list[Candle]) -> tuple[int, float, float] | None:
        if len(history) < 3:
            return None
        first, last = history[-3], history[-1]
        if last.low > first.high:
            return 1, first.high, last.low
        if last.high < first.low:
            return -1, last.high, first.low
        return None


@dataclass
class MarketEvent:
    candle: Candle
    breaks: list[StructureBreak] = field(default_factory=list)
    created: list[Zone] = field(default_factory=list)
    violated: list[Zone] = field(default_factory=list)
    mitigated: list[Zone] = field(default_factory=list)
    new_swings: list[Swing] = field(default_factory=list)


class ZoneTracker:
    """Keep identities and terminal violations; mitigated zones remain eligible."""

    def __init__(self, timeframe: str, swing_n: int = 2):
        self.timeframe = timeframe
        self.structure = MarketStructure(swing_n)
        self.zones: dict[str, Zone] = {}
        self.live: dict[str, Zone] = {}

    def mitigate(self, candle: Candle) -> list[Zone]:
        changed = []
        for zone in self.live.values():
            if zone.state == ZoneState.ACTIVE and zone.created_at <= candle.opened_at and candle.intersects(
                zone.bottom, zone.top
            ):
                zone.state, zone.mitigated_at = ZoneState.MITIGATED, candle.closed_at
                changed.append(zone)
        return changed

    def _create(
        self, kind: str, side: int, bottom: float, top: float,
        origin: pd.Timestamp, at: pd.Timestamp, parent: str | None = None,
    ) -> Zone | None:
        identity = f"{self.timeframe}:{kind}:{side}:{origin.isoformat()}"
        if identity in self.zones:
            return None
        zone = Zone(identity, self.timeframe, kind, side, bottom, top, origin, at, parent_id=parent)
        self.zones[identity] = self.live[identity] = zone
        return zone

    def update(self, candle: Candle) -> MarketEvent:
        event = MarketEvent(candle)
        for zone in list(self.live.values()):
            violated = candle.close < zone.bottom if zone.side == 1 else candle.close > zone.top
            if violated:
                zone.state, zone.violated_at = ZoneState.VIOLATED, candle.closed_at
                event.violated.append(zone)
                del self.live[zone.id]
                if zone.kind == "OB":
                    breaker = self._create(
                        "BB", -zone.side, zone.bottom, zone.top,
                        zone.origin, candle.closed_at, zone.id,
                    )
                    if breaker is not None:
                        event.created.append(breaker)
        event.mitigated = self.mitigate(candle)
        previous_count = len(self.structure.swings)
        event.breaks = self.structure.update(candle)
        event.new_swings = self.structure.swings[previous_count:]
        history = self.structure.history
        gap = FVGDetector.detect(history)
        if gap is not None:
            side, bottom, top = gap
            zone = self._create("FVG", side, bottom, top, candle.opened_at, candle.closed_at)
            if zone is not None:
                event.created.append(zone)
        for broken in event.breaks:
            for prior in reversed(history[:-1]):
                if prior.opened_at < broken.swing.pivot_at:
                    break
                if broken.side * (prior.close - prior.open) < 0:
                    zone = self._create(
                        "OB", broken.side, prior.low, prior.high, prior.opened_at, candle.closed_at,
                    )
                    if zone is not None:
                        event.created.append(zone)
                    break
        return event
