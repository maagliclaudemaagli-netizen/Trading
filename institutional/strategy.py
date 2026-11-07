"""Institutional protocol: macro tap -> BOS -> two violations/two creations -> close."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .market import Candle, MarketEvent, StructureBreak, Swing, Zone, ZoneState, ZoneTracker

ALIGNMENTS = {
    "intraday": ("D1", "H1", "H4"),
    "weekly": ("W1", "H4", "D1"),
    "swing": ("MN", "D1", "W1"),
}


@dataclass(frozen=True)
class StrategyConfig:
    alignment: str = "intraday"
    swing_n: int = 2

    def __post_init__(self) -> None:
        if self.alignment not in ALIGNMENTS:
            raise ValueError("Unknown fractal alignment")
        if self.swing_n < 1:
            raise ValueError("swing_n must be positive")

    @property
    def timeframes(self) -> set[str]:
        return {"H1", "H4", "D1", *ALIGNMENTS[self.alignment]}


@dataclass
class Setup:
    side: int
    started_at: pd.Timestamp
    macro_zone: Zone
    impulse_extreme: float
    opposing_ids: set[str]
    violated_ids: set[str] = field(default_factory=set)
    created_ids: list[str] = field(default_factory=list)
    bos: StructureBreak | None = None


@dataclass(frozen=True)
class Signal:
    side: int
    at: pd.Timestamp
    zone_id: str
    macro_id: str
    entry_level: float
    structural_stop: float
    target: float
    target_id: str
    violated_ids: tuple[str, ...]
    created_ids: tuple[str, ...]
    key_level: float


class InstitutionalStrategy:
    """Closed higher-timeframe information and M5 observations, no DBS/OTE/sweep gate."""

    def __init__(self, config: StrategyConfig = StrategyConfig()):
        self.config = config
        self.macro_tf, self.confirm_tf, self.validation_tf = ALIGNMENTS[config.alignment]
        self.trackers = {tf: ZoneTracker(tf, config.swing_n) for tf in sorted(config.timeframes)}
        self.setups: dict[int, Setup] = {}
        self.audit: list[dict] = []
        self.signals: list[Signal] = []
        self.external_available: dict[str, Swing] = {}
        self.external_consumed: set[str] = set()

    def log(self, at: pd.Timestamp, event: str, **details: object) -> None:
        self.audit.append({"time": at, "event": event, **details})

    def zone(self, identity: str) -> Zone:
        return self.trackers[identity.split(":")[0]].zones[identity]

    def observe_price(self, candle: Candle, busy: bool = False) -> None:
        """Observe a tap of an already-known macro zone without waiting for Daily close."""
        for identity, swing in list(self.external_available.items()):
            touched = candle.high >= swing.price if swing.side == 1 else candle.low <= swing.price
            if touched:
                self.external_consumed.add(identity)
                del self.external_available[identity]
                self.log(candle.closed_at, "external_liquidity_taken", swing=identity)
        for tracker in self.trackers.values():
            for zone in tracker.mitigate(candle):
                self.log(candle.closed_at, "zone_mitigated", zone=zone.id, timeframe=zone.timeframe)
        if busy:
            return
        for setup in self.setups.values():
            # Freeze the origin of the impulse when the second opposing zone breaks.
            if len(setup.violated_ids) < 2:
                setup.impulse_extreme = (
                    min(setup.impulse_extreme, candle.low) if setup.side == 1
                    else max(setup.impulse_extreme, candle.high)
                )
        for side in (1, -1):
            if side in self.setups:
                continue
            tapped = [
                z for z in self.trackers[self.macro_tf].live.values()
                if z.side == side and z.created_at <= candle.opened_at
                and candle.intersects(z.bottom, z.top)
            ]
            if not tapped:
                continue
            zone = max(tapped, key=lambda z: (z.created_at, z.id))
            opposing = {
                z.id for z in self.trackers[self.confirm_tf].live.values()
                if z.side == -side and z.created_at < candle.closed_at
            }
            extreme = candle.low if side == 1 else candle.high
            self.setups[side] = Setup(side, candle.closed_at, zone, extreme, opposing)
            self.log(candle.closed_at, "internal_tap", side=side, zone=zone.id)

    def on_close(self, batch: list[Candle], busy: bool = False) -> Signal | None:
        if not batch or any(c.closed_at != batch[0].closed_at for c in batch):
            raise ValueError("A batch must contain candles sharing one close time")
        events = {c.timeframe: self.trackers[c.timeframe].update(c) for c in batch}
        at = batch[0].closed_at
        for tf, event in events.items():
            if tf in (self.macro_tf, self.validation_tf):
                self.external_available.update((s.id, s) for s in event.new_swings)
            for name, zones in (
                ("zone_created", event.created), ("zone_violated", event.violated),
                ("zone_mitigated", event.mitigated),
            ):
                for zone in zones:
                    self.log(at, name, timeframe=tf, zone=zone.id, kind=zone.kind)
        if busy:
            return None
        for side, setup in list(self.setups.items()):
            if setup.macro_zone.state == ZoneState.VIOLATED:
                self.log(at, "setup_cancelled", side=side, reason="macro zone violated")
                del self.setups[side]
                continue
            confirmation = events.get(self.confirm_tf)
            if confirmation is not None and confirmation.candle.closed_at > setup.started_at:
                if not self._confirm_structure(confirmation, setup):
                    del self.setups[side]
                    continue
            validation = events.get(self.validation_tf)
            if validation is not None:
                signal = self._validate(validation.candle, setup)
                if signal is not None:
                    self.setups.clear()
                    return signal
        return None

    def _confirm_structure(self, event: MarketEvent, setup: Setup) -> bool:
        for broken in event.breaks:
            if broken.side != setup.side:
                self.log(broken.at, "setup_cancelled", side=setup.side, reason="opposite BOS")
                return False
            if setup.bos is None:
                setup.bos = broken
                self.log(broken.at, "bos", side=setup.side, swing=broken.swing.id)
        if setup.bos is None:
            return True
        old_counts = len(setup.violated_ids), len(setup.created_ids)
        setup.violated_ids.update(z.id for z in event.violated if z.id in setup.opposing_ids)
        for zone in event.created:
            if zone.side == setup.side and zone.id not in setup.created_ids:
                setup.created_ids.append(zone.id)
        if old_counts != (len(setup.violated_ids), len(setup.created_ids)):
            self.log(event.candle.closed_at, "zone_counts", side=setup.side,
                     invalidated=len(setup.violated_ids), created=len(setup.created_ids))
        return True

    def _validate(self, candle: Candle, setup: Setup) -> Signal | None:
        if setup.bos is None or len(setup.violated_ids) < 2 or len(setup.created_ids) < 2:
            return None
        side, key = setup.side, setup.bos.swing.price
        if candle.closed_at < setup.bos.at:
            return None
        if side * (candle.close - key) <= 0 or side * (candle.close - candle.open) <= 0:
            return None
        targets = [
            swing
            for tf in (self.macro_tf, self.validation_tf)
            for swing in self.trackers[tf].structure.swings
            if swing.side == side and swing.confirmed_at < setup.started_at
            and swing.taken_at is None and swing.id not in self.external_consumed
            and side * (swing.price - candle.close) > 0
        ]
        if not targets:
            self.log(candle.closed_at, "validation_rejected", side=side, reason="no external liquidity")
            return None
        target = min(targets, key=lambda s: (abs(s.price - candle.close), s.id))
        available = [
            self.zone(identity) for identity in setup.created_ids
            if self.zone(identity).state != ZoneState.VIOLATED
            and side * (candle.close - (self.zone(identity).bottom + self.zone(identity).top) / 2) > 0
        ]
        if not available:
            self.log(candle.closed_at, "validation_rejected", side=side, reason="no surviving retracement zone")
            return None
        zone = max(available, key=lambda z: (z.created_at, z.id))
        entry = (zone.bottom + zone.top) / 2
        stop = setup.impulse_extreme
        if side * (entry - stop) <= 0 or side * (target.price - entry) <= 0:
            self.log(candle.closed_at, "validation_rejected", side=side, reason="invalid price ordering")
            return None
        signal = Signal(
            side, candle.closed_at, zone.id, setup.macro_zone.id, entry, stop, target.price,
            target.id, tuple(sorted(setup.violated_ids)), tuple(setup.created_ids), key,
        )
        self.signals.append(signal)
        self.log(signal.at, "confirmed", side=side, invalidated=len(signal.violated_ids),
                 created=len(signal.created_ids), entry=entry, stop=stop, target=signal.target,
                 violated_ids=signal.violated_ids, created_ids=signal.created_ids)
        return signal
