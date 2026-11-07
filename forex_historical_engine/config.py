from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

MAJORS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
MINORS = (
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD", "GBPJPY",
    "GBPCHF", "GBPAUD", "GBPCAD", "AUDJPY", "AUDNZD", "AUDCAD", "CADJPY",
    "CHFJPY", "NZDJPY",
)
RULES = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D", "W1": "W-MON", "MN1": "MS",
}
NEW_YORK = ZoneInfo("America/New_York")


def instrument_code(value: str) -> str:
    pair = value.upper().replace("/", "")
    if len(pair) != 6 or not pair.isascii() or not pair.isalpha():
        raise ValueError("Instrument attendu: six lettres, par exemple EURUSD.")
    return pair


def instrument(value: str, price_scale: int | None = None) -> tuple[str, int]:
    pair = instrument_code(value)
    if price_scale is None:
        if pair not in MAJORS + MINORS:
            raise ValueError("Instrument non configure: fournir --price-scale explicitement.")
        price_scale = 1000 if pair.endswith("JPY") else 100000
    if price_scale <= 0:
        raise ValueError("price_scale doit etre strictement positif.")
    return pair, price_scale


def utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Date invalide.")
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def date_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = utc(start), utc(end)
    if first != first.normalize() or last != last.normalize() or first > last:
        raise ValueError("Dates UTC de jour entier attendues, debut <= fin.")
    if last >= pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("La date de fin doit preceder aujourd'hui: archives quotidiennes closes uniquement.")
    return first, last


def market_open(index: pd.DatetimeIndex, holidays: tuple[str, ...] = ()) -> pd.Series:
    local = index.tz_convert(NEW_YORK)
    weekday, hour = local.day_of_week, local.hour
    opened = ((weekday < 4) | ((weekday == 4) & (hour < 17)) | ((weekday == 6) & (hour >= 17)))
    opened &= ~index.strftime("%Y-%m-%d").isin(holidays)
    return pd.Series(opened, index=index)


@dataclass(frozen=True)
class Settings:
    root: Path = Path("data")
    workers: int = 2
    delay: float = 1.0
    retries: int = 3
    timeout: float = 30
    holidays: tuple[str, ...] = ()
    cache: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())
        if self.cache is not None:
            object.__setattr__(self, "cache", self.cache.expanduser().resolve())
        if self.workers < 1 or self.workers > 8:
            raise ValueError("workers doit etre compris entre 1 et 8.")
        if not math.isfinite(self.delay) or not math.isfinite(self.timeout) or (
            self.delay < 1.0 or not 0 <= self.retries <= 3 or self.timeout <= 0
        ):
            raise ValueError("Delai >= 1.0, 0 <= retries <= 3 et timeout > 0 finis requis.")

    @property
    def cache_root(self) -> Path:
        return self.cache if self.cache is not None else self.root / "cache"
