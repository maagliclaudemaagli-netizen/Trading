from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import RULES, market_open, utc

OHLC = ["open", "high", "low", "close"]


def _source_urls(values: pd.Series, nested: bool) -> str:
    urls: set[str] = set()
    for value in values:
        items = json.loads(value) if nested else [str(value)]
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError("Provenance source invalide.")
        urls.update(items)
    return json.dumps(sorted(urls))


def timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(frame.timestamp)
    if index.tz is None or index.hasnans:
        raise ValueError("Timestamps non nuls avec fuseau horaire requis.")
    return index.tz_convert("UTC")


def build_candles(frame: pd.DataFrame, timeframe: str, from_ticks: bool = False) -> pd.DataFrame:
    if timeframe not in RULES or frame.empty:
        raise ValueError("Timeframe valide et donnees non vides requis.")
    index = timestamps(frame)
    if not index.is_monotonic_increasing:
        raise ValueError("Timestamps des donnees source non ordonnes.")
    if not from_ticks and frame.duplicated(["timestamp", "side"]).any():
        raise ValueError("Bougies source dupliquees: reconstruction ambigue.")
    rule = RULES[timeframe]
    data = frame.set_index(index)
    results = []
    for side in ("bid", "ask", "mid") if from_ticks else ("bid", "ask"):
        if from_ticks:
            price = (data.bid + data.ask) / 2 if side == "mid" else data[side]
            bars = price.resample(rule, label="left", closed="left").ohlc()
            bars["source_count"] = price.resample(rule, label="left", closed="left").count()
            # Quote sizes are not executed market volume; don't sum tick quotes.
            bars["volume"] = np.nan
            volume_kind = "not_available"
            selected = data
        else:
            selected = data.loc[data.side == side]
            if selected.empty:
                continue
            bars = selected.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last",
                 "volume": lambda values: values.sum(min_count=1)}
            )
            bars["source_count"] = selected.open.resample(rule, label="left", closed="left").count()
            volume_kind = str(selected.volume_kind.iloc[0])
        bars = bars.dropna(subset=OHLC)
        if bars.empty:
            continue
        bars["timestamp"] = bars.index
        bars["closed_at"] = bars.index + pd.tseries.frequencies.to_offset(rule)
        first = selected.timestamp.resample(rule, label="left", closed="left").min()
        last = selected.timestamp.resample(rule, label="left", closed="left").max()
        bars["first_source_timestamp"] = first.reindex(bars.index)
        bars["last_source_timestamp"] = last.reindex(bars.index)
        bars["side"] = side
        bars["timeframe"] = timeframe
        bars["instrument"] = str(frame.instrument.iloc[0])
        bars["source"] = "Dukascopy"
        bars["source_url"] = "derived:tick" if from_ticks else "derived:M1"
        bars["source_record"] = np.arange(len(bars))
        bars["reconstructed"] = True
        bars["volume_kind"] = volume_kind
        bars["input_timeframe"] = "tick" if from_ticks else "M1"
        provenance_column = "input_sources" if "input_sources" in selected else "source_url"
        provenance = selected[provenance_column].resample(rule, label="left", closed="left").agg(
            lambda values: _source_urls(values, provenance_column == "input_sources")
        )
        bars["input_sources"] = provenance.reindex(bars.index)
        bars["mid_method"] = "(bid + ask) / 2 per tick" if side == "mid" else ""
        bars["completeness"] = "not_certified; consult input quality report"
        results.append(bars.reset_index(drop=True))
    if not results:
        raise ValueError("Aucune bougie reconstructible.")
    return pd.concat(results, ignore_index=True).sort_values(["timestamp", "side"], kind="stable")


def quality(frame: pd.DataFrame, timeframe: str) -> dict:
    index = timestamps(frame)
    tick = timeframe == "tick"
    prices = ["bid", "ask"] if tick else OHLC
    values = frame[prices].to_numpy(dtype=float)
    invalid = ~np.isfinite(values).all(axis=1) | (values <= 0).any(axis=1)
    flags = {
        "invalid_prices": int(invalid.sum()),
        "unordered_timestamps": int((index[1:] < index[:-1]).sum()),
        "duplicate_source_records": int(
            (frame if tick else frame.loc[~frame.reconstructed]).duplicated(["source_url", "source_record"]).sum()
        ),
    }
    volumes = ["bid_volume", "ask_volume"] if tick else ["volume"]
    volume_values = frame[volumes].to_numpy(dtype=float)
    flags["negative_volume_candidates"] = int((volume_values < 0).any(axis=1).sum())
    bad = pd.Series(invalid, index=frame.index)
    if tick:
        crossed = frame.bid > frame.ask
        flags["negative_spreads"] = int(crossed.sum())
        flags["identical_tick_candidates"] = int(frame.duplicated(
            ["timestamp", "bid_raw", "ask_raw", "bid_volume", "ask_volume"]
        ).sum())
        flags["simultaneous_distinct_ticks"] = int(frame.timestamp.duplicated(keep=False).sum())
        bad |= crossed
    else:
        bounds = (
            (frame.high < frame[["open", "close", "low"]].max(axis=1))
            | (frame.low > frame[["open", "close", "high"]].min(axis=1))
        )
        duplicates = frame.duplicated(["timestamp", "side"])
        flags["invalid_ohlc"] = int(bounds.sum())
        flags["duplicate_candles"] = int(duplicates.sum())
        base_timeframe = timeframe.removesuffix("_from_ticks")
        if base_timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
            flags["misaligned_candles"] = int((index != index.floor(RULES[base_timeframe])).sum())
        elif base_timeframe == "W1":
            flags["misaligned_candles"] = int(((index.day_of_week != 0) | (index != index.normalize())).sum())
        elif base_timeframe == "MN1":
            flags["misaligned_candles"] = int(((index.day != 1) | (index != index.normalize())).sum())
        bad |= bounds | duplicates
        aligned = frame.pivot_table(index="timestamp", columns="side", values=OHLC, aggfunc="first")
        crossed_count = 0
        if {"bid", "ask"}.issubset(frame.side.unique()):
            crossed = aligned.xs("bid", axis=1, level=1) > aligned.xs("ask", axis=1, level=1)
            crossed_count = int(crossed.to_numpy().any(axis=1).sum())
        flags["crossed_bid_ask_candle_candidates"] = crossed_count
    quote = "bid" if tick else "close"
    series = frame[quote] if tick else frame.loc[frame.side == "bid", quote]
    change = series.pct_change(fill_method=None).abs()
    flags["large_price_jump_candidates_1pct"] = int((change > 0.01).sum())
    flags["non_utc_input"] = int(str(pd.DatetimeIndex(frame.timestamp).tz) != "UTC")
    return {"rows": len(frame), "bad_rows": int(bad.sum()), "flags": flags}


def clean(frame: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit opt-in filtering; caller must save the audit separately."""
    timestamps(frame)
    fields = ["bid", "ask"] if timeframe == "tick" else OHLC
    numeric = frame[fields].to_numpy(dtype=float)
    bad = ~np.isfinite(numeric).all(axis=1) | (numeric <= 0).any(axis=1)
    reasons = pd.Series("", index=frame.index)
    reasons.loc[bad] = "nonfinite_or_nonpositive_price"
    if timeframe == "tick":
        crossed = frame.bid > frame.ask
        reasons.loc[crossed] += ";bid_above_ask"
    else:
        crossed = (
            (frame.high < frame[["open", "close", "low"]].max(axis=1))
            | (frame.low > frame[["open", "close", "high"]].min(axis=1))
        )
        reasons.loc[crossed] += ";invalid_ohlc"
    rejected = reasons != ""
    audit = frame.loc[rejected, ["timestamp", "source_url", "source_record"]].copy()
    audit["reason"] = reasons.loc[rejected]
    return frame.loc[~rejected].copy(), audit


def expected_minutes(start: pd.Timestamp, end: pd.Timestamp, holidays: tuple[str, ...] = ()) -> pd.DatetimeIndex:
    index = pd.date_range(utc(start), utc(end), freq="min", inclusive="left")
    return index[market_open(index, holidays).to_numpy()]
