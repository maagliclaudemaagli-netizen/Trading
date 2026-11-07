from __future__ import annotations

from collections import Counter
from contextlib import closing
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from .config import RULES
from .processing import expected_minutes, quality
from .storage import Storage


def spread_distribution(connection: sqlite3.Connection, first_hour: int, last_hour: int) -> dict:
    count, mean, minimum, maximum, square_mean = connection.execute(
        "SELECT COUNT(*),AVG(value),MIN(value),MAX(value),AVG(value*value) FROM spreads WHERE hour BETWEEN ? AND ?",
        (first_hour, last_hour),
    ).fetchone()
    median = None
    if count:
        middle = connection.execute(
            "SELECT value FROM spreads WHERE hour BETWEEN ? AND ? ORDER BY value LIMIT ? OFFSET ?",
            (first_hour, last_hour, 1 if count % 2 else 2, (count - 1) // 2),
        ).fetchall()
        median = sum(value[0] for value in middle) / len(middle)
    return {
        "observations": count, "mean": mean, "median": median, "minimum": minimum, "maximum": maximum,
        "std": float(np.sqrt(max(0.0, square_mean - mean * mean))) if count else None,
    }


def report(storage: Storage, pair: str, timeframe: str, holidays: tuple[str, ...] = ()) -> dict:
    with tempfile.TemporaryDirectory(prefix="dukascopy-statistics-") as directory:
        with storage.connect() as metadata:
            suffix = "%h_ticks.bi5" if timeframe == "tick" or timeframe.endswith("_from_ticks") else "%candles_min_1.bi5"
            requested = metadata.execute(
                "SELECT MIN(period), MAX(period) FROM downloads WHERE instrument=? AND url LIKE ?", (pair, suffix),
            ).fetchone()
        with closing(sqlite3.connect(f"{directory}/spreads.sqlite")) as statistics:
            statistics.execute("PRAGMA cache_size=-4096")
            statistics.execute("PRAGMA temp_store=FILE")
            statistics.execute("CREATE TABLE spreads (value REAL NOT NULL, hour INTEGER NOT NULL)")
            return _report(storage, pair, timeframe, holidays, statistics, requested)


def _report(
    storage: Storage, pair: str, timeframe: str, holidays: tuple[str, ...],
    statistics: sqlite3.Connection, requested: tuple,
) -> dict:
    files = storage.files(pair, timeframe)
    if not files:
        raise ValueError(f"Aucune donnee locale pour {pair}/{timeframe}.")
    flags: Counter = Counter()
    rows = bad = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    days: set[str] = set()
    spread_count = 0
    spread_sum = spread_min = spread_max = 0.0
    amplitude_sum = 0.0
    amplitude_count = 0
    return_sum = return_squares = 0.0
    return_count = 0
    previous_price: float | None = None
    covered = expected = 0
    missing_minutes = missing_hours = missing_bars = 0
    missing_days: set[str] = set()
    processed_months: set[str] = set()
    base_timeframe = timeframe.removesuffix("_from_ticks")
    for _, frame in storage.partitions(pair, timeframe):
        stats = quality(frame, timeframe)
        rows += stats["rows"]
        bad += stats["bad_rows"]
        flags.update(stats["flags"])
        dates = pd.DatetimeIndex(frame.timestamp)
        processed_months.add(dates.min().strftime("%Y-%m"))
        first = dates.min() if first is None else min(first, dates.min())
        last = dates.max() if last is None else max(last, dates.max())
        days.update(dates.strftime("%Y-%m-%d"))
        if timeframe == "tick":
            spreads = pd.Series((frame.ask - frame.bid).to_numpy(), index=dates)
            prices = frame.bid
        else:
            quotes = frame.pivot_table(index="timestamp", columns="side", values="close", aggfunc="first")
            spreads = quotes.ask - quotes.bid if {"bid", "ask"}.issubset(quotes.columns) else pd.Series(dtype=float)
            prices = frame.loc[frame.side == "bid", "close"]
            amplitude = frame.high - frame.low
            valid_amplitude = amplitude[np.isfinite(amplitude)]
            amplitude_sum += float(valid_amplitude.sum())
            amplitude_count += len(valid_amplitude)
        spreads = spreads[np.isfinite(spreads)]
        if len(spreads):
            minimum, maximum = float(spreads.min()), float(spreads.max())
            spread_min = minimum if not spread_count else min(spread_min, minimum)
            spread_max = maximum if not spread_count else max(spread_max, maximum)
            spread_count += len(spreads)
            spread_sum += float(spreads.sum())
            statistics.executemany(
                "INSERT INTO spreads VALUES (?,?)",
                ((float(value), int(hour)) for value, hour in zip(spreads.to_numpy(), pd.DatetimeIndex(spreads.index).hour)),
            )
        valid_prices = prices[np.isfinite(prices) & (prices > 0)].to_numpy(dtype=float)
        if len(valid_prices):
            values = np.insert(valid_prices, 0, previous_price) if previous_price is not None else valid_prices
            returns = np.diff(np.log(values))
            return_count += len(returns)
            return_sum += float(returns.sum())
            return_squares += float(np.square(returns).sum())
            previous_price = float(valid_prices[-1])
        # Consume coverage now; only compact summaries survive this iteration.
        if timeframe == "tick":
            start = pd.Timestamp(frame.origin_utc.iloc[0])
            end = start + pd.Timedelta(hours=1)
        else:
            start = dates.min().normalize()
            end = dates.max().normalize() + pd.Timedelta(days=1)
            if requested[0] is not None:
                month = pd.Timestamp(f"{dates.min().strftime('%Y-%m')}-01", tz="UTC")
                start = max(month, pd.Timestamp(requested[0]).normalize())
                calendar_end = month + pd.offsets.MonthBegin(1)
                if base_timeframe == "W1":
                    calendar_end += pd.Timedelta(days=7)
                end = min(calendar_end, pd.Timestamp(requested[1]).normalize() + pd.Timedelta(days=1))
            if end <= start:
                continue
        if base_timeframe in ("tick", "M1"):
            schedule = expected_minutes(start, end, holidays)
            actual = dates.floor("min").unique()
            if base_timeframe == "M1":
                for side in ("bid", "ask"):
                    actual_side = pd.DatetimeIndex(frame.loc[frame.side == side, "timestamp"])
                    available = schedule.intersection(actual_side)
                    expected += len(schedule)
                    covered += len(available)
                    missing = schedule.difference(actual_side)
                    missing_minutes += len(missing)
                    missing_bars += len(missing)
                    missing_hours += len(schedule.floor("h").unique().difference(actual_side.floor("h")))
                    missing_days.update(schedule.strftime("%Y-%m-%d").unique().difference(actual_side.strftime("%Y-%m-%d").unique()))
            else:
                available = schedule.intersection(actual)
                expected += len(schedule)
                covered += len(available)
                missing = schedule.difference(actual)
                missing_minutes += len(missing)
                missing_hours += len(schedule.floor("h").unique().difference(actual.floor("h")))
            missing_days.update(schedule.strftime("%Y-%m-%d").unique().difference(dates.strftime("%Y-%m-%d").unique()))
        elif base_timeframe in RULES:
            schedule = expected_minutes(start, end, holidays)
            slots = pd.Series(1, index=schedule).resample(RULES[base_timeframe], label="left", closed="left").count()
            slots = pd.DatetimeIndex(slots.index[slots > 0])
            if base_timeframe == "W1":
                slots = slots[slots.strftime("%Y-%m") == dates.min().strftime("%Y-%m")]
            for side in ("bid", "ask"):
                actual = pd.DatetimeIndex(frame.loc[frame.side == side, "timestamp"]).unique()
                expected += len(slots)
                covered += len(slots.intersection(actual))
                missing_bars += len(slots.difference(actual))
    statistics.commit()
    statistics.execute("CREATE INDEX spread_values ON spreads(value)")
    statistics.execute("CREATE INDEX spread_hours ON spreads(hour,value)")
    median = None
    if spread_count:
        central = statistics.execute(
            "SELECT value FROM spreads ORDER BY value LIMIT ? OFFSET ?",
            (1 if spread_count % 2 else 2, (spread_count - 1) // 2),
        ).fetchall()
        median = sum(value[0] for value in central) / len(central)
    variance = max(0.0, (return_squares - return_sum ** 2 / return_count) / (return_count - 1)) if return_count > 1 else None
    with storage.connect() as connection:
        connection.row_factory = sqlite3.Row
        suffix = "%h_ticks.bi5" if timeframe == "tick" or timeframe.endswith("_from_ticks") else "%candles_min_1.bi5"
        downloads = [dict(row) for row in connection.execute(
            "SELECT period,status,error,url FROM downloads WHERE instrument=? AND url LIKE ? ORDER BY period", (pair, suffix)
        )]
    failures = [row for row in downloads if row["status"] != "verified"]
    base_timeframe = timeframe.removesuffix("_from_ticks")
    if timeframe == "tick":
        for failure in failures:
            start = pd.Timestamp(failure["period"])
            chunk = storage.root / "processed" / pair / "tick" / str(start.year) / f"{start.month:02}" / f"{start.strftime('%Y-%m-%d-%H')}.parquet"
            if chunk.exists():
                continue
            schedule = expected_minutes(start, start + pd.Timedelta(hours=1), holidays)
            expected += len(schedule)
            missing_minutes += len(schedule)
            missing_hours += int(len(schedule) > 0)
    elif requested[0] is not None:
        first_requested = pd.Timestamp(requested[0]).normalize()
        end_requested = pd.Timestamp(requested[1]).normalize() + pd.Timedelta(days=1)
        first_label = first_requested - pd.Timedelta(days=7) if base_timeframe == "W1" else first_requested
        for month in pd.date_range(first_label.replace(day=1), end_requested, freq="MS", inclusive="left"):
            if month.strftime("%Y-%m") in processed_months:
                continue
            calendar_end = month + pd.offsets.MonthBegin(1)
            if base_timeframe == "W1":
                calendar_end += pd.Timedelta(days=7)
            begin, finish = max(month, first_requested), min(calendar_end, end_requested)
            if finish <= begin:
                continue
            schedule = expected_minutes(begin, finish, holidays)
            slots = pd.Series(1, index=schedule).resample(RULES[base_timeframe], label="left", closed="left").count()
            if base_timeframe == "W1":
                slots = slots.loc[pd.DatetimeIndex(slots.index).strftime("%Y-%m") == month.strftime("%Y-%m")]
            expected += 2 * int((slots > 0).sum())
            missing_bars += 2 * int((slots > 0).sum())
            if base_timeframe == "M1":
                missing_minutes += 2 * len(schedule)
                missing_hours += 2 * len(schedule.floor("h").unique())
            missing_days.update(schedule.strftime("%Y-%m-%d"))
    hourly_report = {str(hour): spread_distribution(statistics, hour, hour) for hour in range(24)}
    sessions = {name: spread_distribution(statistics, first, last) for name, (first, last) in {
        "asia_utc": (0, 7), "europe_utc": (8, 15), "america_utc": (16, 23),
    }.items()}
    return {
        "instrument": pair, "timeframe": timeframe, "source": "Dukascopy",
        "rows": rows, "tick_count": rows if timeframe == "tick" else None,
        "candle_count": rows if timeframe != "tick" else None,
        "available_days": len(days),
        "first_available": first.isoformat() if first is not None else None,
        "last_available": last.isoformat() if last is not None else None,
        "quality_score": round(100 * (1 - bad / rows), 3),
        "score_definition": "100 * (1 - invalid_price_or_ohlc_or_crossed_tick_rows / rows); not completeness certification",
        "flags": dict(flags), "bad_rows": bad,
        "nominal_coverage_pct": 100 * covered / expected if expected else None,
        "coverage_scope": "requested archive span when metadata exists; otherwise observed partition span",
        "requested_archive_span": list(requested),
        "coverage_unit": "minute_quote" if base_timeframe in ("M1", "tick") else "candle",
        "expected_minute_observations": expected if base_timeframe in ("M1", "tick") else None,
        "covered_minute_observations": covered if base_timeframe in ("M1", "tick") else None,
        "expected_bar_observations": expected if base_timeframe != "tick" else None,
        "covered_bar_observations": covered if base_timeframe != "tick" else None,
        "missing_minute_candidates": missing_minutes, "missing_hour_candidates": missing_hours,
        "missing_bar_candidates": missing_bars,
        "missing_day_candidates": sorted(missing_days),
        "unavailable_or_failed_archives": failures,
        "spread": {
            "observations": spread_count, "mean": spread_sum / spread_count if spread_count else None,
            "median": median,
            "minimum": spread_min if spread_count else None, "maximum": spread_max if spread_count else None,
            "method": "tick ask-bid or synchronized M1/derived close ask-bid; exact median using disk-backed SQLite",
        },
        "spread_by_hour_utc": hourly_report, "spread_by_session_utc": sessions,
        "log_return_std": float(np.sqrt(variance)) if variance is not None else None,
        "volatility_definition": "sample std of consecutive observed bid log returns, not annualized; gaps retained",
        "mean_candle_range": amplitude_sum / amplitude_count if amplitude_count else None,
        "calendar": {
            "timezone": "America/New_York", "nominal_open": "Sunday 17:00 through Friday 17:00",
            "excluded_utc_dates": list(holidays), "holidays_verified": False,
            "warning": "Missing quotes are candidates, not proven outages. Holiday schedules and provider interruptions require review.",
        },
        "files": [str(path) for path in files],
    }
