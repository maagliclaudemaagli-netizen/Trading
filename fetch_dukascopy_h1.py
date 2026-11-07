"""Download long H1 history (10+ years) from Dukascopy's public monthly .bi5 files.

Module purpose
--------------
Dukascopy publishes one LZMA-compressed file of hourly BID candles per symbol and
month. This module downloads them, decodes them and writes one CSV per symbol.

    build_url()          -> URL of one monthly file
    http_get()           -> download with curl + retries (network only)
    decode_month()       -> bytes -> OHLCV DataFrame (pure, testable offline)
    months_to_fetch()    -> list of complete (year, month) pairs to download
    fetch_symbol()       -> full history for one symbol
    clean_history()      -> drop flat bars / duplicates, sort

File format: records of 24 bytes, big-endian: seconds since the month start (int32),
open, close, low, high (int32 / price divisor), volume (float32). Month is 0-indexed
in the URL.

curl is used instead of urllib because urllib hangs on the corporate proxy's
TLS handshake while curl goes through.

Usage:
    .venv/bin/python fetch_dukascopy_h1.py [EURUSD GBPUSD USDJPY] [--years 10]
Output:
    data/<PAIR>_H1.csv  (timestamp UTC, open, high, low, close, volume)
"""
from __future__ import annotations

import argparse
import lzma
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from strategy import ASSETS, DATA

URL_TEMPLATE = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month0:02d}/BID_candles_hour_1.bi5"
PRICE_DIVISOR: dict[str, float] = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3, "XAUUSD": 1e3}
RECORD_DTYPE = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"),
                         ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
MAX_RETRIES = 8
REQUEST_PAUSE_S = 3.0            # politeness delay between requests (avoids HTTP 429/503)
CURL_TIMEOUT_S = 30
CACHE_DIR = DATA / "cache"       # raw monthly files: a rerun resumes where it stopped


class DownloadError(RuntimeError):
    """Raised when a file cannot be downloaded after all retries."""


# ---------------------------------------------------------------------------- network
def build_url(symbol: str, year: int, month: int) -> str:
    """URL of the hourly candles file for `symbol`, `year`, `month` (1-12)."""
    return URL_TEMPLATE.format(symbol=symbol, year=year, month0=month - 1)


def http_get(url: str) -> bytes:
    """Download `url` with curl, retrying with exponential backoff.

    Args:
        url: file URL.

    Returns:
        Response body, or b"" if the server answers 404 (no data for that month).

    Raises:
        DownloadError: after MAX_RETRIES failed attempts (429, 5xx, timeouts...).
        FileNotFoundError: if curl is not installed.
    """
    last_status = "?"
    for attempt in range(MAX_RETRIES):
        proc = subprocess.run(
            ["curl", "-s", "-m", str(CURL_TIMEOUT_S), "-A", "Mozilla/5.0",
             "-w", "\n%{http_code}", url],
            capture_output=True,
        )
        body, _, status = proc.stdout.rpartition(b"\n")
        last_status = status.decode(errors="replace") or f"curl exit {proc.returncode}"
        if status == b"200":
            return body
        if status == b"404":
            return b""
        time.sleep(min(300, 10 * 2 ** attempt))
    raise DownloadError(f"{url} failed after {MAX_RETRIES} attempts (last status {last_status})")


# ---------------------------------------------------------------------------- decoding
def decode_month(raw: bytes, symbol: str, year: int, month: int) -> pd.DataFrame:
    """Decode one compressed monthly file into OHLCV rows.

    Args:
        raw: LZMA-compressed .bi5 content (b"" means no data).
        symbol: key of PRICE_DIVISOR.
        year, month: month the file belongs to (month 1-12).

    Returns:
        DataFrame with columns timestamp (UTC), open, high, low, close, volume.
        Empty DataFrame if `raw` is empty.

    Raises:
        KeyError: unknown symbol (no price divisor).
        ValueError: corrupted file (bad LZMA stream or size not a multiple of 24).
    """
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not raw:
        return pd.DataFrame(columns=columns)
    if symbol not in PRICE_DIVISOR:
        raise KeyError(f"No price divisor for {symbol}; add it to PRICE_DIVISOR")
    try:
        payload = lzma.decompress(raw)
    except lzma.LZMAError as exc:
        raise ValueError(f"{symbol} {year}-{month:02d}: invalid LZMA data") from exc
    if len(payload) % RECORD_DTYPE.itemsize:
        raise ValueError(f"{symbol} {year}-{month:02d}: truncated file ({len(payload)} bytes)")

    records = np.frombuffer(payload, RECORD_DTYPE)
    divisor = PRICE_DIVISOR[symbol]
    month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    return pd.DataFrame({
        "timestamp": month_start + pd.to_timedelta(records["t"], unit="s"),
        "open": records["o"] / divisor, "high": records["h"] / divisor,
        "low": records["l"] / divisor, "close": records["c"] / divisor,
        "volume": records["v"].astype(float),
    })


def clean_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate monthly frames, drop flat bars (closed market) and duplicates.

    Args:
        frames: decoded monthly DataFrames.

    Returns:
        DataFrame indexed by timestamp, sorted, with high > low on every row.

    Raises:
        ValueError: if no data at all was decoded.
    """
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        raise ValueError("No data downloaded")
    df = pd.concat(non_empty).set_index("timestamp").sort_index()
    df = df[df.high > df.low]
    return df[~df.index.duplicated()]


# ---------------------------------------------------------------------------- orchestration
def months_to_fetch(years: int, now: datetime | None = None) -> list[tuple[int, int]]:
    """Complete months covering the last `years` calendar years up to last month.

    Args:
        years: number of past years to include (>= 0); 0 = current year only.
        now: reference time (defaults to current UTC time; injectable for tests).

    Returns:
        List of (year, month) with month 1-12, oldest first, excluding the current
        (incomplete) month.

    Raises:
        ValueError: if years < 0.
    """
    if years < 0:
        raise ValueError("years must be >= 0")
    now = now or datetime.now(timezone.utc)
    return [(y, m) for y in range(now.year - years, now.year + 1) for m in range(1, 13)
            if (y, m) < (now.year, now.month)]


def fetch_symbol(symbol: str, years: int) -> pd.DataFrame:
    """Download and decode the full H1 history of `symbol`.

    Args:
        symbol: e.g. "EURUSD".
        years: how many past years to fetch.

    Returns:
        Cleaned OHLCV DataFrame indexed by UTC timestamp.
    """
    frames: list[pd.DataFrame] = []
    current_year = None
    for year, month in months_to_fetch(years):
        if year != current_year and current_year is not None:
            print(f"  {symbol} {current_year} ok", flush=True)
        current_year = year
        frames.append(decode_month(get_month(symbol, year, month), symbol, year, month))
    return clean_history(frames)


def get_month(symbol: str, year: int, month: int) -> bytes:
    """Raw monthly file, read from CACHE_DIR if present, otherwise downloaded and cached.

    Empty (404) months are cached too, as 0-byte files.
    """
    cache = CACHE_DIR / symbol / f"{year}-{month:02d}.bi5"
    if cache.exists():
        return cache.read_bytes()
    raw = http_get(build_url(symbol, year, month))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    time.sleep(REQUEST_PAUSE_S)
    return raw


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("assets", nargs="*", default=ASSETS)
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()

    exit_code = 0
    for symbol in args.assets:
        try:
            df = fetch_symbol(symbol, args.years)
        except (DownloadError, ValueError, KeyError) as exc:
            print(f"{symbol}: ÉCHEC — {exc}", file=sys.stderr)
            exit_code = 1
            continue
        out = DATA / f"{symbol}_H1.csv"
        df.to_csv(out)
        print(f"{symbol}: {len(df)} bougies {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} -> {out.name}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
