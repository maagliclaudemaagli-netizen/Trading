"""Download long H1 history (2002 -> today) from the open OANDA datasets on Hugging Face.

Module purpose
--------------
The public datasets `bunqi/OANDA_<PAIR>_<PAIR>` publish one parquet file per month
and timeframe (folder "60" = H1). This module lists the monthly files, downloads them
with curl (works through the corporate proxy), and writes one CSV per pair.

    list_files()      -> monthly parquet paths for a pair (Hugging Face API)
    download()        -> one file, cached in data/cache/oanda/
    to_ohlcv()        -> raw parquet frame -> clean OHLCV indexed by UTC (pure)
    fetch_pair()      -> full history for one pair

Usage:
    .venv/bin/python fetch_oanda_h1.py [EURUSD GBPUSD USDJPY] [--since 2010]
Output:
    data/<PAIR>_H1.csv  (timestamp UTC, open, high, low, close, volume)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from strategy import ASSETS, DATA

REPO = "bunqi/OANDA_{pair}_{pair}"
TREE_URL = "https://huggingface.co/api/datasets/{repo}/tree/main/60"
FILE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
CACHE_DIR = DATA / "cache" / "oanda"
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DownloadError(RuntimeError):
    """Raised when a remote file or listing cannot be retrieved."""


def curl(url: str, output: Path | None = None) -> bytes:
    """Fetch `url` with curl (follows redirects). Writes to `output` if given.

    Raises:
        DownloadError: non-zero curl exit code or HTTP error.
    """
    cmd = ["curl", "-sfL", "-m", "60", url]
    if output:
        cmd += ["-o", str(output)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DownloadError(f"curl exit {proc.returncode} for {url}")
    return proc.stdout


def list_files(pair: str) -> list[str]:
    """Sorted list of H1 parquet paths for `pair` (e.g. "60/OANDA_..._2020-01-01_to_....parquet")."""
    raw = curl(TREE_URL.format(repo=REPO.format(pair=pair)))
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DownloadError(f"{pair}: invalid listing") from exc
    return sorted(e["path"] for e in entries if e.get("path", "").endswith(".parquet"))


def file_year(path: str) -> int:
    """Year of a monthly file, parsed from its name (..._60_YYYY-MM-DD_to_...)."""
    return int(path.split("_60_")[1][:4])


def download(pair: str, path: str) -> Path:
    """Download one monthly file into the cache (skipped if already cached)."""
    target = CACHE_DIR / pair / Path(path).name
    if not target.exists() or target.stat().st_size == 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        curl(FILE_URL.format(repo=REPO.format(pair=pair), path=path), tmp)
        tmp.rename(target)
    return target


def to_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw OANDA frame (time = unix seconds) to clean OHLCV.

    Returns:
        DataFrame indexed by UTC "timestamp", sorted, no duplicates, no flat bars.

    Raises:
        ValueError: missing columns.
    """
    missing = {"time", *OHLCV_COLUMNS} - set(raw.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    df = raw[OHLCV_COLUMNS].astype(float)
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["time"], unit="s", utc=True), name="timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[df.high > df.low]


def fetch_pair(pair: str, since: int) -> pd.DataFrame:
    """Download (with cache) and assemble the H1 history of `pair` from year `since`."""
    paths = [p for p in list_files(pair) if file_year(p) >= since]
    if not paths:
        raise DownloadError(f"{pair}: no H1 file since {since}")
    frames = []
    for i, path in enumerate(paths, 1):
        frames.append(pd.read_parquet(download(pair, path)))
        if i % 24 == 0 or i == len(paths):
            print(f"  {pair}: {i}/{len(paths)} mois", flush=True)
    return to_ohlcv(pd.concat(frames, ignore_index=True))


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description="Download H1 history from OANDA datasets")
    parser.add_argument("pairs", nargs="*", default=ASSETS)
    parser.add_argument("--since", type=int, default=2010, help="first year to download")
    args = parser.parse_args()

    exit_code = 0
    for pair in args.pairs:
        try:
            df = fetch_pair(pair, args.since)
        except (DownloadError, ValueError) as exc:
            print(f"{pair}: ÉCHEC — {exc}", file=sys.stderr)
            exit_code = 1
            continue
        out = DATA / f"{pair}_H1.csv"
        df.to_csv(out)
        print(f"{pair}: {len(df)} bougies {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} -> {out.name}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
