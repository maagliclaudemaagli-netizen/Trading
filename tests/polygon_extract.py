"""Manual Polygon extraction example; not part of the backtest data pipeline."""
from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from polygon import RESTClient


class RateLimiter:
    def __init__(self) -> None:
        self.next_request = 0.0

    def wait(self) -> None:
        remaining = self.next_request - time.monotonic()
        while remaining > 0:
            time.sleep(remaining)
            remaining = self.next_request - time.monotonic()
        self.next_request = time.monotonic() + 12.1


def extract(
    client: RESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    limiter: RateLimiter,
) -> pd.DataFrame:
    start, end = date.fromisoformat(from_date), date.fromisoformat(to_date)
    if start > end:
        raise ValueError("La date de début doit précéder la date de fin.")
    if multiplier != 1 or timespan not in {"minute", "hour", "day"}:
        raise ValueError("Cet exemple utilise le multiplicateur 1 et minute, hour ou day.")
    if not ticker.startswith("C:") or len(ticker) != 8 or not ticker[2:].isalpha():
        raise ValueError("Le ticker Forex doit être au format C:EURUSD.")

    rows = []
    # 30 days contain at most 43,200 minute bars, below Polygon's 50,000 limit.
    while start <= end:
        chunk_end = min(start + timedelta(days=29), end)
        limiter.wait()
        bars = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start.isoformat(),
            to=chunk_end.isoformat(),
            sort="asc",
            limit=50_000,
        )
        if not isinstance(bars, list):
            raise TypeError("Réponse Polygon inattendue : une liste de bougies est requise.")
        for bar in bars:
            rows.append({
                "Timestamp": pd.to_datetime(bar.timestamp, unit="ms", utc=True),
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            })
        start = chunk_end + timedelta(days=1)

    if not rows:
        raise ValueError("Aucune donnée renvoyée pour cette période et ce ticker.")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset="Timestamp")
        .set_index("Timestamp")
        .sort_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="C:EURUSD")
    parser.add_argument("--multiplier", type=int, choices=[1], default=1)
    parser.add_argument("--timespan", choices=["minute", "hour", "day"], default="day")
    parser.add_argument("--from-date", default="2025-01-01")
    parser.add_argument("--to-date", default="2025-12-31")
    parser.add_argument("--output", type=Path, help="Fichier CSV facultatif.")
    args = parser.parse_args()
    key = os.environ.get("POLYGON_API_KEY")
    if not key or not key.strip():
        parser.error("Définissez POLYGON_API_KEY dans votre environnement.")

    from polygon import RESTClient

    # SDK retries would bypass our limiter; propagate HTTP errors instead.
    client = RESTClient(api_key=key, retries=0)
    frame = extract(
        client, args.ticker, args.multiplier, args.timespan,
        args.from_date, args.to_date, RateLimiter(),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output)
    print(f"Données récupérées : {len(frame)} bougies.")
    print(frame.head())


if __name__ == "__main__":
    main()
