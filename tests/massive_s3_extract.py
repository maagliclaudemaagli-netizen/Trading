"""Manual Massive S3 Forex extraction, isolated from the backtest pipeline."""
from __future__ import annotations

import argparse
import os
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict

import pandas as pd

ENDPOINT = "https://files.massive.com"
BUCKET = "flatfiles"
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


class StreamingBody(Protocol):
    def read(self, amt: int | None = None) -> bytes: ...

    def close(self) -> None: ...


class ObjectResponse(TypedDict):
    Body: StreamingBody


class ListedObject(TypedDict):
    Key: str


class ListingResponse(TypedDict):
    IsTruncated: bool
    Contents: NotRequired[list[ListedObject]]
    NextContinuationToken: NotRequired[str]


class S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> ObjectResponse: ...

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, ContinuationToken: str = "",
    ) -> ListingResponse: ...

    def close(self) -> None: ...


def create_client(key: str, secret: str) -> S3Client:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 3},
            s3={"addressing_style": "path"},
        ),
    )


def object_key(day: str, timespan: str) -> str:
    parsed = date.fromisoformat(day)
    if timespan not in {"minute", "day"}:
        raise ValueError("Utilisez minute ou day pour les fichiers Forex.")
    return f"global_forex/{timespan}_aggs_v1/{parsed:%Y/%m/%Y-%m-%d}.csv.gz"


def extract(client: S3Client, day: str, timespan: str, ticker: str) -> pd.DataFrame:
    key = object_key(day, timespan)
    if not ticker.startswith("C:") or len(ticker) != 8 or not ticker[2:].isalpha():
        raise ValueError("Le ticker Forex doit être au format C:EURUSD.")

    response = client.get_object(Bucket=BUCKET, Key=key)
    frames = []
    required = {"ticker", "window_start", *PRICE_COLUMNS}
    with closing(response["Body"]) as body:
        for chunk in pd.read_csv(body, compression="gzip", chunksize=100_000):
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"Colonnes S3 manquantes : {', '.join(sorted(missing))}.")
            selected = chunk.loc[chunk["ticker"] == ticker, ["window_start", *PRICE_COLUMNS]]
            if not selected.empty:
                frames.append(selected)

    if not frames:
        raise ValueError(f"Aucune bougie {ticker} dans le fichier {key}.")
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame.pop("window_start"), unit="ns", utc=True)
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or frame[PRICE_COLUMNS].isin([float("inf"), -float("inf")]).any().any():
        raise ValueError("Le fichier contient des valeurs manquantes ou non finies.")
    if (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame["volume"] < 0)
    ).any():
        raise ValueError("Le fichier contient des bougies OHLCV incohérentes.")
    if frame["timestamp"].dt.date.ne(date.fromisoformat(day)).any():
        raise ValueError("Le fichier contient des bougies hors de la date demandée.")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Le fichier contient des horodatages dupliqués pour cette paire.")
    return frame.set_index("timestamp").sort_index()[PRICE_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2025-01-02")
    parser.add_argument("--timespan", choices=["minute", "day"], default="day")
    parser.add_argument("--ticker", default="C:EURUSD")
    parser.add_argument("--output", type=Path, help="Fichier CSV facultatif.")
    args = parser.parse_args()
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not key.strip() or not secret or not secret.strip():
        parser.error("Définissez AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY dans l'environnement.")

    client = create_client(key, secret)
    try:
        frame = extract(client, args.date, args.timespan, args.ticker)
    finally:
        client.close()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output)
    print(f"S3 : {len(frame)} bougies {args.ticker}, horodatages UTC.")
    print(frame.head())


if __name__ == "__main__":
    main()
