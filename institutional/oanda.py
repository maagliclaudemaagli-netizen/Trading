"""Read-only OANDA v20 M5 importer; credentials are read only from the environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import pandas as pd
import v20
from v20.errors import V20ConnectionError, V20Timeout
from v20.response import Response

from .__main__ import utc
from .market import OHLCV, validate_m5

HOSTS = {"practice": "api-fxpractice.oanda.com", "live": "api-fxtrade.oanda.com"}
STEP = pd.Timedelta(minutes=5)
QUOTE_FIELDS = ("open", "high", "low", "close")
QUOTE_COLUMNS = [f"{side}_{field}" for side in ("bid", "ask") for field in QUOTE_FIELDS]


class OandaError(RuntimeError):
    """An explicit API, data-quality or configuration failure."""


@dataclass
class Download:
    data: pd.DataFrame
    incomplete_bars: int
    empty_windows: int
    requests: int

    def quality(self) -> dict:
        gaps = self.data.index[1:] - self.data.index[:-1]
        interruptions = gaps[gaps > STEP]
        return {
            "bars": len(self.data),
            "first_open": self.data.index[0].isoformat(),
            "last_open": self.data.index[-1].isoformat(),
            "incomplete_bars_excluded": self.incomplete_bars,
            "empty_request_windows": self.empty_windows,
            "requests": self.requests,
            "flat_mid_bars": int((self.data.high == self.data.low).sum()),
            "gaps_over_5_minutes": len(interruptions),
            "unobserved_5m_slots_between_bars": int(
                sum((gap // STEP) - 1 for gap in interruptions)
            ),
            "largest_gap_minutes": float(gaps.max() / pd.Timedelta(minutes=1)) if len(gaps) else 0.0,
            "close_spread_min": float((self.data.ask_close - self.data.bid_close).min()),
            "close_spread_max": float((self.data.ask_close - self.data.bid_close).max()),
            "gap_note": "Includes weekends, holidays and market closures; not a certified missing-data count.",
        }


class OandaImporter:
    def __init__(self, token: str, environment: str = "practice"):
        if not token.strip():
            raise OandaError("Set OANDA_TOKEN locally; never put the token in CLI arguments or Git.")
        if environment not in HOSTS:
            raise OandaError("environment must be practice or live")
        self.environment = environment
        self.context = v20.Context(
            HOSTS[environment], token=token, datetime_format="RFC3339", poll_timeout=30,
        )

    def _request(self, operation: Callable[[], Response]) -> Response:
        for attempt in range(3):
            time.sleep(0.2)
            try:
                response = operation()
            except (V20ConnectionError, V20Timeout) as exc:
                if attempt == 2:
                    raise OandaError("OANDA connection failed after 3 attempts; check network access.") from exc
                print("OANDA connection interrupted; retrying.", file=sys.stderr)
            else:
                if response.status == 200:
                    if not isinstance(response.body, dict):
                        raise OandaError("OANDA returned an unexpected non-JSON response.")
                    return response
                if response.status not in (429, 500, 502, 503, 504) or attempt == 2:
                    hint = " Check token, account and practice/live environment." if response.status in (401, 403) else ""
                    raise OandaError(f"OANDA HTTP {response.status}.{hint}")
                print(f"OANDA HTTP {response.status}; retrying.", file=sys.stderr)
            time.sleep(2 ** attempt)
        raise OandaError("OANDA request exhausted its retries.")

    def forex_instruments(self, account_id: str) -> list[str]:
        if not account_id.strip():
            raise OandaError("Set OANDA_ACCOUNT_ID locally to list the Forex instruments of your account.")
        response = self._request(lambda: self.context.account.instruments(account_id))
        instruments = response.body.get("instruments")
        if not isinstance(instruments, list):
            raise OandaError("OANDA response lacks an instruments list.")
        names = []
        for instrument in instruments:
            if instrument.type == "CURRENCY":
                if not isinstance(instrument.name, str) or not re.fullmatch(r"[A-Z]{3}_[A-Z]{3}", instrument.name):
                    raise OandaError("OANDA returned an invalid Forex instrument name.")
                names.append(instrument.name)
        if not names or len(names) != len(set(names)):
            raise OandaError("OANDA returned an empty or duplicate Forex instrument catalogue.")
        return sorted(names)

    def download(self, instrument: str, start: pd.Timestamp, end: pd.Timestamp) -> Download:
        if not re.fullmatch(r"[A-Z]{3}_[A-Z]{3}", instrument):
            raise OandaError("Use an OANDA Forex symbol such as EUR_USD.")
        validate_range(start, end)
        now = pd.Timestamp.now(tz="UTC")
        rows = []
        incomplete, empty, requests = 0, 0, 0
        cursor = start
        while cursor < end:
            boundary = min(cursor + pd.Timedelta(days=7), end)
            response = self._request(lambda: self.context.instrument.candles(
                instrument, granularity="M5", price="MBA", smooth=False,
                fromTime=cursor.isoformat(), toTime=boundary.isoformat(), includeFirst=True,
            ))
            if response.body.get("instrument") != instrument or response.body.get("granularity") != "M5":
                raise OandaError("OANDA response instrument or granularity does not match the request.")
            candles = response.body.get("candles")
            if not isinstance(candles, list):
                raise OandaError("OANDA response lacks a candles list.")
            requests += 1
            accepted = 0
            for candle in candles:
                at = utc(candle.time)
                if at < cursor or at > boundary:
                    raise OandaError("OANDA returned a candle outside the requested window.")
                if at == boundary:
                    continue
                if candle.complete is False or at + STEP > min(end, now):
                    incomplete += 1
                    continue
                if candle.complete is not True:
                    raise OandaError("OANDA candle lacks an explicit completion flag.")
                row = {"timestamp": at, "volume": candle.volume}
                for side, quote in (("mid", candle.mid), ("bid", candle.bid), ("ask", candle.ask)):
                    if quote is None:
                        raise OandaError("OANDA candle is missing a requested mid/bid/ask component.")
                    for field, attribute in zip(QUOTE_FIELDS, ("o", "h", "l", "c")):
                        value = getattr(quote, attribute)
                        if value is None:
                            raise OandaError("OANDA candle is missing an OHLC price.")
                        row[field if side == "mid" else f"{side}_{field}"] = float(value)
                rows.append(row)
                accepted += 1
            if not accepted:
                empty += 1
            print(f"{instrument}: {cursor:%Y-%m-%d} -> {boundary:%Y-%m-%d}, {accepted} closed M5", file=sys.stderr)
            cursor = boundary
        if not rows:
            raise OandaError(f"{instrument}: no complete M5 candles in the requested period.")
        data = pd.DataFrame(rows).set_index("timestamp")
        validate_m5(data)
        for side in ("bid", "ask"):
            quoted = data[[f"{side}_{field}" for field in QUOTE_FIELDS] + ["volume"]].copy()
            quoted.columns = OHLCV
            validate_m5(quoted)
        if any((data[f"bid_{field}"] > data[f"ask_{field}"]).any() for field in QUOTE_FIELDS):
            raise OandaError("OANDA returned crossed bid/ask quotes.")
        if not np.isfinite(data[QUOTE_COLUMNS].to_numpy()).all():
            raise OandaError("OANDA bid/ask prices must be finite.")
        if (data.volume != data.volume.astype(int)).any():
            raise OandaError("OANDA candle volume must be an integer tick count.")
        return Download(data, incomplete, empty, requests)


def validate_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    if pd.isna(start) or pd.isna(end) or start.tz is None or end.tz is None:
        raise ValueError("Dates must be non-null and timezone-aware.")
    if start >= end:
        raise ValueError("start must precede end")
    if start != start.floor("5min") or end != end.floor("5min"):
        raise ValueError("start and end must be aligned to five minutes")
    if end > pd.Timestamp.now(tz="UTC"):
        raise ValueError("end must not be in the future")


def save_download(
    download: Download, directory: Path, instrument: str, environment: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[Path, Path]:
    stem = f"{instrument.replace('_', '')}_M5_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"
    csv_path, report_path = directory / f"{stem}.csv", directory / f"{stem}.quality.json"
    if csv_path.exists() or report_path.exists():
        raise OandaError(f"Output already exists for {instrument}; choose a new directory or period.")
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".oanda-", dir=directory) as temporary:
        csv = Path(temporary) / csv_path.name
        report = Path(temporary) / report_path.name
        download.data.to_csv(csv, index_label="timestamp")
        metadata = {
            "source": "OANDA REST v20 via official v20 Python SDK",
            "environment": environment, "instrument": instrument,
            "granularity": "M5", "price": "MBA", "smooth": False,
            "requested_start": start.isoformat(), "requested_end_exclusive": end.isoformat(),
            "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
            "quality": download.quality(),
            "execution_note": "Backtest reads mid OHLCV only and still applies its configured fixed spread.",
        }
        report.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        csv.rename(csv_path)
        report.rename(report_path)
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=HOSTS, default="practice")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list-forex", action="store_true")
    choice.add_argument("--all-forex", action="store_true")
    choice.add_argument("--pair", nargs="+", help="OANDA symbols, e.g. EUR_USD GBP_USD")
    parser.add_argument("--start", help="Inclusive UTC date, required for download")
    parser.add_argument("--end", help="Exclusive UTC date, required for download")
    parser.add_argument("--output-dir", type=Path, default=Path("data/oanda"))
    args = parser.parse_args()
    try:
        start = end = None
        if not args.list_forex:
            if not args.start or not args.end:
                raise ValueError("Downloads require --start and --end.")
            start, end = utc(args.start), utc(args.end)
            validate_range(start, end)
        importer = OandaImporter(os.environ.get("OANDA_TOKEN", ""), args.environment)
        instruments = importer.forex_instruments(os.environ.get("OANDA_ACCOUNT_ID", ""))
        if args.list_forex:
            print("\n".join(instruments))
            return
        selected = instruments if args.all_forex else list(dict.fromkeys(args.pair))
        unavailable = set(selected) - set(instruments)
        if unavailable:
            raise ValueError(f"Forex instruments unavailable for this account: {', '.join(sorted(unavailable))}")
        if start is None or end is None:
            raise ValueError("Downloads require a validated date range.")
        for instrument in selected:
            download = importer.download(instrument, start, end)
            csv, report = save_download(download, args.output_dir, instrument, args.environment, start, end)
            print(f"Saved {len(download.data)} M5: {csv}; quality report: {report}", flush=True)
    except (OandaError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"OANDA import failed: {exc}\n")


if __name__ == "__main__":
    main()
