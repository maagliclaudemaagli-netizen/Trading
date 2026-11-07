"""Run: python -m institutional --csv data/EURUSD_M5.csv --output reports/institutional.json."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtest import ExecutionConfig, backtest, data_fingerprint
from .market import validate_m5
from .strategy import ALIGNMENTS, StrategyConfig


def utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Date cannot be NaT")
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def load_m5(csv: Path, start: str | None, end: str | None) -> pd.DataFrame:
    first, last = utc(start) if start else None, utc(end) if end else None
    if first is not None and last is not None and first >= last:
        raise ValueError("start must precede end")
    raw = pd.read_csv(csv, float_precision="round_trip")
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "timestamp" in raw:
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        frame = raw.set_index("timestamp")
    elif "time" in raw:
        frame = raw.drop(columns="time")
        frame.index = pd.DatetimeIndex(pd.to_datetime(raw.time, unit="s", utc=True), name="timestamp")
    else:
        raise ValueError("CSV requires timestamp or Unix-seconds time")
    if first is not None:
        frame = frame.loc[frame.index >= first]
    if last is not None:
        frame = frame.loc[frame.index < last]
    return validate_m5(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Local Dukascopy M5 CSV export")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start", help="Inclusive UTC date")
    parser.add_argument("--end", help="Exclusive UTC date")
    parser.add_argument("--alignment", choices=ALIGNMENTS, default="intraday")
    parser.add_argument("--price-basis", choices=("mid", "bid"), default="mid")
    parser.add_argument("--export-m5", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="JSON report; sibling trades/audit CSV")
    args = parser.parse_args()
    try:
        data = load_m5(args.csv, args.start, args.end)
        config = StrategyConfig(args.alignment)
        execution = ExecutionConfig(pair=args.pair, price_basis=args.price_basis)
        result = backtest(data, config, execution)
        report = {
            "source": str(args.csv),
            "first_open": str(data.index[0]), "last_open": str(data.index[-1]), "m5_bars": len(data),
            "data_sha256": data_fingerprint(data),
            "strategy": asdict(config), "execution": asdict(execution), "metrics": result.metrics,
            "funnel": result.audit.event.value_counts().to_dict() if not result.audit.empty else {},
            "conventions": "See INSTITUTIONAL.md for the executable protocol and modelling definitions.",
            "profit_factor_encoding": "null = undefined; string Infinity = gains without losses.",
        }
        if args.export_m5 is not None:
            args.export_m5.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(args.export_m5, index_label="timestamp")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        result.trades.to_csv(args.output.with_suffix(".trades.csv"), index=False)
        result.audit.to_csv(args.output.with_suffix(".audit.csv"), index=False)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Backtest failed: {exc}\n")


if __name__ == "__main__":
    main()
