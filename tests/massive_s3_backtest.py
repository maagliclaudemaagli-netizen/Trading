"""Compare EURUSD S3 minute files with local OANDA M5 and run the same strategy."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from massive_s3_extract import BUCKET, ENDPOINT, S3Client, create_client, extract, object_key

# Direct execution from tests/ must also resolve the unchanged backtest package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from institutional.__main__ import load_m5
from institutional.backtest import BacktestResult, ExecutionConfig, backtest, data_fingerprint
from institutional.market import AGG, OHLCV, validate_m5
from institutional.strategy import ALIGNMENTS, StrategyConfig


def period_days(start: str, end: str) -> list[date]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first >= last:
        raise ValueError("start must precede end; end is exclusive")
    return [first + timedelta(days=i) for i in range((last - first).days)]


def available_days(client: S3Client, days: list[date]) -> list[date]:
    expected = {object_key(day.isoformat(), "minute"): day for day in days}
    found: set[date] = set()
    for month in sorted({day.strftime("%Y/%m") for day in days}):
        prefix = f"global_forex/minute_aggs_v1/{month}/"
        token = ""
        tokens: set[str] = set()
        while True:
            if token:
                page = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix, ContinuationToken=token)
            else:
                page = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
            for item in page.get("Contents", []):
                if item["Key"] in expected:
                    found.add(expected[item["Key"]])
            if not page["IsTruncated"]:
                break
            token = page.get("NextContinuationToken", "")
            if not token or token in tokens:
                raise ValueError("S3 listing returned a missing or repeated continuation token")
            tokens.add(token)
    if not found:
        raise ValueError("No S3 minute files listed for the requested period")
    return sorted(found)


def minute_to_m5(minutes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if minutes.empty or not minutes.columns.is_unique or set(OHLCV) - set(minutes.columns):
        raise ValueError("M1: empty data, duplicate or missing OHLCV columns")
    index = pd.DatetimeIndex(minutes.index)
    if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("M1: timestamps must be unique, sorted, timezone-aware and non-null")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor("min")):
        raise ValueError("M1: timestamps must be aligned to one minute")
    frame = minutes[OHLCV].astype(float).copy()
    frame.index = index
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("M1: OHLCV must be finite")
    if (
        (frame.high < frame[["open", "close", "low"]].max(axis=1))
        | (frame.low > frame[["open", "close", "high"]].min(axis=1))
        | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (frame.volume < 0)
    ).any():
        raise ValueError("M1: inconsistent OHLCV")
    grouped = frame.resample("5min", label="left", closed="left")
    counts = grouped.size()
    incomplete = counts[(counts > 0) & (counts != 5)]
    if not incomplete.empty:
        raise ValueError(
            f"M1: {len(incomplete)} incomplete M5 bins; first {incomplete.index[0]} "
            f"has {incomplete.iloc[0]} of 5 minutes. No candles were filled or fabricated."
        )
    m5 = validate_m5(grouped.agg(AGG).loc[counts == 5])
    return m5, {
        "m1_bars": len(minutes), "m5_bars": len(m5),
        "incomplete_m5_bins": 0, "empty_m5_bins": int((counts == 0).sum()),
    }


def compare_data(s3: pd.DataFrame, oanda: pd.DataFrame) -> dict:
    s3, oanda = validate_m5(s3), validate_m5(oanda)
    common = s3.index.intersection(oanda.index)
    if common.empty:
        raise ValueError("S3 and OANDA have no common M5 timestamps")
    prices = OHLCV[:-1]
    left, right = s3.loc[common, prices], oanda.loc[common, prices]
    differences = (left - right).abs()
    return {
        "common_m5_bars": len(common),
        "s3_only_m5_bars": len(s3.index.difference(oanda.index)),
        "oanda_only_m5_bars": len(oanda.index.difference(s3.index)),
        "s3_overlap_pct": 100 * len(common) / len(s3),
        "oanda_overlap_pct": 100 * len(common) / len(oanda),
        "different_ohlc_bars": int((left != right).any(axis=1).sum()),
        "absolute_price_differences": {
            column: {"mean": float(differences[column].mean()), "max": float(differences[column].max())}
            for column in prices
        },
        "s3_data_sha256": data_fingerprint(s3),
        "oanda_data_sha256": data_fingerprint(oanda),
        "acceptance_threshold": None,
        "interpretation": "Structural checks passed; price agreement is measured, not certified. "
                          "Different feeds and quote bases can differ; volume is not compared.",
    }


def result_report(data: pd.DataFrame, result: BacktestResult) -> dict:
    return {
        "first_open": str(data.index[0]), "last_open": str(data.index[-1]),
        "m5_bars": len(data), "data_sha256": data_fingerprint(data),
        "metrics": result.metrics,
        "funnel": result.audit.event.value_counts().to_dict() if not result.audit.empty else {},
    }


def run_comparison(
    client: S3Client, oanda: pd.DataFrame, start: str, end: str, output: Path,
    reference_source: str,
    strategy: StrategyConfig = StrategyConfig(),
    execution: ExecutionConfig = ExecutionConfig(),
) -> dict:
    if execution.pair != "EURUSD":
        raise ValueError("This comparison is configured for EURUSD")
    days = period_days(start, end)
    first, last = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    oanda = validate_m5(oanda.loc[(oanda.index >= first) & (oanda.index < last)])
    listed = available_days(client, days)
    minutes = pd.concat([extract(client, day.isoformat(), "minute", "C:EURUSD") for day in listed])
    s3, quality = minute_to_m5(minutes)
    comparison = compare_data(s3, oanda)
    results = {"s3": backtest(s3, strategy, execution), "oanda": backtest(oanda, strategy, execution)}
    frames = {"s3": s3, "oanda": oanda}
    report = {
        "pair": "EURUSD", "start_inclusive": start, "end_exclusive": end,
        "strategy": asdict(strategy), "execution": asdict(execution),
        "sources": {
            "s3_endpoint": ENDPOINT, "s3_bucket": BUCKET,
            "s3_keys": [object_key(day.isoformat(), "minute") for day in listed],
            "oanda": reference_source,
        },
        "quality": {
            **quality,
            "dates_without_listed_file": [day.isoformat() for day in days if day not in set(listed)],
            "calendar_note": "Unlisted dates and empty bins may include closures; no candles are filled.",
        },
        "comparison": comparison,
        "backtests": {name: result_report(frames[name], result) for name, result in results.items()},
        "limitations": "Same requested period, each feed's own candles; no prior-period warmup. "
                       "Quote basis and fixed spread are modelling assumptions. "
                       "Backtest profitability does not validate market data.",
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        frames[name].to_csv(output.with_suffix(f".{name}.m5.csv"), index_label="timestamp")
        result.trades.to_csv(output.with_suffix(f".{name}.trades.csv"), index=False)
        result.audit.to_csv(output.with_suffix(f".{name}.audit.csv"), index=False)
    output.write_text(serialized, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01", help="Inclusive UTC date")
    parser.add_argument("--end", default="2025-02-01", help="Exclusive UTC date")
    parser.add_argument("--cache", type=Path, default=Path("data/cache/oanda/EURUSD_M5"))
    parser.add_argument("--alignment", choices=ALIGNMENTS, default="intraday")
    parser.add_argument("--price-basis", choices=["mid", "bid"], default="mid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not key.strip() or not secret or not secret.strip():
        parser.error("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment")
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        period_days(args.start, args.end)
        oanda = load_m5(None, args.cache, args.start, args.end)
        client = create_client(key, secret)
        try:
            report = run_comparison(
                client, oanda, args.start, args.end, args.output,
                str(args.cache),
                StrategyConfig(args.alignment), ExecutionConfig(price_basis=args.price_basis),
            )
        finally:
            client.close()
        print(json.dumps(report, indent=2, allow_nan=False))
    except ClientError as exc:
        parser.exit(1, f"S3 comparison failed: HTTP {exc.response['ResponseMetadata']['HTTPStatusCode']} "
                       f"({exc.response['Error']['Code']}); no validated S3 backtest was produced.\n")
    except (BotoCoreError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"S3 comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
