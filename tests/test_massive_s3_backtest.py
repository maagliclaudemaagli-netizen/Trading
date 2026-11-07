"""Synthetic tests of S3 listing, strict M1 conversion and strategy comparison."""
from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from massive_s3_backtest import (
    available_days, compare_data, minute_to_m5, period_days, run_comparison,
)
from massive_s3_extract import object_key
from institutional.backtest import backtest, data_fingerprint


def sample_minutes(periods: int = 60, start: str = "2025-01-02") -> pd.DataFrame:
    opening = 1 + np.arange(periods) / 1024
    return pd.DataFrame(
        {"open": opening, "high": opening + 0.25, "low": opening - 0.25,
         "close": opening + 0.125, "volume": np.ones(periods)},
        index=pd.date_range(start, periods=periods, freq="min", tz="UTC", name="timestamp"),
    )


def fake_client(minutes: pd.DataFrame) -> Mock:
    client = Mock()
    key = object_key("2025-01-02", "minute")
    client.list_objects_v2.return_value = {"Contents": [{"Key": key}], "IsTruncated": False}
    csv = minutes.reset_index().rename(columns={"timestamp": "window_start"})
    csv["window_start"] = minutes.index.as_unit("ns").asi8
    csv["ticker"] = "C:EURUSD"
    client.get_object.return_value = {"Body": io.BytesIO(gzip.compress(csv.to_csv(index=False).encode()))}
    return client


class S3BacktestTests(unittest.TestCase):
    def test_period_uses_an_exclusive_end(self) -> None:
        days = period_days("2024-02-28", "2024-03-01")
        self.assertEqual([day.isoformat() for day in days], ["2024-02-28", "2024-02-29"])
        with self.assertRaisesRegex(ValueError, "start must precede end"):
            period_days("2025-01-02", "2025-01-02")

    def test_listing_paginates_and_filters_exact_dates(self) -> None:
        client = Mock()
        client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": object_key("2025-01-01", "minute")},
                          {"Key": object_key("2025-01-02", "minute")}],
             "IsTruncated": True, "NextContinuationToken": "next"},
            {"Contents": [{"Key": object_key("2025-01-03", "minute")},
                          {"Key": object_key("2025-01-04", "minute")}], "IsTruncated": False},
        ]
        days = period_days("2025-01-02", "2025-01-04")
        self.assertEqual(available_days(client, days), days)
        self.assertEqual(client.list_objects_v2.call_count, 2)
        self.assertEqual(client.list_objects_v2.call_args.kwargs["ContinuationToken"], "next")

    def test_listing_handles_month_boundaries(self) -> None:
        client = Mock()
        days = period_days("2025-01-31", "2025-02-02")
        client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": object_key(day.isoformat(), "minute")}], "IsTruncated": False}
            for day in days
        ]
        self.assertEqual(available_days(client, days), days)
        self.assertEqual(
            [call.kwargs["Prefix"] for call in client.list_objects_v2.call_args_list],
            ["global_forex/minute_aggs_v1/2025/01/", "global_forex/minute_aggs_v1/2025/02/"],
        )

    def test_broken_pagination_and_empty_catalogue_fail_explicitly(self) -> None:
        for pages, message in (
            ([{"IsTruncated": True}], "continuation token"),
            ([{"IsTruncated": True, "NextContinuationToken": "same"}] * 2, "continuation token"),
            ([{"IsTruncated": False}], "No S3 minute files"),
        ):
            with self.subTest(message=message, pages=len(pages)):
                client = Mock()
                client.list_objects_v2.side_effect = pages
                with self.assertRaisesRegex(ValueError, message):
                    available_days(client, period_days("2025-01-02", "2025-01-03"))

    def test_m5_aggregates_exactly_five_closed_minutes(self) -> None:
        minutes = sample_minutes()
        m5, quality = minute_to_m5(minutes)
        first = minutes.iloc[:5]
        self.assertEqual(m5.iloc[0].open, first.iloc[0].open)
        self.assertEqual(m5.iloc[0].high, first.high.max())
        self.assertEqual(m5.iloc[0].low, first.low.min())
        self.assertEqual(m5.iloc[0].close, first.iloc[-1].close)
        self.assertEqual(m5.iloc[0].volume, 5)
        self.assertEqual(m5.index[0], minutes.index[0])
        self.assertEqual(quality, {
            "m1_bars": 60, "m5_bars": 12, "incomplete_m5_bins": 0, "empty_m5_bins": 0,
        })

    def test_empty_bins_are_not_filled(self) -> None:
        minutes = sample_minutes().drop(sample_minutes().index[5:10])
        m5, quality = minute_to_m5(minutes)
        self.assertEqual(len(m5), 11)
        self.assertEqual(quality["empty_m5_bins"], 1)
        self.assertNotIn(pd.Timestamp("2025-01-02 00:05", tz="UTC"), m5.index)

    def test_incomplete_bins_are_rejected_not_fabricated(self) -> None:
        minutes = sample_minutes()
        with self.assertRaisesRegex(ValueError, "4 of 5 minutes"):
            minute_to_m5(minutes.drop(minutes.index[2]))
        with self.assertRaisesRegex(ValueError, "incomplete M5 bins"):
            minute_to_m5(minutes.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete M5 bins"):
            minute_to_m5(minutes.iloc[::5])

    def test_invalid_source_minutes_cannot_be_hidden_by_aggregation(self) -> None:
        for kind in ("bounds", "negative", "nan", "duplicate", "unsorted", "naive", "unaligned"):
            with self.subTest(kind=kind):
                frame = sample_minutes()
                if kind == "bounds":
                    frame.iloc[2, frame.columns.get_loc("high")] = 0.1
                elif kind == "negative":
                    frame.iloc[2, frame.columns.get_loc("volume")] = -1
                elif kind == "nan":
                    frame.iloc[2, frame.columns.get_loc("close")] = np.nan
                elif kind == "duplicate":
                    frame = pd.concat([frame, frame.iloc[:1]])
                elif kind == "unsorted":
                    frame = frame.iloc[::-1]
                elif kind == "naive":
                    frame.index = frame.index.tz_localize(None)
                else:
                    frame.index = frame.index + pd.Timedelta(seconds=1)
                with self.assertRaises(ValueError):
                    minute_to_m5(frame)

    def test_comparison_measures_price_differences_and_timestamp_coverage(self) -> None:
        base, _ = minute_to_m5(sample_minutes())
        s3, oanda = base.iloc[1:].copy(), base.iloc[:-1].copy()
        s3["close"] += 0.0001
        report = compare_data(s3, oanda)
        self.assertEqual(report["common_m5_bars"], 10)
        self.assertEqual(report["s3_only_m5_bars"], 1)
        self.assertEqual(report["oanda_only_m5_bars"], 1)
        self.assertEqual(report["different_ohlc_bars"], 10)
        self.assertAlmostEqual(report["absolute_price_differences"]["close"]["mean"], 0.0001)
        self.assertAlmostEqual(report["absolute_price_differences"]["close"]["max"], 0.0001)
        self.assertIsNone(report["acceptance_threshold"])

    def test_volume_is_fingerprinted_but_not_compared_as_price(self) -> None:
        base, _ = minute_to_m5(sample_minutes())
        other = base.copy()
        other["volume"] *= 10
        report = compare_data(base, other)
        self.assertEqual(report["different_ohlc_bars"], 0)
        self.assertNotEqual(report["s3_data_sha256"], report["oanda_data_sha256"])
        self.assertNotIn("volume", report["absolute_price_differences"])

    def test_nonoverlapping_sources_cannot_be_certified(self) -> None:
        first, _ = minute_to_m5(sample_minutes())
        second, _ = minute_to_m5(sample_minutes(start="2025-01-03"))
        with self.assertRaisesRegex(ValueError, "no common M5 timestamps"):
            compare_data(first, second)

    def test_full_synthetic_pipeline_exports_reproducible_strategy_results(self) -> None:
        minutes = sample_minutes(1440)
        oanda, _ = minute_to_m5(minutes)
        client = fake_client(minutes)
        expected = backtest(oanda)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            report = run_comparison(
                client, oanda, "2025-01-02", "2025-01-04", output, "synthetic OANDA fixture",
            )
            self.assertEqual(report, json.loads(output.read_text()))
            self.assertEqual(report["backtests"]["s3"]["metrics"], expected.metrics)
            self.assertEqual(report["backtests"]["oanda"]["metrics"], expected.metrics)
            self.assertEqual(report["comparison"]["s3_data_sha256"], data_fingerprint(oanda))
            self.assertEqual(report["comparison"]["different_ohlc_bars"], 0)
            self.assertEqual(report["quality"]["dates_without_listed_file"], ["2025-01-03"])
            self.assertEqual(report["sources"]["oanda"], "synthetic OANDA fixture")
            for source in ("s3", "oanda"):
                self.assertTrue(output.with_suffix(f".{source}.m5.csv").exists())
                trades = pd.read_csv(output.with_suffix(f".{source}.trades.csv"))
                self.assertEqual(list(trades.columns), list(expected.trades.columns))
                self.assertTrue(output.with_suffix(f".{source}.audit.csv").exists())
        self.assertTrue(client.get_object.return_value["Body"].closed)

    def test_access_denied_produces_no_backtest_or_partial_files(self) -> None:
        minutes = sample_minutes()
        oanda, _ = minute_to_m5(minutes)
        client = fake_client(minutes)
        client.get_object.side_effect = PermissionError("HTTP 403")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            with self.assertRaisesRegex(PermissionError, "HTTP 403"):
                run_comparison(client, oanda, "2025-01-02", "2025-01-03", output, "synthetic")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
