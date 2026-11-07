from __future__ import annotations

import json
from contextlib import chdir, redirect_stdout
from io import StringIO
import lzma
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from forex_historical_engine.analytics import report
from forex_historical_engine.budget import ArchiveBudget
from forex_historical_engine.cli import main as data_cli_main, parser as command_parser
from forex_historical_engine.config import Settings, date_range, instrument, market_open
from forex_historical_engine.downloader import AccessDenied, DukascopyClient, PacketDownloadError, Unavailable
from forex_historical_engine.engine import Engine
from forex_historical_engine.parser import decode, parse
from forex_historical_engine.processing import build_candles, clean, expected_minutes, quality
from forex_historical_engine.raw_ticks import download_raw_ticks
from institutional.__main__ import load_m5

ROOT = Path(__file__).resolve().parents[1]
DAY = pd.Timestamp("2025-01-06", tz="UTC")


def tick_blob() -> bytes:
    return lzma.compress(b"".join(
        struct.pack(">IIIff", offset, ask, bid, av, bv)
        for offset, ask, bid, av, bv in (
            (0, 110002, 110000, 2.0, 3.0),
            (0, 110004, 110001, 2.5, 3.5),
            (61000, 110003, 110002, 4.0, 5.0),
        )
    ))


def candle_blob(side: str = "bid", count: int = 12) -> bytes:
    adjustment = 2 if side == "ask" else 0
    return lzma.compress(b"".join(
        struct.pack(">IIIIIf", offset * 60, 110000 + offset + adjustment,
                    110001 + offset + adjustment, 109999 + offset + adjustment,
                    110002 + offset + adjustment, 5.0)
        for offset in range(count)
    ))


def sample_candles(day: pd.Timestamp = DAY, count: int = 12) -> pd.DataFrame:
    return pd.concat([
        parse(candle_blob(side, count), "M1", day, "EURUSD", 100000, f"{day}/{side}", side)
        for side in ("bid", "ask")
    ], ignore_index=True).sort_values(["timestamp", "side"], kind="stable").reset_index(drop=True)


def response(status: int = 200, content: bytes | None = None) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = candle_blob() if content is None else content
    result.url = "https://datafeed.dukascopy.com/test"
    return result


class SyntheticProvider:
    url = staticmethod(DukascopyClient.url)

    def __init__(self, root: Path):
        self.root = root

    def archive(self, pair, at, kind, side=None):
        url = self.url(pair, at, kind, side)
        if kind == "tick":
            content = tick_blob()
        else:
            if side is None:
                raise ValueError("Cote requis.")
            content = candle_blob(side)
        path = self.root / "raw" / url.removeprefix("https://datafeed.dukascopy.com/datafeed/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return content, url, path


class ParserTests(unittest.TestCase):
    def test_tick_layout_precision_and_simultaneous_events(self):
        frame = parse(tick_blob(), "tick", DAY, "EURUSD", 100000, "archive")
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.timestamp.iloc[0], frame.timestamp.iloc[1])
        self.assertEqual(frame.timestamp.iloc[2], DAY + pd.Timedelta(seconds=61))
        self.assertEqual(frame.bid_raw.tolist(), [110000, 110001, 110002])
        self.assertEqual(frame.ask_volume.tolist(), [2, 2.5, 4])
        self.assertEqual(quality(frame, "tick")["flags"]["simultaneous_distinct_ticks"], 2)
        self.assertEqual(frame.source_record.tolist(), [0, 1, 2])

    def test_candle_order_is_open_close_low_high(self):
        frame = parse(candle_blob(), "M1", DAY, "EURUSD", 100000, "archive", "bid")
        self.assertEqual(frame.iloc[0][["open_raw", "close_raw", "low_raw", "high_raw"]].tolist(),
                         [110000, 110001, 109999, 110002])
        self.assertEqual(frame.timestamp.iloc[-1], DAY + pd.Timedelta(minutes=11))
        self.assertFalse(frame.reconstructed.any())

    def test_invalid_empty_and_truncated_archives(self):
        for content in (b"", b"not lzma", lzma.compress(b""), lzma.compress(b"a")):
            with self.subTest(content=content), self.assertRaises(ValueError):
                decode(content, "tick")

    def test_offsets_and_quote_side_are_validated(self):
        invalid = lzma.compress(struct.pack(">IIIff", 3600000, 1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "heure"):
            parse(invalid, "tick", DAY, "EURUSD", 100000, "archive")
        with self.assertRaisesRegex(ValueError, "cote"):
            parse(candle_blob(), "M1", DAY, "EURUSD", 100000, "archive")
        invalid = lzma.compress(struct.pack(">IIIIIf", 1, 1, 1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "Offset"):
            parse(invalid, "M1", DAY, "EURUSD", 100000, "archive", "bid")


class ProcessingTests(unittest.TestCase):
    def test_ohlc_and_mid_are_built_per_tick(self):
        frame = parse(tick_blob(), "tick", DAY, "EURUSD", 100000, "archive")
        bars = build_candles(frame, "M1", True)
        first = bars.loc[bars.timestamp == DAY].set_index("side")
        self.assertEqual(first.loc["bid", "open"], 1.1)
        self.assertEqual(first.loc["bid", "close"], 1.10001)
        np.testing.assert_allclose(first.loc[["mid"], ["high"]].to_numpy(dtype=float), (1.10004 + 1.10001) / 2)
        self.assertTrue(bars.volume.isna().all())
        self.assertEqual(first.loc["bid", "source_count"], 2)
        provenance = first.loc["bid", "input_sources"]
        if not isinstance(provenance, str):
            self.fail("Provenance JSON textuelle attendue.")
        self.assertEqual(json.loads(provenance), ["archive"])

    def test_all_timeframes_align_and_never_fill_empty_bins(self):
        frame = sample_candles()
        for timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
            with self.subTest(timeframe=timeframe):
                bars = build_candles(frame, timeframe)
                self.assertEqual(set(bars.side), {"bid", "ask"})
                self.assertTrue(bars.reconstructed.all())
                self.assertTrue((bars.closed_at > bars.timestamp).all())
        separated = pd.concat([frame.iloc[:2], frame.iloc[-2:]]).sort_values("timestamp")
        bars = build_candles(separated, "M1")
        self.assertEqual(len(bars), 4)
        weekly = build_candles(frame, "W1")
        self.assertEqual(weekly.timestamp.iloc[0], DAY)
        monthly = build_candles(frame, "MN1")
        self.assertEqual(monthly.timestamp.iloc[0], pd.Timestamp("2025-01-01", tz="UTC"))

    def test_reconstruction_rejects_ambiguous_candles_and_unordered_ticks(self):
        frame = sample_candles()
        with self.assertRaisesRegex(ValueError, "dupliquees"):
            build_candles(pd.concat([frame, frame]).sort_values("timestamp"), "M5")
        ticks = parse(tick_blob(), "tick", DAY, "EURUSD", 100000, "archive")
        with self.assertRaisesRegex(ValueError, "ordonnes"):
            build_candles(ticks.iloc[::-1], "M1", True)

    def test_quality_and_explicit_cleaning_leave_input_unchanged(self):
        frame = sample_candles()
        frame.loc[0, "high"] = 0
        frame.loc[1, "close"] = np.nan
        original = frame.copy(deep=True)
        stats = quality(frame, "M1")
        self.assertEqual(stats["bad_rows"], 2)
        filtered, audit = clean(frame, "M1")
        self.assertEqual(len(audit), 2)
        self.assertEqual(len(filtered), len(frame) - 2)
        pd.testing.assert_frame_equal(frame, original)
        self.assertIn("reason", audit)

    def test_duplicate_ticks_are_not_silently_removed(self):
        frame = parse(tick_blob(), "tick", DAY, "EURUSD", 100000, "archive")
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        filtered, audit = clean(frame, "tick")
        self.assertEqual(len(filtered), 4)
        self.assertTrue(audit.empty)

    def test_negative_spreads_are_detected(self):
        frame = parse(tick_blob(), "tick", DAY, "EURUSD", 100000, "archive")
        frame.loc[0, "ask"] = 1.0
        self.assertEqual(quality(frame, "tick")["flags"]["negative_spreads"], 1)

    def test_calendar_handles_weekend_dst_and_explicit_holidays(self):
        index = pd.DatetimeIndex([
            "2025-01-05T21:59Z", "2025-01-05T22:00Z",
            "2025-03-09T20:59Z", "2025-03-09T21:00Z",
            "2025-03-14T20:59Z", "2025-03-14T21:00Z",
            "2025-03-15T12:00Z",
        ])
        self.assertEqual(market_open(index).tolist(), [False, True, False, True, True, False, False])
        self.assertEqual(len(expected_minutes(DAY, DAY + pd.Timedelta(days=1), ("2025-01-06",))), 0)

    def test_instruments_dates_and_settings_are_checked(self):
        self.assertEqual(instrument("USD/JPY"), ("USDJPY", 1000))
        self.assertEqual(instrument("EURUSD"), ("EURUSD", 100000))
        self.assertEqual(instrument("XAUUSD", 1000), ("XAUUSD", 1000))
        for pair in ("XAUUSD", "../bad", "éURUSD"):
            with self.assertRaises(ValueError):
                instrument(pair)
        for start, end in (("2025-02-01", "2025-01-01"), ("2025-01-01T01:00Z", "2025-01-02")):
            with self.assertRaises(ValueError):
                date_range(start, end)
        with self.assertRaises(ValueError):
            date_range("2025-01-01", pd.Timestamp.now(tz="UTC").normalize())
        with self.assertRaises(ValueError):
            Settings(delay=0)
        for invalid_delay in (0.5, float("nan")):
            with self.assertRaises(ValueError):
                Settings(delay=invalid_delay)
        with self.assertRaises(ValueError):
            Settings(retries=4)


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        sleeper = patch("forex_historical_engine.downloader.time.sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        self.client = DukascopyClient(Settings(root=Path(self.directory.name), retries=1))

    def test_urls_use_zero_based_month_and_utc_hour(self):
        self.assertTrue(self.client.url("EURUSD", DAY, "M1", "ask").endswith("/2025/00/06/ASK_candles_min_1.bi5"))
        self.assertTrue(self.client.url("EURUSD", DAY + pd.Timedelta(hours=12), "tick").endswith("/12h_ticks.bi5"))

    @patch("forex_historical_engine.downloader.time.sleep")
    @patch("forex_historical_engine.downloader.requests.get")
    def test_retry_server_error_then_validate_content(self, get, _sleep):
        get.side_effect = [response(503), response()]
        self.assertEqual(self.client.fetch("archive", "M1"), candle_blob())
        self.assertEqual(get.call_count, 2)
        self.assertTrue(_sleep.called)

    @patch("forex_historical_engine.downloader.requests.get")
    def test_access_restrictions_are_not_retried(self, get):
        for status in (401, 403):
            get.reset_mock()
            get.return_value = response(status)
            self.client = DukascopyClient(Settings(root=Path(self.directory.name), retries=1))
            with self.assertRaises(AccessDenied):
                self.client.fetch("archive", "M1")
            self.assertEqual(get.call_count, 1)
            with self.assertRaises(AccessDenied):
                self.client.fetch("another_archive", "M1")
            self.assertEqual(get.call_count, 1)

    @patch("forex_historical_engine.downloader.requests.get")
    def test_empty_and_missing_archives_are_not_success(self, get):
        for result in (response(404), response(204), response(200, b"")):
            get.return_value = result
            with self.assertRaises(Unavailable):
                self.client.fetch("archive", "M1")

    @patch("forex_historical_engine.downloader.time.sleep")
    @patch("forex_historical_engine.downloader.requests.get")
    def test_corrupt_http_200_is_retried_then_rejected(self, get, _sleep):
        get.return_value = response(200, b"garbage")
        with self.assertRaises(PacketDownloadError):
            self.client.fetch("archive", "M1")
        self.assertEqual(get.call_count, 2)
        self.assertTrue(_sleep.called)

    @patch("forex_historical_engine.downloader.requests.get")
    def test_restart_reuses_raw_without_network_and_keeps_bytes(self, get):
        get.return_value = response()
        first, _, path = self.client.archive("EURUSD", DAY, "M1", "bid")
        second, _, _ = self.client.archive("EURUSD", DAY, "M1", "bid")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(path.read_bytes(), first)
        self.assertFalse(path.with_suffix(".bi5.part").exists())

    @patch("forex_historical_engine.downloader.requests.get")
    def test_429_retries_at_exact_delays_then_succeeds(self, get):
        self.client = DukascopyClient(Settings(root=Path(self.directory.name)))
        get.side_effect = [response(429), response(429), response(429), response()]
        self.assertEqual(self.client.fetch("archive", "M1"), candle_blob())
        self.assertEqual(get.call_count, 4)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1.0, 5.0, 1.0, 15.0, 1.0, 30.0, 1.0])

    @patch("forex_historical_engine.downloader.requests.get")
    def test_exhausted_429_does_not_block_the_next_packet(self, get):
        self.client = DukascopyClient(Settings(root=Path(self.directory.name)))
        get.side_effect = [response(429)] * 4 + [response()]
        with self.assertRaises(PacketDownloadError):
            self.client.fetch("first", "M1")
        self.assertEqual(get.call_count, 4)
        self.assertEqual(self.client.fetch("next", "M1"), candle_blob())
        self.assertEqual(get.call_count, 5)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1.0, 5.0, 1.0, 15.0, 1.0, 30.0, 1.0, 1.0])

    @patch("forex_historical_engine.downloader.requests.get")
    def test_429_respects_a_longer_retry_after(self, get):
        limited = response(429)
        limited.headers["Retry-After"] = "120"
        get.side_effect = [limited, response()]
        self.client.fetch("archive", "M1")
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1.0, 120.0, 1.0])

    @patch("forex_historical_engine.downloader.requests.get")
    def test_explicit_bot_block_is_not_retried(self, get):
        get.return_value = response(429, b'{"error":"Too Many Requests (Bot blocked)."}')
        with self.assertRaises(AccessDenied):
            self.client.fetch("archive", "M1")
        with self.assertRaises(AccessDenied):
            self.client.fetch("next", "M1")
        self.assertEqual(get.call_count, 1)

    @patch("forex_historical_engine.downloader.requests.get")
    def test_every_http_request_has_a_systematic_one_second_pause(self, get):
        def receive(*args, **kwargs):
            self.assertIn(args[0], ("first", "next"))
            self.assertEqual(kwargs["timeout"], self.client.settings.timeout)
            self.assertEqual(self.sleep.call_args.args, (1.0,))
            return response()

        get.side_effect = receive
        self.client.fetch("first", "M1")
        self.client.fetch("next", "M1")
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1.0, 1.0])


class RawTickBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.client = DukascopyClient(Settings(root=self.root, retries=0))
        self.content = tick_blob()

    def test_limit_counts_existing_files_and_resume_reuses_archives(self):
        existing = self.root / "keep.txt"
        existing.write_bytes(b"preserve")
        stored = existing.stat().st_size + 2 * len(self.content)
        limit = stored + len(self.content)
        budget = ArchiveBudget(self.root, limit)
        original_link = os.link
        peaks = []

        def publish(source, destination):
            original_link(source, destination)
            total = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
            peaks.append(total)
            self.assertLessEqual(total, limit)

        with patch.object(self.client, "fetch", return_value=self.content) as fetch, patch(
            "forex_historical_engine.budget.os.link", side_effect=publish,
        ):
            result = download_raw_ticks(self.client, budget, ["EURUSD", "GBPUSD"], "2025-01-06", "2025-01-06")
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(result["stop_reason"], "storage_limit")
        self.assertFalse(result["complete_archive_request"])
        self.assertEqual(result["stored_bytes"], stored)
        self.assertEqual(max(peaks), limit)
        self.assertTrue(result["next_archive"].endswith("/EURUSD/2025/00/06/01h_ticks.bi5"))
        self.assertEqual([item["downloaded_archives"] for item in result["instruments"]], [1, 1])
        self.assertEqual(existing.read_bytes(), b"preserve")
        self.assertEqual(sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()), stored)
        self.assertEqual(len(list(self.root.rglob("*.bi5"))), 2)
        self.assertFalse(list(self.root.rglob("*.part")))

        with patch.object(self.client, "fetch", return_value=self.content) as fetch:
            resumed = download_raw_ticks(
                self.client, ArchiveBudget(self.root, limit), ["EURUSD", "GBPUSD"], "2025-01-06", "2025-01-06",
            )
        fetch.assert_called_once()
        self.assertEqual([item["reused_archives"] for item in resumed["instruments"]], [1, 1])
        self.assertEqual(resumed["stored_bytes"], stored)

    def test_archive_larger_than_remaining_budget_is_not_written(self):
        budget = ArchiveBudget(self.root, len(self.content) - 1)
        with patch.object(self.client, "fetch", return_value=self.content) as fetch:
            result = download_raw_ticks(self.client, budget, ["EURUSD"], "2025-01-06", "2025-01-06")
        fetch.assert_called_once()
        self.assertEqual(result["stored_bytes"], 0)
        self.assertEqual(result["stop_reason"], "storage_limit")
        self.assertFalse(list(self.root.rglob("*.bi5")))
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_unavailable_archive_is_reported_without_claiming_success(self):
        budget = ArchiveBudget(self.root, 2 * len(self.content))
        with patch.object(self.client, "fetch", side_effect=[Unavailable("archive absente"), self.content, self.content]):
            result = download_raw_ticks(self.client, budget, ["EURUSD", "GBPUSD"], "2025-01-06", "2025-01-06")
        self.assertEqual(result["instruments"][0]["unavailable_archives"], 1)
        self.assertEqual(result["instruments"][1]["downloaded_archives"], 1)
        self.assertFalse(result["complete_archive_request"])

    def test_complete_day_preserves_publication_margin_and_reports_all_hours(self):
        stored = 24 * len(self.content)
        limit = stored + len(self.content)
        with patch.object(self.client, "fetch", return_value=self.content) as fetch:
            result = download_raw_ticks(
                self.client, ArchiveBudget(self.root, limit), ["EURUSD"], "2025-01-06", "2025-01-06",
            )
        self.assertEqual(fetch.call_count, 24)
        self.assertTrue(result["scanned_all_hours"])
        self.assertTrue(result["complete_archive_request"])
        self.assertIsNone(result["next_archive"])
        self.assertEqual(result["instruments"][0]["downloaded_archives"], 24)
        self.assertEqual(result["stored_bytes"], stored)
        self.assertEqual(sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()), stored)

    def test_cli_provider_restriction_is_an_error_and_releases_lock(self):
        stdout = StringIO()
        with patch.object(DukascopyClient, "fetch", side_effect=AccessDenied("acces refuse")), redirect_stdout(stdout):
            with self.assertLogs("forex_historical_engine.cli", level="ERROR"):
                exit_code = data_cli_main([
                    "download-ticks", "--pair", "EURUSD",
                    "--start", "2025-01-06", "--end", "2025-01-06",
                    "--max-bytes", "1024", "--output-dir", str(self.root),
                ])
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse((self.root / ".download-ticks.lock").exists())
        self.assertFalse(list(self.root.rglob("*.bi5")))

    def test_existing_files_and_symlinks_are_not_removed(self):
        existing = self.root / "keep.txt"
        existing.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "plus de donnees"):
            ArchiveBudget(self.root, len(b"preserve") - 1)
        link = self.root / "link"
        link.symlink_to(existing)
        with self.assertRaisesRegex(ValueError, "Lien symbolique"):
            ArchiveBudget(self.root, 1024)
        self.assertEqual(existing.read_bytes(), b"preserve")

    def test_existing_partial_file_is_never_overwritten(self):
        partial = self.root / "raw/EURUSD/2025/00/06/00h_ticks.bi5.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"existing partial")
        budget = ArchiveBudget(self.root, 1024)
        with patch.object(self.client, "fetch", return_value=self.content), self.assertRaises(FileExistsError):
            download_raw_ticks(self.client, budget, ["EURUSD"], "2025-01-06", "2025-01-06")
        self.assertEqual(partial.read_bytes(), b"existing partial")

    def test_limit_is_rechecked_after_lock_and_concurrent_download_is_rejected(self):
        budget = ArchiveBudget(self.root, len(self.content))
        existing = self.root / "created_after_budget"
        existing.write_bytes(b"data")
        with budget.exclusive():
            other = ArchiveBudget(self.root, len(self.content))
            with self.assertRaisesRegex(RuntimeError, "Telechargement deja actif"):
                download_raw_ticks(self.client, other, ["EURUSD"], "2025-01-06", "2025-01-06")
        with patch.object(self.client, "fetch", return_value=self.content):
            result = download_raw_ticks(self.client, budget, ["EURUSD"], "2025-01-06", "2025-01-06")
        self.assertEqual(result["stored_bytes"], len(b"data"))
        self.assertEqual(result["stop_reason"], "storage_limit")
        self.assertEqual(existing.read_bytes(), b"data")
        self.assertFalse((self.root / ".download-ticks.lock").exists())

    def test_cli_includes_gold_without_inventing_its_price_scale(self):
        stdout = StringIO()
        with patch.object(DukascopyClient, "fetch", return_value=self.content), redirect_stdout(stdout):
            exit_code = data_cli_main([
                "download-ticks", "--pairs", "all", "--include-xauusd",
                "--start", "2025-01-06", "--end", "2025-01-06",
                "--max-bytes", str(2 * len(self.content)), "--output-dir", str(self.root),
            ])
        result = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(result["instruments"][0]["instrument"], "XAUUSD")
        self.assertEqual(len(result["instruments"]), 24)
        self.assertEqual(result["stored_bytes"], len(self.content))
        self.assertTrue((self.root / "raw/XAUUSD/2025/00/06/00h_ticks.bi5").is_file())
        self.assertFalse((self.root / "metadata.sqlite").exists())
        with self.assertRaisesRegex(ValueError, "price-scale"):
            instrument("XAUUSD")


class StorageEngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        network = patch("forex_historical_engine.downloader.requests.get", side_effect=AssertionError("Internet interdit dans les tests unitaires."))
        network.start()
        self.addCleanup(network.stop)
        self.settings = Settings(root=self.root)
        self.engine = Engine(self.settings, SyntheticProvider(self.root))
        self.storage = self.engine.storage

    def test_parquet_roundtrip_and_idempotent_archive_upsert(self):
        frame = sample_candles()
        paths = self.storage.write(frame, "EURUSD", "M1")
        self.storage.write(frame, "EURUSD", "M1")
        actual = self.storage.read("EURUSD", "M1").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, frame)
        self.assertEqual(paths[0], self.root / "processed/EURUSD/M1/2025/01.parquet")

    def test_ticks_preserve_same_timestamp_events_and_chunk_restart(self):
        frame = parse(tick_blob(), "tick", DAY, "EURUSD", 100000, "archive")
        self.storage.write(frame, "EURUSD", "tick", "2025-01-06-00")
        self.storage.write(frame, "EURUSD", "tick", "2025-01-06-00")
        self.assertEqual(sum(len(part) for _, part in self.storage.partitions("EURUSD", "tick")), 3)
        with self.assertRaisesRegex(ValueError, "globale"):
            self.storage.read("EURUSD", "tick")

    def test_download_rebuild_report_and_bid_csv_for_existing_backtest(self):
        result = self.engine.download(["EURUSD", "GBPUSD"], "M5", "2025-01-06", "2025-01-06")
        self.assertTrue(all(item["complete_archive_request"] for item in result["results"]))
        export = self.root / "exports/EURUSD_M5_bid.csv"
        self.assertEqual(self.storage.export("EURUSD", "M5", export, "bid"), 3)
        loaded = load_m5(export, None, None)
        self.assertEqual(len(loaded), 3)
        output = self.root / "backtest.json"
        process = subprocess.run(
            [sys.executable, "-B", "-m", "institutional", "--csv", str(export),
             "--pair", "EURUSD", "--price-basis", "bid", "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(output.read_text())["m5_bars"], 3)
        self.engine.download(["EURUSD"], "M5", "2025-01-06", "2025-01-06")
        self.assertEqual(len(self.storage.read("EURUSD", "M1")), 24)

    def test_unavailable_is_persisted_and_reported_as_incomplete(self):
        provider = Mock()
        provider.url = DukascopyClient.url
        provider.archive.side_effect = Unavailable("not available")
        engine = Engine(self.settings, provider)
        result = engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
        self.assertFalse(result["results"][0]["complete_archive_request"])
        self.assertEqual(result["results"][0]["unavailable_archives"], 2)
        record = engine.storage.download(DukascopyClient.url("EURUSD", DAY, "M1", "bid"))
        if record is None:
            self.fail("Statut de telechargement manquant.")
        self.assertEqual(record["status"], "unavailable")

    def test_mid_reconstruction_and_higher_bars_do_not_mix_with_original_m1(self):
        self.engine.download(["EURUSD"], "tick", "2025-01-06", "2025-01-06")
        self.engine.rebuild("EURUSD", "M5", True)
        self.assertFalse(self.storage.files("EURUSD", "M1"))
        bars = self.storage.read("EURUSD", "M5_from_ticks")
        self.assertEqual(set(bars.side), {"bid", "ask", "mid"})
        self.assertTrue(bars.volume.isna().all())
        self.engine.validate("EURUSD", "M5_from_ticks")

    def test_weekly_rebuild_crosses_month_and_year_boundaries(self):
        for day in (pd.Timestamp("2024-12-31", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")):
            self.storage.write(sample_candles(day), "EURUSD", "M1")
        self.engine.rebuild("EURUSD", "W1")
        bars = self.storage.read("EURUSD", "W1")
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars.timestamp.iloc[0], pd.Timestamp("2024-12-30", tz="UTC"))
        self.assertEqual(bars.source_count.tolist(), [24, 24])
        self.assertEqual(len(json.loads(bars.input_sources.iloc[0])), 2)
        self.engine.rebuild("EURUSD", "W1")
        self.assertEqual(len(self.storage.read("EURUSD", "W1")), 2)

    def test_weekly_coverage_counts_a_cross_month_week_once(self):
        self.engine.download(["EURUSD"], "W1", "2024-12-31", "2025-01-01")
        result = report(self.storage, "EURUSD", "W1")
        self.assertEqual(result["expected_bar_observations"], 2)
        self.assertEqual(result["nominal_coverage_pct"], 100)
        self.assertEqual(result["missing_bar_candidates"], 0)

    def test_exact_spread_median_and_gap_candidates(self):
        frame = sample_candles()
        self.storage.write(frame, "EURUSD", "M1")
        result = report(self.storage, "EURUSD", "M1")
        aligned = frame.pivot(index="timestamp", columns="side", values="close")
        self.assertAlmostEqual(result["spread"]["median"], float((aligned.ask - aligned.bid).median()))
        self.assertEqual(result["quality_score"], 100)
        self.assertEqual(result["missing_minute_candidates"], 2 * (1440 - 12))
        self.assertEqual(result["spread"]["observations"], 12)
        self.assertAlmostEqual(result["spread_by_hour_utc"]["0"]["median"], result["spread"]["median"])
        json.dumps(result, allow_nan=False)

    def test_checksum_changes_are_rejected_even_after_failed_retry(self):
        self.engine._archive("EURUSD", 100000, DAY, "M1", "bid")
        record = self.storage.download(DukascopyClient.url("EURUSD", DAY, "M1", "bid"))
        if record is None:
            self.fail("Archive verifiee manquante.")
        Path(record["raw_path"]).write_bytes(candle_blob(count=3))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "modifiee"):
                self.engine._archive("EURUSD", 100000, DAY, "M1", "bid")

    @patch("forex_historical_engine.downloader.time.sleep")
    @patch("forex_historical_engine.downloader.requests.get")
    def test_exhausted_packet_is_logged_and_next_packet_is_saved(self, get, _sleep):
        get.side_effect = [response(429)] * 4 + [response(content=candle_blob("ask"))]
        engine = Engine(self.settings)
        events = []
        with self.assertLogs("forex_historical_engine", level="WARNING") as logs:
            result = engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06", progress=events.append)
        summary = result["results"][0]
        self.assertEqual(get.call_count, 5)
        self.assertTrue(_sleep.called)
        self.assertEqual(summary["failed_archives"], 1)
        self.assertEqual(summary["verified_archives"], 1)
        self.assertFalse(summary["complete_archive_request"])
        self.assertEqual([event["status"] for event in events], ["failed", "verified"])
        self.assertTrue(any("Paquet ignore" in line for line in logs.output))
        record = engine.storage.download(DukascopyClient.url("EURUSD", DAY, "M1", "bid"))
        if record is None:
            self.fail("Paquet abandonne non journalise en base.")
        self.assertEqual(record["status"], "failed")
        cache = engine.settings.cache_root / "EURUSD/M1/2025/01/2025-01-06-ask.parquet"
        self.assertTrue(cache.exists())

    def test_custom_parquet_cache_restores_outputs_without_raw_or_network(self):
        cache = self.root / "custom-cache"
        settings = Settings(root=self.root, cache=cache)
        engine = Engine(settings, SyntheticProvider(self.root))
        engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
        for path in (self.root / "raw").glob("**/*.bi5"):
            path.unlink()
        for path in engine.storage.files("EURUSD", "M1"):
            path.unlink()
        provider = Mock()
        provider.url = DukascopyClient.url
        provider.archive.side_effect = AssertionError("Le cache doit eviter la requete reseau.")
        resumed = Engine(settings, provider)
        result = resumed.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
        self.assertTrue(result["results"][0]["complete_archive_request"])
        provider.archive.assert_not_called()
        self.assertEqual(len(resumed.storage.read("EURUSD", "M1")), 24)
        self.assertEqual(len(list(cache.glob("**/*.parquet"))), 2)
        self.assertFalse(list(cache.glob("**/*.part")))

    def test_packet_is_cached_before_an_interrupted_monthly_merge(self):
        with patch.object(self.storage, "write", side_effect=OSError("Merge interrompu")):
            with self.assertRaisesRegex(OSError, "interrompu"):
                self.engine._archive("EURUSD", 100000, DAY, "M1", "bid")
        cache = self.settings.cache_root / "EURUSD/M1/2025/01/2025-01-06-bid.parquet"
        self.assertTrue(cache.exists())
        for path in (self.root / "raw").glob("**/*.bi5"):
            path.unlink()
        provider = Mock()
        provider.url = DukascopyClient.url
        provider.archive.side_effect = AssertionError("Aucune requete pour un paquet deja en cache.")
        resumed = Engine(self.settings, provider)
        result = resumed._archive("EURUSD", 100000, DAY, "M1", "bid")
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["records"], 12)
        provider.archive.assert_not_called()

    def test_cache_option_is_accepted_before_or_after_download_command(self):
        for arguments in (
            ["--cache", str(self.root / "cache"), "download", "--pair", "EURUSD"],
            ["download", "--pair", "EURUSD", "--cache", str(self.root / "cache")],
        ):
            parsed = command_parser().parse_args(arguments)
            self.assertEqual(parsed.cache, self.root / "cache")
            self.assertEqual(parsed.delay, 1.0)
            self.assertEqual(parsed.retries, 3)

    def test_cleaning_has_audit_and_keeps_processed_untouched(self):
        frame = sample_candles()
        frame.loc[0, "low"] = -1
        path = self.storage.write(frame, "EURUSD", "M1")[0]
        original = path.read_bytes()
        result = self.engine.clean("EURUSD", "M1")
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(Path(result["audit_files"][0]).exists())

    def test_availability_is_bounded_and_does_not_claim_all_time_history(self):
        result = self.engine.availability("EURUSD", "2025-01-06", "2025-01-07")
        self.assertEqual(set(result["first_observed_M1_archive"]), {"bid", "ask"})
        self.assertIn("not a certified", result["scope"])

    def test_cli_status_and_invalid_date_exit_codes(self):
        for arguments, expected in (
            (["status"], 0),
            (["download", "--pair", "EURUSD", "--start", "2025-01-07", "--end", "2025-01-06"], 1),
        ):
            process = subprocess.run(
                [sys.executable, "-B", "main.py", "--data-dir", str(self.root), *arguments],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, expected, process.stderr)

    def test_parquet_corruption_is_reported_instead_of_silent_repair(self):
        path = self.storage.write(sample_candles(), "EURUSD", "M1")[0]
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Integrite Parquet"):
            self.storage.read("EURUSD", "M1")

    def test_export_cannot_overwrite_raw_or_processed(self):
        self.storage.write(sample_candles(), "EURUSD", "M1")
        for directory in ("raw", "processed", "cleaned"):
            with self.assertRaisesRegex(ValueError, "preserver"):
                self.storage.export("EURUSD", "M1", self.root / directory / "output.csv")

    def test_resume_recreates_missing_processed_output_without_duplicate_records(self):
        self.engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
        self.storage.files("EURUSD", "M1")[0].unlink()
        self.engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
        self.assertEqual(len(self.storage.read("EURUSD", "M1")), 24)

    def test_full_missing_day_counts_in_requested_coverage(self):
        self.engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
        at = DAY + pd.Timedelta(days=1)
        for side in ("bid", "ask"):
            self.storage.record(DukascopyClient.url("EURUSD", at, "M1", side), "EURUSD", at, "unavailable", error="404")
        result = report(self.storage, "EURUSD", "M1")
        self.assertEqual(result["expected_minute_observations"], 2 * 2 * 1440)
        self.assertIn("2025-01-07", result["missing_day_candidates"])
        self.assertEqual(len(result["unavailable_or_failed_archives"]), 2)

    def test_streamlit_initial_page_and_local_data_view(self):
        from streamlit.testing.v1 import AppTest

        with chdir(self.root):
            app = AppTest.from_file(str(ROOT / "forex_historical_engine/dashboard.py")).run()
            self.assertEqual(len(app.exception), 0)
            engine = Engine(Settings(root=self.root / "data"), SyntheticProvider(self.root / "data"))
            engine.download(["EURUSD"], "M1", "2025-01-06", "2025-01-06")
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertGreaterEqual(len(app.dataframe), 4)
            app.date_input[0].set_value(DAY.date())
            app.date_input[1].set_value(DAY.date())
            app.selectbox[0].select("M1")
            app.button[0].click().run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.success), 1)
            app.button[2].click().run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue((self.root / "data/exports/EURUSD_M1.csv").exists())


@unittest.skipUnless(os.environ.get("DUKASCOPY_INTEGRATION") == "1", "Integration reseau activable explicitement.")
class DukascopyIntegrationTests(unittest.TestCase):
    def test_real_m1_bid_ask_and_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Settings(root=Path(directory)))
            result = engine.download(["EURUSD"], "M5", "2025-01-06", "2025-01-06")
            self.assertTrue(result["results"][0]["complete_archive_request"])
            source = engine.storage.read("EURUSD", "M1")
            self.assertEqual(len(source), 2880)
            self.assertEqual(set(source.side), {"bid", "ask"})
            self.assertEqual(len(engine.storage.read("EURUSD", "M5")), 576)
            at = DAY + pd.Timedelta(hours=12)
            tick_result = engine._archive("EURUSD", 100000, at, "tick", None)
            self.assertGreater(tick_result["records"], 0)
            engine.rebuild("EURUSD", "M1", True)
            report_path = engine.validate("EURUSD", "tick")
            self.assertEqual(json.loads(report_path.read_text())["flags"]["negative_spreads"], 0)


if __name__ == "__main__":
    unittest.main()
