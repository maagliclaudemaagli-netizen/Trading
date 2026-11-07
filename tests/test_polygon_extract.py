"""Offline regression tests for the manual Polygon example."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from polygon_extract import RateLimiter, extract, main


def bar(timestamp: int) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=timestamp, open=1.1, high=1.2, low=1.0, close=1.15, volume=100,
    )


class PolygonExtractionTests(unittest.TestCase):
    def test_cli_uses_the_sdk_without_retries_or_a_context_manager(self) -> None:
        client = SimpleNamespace(get_aggs=Mock(return_value=[bar(1000)]))
        factory = Mock(return_value=client)
        with patch.dict("sys.modules", {"polygon": SimpleNamespace(RESTClient=factory)}), patch.dict(
            "os.environ", {"POLYGON_API_KEY": "offline-test"}, clear=True,
        ), patch(
            "sys.argv", ["polygon_extract.py", "--from-date", "2025-01-01", "--to-date", "2025-01-01"],
        ), patch("builtins.print") as output:
            main()
        factory.assert_called_once_with(api_key="offline-test", retries=0)
        client.get_aggs.assert_called_once()
        self.assertIn("1 bougies", output.call_args_list[0].args[0])

    def test_limits_requests_to_five_per_rolling_minute(self) -> None:
        now = 0.0
        starts = []

        def sleep(seconds: float) -> None:
            nonlocal now
            now += seconds

        with patch("polygon_extract.time.monotonic", side_effect=lambda: now), patch(
            "polygon_extract.time.sleep", side_effect=sleep,
        ):
            limiter = RateLimiter()
            for _ in range(12):
                limiter.wait()
                starts.append(now)

        self.assertEqual(starts[0], 0.0)
        for start in starts:
            self.assertLessEqual(sum(start <= value < start + 60 for value in starts), 5)

    def test_short_sleep_is_rechecked(self) -> None:
        limiter = RateLimiter()
        limiter.next_request = 12.1
        with patch(
            "polygon_extract.time.monotonic", side_effect=[0, 6, 12.1, 12.1],
        ), patch("polygon_extract.time.sleep") as sleep:
            limiter.wait()
        self.assertEqual(sleep.call_count, 2)

    def test_chunks_cover_dates_and_return_sorted_unique_utc_bars(self) -> None:
        client, limiter = Mock(), Mock()
        client.get_aggs.side_effect = [[bar(2000), bar(1000)], [bar(2000), bar(3000)]]
        frame = extract(client, "C:EURUSD", 1, "minute", "2025-01-01", "2025-01-31", limiter)
        calls = client.get_aggs.call_args_list
        self.assertEqual(limiter.wait.call_count, 2)
        self.assertEqual(calls[0].kwargs["from_"], "2025-01-01")
        self.assertEqual(calls[0].kwargs["to"], "2025-01-30")
        self.assertEqual(calls[1].kwargs["from_"], "2025-01-31")
        self.assertEqual(calls[1].kwargs["to"], "2025-01-31")
        self.assertEqual(calls[0].kwargs["limit"], 50_000)
        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        pd.testing.assert_index_equal(
            frame.index,
            pd.DatetimeIndex(pd.to_datetime([1000, 2000, 3000], unit="ms", utc=True), name="Timestamp"),
        )

    def test_empty_result_is_an_explicit_error(self) -> None:
        client = Mock()
        client.get_aggs.return_value = []
        with self.assertRaisesRegex(ValueError, "Aucune donnée"):
            extract(client, "C:EURUSD", 1, "day", "2025-01-01", "2025-01-01", Mock())

    def test_api_errors_propagate(self) -> None:
        client = Mock()
        client.get_aggs.side_effect = RuntimeError("HTTP 429")
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            extract(client, "C:EURUSD", 1, "day", "2025-01-01", "2025-01-01", Mock())

    def test_unexpected_response_is_an_explicit_error(self) -> None:
        client = Mock()
        client.get_aggs.return_value = {"error": "unexpected"}
        with self.assertRaisesRegex(TypeError, "Réponse Polygon inattendue"):
            extract(client, "C:EURUSD", 1, "day", "2025-01-01", "2025-01-01", Mock())

    def test_invalid_parameters_do_not_request_data(self) -> None:
        for ticker, multiplier, timespan, start, end in (
            ("EURUSD", 1, "day", "2025-01-01", "2025-01-02"),
            ("C:EURUSD", 0, "day", "2025-01-01", "2025-01-02"),
            ("C:EURUSD", 2, "day", "2025-01-01", "2025-01-02"),
            ("C:EURUSD", 1, "second", "2025-01-01", "2025-01-02"),
            ("C:EURUSD", 1, "day", "2025-01-02", "2025-01-01"),
            ("C:EURUSD", 1, "day", "invalid", "2025-01-01"),
        ):
            with self.subTest(ticker=ticker, multiplier=multiplier, timespan=timespan, start=start):
                client, limiter = Mock(), Mock()
                with self.assertRaises(ValueError):
                    extract(client, ticker, multiplier, timespan, start, end, limiter)
                client.get_aggs.assert_not_called()
                limiter.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
