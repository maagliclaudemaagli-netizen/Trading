"""Offline S3 tests using genuine gzip CSV structure, without credentials."""
from __future__ import annotations

import gzip
import io
import unittest
from unittest.mock import Mock

import pandas as pd

from massive_s3_extract import extract, object_key


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["C:EURUSD", "C:GBPUSD", "C:EURUSD"],
        "window_start": [1735776060000000000, 1735776000000000000, 1735776000000000000],
        "open": [1.1, 1.3, 1.1],
        "high": [1.2, 1.4, 1.2],
        "low": [1.0, 1.2, 1.0],
        "close": [1.15, 1.35, 1.15],
        "volume": [100, 200, 100],
    })


def response(frame: pd.DataFrame) -> tuple[Mock, io.BytesIO]:
    body = io.BytesIO(gzip.compress(frame.to_csv(index=False).encode()))
    client = Mock()
    client.get_object.return_value = {"Body": body}
    return client, body


class MassiveS3Tests(unittest.TestCase):
    def test_key_uses_the_forex_flat_file_layout(self) -> None:
        self.assertEqual(
            object_key("2025-01-02", "day"),
            "global_forex/day_aggs_v1/2025/01/2025-01-02.csv.gz",
        )
        self.assertEqual(
            object_key("2024-02-29", "minute"),
            "global_forex/minute_aggs_v1/2024/02/2024-02-29.csv.gz",
        )

    def test_extract_filters_the_pair_sorts_and_converts_nanoseconds_to_utc(self) -> None:
        client, body = response(sample())
        frame = extract(client, "2025-01-02", "minute", "C:EURUSD")
        client.get_object.assert_called_once_with(
            Bucket="flatfiles", Key="global_forex/minute_aggs_v1/2025/01/2025-01-02.csv.gz",
        )
        self.assertTrue(body.closed)
        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
        pd.testing.assert_index_equal(
            frame.index,
            pd.DatetimeIndex(
                pd.to_datetime([1735776000000000000, 1735776060000000000], unit="ns", utc=True),
                name="timestamp",
            ),
        )

    def test_missing_pair_is_an_explicit_error(self) -> None:
        client, body = response(sample())
        with self.assertRaisesRegex(ValueError, "Aucune bougie"):
            extract(client, "2025-01-02", "minute", "C:USDJPY")
        self.assertTrue(body.closed)

    def test_missing_column_is_an_explicit_error_and_closes_body(self) -> None:
        client, body = response(sample().drop(columns="window_start"))
        with self.assertRaisesRegex(ValueError, "Colonnes S3 manquantes"):
            extract(client, "2025-01-02", "minute", "C:EURUSD")
        self.assertTrue(body.closed)

    def test_bad_candles_are_rejected(self) -> None:
        for column, value, message in (
            ("high", 0.5, "incohérentes"),
            ("volume", -1, "incohérentes"),
            ("close", float("inf"), "non finies"),
            ("open", float("nan"), "manquantes"),
            ("window_start", 0, "hors de la date"),
            ("window_start", 1735776000000000000, "dupliqués"),
        ):
            with self.subTest(column=column, value=value):
                data = sample()
                data.loc[0, column] = value
                client, body = response(data)
                with self.assertRaisesRegex(ValueError, message):
                    extract(client, "2025-01-02", "minute", "C:EURUSD")
                self.assertTrue(body.closed)

    def test_invalid_parameters_do_not_contact_s3(self) -> None:
        for day, span, ticker in (
            ("2025-02-30", "day", "C:EURUSD"),
            ("2025-01-02", "hour", "C:EURUSD"),
            ("2025-01-02", "day", "EURUSD"),
        ):
            with self.subTest(day=day, span=span, ticker=ticker):
                client = Mock()
                with self.assertRaises(ValueError):
                    extract(client, day, span, ticker)
                client.get_object.assert_not_called()

    def test_s3_errors_propagate(self) -> None:
        client = Mock()
        client.get_object.side_effect = PermissionError("AccessDenied")
        with self.assertRaisesRegex(PermissionError, "AccessDenied"):
            extract(client, "2025-01-02", "day", "C:EURUSD")

    def test_corrupt_gzip_is_rejected_and_closed(self) -> None:
        client = Mock()
        body = io.BytesIO(b"not a gzip file")
        client.get_object.return_value = {"Body": body}
        with self.assertRaises(gzip.BadGzipFile):
            extract(client, "2025-01-02", "day", "C:EURUSD")
        self.assertTrue(body.closed)


if __name__ == "__main__":
    unittest.main()
