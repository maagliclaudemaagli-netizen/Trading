"""Synthetic regression tests; no market data or network access required."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from institutional import backtest
from institutional.__main__ import load_m5
from institutional.backtest import TRADE_COLUMNS, data_fingerprint, performance
from institutional.market import closed_bars, validate_m5

ROOT = Path(__file__).resolve().parents[1]


def sample_m5(periods: int = 576, start: str = "2026-01-01") -> pd.DataFrame:
    phase = np.arange(periods)
    opening = 1.1 + 0.002 * np.sin(phase / 15)
    closing = opening + 0.0002 * np.sin(phase / 3)
    return pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, closing) + 0.0001,
            "low": np.minimum(opening, closing) - 0.0001,
            "close": closing,
            "volume": np.ones(periods),
        },
        index=pd.date_range(start, periods=periods, freq="5min", tz="UTC", name="timestamp"),
    )


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "institutional", *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


class InstitutionalTests(unittest.TestCase):
    def test_closed_candles_are_available_only_at_their_close(self) -> None:
        for timeframe, periods in (("M15", 3), ("H1", 12), ("H4", 48), ("D1", 288)):
            with self.subTest(timeframe=timeframe):
                data = sample_m5(periods)
                self.assertTrue(closed_bars(data.iloc[:-1], timeframe).empty)
                closed = closed_bars(data, timeframe)
                self.assertEqual(len(closed), 1)
                self.assertEqual(closed.index[0], data.index[-1] + pd.Timedelta(minutes=5))
                self.assertEqual(closed.iloc[0].opened_at, data.index[0])
                self.assertEqual(closed.iloc[0].close, data.iloc[-1].close)

    def test_closed_candles_do_not_shift_across_the_weekend(self) -> None:
        friday = sample_m5(48, "2026-01-02 16:00")
        monday = sample_m5(48, "2026-01-05 00:00")
        closed = closed_bars(pd.concat([friday, monday]), "H4")
        expected = pd.DatetimeIndex(["2026-01-02 20:00", "2026-01-05 04:00"], tz="UTC")
        pd.testing.assert_index_equal(closed.index, expected, check_names=False)

    def test_malformed_candles_are_rejected(self) -> None:
        data = sample_m5(12)
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_m5(data.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_m5(data.drop(columns="volume"))

    def test_performance_includes_the_initial_capital_in_drawdown(self) -> None:
        metrics = performance(np.array([-1.0, 2.0, -1.0]))
        self.assertEqual(metrics["trades"], 3)
        self.assertEqual(metrics["expectancy_r"], 0.0)
        self.assertAlmostEqual(metrics["win_rate_pct"], 100 / 3)
        self.assertEqual(metrics["profit_factor"], 1.0)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], 1.0)
        self.assertAlmostEqual(metrics["final_equity"], 9997.02)

    def test_no_trades_have_no_invented_performance(self) -> None:
        metrics = performance(np.array([]))
        self.assertEqual(metrics["trades"], 0)
        self.assertEqual(metrics["final_equity"], 10_000)
        for key in ("expectancy_r", "win_rate_pct", "profit_factor"):
            self.assertIsNone(metrics[key])

    def test_cli_exports_the_current_backtest_results(self) -> None:
        data = sample_m5()
        expected = backtest(data)
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "synthetic_m5.csv"
            output = Path(directory) / "reports" / "institutional.json"
            data.to_csv(csv)
            loaded = load_m5(csv, None, None, None)
            pd.testing.assert_frame_equal(data, loaded, check_freq=False)
            process = run_cli("--csv", str(csv), "--output", str(output))
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(json.loads(process.stdout), report)
            self.assertEqual(set(report), {
                "source", "first_open", "last_open", "m5_bars", "data_sha256",
                "strategy", "execution", "metrics", "funnel", "conventions",
                "profit_factor_encoding",
            })
            self.assertEqual(report["metrics"], expected.metrics)
            self.assertEqual(report["data_sha256"], data_fingerprint(data))
            self.assertEqual(report["m5_bars"], len(data))
            trades = pd.read_csv(output.with_suffix(".trades.csv"))
            self.assertEqual(list(trades.columns), TRADE_COLUMNS)
            self.assertEqual(len(trades), len(expected.trades))
            self.assertEqual(
                output.with_suffix(".audit.csv").read_text(),
                expected.audit.to_csv(index=False),
            )

    def test_missing_csv_is_an_explicit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "institutional.json"
            process = run_cli(
                "--csv", str(Path(directory) / "missing.csv"), "--output", str(output),
            )
            self.assertEqual(process.returncode, 1)
            self.assertIn("Backtest failed:", process.stderr)
            self.assertEqual(process.stdout, "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
