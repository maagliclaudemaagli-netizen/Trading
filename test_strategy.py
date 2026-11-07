"""Unit tests for strategy.py.

Run with: .venv/bin/python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

import strategy as S


# ---------------------------------------------------------------------------- helpers
def make_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Build an hourly OHLC frame from [open, high, low, close] rows."""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=S.OHLC_COLUMNS, index=idx)


def make_long_setup(bars_after: list[list[float]], atr: float = 0.01) -> pd.DataFrame:
    """Three flat bars with a LONG signal on the third, followed by `bars_after`.

    Entry is at 1.0, risk = 1.5 * atr = 0.015, target = 1.03 at R:R 1:2.
    """
    d = make_frame([[1.0, 1.001, 0.999, 1.0]] * 3 + bars_after)
    d["atr"], d["long"], d["short"] = atr, False, False
    d.loc[d.index[2], "long"] = True
    return d


# ---------------------------------------------------------------------------- patterns
def test_bullish_engulfing_detected():
    d = make_frame([[1.10, 1.11, 1.09, 1.095], [1.094, 1.12, 1.09, 1.105]])
    bullish, bearish = S.detect_engulfing(d)
    assert bool(bullish.iloc[1]) and not bool(bearish.iloc[1])


def test_bearish_engulfing_detected():
    d = make_frame([[1.095, 1.11, 1.09, 1.10], [1.105, 1.11, 1.08, 1.09]])
    bullish, bearish = S.detect_engulfing(d)
    assert bool(bearish.iloc[1]) and not bool(bullish.iloc[1])


def test_signals_need_indicator_warmup():
    d = make_frame([[1.10, 1.11, 1.09, 1.095], [1.094, 1.12, 1.09, 1.105]])
    s = S.signals(d)
    assert not s.long.any() and not s.short.any()      # SMA50 / ATR not ready yet


# ---------------------------------------------------------------------------- engine
def test_take_profit_is_rr():
    tr = S.backtest(make_long_setup([[1.0, 1.04, 0.995, 1.03]]), 0.0)
    assert tr.reason.iloc[0] == "take_profit"
    assert np.isclose(tr.pnl_r.iloc[0], S.RR)


def test_stop_wins_when_both_hit_same_bar():
    tr = S.backtest(make_long_setup([[1.0, 1.05, 0.95, 1.0]]), 0.0)
    assert tr.reason.iloc[0] == "stop_loss"
    assert np.isclose(tr.pnl_r.iloc[0], -1.0)


def test_gap_through_stop_exits_at_open():
    tr = S.backtest(make_long_setup([[1.0, 1.001, 0.999, 1.0], [0.97, 0.975, 0.96, 0.97]]), 0.0)
    assert tr.reason.iloc[0] == "stop_loss"
    assert np.isclose(tr.exit.iloc[0], 0.97)            # worse than the 0.985 stop
    assert tr.pnl_r.iloc[0] < -1.0


def test_costs_reduce_result():
    bars = [[1.0, 1.04, 0.995, 1.03]]
    assert S.backtest(make_long_setup(bars), 0.0001).pnl_r.iloc[0] < S.RR


def test_open_trade_at_end_is_not_reported():
    assert S.backtest(make_long_setup([[1.0, 1.001, 0.999, 1.0]]), 0.0).empty


def test_backtest_rejects_bad_input():
    with pytest.raises(ValueError):
        S.backtest(make_frame([[1, 1, 1, 1]]), 0.0)    # signals() not applied
    with pytest.raises(ValueError):
        S.backtest(make_long_setup([]), -1.0)


# ---------------------------------------------------------------------------- metrics
def test_stats_values():
    s = S.stats(np.array([2.0, -1.0, -1.0]), weeks=1)
    assert s["trades"] == 3 and np.isclose(s["pf"], 1.0) and np.isclose(s["win"], 100 / 3)
    assert s["maxdd"] < 0


def test_stats_edge_cases():
    assert S.stats(np.array([]), 1) == {"trades": 0}
    assert S.stats(np.array([1.0]), 1)["pf"] == np.inf
    with pytest.raises(ValueError):
        S.stats(np.array([1.0]), 0)


# ---------------------------------------------------------------------------- data
def test_load_csv_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_csv(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,open,high\n2026-01-01,1,1\n")
    with pytest.raises(ValueError):
        S.load_csv(bad)


def test_load_csv_resamples(tmp_path):
    f = tmp_path / "m15.csv"
    f.write_text("Timestamp,Open,High,Low,Close\n"
                 "2026-01-01 00:00,1,2,0.5,1.5\n2026-01-01 00:15,1.5,3,1,2\n")
    d = S.load_csv(f, "1h")
    assert len(d) == 1 and d.high.iloc[0] == 3 and d.low.iloc[0] == 0.5 and d.close.iloc[0] == 2
