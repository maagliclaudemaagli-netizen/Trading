"""Offline unit tests for the data downloaders (no network access).

Run with: .venv/bin/python -m pytest -q
"""
import lzma
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import fetch_dukascopy_h1 as D


# ---------------------------------------------------------------------------- Dukascopy
def make_bi5(rows: list[tuple[int, int, int, int, int, float]]) -> bytes:
    """Build a compressed .bi5 payload from (t, o, c, l, h, v) rows."""
    return lzma.compress(np.array(rows, dtype=D.RECORD_DTYPE).tobytes())


def test_build_url_uses_zero_based_month():
    assert D.build_url("EURUSD", 2020, 1).endswith("/EURUSD/2020/00/BID_candles_hour_1.bi5")
    assert "/2020/11/" in D.build_url("EURUSD", 2020, 12)


def test_decode_month_prices_and_timestamps():
    raw = make_bi5([(0, 112076, 112100, 112000, 112200, 5.0),
                    (3600, 112100, 112150, 112050, 112300, 7.0)])
    df = D.decode_month(raw, "EURUSD", 2020, 1)
    assert len(df) == 2
    assert df.timestamp.iloc[1] == pd.Timestamp("2020-01-01 01:00", tz="UTC")
    assert np.isclose(df.open.iloc[0], 1.12076) and np.isclose(df.high.iloc[0], 1.122)
    assert np.isclose(df.close.iloc[0], 1.121) and np.isclose(df.low.iloc[0], 1.12)


def test_decode_month_jpy_divisor():
    df = D.decode_month(make_bi5([(0, 108500, 108600, 108400, 108700, 1.0)]), "USDJPY", 2020, 1)
    assert np.isclose(df.open.iloc[0], 108.5)


def test_decode_month_edge_cases():
    assert D.decode_month(b"", "EURUSD", 2020, 1).empty
    with pytest.raises(ValueError):
        D.decode_month(b"not lzma", "EURUSD", 2020, 1)
    with pytest.raises(ValueError):
        D.decode_month(lzma.compress(b"x" * 25), "EURUSD", 2020, 1)
    with pytest.raises(KeyError):
        D.decode_month(make_bi5([(0, 1, 1, 1, 1, 0.0)]), "UNKNOWN", 2020, 1)


def test_clean_history_drops_flat_and_duplicates():
    a = D.decode_month(make_bi5([(0, 100, 100, 100, 100, 0.0),          # flat (closed)
                                 (3600, 100, 101, 99, 102, 1.0)]), "EURUSD", 2020, 1)
    df = D.clean_history([a, a.copy(), pd.DataFrame()])
    assert len(df) == 1 and df.index.is_monotonic_increasing
    with pytest.raises(ValueError):
        D.clean_history([pd.DataFrame()])


def test_months_to_fetch_excludes_current_month():
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    months = D.months_to_fetch(1, now)
    assert months[0] == (2025, 1) and months[-1] == (2026, 2) and len(months) == 14
    with pytest.raises(ValueError):
        D.months_to_fetch(-1, now)
