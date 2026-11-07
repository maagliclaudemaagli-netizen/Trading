"""Final strategy: Engulfing + SMA50 on H1, stop at 1.5 x ATR, risk/reward 1:2.

Module purpose
--------------
Single source of truth for the trading rules and the backtest engine. It is split into
small, single-purpose functions:

    load_csv()            -> read and resample OHLC data
    compute_atr()         -> Wilder ATR
    detect_engulfing()    -> bullish / bearish engulfing masks
    signals()             -> adds indicators + long/short entry flags
    backtest()            -> simulates trades, one position at a time
    stats()               -> performance metrics from R multiples

Trading rules (all known at the close of bar N, no look-ahead):
    LONG  : N-1 bearish, N bullish, body N engulfs body N-1
            (Open[N] <= Close[N-1] and Close[N] >= Open[N-1]) AND Close[N] > SMA50.
    SHORT : mirror image, Close[N] < SMA50.
    Entry : Open[N+1] +/- cost.
    Stop  : entry -/+ SL_ATR x ATR14(N).
    Target: entry +/- RR x risk.
    One position at a time; stop wins if stop and target are hit in the same bar;
    a gap beyond the stop/target exits at that bar's open.

Public names (load_csv, signals, backtest, stats and the constants) are kept stable
because run.py and the tests import them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- settings
TIMEFRAME: str = "1h"
SMA_PERIOD: int = 50
ATR_PERIOD: int = 14
SL_ATR: float = 1.5
RR: float = 2.0
RISK: float = 0.01                      # fraction of equity risked per trade
CAPITAL: float = 10_000.0
COSTS: dict[str, float] = {             # cost per side, in price units
    "EURUSD": 0.00006,
    "GBPUSD": 0.00008,
    "USDJPY": 0.006,
}
ASSETS: list[str] = list(COSTS)
DATA: Path = Path(__file__).parent / "data"

OHLC_COLUMNS: list[str] = ["open", "high", "low", "close"]
LONG, SHORT, FLAT = 1, -1, 0


# ---------------------------------------------------------------------------- data
def load_csv(path: str | Path, tf: str | None = TIMEFRAME) -> pd.DataFrame:
    """Load an OHLC CSV and optionally resample it.

    Args:
        path: CSV file with at least columns timestamp, open, high, low, close
              (case-insensitive). Timestamps are parsed as UTC.
        tf: pandas resample rule (e.g. "1h"). None or "" keeps the native timeframe.

    Returns:
        DataFrame indexed by UTC timestamps with float columns open, high, low, close,
        sorted, without duplicated timestamps and without empty resampled bars.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = {"timestamp", *OHLC_COLUMNS} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if raw.empty:
        raise ValueError(f"{path.name}: file contains no rows")

    df = raw[OHLC_COLUMNS].astype(float)
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"], utc=True))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if tf:
        df = resample_ohlc(df, tf)
    return df


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample OHLC bars to a higher timeframe (left-labelled, left-closed).

    Args:
        df: OHLC DataFrame indexed by timestamp.
        tf: pandas resample rule, e.g. "1h" or "4h".

    Returns:
        Resampled DataFrame; periods with no data are dropped.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return df.resample(tf, label="left", closed="left").agg(agg).dropna()


# ---------------------------------------------------------------------------- indicators
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder Average True Range.

    Args:
        df: OHLC DataFrame.
        period: smoothing period.

    Returns:
        ATR series (NaN until `period` bars are available).
    """
    prev_close = df.close.shift(1)
    true_range = pd.concat(
        [df.high - df.low, (df.high - prev_close).abs(), (df.low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def detect_engulfing(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Detect body-engulfing candles.

    Args:
        df: OHLC DataFrame.

    Returns:
        (bullish, bearish) boolean Series. Bullish = previous bar bearish, current bar
        bullish and its body covers the previous body; bearish is the mirror image.
    """
    prev_open, prev_close = df.open.shift(1), df.close.shift(1)
    bullish = (prev_close < prev_open) & (df.close > df.open) \
        & (df.open <= prev_close) & (df.close >= prev_open)
    bearish = (prev_close > prev_open) & (df.close < df.open) \
        & (df.open >= prev_close) & (df.close <= prev_open)
    return bullish, bearish


# ---------------------------------------------------------------------------- signals
def signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators and entry flags to a copy of the OHLC data.

    Args:
        df: OHLC DataFrame (see load_csv).

    Returns:
        Copy of `df` with extra columns: atr, sma, long (bool), short (bool).
    """
    out = df.copy()
    out["atr"] = compute_atr(out)
    out["sma"] = out.close.rolling(SMA_PERIOD).mean()
    bullish, bearish = detect_engulfing(out)
    ready = out.atr.notna() & out.sma.notna()
    out["long"] = bullish & (out.close > out.sma) & ready
    out["short"] = bearish & (out.close < out.sma) & ready
    return out


# ---------------------------------------------------------------------------- engine
def _first_hit(mask: np.ndarray, default: int) -> int:
    """Index of the first True in `mask`, or `default` if none."""
    return int(np.argmax(mask)) if mask.any() else default


def _exit_price(bar_open: float, level: float, side: int, is_stop: bool) -> float:
    """Fill price when a stop or target is touched, honouring opening gaps.

    Args:
        bar_open: open of the exit bar.
        level: stop or target price.
        side: LONG (1) or SHORT (-1).
        is_stop: True for a stop, False for a target.

    Returns:
        The open if the bar gapped beyond the level, otherwise the level itself.
    """
    gapped = side * (bar_open - level) <= 0 if is_stop else side * (bar_open - level) >= 0
    return bar_open if gapped else level


def backtest(d: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Simulate trades on a DataFrame produced by `signals`.

    Args:
        d: DataFrame with columns open, high, low, atr, long, short.
        cost: cost per side in price units (added at entry and exit).

    Returns:
        One row per closed trade with columns side, entry_time, entry, sl, tp,
        exit_time, exit, reason, bars, pnl_r. Empty DataFrame if no trade.
        A trade still open at the end of the data is not reported.

    Raises:
        ValueError: if required columns are missing or cost is negative.
    """
    required = {"open", "high", "low", "atr", "long", "short"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"backtest: missing columns {sorted(missing)} (call signals() first)")
    if cost < 0:
        raise ValueError("backtest: cost must be >= 0")

    opens, highs, lows, atrs = (d[c].to_numpy() for c in ("open", "high", "low", "atr"))
    direction = np.where(d.long, LONG, np.where(d.short, SHORT, FLAT))
    times, n_bars = d.index, len(d)
    trades: list[dict[str, Any]] = []
    next_free_bar = 0

    for signal_bar in np.flatnonzero(direction):
        if signal_bar < next_free_bar or signal_bar + 1 >= n_bars:
            continue                                     # position open / no next bar
        side, entry_bar = int(direction[signal_bar]), signal_bar + 1
        entry = opens[entry_bar] + side * cost
        risk = SL_ATR * atrs[signal_bar]
        stop, target = entry - side * risk, entry + side * RR * risk

        future_high, future_low = highs[entry_bar:], lows[entry_bar:]
        stop_hits = future_low <= stop if side == LONG else future_high >= stop
        target_hits = future_high >= target if side == LONG else future_low <= target
        k_stop, k_target = _first_hit(stop_hits, n_bars), _first_hit(target_hits, n_bars)
        if k_stop == n_bars and k_target == n_bars:
            break                                        # still open at end of data

        is_stop = k_stop <= k_target                     # pessimistic: stop wins ties
        k = k_stop if is_stop else k_target
        level = stop if is_stop else target
        exit_price = _exit_price(opens[entry_bar + k], level, side, is_stop) - side * cost

        trades.append({
            "side": "long" if side == LONG else "short",
            "entry_time": times[entry_bar], "entry": entry, "sl": stop, "tp": target,
            "exit_time": times[entry_bar + k], "exit": exit_price,
            "reason": "stop_loss" if is_stop else "take_profit",
            "bars": k + 1, "pnl_r": side * (exit_price - entry) / risk,
        })
        next_free_bar = entry_bar + k + 1
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------- metrics
def stats(r: np.ndarray, weeks: float) -> dict[str, float]:
    """Performance metrics from a sequence of trade results in R.

    Args:
        r: array of R multiples, in chronological order.
        weeks: length of the tested period in weeks (must be > 0).

    Returns:
        {"trades": 0} if `r` is empty, otherwise keys trades, per_week, win (%),
        pf (inf if no losing trade), exp (mean R), maxdd (%, <= 0) and final
        (equity after compounding RISK per trade from CAPITAL).

    Raises:
        ValueError: if weeks <= 0.
    """
    if weeks <= 0:
        raise ValueError("stats: weeks must be > 0")
    if len(r) == 0:
        return {"trades": 0}
    equity = np.concatenate([[CAPITAL], CAPITAL * np.cumprod(1 + RISK * r)])
    peak = np.maximum.accumulate(equity)
    gross_loss = -r[r < 0].sum()
    return {
        "trades": len(r),
        "per_week": len(r) / weeks,
        "win": 100 * (r > 0).mean(),
        "pf": r[r > 0].sum() / gross_loss if gross_loss else np.inf,
        "exp": r.mean(),
        "maxdd": 100 * ((equity - peak) / peak).min(),
        "final": equity[-1],
    }
