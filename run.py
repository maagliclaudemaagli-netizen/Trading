"""Command-line entry point: backtest the final strategy on every configured asset.

Module purpose
--------------
For each asset in strategy.ASSETS: locate its data file, run signals + backtest,
print one summary line, then print a global summary and export every trade to
data/trades.csv.

Usage:
    .venv/bin/python run.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy import ASSETS, COSTS, DATA, RR, SL_ATR, TIMEFRAME, backtest, load_csv, signals, stats

TRADES_FILE: Path = DATA / "trades.csv"
DATA_PATTERNS: tuple[str, ...] = ("{asset}_H1.csv",)   # Dukascopy (fetch_dukascopy_h1.py)


def find_data_file(asset: str) -> Path | None:
    """Return the preferred data file for `asset` (long H1 history first), or None."""
    for pattern in DATA_PATTERNS:
        path = DATA / pattern.format(asset=asset)
        if path.exists():
            return path
    return None


def period_in_weeks(df: pd.DataFrame) -> float:
    """Length of the data in weeks (tiny positive value for a single-day sample)."""
    return max((df.index[-1] - df.index[0]).days / 7, 1e-9)


def run_asset(asset: str, path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest one asset.

    Args:
        asset: symbol, must be a key of COSTS.
        path: OHLC CSV file.

    Returns:
        (trades with an `asset` column first, metrics dict from strategy.stats).
    """
    df = load_csv(path)
    trades = backtest(signals(df), COSTS[asset])
    results = trades.pnl_r.to_numpy() if len(trades) else np.array([])
    metrics = stats(results, period_in_weeks(df))
    trades.insert(0, "asset", asset)
    return trades, metrics


def format_asset_line(asset: str, m: dict[str, float]) -> str:
    """One-line human summary of an asset's metrics."""
    return (f"{asset}: {m['trades']:3d} trades ({m['per_week']:.1f}/sem) | win {m['win']:.1f}% | "
            f"PF {m['pf']:.2f} | E[R] {m['exp']:+.3f} | DD {m['maxdd']:.1f}% | "
            f"10 000$ -> {m['final']:,.0f}$")


def format_global_line(trades: pd.DataFrame) -> str:
    """One-line summary over all assets' trades (assumes at least one losing trade)."""
    r = trades.pnl_r.to_numpy()
    gross_loss = -r[r < 0].sum()
    pf = r[r > 0].sum() / gross_loss if gross_loss else float("inf")
    return (f"\nGLOBAL : {len(r)} trades | win {100 * (r > 0).mean():.1f}% | "
            f"PF {pf:.2f} | total {r.sum():+.1f} R")


def main() -> None:
    """Run the backtest on all assets, print summaries and export trades."""
    print(f"Englobante + SMA50 | {TIMEFRAME} | SL {SL_ATR} x ATR | R:R 1:{RR:g}\n")
    all_trades: list[pd.DataFrame] = []
    for asset in ASSETS:
        path = find_data_file(asset)
        if path is None:
            print(f"{asset}: pas de données ({asset}_H1.csv) -> lancer fetch_dukascopy_h1.py")
            continue
        trades, metrics = run_asset(asset, path)
        if metrics["trades"]:
            print(format_asset_line(asset, metrics))
        all_trades.append(trades)

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if combined.empty:
        print("Aucun trade.")
        return
    print(format_global_line(combined))
    combined.to_csv(TRADES_FILE, index=False)
    print(f"Export : {TRADES_FILE}")


if __name__ == "__main__":
    main()
