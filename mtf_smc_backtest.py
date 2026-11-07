"""Backtest MTF Smart Money 4H -> 15M (OB / FVG / Liquidity Sweep + Engulfing).

Logique
-------
4H (HTF, uniquement bougies CLÔTURÉES) :
  - Biais : dernière englobante 4H (haussière -> long, baissière -> short).
  - OB haussier  : dernière bougie baissière dont le plus haut est cassé en clôture par la bougie suivante.
    OB baissier  : symétrique. Zone = [low, high] de la bougie OB.
  - FVG haussier : low[i] > high[i-2] -> zone [high[i-2], low[i]]. FVG baissier symétrique.
  - Une zone devient active à la clôture de la bougie qui la confirme, est invalidée
    par une clôture 4H au-delà de son bord opposé, expire après ZONE_TTL bougies 4H,
    et n'est tradée qu'une fois.
15M (LTF) :
  - Englobante dans le sens du biais dont la mèche touche une zone 4H active.
  - Option sweep : la mèche de l'englobante (ou de la bougie précédente) prend le
    plus bas/haut des SWEEP_LOOKBACK bougies précédentes.
Exécution :
  - Entrée à l'open de la bougie 15M suivante ; SL = extrême de l'englobante ± 0.1 ATR(14) 15M.
  - TP = RR x risque. SL prioritaire si SL et TP touchés dans la même bougie.
  - Coût 0.6 pip par trade (spread + slippage). Une position à la fois.
Look-ahead : une bougie 4H ouverte à T n'est utilisable qu'à partir de T+4h
(=> ouverture de la 15M de T+4h), jamais avant.

Usage:  .venv/bin/python mtf_smc_backtest.py [--pair EURUSD] [--risk 1.0]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
PIP = 0.0001
COST = 0.6 * PIP
ATR_LEN = 14
ATR_BUFFER = 0.1
ZONE_TTL = 30          # bougies 4H (~5 jours)
SWEEP_LOOKBACK = 12    # bougies 15M (3h)
H4 = pd.Timedelta(hours=4)


# ---------------------------------------------------------------- données
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()[["open", "high", "low", "close"]].astype(float)


def engulfing(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    o, c = df.open.values, df.close.values
    po, pc = np.roll(o, 1), np.roll(c, 1)
    bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)
    bull[0] = bear[0] = False
    return bull, bear


def atr(df: pd.DataFrame, n: int = ATR_LEN) -> np.ndarray:
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().values


# ---------------------------------------------------------------- zones 4H
@dataclass
class Zone:
    kind: str        # "OB" | "FVG"
    side: int        # +1 demande (long) / -1 offre (short)
    bottom: float
    top: float
    born: int        # index 4H de confirmation
    used: bool = False


def build_h4_events(h4: pd.DataFrame) -> list[tuple[pd.Timestamp, int, list[Zone]]]:
    """Pour chaque bougie 4H : (heure de clôture, biais après clôture, zones créées)."""
    o, h, l, c = (h4[k].values for k in ("open", "high", "low", "close"))
    bull, bear = engulfing(h4)
    events, bias = [], 0
    for i in range(len(h4)):
        if bull[i]:
            bias = 1
        elif bear[i]:
            bias = -1
        new: list[Zone] = []
        if i >= 1:
            if c[i - 1] < o[i - 1] and c[i] > h[i - 1]:
                new.append(Zone("OB", 1, l[i - 1], h[i - 1], i))
            if c[i - 1] > o[i - 1] and c[i] < l[i - 1]:
                new.append(Zone("OB", -1, l[i - 1], h[i - 1], i))
        if i >= 2:
            if l[i] > h[i - 2]:
                new.append(Zone("FVG", 1, h[i - 2], l[i], i))
            if h[i] < l[i - 2]:
                new.append(Zone("FVG", -1, h[i], l[i - 2], i))
        events.append((h4.index[i] + H4, bias, new, c[i]))
    return events


# ---------------------------------------------------------------- signaux 15M
def signals(h4: pd.DataFrame, m15: pd.DataFrame, kinds: set[str], need_sweep: bool) -> list[tuple[int, int]]:
    """Liste (index de la bougie englobante 15M, direction)."""
    events = build_h4_events(h4)
    bull, bear = engulfing(m15)
    hi, lo = m15.high.values, m15.low.values
    roll_lo = pd.Series(lo).shift(2).rolling(SWEEP_LOOKBACK).min().values
    roll_hi = pd.Series(hi).shift(2).rolling(SWEEP_LOOKBACK).max().values
    t15 = m15.index
    out, zones, bias, k, h4_i = [], [], 0, 0, -1
    for j in range(len(m15)):
        # intègre toutes les bougies 4H clôturées au plus tard à l'ouverture de cette 15M
        while k < len(events) and events[k][0] <= t15[j]:
            _, bias, new, close4 = events[k]
            h4_i = k
            zones = [z for z in zones
                     if not z.used and h4_i - z.born <= ZONE_TTL
                     and not (z.side == 1 and close4 < z.bottom)
                     and not (z.side == -1 and close4 > z.top)]
            zones += [z for z in new if z.kind in kinds]
            k += 1
        if bias == 0 or not (bull[j] or bear[j]):
            continue
        d = 1 if bull[j] else -1
        if d != bias:
            continue
        if need_sweep:
            if d == 1 and not (min(lo[j], lo[j - 1]) < roll_lo[j]):
                continue
            if d == -1 and not (max(hi[j], hi[j - 1]) > roll_hi[j]):
                continue
        for z in zones:
            if z.side == d and lo[j] <= z.top and hi[j] >= z.bottom:
                z.used = True
                out.append((j, d))
                break
    return out


# ---------------------------------------------------------------- simulation
def simulate(m15: pd.DataFrame, atr15: np.ndarray, sigs: list[tuple[int, int]], rr: float) -> pd.DataFrame:
    o, h, l = m15.open.values, m15.high.values, m15.low.values
    n, busy_until, rows = len(m15), -1, []
    for j, d in sigs:
        e = j + 1
        if e >= n or j < busy_until or np.isnan(atr15[j]):
            continue
        entry = o[e]
        sl = l[j] - ATR_BUFFER * atr15[j] if d == 1 else h[j] + ATR_BUFFER * atr15[j]
        risk = (entry - sl) * d
        if risk <= COST:
            continue
        tp = entry + d * rr * risk
        hs, ls = h[e:], l[e:]
        hit_sl = np.flatnonzero(ls <= sl) if d == 1 else np.flatnonzero(hs >= sl)
        hit_tp = np.flatnonzero(hs >= tp) if d == 1 else np.flatnonzero(ls <= tp)
        i_sl = hit_sl[0] if hit_sl.size else np.inf
        i_tp = hit_tp[0] if hit_tp.size else np.inf
        if i_sl == np.inf and i_tp == np.inf:
            break  # trade non clôturé en fin de données
        if i_sl <= i_tp:
            exit_px, x = sl, int(i_sl)
        else:
            exit_px, x = tp, int(i_tp)
        r = ((exit_px - entry) * d - COST) / risk
        rows.append((m15.index[e], d, entry, sl, tp, m15.index[e + x], r))
        busy_until = e + x
    return pd.DataFrame(rows, columns=["entry_time", "dir", "entry", "sl", "tp", "exit_time", "R"])


def metrics(tr: pd.DataFrame, risk_pct: float) -> dict:
    if tr.empty:
        return {"Trades": 0}
    r = tr.R.values
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumprod(1 + r * risk_pct / 100)
    dd = (1 - eq / np.maximum.accumulate(eq)).max() * 100
    return {
        "Trades": len(r),
        "Win Rate %": round((r > 0).mean() * 100, 1),
        "Profit Factor": round(gains / losses, 2) if losses else np.inf,
        "Expectancy R": round(r.mean(), 3),
        "Total R": round(r.sum(), 1),
        "Max DD %": round(dd, 1),
        "Return %": round((eq[-1] - 1) * 100, 1),
    }


COMBOS = {
    "1. OB 4H": ({"OB"}, False),
    "2. FVG 4H": ({"FVG"}, False),
    "3. OB/FVG + Sweep 15M": ({"OB", "FVG"}, True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--risk", type=float, default=1.0, help="%% du capital risqué par trade (pour le DD)")
    args = ap.parse_args()

    h4 = load(DATA / f"{args.pair}_4H.csv")
    m15 = load(DATA / f"{args.pair}_15M.csv")
    atr15 = atr(m15)
    print(f"{args.pair}  15M: {m15.index[0]:%Y-%m-%d} -> {m15.index[-1]:%Y-%m-%d} "
          f"({len(m15)} bougies, 4H: {len(h4)})  coût {COST / PIP:.1f} pip, risque {args.risk}%/trade\n")

    rows, all_trades = [], []
    for name, (kinds, sweep) in COMBOS.items():
        sigs = signals(h4, m15, kinds, sweep)
        for rr in (2, 3, 4):
            tr = simulate(m15, atr15, sigs, rr)
            rows.append({"Combinaison": name, "R:R": f"1:{rr}", **metrics(tr, args.risk)})
            all_trades.append(tr.assign(combo=name, rr=rr))

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    pd.concat(all_trades).to_csv(DATA / f"mtf_smc_trades_{args.pair}.csv", index=False)


if __name__ == "__main__":
    main()
