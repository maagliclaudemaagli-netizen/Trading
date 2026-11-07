"""Télécharge l'historique M5 OANDA (Hugging Face) et produit EURUSD_15M.csv + EURUSD_4H.csv.

Le dataset n'a pas de dossier 15M : on télécharge le M5 et on rééchantillonne
(15M et 4H sont construits depuis la même source -> alignement strict, UTC, label = ouverture).

Usage:
    .venv/bin/python fetch_oanda_mtf.py [EURUSD] [--years 6]
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from fetch_oanda_h1 import CACHE_DIR, FILE_URL, REPO, DownloadError, curl, to_ohlcv
from strategy import DATA

import json

TREE_M5 = "https://huggingface.co/api/datasets/{repo}/tree/main/5"
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def list_m5(pair: str, since: int) -> list[str]:
    entries = json.loads(curl(TREE_M5.format(repo=REPO.format(pair=pair))))
    paths = sorted(e["path"] for e in entries if e.get("path", "").endswith(".parquet"))
    return [p for p in paths if int(p.split("_5_")[1][:4]) >= since]


def download_m5(pair: str, path: str):
    target = CACHE_DIR / f"{pair}_M5" / path.split("/")[-1]
    if not target.exists() or target.stat().st_size == 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        curl(FILE_URL.format(repo=REPO.format(pair=pair), path=path), tmp)
        tmp.rename(target)
    return target


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left").agg(AGG)
    return out.dropna(subset=["open"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=["EURUSD"])
    ap.add_argument("--years", type=int, default=6)
    args = ap.parse_args()
    since = pd.Timestamp.now(tz="UTC").year - args.years
    for pair in args.pairs:
        try:
            paths = list_m5(pair, since)
            frames = []
            for i, p in enumerate(paths, 1):
                frames.append(pd.read_parquet(download_m5(pair, p)))
                if i % 12 == 0 or i == len(paths):
                    print(f"  {pair}: {i}/{len(paths)} mois", flush=True)
            m5 = to_ohlcv(pd.concat(frames, ignore_index=True))
        except (DownloadError, ValueError) as exc:
            print(f"{pair}: ÉCHEC — {exc}", file=sys.stderr)
            return 1
        m15, h4 = resample(m5, "15min"), resample(m5, "4h")
        m15.to_csv(DATA / f"{pair}_15M.csv")
        h4.to_csv(DATA / f"{pair}_4H.csv")
        print(f"{pair}: 15M={len(m15)} 4H={len(h4)} {m15.index[0]:%Y-%m-%d} -> {m15.index[-1]:%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
