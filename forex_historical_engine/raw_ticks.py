from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

import pandas as pd

from .budget import ArchiveBudget, BudgetReached
from .config import date_range, instrument_code
from .downloader import DukascopyClient, Unavailable

LOG = logging.getLogger(__name__)


def download_raw_ticks(
    client: DukascopyClient,
    budget: ArchiveBudget,
    pairs: Sequence[str],
    start: str,
    end: str,
) -> dict:
    first, last = date_range(start, end)
    instruments = list(dict.fromkeys(instrument_code(pair) for pair in pairs))
    if not instruments:
        raise ValueError("Choisir au moins un instrument.")
    if client.settings.root != budget.root:
        raise ValueError("Le client et le budget doivent utiliser le meme dossier.")
    downloaded: Counter[str] = Counter()
    reused: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()

    def summary(scanned: bool, next_url: str | None = None) -> dict:
        return {
            "source": "Dukascopy",
            "start_utc": first.isoformat(),
            "end_utc_inclusive": last.isoformat(),
            "max_bytes": budget.max_bytes,
            "stored_bytes": budget.used_bytes,
            "scanned_all_hours": scanned,
            "complete_archive_request": scanned and not sum(unavailable.values()),
            "stop_reason": None if scanned else "storage_limit",
            "next_archive": next_url,
            "instruments": [
                {
                    "instrument": pair,
                    "downloaded_archives": downloaded[pair],
                    "reused_archives": reused[pair],
                    "unavailable_archives": unavailable[pair],
                }
                for pair in instruments
            ],
        }

    with budget.exclusive():
        hours = pd.date_range(first, last + pd.Timedelta(days=1), freq="h", inclusive="left")
        for at in hours:
            for pair in instruments:
                previous_size = budget.used_bytes
                try:
                    client.archive(pair, at, "tick", budget=budget)
                except Unavailable as error:
                    unavailable[pair] += 1
                    LOG.warning("%s", error)
                    continue
                except BudgetReached:
                    return summary(False, client.url(pair, at, "tick"))
                if budget.used_bytes == previous_size:
                    reused[pair] += 1
                else:
                    downloaded[pair] += 1
        return summary(True)
