from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from .budget import ArchiveBudget, BudgetReached
from .config import Settings
from .parser import decode

LOG = logging.getLogger(__name__)
BASE = "https://datafeed.dukascopy.com/datafeed"
RATE_LIMIT_DELAYS = (5.0, 15.0, 30.0)


class AccessDenied(RuntimeError):
    """Provider restrictions are never bypassed."""


class Unavailable(RuntimeError):
    """Archive not available; not evidence of market closure."""


class PacketDownloadError(RuntimeError):
    """An exhausted packet can be recorded and skipped without hiding failure."""


def retry_after(response: requests.Response) -> float:
    value = response.headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            at = parsedate_to_datetime(value)
            if at.tzinfo is None:
                raise ValueError("Retry-After sans fuseau.")
            seconds = (at - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            LOG.warning("Retry-After invalide: %r; delai de reessai configure utilise.", value)
            return 0.0
    if not math.isfinite(seconds):
        LOG.warning("Retry-After non fini: %r; delai de reessai configure utilise.", value)
        return 0.0
    return max(0.0, seconds)


class DukascopyClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._blocked = threading.Event()

    @staticmethod
    def url(pair: str, at: pd.Timestamp, kind: str, side: str | None = None) -> str:
        prefix = f"{BASE}/{pair}/{at.year}/{at.month - 1:02}/{at.day:02}"
        if kind == "tick":
            return f"{prefix}/{at.hour:02}h_ticks.bi5"
        if kind == "M1" and side in ("bid", "ask"):
            return f"{prefix}/{side.upper()}_candles_min_1.bi5"
        raise ValueError("Archive attendue: tick ou M1 bid/ask.")

    def fetch(self, url: str, kind: str) -> bytes:
        # One shared gate also prevents other workers from ignoring a 429 cooldown.
        with self._lock:
            if self._blocked.is_set():
                raise AccessDenied("Fournisseur bloque apres un refus d'acces; aucune nouvelle requete.")
            for attempt in range(self.settings.retries + 1):
                time.sleep(self.settings.delay)
                try:
                    response = requests.get(url, timeout=self.settings.timeout)
                    if response.status_code in (401, 403) or (
                        response.status_code == 429 and "bot blocked" in response.text.casefold()
                    ):
                        self._blocked.set()
                        raise AccessDenied(f"Acces refuse HTTP {response.status_code}: {url}")
                    if response.status_code in (404, 204):
                        raise Unavailable(f"Archive absente HTTP {response.status_code}: {url}")
                    response.raise_for_status()
                    if not response.content:
                        raise Unavailable(f"Archive source vide: {url}")
                    decode(response.content, kind)
                    return response.content
                except (requests.RequestException, ValueError) as error:
                    LOG.warning("Tentative %s/%s %s: %s", attempt + 1, self.settings.retries + 1, url, error)
                    rate_limited = isinstance(error, requests.HTTPError) and (
                        error.response is not None and error.response.status_code == 429
                    )
                    non_retryable = isinstance(error, requests.HTTPError) and (
                        error.response is not None and error.response.status_code < 500 and not rate_limited
                    )
                    if attempt == self.settings.retries or non_retryable:
                        raise PacketDownloadError(
                            f"Paquet abandonne apres {attempt + 1} requete(s): {url}: {error}"
                        ) from error
                    delay = RATE_LIMIT_DELAYS[attempt] if rate_limited else min(2 ** attempt, 30)
                    if rate_limited and isinstance(error, requests.HTTPError) and error.response is not None:
                        delay = max(delay, retry_after(error.response))
                    LOG.warning("Reessai dans %.1f secondes: %s", delay, url)
                    time.sleep(delay)
        raise RuntimeError("Boucle de telechargement incoherente.")

    def archive(
        self, pair: str, at: pd.Timestamp, kind: str, side: str | None = None,
        *, budget: ArchiveBudget | None = None,
    ) -> tuple[bytes, str, Path]:
        url = self.url(pair, at, kind, side)
        relative = url.removeprefix(BASE + "/")
        path = self.settings.root / "raw" / relative
        if path.exists():
            content = path.read_bytes()
            decode(content, kind)
            LOG.info("Archive conservee reutilisee: %s", path)
            return content, url, path
        if budget is not None and budget.used_bytes >= budget.max_bytes:
            raise BudgetReached("Le budget autorise est atteint.")
        content = self.fetch(url, kind)
        if budget is not None:
            budget.save(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".bi5.part")
            temporary.write_bytes(content)
            temporary.replace(path)
        LOG.info("Archive verifiee: %s (%s octets, SHA256=%s)", url, len(content), hashlib.sha256(content).hexdigest())
        return content, url, path
