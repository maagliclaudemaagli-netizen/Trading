from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd
import requests

from .analytics import report
from .config import RULES, Settings, date_range, instrument
from .downloader import BASE, DukascopyClient, PacketDownloadError, Unavailable
from .parser import parse
from .processing import build_candles, clean, quality
from .storage import Storage, file_fingerprint, fingerprint

LOG = logging.getLogger(__name__)


class Provider(Protocol):
    def url(self, pair: str, at: pd.Timestamp, kind: str, side: str | None = None) -> str: ...

    def archive(self, pair: str, at: pd.Timestamp, kind: str, side: str | None = None) -> tuple[bytes, str, Path]: ...


class Engine:
    def __init__(self, settings: Settings, provider: Provider | None = None):
        self.settings = settings
        self.storage = Storage(settings.root)
        self.provider = provider if provider is not None else DukascopyClient(settings)

    def _archive(self, pair: str, scale: int, at: pd.Timestamp, kind: str, side: str | None) -> dict:
        url = self.provider.url(pair, at, kind, side)
        content = None
        try:
            previous = self.storage.download(url)
            name = at.strftime("%Y-%m-%d-%H") if kind == "tick" else f"{at.strftime('%Y-%m-%d')}-{side}"
            cache_path = self.settings.cache_root / pair / kind / str(at.year) / f"{at.month:02}" / f"{name}.parquet"
            if cache_path.exists():
                frame = self.storage.read_partition(cache_path)
                required = {"instrument", "source_url", "origin_utc", "price_scale", "archive_sha256"}
                if frame.empty or not required.issubset(frame.columns) or (
                    (frame.instrument != pair).any() or (frame.source_url != url).any()
                    or (frame.origin_utc != at).any() or (frame.price_scale != scale).any()
                    or frame.archive_sha256.nunique() != 1
                ):
                    raise ValueError(f"Cache incompatible avec le paquet ou son echelle de prix: {cache_path}")
                digest = str(frame.archive_sha256.iloc[0])
                path = Path(previous["raw_path"]) if previous and previous["raw_path"] else (
                    self.settings.root / "raw" / url.removeprefix(BASE + "/")
                )
                if path.exists() and file_fingerprint(path) != digest:
                    raise ValueError(f"Archive brute modifiee depuis sa mise en cache: {path}")
                LOG.info("Paquet Parquet reutilise sans reseau: %s", cache_path)
            else:
                content, url, path = self.provider.archive(pair, at, kind, side)
                digest = fingerprint(content)
                if previous and previous["sha256"] and previous["sha256"] != digest:
                    raise ValueError(f"Archive brute modifiee depuis sa verification: {path}")
                frame = parse(content, kind, at, pair, scale, url, side)
                frame["archive_sha256"] = digest
                # Cache the packet before merging monthly outputs, so an interrupted merge is resumable.
                self.storage.write_partition(frame, cache_path)
                LOG.info("Paquet sauvegarde dans le cache: %s", cache_path)
            if previous and previous["sha256"] and previous["sha256"] != digest:
                raise ValueError(f"Archive brute modifiee depuis sa verification: {path}")
            if previous and previous["status"] == "verified" and self.storage.cached_archive(
                pair, at, kind, url, previous["records"],
            ):
                LOG.info("Archive deja traitee et integrite verifiee: %s", url)
                return {"url": url, "status": "verified", "records": previous["records"], "cached": True}
            stats = quality(frame, kind)
            if stats["bad_rows"] or any(stats["flags"].values()):
                LOG.warning("Controle %s: %s", url, stats)
            paths = self.storage.write(
                frame, pair, kind, at.strftime("%Y-%m-%d-%H") if kind == "tick" else None,
            )
            self.storage.record(url, pair, at, "verified", path, content, len(frame), sha256=digest)
            return {"url": url, "status": "verified", "records": len(frame), "files": [str(p) for p in paths], "cache": str(cache_path)}
        except Unavailable as error:
            LOG.warning("%s", error)
            self.storage.record(url, pair, at, "unavailable", error=str(error))
            return {"url": url, "status": "unavailable", "error": str(error)}
        except PacketDownloadError as error:
            LOG.warning("Paquet ignore, poursuite du telechargement: %s", error)
            self.storage.record(url, pair, at, "failed", error=str(error))
            return {"url": url, "status": "failed", "error": str(error)}
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            LOG.exception("Echec de traitement %s", url)
            self.storage.record(url, pair, at, "failed", error=str(error))
            raise

    def download(
        self, pairs: Sequence[str], timeframe: str, start: str, end: str,
        price_scale: int | None = None, progress: Callable[[dict], None] | None = None,
    ) -> dict:
        if timeframe not in ("tick", *RULES):
            raise ValueError(f"Timeframe inconnu: {timeframe}")
        first, last = date_range(start, end)
        instruments = list(dict.fromkeys(instrument(pair, price_scale) for pair in pairs))
        if not instruments:
            raise ValueError("Choisir au moins un instrument.")
        kind = "tick" if timeframe == "tick" else "M1"

        def download_pair(item: tuple[str, int]) -> dict:
            pair, scale = item
            verified = unavailable = failed = records = 0
            # A single instrument has one writer; different instruments can run concurrently.
            for day in pd.date_range(first, last, freq="D"):
                periods = pd.date_range(day, periods=24, freq="h") if kind == "tick" else [day]
                for at in periods:
                    for side in (None,) if kind == "tick" else ("bid", "ask"):
                        result = self._archive(pair, scale, at, kind, side)
                        verified += result["status"] == "verified"
                        unavailable += result["status"] == "unavailable"
                        failed += result["status"] == "failed"
                        records += result.get("records", 0)
                        if progress:
                            progress({"instrument": pair, "period": at.isoformat(), **result})
            if verified and timeframe not in ("M1", "tick"):
                self.rebuild(pair, timeframe)
            report_path = None
            if self.storage.files(pair, timeframe):
                report_path = self.validate(pair, timeframe)
            return {
                "instrument": pair, "timeframe": timeframe, "verified_archives": verified,
                "unavailable_archives": unavailable, "failed_archives": failed, "source_records": records,
                "complete_archive_request": unavailable == 0 and failed == 0,
                "quality_report": str(report_path) if report_path else None,
            }

        with ThreadPoolExecutor(max_workers=self.settings.workers) as pool:
            results = list(pool.map(download_pair, instruments))
        return {"start_utc": first.isoformat(), "end_utc_inclusive": last.isoformat(), "results": results}

    def rebuild(self, pair: str, timeframe: str, from_ticks: bool = False) -> list[Path]:
        if timeframe not in RULES:
            raise ValueError("Timeframe de bougies requis.")
        if timeframe == "M1" and not from_ticks:
            raise ValueError("M1 original existe deja; utiliser --from-ticks pour une reconstruction.")
        source = "tick" if from_ticks else "M1"
        target = f"{timeframe}_from_ticks" if from_ticks else timeframe
        source_files = self.storage.files(pair, source)
        if not source_files:
            raise ValueError(f"Aucune source locale pour {pair}/{source}.")
        self.validate(pair, source)
        paths = []
        carry = pd.DataFrame()
        if from_ticks:
            # Keep only one hour of ticks in memory; larger bars are built from tick-derived M1.
            for _, frame in self.storage.partitions(pair, "tick"):
                bars = build_candles(frame, "M1", from_ticks=True)
                paths.extend(self.storage.write(bars, pair, "M1_from_ticks"))
            if timeframe == "M1":
                return paths
            source = "M1_from_ticks"
            carry = pd.DataFrame()
        for _, frame in self.storage.partitions(pair, source):
            if not carry.empty:
                frame = pd.concat([carry, frame], ignore_index=True).sort_values("timestamp", kind="stable")
            if source == "M1_from_ticks":
                # Preserve per-tick midpoint aggregation, never average Bid/Ask OHLC.
                mid = frame.loc[frame.side == "mid"].copy()
                quotes = frame.loc[frame.side != "mid"]
                bars = build_candles(quotes, timeframe)
                if not mid.empty:
                    mid_bid = mid.assign(side="bid")
                    mid_bars = build_candles(mid_bid, timeframe).assign(
                        side="mid", mid_method="OHLC aggregation of per-tick (bid + ask) / 2",
                    )
                    bars = pd.concat([bars, mid_bars], ignore_index=True)
                bars["input_timeframe"] = "M1_from_ticks"
                bars["source_url"] = "derived:M1_from_ticks"
            else:
                bars = build_candles(frame, timeframe)
            paths.extend(self.storage.write(bars, pair, target))
            if timeframe == "W1":
                last_start = bars.timestamp.max()
                carry = frame.loc[frame.timestamp >= last_start].copy()
            else:
                carry = pd.DataFrame()
        return list(dict.fromkeys(paths))

    def validate(self, pair: str, timeframe: str) -> Path:
        result = report(self.storage, pair, timeframe, self.settings.holidays)
        return self.storage.save_quality(pair, timeframe, result)

    def availability(self, pair: str, start: str, end: str, price_scale: int | None = None) -> dict:
        pair, scale = instrument(pair, price_scale)
        first, last = date_range(start, end)
        found = {}
        for day in pd.date_range(first, last, freq="D"):
            for side in ("bid", "ask"):
                if side in found:
                    continue
                result = self._archive(pair, scale, day, "M1", side)
                if result["status"] == "verified":
                    found[side] = day.isoformat()
            if len(found) == 2:
                break
        return {
            "instrument": pair, "search_start": first.isoformat(), "search_end": last.isoformat(),
            "first_observed_M1_archive": found,
            "scope": "first available within the scanned interval, not a certified all-time start; ticks require independent verification",
        }

    def clean(self, pair: str, timeframe: str) -> dict:
        files = self.storage.files(pair, timeframe)
        if not files:
            raise ValueError("Aucune donnee locale a nettoyer.")
        rejected = kept = 0
        audit_files = []
        for path, frame in self.storage.partitions(pair, timeframe):
            filtered, audit = clean(frame, timeframe)
            relative = path.relative_to(self.settings.root / "processed")
            output = self.settings.root / "cleaned" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_parquet(output, index=False, compression="zstd")
            audit_path = (self.settings.root / "reports" / "cleaning" / relative).with_suffix(".audit.csv")
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit.to_csv(audit_path, index=False)
            kept += len(filtered)
            rejected += len(audit)
            audit_files.append(str(audit_path))
            LOG.warning("Nettoyage explicite %s: %s lignes rejetees; audit %s", path, len(audit), audit_path)
        return {"kept": kept, "rejected": rejected, "audit_files": audit_files, "originals_unchanged": True}
