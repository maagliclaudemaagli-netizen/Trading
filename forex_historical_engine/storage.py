from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class Storage:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "metadata.sqlite"
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS downloads (
                    url TEXT PRIMARY KEY, instrument TEXT NOT NULL, period TEXT NOT NULL,
                    status TEXT NOT NULL, raw_path TEXT, sha256 TEXT, records INTEGER,
                    error TEXT, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quality (
                    instrument TEXT, timeframe TEXT, report_path TEXT, score REAL,
                    updated_at TEXT, PRIMARY KEY (instrument, timeframe)
                );
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, records INTEGER NOT NULL
                );
            """)

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    def record(
        self, url: str, pair: str, at: pd.Timestamp, status: str,
        path: Path | None = None, content: bytes | None = None,
        records: int | None = None, error: str | None = None,
        sha256: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO downloads VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(url) DO UPDATE SET
                    status=excluded.status, raw_path=COALESCE(excluded.raw_path,downloads.raw_path),
                    sha256=COALESCE(excluded.sha256,downloads.sha256),
                    records=COALESCE(excluded.records,downloads.records),
                    error=excluded.error, updated_at=excluded.updated_at""",
                (url, pair, at.isoformat(), status, str(path) if path else None,
                 fingerprint(content) if content is not None else sha256, records, error,
                 pd.Timestamp.now(tz="UTC").isoformat()),
            )

    def download(self, url: str) -> dict | None:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            result = connection.execute("SELECT * FROM downloads WHERE url=?", (url,)).fetchone()
        return dict(result) if result is not None else None

    def write(self, frame: pd.DataFrame, pair: str, timeframe: str, chunk: str | None = None) -> list[Path]:
        if frame.empty:
            raise ValueError("Impossible d'enregistrer des donnees vides.")
        paths = []
        dates = pd.DatetimeIndex(frame.timestamp)
        for (year, month), positions in pd.Series(range(len(frame))).groupby([dates.year, dates.month]):
            current = frame.iloc[positions.to_numpy()].copy()
            base = self.root / "processed" / pair / timeframe / str(year)
            path = base / (f"{month:02}/{chunk}.parquet" if chunk else f"{month:02}.parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not chunk:
                old = self.read_partition(path)
                if current.reconstructed.all():
                    keys = pd.MultiIndex.from_frame(current[["timestamp", "side"]])
                    old_keys = pd.MultiIndex.from_frame(old[["timestamp", "side"]])
                    old = old.loc[~old_keys.isin(keys)]
                else:
                    # Replace by archive identity, not timestamp: simultaneous ticks may differ.
                    sources = current.source_url.unique()
                    old = old.loc[~old.source_url.isin(sources)]
                current = pd.concat([old, current], ignore_index=True)
            order = ["timestamp"] + (["side"] if "side" in current else [])
            current = current.sort_values(order, kind="stable").reset_index(drop=True)
            self.write_partition(current, path)
            paths.append(path)
        return paths

    def write_partition(self, frame: pd.DataFrame, path: Path) -> None:
        if frame.empty:
            raise ValueError("Impossible d'enregistrer une partition vide.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.part")
        frame.to_parquet(temporary, index=False, compression="zstd")
        pd.read_parquet(temporary, columns=["timestamp"])
        temporary.replace(path)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?)",
                (str(path), file_fingerprint(path), len(frame)),
            )
    def files(self, pair: str, timeframe: str) -> list[Path]:
        return sorted((self.root / "processed" / pair / timeframe).glob("**/*.parquet"))

    def partitions(self, pair: str, timeframe: str):
        for path in self.files(pair, timeframe):
            yield path, self.read_partition(path)

    def read_partition(self, path: Path) -> pd.DataFrame:
        with self.connect() as connection:
            record = connection.execute("SELECT sha256 FROM files WHERE path=?", (str(path),)).fetchone()
        if record is not None and file_fingerprint(path) != record[0]:
            raise ValueError(f"Integrite Parquet invalide: {path}")
        return pd.read_parquet(path)

    def cached_archive(self, pair: str, at: pd.Timestamp, kind: str, url: str, records: int) -> bool:
        base = self.root / "processed" / pair / kind / str(at.year)
        path = base / (f"{at.month:02}/{at.strftime('%Y-%m-%d-%H')}.parquet" if kind == "tick" else f"{at.month:02}.parquet")
        if not path.exists():
            return False
        frame = self.read_partition(path)
        return int((frame.source_url == url).sum()) == records

    def read(self, pair: str, timeframe: str) -> pd.DataFrame:
        if timeframe == "tick":
            raise ValueError("Lecture globale des ticks interdite; utiliser partitions() ou export().")
        frames = [frame for _, frame in self.partitions(pair, timeframe)]
        if not frames:
            raise ValueError(f"Aucune donnee locale pour {pair}/{timeframe}.")
        return pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")

    def save_quality(self, pair: str, timeframe: str, report: dict) -> Path:
        path = self.root / "reports" / pair / f"{timeframe}.json"
        atomic_json(path, report)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO quality VALUES (?,?,?,?,?)",
                (pair, timeframe, str(path), report["quality_score"],
                 pd.Timestamp.now(tz="UTC").isoformat()),
            )
        return path

    def export(self, pair: str, timeframe: str, path: Path, side: str | None = None) -> int:
        path = path.resolve()
        for protected in ("raw", "processed", "cleaned"):
            if path.is_relative_to((self.root / protected).resolve()):
                raise ValueError("Exporter hors des dossiers raw/processed/cleaned pour preserver les originaux.")
        if not self.files(pair, timeframe):
            raise ValueError(f"Aucune donnee locale pour {pair}/{timeframe}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        count = 0
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            for _, frame in self.partitions(pair, timeframe):
                if side is not None:
                    if "side" not in frame:
                        raise ValueError("Les ticks contiennent bid et ask, pas une colonne side.")
                    frame = frame.loc[frame.side == side]
                if frame.empty:
                    continue
                frame.to_csv(handle, index=False, header=count == 0)
                count += len(frame)
        if count == 0:
            temporary.unlink()
            raise ValueError("Aucune donnee pour cet export.")
        temporary.replace(path)
        return count
