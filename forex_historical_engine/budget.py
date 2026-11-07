from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class BudgetReached(RuntimeError):
    """The next archive cannot fit in the requested storage budget."""


class ArchiveBudget:
    def __init__(self, root: Path, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("Le budget doit etre strictement positif.")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self._lock = threading.Lock()
        self._active = False
        self._refresh()

    def _refresh(self) -> None:
        self.used_bytes = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Lien symbolique interdit dans le dossier limite: {path}")
            if path.is_file():
                self.used_bytes += path.stat().st_size
        if self.used_bytes > self.max_bytes:
            raise ValueError("Le dossier contient deja plus de donnees que le budget autorise.")

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        lock_path = self.root / ".download-ticks.lock"
        try:
            lock_path.mkdir()
        except FileExistsError as error:
            raise RuntimeError(
                f"Telechargement deja actif ou verrou abandonne: {lock_path}. "
                "Verifier les processus avant toute suppression manuelle."
            ) from error
        try:
            self._refresh()
            self._active = True
            yield
        finally:
            self._active = False
            lock_path.rmdir()

    def save(self, path: Path, content: bytes) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("Le verrou du dossier limite est requis.")
            resolved = path.resolve()
            if resolved != path.absolute() or not resolved.is_relative_to(self.root):
                raise ValueError(f"Chemin d'archive hors du dossier limite: {path}")
            # Both temporary and published filenames count during the no-overwrite publication.
            if self.used_bytes + 2 * len(content) > self.max_bytes:
                raise BudgetReached("L'archive et son fichier temporaire depasseraient le budget autorise.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            created = False
            try:
                with temporary.open("xb") as handle:
                    created = True
                    handle.write(content)
                os.link(temporary, path)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
            self.used_bytes += len(content)
