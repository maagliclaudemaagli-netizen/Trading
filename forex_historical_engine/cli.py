from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd
import requests

from .budget import ArchiveBudget
from .config import MAJORS, MINORS, RULES, Settings, instrument
from .downloader import DukascopyClient
from .engine import Engine
from .raw_ticks import download_raw_ticks


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Archives historiques Dukascopy, sans remplissage artificiel.")
    root.add_argument("--data-dir", type=Path, default=Path("data"))
    root.add_argument("--cache", type=Path, help="Cache Parquet par paquet; data-dir/cache par defaut.")
    root.add_argument("--workers", type=int, default=2)
    root.add_argument("--delay", type=float, default=1.0, help="Pause entre requetes, minimum 1 seconde.")
    root.add_argument("--retries", type=int, choices=range(4), default=3, help="Reessais apres la requete initiale, maximum 3.")
    root.add_argument("--timeout", type=float, default=30)
    root.add_argument("--holiday", action="append", default=[], help="Date UTC de fermeture verifiee, YYYY-MM-DD.")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("download", "availability", "validate", "report", "export", "rebuild", "clean"):
        command = commands.add_parser(name)
        if name == "download":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--pair")
            selection.add_argument("--pairs", choices=("majors", "minors", "all"))
        else:
            command.add_argument("--pair", required=True)
        command.add_argument("--price-scale", type=int, help="Diviseur des prix entiers pour un instrument ajoute.")
        if name in ("download", "availability"):
            command.add_argument("--cache", type=Path, default=argparse.SUPPRESS, help="Cache Parquet local par paquet.")
            dates = command.add_mutually_exclusive_group()
            dates.add_argument("--start")
            dates.add_argument("--years", type=int, choices=(1, 5, 10, 15, 20))
            dates.add_argument("--since-first", action="store_true", help="Chercher M1 dans l'intervalle --search-start/--end.")
            command.add_argument("--search-start", help="Borne explicite de recherche; debut absolu non certifie.")
            command.add_argument("--end", help="Dernier jour UTC inclus; hier par defaut.")
        if name != "availability":
            choices = ["tick", *RULES, *(f"{key}_from_ticks" for key in RULES)]
            command.add_argument("--timeframe", choices=choices, default="M1")
        if name == "rebuild":
            command.add_argument("--from-ticks", action="store_true")
        if name == "export":
            command.add_argument("--format", choices=("csv",), default="csv")
            command.add_argument("--side", choices=("bid", "ask", "mid"))
            command.add_argument("--output", type=Path, required=True)
    raw_ticks = commands.add_parser("download-ticks", help="Telecharger les ticks bruts avec un plafond de stockage.")
    selection = raw_ticks.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pair")
    selection.add_argument("--pairs", choices=("majors", "minors", "all"))
    raw_ticks.add_argument("--include-xauusd", action="store_true", help="Essayer aussi XAUUSD sans supposer son echelle.")
    raw_ticks.add_argument("--start", required=True, help="Premier jour UTC inclus.")
    raw_ticks.add_argument("--end", required=True, help="Dernier jour UTC inclus.")
    raw_ticks.add_argument("--max-bytes", type=int, required=True, help="Plafond des fichiers dans le dossier de sortie.")
    raw_ticks.add_argument("--output-dir", type=Path, required=True, help="Dossier dedie aux archives brutes.")
    commands.add_parser("status")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        holidays = tuple(arguments.holiday)
        for date in holidays:
            if pd.Timestamp(date).strftime("%Y-%m-%d") != date:
                raise ValueError("--holiday attend YYYY-MM-DD.")
        root = arguments.output_dir if arguments.command == "download-ticks" else arguments.data_dir
        settings = Settings(
            root, arguments.workers, arguments.delay, arguments.retries,
            arguments.timeout, holidays, cache=arguments.cache,
        )
        if arguments.command == "download-ticks":
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            if arguments.cache is not None:
                raise ValueError("--cache n'est pas utilise par download-ticks.")
            if arguments.pair and arguments.include_xauusd:
                raise ValueError("--include-xauusd s'utilise avec --pairs.")
            groups = {"majors": MAJORS, "minors": MINORS, "all": MAJORS + MINORS}
            pairs = [arguments.pair] if arguments.pair else list(groups[arguments.pairs])
            if arguments.include_xauusd:
                pairs.insert(0, "XAUUSD")
            budget = ArchiveBudget(settings.root, arguments.max_bytes)
            result = download_raw_ticks(DukascopyClient(settings), budget, pairs, arguments.start, arguments.end)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if result["complete_archive_request"] else 2
        logs = settings.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(logs / "download.log", encoding="utf-8")],
        )
        engine = Engine(settings)
        if arguments.command == "status":
            with engine.storage.connect() as connection:
                statuses = dict(connection.execute("SELECT status,COUNT(*) FROM downloads GROUP BY status"))
                quality = [list(row) for row in connection.execute("SELECT instrument,timeframe,score,report_path FROM quality")]
            result = {"downloads": statuses, "quality_reports": quality, "database": str(engine.storage.database)}
        else:
            pair = arguments.pair if hasattr(arguments, "pair") else None
            if pair:
                pair, _ = instrument(pair, arguments.price_scale)
            if arguments.command != "download" and pair is None:
                raise ValueError("Instrument requis.")
            if arguments.command in ("download", "availability"):
                end = arguments.end or (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if arguments.years:
                    start = (pd.Timestamp(end) - pd.DateOffset(years=arguments.years)).strftime("%Y-%m-%d")
                else:
                    start = arguments.start or (pd.Timestamp(end) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
                if arguments.since_first:
                    if not pair or not arguments.search_start:
                        raise ValueError("--since-first exige --pair et --search-start; metadata officielle inaccessible.")
                    availability = engine.availability(pair, arguments.search_start, end, arguments.price_scale)
                    found = availability["first_observed_M1_archive"]
                    if len(found) != 2:
                        raise ValueError(f"Debut Bid/Ask non trouve dans l'intervalle: {availability}")
                    start = min(found.values())[:10]
                if arguments.command == "availability":
                    if not pair:
                        raise ValueError("Instrument requis.")
                    result = engine.availability(pair, start, end, arguments.price_scale)
                else:
                    if arguments.timeframe.endswith("_from_ticks"):
                        raise ValueError("download accepte les timeframes originaux; utiliser rebuild --from-ticks ensuite.")
                    groups = {"majors": MAJORS, "minors": MINORS, "all": MAJORS + MINORS}
                    pairs = [pair] if pair else list(groups[arguments.pairs])
                    result = engine.download(pairs, arguments.timeframe, start, end, arguments.price_scale)
            elif pair is None:
                raise ValueError("Instrument requis.")
            elif arguments.command in ("validate", "report"):
                path = engine.validate(pair, arguments.timeframe)
                result = json.loads(path.read_text(encoding="utf-8"))
                result["report_path"] = str(path)
            elif arguments.command == "rebuild":
                paths = engine.rebuild(pair, arguments.timeframe, arguments.from_ticks)
                target = f"{arguments.timeframe}_from_ticks" if arguments.from_ticks else arguments.timeframe
                report_path = engine.validate(pair, target)
                result = {"files": [str(path) for path in paths], "report_path": str(report_path)}
            elif arguments.command == "clean":
                result = engine.clean(pair, arguments.timeframe)
            else:
                count = engine.storage.export(pair, arguments.timeframe, arguments.output, arguments.side)
                result = {"rows": count, "output": str(arguments.output)}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if arguments.command == "download":
            return 0 if all(item["complete_archive_request"] for item in result["results"]) else 2
        if arguments.command == "availability":
            return 0 if len(result["first_observed_M1_archive"]) == 2 else 2
        return 0
    except (ValueError, OSError, RuntimeError, requests.RequestException, sqlite3.Error) as error:
        logging.getLogger(__name__).exception("Commande interrompue: %s", error)
        return 1
