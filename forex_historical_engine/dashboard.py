from __future__ import annotations

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue

import pandas as pd
import requests
import streamlit as st

from forex_historical_engine.config import MAJORS, MINORS, RULES, Settings
from forex_historical_engine.engine import Engine


def main() -> None:
    st.set_page_config(page_title="Dukascopy Historical Engine", layout="wide")
    st.title("Historique Forex — Dukascopy")
    st.caption("Cotations de cette source, pas un historique universel. Aucun remplissage artificiel.")
    root = Path(st.sidebar.text_input("Dossier des données", "data"))
    cache = Path(st.sidebar.text_input("Cache Parquet par paquet", str(root / "cache")))
    workers = st.sidebar.slider("Téléchargements simultanés", 1, 8, 2)
    engine = Engine(Settings(root=root, workers=workers, cache=cache))
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("forex_historical_engine")
    logger.setLevel(logging.INFO)
    log_path = (logs / "download.log").resolve()
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path) for handler in logger.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    pairs = st.multiselect("Paires", MAJORS + MINORS, default=["EURUSD"])
    timeframe = st.selectbox("Timeframe", ["tick", *RULES], index=1)
    yesterday = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    left, right = st.columns(2)
    start = left.date_input("Début UTC", yesterday - timedelta(days=6), max_value=yesterday)
    end = right.date_input("Fin UTC incluse", yesterday, max_value=yesterday)
    st.warning("Un historique long de ticks peut être volumineux. Les fermetures et archives absentes restent signalées.")
    if st.button("Télécharger / reprendre les périodes incomplètes", type="primary"):
        output = st.empty()
        bar = st.progress(0.0)
        total = max(1, len(pairs) * ((end - start).days + 1) * (24 if timeframe == "tick" else 2))
        events: Queue[dict] = Queue()

        def progress(event: dict) -> None:
            events.put(event)

        try:
            with st.spinner("Téléchargement et vérifications en cours…"), ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(engine.download, pairs, timeframe, start.isoformat(), end.isoformat(), progress=progress)
                completed = 0
                while not future.done() or not events.empty():
                    try:
                        event = events.get(timeout=0.1)
                    except Empty:
                        continue
                    completed += 1
                    bar.progress(min(1.0, completed / total))
                    output.caption(f"{event['instrument']} — {event['period']} — {event['status']}")
                result = future.result()
            output.json(result)
            if all(item["complete_archive_request"] for item in result["results"]):
                st.success("Archives demandées vérifiées. Consulter également les rapports qualité.")
            else:
                st.warning("Demande incomplète : des archives sont absentes. Voir les rapports et journaux.")
        except (ValueError, OSError, RuntimeError, requests.RequestException, sqlite3.Error) as error:
            logger.exception("Téléchargement interrompu")
            st.error(str(error))
    st.subheader("État de la base")
    with engine.storage.connect() as connection:
        status = pd.read_sql_query("SELECT status,COUNT(*) AS archives FROM downloads GROUP BY status", connection)
        reports = pd.read_sql_query("SELECT * FROM quality ORDER BY instrument,timeframe", connection)
        incomplete = pd.read_sql_query(
            "SELECT instrument,period,status,error FROM downloads WHERE status!='verified' ORDER BY period DESC LIMIT 100", connection,
        )
    st.dataframe(status)
    st.dataframe(reports)
    st.dataframe(incomplete)
    available = sorted({(path.parts[-4], path.parts[-3]) for path in (root / "processed").glob("*/*/*/*.parquet")})
    available += sorted({(path.parts[-5], path.parts[-4]) for path in (root / "processed").glob("*/*/*/*/*.parquet")})
    if available:
        pair, local_timeframe = st.selectbox("Données locales", sorted(set(available)))
        paths = engine.storage.files(pair, local_timeframe)
        st.write(f"{len(paths)} partition(s)")
        if paths:
            st.dataframe(engine.storage.read_partition(paths[-1]).head(100))
        report_path = root / "reports" / pair / f"{local_timeframe}.json"
        if report_path.exists():
            st.json(json.loads(report_path.read_text(encoding="utf-8")))
        if st.button("Vérifier et générer le rapport"):
            try:
                st.json(json.loads(engine.validate(pair, local_timeframe).read_text(encoding="utf-8")))
            except (ValueError, OSError, RuntimeError) as error:
                logger.exception("Validation interrompue")
                st.error(str(error))
        if st.button("Préparer un export CSV"):
            export = root / "exports" / f"{pair}_{local_timeframe}.csv"
            try:
                count = engine.storage.export(pair, local_timeframe, export)
                st.info(f"{count} lignes exportées vers {export}. Export sur disque pour limiter la mémoire.")
            except (ValueError, OSError, RuntimeError) as error:
                logger.exception("Export interrompu")
                st.error(str(error))
    st.subheader("Journaux récents")
    if log_path.exists():
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 12000))
            st.code(handle.read().decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
