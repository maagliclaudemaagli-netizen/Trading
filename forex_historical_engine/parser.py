from __future__ import annotations

import lzma
import struct

import numpy as np
import pandas as pd

from .config import utc

TICK = struct.Struct(">IIIff")
CANDLE = struct.Struct(">IIIIIf")


def decode(content: bytes, kind: str) -> bytes:
    if kind not in ("tick", "M1"):
        raise ValueError(f"Format inconnu: {kind}")
    if not content:
        raise ValueError("Archive vide.")
    try:
        decoded = lzma.decompress(content)
    except lzma.LZMAError as error:
        raise ValueError("Archive LZMA invalide.") from error
    size = TICK.size if kind == "tick" else CANDLE.size
    if not decoded or len(decoded) % size:
        raise ValueError("Archive vide ou longueur binaire incoherente.")
    return decoded


def parse(
    content: bytes, kind: str, origin: pd.Timestamp, pair: str,
    scale: int, url: str, side: str | None = None,
) -> pd.DataFrame:
    if scale <= 0:
        raise ValueError("Echelle de prix invalide.")
    origin = utc(origin)
    decoded = decode(content, kind)
    if kind == "tick":
        frame = pd.DataFrame(
            TICK.iter_unpack(decoded),
            columns=["timestamp_original", "ask_raw", "bid_raw", "ask_volume", "bid_volume"],
        )
        offsets = frame.timestamp_original
        if (offsets >= 3_600_000).any():
            raise ValueError("Offset de tick hors de son heure UTC.")
        frame["timestamp"] = origin + pd.to_timedelta(offsets, unit="ms")
        for quote in ("bid", "ask"):
            frame[quote] = frame[f"{quote}_raw"] / scale
        frame["volume_kind"] = "source_quote_volume"
    else:
        if side not in ("bid", "ask"):
            raise ValueError("M1 exige un cote bid ou ask.")
        frame = pd.DataFrame(
            CANDLE.iter_unpack(decoded),
            columns=["timestamp_original", "open_raw", "close_raw", "low_raw", "high_raw", "volume"],
        )
        offsets = frame.timestamp_original
        if ((offsets >= 86400) | (offsets % 60 != 0)).any():
            raise ValueError("Offset M1 invalide.")
        frame["timestamp"] = origin + pd.to_timedelta(offsets, unit="s")
        for field in ("open", "high", "low", "close"):
            frame[field] = frame[f"{field}_raw"] / scale
        frame["side"] = side
        frame["timeframe"] = "M1"
        frame["reconstructed"] = False
        frame["volume_kind"] = "source_quote_volume"
    frame["instrument"] = pair
    frame["source"] = "Dukascopy"
    frame["source_url"] = url
    frame["source_record"] = np.arange(len(frame), dtype=np.int64)
    frame["origin_utc"] = origin
    frame["price_scale"] = scale
    return frame
