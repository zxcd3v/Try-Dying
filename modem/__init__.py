"""Модем — «глупая труба»: блок байтов ↔ звук. Контракт v1.2."""

from modem.audio import AudioEndpoint
from modem.probe import band_snr_db, best_band, probe
from modem.profiles import (
    ANCHOR,
    MAX_BLOCK,
    PROFILES,
    SAMPLE_RATE,
    Band,
    bands,
    default_band,
)
from modem.rx import demodulate, demodulate_stream
from modem.tx import modulate

__all__ = [
    "ANCHOR",
    "MAX_BLOCK",
    "PROFILES",
    "SAMPLE_RATE",
    "AudioEndpoint",
    "Band",
    "band_snr_db",
    "bands",
    "best_band",
    "default_band",
    "demodulate",
    "demodulate_stream",
    "modulate",
    "probe",
]
