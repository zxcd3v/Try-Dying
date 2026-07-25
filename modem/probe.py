"""Зонд калибровки: озвучить полосы и померить их SNR одним FFT (ADR-0003).

Зонд — широкополосный сигнал (~0.5 с), по одному приёму которого RX меряет
SNR всех полос профиля сразу: последовательный перебор полос не нужен, вся
информация есть в одном спектре. Геометрия самих полос — в profiles.py.

Здесь только чистые функции «массив сэмплов ↔ измерение»; кто и когда играет
зонд, решает эндпоинт (modem/audio.py) и рукопожатие (protocol/calibration.py).
"""

import numpy as np

from modem.profiles import ANCHOR, PROFILES, SAMPLE_RATE, bands
from modem.rx import find_preamble
from modem.tx import chirp

# 0.5 с мультитона: меньше — грубее оценка шума, больше — дольше
# рукопожатие (ADR-0003 закладывает 1–2 с на сеанс).
PROBE_SAMPLES = 24_000

# Весь зонд целиком (преамбула + пауза + мультитон) — длина одинакова для
# всех профилей: эндпоинт по ней понимает, когда есть смысл мерить.
PROBE_TOTAL_SAMPLES = (
    PROFILES[ANCHOR].preamble_samples + PROFILES[ANCHOR].preamble_gap_samples
) + PROBE_SAMPLES

# Ширина окна энергии вокруг частоты: тон под окном Ханна размазан на ~3
# бина (бин 2 Гц), дрейф часов ±0.1% уводит верхние тона ещё на ~9 Гц.
_ENERGY_TOL_HZ = 15.0

_EPS = 1e-20
_MAX_SNR_DB = 60.0  # синтетический канал без шума иначе дал бы +inf


def probe_tones_hz(profile_name: str) -> tuple[float, ...]:
    """Все тона всех полос профиля — сетка, которую озвучивает зонд."""
    tones = PROFILES[profile_name].tones_hz
    return tuple(sorted({t + b.offset_hz for b in bands(profile_name) for t in tones}))


def probe(profile_name: str) -> np.ndarray:
    """Зонд профиля: якорная преамбула + мультитон всех полос.

    Преамбула — тот же chirp, что метит аир-кадры: приёмник находит зонд и
    оценивает дрейф часов уже отлаженным банком шаблонов (rx.find_preamble)
    вместо хрупкой эвристики «самый громкий кусок записи». Преамбула
    якорная (robust): зонд — служебный сигнал, а служебное живёт на якоре.
    """
    anchor = PROFILES[ANCHOR]
    return np.concatenate(
        [
            chirp(anchor),
            np.zeros(anchor.preamble_gap_samples),
            _multitone(probe_tones_hz(profile_name), anchor.amplitude),
        ]
    ).astype(np.float32)


def find_probe(samples: np.ndarray) -> tuple[int, float] | None:
    """Начало мультитона в записи и масштаб времени; None — зонда нет."""
    hit = find_preamble(samples, PROFILES[ANCHOR])
    if hit is None:
        return None
    start, scale = hit
    anchor = PROFILES[ANCHOR]
    offset = anchor.preamble_samples + anchor.preamble_gap_samples
    return start + round(offset * scale), scale


def band_snr_db(samples: np.ndarray, profile_name: str) -> list[float] | None:
    """SNR каждой полосы профиля по одному приёму зонда — одним FFT.

    None — зонд не найден либо не дозвучал до конца записи (эндпоинт по
    этому признаку понимает, что слушать ещё рано).

    Мера полосы — среднее в дБ по её тонам, а не отношение суммарных
    мощностей: полоса хороша ровно настолько, насколько слышен её худший
    тон, а среднее арифметическое мощностей замаскировало бы пару убитых
    тонов соседними громкими.
    """
    samples = np.asarray(samples, dtype=np.float64)
    located = find_probe(samples)
    if located is None:
        return None
    start, scale = located
    length = round(PROBE_SAMPLES * scale)
    if start < 0 or start + length > len(samples):
        return None

    spectrum = np.square(
        np.abs(np.fft.rfft(samples[start : start + length] * np.hanning(length)))
    )
    bin_hz = SAMPLE_RATE / length
    tol = max(1, round(_ENERGY_TOL_HZ / bin_hz))

    def energy(freq_hz: float) -> float:
        center = round(freq_hz / bin_hz)
        return float(np.sum(spectrum[max(center - tol, 0) : center + tol + 1]))

    profile = PROFILES[profile_name]
    # Опорная точка шума — середина между соседними тонами сетки: сигнала
    # там нет по построению, а тракт и шум те же, что у тона рядом.
    half_step = (profile.tones_hz[1] - profile.tones_hz[0]) / 2
    out = []
    for band in bands(profile_name):
        per_tone = [
            10
            * np.log10(
                max(
                    energy(tone + band.offset_hz)
                    / (energy(tone + band.offset_hz + half_step) + _EPS),
                    _EPS,
                )
            )
            for tone in profile.tones_hz
        ]
        out.append(min(float(np.mean(per_tone)), _MAX_SNR_DB))
    return out


def best_band(snr_db: list[float]) -> int:
    """Номер полосы с лучшим SNR."""
    return int(np.argmax(snr_db))


def _multitone(tones_hz: tuple[float, ...], amplitude: float) -> np.ndarray:
    """Сумма тонов с фазами Шрёдера — плоский спектр без пикового клиппинга.

    Фазы −πk²/N разводят тона во времени: сумма 20+ синусоид в нулевой фазе
    дала бы одиночный всплеск и мизерную среднюю мощность на тон.
    """
    t = np.arange(PROBE_SAMPLES) / SAMPLE_RATE
    n = len(tones_hz)
    wave = sum(
        np.sin(2 * np.pi * f * t - np.pi * k**2 / n) for k, f in enumerate(tones_hz)
    )
    return amplitude * wave / np.max(np.abs(wave))
