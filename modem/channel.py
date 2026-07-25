"""Канал-симулятор: чистые функции порчи сигнала (тикет 03).

Каждая порча — отдельная функция «массив сэмплов → массив сэмплов» с
параметрами; сложные каналы собираются композицией вызовов. Случайность —
только через явный np.random.Generator: одинаковый seed даёт одинаковый канал.
Звукового железа здесь нет — симулятор питает петлю tx→канал→rx в pytest.

clock_drift — не только порча: rx строит им банк шаблонов преамбулы,
модель времени дрейфующих часов у симулятора и приёмника общая.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from modem.profiles import SAMPLE_RATE


def band_stop(
    samples: np.ndarray, lo_hz: float, hi_hz: float, order: int = 8
) -> np.ndarray:
    """Провал АЧХ: полоса lo..hi вырезана фильтром (тикет 07).

    Модель «динамик/комната съели кусок спектра» для проверки калибровки.
    Фильтрация нуль-фазовая (sosfiltfilt): провал портит спектр, но не
    сдвигает сигнал во времени — тест меряет только потерю полосы.
    """
    nyquist = SAMPLE_RATE / 2
    sos = butter(order, [lo_hz / nyquist, hi_hz / nyquist], btype="bandstop", output="sos")
    return sosfiltfilt(sos, samples)


def awgn(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Аддитивный белый гауссов шум с заданным SNR (дБ) к мощности сигнала."""
    signal_power = float(np.mean(np.square(samples)))
    noise_power = signal_power / 10 ** (snr_db / 10)
    return samples + rng.normal(0.0, np.sqrt(noise_power), len(samples))


def attenuate(samples: np.ndarray, gain: float) -> np.ndarray:
    """Затухание (или усиление): умножение на линейный коэффициент."""
    return samples * gain


def shift_start(samples: np.ndarray, offset_samples: int) -> np.ndarray:
    """Случайный момент старта записи: тишина перед сигналом."""
    return np.concatenate([np.zeros(offset_samples), samples])


def echo(samples: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    """Простое эхо: к сигналу добавляется его задержанная ослабленная копия."""
    out = np.concatenate([samples, np.zeros(delay_samples)])
    out[delay_samples:] += gain * samples
    return out


def clock_drift(samples: np.ndarray, drift: float) -> np.ndarray:
    """Рассинхрон часов: drift=+0.001 — часы передатчика на 0.1% быстрее
    часов приёмника, сигнал в сетке приёмника сжат в (1+drift) раз
    (drift<0 — растянут). Линейная передискретизация всего сигнала."""
    positions = np.arange(round(len(samples) / (1 + drift))) * (1 + drift)
    return np.interp(positions, np.arange(len(samples)), samples)
