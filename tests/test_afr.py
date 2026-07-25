"""АЧХ-свип (тикет 06): чистые функции анализа спектра записи.

Сам инструмент (проиграть/записать) — железо, руками; тестируем только
анализ: энергия видна там, где она есть, и не видна там, где её нет.
"""

import numpy as np

from modem import SAMPLE_RATE
from tools.afr_sweep import band_db, spectrum_db


def test_band_db_sees_energy_in_its_band_and_not_elsewhere():
    # Свип, заполняющий ровно полосу robust: медиана полосы должна быть
    # высокой в ней и на уровне шума вне её.
    seconds = 2.0
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    rate = (6_900.0 - 4_500.0) / seconds
    signal = 0.5 * np.sin(2 * np.pi * (4_500.0 * t + rate * t**2 / 2))
    rng = np.random.default_rng(0)
    signal = signal + 1e-4 * rng.standard_normal(len(signal))

    freqs, db = spectrum_db(signal)

    in_band = band_db(freqs, db, 4_500.0, 6_900.0)     # полоса robust
    out_band = band_db(freqs, db, 10_000.0, 12_000.0)  # там энергии нет
    assert in_band > out_band + 20.0
