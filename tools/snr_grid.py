"""Разведка порога SNR профиля: петля tx→канал→rx по сетке SNR × сиды.

Запуск из корня репо:  python tools/snr_grid.py [robust|medium|fast]
Печатает исход каждой ячейки (ok / bits / drop) для чистого шума и для
жёсткого канала (шум + дрейф часов +0.1% + эхо 50 мс). Этой разведкой
найдены регрессионные границы tests/test_channel.py и таблица README.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem import channel, demodulate, modulate

BLOCKS = [b"nuclear it hack", bytes(range(64)), b"\x00\xff" * 12]
# Сетка SNR своя на профиль: пороги fast мягкие (аккорд делит амплитуду
# на 9 поднесущих), его пол лежит там, где robust/medium давно чисты.
SNR_GRIDS_DB = {
    "robust": (6, 3, 0, -3, -6, -9, -12, -15, -18),
    "medium": (6, 3, 0, -3, -6, -9, -12, -15, -18),
    "fast": (27, 24, 21, 18, 15, 12, 9, 6, 3),
}
SEEDS = range(8)


def outcome(
    clean: np.ndarray, profile: str, snr_db: float, seed: int, hard: bool
) -> str:
    signal = clean
    if hard:
        signal = channel.echo(
            channel.clock_drift(signal, +0.001), delay_samples=2_400, gain=0.35
        )
    signal = channel.awgn(
        channel.attenuate(signal, 0.5), snr_db, np.random.default_rng(seed)
    )
    got = demodulate(signal, profile)
    if got == BLOCKS:
        return "ok"
    return "drop" if len(got) < len(BLOCKS) else "bits"


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "robust"
    clean = modulate(BLOCKS, profile)
    print(f"Профиль {profile}: ok / bits (битые байты) / drop (потеря кадра)")
    for snr_db in SNR_GRIDS_DB[profile]:
        for hard in (False, True):
            row = [outcome(clean, profile, snr_db, seed, hard) for seed in SEEDS]
            label = "hard " if hard else "clean"
            print(f"SNR {snr_db:+3d} дБ {label}: {' '.join(row)}")


if __name__ == "__main__":
    main()
