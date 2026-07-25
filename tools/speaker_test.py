"""Быстрая проверка динамика: три тона + chirp модема.

Запуск из корня репо:  python tools/speaker_test.py [устройство]
Устройство — номер или подстрока имени (как --out-device у send.py);
без аргумента — устройство вывода по умолчанию. Если слышно писки и
«свист» вверх — динамик работает и годится для модема.
"""

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem.profiles import PROFILES, SAMPLE_RATE
from modem.tx import chirp


def tone(freq_hz: float, duration_s: float = 0.4) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    wave = 0.6 * np.sin(2 * np.pi * freq_hz * t)
    ramp = np.minimum(1.0, np.arange(len(wave)) / 480)
    return (wave * ramp * ramp[::-1]).astype(np.float32)


def main() -> None:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        sd.default.device = (None, int(arg) if arg.isdigit() else arg)
    device = sd.query_devices(sd.default.device[1] if sd.default.device else None,
                              "output")
    print(f"Устройство вывода: {device['name']}")

    gap = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
    parts = []
    for freq in (500.0, 1_000.0, 4_000.0):  # 4 кГц — середина сеток модема
        print(f"  тон {freq:.0f} Гц…")
        parts += [tone(freq), gap]
    print("  chirp-преамбула robust (свист 2→8 кГц)…")
    parts.append(chirp(PROFILES["robust"]).astype(np.float32))

    sd.play(np.concatenate(parts), SAMPLE_RATE, blocking=True)
    print("Готово. Слышали три писка и свист вверх? Значит, динамик работает.")


if __name__ == "__main__":
    main()
