"""Уровнемер входа (тикет 06, диагностика): слышит ли нода микрофон.

Открывает вход ровно тем же кодом, что AudioEndpoint (_MicSource:
sounddevice.InputStream, 48 кГц, mono, float32), и печатает живой уровень.
Говорите в микрофон — полоска должна прыгать. Мёртвый ноль при живом
голосе = нода слушает не то устройство: выбрать явно через --in-device
(номер или подстрока имени из sounddevice.query_devices()).

Запуск из корня репо:
  python tools/mic_meter.py                 # вход по умолчанию, 15 с
  python tools/mic_meter.py --in-device 14  # явное устройство
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem.audio import _MicSource

_WINDOW_S = 0.25  # шаг полоски
_DEAD_RMS = 1e-4  # тише — считаем, что устройство отдаёт нули
_BAR_WIDTH = 50


def _bar(rms: float) -> str:
    """Полоска в дБFS: -60 дБ = пусто, 0 дБ = полная."""
    db = 20 * np.log10(max(rms, 1e-9))
    filled = int(np.clip((db + 60) / 60, 0, 1) * _BAR_WIDTH)
    return f"[{'#' * filled}{'.' * (_BAR_WIDTH - filled)}] {db:6.1f} дБFS"


def main() -> None:
    parser = argparse.ArgumentParser(description="Живой уровень входа нодовским путём.")
    parser.add_argument("--in-device", default=None, metavar="УСТР",
                        help="устройство записи: номер или подстрока имени")
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="сколько секунд мерить (по умолчанию 15)")
    args = parser.parse_args()

    device = args.in_device
    if device is not None and device.isdigit():
        device = int(device)

    source = _MicSource(device)
    source.start()
    print("Говорите в микрофон; полоска должна прыгать. Ctrl+C — выход.")

    from modem.profiles import SAMPLE_RATE

    peak_rms = 0.0
    window: list[np.ndarray] = []
    window_len = 0
    got = 0
    try:
        while got < args.seconds * SAMPLE_RATE:
            chunk = source.get(0.5)
            if chunk is None:
                print("!! поток открыт, но сэмплы не идут — устройство молчит")
                continue
            got += len(chunk)
            window.append(chunk)
            window_len += len(chunk)
            if window_len >= _WINDOW_S * SAMPLE_RATE:
                joined = np.concatenate(window)
                window, window_len = [], 0
                rms = float(np.sqrt((joined ** 2).mean()))
                peak_rms = max(peak_rms, rms)
                print(f"\r{_bar(rms)}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    if peak_rms < _DEAD_RMS:
        print("ВЕРДИКТ: мёртвая тишина — нода слушает не то устройство.")
        print("Выберите вход явно: --in-device НОМЕР (см. sounddevice.query_devices()).")
    else:
        db = 20 * np.log10(peak_rms)
        print(f"ВЕРДИКТ: вход живой, пик {db:.1f} дБFS — этим устройством можно принимать.")


if __name__ == "__main__":
    main()
