"""АЧХ-свип (тикет 06): проиграть свип, записать, показать спектр.

Меряем сквозную АЧХ пары «динамик TX → воздух → микрофон RX», чтобы
сверить границы полос профилей с фактами двух реальных ноутбуков.

Запуск из корня репо:
  python tools/afr_sweep.py play                 # на передающем ноутбуке
  python tools/afr_sweep.py record               # на принимающем (запустить первым)
  python tools/afr_sweep.py both                 # один ноутбук: сам играет и пишет

record/both сохраняют WAV и картинку спектра (out/afr.wav, out/afr.png)
и печатают уровень каждой полосы профилей относительно пика записи.
Свип линейный — равная энергия на герц, PSD полос сравнима напрямую.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem import PROFILES, SAMPLE_RATE

SWEEP_LO_HZ = 300.0
SWEEP_HI_HZ = 12_000.0
_RAMP_SAMPLES = 480  # 10 мс плавных краёв — без щелчков
_EPS = 1e-12


def sweep(seconds: float, amplitude: float = 0.8) -> np.ndarray:
    """Линейный свип SWEEP_LO_HZ → SWEEP_HI_HZ, 48 кГц mono float32."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    rate = (SWEEP_HI_HZ - SWEEP_LO_HZ) / seconds
    phase = 2 * np.pi * (SWEEP_LO_HZ * t + rate * t**2 / 2)
    wave = amplitude * np.sin(phase)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(_RAMP_SAMPLES) / _RAMP_SAMPLES))
    wave[:_RAMP_SAMPLES] *= ramp
    wave[-_RAMP_SAMPLES:] *= ramp[::-1]
    return wave.astype(np.float32)


def spectrum_db(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Спектр записи по Уэлчу: (частоты, дБ относительно пика)."""
    freqs, psd = welch(np.asarray(samples, dtype=np.float64), SAMPLE_RATE, nperseg=4_096)
    db = 10 * np.log10(psd + _EPS)
    return freqs, db - db.max()


def band_db(freqs: np.ndarray, db: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    """Медианный уровень полосы [lo_hz, hi_hz], дБ."""
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    return float(np.median(db[mask]))


def band_report(freqs: np.ndarray, db: np.ndarray) -> str:
    """Уровни полос тонов всех профилей — сверка границ с фактической АЧХ.

    Отсчёт — медиана всего диапазона свипа, а не пик записи: у ноутбучных
    динамиков низкочастотный резонанс на десятки дБ громче рабочих полос,
    и относительно пика любая полоса выглядела бы «провалом».
    """
    reference = band_db(freqs, db, SWEEP_LO_HZ, SWEEP_HI_HZ)
    lines = ["Полосы профилей (медиана PSD относительно медианы свипа):"]
    for name, profile in PROFILES.items():
        lo, hi = min(profile.tones_hz), max(profile.tones_hz)
        level = band_db(freqs, db, lo, hi) - reference
        lines.append(f"  {name:7s} {lo:7.0f}–{hi:5.0f} Гц: {level:+6.1f} дБ")
    return "\n".join(lines)


def _save_plot(freqs: np.ndarray, db: np.ndarray, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, db, lw=1)
    for name, profile in PROFILES.items():
        lo, hi = min(profile.tones_hz), max(profile.tones_hz)
        ax.axvspan(lo, hi, alpha=0.12, label=f"{name} {lo:.0f}–{hi:.0f} Гц")
    ax.set_xlim(0, 14_000)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("PSD, дБ отн. пика")
    ax.set_title("АЧХ тракта: свип динамик → микрофон")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f"Спектр: {png}")


def _analyze(recording: np.ndarray, wav: Path, png: Path) -> None:
    from scipy.io import wavfile

    wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav, SAMPLE_RATE, recording)
    print(f"Запись: {wav}")
    if float(np.abs(recording).max()) < 1e-4:
        print(
            "ВНИМАНИЕ: запись почти нулевая — микрофон молчит. Проверьте "
            "приватность микрофона Windows, «улучшения звука» и устройство "
            "(--in-device; MME-устройства бывают немыми, возьмите WASAPI)."
        )
    freqs, db = spectrum_db(recording)
    _save_plot(freqs, db, png)
    print(band_report(freqs, db))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="АЧХ-свип: проиграть, записать, показать спектр."
    )
    parser.add_argument("command", choices=("play", "record", "both"))
    parser.add_argument(
        "--seconds", type=float, default=6.0, help="длительность свипа (по умолчанию 6)"
    )
    parser.add_argument("--wav", default="out/afr.wav", help="куда сохранить запись")
    parser.add_argument("--png", default="out/afr.png", help="куда сохранить спектр")
    parser.add_argument("--in-device", default=None, help="устройство записи sounddevice")
    parser.add_argument("--out-device", default=None, help="устройство вывода sounddevice")
    args = parser.parse_args()

    import sounddevice as sd

    signal = sweep(args.seconds)
    # record слушает дольше свипа: запас на ручной запуск двух ноутбуков.
    record_samples = int((args.seconds + 4.0) * SAMPLE_RATE)

    if args.command == "play":
        print(f"Играю свип {SWEEP_LO_HZ:.0f}–{SWEEP_HI_HZ:.0f} Гц, {args.seconds:.0f} с…")
        sd.play(signal, SAMPLE_RATE, device=_device(args.out_device), blocking=True)
        return

    print(f"Пишу микрофон {record_samples / SAMPLE_RATE:.0f} с…")
    if args.command == "both":
        # Одновременно играть и писать можно только одним playrec:
        # сами по себе sd.rec и sd.play — «удобные» функции с общим
        # потоком, второй вызов молча убивает первый.
        padded = np.zeros(record_samples, dtype=np.float32)
        lead = SAMPLE_RATE  # секунда тишины перед свипом — уровень шума
        padded[lead : lead + len(signal)] = signal
        recording = sd.playrec(
            padded,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=(_device(args.in_device), _device(args.out_device)),
        )
    else:
        recording = sd.rec(
            record_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=_device(args.in_device),
        )
    sd.wait()
    _analyze(recording[:, 0], Path(args.wav), Path(args.png))


def _device(value: str | None) -> int | str | None:
    """Номер или подстрока имени устройства sounddevice."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
