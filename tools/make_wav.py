"""Тестовая партия блоков → WAV + спектрограмма (сигнал ушами и глазами).

Запуск из корня репо:  python tools/make_wav.py [профиль]
Кладёт out/<профиль>_batch.wav и out/<профиль>_batch_spectrogram.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окон — только файл
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem import MAX_BLOCK, SAMPLE_RATE, modulate


def demo_batch(profile: str) -> list[bytes]:
    """Партия: текст, счётчик байтов на полный блок, короткий блок."""
    return [
        "Аудиомодем: привет, воздух!".encode("utf-8"),
        bytes(k % 256 for k in range(MAX_BLOCK[profile])),
        b"\xa7\x55",
    ]


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "robust"
    samples = modulate(demo_batch(profile), profile)
    out_dir = Path(__file__).resolve().parents[1] / "out"
    out_dir.mkdir(exist_ok=True)

    wav_path = out_dir / f"{profile}_batch.wav"
    wavfile.write(wav_path, SAMPLE_RATE, (samples * 32_767).astype(np.int16))

    freqs, times, power = spectrogram(
        samples, fs=SAMPLE_RATE, nperseg=512, noverlap=384
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    mesh = ax.pcolormesh(
        times, freqs / 1_000, 10 * np.log10(power + 1e-12), shading="gouraud"
    )
    ax.set_ylim(0, 12)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, кГц")
    ax.set_title(f"Профиль {profile}: chirp-преамбулы и MFSK-символы")
    fig.colorbar(mesh, ax=ax, label="дБ")
    png_path = out_dir / f"{profile}_batch_spectrogram.png"
    fig.savefig(png_path, dpi=110, bbox_inches="tight")

    duration = len(samples) / SAMPLE_RATE
    print(f"{wav_path}  ({duration:.2f} с)")
    print(png_path)


if __name__ == "__main__":
    main()
