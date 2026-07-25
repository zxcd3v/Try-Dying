"""Передатчик: список блоков байтов → массив сэмплов (чистые функции).

Аир-кадр (ADR-0002): chirp-преамбула → тишина → аир-заголовок ×3 →
символы данных с пилотами → межкадровая тишина. Каждый кадр самодостаточен.
Звуковое железо здесь не трогаем — sounddevice появится тонкой обёрткой позже.
"""

import numpy as np

from modem.framing import build_air_header
from modem.profiles import PROFILES, SAMPLE_RATE, Profile, banded


def modulate(
    blocks: list[bytes], profile_name: str, band: int | None = None
) -> np.ndarray:
    """Превращает партию блоков в сигнал 48 кГц mono float32.

    Детерминирована: одинаковый вход даёт одинаковые сэмплы. band —
    полоса профиля (ADR-0003); None — полоса по умолчанию.
    """
    if profile_name not in PROFILES:
        raise ValueError(f"неизвестный профиль {profile_name!r}")
    if not blocks:
        raise ValueError("партия блоков пуста")
    profile = banded(profile_name, band)
    for i, block in enumerate(blocks):
        if not 1 <= len(block) <= profile.max_block:
            raise ValueError(
                f"блок #{i}: длина {len(block)} вне диапазона 1..{profile.max_block}"
            )
    return np.concatenate(
        [_air_frame(block, profile) for block in blocks]
    ).astype(np.float32)


def _air_frame(block: bytes, profile: Profile) -> np.ndarray:
    header = build_air_header(len(block), repeats=profile.header_repeats)
    parts = [
        chirp(profile),
        np.zeros(profile.preamble_gap_samples),
        _symbols(_bytes_to_symbols(header, profile), profile, pilots=False),
        _symbols(_bytes_to_symbols(block, profile), profile, pilots=True),
        np.zeros(profile.frame_gap_samples),
    ]
    return np.concatenate(parts)


def _bytes_to_symbols(data: bytes, profile: Profile) -> list[int]:
    """Байты → значения символов, старшие биты вперёд.

    Бит на символ может не делить 8 (аккорды fast): байты складываются в
    один битовый поток, хвост последнего символа добивается нулями.
    """
    bits = profile.bits_per_symbol
    n_symbols = profile.symbols_for_bytes(len(data))
    stream = int.from_bytes(data, "big") << (n_symbols * bits - 8 * len(data))
    mask = (1 << bits) - 1
    return [(stream >> (bits * (n_symbols - 1 - k))) & mask for k in range(n_symbols)]


def _symbols(values: list[int], profile: Profile, *, pilots: bool) -> np.ndarray:
    """Символы + защитные интервалы; в данных — пилоты каждые pilot_every."""
    guard = np.zeros(profile.guard_samples)
    out = []
    for i, value in enumerate(values):
        if pilots and i > 0 and i % profile.pilot_every == 0:
            for pilot in profile.pilot_pattern:
                out += [_tone(pilot, profile), guard]
        out += [_tone(value, profile), guard]
    return np.concatenate(out)


def _tone(value: int, profile: Profile) -> np.ndarray:
    """Аккорд символа: по синусоиде на поднесущую (одной — вне fast).

    Амплитуда делится на число поднесущих: даже сложившись в фазе,
    аккорд не выходит за пиковую амплитуду профиля.
    """
    t = np.arange(profile.symbol_samples) / SAMPLE_RATE
    wave = sum(np.sin(2 * np.pi * tone * t) for tone in profile.chord_tones(value))
    amplitude = profile.amplitude / profile.subcarriers
    return amplitude * _ramped(wave, profile.ramp_samples)


def chirp(profile: Profile) -> np.ndarray:
    """Линейный частотный свип — преамбула; rx использует его же как
    шаблон корреляционного поиска."""
    t = np.arange(profile.preamble_samples) / SAMPLE_RATE
    duration = profile.preamble_samples / SAMPLE_RATE
    sweep_rate = (profile.chirp_hi_hz - profile.chirp_lo_hz) / duration
    phase = 2 * np.pi * (profile.chirp_lo_hz * t + sweep_rate * t**2 / 2)
    return profile.amplitude * _ramped(np.sin(phase), profile.ramp_samples)


def _ramped(wave: np.ndarray, ramp_samples: int) -> np.ndarray:
    """Плавные края (полупериод косинуса) — без щелчков на стыках."""
    if ramp_samples == 0:
        return wave
    envelope = np.ones(len(wave))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
    envelope[:ramp_samples] = ramp
    envelope[-ramp_samples:] = ramp[::-1]
    return wave * envelope
