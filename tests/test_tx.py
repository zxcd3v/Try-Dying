"""Тесты передатчика на шве «массив сэмплов» (Testing Decisions спеки).

Внутрь (нарезка на символы, chirp, пилоты) не заглядываем: полная петля
демодуляция(канал(модуляция(блоки))) появится в тикете 02. Здесь — свойства,
проверяемые по самому массиву: детерминизм, формат, диапазон длин, спектр.
"""

import numpy as np
import pytest

from modem.profiles import MAX_BLOCK, SAMPLE_RATE
from modem.tx import modulate

BLOCKS = [b"hello, air", bytes(range(96)), b"\x00" * 5]


def test_modulate_is_deterministic():
    a = modulate(BLOCKS, "robust")
    b = modulate(BLOCKS, "robust")
    assert np.array_equal(a, b)


def test_output_is_mono_float32_within_unit_range():
    samples = modulate(BLOCKS, "robust")
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.max(np.abs(samples)) <= 1.0
    assert np.max(np.abs(samples)) > 0.1  # сигнал не тишина


def test_accepts_any_block_length_in_contract_range():
    for length in (1, 2, MAX_BLOCK["robust"]):
        samples = modulate([b"\xa7" * length], "robust")
        assert len(samples) > 0


def test_rejects_blocks_outside_contract_range():
    with pytest.raises(ValueError):
        modulate([b""], "robust")
    with pytest.raises(ValueError):
        modulate([b"x" * (MAX_BLOCK["robust"] + 1)], "robust")


def test_rejects_unknown_profile():
    with pytest.raises(ValueError):
        modulate([b"data"], "turbo")


def test_rejects_empty_batch():
    with pytest.raises(ValueError):
        modulate([], "robust")


def test_each_block_gets_its_own_air_frame():
    # ADR-0002: аир-кадр самодостаточен — два блока звучат дольше одного.
    one = modulate([b"x" * 10], "robust")
    two = modulate([b"x" * 10, b"y" * 10], "robust")
    assert len(two) > len(one) * 1.8


def test_medium_modem_rate_at_least_5x_robust():
    # Тикет 08: скорость medium ≥ 5× robust. Закрепляем на уровне модема:
    # один и тот же файл, режем на блоки по MAX_BLOCK профиля, сравниваем
    # байты/сэмпл целых партий (преамбулы, заголовки и паузы учтены).
    payload = bytes(range(256)) * 15  # 3840 Б — кратно 96 и 128
    rates = {}
    for name in ("robust", "medium"):
        size = MAX_BLOCK[name]
        blocks = [payload[i : i + size] for i in range(0, len(payload), size)]
        rates[name] = len(payload) / len(modulate(blocks, name))
    assert rates["medium"] >= 5 * rates["robust"], rates


def test_fast_modem_rate_at_least_3x_medium():
    # Тикет 09: fast — параллельные поднесущие. Цель тикета «2–5 КБ/с»
    # физически недостижима MFSK в окне 1.5–9.5 кГц (см. Comments тикета);
    # закрепляем достигнутое: ≥3× medium на уровне модема.
    payload = bytes(range(256)) * 15  # 3840 Б
    rates = {}
    for name in ("medium", "fast"):
        size = MAX_BLOCK[name]
        blocks = [payload[i : i + size] for i in range(0, len(payload), size)]
        rates[name] = len(payload) / len(modulate(blocks, name))
    assert rates["fast"] >= 3 * rates["medium"], rates


def test_fast_chord_of_all_top_tones_stays_within_unit_range():
    # Аккорд — сумма синусов всех поднесущих: пик обязан остаться ≤1.0
    # даже когда все поднесущие сложились в фазе.
    samples = modulate([b"\xff" * 255, b"\x00" * 255], "fast")
    assert np.max(np.abs(samples)) <= 1.0


def test_spectrum_respects_hardware_ceiling():
    # Handoff §4: аппаратный потолок приёма — 10 кГц. Практически вся
    # энергия сигнала обязана лежать ниже.
    for profile in ("robust", "medium", "fast"):
        samples = modulate(BLOCKS, profile)
        spectrum = np.abs(np.fft.rfft(samples.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(samples), d=1 / SAMPLE_RATE)
        energy_above = spectrum[freqs > 10_000].sum()
        assert energy_above < 0.01 * spectrum.sum(), profile
