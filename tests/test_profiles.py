"""Тесты контракта модуля профилей (шов: публичные константы сигнала).

Ожидаемые значения взяты из независимых источников истины:
контракт v1.2 (MAX_BLOCK), ADR-0001 (тона строго в FFT-бины),
handoff §4 (аппаратный потолок 10 кГц, пол ~1.5 кГц, 48 кГц mono).
"""

import math

from modem.profiles import MAX_BLOCK, PROFILES, SAMPLE_RATE


def test_sample_rate_is_48k():
    assert SAMPLE_RATE == 48_000


def test_max_block_matches_contract_v12():
    assert MAX_BLOCK == {"robust": 96, "medium": 128, "fast": 255}


def test_profiles_has_exactly_three_profiles():
    assert set(PROFILES) == {"robust", "medium", "fast"}


def test_every_tone_lands_exactly_on_fft_bin():
    # ADR-0001: тон обязан попадать ровно в FFT-бин окна символа,
    # иначе энергия размазывается по соседним бинам.
    for name, profile in PROFILES.items():
        bin_hz = SAMPLE_RATE / profile.symbol_samples
        for tone in profile.tones_hz:
            assert tone % bin_hz == 0, (
                f"{name}: тон {tone} Гц не кратен ширине бина {bin_hz} Гц"
            )


def test_tones_split_evenly_into_power_of_two_carrier_grids():
    # Сетка делится поровну между поднесущими; тонов на поднесущую —
    # степень двойки: целое число бит на поднесущую.
    for name, profile in PROFILES.items():
        assert len(profile.tones_hz) % profile.subcarriers == 0, name
        per_carrier = len(profile.tones_hz) // profile.subcarriers
        assert per_carrier >= 2 and math.log2(per_carrier).is_integer(), name


def test_single_carrier_profiles_stay_single_fast_is_parallel():
    # Тикет 09: fast — параллельные MFSK-поднесущие («OFDM-лайт»);
    # robust и medium остаются одиночным тоном.
    assert PROFILES["robust"].subcarriers == 1
    assert PROFILES["medium"].subcarriers == 1
    assert PROFILES["fast"].subcarriers >= 4


def test_bits_per_symbol_counts_all_subcarriers():
    for name, profile in PROFILES.items():
        per_carrier = len(profile.tones_hz) // profile.subcarriers
        expected = profile.subcarriers * (per_carrier - 1).bit_length()
        assert profile.bits_per_symbol == expected, name


def test_pilot_values_fit_symbol_bit_width():
    # Пилот — обычное значение символа: обязан помещаться в его биты,
    # иначе смена числа поднесущих молча сломает пилоты профиля.
    for name, profile in PROFILES.items():
        for pilot in profile.pilot_pattern:
            assert 0 <= pilot < 2**profile.bits_per_symbol, name


def test_robust_is_4fsk():
    assert len(PROFILES["robust"].tones_hz) == 4


def test_tones_inside_hardware_window():
    # Handoff §4: ниже ~1.5 кГц динамики не звучат, выше 9.5 кГц режет тракт.
    for name, profile in PROFILES.items():
        for tone in profile.tones_hz:
            assert 1_500 <= tone <= 9_500, f"{name}: тон {tone} Гц вне окна"
        assert profile.chirp_lo_hz >= 1_000
        assert profile.chirp_hi_hz <= 9_500


def test_guard_and_symbol_durations_positive():
    for name, profile in PROFILES.items():
        assert profile.symbol_samples > 0, name
        assert profile.guard_samples >= 0, name
        assert profile.preamble_samples > 0, name
        assert 0 < profile.amplitude <= 1.0, name
