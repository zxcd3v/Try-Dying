"""Тесты аир-заголовка (ADR-0002: длина 2 Б + CRC-8, повтор ×3).

Осознанное отступление от правила «только внешние швы»: до появления RX
(тикет 02) формат заголовка больше нечем закрепить. Когда петля
демодуляция(канал(модуляция(блоки))) заработает, эти тесты можно свернуть.
"""

import pytest

from modem.framing import build_air_header, crc8


def test_crc8_known_check_value():
    # Стандартное проверочное значение CRC-8 (poly 0x07, init 0x00)
    # для строки "123456789" — 0xF4. Источник: каталог CRC RevEng.
    assert crc8(b"123456789") == 0xF4


def test_crc8_empty_is_zero():
    assert crc8(b"") == 0x00


def test_air_header_is_three_identical_copies():
    header = build_air_header(96, repeats=3)
    assert len(header) == 9
    assert header[0:3] == header[3:6] == header[6:9]


def test_air_header_encodes_length_big_endian_with_crc():
    header = build_air_header(0x0160, repeats=1)
    assert header[0] == 0x01
    assert header[1] == 0x60
    assert header[2] == crc8(bytes([0x01, 0x60]))


def test_air_header_rejects_bad_length():
    with pytest.raises(ValueError):
        build_air_header(0, repeats=3)
    with pytest.raises(ValueError):
        build_air_header(65_536, repeats=3)
