"""Аир-заголовок: длина блока (2 Б, big-endian) + CRC-8, повтор ×N (ADR-0002).

Заголовок передаётся надёжнее данных: приёмник собирает копии и голосует
большинством; если заголовок не сошёлся — весь аир-кадр отбрасывается.
"""

_CRC8_POLY = 0x07  # CRC-8/SMBUS: x^8 + x^2 + x + 1

HEADER_COPY_BYTES = 3  # одна копия аир-заголовка: длина (2 Б) + CRC-8 (1 Б)


def crc8(data: bytes) -> int:
    """CRC-8, poly 0x07, init 0x00, без отражения и финального XOR."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC8_POLY if crc & 0x80 else crc << 1) & 0xFF
    return crc


def build_air_header(block_len: int, repeats: int) -> bytes:
    """Собирает аир-заголовок: (len_hi, len_lo, crc8) × repeats."""
    if not 1 <= block_len <= 0xFFFF:
        raise ValueError(f"длина блока {block_len} вне диапазона 1..65535")
    body = block_len.to_bytes(2, "big")
    copy = body + bytes([crc8(body)])
    assert len(copy) == HEADER_COPY_BYTES
    return copy * repeats
