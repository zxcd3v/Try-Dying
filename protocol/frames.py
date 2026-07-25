"""Протокольный кадр: `magic 0xA7 0x55 | type | seq | total | len | payload | CRC32`.

Для модема кадр — просто содержимое блока (CONTEXT.md); битовые ошибки внутри
блока допустимы: их чинит Reed-Solomon, а окончательную целостность кадра
подтверждает CRC32. Кадр, не прошедший RS+CRC, отбрасывается целиком — его
добирает NACK-цикл.
"""

import struct
import zlib

from reedsolo import RSCodec, ReedSolomonError

from modem.profiles import MAX_BLOCK

MAGIC = b"\xa7\x55"

# Паритетные байты RS на блок; чинится до половины этого числа байтов.
# Доля от кадра: robust 32/64 = 50%, medium 26/102 ≈ 25%, fast 32/223 ≈ 14%.
RS_PARITY = {"robust": 32, "medium": 26, "fast": 32}

# Приёмник не знает профиль пришедшего блока (данные идут в режиме файла,
# служебные — в robust): пробуем каждый вариант паритета, CRC32 отсекает
# неверно раскодированные.
_CODECS = {parity: RSCodec(parity) for parity in set(RS_PARITY.values())}
_DECODE_PARITIES = sorted(_CODECS, reverse=True)

HEADER = 0x01
DATA = 0x02
NACK = 0x03
DONE = 0x04
BAND = 0x05  # полоса калибровки: payload = 1 байт номера полосы (ADR-0003)

_HEAD = struct.Struct(">2sBHHH")  # magic | type | seq | total | len
FRAME_OVERHEAD = _HEAD.size + 4  # + CRC32


def build_frame(frame_type: int, seq: int, total: int, payload: bytes) -> bytes:
    body = _HEAD.pack(MAGIC, frame_type, seq, total, len(payload)) + payload
    return body + zlib.crc32(body).to_bytes(4, "big")


def parse_frame(raw: bytes) -> tuple[int, int, int, bytes] | None:
    """(type, seq, total, payload) либо None — кадр битый, не наш или обрезан."""
    if len(raw) < FRAME_OVERHEAD:
        return None
    magic, frame_type, seq, total, payload_len = _HEAD.unpack_from(raw)
    end = _HEAD.size + payload_len
    if magic != MAGIC or len(raw) < end + 4:
        return None
    body = raw[:end]
    if zlib.crc32(body) != int.from_bytes(raw[end : end + 4], "big"):
        return None
    return frame_type, seq, total, body[_HEAD.size : end]


def payload_capacity(mode: str) -> int:
    """Максимум байтов payload в одном кадре профиля."""
    return MAX_BLOCK[mode] - RS_PARITY[mode] - FRAME_OVERHEAD


def encode_block(frame: bytes, mode: str) -> bytes:
    """Кадр → RS-закодированный блок модему."""
    block = bytes(_CODECS[RS_PARITY[mode]].encode(frame))
    assert len(block) <= MAX_BLOCK[mode]
    return block


def decode_frame(block: bytes) -> tuple[int, int, int, bytes] | None:
    """Блок от модема → починенный и разобранный кадр либо None."""
    for parity in _DECODE_PARITIES:
        if len(block) <= parity:
            continue
        try:
            frame = bytes(_CODECS[parity].decode(block)[0])
        except ReedSolomonError:
            continue
        parsed = parse_frame(frame)
        if parsed is not None:
            return parsed
    return None
