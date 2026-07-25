"""Порча канала, общая для фейкового и спул-модема (тикеты 04–05).

Переворот случайных бит с заданной вероятностью на бит и потеря блока
целиком; длина блока при порче сохраняется (контракт v1.2). Плюс общая
проверка длин блоков на отправке.
"""

import numpy as np

from modem.profiles import MAX_BLOCK


def check_block_lengths(blocks: list[bytes], profile: str) -> None:
    limit = MAX_BLOCK[profile]
    for i, block in enumerate(blocks):
        if not 1 <= len(block) <= limit:
            raise ValueError(
                f"блок #{i}: длина {len(block)} вне диапазона 1..{limit}"
            )


def block_lost(rng: np.random.Generator, block_loss_prob: float) -> bool:
    return bool(block_loss_prob and rng.random() < block_loss_prob)


def corrupt_block(
    rng: np.random.Generator, block: bytes, bit_flip_prob: float
) -> bytes:
    if not bit_flip_prob:
        return block
    arr = np.frombuffer(block, dtype=np.uint8)
    flips = np.packbits(rng.random(arr.size * 8) < bit_flip_prob)
    return (arr ^ flips).tobytes()
