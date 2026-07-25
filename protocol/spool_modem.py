"""Спул-модем: контракт v1.2 между двумя процессами через общий каталог.

Межпроцессный собрат FakeLink для демо CLI на одной машине (тикет 05):
каждый конец пишет блоки нумерованными файлами `<сторона>-NNNNNN.blk` и
читает файлы противоположной стороны. Файл пишется атомарно (tmp → replace),
прочитанный — удаляется, поэтому блок потребляется ровно один раз.

Порча канала — как у FakeLink: переворот случайных бит и потеря блоков
целиком, применяется на отправке. Партия на приёме заканчивается по тишине:
после последнего нового блока прошло `quiet_gap_s` либо истёк `timeout_s`.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from protocol.corruption import block_lost, check_block_lengths, corrupt_block

_OTHER = {"a": "b", "b": "a"}


class SpoolEndpoint:
    # Блоки ходят файлами, спектра в канале нет — калибровать нечего
    # (ADR-0003 про полосы живого звука).
    supports_calibration = False

    def __init__(
        self,
        link_dir: str | Path,
        side: str,
        *,
        bit_flip_prob: float = 0.0,
        block_loss_prob: float = 0.0,
        seed: int = 0,
        quiet_gap_s: float = 0.25,
        poll_s: float = 0.02,
    ):
        if side not in _OTHER:
            raise ValueError(f"сторона линка должна быть 'a' или 'b', не {side!r}")
        self._dir = Path(link_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._side = side
        self._sent = 0
        self.bit_flip_prob = bit_flip_prob
        self.block_loss_prob = block_loss_prob
        self.rng = np.random.default_rng(seed)
        self.quiet_gap_s = quiet_gap_s
        self.poll_s = poll_s
        for stale in self._incoming():  # хвосты прошлых запусков
            stale.unlink(missing_ok=True)

    def _incoming(self) -> list[Path]:
        return sorted(self._dir.glob(f"{_OTHER[self._side]}-*.blk"))

    def send_blocks(self, blocks: list[bytes], profile: str) -> None:
        check_block_lengths(blocks, profile)
        for block in blocks:
            name = f"{self._side}-{self._sent:06d}.blk"
            self._sent += 1
            if block_lost(self.rng, self.block_loss_prob):
                continue
            tmp = self._dir / (name + ".tmp")
            tmp.write_bytes(corrupt_block(self.rng, block, self.bit_flip_prob))
            os.replace(tmp, self._dir / name)

    def receive_blocks(self, profile: str, timeout_s: float) -> Iterator[bytes]:
        deadline = time.monotonic() + timeout_s
        last_block_at: float | None = None
        while True:
            got = self._incoming()
            for path in got:
                data = path.read_bytes()
                path.unlink()
                last_block_at = time.monotonic()
                yield data
            now = time.monotonic()
            if last_block_at is not None and now - last_block_at >= self.quiet_gap_s:
                return  # партия закончилась: тишина после последнего блока
            if now >= deadline:
                return
            time.sleep(self.poll_s)
