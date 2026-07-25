"""Фейковый модем: контракт v1.2 без звука, для отладки протокола (тикет 04).

Два конца — `link.a` (отправитель) и `link.b` (приёмник); блоки, отданные в
send_blocks одного конца, приходят из receive_blocks другого. Журнал `log`
хранит (конец, профиль, число блоков) каждого send — тесты сверяют по нему,
что служебные кадры идут в robust.

Порча канала: переворот случайных бит с вероятностью `bit_flip_prob` на бит,
потеря блока целиком с вероятностью `block_loss_prob` (аналог погибшего
аир-заголовка: модем такой кадр не отдаёт вовсе) либо прицельно по номеру в
эфире через `drop_indices` (сквозная нумерация обоих направлений). Один seed
на весь линк — прогон детерминирован.
"""

from collections import deque
from collections.abc import Iterator

import numpy as np

from protocol.corruption import block_lost, check_block_lengths, corrupt_block


class FakeLink:
    def __init__(
        self,
        *,
        bit_flip_prob: float = 0.0,
        block_loss_prob: float = 0.0,
        drop_indices: set[int] = frozenset(),
        seed: int = 0,
    ):
        self.bit_flip_prob = bit_flip_prob
        self.block_loss_prob = block_loss_prob
        self.drop_indices = drop_indices
        self._sent_count = 0
        self.rng = np.random.default_rng(seed)
        self.log: list[tuple[str, str, int]] = []
        a_inbox: deque[bytes] = deque()
        b_inbox: deque[bytes] = deque()
        self.a = _Endpoint("a", self, outbox=b_inbox, inbox=a_inbox)
        self.b = _Endpoint("b", self, outbox=a_inbox, inbox=b_inbox)

    def _lost(self) -> bool:
        index = self._sent_count
        self._sent_count += 1
        if index in self.drop_indices:
            return True
        return block_lost(self.rng, self.block_loss_prob)

    def _corrupt(self, block: bytes) -> bytes:
        return corrupt_block(self.rng, block, self.bit_flip_prob)


class _Endpoint:
    def __init__(self, name: str, link: FakeLink, outbox: deque, inbox: deque):
        self._name = name
        self._link = link
        self._outbox = outbox
        self._inbox = inbox

    def send_blocks(self, blocks: list[bytes], profile: str) -> None:
        check_block_lengths(blocks, profile)
        self._link.log.append((self._name, profile, len(blocks)))
        self._outbox.extend(
            self._link._corrupt(block) for block in blocks if not self._link._lost()
        )

    def receive_blocks(self, profile: str, timeout_s: float) -> Iterator[bytes]:
        """Фейк мгновенен: отдаёт всё накопленное, конец = таймаут партии."""
        while self._inbox:
            yield self._inbox.popleft()
