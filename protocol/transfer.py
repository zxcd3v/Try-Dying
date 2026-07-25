"""Конвейер передачи файла: zlib → кадры → блоки → NACK-цикл → SHA-256-вердикт.

Отправитель шлёт партию (HEADER ×3 + DATA), приёмник после партии отвечает
одним ответом (NACK со списком недостающих либо DONE), отправитель повторяет
только битые — цикл до DONE (ADR-0002).
"""

import hashlib
import zlib
from dataclasses import dataclass

from protocol import frames

MODES = ("robust", "medium", "fast")

_HEADER_REPEATS = 3
_HEADER_FIXED = 32 + 4 + 1  # sha256 + размер файла (4 Б) + режим (1 Б) + имя
_NACK_HEADER = 0xFFFF  # сентинел в NACK: «HEADER не принят, повтори его»


@dataclass(frozen=True)
class TransferResult:
    ok: bool
    name: str
    data: bytes
    rounds: int
    retransmitted: int  # DATA-кадров послано повторно за все NACK-циклы


class FileSender:
    def __init__(self, data: bytes, name: str, mode: str):
        self.mode = mode
        self.done = False
        self.frames_sent = 0  # DATA-кадров ушло всего, включая повторы
        self.retransmitted = 0
        self._first_batch = True
        self._nacked: set[int] = set()
        self._resend_header = False

        compressed = zlib.compress(data, level=9)
        step = frames.payload_capacity(mode)
        chunks = [compressed[i : i + step] for i in range(0, len(compressed), step)]
        total = len(chunks)
        self._data_frames = {
            seq: frames.build_frame(frames.DATA, seq, total, chunk)
            for seq, chunk in enumerate(chunks)
        }

        header_payload = (
            hashlib.sha256(data).digest()
            + len(data).to_bytes(4, "big")
            + bytes([MODES.index(mode)])
            + name.encode("utf-8")
        )
        if len(header_payload) > frames.payload_capacity("robust"):
            raise ValueError(f"имя файла {name!r} не влезает в HEADER-кадр")
        self._header_frame = frames.build_frame(frames.HEADER, 0, total, header_payload)

    @property
    def frames_total(self) -> int:
        return len(self._data_frames)

    @property
    def pending(self) -> int:
        """DATA-кадров, затребованных последним NACK и ещё не отправленных."""
        return len(self._nacked)

    def outgoing(self) -> list[tuple[str, list[bytes]]]:
        """Партия на отправку: список (профиль, блоки).

        Первая партия — HEADER ×3 + все DATA; дальше — только затребованное
        NACK'ом. Пустой запрос (ответ приёмника погиб в канале) → пингуем
        HEADER'ом, чтобы приёмник ответил ещё раз.
        """
        seqs: list[int] = []
        with_header = self._resend_header
        if self._first_batch:
            self._first_batch = False
            with_header = True
            seqs = list(self._data_frames)
        elif self._nacked:
            seqs = sorted(self._nacked)
            self.retransmitted += len(seqs)
        else:
            with_header = True  # пинг
        self._nacked.clear()
        self._resend_header = False
        self.frames_sent += len(seqs)

        batches = []
        if with_header:  # служебные кадры — всегда robust (CONTEXT.md)
            header_block = frames.encode_block(self._header_frame, "robust")
            batches.append(("robust", [header_block] * _HEADER_REPEATS))
        if seqs:
            batches.append(
                (
                    self.mode,
                    [
                        frames.encode_block(self._data_frames[seq], self.mode)
                        for seq in seqs
                    ],
                )
            )
        return batches

    def feed(self, block: bytes) -> None:
        parsed = frames.decode_frame(block)
        if parsed is None:
            return
        frame_type, _, _, payload = parsed
        if frame_type == frames.DONE:
            self.done = True
        elif frame_type == frames.NACK:
            for i in range(0, len(payload) - 1, 2):
                seq = int.from_bytes(payload[i : i + 2], "big")
                if seq == _NACK_HEADER:
                    self._resend_header = True
                elif seq in self._data_frames:
                    self._nacked.add(seq)


class FileReceiver:
    def __init__(self):
        self.ok = False
        self.name = ""
        self.data = b""
        self._sha256 = b""
        self._size = 0
        self._total: int | None = None
        self._chunks: dict[int, bytes] = {}

    @property
    def frames_total(self) -> int | None:
        """Сколько DATA-кадров всего; None, пока HEADER не принят."""
        return self._total

    @property
    def frames_received(self) -> int:
        return len(self._chunks)

    @property
    def bytes_received(self) -> int:
        return sum(len(c) for c in self._chunks.values())

    def feed(self, block: bytes) -> None:
        parsed = frames.decode_frame(block)
        if parsed is None:
            return
        frame_type, seq, total, payload = parsed
        if frame_type == frames.HEADER and self._total is None:
            self._total = total
            self._sha256 = payload[:32]
            self._size = int.from_bytes(payload[32:36], "big")
            self.name = payload[_HEADER_FIXED:].decode("utf-8")
        elif frame_type == frames.DATA:
            self._chunks[seq] = payload

    def response(self) -> list[tuple[str, list[bytes]]]:
        """Ответ на принятую партию: DONE либо NACK с недостающими."""
        if self._total is not None and len(self._chunks) == self._total:
            if not self.ok:
                data = zlib.decompress(
                    b"".join(self._chunks[i] for i in range(self._total))
                )
                if (
                    hashlib.sha256(data).digest() != self._sha256
                    or len(data) != self._size
                ):
                    raise ValueError(
                        "SHA-256 не совпал: все кадры собраны, "
                        "но контрольная сумма файла другая"
                    )
                self.data = data
                self.ok = True
            done = frames.build_frame(frames.DONE, 0, 0, b"")
            return [("robust", [frames.encode_block(done, "robust")])]

        if self._total is None:
            missing = [_NACK_HEADER]  # без HEADER не знаем даже total
        else:
            missing = [s for s in range(self._total) if s not in self._chunks]
        step = frames.payload_capacity("robust") // 2
        nacks = []
        for i in range(0, len(missing), step):
            payload = b"".join(s.to_bytes(2, "big") for s in missing[i : i + step])
            nacks.append(frames.build_frame(frames.NACK, 0, 0, payload))
        return [("robust", [frames.encode_block(n, "robust") for n in nacks])]


def transfer(
    data: bytes, name: str, mode: str, link, max_rounds: int = 50
) -> TransferResult:
    """Гоняет полный обмен через фейковый модем до DONE (шов для тестов)."""
    sender = FileSender(data, name, mode)
    receiver = FileReceiver()
    rounds = 0
    while not sender.done:
        if rounds == max_rounds:
            raise RuntimeError(f"передача не завершилась за {max_rounds} партий")
        rounds += 1
        for profile, blocks in sender.outgoing():
            link.a.send_blocks(blocks, profile)
        for block in link.b.receive_blocks(mode, timeout_s=0.0):
            receiver.feed(block)
        for profile, blocks in receiver.response():
            link.b.send_blocks(blocks, profile)
        for block in link.a.receive_blocks("robust", timeout_s=0.0):
            sender.feed(block)
    return TransferResult(
        receiver.ok, receiver.name, receiver.data, rounds, sender.retransmitted
    )
