"""Аудио-эндпоинт — контракт v1.2 через настоящий звук (тикет 06).

Тесты дёргают только внешний шов (send_blocks / receive_blocks); железо
подменено фейками: «динамик» копит сэмплы, «микрофон» отдаёт заранее
заданные куски. Петля тестов — честный звук: send_blocks одного конца
модулирует сигнал, receive_blocks другого разбирает его из кусков.
"""

from collections import deque

import numpy as np
import pytest

from modem import SAMPLE_RATE, modulate
from modem.audio import PLAY_PAD_SAMPLES, AudioEndpoint


class FakeSource:
    """Фейковый микрофон: очередь заранее заданных кусков сигнала."""

    def __init__(self, chunks=()):
        self._chunks = deque()
        self.push(chunks)

    def push(self, chunks) -> None:
        self._chunks.extend(np.asarray(c, dtype=np.float32) for c in chunks)

    def start(self) -> None:
        pass

    def get(self, wait_s: float):
        return self._chunks.popleft() if self._chunks else None

    @property
    def exhausted(self) -> bool:
        return not self._chunks

    def drain(self) -> None:
        self._chunks.clear()

    @property
    def remaining_samples(self) -> int:
        return sum(len(c) for c in self._chunks)


def chunked(samples, size=3_331):
    """Нарезает сигнал кусками «неудобного» размера — как отдаёт железо."""
    samples = np.asarray(samples, dtype=np.float32)
    return [samples[i : i + size] for i in range(0, len(samples), size)]


def transmitted(blocks, profile, noise_amp=0.01, seed=0, band=None):
    """Что «слышит» микрофон: сигнал send_blocks + лёгкий шум."""
    played = []
    tx = AudioEndpoint(player=played.append, source=FakeSource())
    if band is not None:
        tx.set_band(profile, band)
    tx.send_blocks(blocks, profile)
    rng = np.random.default_rng(seed)
    return played[0] + noise_amp * rng.standard_normal(len(played[0]))


def test_send_blocks_plays_padded_modulated_batch():
    played = []
    endpoint = AudioEndpoint(player=played.append, source=FakeSource())
    blocks = [b"nuclear it hack", bytes(range(64))]

    endpoint.send_blocks(blocks, "robust")

    (samples,) = played
    expected = modulate(blocks, "robust")
    assert samples.dtype == np.float32
    assert len(samples) == len(expected) + 2 * PLAY_PAD_SAMPLES
    pad = np.zeros(PLAY_PAD_SAMPLES, dtype=np.float32)
    assert np.array_equal(samples[:PLAY_PAD_SAMPLES], pad)
    assert np.array_equal(samples[-PLAY_PAD_SAMPLES:], pad)
    assert np.array_equal(samples[PLAY_PAD_SAMPLES:-PLAY_PAD_SAMPLES], expected)


def test_loopback_blocks_cross_the_air():
    blocks = [b"nuclear it hack", bytes(range(96)), b"\x00\xff" * 12]
    air = transmitted(blocks, "robust")
    stream = np.concatenate([np.zeros(24_000), air, np.zeros(24_000)])
    rx = AudioEndpoint(source=FakeSource(chunked(stream)))

    got = list(rx.receive_blocks("robust", timeout_s=60.0))

    assert got == blocks


def test_batch_ends_by_quiet_gap_not_by_stream_end():
    air = transmitted([b"lone frame"], "robust")
    stream = np.concatenate([np.zeros(24_000), air, np.zeros(30 * SAMPLE_RATE)])
    source = FakeSource(chunked(stream))
    rx = AudioEndpoint(source=source)

    got = list(rx.receive_blocks("robust", timeout_s=1000.0))

    assert got == [b"lone frame"]
    # Вернулись по тишине после кадра, а не дожевав весь хвост до конца.
    assert source.remaining_samples > 20 * SAMPLE_RATE


def test_receive_times_out_on_silence():
    source = FakeSource(chunked(np.zeros(30 * SAMPLE_RATE)))
    rx = AudioEndpoint(source=source)

    got = list(rx.receive_blocks("robust", timeout_s=2.0))

    assert got == []
    # Вернулись по таймауту аудио-времени, тишина в запасе осталась.
    assert source.remaining_samples > 20 * SAMPLE_RATE


def test_send_blocks_drops_own_echo_from_mic():
    source = FakeSource()
    endpoint = AudioEndpoint(player=lambda samples: None, source=source)
    # Пока мы играли партию, микрофон записал наш же звук.
    source.push(chunked(transmitted([b"own echo"], "robust")))

    endpoint.send_blocks([b"own echo"], "robust")
    got = list(endpoint.receive_blocks("robust", timeout_s=1.0))

    assert got == []


def test_receive_in_data_profile_still_hears_anchor_robust():
    """Служебные кадры всегда летят в robust (якорная полоса): приём
    в medium обязан разобрать и их."""
    air = transmitted([b"service in robust"], "robust")
    stream = np.concatenate([np.zeros(24_000), air, np.zeros(3 * SAMPLE_RATE)])
    rx = AudioEndpoint(source=FakeSource(chunked(stream)))

    got = list(rx.receive_blocks("medium", timeout_s=60.0))

    assert got == [b"service in robust"]


class DeadSource(FakeSource):
    """Микрофон, который «жив», но сэмплов не отдаёт (сломанное железо)."""

    @property
    def exhausted(self) -> bool:
        return False


def test_dead_microphone_raises_instead_of_hanging():
    rx = AudioEndpoint(source=DeadSource(), stall_s=0.2)
    with pytest.raises(RuntimeError, match="микрофон"):
        list(rx.receive_blocks("robust", timeout_s=60.0))


def test_full_transfer_over_air_with_nack_cycle():
    """NACK-цикл через звук в памяти: один DATA-кадр глушится «рукой на
    микрофоне», повтор добирает битое, SHA-256 сходится."""
    from protocol.transfer import FileReceiver, FileSender

    src_a, src_b = FakeSource(), FakeSource()
    sabotage = {"calls": 0}

    def a_speaker_to_b_mic(samples: np.ndarray) -> None:
        sabotage["calls"] += 1
        if sabotage["calls"] == 2:  # второй send_blocks = партия DATA
            # Глушим кусок длиннее аир-кадра: RS лечит короткие дырки,
            # а потерю целого кадра добирает только NACK-цикл.
            samples = samples.copy()
            middle = len(samples) // 2
            samples[middle : middle + 300_000] = 0.0
        src_b.push(chunked(samples))

    ep_a = AudioEndpoint(player=a_speaker_to_b_mic, source=src_a)
    ep_b = AudioEndpoint(player=lambda s: src_a.push(chunked(s)), source=src_b)

    data = bytes(np.random.default_rng(6).integers(0, 256, 1_024, dtype=np.uint8))
    sender = FileSender(data, "demo.bin", "robust")
    receiver = FileReceiver()
    for _ in range(5):
        if sender.done:
            break
        for profile, blocks in sender.outgoing():
            ep_a.send_blocks(blocks, profile)
        for block in ep_b.receive_blocks("robust", timeout_s=10.0):
            receiver.feed(block)
        for profile, blocks in receiver.response():
            ep_b.send_blocks(blocks, profile)
        for block in ep_a.receive_blocks("robust", timeout_s=10.0):
            sender.feed(block)

    assert sender.done
    assert receiver.ok
    assert receiver.data == data
    assert sender.retransmitted >= 1


def test_full_transfer_over_air_in_medium_data_frames():
    """Тикет 08: файл через звук в памяти в medium — DATA-кадры летят в
    medium, служебные (HEADER/NACK/DONE) остаются в якорном robust."""
    from protocol.transfer import FileReceiver, FileSender

    src_a, src_b = FakeSource(), FakeSource()
    ep_a = AudioEndpoint(player=lambda s: src_b.push(chunked(s)), source=src_a)
    ep_b = AudioEndpoint(player=lambda s: src_a.push(chunked(s)), source=src_b)

    data = bytes(np.random.default_rng(8).integers(0, 256, 2_048, dtype=np.uint8))
    sender = FileSender(data, "demo.bin", "medium")
    receiver = FileReceiver()
    outgoing_profiles, response_profiles = set(), set()
    for _ in range(5):
        if sender.done:
            break
        for profile, blocks in sender.outgoing():
            outgoing_profiles.add(profile)
            ep_a.send_blocks(blocks, profile)
        for block in ep_b.receive_blocks("medium", timeout_s=10.0):
            receiver.feed(block)
        for profile, blocks in receiver.response():
            response_profiles.add(profile)
            ep_b.send_blocks(blocks, profile)
        for block in ep_a.receive_blocks("robust", timeout_s=10.0):
            sender.feed(block)

    assert sender.done
    assert receiver.ok
    assert receiver.data == data
    # Разделение профилей закреплено явно: приёмник слушает якорь всегда,
    # поэтому сам успех передачи его не доказывает.
    assert outgoing_profiles == {"robust", "medium"}  # HEADER — robust, DATA — medium
    assert response_profiles == {"robust"}            # NACK/DONE — только robust


def test_full_transfer_over_air_in_fast_data_frames():
    """Тикет 09: файл через звук в памяти в fast — DATA-кадры летят
    аккордами поднесущих, служебные остаются в якорном robust."""
    from protocol.transfer import FileReceiver, FileSender

    src_a, src_b = FakeSource(), FakeSource()
    ep_a = AudioEndpoint(player=lambda s: src_b.push(chunked(s)), source=src_a)
    ep_b = AudioEndpoint(player=lambda s: src_a.push(chunked(s)), source=src_b)

    data = bytes(np.random.default_rng(11).integers(0, 256, 4_096, dtype=np.uint8))
    sender = FileSender(data, "demo.bin", "fast")
    receiver = FileReceiver()
    outgoing_profiles = set()
    for _ in range(5):
        if sender.done:
            break
        for profile, blocks in sender.outgoing():
            outgoing_profiles.add(profile)
            ep_a.send_blocks(blocks, profile)
        for block in ep_b.receive_blocks("fast", timeout_s=10.0):
            receiver.feed(block)
        for profile, blocks in receiver.response():
            ep_b.send_blocks(blocks, profile)
        for block in ep_a.receive_blocks("robust", timeout_s=10.0):
            sender.feed(block)

    assert sender.done
    assert receiver.ok
    assert receiver.data == data
    assert outgoing_profiles == {"robust", "fast"}  # HEADER — robust, DATA — fast
