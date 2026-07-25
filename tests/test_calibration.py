"""Рукопожатие калибровки полосы — шов протокола (тикет 07, ADR-0003).

Логика рукопожатия (кто кого ждёт и что делает при молчании) проверяется на
скриптованном эндпоинте, сигнальная часть — уже в test_bands.py. Один тест
идёт через настоящий звук в памяти: зонд → провал АЧХ → измерение → кадр
BAND в robust — чтобы связка слоёв не разъехалась.
"""

import numpy as np
import pytest
from test_audio import FakeSource, chunked, transmitted
from test_bands import notched

from modem.audio import AudioEndpoint
from modem.probe import best_band, probe
from modem.profiles import ANCHOR, SAMPLE_RATE, bands, default_band
from protocol import frames
from protocol.calibration import (
    PROBE_ATTEMPTS,
    calibrate_receiver,
    calibrate_sender,
)
from protocol.spool_modem import SpoolEndpoint

MODE = "medium"
GOOD_BAND = 3  # полоса выше провала NOTCH из test_bands


class ScriptedEndpoint:
    """Эндпоинт-марионетка: заданное измерение и очередь входящих партий."""

    supports_calibration = True

    def __init__(self, incoming=(), snr_db=None):
        self.incoming = [list(batch) for batch in incoming]
        self.snr_db = snr_db
        self.probes = 0
        self.sent: list[tuple[str, list[bytes]]] = []
        self.band: int | None = None

    def send_probe(self, profile):
        self.probes += 1

    def measure_bands(self, profile, timeout_s):
        return self.snr_db

    def receive_blocks(self, profile, timeout_s):
        return self.incoming.pop(0) if self.incoming else []

    def send_blocks(self, blocks, profile):
        self.sent.append((profile, list(blocks)))

    def set_band(self, profile, band):
        self.band = band


def band_batch(band: int) -> list[bytes]:
    """Партия кадров BAND, как её шлёт другая сторона."""
    frame = frames.build_frame(frames.BAND, band, 0, bytes([band]))
    return [frames.encode_block(frame, ANCHOR)] * 3


def announced_band(endpoint) -> int | None:
    """Номер полосы из последней отправленной эндпоинтом партии BAND."""
    for profile, blocks in reversed(endpoint.sent):
        parsed = frames.decode_frame(blocks[0])
        if parsed and parsed[0] == frames.BAND:
            assert profile == ANCHOR  # служебное — только на якоре
            return parsed[3][0]
    return None


# --- сторона отправителя -------------------------------------------------


def test_sender_takes_the_band_the_receiver_measured():
    endpoint = ScriptedEndpoint(incoming=[band_batch(GOOD_BAND)])

    result = calibrate_sender(endpoint, MODE, wait_s=1.0)

    assert endpoint.probes == 1
    assert result.band == GOOD_BAND and result.confirmed
    assert endpoint.band == GOOD_BAND
    # Решение объявлено обратно: без него приёмник не знает, принят ли выбор.
    assert announced_band(endpoint) == GOOD_BAND


def test_silent_receiver_leaves_sender_on_the_default_band():
    """Критерий тикета: молчащий приёмник → таймаут → полоса по умолчанию."""
    endpoint = ScriptedEndpoint(incoming=[])  # тишина в ответ на любой зонд

    result = calibrate_sender(endpoint, MODE, wait_s=0.0)

    assert endpoint.probes == PROBE_ATTEMPTS  # зонд повторили, потом сдались
    assert result.band == default_band(MODE)
    assert not result.confirmed and "не ответил" in result.reason
    assert endpoint.band == default_band(MODE)


def test_sender_ignores_a_band_outside_the_profile():
    """Битый номер полосы (RS починил не всё) не должен уводить в никуда."""
    frame = frames.build_frame(frames.BAND, 0, 0, bytes([200]))
    endpoint = ScriptedEndpoint(incoming=[[frames.encode_block(frame, ANCHOR)]] * 2)

    result = calibrate_sender(endpoint, MODE, wait_s=0.0)

    assert result.band == default_band(MODE) and not result.confirmed


# --- сторона приёмника ---------------------------------------------------


def test_receiver_announces_the_best_band_and_obeys_the_sender():
    snr = [10.0, 12.0, 20.0, 30.0]
    endpoint = ScriptedEndpoint(incoming=[band_batch(GOOD_BAND)], snr_db=snr)

    result = calibrate_receiver(endpoint, MODE, probe_timeout_s=1.0, wait_s=1.0)

    assert announced_band(endpoint) == best_band(snr) == GOOD_BAND
    assert result.band == GOOD_BAND and result.confirmed
    assert result.snr_db == snr  # измерение уезжает в CLI целиком


def test_receiver_follows_the_sender_when_the_sender_chose_otherwise():
    """Решает отправитель: приёмник лишь предлагает лучшую по зонду."""
    endpoint = ScriptedEndpoint(
        incoming=[band_batch(0)], snr_db=[10.0, 12.0, 20.0, 30.0]
    )

    result = calibrate_receiver(endpoint, MODE, probe_timeout_s=1.0, wait_s=1.0)

    assert result.band == 0 and endpoint.band == 0
    assert "другую полосу" in result.reason


def test_receiver_without_a_probe_falls_back_to_the_default_band():
    endpoint = ScriptedEndpoint(snr_db=None)  # зонда в эфире не было

    result = calibrate_receiver(endpoint, MODE, probe_timeout_s=0.0, wait_s=0.0)

    assert result.band == default_band(MODE) and not result.confirmed
    assert endpoint.sent == []  # нечего объявлять — и эфир не засоряем


def test_receiver_unconfirmed_choice_falls_back_to_the_default_band():
    """Отправитель не подтвердил — значит, он остался на полосе по умолчанию,
    и приёмник обязан уйти туда же, а не на свою лучшую."""
    endpoint = ScriptedEndpoint(snr_db=[10.0, 12.0, 20.0, 30.0])

    result = calibrate_receiver(endpoint, MODE, probe_timeout_s=1.0, wait_s=0.0)

    assert result.band == default_band(MODE) and not result.confirmed


# --- когда калибровки не бывает ------------------------------------------


def test_anchor_profile_skips_calibration_entirely():
    """robust живёт на якорной полосе: ни зонда, ни ожидания (ADR-0003)."""
    endpoint = ScriptedEndpoint()

    result = calibrate_sender(endpoint, ANCHOR, wait_s=1.0)

    assert endpoint.probes == 0 and endpoint.sent == []
    assert result.band == default_band(ANCHOR)


def test_spool_link_skips_calibration(tmp_path):
    """Спул-линк носит блоки файлами — спектра, а значит и полос, там нет."""
    endpoint = SpoolEndpoint(tmp_path / "link", "a")

    result = calibrate_sender(endpoint, MODE, wait_s=1.0)

    assert result.band == default_band(MODE)
    assert "--audio" in result.reason


# --- независимость служебного канала -------------------------------------


def test_service_frames_stay_on_the_anchor_after_calibration():
    """Критерий тикета: robust-канал управления работает независимо от
    результата калибровки — HEADER/NACK/DONE звучат на якорной сетке."""
    played = []
    endpoint = AudioEndpoint(player=played.append, source=FakeSource())
    endpoint.set_band(MODE, GOOD_BAND)

    endpoint.send_blocks([b"service"], ANCHOR)
    endpoint.send_blocks([b"data"], MODE)

    anchor_only = AudioEndpoint(player=played.append, source=FakeSource())
    anchor_only.send_blocks([b"service"], ANCHOR)
    assert endpoint.band(ANCHOR) == default_band(ANCHOR)
    assert np.array_equal(played[0], played[2])  # якорь не сдвинулся
    assert not np.array_equal(played[1][: len(played[0])], played[0])


def test_receiver_hears_the_anchor_while_tuned_to_another_band():
    stream = np.concatenate(
        [np.zeros(24_000), transmitted([b"service in robust"], ANCHOR),
         np.zeros(3 * SAMPLE_RATE)]
    )
    rx = AudioEndpoint(source=FakeSource(chunked(stream)))
    rx.set_band(MODE, GOOD_BAND)

    assert list(rx.receive_blocks(MODE, timeout_s=60.0)) == [b"service in robust"]


# --- страховка от потерянного подтверждения ------------------------------


@pytest.mark.parametrize("air_band", [None, GOOD_BAND])
def test_receiver_listens_to_both_the_chosen_and_the_default_band(air_band):
    """Подтверждение полосы могло не долететь — тогда отправитель остался на
    полосе по умолчанию. Приёмник слушает обе и запирается на той, откуда
    пришёл первый кадр."""
    blocks = [b"whichever band", bytes(range(64))]
    stream = np.concatenate(
        [np.zeros(24_000), transmitted(blocks, MODE, band=air_band),
         np.zeros(3 * SAMPLE_RATE)]
    )
    rx = AudioEndpoint(source=FakeSource(chunked(stream)))
    rx.set_band(MODE, GOOD_BAND)

    assert list(rx.receive_blocks(MODE, timeout_s=60.0)) == blocks
    assert rx.band(MODE) == (air_band if air_band is not None else default_band(MODE))


def test_full_transfer_over_air_in_a_calibrated_band():
    """Полоса протянута через весь стек: файл уходит по звуку в полосе,
    выбранной калибровкой, DATA — на её сетке, служебное — на якоре."""
    from protocol.transfer import FileReceiver, FileSender

    src_a, src_b = FakeSource(), FakeSource()
    ep_a = AudioEndpoint(player=lambda s: src_b.push(chunked(s)), source=src_a)
    ep_b = AudioEndpoint(player=lambda s: src_a.push(chunked(s)), source=src_b)
    for endpoint in (ep_a, ep_b):  # исход калибровки, уже согласованный
        endpoint.set_band(MODE, GOOD_BAND)

    data = bytes(np.random.default_rng(7).integers(0, 256, 2_048, dtype=np.uint8))
    sender = FileSender(data, "demo.bin", MODE)
    receiver = FileReceiver()
    for _ in range(5):
        if sender.done:
            break
        for profile, blocks in sender.outgoing():
            ep_a.send_blocks(blocks, profile)
        for block in ep_b.receive_blocks(MODE, timeout_s=10.0):
            receiver.feed(block)
        for profile, blocks in receiver.response():
            ep_b.send_blocks(blocks, profile)
        for block in ep_a.receive_blocks(ANCHOR, timeout_s=10.0):
            sender.feed(block)

    assert sender.done and receiver.ok and receiver.data == data
    assert ep_b.band(MODE) == GOOD_BAND  # приём заперся на полосе эфира
    assert ep_b.band(ANCHOR) == default_band(ANCHOR)


# --- сквозной прогон через настоящий звук --------------------------------


class NoEchoSource(FakeSource):
    """Микрофон без собственного эха: играем в пустоту — чистить нечего,
    поэтому реплики, положенные заранее, доживают до своей очереди."""

    def drain(self) -> None:
        pass


def test_probe_crosses_the_air_from_one_endpoint_to_the_other():
    """Зонд одной стороны и измерение другой — через канал с провалом."""
    src_b = FakeSource()
    tx = AudioEndpoint(
        player=lambda samples: src_b.push(chunked(notched(samples))),
        source=FakeSource(),
    )
    rx = AudioEndpoint(player=lambda samples: None, source=src_b)

    tx.send_probe(MODE)
    snr = rx.measure_bands(MODE, timeout_s=30.0)

    assert snr is not None and best_band(snr) == GOOD_BAND


def test_receiver_calibrates_over_real_air_through_a_notch():
    """Слои вместе: зонд через провал АЧХ → измерение → BAND в robust →
    обе стороны на полосе, которую провал не задевает."""
    src = NoEchoSource()
    ep = AudioEndpoint(player=lambda samples: None, source=src)
    src.push(chunked(notched(np.concatenate([np.zeros(12_000), probe(MODE)]))))
    # Ответ отправителя: он услышал выбор и объявил его решением.
    src.push(chunked(transmitted(band_batch(GOOD_BAND), ANCHOR)))

    result = calibrate_receiver(ep, MODE, probe_timeout_s=30.0, wait_s=30.0)

    assert result.confirmed and result.band == GOOD_BAND
    assert best_band(result.snr_db) == GOOD_BAND
    assert bands(MODE)[result.band].lo_hz >= 4_500.0  # выше провала
    assert ep.band(MODE) == GOOD_BAND  # эндпоинт перестроен на неё
