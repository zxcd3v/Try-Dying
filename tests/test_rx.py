"""Тесты приёмника на внешнем шве модема: демодуляция(модуляция(блоки)).

Testing Decisions спеки: внутрь (корреляция, FFT-окна, стейт-машина) не
заглядываем — проверяем только свойства петли на чистом канале. Канал с
шумом/затуханием/дрейфом — тикет 03.
"""

import numpy as np

from modem.framing import HEADER_COPY_BYTES
from modem.profiles import MAX_BLOCK, PROFILES
from modem.rx import demodulate, demodulate_stream
from modem.tx import modulate

BLOCKS = [
    b"\xa7",                       # минимум: 1 байт
    b"hello, air",
    bytes(range(96)),              # MAX_BLOCK robust — все значения байтов
    b"\x00\x00\x00",               # нулевые байты не теряются
    b"\xff" * 17,
]


def test_loop_recovers_batch_byte_for_byte_in_order():
    assert len(BLOCKS[2]) == MAX_BLOCK["robust"]
    samples = modulate(BLOCKS, "robust")
    assert demodulate(samples, "robust") == BLOCKS


def test_medium_loop_recovers_batch_byte_for_byte_in_order():
    # Тикет 08: круговая петля medium, включая блок в полный MAX_BLOCK.
    blocks = [*BLOCKS, bytes(range(128, 0, -1))]
    assert len(blocks[-1]) == MAX_BLOCK["medium"]
    samples = modulate(blocks, "medium")
    assert demodulate(samples, "medium") == blocks
    assert list(demodulate_stream(_chunked(samples, [1024]), "medium")) == blocks


def test_fast_loop_recovers_batch_byte_for_byte_in_order():
    # Тикет 09: круговая петля fast (параллельные поднесущие). Блок в
    # полный MAX_BLOCK и длины, не кратные битам символа, — упаковка бит
    # в «аккорды» обязана вернуть байты точно.
    blocks = [*BLOCKS, bytes(range(255, 0, -1))]
    assert len(blocks[-1]) == MAX_BLOCK["fast"]
    samples = modulate(blocks, "fast")
    assert demodulate(samples, "fast") == blocks
    assert list(demodulate_stream(_chunked(samples, [1024]), "fast")) == blocks


def test_streaming_result_does_not_depend_on_chunking():
    samples = modulate(BLOCKS, "robust")
    whole = list(demodulate_stream([samples], "robust"))
    fixed = list(demodulate_stream(_chunked(samples, [1024]), "robust"))
    rng = np.random.default_rng(seed=7)
    sizes = rng.integers(1, 20_000, size=1_000).tolist()
    random_cut = list(demodulate_stream(_chunked(samples, sizes), "robust"))
    assert whole == fixed == random_cut == BLOCKS


def test_silence_around_signal_and_shifted_start_still_decode():
    # Некруглые длины тишины: старт кадра не совпадает ни с какой сеткой.
    samples = modulate(BLOCKS, "robust")
    padded = np.concatenate(
        [np.zeros(13_337, dtype=np.float32), samples, np.zeros(7_777, dtype=np.float32)]
    )
    assert demodulate(padded, "robust") == BLOCKS
    assert list(demodulate_stream(_chunked(padded, [1024]), "robust")) == BLOCKS


def test_frame_with_dead_air_header_is_dropped_neighbours_survive():
    # Убиваем аир-заголовок среднего кадра целиком (все 3 копии), заменив
    # его шумом. Кадр обязан выпасть весь, соседние — приняться без потерь.
    profile = PROFILES["robust"]
    first, second, third = b"first" * 3, b"victim" * 4, b"third" * 5
    samples = modulate([first, second, third], "robust").copy()

    mid_frame = len(modulate([first], "robust"))
    header_start = mid_frame + profile.preamble_samples + profile.preamble_gap_samples
    header_len = (
        HEADER_COPY_BYTES
        * profile.header_repeats
        * (8 // profile.bits_per_symbol)
        * profile.slot_samples
    )
    rng = np.random.default_rng(seed=42)
    samples[header_start : header_start + header_len] = 0.5 * rng.standard_normal(
        header_len
    ).astype(np.float32)

    assert demodulate(samples, "robust") == [first, third]
    assert list(demodulate_stream(_chunked(samples, [1024]), "robust")) == [
        first,
        third,
    ]


def _chunked(samples, sizes):
    """Режет сигнал на куски заданных размеров (циклически), имитируя микрофон."""
    pos = 0
    while pos < len(samples):
        size = sizes[0]
        sizes = sizes[1:] + sizes[:1]
        yield samples[pos : pos + size]
        pos += size
