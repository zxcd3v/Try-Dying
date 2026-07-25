"""Стресс-тесты петли демодуляция(канал(модуляция(блоки))) — тикет 03.

Канал-симулятор — чистые функции порчи сигнала (modem/channel.py); тесты
дёргают только внешний шов модема, как и test_rx. Пороги SNR, закреплённые
здесь, — регрессионные границы уверенного приёма (документированы в README).
"""

import numpy as np

from modem import channel
from modem.rx import demodulate
from modem.tx import modulate

SEED = 20260725

BLOCKS = [b"nuclear it hack", bytes(range(64)), b"\x00\xff" * 12]


def _modulated(profile: str = "robust") -> np.ndarray:
    return modulate(BLOCKS, profile)


def test_awgn_adds_noise_at_requested_snr():
    # Истина — определение SNR: 10·log10(P_signal / P_noise).
    rng = np.random.default_rng(SEED)
    signal = np.sin(2 * np.pi * 5_000.0 * np.arange(48_000) / 48_000)
    for snr_db in (20.0, 0.0, -10.0):
        noisy = channel.awgn(signal, snr_db, rng)
        noise = noisy - signal
        measured = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
        assert abs(measured - snr_db) < 0.5, f"SNR {measured:.2f} != {snr_db}"


def test_quiet_shifted_signal_with_noise_floor_still_decodes():
    # Тихий (затухание до 3% амплитуды), начавшийся в случайный момент
    # сигнал с щедрым SNR обязан пройти байт-в-байт: НКК преамбулы и argmax
    # по бинам не зависят от абсолютного уровня.
    rng = np.random.default_rng(SEED)
    corrupted = channel.awgn(
        channel.shift_start(channel.attenuate(_modulated(), 0.03), 4_321),
        20.0,
        rng,
    )
    assert demodulate(corrupted, "robust") == BLOCKS


def test_clock_drift_up_to_criterion_does_not_break_clean_decode():
    # ±0.1% — критерий тикета (рассинхрон часов двух ноутбуков). Кадры
    # самодостаточны: преамбула ресинхронизирует каждый, дрейф внутри
    # кадра не должен уводить окна символов с сетки.
    samples = _modulated()
    for drift in (+0.001, -0.001):
        assert demodulate(channel.clock_drift(samples, drift), "robust") == BLOCKS


# Эхо критерия тикета: ~50 мс задержки (много больше защитного интервала),
# ощутимая, но не перебивающая прямой тон копия.
ECHO_DELAY_SAMPLES = 2_400
ECHO_GAIN = 0.35


def test_echo_up_to_criterion_does_not_break_clean_decode():
    # Эхо прошлых символов ложится на текущие — ослабленная копия не
    # должна перебивать прямой тон.
    samples = _modulated()
    echoed = channel.echo(samples, ECHO_DELAY_SAMPLES, ECHO_GAIN)
    assert demodulate(echoed, "robust") == BLOCKS


# Регрессионные границы уверенного приёма robust. Найдены разведкой сетки
# SNR × сиды после тюнинга preamble_threshold 0.5→0.25 — воспроизводится
# командой `python tools/snr_grid.py`. Документированы в README. Выше
# границы партия обязана доходить без потерь; провал теста = регрессия
# чувствительности приёмника.
ROBUST_SNR_FLOOR_DB = -9.0        # белый шум поверх тихого сигнала
ROBUST_HARD_SNR_FLOOR_DB = -6.0   # шум + дрейф ±0.1% + эхо 50 мс


def _hard_channel(samples: np.ndarray, drift: float) -> np.ndarray:
    """Худший канал критериев тикета: дрейф и эхо разом."""
    return channel.echo(
        channel.clock_drift(samples, drift), ECHO_DELAY_SAMPLES, ECHO_GAIN
    )


def test_batch_survives_noise_at_documented_snr_floor():
    samples = channel.attenuate(_modulated(), 0.5)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noisy = channel.awgn(samples, ROBUST_SNR_FLOOR_DB, rng)
        assert demodulate(noisy, "robust") == BLOCKS, f"seed {seed}"


def test_batch_survives_drift_echo_and_noise_at_documented_snr_floor():
    for drift in (+0.001, -0.001):
        samples = channel.attenuate(_hard_channel(_modulated(), drift), 0.5)
        for seed in range(3):
            rng = np.random.default_rng(seed)
            noisy = channel.awgn(samples, ROBUST_HARD_SNR_FLOOR_DB, rng)
            assert demodulate(noisy, "robust") == BLOCKS, f"drift {drift} seed {seed}"


# --- профиль medium (тикет 08) ---------------------------------------------

# Регрессионная граница уверенного приёма medium: символ короче и тонов 16
# вместо 4, поэтому пол выше robust. Найдена разведкой
# `python tools/snr_grid.py medium`: при −3 дБ все сиды чисты на обоих
# каналах (чистый шум и шум+дрейф+эхо), первые битовые ошибки — при −6 дБ.
MEDIUM_SNR_FLOOR_DB = -3.0


def test_medium_quiet_shifted_signal_with_noise_floor_still_decodes():
    rng = np.random.default_rng(SEED)
    corrupted = channel.awgn(
        channel.shift_start(channel.attenuate(_modulated("medium"), 0.03), 4_321),
        20.0,
        rng,
    )
    assert demodulate(corrupted, "medium") == BLOCKS


def test_medium_clock_drift_up_to_criterion_does_not_break_clean_decode():
    samples = _modulated("medium")
    for drift in (+0.001, -0.001):
        assert demodulate(channel.clock_drift(samples, drift), "medium") == BLOCKS


def test_medium_echo_up_to_criterion_does_not_break_clean_decode():
    echoed = channel.echo(_modulated("medium"), ECHO_DELAY_SAMPLES, ECHO_GAIN)
    assert demodulate(echoed, "medium") == BLOCKS


def test_medium_batch_survives_noise_at_documented_snr_floor():
    samples = channel.attenuate(_modulated("medium"), 0.5)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noisy = channel.awgn(samples, MEDIUM_SNR_FLOOR_DB, rng)
        assert demodulate(noisy, "medium") == BLOCKS, f"seed {seed}"


def test_medium_batch_survives_drift_echo_and_noise_at_documented_snr_floor():
    # У medium один пол на оба канала: порог задаёт различение 16 тонов
    # в шуме, а не поиск преамбулы, поэтому дрейф и эхо погоды не делают.
    for drift in (+0.001, -0.001):
        samples = channel.attenuate(_hard_channel(_modulated("medium"), drift), 0.5)
        for seed in range(3):
            rng = np.random.default_rng(seed)
            noisy = channel.awgn(samples, MEDIUM_SNR_FLOOR_DB, rng)
            assert demodulate(noisy, "medium") == BLOCKS, f"drift {drift} seed {seed}"


# --- профиль fast (тикет 09) ------------------------------------------------

# Пороги fast мягкие — профиль для тишины: 9 параллельных поднесущих делят
# амплитуду, каждый тон в 9 раз тише одиночного. Найдены разведкой
# `python tools/snr_grid.py fast`: чистый шум — все сиды чисты при +12 дБ,
# первые битовые ошибки при +9; жёсткий канал (шум + дрейф ±0.1% + эхо
# 50 мс) чист при +21 дБ, ниже — битовые ошибки без потери кадров.
FAST_SNR_FLOOR_DB = 12.0
FAST_HARD_SNR_FLOOR_DB = 21.0


def test_fast_quiet_shifted_signal_with_noise_floor_still_decodes():
    rng = np.random.default_rng(SEED)
    corrupted = channel.awgn(
        channel.shift_start(channel.attenuate(_modulated("fast"), 0.03), 4_321),
        20.0,
        rng,
    )
    assert demodulate(corrupted, "fast") == BLOCKS


def test_fast_clock_drift_up_to_criterion_does_not_break_clean_decode():
    samples = _modulated("fast")
    for drift in (+0.001, -0.001):
        assert demodulate(channel.clock_drift(samples, drift), "fast") == BLOCKS


def test_fast_echo_up_to_criterion_does_not_break_clean_decode():
    # Эхо преамбулы в 9 раз громче каждого тона аккорда: заголовок спасает
    # пауза preamble_gap длиннее задержки эха (см. комментарий в profiles).
    echoed = channel.echo(_modulated("fast"), ECHO_DELAY_SAMPLES, ECHO_GAIN)
    assert demodulate(echoed, "fast") == BLOCKS


def test_fast_batch_survives_noise_at_documented_snr_floor():
    samples = channel.attenuate(_modulated("fast"), 0.5)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noisy = channel.awgn(samples, FAST_SNR_FLOOR_DB, rng)
        assert demodulate(noisy, "fast") == BLOCKS, f"seed {seed}"


def test_fast_batch_survives_drift_echo_and_noise_at_documented_snr_floor():
    for drift in (+0.001, -0.001):
        samples = channel.attenuate(_hard_channel(_modulated("fast"), drift), 0.5)
        for seed in range(3):
            rng = np.random.default_rng(seed)
            noisy = channel.awgn(samples, FAST_HARD_SNR_FLOOR_DB, rng)
            assert demodulate(noisy, "fast") == BLOCKS, f"drift {drift} seed {seed}"


def test_below_floor_frames_die_whole_or_get_bit_errors_never_byte_shift():
    _assert_degrades_without_byte_shift("robust", (-12.0, -15.0))


def test_medium_below_floor_frames_die_whole_or_get_bit_errors_never_byte_shift():
    _assert_degrades_without_byte_shift("medium", (-6.0, -9.0))


def test_fast_below_floor_frames_die_whole_or_get_bit_errors_never_byte_shift():
    _assert_degrades_without_byte_shift("fast", (9.0, 6.0))


def _assert_degrades_without_byte_shift(
    profile: str, snr_grid_db: tuple[float, ...]
) -> None:
    # Контракт «Блока» (CONTEXT.md): модем сохраняет длину и порядок байтов;
    # ниже порога допустимы только потеря кадра целиком (аир-заголовок ×3
    # не сошёлся) и битовые ошибки на месте — байты не съезжают.
    length = 32
    rng = np.random.default_rng(SEED)
    sent = [bytes(rng.integers(0, 256, length, dtype=np.uint8)) for _ in range(6)]
    samples = modulate(sent, profile)

    survived = errored_bits = total_bits = 0
    for snr_db in snr_grid_db:
        for seed in range(3):
            noise_rng = np.random.default_rng(seed)
            got = demodulate(channel.awgn(samples, snr_db, noise_rng), profile)
            assert len(got) <= len(sent)
            matched = []
            for block in got:
                assert len(block) == length, "длина блока обязана сохраниться"
                # Ближайший по Хэммингу отправленный блок — его прообраз;
                # съехавшие байты дали бы ~50% битовых ошибок к любому.
                distances = [_bit_errors(block, s) for s in sent]
                assert min(distances) / (8 * length) < 0.25, "байты съехали"
                matched.append(distances.index(min(distances)))
                errored_bits += min(distances)
                total_bits += 8 * length
                survived += 1
            # Уцелевшие — подпоследовательность отправленных: порядок цел.
            assert matched == sorted(set(matched)), f"порядок нарушен: {matched}"
    # Ниже порога канал жёсткий, но не пустой: часть кадров доходит,
    # суммарная доля битовых ошибок в уцелевших — малая.
    assert survived > 0
    assert errored_bits / total_bits < 0.05


def _bit_errors(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))
