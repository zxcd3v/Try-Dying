"""Полосы и зонд калибровки — шов модема (тикет 07, ADR-0003).

Дёргаем те же внешние швы, что и остальные тесты модема: «модуляция →
канал → демодуляция» для полос и «зонд → канал → измерение» для калибровки.
Канал с вырезанной полосой собирается композицией band_stop + awgn — именно
в таком порядке: провал АЧХ виден только на фоне шума приёмника, режектор
до шума глушит сигнал вместе с шумом и SNR не портит.
"""

import numpy as np
import pytest

from modem import channel, demodulate, modulate
from modem.probe import band_snr_db, best_band, probe, probe_tones_hz
from modem.profiles import (
    ANCHOR,
    PROFILES,
    SAMPLE_RATE,
    WINDOW_HI_HZ,
    WINDOW_LO_HZ,
    bands,
    banded,
    default_band,
)

BLOCKS = [b"nuclear it hack", bytes(range(64)), b"\x00\xff" * 12]
NOTCH = (1_500.0, 4_500.0)  # «провал» тракта, накрывающий низ рабочего окна


def notched(samples: np.ndarray, snr_db: float = 20.0, seed: int = 0) -> np.ndarray:
    """Канал с вырезанной полосой: провал АЧХ, поверх него — шум приёмника."""
    return channel.awgn(
        channel.band_stop(samples, *NOTCH), snr_db, np.random.default_rng(seed)
    )


# --- геометрия полос -----------------------------------------------------


@pytest.mark.parametrize("name", list(PROFILES))
def test_bands_fit_the_window_and_stay_in_fft_bins(name):
    """Полоса обязана целиком лежать в окне тракта, а тона — в бинах FFT
    (ADR-0001): иначе энергия тона размазывается по соседям."""
    bin_hz = SAMPLE_RATE / PROFILES[name].symbol_samples
    for band in bands(name):
        assert WINDOW_LO_HZ <= band.lo_hz and band.hi_hz <= WINDOW_HI_HZ
        for tone in banded(name, band.index).tones_hz:
            assert tone % bin_hz == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", ["medium", "fast"])
def test_data_profiles_have_three_or_four_bands_including_the_tuned_one(name):
    """ADR-0003: 3–4 полосы. Сетка профиля как есть — среди них всегда:
    на ней сняты пороги SNR и таблица скоростей, она же фолбэк."""
    assert 3 <= len(bands(name)) <= 4
    assert banded(name, default_band(name)).tones_hz == PROFILES[name].tones_hz


def test_anchor_profile_never_moves():
    """Якорная полоса фиксирована — служебный канал от калибровки не зависит."""
    assert len(bands(ANCHOR)) == 1
    assert bands(ANCHOR)[0].offset_hz == 0.0


def test_unknown_band_is_rejected():
    with pytest.raises(ValueError, match="полоса"):
        banded("medium", 99)


# --- передача в выбранной полосе -----------------------------------------


def test_blocks_survive_a_notch_in_the_band_calibration_would_pick():
    """Ключевой сценарий тикета: полоса «завалена», но передача в другой
    полосе того же профиля проходит целиком."""
    signal = notched(modulate(BLOCKS, "medium", band=3))

    assert demodulate(signal, "medium", band=3) == BLOCKS


def test_default_band_dies_in_the_same_notch():
    """Контроль: без калибровки та же посылка в полосе по умолчанию гибнет —
    значит, успех соседней полосы обеспечен именно выбором полосы."""
    signal = notched(modulate(BLOCKS, "medium"))

    assert demodulate(signal, "medium") != BLOCKS


def test_band_grids_do_not_decode_each_other():
    """Полосы различимы: сигнал одной полосы не читается сеткой другой."""
    signal = modulate(BLOCKS, "medium", band=3)

    assert demodulate(signal, "medium", band=3) == BLOCKS
    assert demodulate(signal, "medium", band=0) != BLOCKS


# --- зонд и измерение ----------------------------------------------------


def test_probe_sounds_every_tone_of_every_band():
    tones = probe_tones_hz("medium")
    for band in bands("medium"):
        for tone in banded("medium", band.index).tones_hz:
            assert tone in tones


def test_probe_measures_all_bands_at_once_on_a_clean_channel():
    """Чистый канал: все полосы примерно равны, лучшая не хуже остальных."""
    heard = channel.awgn(
        _recording(probe("medium")), 20.0, np.random.default_rng(0)
    )

    snr = band_snr_db(heard, "medium")

    assert snr is not None and len(snr) == len(bands("medium"))
    assert max(snr) - min(snr) < 5.0


def test_probe_picks_the_band_away_from_the_notch():
    """Одним приёмом зонда — SNR всех полос: чем дальше полоса от провала,
    тем выше её SNR, и лучшая полоса провала не касается."""
    snr = band_snr_db(notched(_recording(probe("medium"))), "medium")

    assert snr is not None
    assert snr == sorted(snr)  # полосы упорядочены по частоте, провал — снизу
    chosen = best_band(snr)
    assert bands("medium")[chosen].lo_hz >= NOTCH[1]


def test_calibration_choice_is_the_band_that_actually_survives():
    """Смычка измерения и передачи: полоса, выбранная по зонду, работает,
    а полоса по умолчанию в этом же канале — нет."""
    snr = band_snr_db(notched(_recording(probe("medium"))), "medium")
    chosen = best_band(snr)

    assert demodulate(notched(modulate(BLOCKS, "medium", band=chosen)), "medium",
                      band=chosen) == BLOCKS
    assert demodulate(notched(modulate(BLOCKS, "medium")), "medium") != BLOCKS


def test_silence_is_not_mistaken_for_a_probe():
    """Нет зонда — нет измерения: приёмник обязан уйти в полосу по умолчанию,
    а не выбрать полосу по шуму."""
    silence = channel.awgn(np.zeros(4 * SAMPLE_RATE), 0.0, np.random.default_rng(1))

    assert band_snr_db(silence, "medium") is None


def test_half_heard_probe_is_not_measured():
    """Зонд ещё звучит — мерить рано (эндпоинт по None понимает, что ждать)."""
    partial = _recording(probe("medium"))[: -len(probe("medium")) // 2]

    assert band_snr_db(partial, "medium") is None


def _recording(signal: np.ndarray) -> np.ndarray:
    """Запись микрофона: тишина, сигнал, тишина."""
    pad = np.zeros(SAMPLE_RATE // 4)
    return np.concatenate([pad, signal, pad])
