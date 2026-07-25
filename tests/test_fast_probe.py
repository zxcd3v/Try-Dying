"""Зонд fast (диагностика тикета 09): чистые функции анализа.

Железо руками; тестируем петлю «зонд → канал → анализ»: чистый канал даёт
ноль ошибок, шум с дрейфом переживается, а вырезанная фильтром полоса
одной поднесущей видна в отчёте как мёртвая — ради этого зонд и писался.
"""

import numpy as np

from modem.channel import awgn, clock_drift, echo, shift_start
from modem.profiles import PROFILES, SAMPLE_RATE
from tools.fast_probe import analyze, format_report, probe_signal

FAST = PROFILES["fast"]
SYMBOLS = 32
BURSTS = 3


def _run(recording: np.ndarray):
    return analyze(recording, FAST, SYMBOLS, BURSTS)


def test_clean_loopback_zero_errors():
    signal = probe_signal(FAST, SYMBOLS, BURSTS)
    recording = shift_start(0.05 * signal, 5_000)  # тихо и с поздним стартом

    report = _run(recording)

    assert report.bursts_found == BURSTS
    assert report.errors == 0
    assert all(s.margin_db > 3.0 for s in report.subcarriers)


def test_noise_and_drift_kept_low():
    rng = np.random.default_rng(9)
    signal = probe_signal(FAST, SYMBOLS, BURSTS)
    recording = awgn(clock_drift(shift_start(signal, 5_000), 0.0005), 25.0, rng)

    report = _run(recording)

    assert report.bursts_found == BURSTS
    assert report.errors / report.total < 0.02
    # Белый шум не помнит прошлый символ: следа эха нет.
    assert report.echo_prev_db - report.echo_other_db < 3.0


def test_echo_trace_flags_reverb():
    # Эхо длиннее слота символа: хвост прошлого аккорда звучит в окне
    # текущего — фон бинов прошлого символа заметно выше прочих.
    signal = probe_signal(FAST, SYMBOLS, BURSTS)
    recording = echo(shift_start(signal.astype(np.float64), 5_000), 300, 0.3)

    report = _run(recording)

    assert report.bursts_found == BURSTS
    assert report.echo_prev_db > report.echo_other_db + 3.0


def test_notched_band_flags_its_subcarrier_dead():
    # Вырезаем полосу поднесущей №5 (6.0–6.6 кГц) — модель провала АЧХ.
    signal = probe_signal(FAST, SYMBOLS, BURSTS)
    recording = shift_start(signal.astype(np.float64), 5_000)
    spectrum = np.fft.rfft(recording)
    freqs = np.fft.rfftfreq(len(recording), 1 / SAMPLE_RATE)
    spectrum[(freqs >= 5_900.0) & (freqs <= 6_700.0)] = 0.0
    recording = np.fft.irfft(spectrum, n=len(recording))

    report = _run(recording)

    assert report.bursts_found == BURSTS
    notched = report.subcarriers[5]
    assert notched.error_rate > 0.30
    assert notched.verdict == "МЕРТВА"
    healthy = [s for c, s in enumerate(report.subcarriers) if c != 5]
    assert all(s.errors == 0 for s in healthy)
    # Отчёт называет мёртвую полосу по имени — по нему принимается решение.
    assert "6000–6600" in format_report(report, FAST)
