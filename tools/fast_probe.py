"""Зонд fast (диагностика тикета 09): статистика приёма по поднесущим.

Передатчик играет партии из известной (seed) псевдослучайной
последовательности аккордов fast; приёмник записывает и по каждой
поднесущей печатает: долю ошибок argmax, отрыв переданного тона от
лучшего соседа по куску сетки, уровень переданного тона и фон его бинов,
когда тон молчит. Один живой прогон отвечает на вопрос «какие поднесущие
мертвы»: провал АЧХ виден как тихий тон, интермодуляция динамика — как
высокий фон молчащих бинов.

Запуск из корня репо (сначала приёмник, как у afr_sweep):
  python tools/fast_probe.py record              # на принимающем ноутбуке
  python tools/fast_probe.py play                # на передающем
  python tools/fast_probe.py both                # один ноутбук сам играет и пишет
  python tools/fast_probe.py analyze --wav out/fast_probe.wav  # пересчитать запись

record/both сохраняют запись (out/fast_probe.wav) и сразу печатают отчёт.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem.channel import clock_drift
from modem.profiles import PROFILES, SAMPLE_RATE, Profile
from modem.tx import _tone, chirp

_EPS = 1e-12
SEED = 20_260_725          # известен обеим сторонам: приёмник знает партитуру
BURST_GAP_SAMPLES = 12_000  # 0.25 с тишины между партиями — эху затихнуть


def probe_values(profile: Profile, symbols: int, seed: int = SEED) -> list[int]:
    """Партитура одной партии: symbols случайных значений символа.

    Все партии играют одну и ту же последовательность — пропуск партии
    приёмником не сдвигает сверку остальных.
    """
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, 2 ** profile.bits_per_symbol, symbols)]


def burst_signal(profile: Profile, symbols: int, seed: int = SEED) -> np.ndarray:
    """Одна партия: chirp-преамбула + пауза + известные аккорды."""
    guard = np.zeros(profile.guard_samples)
    parts = [chirp(profile), np.zeros(profile.preamble_gap_samples)]
    for value in probe_values(profile, symbols, seed):
        parts += [_tone(value, profile), guard]
    return np.concatenate(parts).astype(np.float32)


def probe_signal(
    profile: Profile, symbols: int, bursts: int, seed: int = SEED
) -> np.ndarray:
    """Полный сигнал зонда: bursts одинаковых партий через паузы."""
    burst = burst_signal(profile, symbols, seed)
    gap = np.zeros(BURST_GAP_SAMPLES, dtype=np.float32)
    return np.concatenate([burst if i % 2 == 0 else gap
                           for i in range(2 * bursts - 1)])


@dataclass
class SubcarrierStats:
    """Сводка одной поднесущей по всем принятым символам."""

    lo_hz: float
    hi_hz: float
    errors: int
    total: int
    margin_db: float   # медиана отрыва переданного тона от лучшего соседа
    active_db: float   # медиана уровня переданного тона (отн. общей медианы)
    idle_db: float     # медиана фона бинов поднесущей, когда их тон молчит

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 1.0

    @property
    def verdict(self) -> str:
        # Случайный argmax из 4 тонов ошибается в 75% случаев.
        if self.error_rate > 0.30:
            return "МЕРТВА"
        if self.error_rate > 0.05:
            return "слаба"
        return "ок"


@dataclass
class ProbeReport:
    """Итог анализа записи зонда."""

    bursts_found: int
    bursts_expected: int
    drifts: list[float]                 # дрейф победившего шаблона по партиям
    subcarriers: list[SubcarrierStats]
    bin_active_db: list[float]          # уровень каждого из 36 тонов сетки
    bin_idle_db: list[float]            # фон каждого бина, когда тон молчит
    errors: int
    total: int
    # След реверберации: фон молчащего бина, звучавшего в ПРЕДЫДУЩЕМ символе,
    # против фона бина, молчавшего и там. Большая разница — хвост прошлого
    # аккорда не затухает за защитный интервал (эхо комнаты, не шум).
    echo_prev_db: float = float("nan")
    echo_other_db: float = float("nan")
    # Ошибки половин партии: рост ко второй половине — уплывание тайминга.
    half_errors: tuple[int, int] = (0, 0)
    half_totals: tuple[int, int] = (0, 0)


def analyze(
    recording: np.ndarray,
    profile: Profile,
    symbols: int,
    bursts: int,
    seed: int = SEED,
) -> ProbeReport:
    """Запись зонда → статистика по поднесущим (чистая функция).

    Синхронизация — как у rx: банк шаблонов chirp по сетке дрейфов,
    позиции окон символов масштабируются оценкой дрейфа победителя.
    """
    rec = np.asarray(recording, dtype=np.float64)
    p = profile
    expected = probe_values(p, symbols, seed)
    burst_len = p.preamble_samples + p.preamble_gap_samples + symbols * p.slot_samples

    stack, templates = _preamble_ncc(rec, p)
    peaks = _pick_peaks(stack.max(axis=0), p.preamble_threshold,
                        min_sep=int(burst_len * 0.7), limit=bursts)

    bins = [round(tone * p.symbol_samples / SAMPLE_RATE) for tone in p.tones_hz]
    per = p.tones_per_carrier
    sent: list[list[float]] = [[] for _ in bins]    # магнитуды переданного тона
    silent: list[list[float]] = [[] for _ in bins]  # магнитуды молчащего бина
    margins: list[list[float]] = [[] for _ in range(p.subcarriers)]
    errors = [0] * p.subcarriers
    totals = [0] * p.subcarriers
    drifts = []
    echo_prev: list[float] = []   # молчащий бин звучал в предыдущем символе
    echo_other: list[float] = []  # молчащий бин молчал и в предыдущем
    half_errors, half_totals = [0, 0], [0, 0]

    for peak in peaks:
        drift = templates[int(np.argmax(stack[:, peak]))]
        scale = 1 / (1 + drift)
        data_start = p.preamble_samples + p.preamble_gap_samples
        # Конец последнего ОКНА, без финального guard: его в записи может
        # не быть (запись оборвана сразу за сигналом), а читаем только окна.
        end_nominal = data_start + (symbols - 1) * p.slot_samples + p.symbol_samples
        if peak + round(end_nominal * scale) > len(rec):
            continue  # партия обрезана краем записи
        drifts.append(drift)
        for i, value in enumerate(expected):
            pos = peak + round((data_start + i * p.slot_samples) * scale)
            window = rec[pos : pos + p.symbol_samples]
            mags = np.abs(np.fft.rfft(window, n=p.symbol_samples))[bins]
            half = 0 if i < symbols // 2 else 1
            for c in range(p.subcarriers):
                group = mags[c * per : (c + 1) * per]
                shift = p.bits_per_carrier * (p.subcarriers - 1 - c)
                tone = (value >> shift) & (per - 1)
                prev_tone = (
                    (expected[i - 1] >> shift) & (per - 1) if i > 0 else None
                )
                others = np.delete(group, tone)
                margins[c].append(
                    20 * np.log10((group[tone] + _EPS) / (others.max() + _EPS))
                )
                totals[c] += 1
                half_totals[half] += 1
                if int(np.argmax(group)) != tone:
                    errors[c] += 1
                    half_errors[half] += 1
                for k in range(per):
                    if k == tone:
                        sent[c * per + k].append(group[k])
                        continue
                    silent[c * per + k].append(group[k])
                    if prev_tone is not None:
                        (echo_prev if k == prev_tone else echo_other).append(group[k])

    # 0 дБ — медиана магнитуды всех переданных тонов: уровни сравнимы между собой.
    ref = float(np.median(np.concatenate([m for m in sent if m] or [[1.0]])))
    to_db = lambda mags: (
        20 * float(np.log10((np.median(mags) + _EPS) / (ref + _EPS)))
        if mags else float("nan")
    )
    subs = [
        SubcarrierStats(
            lo_hz=p.tones_hz[c * per],
            hi_hz=p.tones_hz[(c + 1) * per - 1],
            errors=errors[c],
            total=totals[c],
            margin_db=float(np.median(margins[c])) if margins[c] else float("nan"),
            active_db=to_db(sum((sent[c * per + k] for k in range(per)), [])),
            idle_db=to_db(sum((silent[c * per + k] for k in range(per)), [])),
        )
        for c in range(p.subcarriers)
    ]
    return ProbeReport(
        bursts_found=len(drifts),
        bursts_expected=bursts,
        drifts=drifts,
        subcarriers=subs,
        bin_active_db=[to_db(m) for m in sent],
        bin_idle_db=[to_db(m) for m in silent],
        errors=sum(errors),
        total=sum(totals),
        echo_prev_db=to_db(echo_prev),
        echo_other_db=to_db(echo_other),
        half_errors=(half_errors[0], half_errors[1]),
        half_totals=(half_totals[0], half_totals[1]),
    )


def _preamble_ncc(rec: np.ndarray, p: Profile) -> tuple[np.ndarray, list[float]]:
    """НКК записи с банком шаблонов chirp (модель дрейфа — как у rx)."""
    nominal = chirp(p).astype(np.float64)
    nccs, drifts = [], []
    squares = np.concatenate([[0.0], np.cumsum(rec**2)])
    for drift in p.drift_grid:
        tpl = clock_drift(nominal, drift)
        dots = fftconvolve(rec, tpl[::-1], mode="valid")
        energies = np.sqrt(np.maximum(squares[len(tpl):] - squares[:-len(tpl)], 0.0))
        nccs.append(dots / (energies * float(np.linalg.norm(tpl)) + _EPS))
        drifts.append(drift)
    length = min(len(n) for n in nccs)
    return np.stack([n[:length] for n in nccs]), drifts


def _pick_peaks(
    ncc: np.ndarray, threshold: float, min_sep: int, limit: int
) -> list[int]:
    """Пики НКК: жадно от сильнейшего, подавляя соседей ближе min_sep."""
    ncc = ncc.copy()
    peaks: list[int] = []
    while len(peaks) < limit:
        idx = int(np.argmax(ncc))
        if ncc[idx] < threshold:
            break
        peaks.append(idx)
        ncc[max(0, idx - min_sep) : idx + min_sep] = -np.inf
    return sorted(peaks)


def format_report(report: ProbeReport, profile: Profile, show_bins: bool = False) -> str:
    """Отчёт зонда: таблица поднесущих + вердикты."""
    lines = [
        f"Партий найдено: {report.bursts_found}/{report.bursts_expected}"
        + (f", дрейфы шаблонов: {sorted(set(report.drifts))}" if report.drifts else "")
    ]
    if not report.bursts_found:
        lines.append(
            "Ни одной преамбулы не найдено: проверьте запись (уровень, "
            "устройство, «улучшения звука») — как в README перед живыми тестами."
        )
        return "\n".join(lines)
    lines += [
        "0 дБ — медиана уровня переданных тонов; «фон» — бины поднесущей, когда её тон молчит.",
        "",
        "П/н  Полоса, Гц  Ошибки            Отрыв, дБ  Тон, дБ  Фон, дБ  Вердикт",
    ]
    for c, s in enumerate(report.subcarriers):
        lines.append(
            f"{c:3d}  {s.lo_hz:4.0f}–{s.hi_hz:4.0f}  "
            f"{s.errors:4d}/{s.total:<4d} ({100 * s.error_rate:5.1f}%)  "
            f"{s.margin_db:+9.1f}  {s.active_db:+7.1f}  {s.idle_db:+7.1f}  {s.verdict}"
        )
    lines.append(
        f"Итого ошибок: {report.errors}/{report.total} "
        f"({100 * report.errors / max(report.total, 1):.1f}%)"
    )
    echo_gap = report.echo_prev_db - report.echo_other_db
    lines.append(
        f"След эха: фон бинов прошлого символа {report.echo_prev_db:+.1f} дБ "
        f"против прочих {report.echo_other_db:+.1f} дБ (разница {echo_gap:+.1f}; "
        ">3 — хвост прошлого аккорда не затухает, громкость не спасёт)"
    )
    (e1, e2), (t1, t2) = report.half_errors, report.half_totals
    lines.append(
        f"Половины партии: 1-я {e1}/{t1} ({100 * e1 / max(t1, 1):.1f}%), "
        f"2-я {e2}/{t2} ({100 * e2 / max(t2, 1):.1f}%) "
        "(рост ко 2-й — уплывание тайминга)"
    )
    dead = [f"{s.lo_hz:.0f}–{s.hi_hz:.0f} Гц" for s in report.subcarriers
            if s.verdict != "ок"]
    if dead:
        lines.append("Проблемные поднесущие: " + ", ".join(dead))
    if show_bins:
        lines += ["", "Бин  Тон, Гц  Уровень, дБ  Фон, дБ"]
        for k, tone in enumerate(profile.tones_hz):
            lines.append(
                f"{k:3d}  {tone:7.0f}  {report.bin_active_db[k]:+11.1f}"
                f"  {report.bin_idle_db[k]:+7.1f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Зонд fast: известные аккорды в эфир, статистика по поднесущим."
    )
    parser.add_argument("command", choices=("play", "record", "both", "analyze"))
    parser.add_argument("--profile", default="fast", choices=sorted(PROFILES))
    parser.add_argument("--symbols", type=int, default=64, help="символов в партии")
    parser.add_argument("--bursts", type=int, default=8, help="число партий")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bins", action="store_true", help="таблица всех 36 бинов")
    parser.add_argument("--wav", default="out/fast_probe.wav", help="файл записи")
    parser.add_argument("--in-device", default=None, help="устройство записи")
    parser.add_argument("--out-device", default=None, help="устройство вывода")
    args = parser.parse_args()

    profile = PROFILES[args.profile]

    if args.command == "analyze":
        from scipy.io import wavfile

        rate, rec = wavfile.read(args.wav)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"ожидался WAV {SAMPLE_RATE} Гц, в файле {rate}")
        if rec.dtype == np.int16:
            rec = rec / 32_768.0
        report = analyze(np.asarray(rec, dtype=np.float64), profile,
                         args.symbols, args.bursts, args.seed)
        print(format_report(report, profile, args.bins))
        return

    import sounddevice as sd

    signal = probe_signal(profile, args.symbols, args.bursts, args.seed)
    if args.command == "play":
        print(f"Играю зонд {args.profile}: {args.bursts} партий × "
              f"{args.symbols} символов, {len(signal) / SAMPLE_RATE:.1f} с…")
        sd.play(signal, SAMPLE_RATE, device=_device(args.out_device), blocking=True)
        return

    # record слушает дольше зонда: запас на ручной запуск двух ноутбуков.
    record_samples = len(signal) + 8 * SAMPLE_RATE
    print(f"Пишу микрофон {record_samples / SAMPLE_RATE:.0f} с…")
    if args.command == "both":
        padded = np.zeros(record_samples, dtype=np.float32)
        padded[SAMPLE_RATE : SAMPLE_RATE + len(signal)] = signal
        recording = sd.playrec(
            padded, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=(_device(args.in_device), _device(args.out_device)),
        )
    else:
        recording = sd.rec(record_samples, samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32",
                           device=_device(args.in_device))
    sd.wait()
    recording = recording[:, 0]

    from scipy.io import wavfile

    wav = Path(args.wav)
    wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav, SAMPLE_RATE, recording)
    print(f"Запись: {wav}")
    report = analyze(np.asarray(recording, dtype=np.float64), profile,
                     args.symbols, args.bursts, args.seed)
    print(format_report(report, profile, args.bins))


def _device(value: str | None) -> int | str | None:
    """Номер или подстрока имени устройства sounddevice."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
