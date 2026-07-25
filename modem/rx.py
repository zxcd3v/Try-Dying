"""Приёмник: поток сэмплов → блоки байтов (чистые функции, без железа).

Потоковый разбор «сэмплы → блоки»: скользящая нормированная корреляция ищет
chirp-преамбулу, аир-заголовок принимается по большинству из ×N копий
(ADR-0002), окна символов режутся по известной сетке, тон выбирается argmax
по FFT-бинам сетки профиля.

Результат не зависит от нарезки входа на куски: каждое решение — чистая
функция сэмплов на абсолютных позициях потока. Кандидат преамбулы
принимается только когда доступен полный lookahead после него (иначе ждём
следующих кусков), а отвергнутый офсет отвергнут навсегда.

Дрейф частоты дискретизации (рассинхрон часов ноутбуков, критерий ±0.1%,
тикет 03): длинный chirp теряет корреляцию уже при 0.05% масштаба, поэтому
ищем по банку шаблонов, передискретизированных на сетке дрейфов; победивший
шаблон заодно даёт оценку дрейфа, которой масштабируются позиции окон
символов внутри кадра.
"""

from collections import Counter
from collections.abc import Iterable, Iterator

import numpy as np
from scipy.signal import fftconvolve

from modem.channel import clock_drift
from modem.framing import HEADER_COPY_BYTES, crc8
from modem.profiles import PROFILES, SAMPLE_RATE, Profile, banded
from modem.tx import chirp

_EPS = 1e-12


def demodulate_stream(
    chunks: Iterable[np.ndarray], profile_name: str, band: int | None = None
) -> Iterator[bytes]:
    """Куски сигнала произвольного размера → блоки, по мере готовности."""
    if profile_name not in PROFILES:
        raise ValueError(f"неизвестный профиль {profile_name!r}")
    demod = _Demodulator(banded(profile_name, band))
    for chunk in chunks:
        yield from demod.feed(chunk)
    yield from demod.flush()


def demodulate(
    samples: np.ndarray, profile_name: str, band: int | None = None
) -> list[bytes]:
    """Оффлайн-обёртка: весь сигнал целиком → список блоков."""
    return list(demodulate_stream([samples], profile_name, band))


def find_preamble(samples: np.ndarray, profile: Profile) -> tuple[int, float] | None:
    """Первая chirp-преамбула в записи: (позиция, масштаб времени) или None.

    Оффлайн-двойник поиска _Demodulator для служебных сигналов, у которых
    нет аир-кадра, — зонда калибровки (ADR-0003). Дрейф часов ловится тем
    же банком шаблонов, поэтому зонд не нуждается в собственной эвристике
    «самый громкий кусок записи».
    """
    samples = np.asarray(samples, dtype=np.float64)
    bank = _template_bank(profile)
    if len(samples) < max(len(tpl) for _, tpl, _ in bank):
        return None
    stack = _correlation_stack(samples, bank)
    ncc = stack.max(axis=0)
    hits = np.flatnonzero(ncc >= profile.preamble_threshold)
    if not len(hits):
        return None
    # Первый превысивший порог офсет — ещё не обязательно вершина: берём
    # максимум в окне рефрактерности после него (как в потоковом поиске).
    first = int(hits[0])
    start = first + int(np.argmax(ncc[first : first + profile.preamble_samples // 2]))
    drift = bank[int(np.argmax(stack[:, start]))][0]
    return start, 1 / (1 + drift)


def _template_bank(profile: Profile) -> list[tuple[float, np.ndarray, float]]:
    """Преамбула, как её услышат часы с каждым дрейфом сетки профиля.

    clock_drift канал-симулятора — та же модель времени, что у приёмника.
    """
    nominal = chirp(profile).astype(np.float64)
    return [
        (drift, tpl, float(np.linalg.norm(tpl)))
        for drift in profile.drift_grid
        for tpl in [clock_drift(nominal, drift)]
    ]


def _correlation_stack(
    buf: np.ndarray, bank: list[tuple[float, np.ndarray, float]]
) -> np.ndarray:
    """НКК буфера с каждым шаблоном банка, обрезанные до общей длины."""
    nccs = [_normalized_correlation(buf, tpl, norm) for _, tpl, norm in bank]
    length = min(len(n) for n in nccs)
    return np.stack([n[:length] for n in nccs])


def _normalized_correlation(
    buf: np.ndarray, template: np.ndarray, template_norm: float
) -> np.ndarray:
    """НКК буфера с шаблоном по всем полным окнам."""
    dots = fftconvolve(buf, template[::-1], mode="valid")
    squares = np.concatenate([[0.0], np.cumsum(buf**2)])
    tpl = len(template)
    energies = np.sqrt(np.maximum(squares[tpl:] - squares[:-tpl], 0.0))
    return dots / (energies * template_norm + _EPS)


class _Demodulator:
    """Стейт-машина потокового разбора: поиск преамбулы → разбор кадра."""

    def __init__(self, profile: Profile):
        self.profile = profile
        self.templates = _template_bank(profile)
        # Пик обязан быть максимумом в этом окне после себя — отсекает
        # боковые лепестки корреляции chirp'а рядом с истинным пиком.
        self.refractory = profile.preamble_samples // 2
        self.buf = np.zeros(0, dtype=np.float64)
        self.base = 0  # абсолютная позиция buf[0] в потоке
        self.frame_start = None  # абсолютный старт найденной преамбулы
        self.scale = 1.0  # растяжение времени принятого кадра (оценка по банку)

    @property
    def frame_in_progress(self) -> bool:
        """Преамбула найдена, кадр ещё разбирается (активность для audio)."""
        return self.frame_start is not None

    def feed(self, chunk: np.ndarray) -> list[bytes]:
        self.buf = np.concatenate(
            [self.buf, np.asarray(chunk, dtype=np.float64)]
        )
        return self._drain(flush=False)

    def flush(self) -> list[bytes]:
        """Конец потока: дорешать что можно, недокачанные кадры — выбросить."""
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> list[bytes]:
        out = []
        progressed = True
        while progressed:
            progressed = False
            if self.frame_start is None:
                progressed = self._search(flush)
            if self.frame_start is not None:
                block, progressed_frame = self._parse_frame(flush)
                progressed = progressed or progressed_frame
                if block is not None:
                    out.append(block)
        return out

    # --- поиск преамбулы -------------------------------------------------

    def _search(self, flush: bool) -> bool:
        """Ищет chirp корреляцией с банком шаблонов; True — преамбула
        найдена, дрейф оценён по победившему шаблону."""
        if len(self.buf) < max(len(tpl) for _, tpl, _ in self.templates):
            return False
        stack = _correlation_stack(self.buf, self.templates)
        length = stack.shape[1]
        ncc = stack.max(axis=0)
        # Офсеты < cutoff имеют полный lookahead — решение по ним финально.
        cutoff = length if flush else length - self.refractory
        for idx in np.flatnonzero(ncc >= self.profile.preamble_threshold):
            if idx >= cutoff:
                break  # мало lookahead — дождёмся следующих кусков
            tail = ncc[idx + 1 : idx + 1 + self.refractory]
            if ncc[idx] >= np.max(tail, initial=-np.inf):
                drift = self.templates[int(np.argmax(stack[:, idx]))][0]
                self.scale = 1 / (1 + drift)
                self.frame_start = self.base + int(idx)
                self._discard_before(self.frame_start)
                return True
        # Никого не приняли: всё финально-отвергнутое выбрасываем,
        # неподтверждённых кандидатов (>= cutoff) храним до новых кусков.
        self._discard_before(self.base + max(int(cutoff), 0))
        return False

    # --- разбор аир-кадра ------------------------------------------------

    def _parse_frame(self, flush: bool) -> tuple[bytes | None, bool]:
        """Разбирает кадр от зафиксированной преамбулы.

        Возвращает (блок | None, был ли прогресс). Кадр с непринятым
        аир-заголовком отбрасывается целиком: поиск продолжается сразу за
        преамбулой, соседние кадры не страдают (ADR-0002).
        """
        p = self.profile
        start = self.frame_start - self.base
        # Офсеты внутри кадра считаем в номинальном времени передатчика,
        # в позиции буфера переводим через оценку дрейфа (_received_offset).
        data_start = p.preamble_samples + p.preamble_gap_samples
        header_bytes = HEADER_COPY_BYTES * p.header_repeats
        header_end = data_start + p.symbols_for_bytes(header_bytes) * p.slot_samples
        if len(self.buf) < start + self._received_offset(header_end):
            if flush:
                self._resume_search_at(self.base + len(self.buf))
                return None, True
            return None, False

        header = self._read_bytes(data_start, header_bytes, pilots=False)
        block_len = self._vote_block_len(header)
        if block_len is None:
            # Заголовок погиб — выбрасываем кадр, ищем сразу за преамбулой.
            self._resume_search_at(
                self.frame_start + self._received_offset(p.preamble_samples)
            )
            return None, True

        data_end = header_end + self._data_span(p.symbols_for_bytes(block_len))
        if len(self.buf) < start + self._received_offset(data_end):
            if flush:
                self._resume_search_at(self.base + len(self.buf))
                return None, True
            return None, False

        block = self._read_bytes(header_end, block_len, pilots=True)
        self._resume_search_at(self.frame_start + self._received_offset(data_end))
        return block, True

    def _data_span(self, n_symbols: int) -> int:
        """Длина участка данных в сэмплах с учётом вставок пилотов."""
        p = self.profile
        inserts = (n_symbols - 1) // p.pilot_every if n_symbols else 0
        total = n_symbols + inserts * len(p.pilot_pattern)
        return total * p.slot_samples

    def _read_bytes(self, start: int, n_bytes: int, *, pilots: bool) -> bytes:
        """Читает n_bytes байтов от номинального офсета start в кадре,
        пропуская пилоты."""
        p = self.profile
        frame = self.frame_start - self.base
        values = []
        pos = start
        for i in range(p.symbols_for_bytes(n_bytes)):
            if pilots and i > 0 and i % p.pilot_every == 0:
                pos += p.slot_samples * len(p.pilot_pattern)  # зеркало вставки в tx
            values.append(self._symbol_at(frame + self._received_offset(pos)))
            pos += p.slot_samples
        return _symbols_to_bytes(values, p, n_bytes)

    def _received_offset(self, nominal: int) -> int:
        """Номинальный офсет от старта кадра → офсет в принятом времени."""
        return round(nominal * self.scale)

    def _symbol_at(self, pos: int) -> int:
        """FFT окна символа → по каждой поднесущей argmax магнитуды в её
        куске сетки; биты поднесущих склеиваются в значение символа
        (зеркало Profile.chord_tones: нижняя поднесущая — старшие биты)."""
        p = self.profile
        window = self.buf[pos : pos + p.symbol_samples]
        spectrum = np.abs(np.fft.rfft(window, n=p.symbol_samples))
        bins = [round(tone * p.symbol_samples / SAMPLE_RATE) for tone in p.tones_hz]
        magnitudes = spectrum[bins]
        per = p.tones_per_carrier
        value = 0
        for c in range(p.subcarriers):
            value = (value << p.bits_per_carrier) | int(
                np.argmax(magnitudes[c * per : (c + 1) * per])
            )
        return value

    def _vote_block_len(self, header: bytes) -> int | None:
        """Длина блока из ×N копий (длина 2 Б + CRC-8) по большинству.

        Сначала побайтное большинство всех копий, затем каждая копия
        отдельно; побеждает первый кандидат с валидным CRC и длиной в
        пределах профиля. None — заголовок не принят.

        Запасной проход по одиночным копиям — осознанное расширение
        «большинства» ADR-0002: две по-разному битые копии отравляют
        побайтное голосование даже при целой третьей; CRC-8 плюс проверка
        диапазона длины держат ложный приём копии на пренебрежимом уровне.
        """
        copies = [
            header[i * HEADER_COPY_BYTES : (i + 1) * HEADER_COPY_BYTES]
            for i in range(self.profile.header_repeats)
        ]
        majority = bytes(
            Counter(column).most_common(1)[0][0] for column in zip(*copies)
        )
        for candidate in (majority, *copies):
            if crc8(candidate[:2]) == candidate[2]:
                block_len = int.from_bytes(candidate[:2], "big")
                if 1 <= block_len <= self.profile.max_block:
                    return block_len
        return None

    # --- служебное -------------------------------------------------------

    def _resume_search_at(self, absolute: int) -> None:
        self.frame_start = None
        self._discard_before(absolute)

    def _discard_before(self, absolute: int) -> None:
        drop = absolute - self.base
        if drop > 0:
            self.buf = self.buf[drop:]
            self.base = absolute


def _symbols_to_bytes(values: list[int], profile: Profile, n_bytes: int) -> bytes:
    """Значения символов → n_bytes байтов, старшие биты вперёд (зеркало tx):
    символы склеиваются в битовый поток, нулевая добивка хвоста срезается."""
    bits = profile.bits_per_symbol
    stream = 0
    for value in values:
        stream = (stream << bits) | value
    stream >>= len(values) * bits - 8 * n_bytes
    return stream.to_bytes(n_bytes, "big")
