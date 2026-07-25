"""Живой звук: контракт v1.2 поверх sounddevice (тикет 06).

Тонкая обёртка над чистыми modulate/demodulate_stream: send_blocks играет
партию динамиком (блокирующе), receive_blocks стримит микрофон в потоковый
демодулятор. Полудуплекс: после собственного проигрывания всё записанное
выбрасывается (своё эхо — мусор).

Железо инжектируется швами player/source, поэтому логика тестируется
петлёй в памяти без звуковой карты; sounddevice импортируется лениво.
"""

import functools
import queue
import sys
import time
from collections.abc import Iterator

import numpy as np

from modem.probe import PROBE_TOTAL_SAMPLES, band_snr_db, probe
from modem.profiles import ANCHOR, PROFILES, SAMPLE_RATE, banded, bands, default_band
from modem.rx import _Demodulator  # тот же слой: потоковый разбор без железа
from modem.tx import modulate

# Тишина по краям проигрывания: звуковой карте нужно время на разгон,
# иначе первые миллисекунды (начало преамбулы) срезаются.
PLAY_PAD_SAMPLES = int(0.2 * SAMPLE_RATE)

# Сколько записи держим в поиске зонда: зонд короче 0.6 с, поэтому окно
# втрое длиннее гарантированно вмещает целый зонд и ограничивает счёт.
_PROBE_BUFFER_SAMPLES = 3 * PROBE_TOTAL_SAMPLES

# Конец партии ищем по тишине, но «тишиной» выглядит и кадр, чью преамбулу
# приёмник не нашёл: пока идут символы данных, для него это просто шум.
# Поэтому порог тишины считается от длительности аир-кадра профиля —
# QUIET_FRAMES кадров подряд. Потолок держит ответ внутри окна ожидания
# отправителя (cli_common.RESPONSE_WAIT_S).
QUIET_FRAMES = 2.5
MAX_QUIET_GAP_S = 6.0


@functools.lru_cache(maxsize=None)
def air_frame_seconds(profile: str) -> float:
    """Длительность самого длинного аир-кадра профиля, секунды."""
    return len(modulate([bytes(PROFILES[profile].max_block)], profile)) / SAMPLE_RATE


class AudioEndpoint:
    """Контракт v1.2 через настоящий звук: динамик наружу, микрофон внутрь."""

    supports_calibration = True  # в отличие от спул-линка: тут есть спектр

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        player=None,
        source=None,
        quiet_gap_s: float = 2.5,
        poll_s: float = 0.05,
        stall_s: float = 5.0,
    ):
        self._player = player or _SounddevicePlayer(output_device)
        self._source = source if source is not None else _MicSource(input_device)
        self.quiet_gap_s = quiet_gap_s
        self.poll_s = poll_s
        self.stall_s = stall_s
        self._demodulators: dict[tuple[str, int], _Demodulator] = {}
        # Профиль → полосы-кандидаты, первая рабочая (ADR-0003).
        self._bands: dict[str, list[int]] = {}
        self._pos = 0  # сэмплов микрофона разобрано за всё время

    # --- полоса (калибровка, ADR-0003) -----------------------------------

    def set_band(self, profile: str, band: int) -> None:
        """Фиксирует рабочую полосу профиля по итогу калибровки.

        Приём остаётся и на полосе по умолчанию: подтверждение выбора
        могло не дойти до другой стороны, и та осталась на дефолтной —
        слушаем обе, пока первый же разобранный кадр не покажет, какая
        полоса реально в эфире.
        """
        if band not in range(len(bands(profile))):
            raise ValueError(f"полоса {band} вне диапазона профиля {profile!r}")
        self._bands[profile] = list(dict.fromkeys([band, default_band(profile)]))
        self._reset_demodulators()

    def band(self, profile: str) -> int:
        """Полоса, в которой этот конец сейчас передаёт."""
        return self._bands.get(profile, [default_band(profile)])[0]

    def send_probe(self, profile: str) -> None:
        """Играет зонд калибровки (ADR-0003)."""
        self._play(probe(profile))

    def measure_bands(self, profile: str, timeout_s: float) -> list[float] | None:
        """Слушает зонд и меряет SNR всех полос; None — зонда не было.

        Часы — аудио-время, как и в receive_blocks: решает то, что реально
        услышано, а не сколько прошло по настенным часам.
        """
        if profile not in PROFILES:
            raise ValueError(f"неизвестный профиль {profile!r}")
        self._source.start()
        start = self._pos
        heard: list[np.ndarray] = []
        total = 0
        fed_at = time.monotonic()
        timeout = int(timeout_s * SAMPLE_RATE)
        while True:
            chunk = self._source.get(self.poll_s)
            if chunk is not None:
                fed_at = time.monotonic()
                self._pos += len(chunk)
                heard.append(chunk)
                total += len(chunk)
                if total >= PROBE_TOTAL_SAMPLES:
                    snr = band_snr_db(np.concatenate(heard), profile)
                    if snr is not None:
                        return snr
                    while total - len(heard[0]) >= _PROBE_BUFFER_SAMPLES:
                        total -= len(heard.pop(0))
            elif self._source.exhausted:  # конец потока бывает только у фейков
                return band_snr_db(np.concatenate(heard), profile) if heard else None
            elif time.monotonic() - fed_at > self.stall_s:
                raise RuntimeError(
                    f"микрофон не отдаёт сэмплы дольше {self.stall_s:.0f} с — "
                    "проверьте устройство записи"
                )
            if self._pos - start >= timeout:
                return None

    # --- контракт v1.2 ---------------------------------------------------

    def send_blocks(self, blocks: list[bytes], profile: str) -> None:
        self._play(modulate(blocks, profile, band=self.band(profile)))

    def _play(self, samples: np.ndarray) -> None:
        pad = np.zeros(PLAY_PAD_SAMPLES, dtype=np.float32)
        self._player(np.concatenate([pad, samples, pad]))
        # Полудуплекс: всё, что микрофон записал во время нашего же
        # проигрывания, — своё эхо, выбрасываем вместе с недоразобранным.
        self._source.drain()
        self._reset_demodulators()

    def receive_blocks(self, profile: str, timeout_s: float) -> Iterator[bytes]:
        """Блоки из микрофона, пока партия не кончится.

        Часы — аудио-время (разобранные сэмплы), а не настенные: партия в
        воздухе длится долго, и решает то, что реально услышано. Конец:
        не было ни одного блока и прошло timeout_s тишины — либо после
        активности (блок или разбираемый кадр) прошло quiet_gap_s.
        """
        if profile not in PROFILES:
            raise ValueError(f"неизвестный профиль {profile!r}")
        self._source.start()
        demods = self._demods(profile)
        start = self._pos
        last_activity: int | None = None
        quiet = int(self.quiet_gap(profile) * SAMPLE_RATE)
        timeout = int(timeout_s * SAMPLE_RATE)
        fed_at = time.monotonic()
        while True:
            chunk = self._source.get(self.poll_s)
            if chunk is None and self._source.exhausted:
                for _, demod in demods:  # конец потока бывает только у фейков
                    yield from demod.flush()
                self._reset_demodulators()
                return
            if chunk is None and time.monotonic() - fed_at > self.stall_s:
                raise RuntimeError(
                    f"микрофон не отдаёт сэмплы дольше {self.stall_s:.0f} с — "
                    "проверьте устройство записи"
                )
            if chunk is not None:
                fed_at = time.monotonic()
                self._pos += len(chunk)
                locked = False
                for key, demod in demods:
                    for block in demod.feed(chunk):
                        last_activity = self._pos
                        locked = self._lock_band(*key) or locked
                        yield block
                if locked:  # лишние полосы-кандидаты отвалились
                    demods = self._demods(profile)
            if any(demod.frame_in_progress for _, demod in demods):
                last_activity = self._pos
            if last_activity is not None:
                if self._pos - last_activity >= quiet:
                    return
            elif self._pos - start >= timeout:
                return

    def quiet_gap(self, profile: str) -> float:
        """Сколько тишины считать концом партии в этом профиле.

        Один потерянный кадр — это quiet_gap_s тишины для приёмника; если
        порог короче кадра, приёмник объявит партию законченной посреди
        передачи, ответит NACK'ом поверх ещё идущего сигнала и потеряет всё,
        что звучало, пока он говорил. Отсюда порог в QUIET_FRAMES кадров.
        """
        return min(
            max(self.quiet_gap_s, QUIET_FRAMES * air_frame_seconds(profile)),
            MAX_QUIET_GAP_S,
        )

    def _lock_band(self, profile: str, band: int) -> bool:
        """Первый разобранный кадр показал полосу — кандидаты больше не нужны."""
        if profile == ANCHOR or self._bands.get(profile, []) == [band]:
            return False
        self._bands[profile] = [band]
        for key in [k for k in self._demodulators if k[0] == profile and k[1] != band]:
            del self._demodulators[key]
        return True

    def _demods(self, profile: str) -> list[tuple[tuple[str, int], _Demodulator]]:
        """Демодуляторы приёма: якорь (служебное) + полосы-кандидаты профиля."""
        keys = [(ANCHOR, default_band(ANCHOR))]
        keys += [(profile, b) for b in self._bands.get(profile, [default_band(profile)])]
        for key in dict.fromkeys(keys):
            if key not in self._demodulators:
                self._demodulators[key] = _Demodulator(banded(*key))
        return [(key, self._demodulators[key]) for key in dict.fromkeys(keys)]

    def _reset_demodulators(self) -> None:
        self._demodulators.clear()


class _SounddevicePlayer:
    """Блокирующее проигрывание через динамик (ленивый импорт sounddevice)."""

    def __init__(self, device: int | str | None):
        self._device = device

    def __call__(self, samples: np.ndarray) -> None:
        import sounddevice as sd

        sd.play(samples, SAMPLE_RATE, device=self._device, blocking=True)


class _MicSource:
    """Микрофон → очередь кусков; поток открывается лениво и живёт до конца."""

    exhausted = False  # настоящий микрофон не кончается

    def __init__(self, device: int | str | None):
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._on_audio,
        )
        self._stream.start()
        # Живой звук капризен к выбору входа — всегда показываем, что открыли.
        info = sd.query_devices(self._stream.device)
        api = sd.query_hostapis(info["hostapi"])["name"]
        print(f"Вход: [{self._stream.device}] {info['name']} ({api})", file=sys.stderr)

    def _on_audio(self, indata, frames, time_info, status) -> None:
        self._queue.put(indata[:, 0].copy())

    def get(self, wait_s: float) -> np.ndarray | None:
        try:
            if wait_s > 0:
                return self._queue.get(timeout=wait_s)
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> None:
        if self._stream is None:
            return
        time.sleep(0.15)  # хвост своего звука ещё летит динамик → микрофон
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
