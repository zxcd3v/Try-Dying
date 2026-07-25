"""Фаза калибровки полосы перед передачей файла (ADR-0003, тикет 07).

Порядок в эфире:

1. TX играет зонд (широкополосный, ~0.5 с).
2. RX меряет SNR всех полос одним FFT и шлёт лучшую кадром BAND — в robust
   на якорной полосе.
3. TX решает (услышал BAND → его полоса, не услышал → полоса по умолчанию)
   и объявляет решение ответным BAND, тоже в robust на якоре.

Третий шаг — сверх ADR: без него стороны знают о выборе несимметрично.
Полностью симметричными их не сделать (задача двух генералов: последний
кадр всегда может пропасть), поэтому остаток риска гасит эндпоинт — он
слушает и выбранную полосу, и полосу по умолчанию, пока первый разобранный
кадр не покажет, что реально в эфире (AudioEndpoint.set_band).

Служебный канал живёт в robust на якорной полосе и от исхода калибровки не
зависит: провалившееся рукопожатие стоит сеансу лишней пары секунд, но не
ломает ни HEADER, ни NACK-цикл.
"""

from dataclasses import dataclass

from modem.probe import best_band
from modem.profiles import ANCHOR, bands, default_band
from protocol import frames

BAND_REPEATS = 3  # кадр BAND дублируется, как и HEADER: терять его дорого
PROBE_ATTEMPTS = 2  # зонд повторяется, если приёмник промолчал


@dataclass(frozen=True)
class Calibration:
    """Итог калибровки для одной стороны — он же материал для CLI."""

    band: int
    snr_db: list[float] | None  # None у передатчика: меряет только приёмник
    confirmed: bool  # полоса согласована, а не взята по умолчанию
    reason: str  # почему так вышло, человеческими словами


def calibrate_sender(endpoint, mode: str, wait_s: float) -> Calibration:
    """Сторона TX: зонд → ждём выбор приёмника → объявляем решение."""
    skipped = _skip(endpoint, mode)
    if skipped is not None:
        return skipped
    for _ in range(PROBE_ATTEMPTS):
        endpoint.send_probe(mode)
        band = _listen_band(endpoint, mode, wait_s)
        if band is not None:
            endpoint.set_band(mode, band)
            _announce(endpoint, band)
            return Calibration(band, None, True, "полосу выбрал приёмник по зонду")
    band = default_band(mode)
    endpoint.set_band(mode, band)
    _announce(endpoint, band)
    return Calibration(
        band, None, False,
        f"приёмник не ответил на зонд за {wait_s:.0f} с ×{PROBE_ATTEMPTS} — "
        "полоса по умолчанию",
    )


def calibrate_receiver(
    endpoint, mode: str, probe_timeout_s: float, wait_s: float
) -> Calibration:
    """Сторона RX: слушаем зонд → меряем полосы → шлём выбор → ждём решение."""
    skipped = _skip(endpoint, mode)
    if skipped is not None:
        return skipped
    snr_db = endpoint.measure_bands(mode, probe_timeout_s)
    if snr_db is None:
        band = default_band(mode)
        endpoint.set_band(mode, band)
        return Calibration(
            band, None, False,
            f"зонд не услышан за {probe_timeout_s:.0f} с — полоса по умолчанию",
        )
    _announce(endpoint, best_band(snr_db))
    decided = _listen_band(endpoint, mode, wait_s)
    band = decided if decided is not None else default_band(mode)
    endpoint.set_band(mode, band)
    if decided is None:
        reason = (
            f"отправитель не подтвердил полосу за {wait_s:.0f} с — по умолчанию"
        )
    elif decided == best_band(snr_db):
        reason = "полоса выбрана по зонду и подтверждена отправителем"
    else:
        reason = "отправитель выбрал другую полосу"
    return Calibration(band, snr_db, decided is not None, reason)


def calibration_applies(endpoint, mode: str) -> bool:
    """Будет ли вообще рукопожатие — CLI по этому решает, что обещать."""
    return _skip(endpoint, mode) is None


def _skip(endpoint, mode: str) -> Calibration | None:
    """Калибровать нечего: якорь неподвижен, а спул-линк — не звук."""
    if mode == ANCHOR or len(bands(mode)) < 2:
        return Calibration(
            default_band(mode), None, False,
            f"профиль {mode} живёт на якорной полосе — калибровка не нужна",
        )
    if not getattr(endpoint, "supports_calibration", False):
        return Calibration(
            default_band(mode), None, False,
            "канал без спектра (спул-линк) — калибровка только с --audio",
        )
    return None


def _announce(endpoint, band: int) -> None:
    frame = frames.build_frame(frames.BAND, band, 0, bytes([band]))
    block = frames.encode_block(frame, ANCHOR)
    endpoint.send_blocks([block] * BAND_REPEATS, ANCHOR)


def _listen_band(endpoint, mode: str, wait_s: float) -> int | None:
    """Номер полосы из кадра BAND в robust; None — не дождались."""
    band = None
    for block in endpoint.receive_blocks(ANCHOR, timeout_s=wait_s):
        parsed = frames.decode_frame(block)
        if parsed is None:
            continue
        frame_type, _, _, payload = parsed
        if frame_type == frames.BAND and payload and payload[0] < len(bands(mode)):
            band = payload[0]
    return band
