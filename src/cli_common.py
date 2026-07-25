"""Общие кусочки CLI `send.py` / `recv.py`: конфиг, аргументы, прогресс, вердикты.

Два транспорта одним контрактом v1.2: по умолчанию — спул-модем (общий
каталог `--link`, демо на одной машине, тикет 05); с флагом `--audio` —
живой звук через динамик и микрофон (тикет 06).

Пути и прочие значения по умолчанию не зашиты в код: они живут в
`config/audiomodem.env` (образец — `config/example.env`), флаги командной
строки их перекрывают.
"""

import argparse
import functools
import os
import sys
import time
from pathlib import Path
from typing import NoReturn

from modem.audio import AudioEndpoint
from modem.profiles import bands
from protocol.calibration import Calibration
from protocol.spool_modem import SpoolEndpoint
from protocol.transfer import MODES

# Отправитель ждёт NACK/DONE после партии столько; дольше — пингует
# HEADER'ом. С запасом на живой звук: приёмник сначала выжидает свою
# тишину конца партии (AudioEndpoint.quiet_gap, до MAX_QUIET_GAP_S = 6 с),
# только потом начинает отвечать. Приёмник после успеха обязан дослушивать
# ДОЛЬШЕ этого срока (serve_done_after_success), иначе пинг прилетит в пустоту.
RESPONSE_WAIT_S = 20.0

# Сколько ждать кадр BAND от другой стороны в рукопожатии калибровки.
# Короче обмена данными: тут в эфире всего три коротких robust-кадра.
CALIBRATION_WAIT_S = 8.0

# --- конфиг (config/) ----------------------------------------------------
#
# Ни один путь, к которому CLI обращается при приёме и отправке, не зашит
# в код: значения приходят из файла конфига, а флаги перекрывают файл.
# Порядок поиска файла — $AUDIOMODEM_CONFIG, потом каталог config/ рядом с
# этим файлом (он же корень репозитория и для собранного audiomodem.py).

CONFIG_ENV_VAR = "AUDIOMODEM_CONFIG"
CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIG_NAMES = ("audiomodem.env", "example.env")

# Встроенные значения — последняя линия обороны: конфига может не быть
# (например, у одиночного `audiomodem.py`, унесённого из репозитория).
_FALLBACK = {
    "MODE": "robust",
    "LINK_DIR": "link",
    "OUTPUT_DIRECTORY": "received",
    "TIMEOUT_S": "120",
    "IN_DEVICE": "",
    "OUT_DEVICE": "",
}


def config_path() -> Path | None:
    """Файл конфига, который будет прочитан; None — конфига нет вовсе."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    for name in CONFIG_NAMES:
        candidate = CONFIG_DIR / name
        if candidate.is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def config() -> dict[str, str]:
    """Значения по умолчанию из конфига поверх встроенных.

    Ошибка в конфиге — человеческое сообщение и выход 2: разбираться в
    трассировке за минуту до демо никто не будет.
    """
    values = dict(_FALLBACK)
    path = config_path()
    if path is None:
        return values
    if not path.is_file():
        die(f"конфиг не найден: {path} (переменная {CONFIG_ENV_VAR})", 2)
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            die(f"{path}, строка {lineno}: ждали КЛЮЧ=значение, видим {raw.strip()!r}", 2)
        key = key.strip().upper()
        if key not in _FALLBACK:
            print(
                f"ВНИМАНИЕ: {path}, строка {lineno}: ключ {key} не используется",
                file=sys.stderr,
            )
            continue
        values[key] = value.strip().strip("\"'")
    if values["MODE"] not in MODES:
        die(f"{path}: MODE={values['MODE']!r}, а бывают только {', '.join(MODES)}", 2)
    try:
        float(values["TIMEOUT_S"])
    except ValueError:
        die(f"{path}: TIMEOUT_S={values['TIMEOUT_S']!r} — нужно число секунд", 2)
    return values


def _from_config(key: str) -> str | None:
    """Значение конфига; пустая строка означает «не задано»."""
    return config()[key] or None


def add_common_args(parser: argparse.ArgumentParser) -> None:
    conf = config()
    parser.add_argument(
        "--mode", choices=MODES, default=conf["MODE"],
        help=f"профиль передачи данных (из конфига: {conf['MODE']})",
    )
    parser.add_argument(
        "--link", default=conf["LINK_DIR"], metavar="DIR",
        help=f"каталог спул-модема, общий для send и recv "
             f"(из конфига: {conf['LINK_DIR']})",
    )
    parser.add_argument(
        "--timeout", type=float, default=float(conf["TIMEOUT_S"]), metavar="СЕК",
        help=f"сколько секунд ждать другую сторону (из конфига: {conf['TIMEOUT_S']})",
    )
    parser.add_argument(
        "--audio", action="store_true",
        help="живой звук: динамик и микрофон вместо спул-каталога",
    )
    parser.add_argument(
        "--in-device", default=_from_config("IN_DEVICE"), metavar="УСТР",
        help="устройство записи sounddevice: номер или подстрока имени",
    )
    parser.add_argument(
        "--out-device", default=_from_config("OUT_DEVICE"), metavar="УСТР",
        help="устройство вывода sounddevice: номер или подстрока имени",
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="без рукопожатия полосы: обе стороны на полосе по умолчанию "
             "(ставить на ОБЕИХ сторонах)",
    )
    parser.add_argument(
        "--bit-flip", type=float, default=0.0, metavar="P",
        help="демо помех (только спул): вероятность переворота каждого бита",
    )
    parser.add_argument(
        "--block-loss", type=float, default=0.0, metavar="P",
        help="демо помех (только спул): вероятность потери блока целиком",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="зерно генератора помех"
    )


def make_endpoint(args: argparse.Namespace, side: str) -> AudioEndpoint | SpoolEndpoint:
    if args.audio:
        return AudioEndpoint(
            input_device=_device(args.in_device),
            output_device=_device(args.out_device),
        )
    return SpoolEndpoint(
        args.link, side,
        bit_flip_prob=args.bit_flip,
        block_loss_prob=args.block_loss,
        seed=args.seed,
    )


def merge_batches(
    batches: list[tuple[str, list[bytes]]]
) -> list[tuple[str, list[bytes]]]:
    """Склеивает соседние партии одного профиля в одну.

    Каждый send_blocks в воздухе — отдельная непрерывная передача, а пауза
    между ними (открытие потока, пады) рискует прозвучать для приёмника как
    конец партии → ответ навстречу ещё идущим кадрам. В robust служебные и
    DATA-кадры одного профиля — склейка убирает паузу целиком.
    """
    merged: list[tuple[str, list[bytes]]] = []
    for profile, blocks in batches:
        if merged and merged[-1][0] == profile:
            merged[-1][1].extend(blocks)
        else:
            merged.append((profile, list(blocks)))
    return merged


def describe_calibration(mode: str, result: Calibration) -> str:
    """Строка о выбранной полосе для пользователя (тикет 07).

    Показываем не только выбор, но и SNR всех полос: на демо это главное
    доказательство, что полоса выбрана по измерению, а не наугад.
    """
    chosen = bands(mode)[result.band]
    line = f"Полоса: {chosen.label} — {result.reason}"
    if result.snr_db is None:
        return line
    measured = ", ".join(
        f"{band.label} {snr:+.0f} дБ"
        for band, snr in zip(bands(mode), result.snr_db)
    )
    return f"{line}\nSNR по полосам: {measured}"


def slower_profile_hint(
    mode: str, frames_total: int, retransmitted: int
) -> str | None:
    """Подсказка сменить профиль: канал явно не держит выбранный (тикет 09).

    Срабатывает, когда повторных кадров накопилось больше, чем кадров в
    файле, — каждый кадр в среднем ходил дважды, на живом канале это
    деградация, а не случайность. robust — самый живучий, ему совета нет.
    """
    slower = {"fast": "medium (или robust)", "medium": "robust"}.get(mode)
    if slower is None or retransmitted <= max(frames_total, 2):
        return None
    return (
        f"канал плохо держит {mode}: повторных кадров уже {retransmitted} "
        f"при {frames_total} кадрах файла — перезапустите обе стороны "
        f"с --mode {slower}"
    )


def serve_done_after_success(
    endpoint, receiver, mode: str, *, ping_wait_s: float, max_wait_s: float
) -> int:
    """Дослуживает DONE после успеха приёмника (тикет 12).

    Отправитель мог не расслышать DONE — тогда его HEADER-пинг придёт
    через RESPONSE_WAIT_S; уйти раньше нельзя, иначе он будет пинговать
    пустоту до своего таймаута. Слушаем окнами ping_wait_s (они обязаны
    быть длиннее цикла пинга): на любой пинг протокол сам отвечает
    очередным DONE, полная тишина в окне — отправитель услышал нас и
    ушёл. max_wait_s страхует от вечного цикла. Возвращает число повторов.
    """
    deadline = time.monotonic() + max_wait_s
    repeats = 0
    while time.monotonic() < deadline:
        got = list(endpoint.receive_blocks(mode, timeout_s=ping_wait_s))
        if not got:
            break
        for block in got:
            receiver.feed(block)
        for profile, blocks in receiver.response():
            endpoint.send_blocks(blocks, profile)
        repeats += 1
    return repeats


def _device(value: str | None) -> int | str | None:
    """Номер или подстрока имени устройства sounddevice."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def die(message: str, code: int) -> NoReturn:
    print(f"ОШИБКА: {message}", file=sys.stderr)
    sys.exit(code)


def fmt_bytes(n: float) -> str:
    for unit in ("Б", "КБ", "МБ"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ГБ"


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "ETA —"
    return f"ETA {seconds:.0f} с"


class ProgressLine:
    """Однострочный прогресс поверх stderr, затирается возвратом каретки."""

    def __init__(self):
        self._width = 0

    def update(self, text: str) -> None:
        pad = " " * max(0, self._width - len(text))
        print(f"\r{text}{pad}", end="", file=sys.stderr, flush=True)
        self._width = len(text)

    def finish(self) -> None:
        if self._width:
            print(file=sys.stderr, flush=True)
            self._width = 0
