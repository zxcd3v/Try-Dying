"""Принять файл: `python recv.py [--mode robust|medium|fast] [--audio] [--out DIR]`.

Слушает каталог `--link` (спул-демо) либо микрофон при `--audio`,
собирает кадры, после каждой партии отвечает
NACK/DONE. Показывает прогресс (кадры, скорость, ETA); в конце — однозначный
вердикт «УСПЕХ: SHA-256 совпал» и путь к принятому файлу.
"""

import argparse
import sys
import time
import zlib
from pathlib import Path

from cli_common import (
    CALIBRATION_WAIT_S,
    RESPONSE_WAIT_S,
    ProgressLine,
    add_common_args,
    config,
    describe_calibration,
    die,
    fmt_bytes,
    fmt_eta,
    make_endpoint,
    serve_done_after_success,
)
from protocol.calibration import calibrate_receiver, calibration_applies
from protocol.transfer import FileReceiver

_BATCH_WAIT_S = 2.0  # одно ожидание партии
# После успеха дослушиваем окнами длиннее цикла пинга отправителя
# (RESPONSE_WAIT_S): не расслышал DONE — переспросит, ответим ещё раз.
_DONE_LINGER_S = RESPONSE_WAIT_S + 3.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Приём файла через аудиомодем (пока — спул-демо на одной машине)."
    )
    add_common_args(parser)
    out_default = config()["OUTPUT_DIRECTORY"]
    parser.add_argument(
        "--out", default=out_default, metavar="DIR",
        help=f"куда положить принятый файл (из конфига: {out_default})",
    )
    args = parser.parse_args()

    receiver = FileReceiver()
    endpoint = make_endpoint(args, "b")
    listening = "микрофон" if args.audio else f"линк {args.link}"
    print(f"Приём: слушаю {listening}, профиль {args.mode}…")
    # Калибровка до данных: зонд отправителя → SNR полос → выбор (ADR-0003).
    # Ждать можем долго (отправитель ещё не запущен) — предупреждаем сразу.
    if calibration_applies(endpoint, args.mode):
        print("Калибровка: жду зонд отправителя…", flush=True)
    print(describe_calibration(args.mode, calibrate_receiver(
        endpoint, args.mode, args.timeout, CALIBRATION_WAIT_S
    )))
    progress = ProgressLine()
    start = time.monotonic()
    last_activity = start

    while True:
        got = list(endpoint.receive_blocks(args.mode, timeout_s=_BATCH_WAIT_S))
        now = time.monotonic()
        if not got:
            if now - last_activity > args.timeout:
                progress.finish()
                where = "слышно ли его" if args.audio else "на том же --link"
                die(
                    f"ни одного блока за {args.timeout:.0f} с — "
                    f"запущен ли send.py, {where}?",
                    1,
                )
            continue
        last_activity = now
        try:
            for block in got:
                receiver.feed(block)
            for profile, blocks in receiver.response():
                endpoint.send_blocks(blocks, profile)
        except (ValueError, zlib.error) as err:
            progress.finish()
            die(f"передача повреждена — {err}", 1)
        if receiver.ok:
            break  # DONE отправлен; вердикт сразу, дослушаем после него

        elapsed = max(now - start, 1e-6)
        total = receiver.frames_total
        eta = None
        if total and receiver.frames_received:
            rate = receiver.frames_received / elapsed
            eta = (total - receiver.frames_received) / rate
        progress.update(
            f"кадров принято: {receiver.frames_received}/{total or '?'} | "
            f"{fmt_bytes(receiver.bytes_received / elapsed)}/с | {fmt_eta(eta)}"
        )

    progress.finish()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(receiver.name).name
    out_path.write_bytes(receiver.data)
    elapsed = max(time.monotonic() - start, 1e-6)
    print(
        f"УСПЕХ: SHA-256 совпал — {receiver.name} "
        f"({fmt_bytes(len(receiver.data))}) за {elapsed:.1f} с, "
        f"{fmt_bytes(len(receiver.data) / elapsed)}/с"
    )
    print(f"Файл: {out_path}")

    # Тикет 12: не уходить раньше пинга отправителя — он мог не расслышать
    # DONE и тогда пинговал бы пустоту до своего таймаута.
    repeats = serve_done_after_success(
        endpoint, receiver, args.mode,
        ping_wait_s=_DONE_LINGER_S, max_wait_s=args.timeout,
    )
    if repeats:
        print(f"Отправитель переспрашивал: DONE повторён ×{repeats}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        sys.exit(130)
