"""Отправить файл: `python send.py <файл> [--mode robust|medium|fast] [--audio]`.

Приёмник (`recv.py`) должен уже слушать: тот же `--link` (спул-демо) либо
воздух при `--audio` на обоих концах. Показывает прогресс
партий, скорость и ETA; в конце — вердикт: приёмник подтвердил DONE (значит,
SHA-256 на его стороне сошёлся) либо понятная ошибка.
"""

import argparse
import sys
import time
from pathlib import Path

from cli_common import (
    CALIBRATION_WAIT_S,
    RESPONSE_WAIT_S,
    ProgressLine,
    add_common_args,
    describe_calibration,
    die,
    fmt_bytes,
    fmt_eta,
    make_endpoint,
    merge_batches,
    slower_profile_hint,
)
from protocol.calibration import calibrate_sender, calibration_applies
from protocol.transfer import FileSender


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Отправка файла через аудиомодем (пока — спул-демо на одной машине)."
    )
    parser.add_argument("file", help="файл для отправки")
    add_common_args(parser)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        die(f"файл не найден: {path}", 2)
    data = path.read_bytes()
    try:
        sender = FileSender(data, path.name, args.mode)
    except ValueError as err:
        die(f"{err} — переименуйте файл короче", 2)
    endpoint = make_endpoint(args, "a")

    channel = "живой звук" if args.audio else f"линк: {args.link}"
    print(
        f"Отправка {path.name} ({fmt_bytes(len(data))}), профиль {args.mode}, "
        f"кадров: {sender.frames_total}, {channel}"
    )
    # Калибровка до данных: зонд → выбор полосы приёмником (ADR-0003).
    calibrate = not args.no_calibration
    if calibration_applies(endpoint, args.mode, enabled=calibrate):
        print("Калибровка: играю зонд…", flush=True)
    print(describe_calibration(args.mode, calibrate_sender(
        endpoint, args.mode, CALIBRATION_WAIT_S, enabled=calibrate
    )))
    progress = ProgressLine()
    start = time.monotonic()
    last_response = start
    # Партия большого файла звучит дольше любого разумного таймаута (21 КБ в
    # medium — это ~5 минут), а приёмник в это время молчит по определению:
    # полудуплекс. Своё проигрывание из молчания приёмника вычитаем, иначе
    # первый же не расслышанный NACK убивает сеанс вместо пинга.
    air_time = 0.0
    wire_bytes = 0
    hint_shown = False

    while not sender.done:
        if time.monotonic() - last_response - air_time > args.timeout:
            progress.finish()
            where = "слышит ли нас его микрофон" if args.audio else "на том же --link"
            die(
                f"приёмник не отвечает {args.timeout:.0f} с — "
                f"запущен ли recv.py, {where}?",
                1,
            )
        sending_since = time.monotonic()
        for profile, blocks in merge_batches(sender.outgoing()):
            endpoint.send_blocks(blocks, profile)
            wire_bytes += sum(len(b) for b in blocks)
        air_time += time.monotonic() - sending_since
        got_response = False
        for block in endpoint.receive_blocks("robust", timeout_s=RESPONSE_WAIT_S):
            sender.feed(block)
            got_response = True
        now = time.monotonic()
        if got_response:
            last_response = now
            air_time = 0.0
        elapsed = max(now - start, 1e-6)
        rate = sender.frames_sent / elapsed
        # pending известен из только что принятого NACK: сколько кадров добирать
        eta = sender.pending / rate if sender.pending and rate else None
        progress.update(
            f"кадров передано: {sender.frames_sent}/{sender.frames_total} "
            f"(повторно: {sender.retransmitted}) | "
            f"{fmt_bytes(wire_bytes / elapsed)}/с | {fmt_eta(eta)}"
        )
        if not hint_shown:
            hint = slower_profile_hint(
                args.mode, sender.frames_total, sender.retransmitted
            )
            if hint:
                hint_shown = True
                progress.finish()
                print(f"СОВЕТ: {hint}", file=sys.stderr)

    progress.finish()
    elapsed = max(time.monotonic() - start, 1e-6)
    print(
        f"УСПЕХ: приёмник подтвердил DONE (SHA-256 сошёлся) — "
        f"{fmt_bytes(len(data))} за {elapsed:.1f} с, "
        f"{fmt_bytes(len(data) / elapsed)}/с, повторных кадров: {sender.retransmitted}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        sys.exit(130)
