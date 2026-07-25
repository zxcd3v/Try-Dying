"""Тестовый стенд (тикет 10): файлы, авто-матрица, таблица результатов.

Запуск из корня репо:

  python tools/bench.py gen                 # тестовые файлы → out/bench/
  python tools/bench.py auto [--quick]      # авто-матрица → docs/bench/results.csv
  python tools/bench.py table               # docs/bench/results.md из CSV
  python tools/bench.py manual --profile medium --file README.md --size 3400 --seconds 61.5 --distance 30 --noise "тихая комната" --ok 3 --tries 3

Авто-режим гоняет полный протокол (zlib → RS → NACK-цикл → SHA-256) через
фейковый модем по сетке условий. Время авто-строк — оценка эфира: точная
длительность всех партий обоих направлений (формула зеркалит tx и закрыта
тестом против modulate) плюс TURNAROUND_S на каждый разворот полудуплекса
(тишина quiet_gap_s приёмника + разгон звуковой карты). Ручные замеры по
воздуху пишутся в ту же таблицу командой manual.
"""

import argparse
import csv
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modem.framing import HEADER_COPY_BYTES
from modem.profiles import PROFILES, SAMPLE_RATE
from protocol.fake_modem import FakeLink
from protocol.transfer import FileReceiver, FileSender

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "docs" / "bench" / "results.csv"
RESULTS_MD = ROOT / "docs" / "bench" / "results.md"
FILES_DIR = ROOT / "out" / "bench"

# Разворот полудуплекса: приёмник ждёт quiet_gap_s (2.5 с) тишины, прежде
# чем счесть партию законченной и ответить, плюс пады разгона карты (0.4 с).
TURNAROUND_S = 3.0

COLUMNS = (
    "источник", "профиль", "файл", "размер, Б", "шум/условия",
    "расстояние, см", "время, с", "скорость, Б/с", "успех, %",
)

_TEXT = (
    "Аудиомодем передаёт файл между двумя ноутбуками только через звук: "
    "динамик поёт MFSK-тонами, микрофон слушает, Reed-Solomon чинит биты, "
    "NACK-цикл добирает погибшие кадры, SHA-256 подводит итог. "
)


# --- время эфира -------------------------------------------------------------

def air_seconds(block_lens: list[int], profile_name: str) -> float:
    """Точная длительность партии блоков в эфире (зеркало tx.modulate)."""
    p = PROFILES[profile_name]
    total = 0
    for block_len in block_lens:
        data_symbols = p.symbols_for_bytes(block_len)
        inserts = (data_symbols - 1) // p.pilot_every if data_symbols else 0
        symbols = (
            p.symbols_for_bytes(HEADER_COPY_BYTES * p.header_repeats)
            + data_symbols
            + inserts * len(p.pilot_pattern)
        )
        total += (
            p.preamble_samples + p.preamble_gap_samples
            + symbols * p.slot_samples + p.frame_gap_samples
        )
    return total / SAMPLE_RATE


# --- авто-прогон через фейковый модем ---------------------------------------

@dataclass(frozen=True)
class RunResult:
    ok: bool
    air_s: float     # чистый эфир: сигнал обеих сторон
    wall_s: float    # эфир + развороты полудуплекса
    rounds: int
    retransmitted: int


def run_auto_transfer(
    data: bytes,
    mode: str,
    *,
    bit_flip: float = 0.0,
    block_loss: float = 0.0,
    seed: int = 0,
    max_rounds: int = 50,
) -> RunResult:
    """Полный протокольный обмен через фейковый модем со счётом эфира."""
    link = FakeLink(bit_flip_prob=bit_flip, block_loss_prob=block_loss, seed=seed)
    sender = FileSender(data, "bench.bin", mode)
    receiver = FileReceiver()
    air = 0.0
    turnarounds = 0
    rounds = 0
    while not sender.done and rounds < max_rounds:
        rounds += 1
        for profile, blocks in sender.outgoing():
            air += air_seconds([len(b) for b in blocks], profile)
            link.a.send_blocks(blocks, profile)
        turnarounds += 1
        for block in link.b.receive_blocks(mode, timeout_s=0.0):
            receiver.feed(block)
        try:
            responses = receiver.response()
        except (ValueError, zlib.error):
            break  # SHA-256 не сошёлся — попытка провалена
        for profile, blocks in responses:
            air += air_seconds([len(b) for b in blocks], profile)
            link.b.send_blocks(blocks, profile)
        turnarounds += 1
        for block in link.a.receive_blocks("robust", timeout_s=0.0):
            sender.feed(block)
    return RunResult(
        ok=sender.done and receiver.ok and receiver.data == data,
        air_s=air,
        wall_s=air + turnarounds * TURNAROUND_S,
        rounds=rounds,
        retransmitted=sender.retransmitted,
    )


def measure_auto(
    data: bytes,
    mode: str,
    *,
    file_label: str,
    conditions: str,
    bit_flip: float = 0.0,
    block_loss: float = 0.0,
    attempts: int = 3,
) -> dict:
    """Ячейка матрицы: attempts попыток с разными сидами → строка таблицы."""
    results = [
        run_auto_transfer(
            data, mode, bit_flip=bit_flip, block_loss=block_loss, seed=seed
        )
        for seed in range(attempts)
    ]
    succeeded = [r for r in results if r.ok]
    wall = sum(r.wall_s for r in succeeded) / len(succeeded) if succeeded else 0.0
    return {
        "источник": "авто",
        "профиль": mode,
        "файл": file_label,
        "размер, Б": len(data),
        "шум/условия": conditions,
        "расстояние, см": "—",
        "время, с": f"{wall:.1f}" if succeeded else "—",
        "скорость, Б/с": f"{len(data) / wall:.1f}" if succeeded else "—",
        "успех, %": round(100 * len(succeeded) / len(results)),
    }


# --- таблица ------------------------------------------------------------------

def drop_auto_rows(csv_path: Path) -> None:
    """Перед новым auto: убрать прошлые авто-строки, сохранив замеры по воздуху."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        kept = [r for r in csv.DictReader(f) if r["источник"] != "авто"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(kept)


def append_row(csv_path: Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def render_markdown(csv_path: Path, md_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [
        "# Таблица результатов (тикет 10)",
        "",
        "Авто-строки: полный протокол через фейковый модем; время — оценка "
        "эфира (точная длительность сигнала + развороты полудуплекса по "
        f"{TURNAROUND_S:.0f} с). Строки «воздух» — живые замеры между двумя "
        "ноутбуками (`python tools/bench.py manual …`).",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    lines += ["| " + " | ".join(r[c] for c in COLUMNS) + " |" for r in rows]
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- тестовые файлы -----------------------------------------------------------

def generate_payload(kind: str, size: int, seed: int = 0) -> bytes:
    """«текст» — сжимаемый, «случайные» — худший случай (несжимаемый)."""
    if kind == "текст":
        body = (_TEXT.encode("utf-8") * (size // len(_TEXT.encode("utf-8")) + 1))
        return body[:size]
    if kind == "случайные":
        rng = np.random.default_rng(seed)
        return bytes(rng.integers(0, 256, size, dtype=np.uint8))
    raise ValueError(f"неизвестный тип файла {kind!r}")


_GEN_SET = [
    ("текст", "text", (1_024, 10_240, 102_400, 1_048_576)),
    ("случайные", "rand", (1_024, 10_240, 102_400, 1_048_576, 10_485_760)),
]


def cmd_gen(_: argparse.Namespace) -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    for kind, stem, sizes in _GEN_SET:
        for size in sizes:
            name = f"{stem}_{_size_label(size)}".replace(" ", "") + (
                ".txt" if kind == "текст" else ".bin"
            )
            path = FILES_DIR / name
            path.write_bytes(generate_payload(kind, size))
            print(f"{path}  ({kind}, {_size_label(size)})")


def _size_label(size: int) -> str:
    return f"{size // 1_048_576} МБ" if size >= 1_048_576 else f"{size // 1024} КБ"


# --- команды ------------------------------------------------------------------

# Сетка авто-матрицы: условия (bit_flip, block_loss) × файлы. Уровни ошибок —
# те же, что демо-флаги CLI (README): точечные биты чинит RS, потерю кадров
# добирает NACK; сочетание — «худший случай» канала.
_CONDITIONS = [
    ("чисто", 0.0, 0.0),
    ("биты 0.2%", 0.002, 0.0),
    ("потеря кадров 10%", 0.0, 0.10),
    ("биты 0.2% + потеря 10%", 0.002, 0.10),
]
_FILES = [
    ("текст 10 КБ", "текст", 10_240),
    ("случайные 1 КБ", "случайные", 1_024),
    ("случайные 10 КБ", "случайные", 10_240),
    ("случайные 100 КБ", "случайные", 102_400),
    ("случайные 1 МБ", "случайные", 1_048_576),
]


def cmd_auto(args: argparse.Namespace) -> None:
    attempts = 1 if args.quick else 3
    files = _FILES[:2] if args.quick else _FILES
    drop_auto_rows(RESULTS_CSV)
    for mode in ("robust", "medium", "fast"):
        for file_label, kind, size in files:
            data = generate_payload(kind, size)
            for cond_label, bit_flip, block_loss in _CONDITIONS:
                row = measure_auto(
                    data, mode,
                    file_label=file_label, conditions=cond_label,
                    bit_flip=bit_flip, block_loss=block_loss, attempts=attempts,
                )
                append_row(RESULTS_CSV, row)
                print(
                    f"{mode:7} | {file_label:16} | {cond_label:23} | "
                    f"{row['скорость, Б/с']:>8} Б/с | успех {row['успех, %']}%"
                )
    render_markdown(RESULTS_CSV, RESULTS_MD)
    print(f"\nТаблица: {RESULTS_CSV}\n         {RESULTS_MD}")


def cmd_manual(args: argparse.Namespace) -> None:
    speed = args.size / args.seconds if args.seconds else 0.0
    append_row(RESULTS_CSV, {
        "источник": "воздух",
        "профиль": args.profile,
        "файл": args.file,
        "размер, Б": args.size,
        "шум/условия": args.noise,
        "расстояние, см": args.distance,
        "время, с": f"{args.seconds:.1f}",
        "скорость, Б/с": f"{speed:.1f}",
        "успех, %": round(100 * args.ok / args.tries),
    })
    render_markdown(RESULTS_CSV, RESULTS_MD)
    print(f"Строка добавлена: {RESULTS_CSV}")


def cmd_table(_: argparse.Namespace) -> None:
    render_markdown(RESULTS_CSV, RESULTS_MD)
    print(f"Таблица: {RESULTS_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gen", help="сгенерировать тестовые файлы в out/bench/")

    auto = sub.add_parser("auto", help="авто-матрица через фейковый модем")
    auto.add_argument("--quick", action="store_true",
                      help="быстрый дым: 1 попытка, малые файлы")

    manual = sub.add_parser("manual", help="добавить живой замер по воздуху")
    manual.add_argument("--profile", required=True, choices=list(PROFILES))
    manual.add_argument("--file", required=True, help="что передавали")
    manual.add_argument("--size", required=True, type=int, help="размер, байт")
    manual.add_argument("--seconds", required=True, type=float,
                        help="время передачи, с (из вердикта send.py)")
    manual.add_argument("--distance", required=True, help="расстояние, см")
    manual.add_argument("--noise", required=True,
                        help="условия шума (например: тихая комната)")
    manual.add_argument("--ok", required=True, type=int, help="успешных попыток")
    manual.add_argument("--tries", required=True, type=int, help="всего попыток")

    sub.add_parser("table", help="перерендерить markdown-таблицу из CSV")

    args = parser.parse_args()
    {"gen": cmd_gen, "auto": cmd_auto, "manual": cmd_manual, "table": cmd_table}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
