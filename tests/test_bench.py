"""Тестовый стенд (тикет 10) — тесты на швах инструмента.

Швы: формула времени эфира против настоящего modulate, авто-прогон через
фейковый модем, CSV-таблица и её markdown-рендер, генератор файлов.
Внутрь NACK-цикла не заглядываем — он уже закрыт тестами протокола.
"""

import csv

import numpy as np

from modem.profiles import MAX_BLOCK, SAMPLE_RATE
from modem.tx import modulate
from tools.bench import (
    COLUMNS,
    air_seconds,
    append_row,
    drop_auto_rows,
    generate_payload,
    measure_auto,
    render_markdown,
    run_auto_transfer,
)


def test_air_seconds_matches_modulate_exactly():
    # Формула стенда обязана совпадать с настоящим сигналом байт-в-байт:
    # разъедутся — таблица отчёта начнёт врать про скорость.
    for profile in ("robust", "medium", "fast"):
        lens = [1, 17, MAX_BLOCK[profile]]
        blocks = [bytes(range(256))[:n] for n in lens]
        exact = len(modulate(blocks, profile)) / SAMPLE_RATE
        assert air_seconds(lens, profile) == exact


def test_clean_auto_transfer_succeeds():
    data = ("to be or not to be\n" * 100).encode()
    result = run_auto_transfer(data, "medium", seed=0)
    assert result.ok
    assert result.rounds >= 1
    assert result.air_s > 0
    assert result.wall_s > result.air_s  # развороты полудуплекса учтены


def test_lossy_auto_transfer_still_succeeds_via_nack():
    data = bytes(np.random.default_rng(3).integers(0, 256, 4_000, dtype=np.uint8))
    result = run_auto_transfer(data, "medium", block_loss=0.15, seed=1)
    assert result.ok
    assert result.retransmitted > 0


def test_measure_auto_builds_table_row():
    data = b"x" * 2_000
    row = measure_auto(
        data, "robust", file_label="текст 2 КБ", conditions="чисто", attempts=2
    )
    assert row["источник"] == "авто"
    assert row["профиль"] == "robust"
    assert row["размер, Б"] == 2_000
    assert row["успех, %"] == 100
    assert float(row["скорость, Б/с"]) > 0
    assert row["расстояние, см"] == "—"


def test_csv_append_and_markdown_render(tmp_path):
    csv_path = tmp_path / "results.csv"
    md_path = tmp_path / "results.md"
    row = {c: "—" for c in COLUMNS}
    row.update({"источник": "воздух", "профиль": "medium", "успех, %": 100})
    append_row(csv_path, row)
    append_row(csv_path, row)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["профиль"] == "medium"

    render_markdown(csv_path, md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "| " + " | ".join(COLUMNS) + " |" in text
    assert text.count("| воздух |") == 2


def test_rerun_auto_drops_stale_auto_rows_but_keeps_air_rows(tmp_path):
    # Повторный `auto` не должен плодить дубли: старые авто-строки уходят,
    # живые замеры по воздуху остаются.
    csv_path = tmp_path / "results.csv"
    for source in ("авто", "воздух", "авто"):
        row = {c: "—" for c in COLUMNS}
        row["источник"] = source
        append_row(csv_path, row)

    drop_auto_rows(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["источник"] for r in rows] == ["воздух"]

    drop_auto_rows(tmp_path / "нет-такого.csv")  # нет файла — просто нет работы


def test_generate_payload_text_compresses_random_does_not():
    import zlib

    text = generate_payload("текст", 10_000, seed=0)
    rand = generate_payload("случайные", 10_000, seed=0)
    assert len(text) == len(rand) == 10_000
    assert len(zlib.compress(text, 9)) < 0.2 * len(text)
    assert len(zlib.compress(rand, 9)) > 0.9 * len(rand)
