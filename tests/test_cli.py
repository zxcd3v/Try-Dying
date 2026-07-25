"""CLI отправки и приёма — шов «две команды через спул-модем» (тикет 05).

Каждый тест запускает настоящие `recv.py` и `send.py` отдельными процессами
на общем каталоге-линке — ровно так, как их запустит эксперт по README.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_ENV = {**os.environ, "PYTHONUTF8": "1"}


def _run_pair(tmp_path, data, name="demo.bin", mode="robust", send_extra=()):
    src = tmp_path / name
    src.write_bytes(data)
    link = tmp_path / "link"
    out = tmp_path / "received"

    recv = subprocess.Popen(
        [sys.executable, str(ROOT / "recv.py"), "--mode", mode,
         "--link", str(link), "--out", str(out), "--timeout", "60"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", env=_ENV, cwd=ROOT,
    )
    time.sleep(1.0)  # приёмник должен успеть занять линк (README: recv первым)
    send = subprocess.Popen(
        [sys.executable, str(ROOT / "send.py"), str(src), "--mode", mode,
         "--link", str(link), "--timeout", "60", *send_extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", env=_ENV, cwd=ROOT,
    )
    send_out, send_err = send.communicate(timeout=90)
    recv_out, recv_err = recv.communicate(timeout=90)
    return send, recv, send_out + send_err, recv_out + recv_err, out / name


def test_clean_link_transfers_file_with_verdicts(tmp_path):
    data = b"to be or not to be\n" * 40
    send, recv, send_text, recv_text, received = _run_pair(tmp_path, data)

    assert send.returncode == 0, send_text
    assert recv.returncode == 0, recv_text
    assert "УСПЕХ" in send_text
    assert "УСПЕХ: SHA-256 совпал" in recv_text
    assert received.read_bytes() == data


def test_noisy_link_still_succeeds(tmp_path):
    """Полный прогон с ошибками: биты портятся, блоки теряются — успех."""
    data = bytes(np.random.default_rng(2).integers(0, 256, 2_000, dtype=np.uint8))
    send, recv, send_text, recv_text, received = _run_pair(
        tmp_path, data, mode="medium",
        send_extra=("--bit-flip", "0.002", "--block-loss", "0.1"),
    )

    assert send.returncode == 0, send_text
    assert recv.returncode == 0, recv_text
    assert "УСПЕХ: SHA-256 совпал" in recv_text
    assert received.read_bytes() == data


def test_band_choice_is_visible_to_the_user(tmp_path):
    """Тикет 07: обе стороны обязаны сказать, в какой полосе работают —
    на спул-линке полоса по умолчанию, но молчать об этом нельзя."""
    send, recv, send_text, recv_text, _ = _run_pair(
        tmp_path, b"band line\n" * 10, mode="medium"
    )

    assert send.returncode == 0, send_text
    for text in (send_text, recv_text):
        assert "Полоса: 2400–6900 Гц" in text
        assert "--audio" in text  # почему калибровки не было


def test_describe_calibration_shows_the_choice_and_every_band_snr():
    """Тикет 07: выбор полосы и измерение по всем полосам — в CLI."""
    from cli_common import describe_calibration
    from protocol.calibration import Calibration

    text = describe_calibration(
        "medium",
        Calibration(band=3, snr_db=[16.5, 21.8, 31.2, 38.2], confirmed=True,
                    reason="полосу выбрал приёмник по зонду"),
    )

    assert "Полоса: 4800–9300 Гц" in text
    assert "полосу выбрал приёмник по зонду" in text
    for label in ("1500–6000 Гц +16", "2400–6900 Гц +22", "4800–9300 Гц +38"):
        assert label in text

    # Без измерения (сторона отправителя) — только строка выбора.
    silent = describe_calibration(
        "medium", Calibration(band=1, snr_db=None, confirmed=False, reason="по умолчанию")
    )
    assert "SNR" not in silent


def test_missing_file_gives_human_error(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "send.py"), str(tmp_path / "нет.bin"),
         "--link", str(tmp_path / "link")],
        capture_output=True, encoding="utf-8", env=_ENV, cwd=ROOT, timeout=30,
    )
    assert proc.returncode == 2
    assert "ОШИБКА" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_audio_flag_builds_audio_endpoint(tmp_path):
    """--audio переключает CLI на живой звук тем же контрактом v1.2;
    железо при этом не трогается (sounddevice ленив)."""
    import argparse

    from cli_common import add_common_args, make_endpoint
    from modem.audio import AudioEndpoint

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--audio", "--in-device", "1", "--out-device", "Speakers"])
    assert isinstance(make_endpoint(args, "a"), AudioEndpoint)

    args = parser.parse_args(["--link", str(tmp_path / "link")])
    assert not isinstance(make_endpoint(args, "a"), AudioEndpoint)


def test_clean_link_transfers_file_in_fast(tmp_path):
    """Тикет 09: блоки fast (MAX_BLOCK 255) проходят весь CLI-конвейер."""
    data = bytes(np.random.default_rng(9).integers(0, 256, 3_000, dtype=np.uint8))
    send, recv, send_text, recv_text, received = _run_pair(
        tmp_path, data, mode="fast"
    )

    assert send.returncode == 0, send_text
    assert recv.returncode == 0, recv_text
    assert "УСПЕХ: SHA-256 совпал" in recv_text
    assert received.read_bytes() == data


def test_slower_profile_hint_triggers_only_on_heavy_retransmission():
    """Тикет 09: при деградации канала пользователю очевидно, что пора
    на профиль живучее; на честном канале подсказка молчит."""
    from cli_common import slower_profile_hint

    # Повторов больше, чем кадров в файле, — канал явно не держит.
    hint = slower_profile_hint("fast", frames_total=10, retransmitted=11)
    assert hint is not None and "--mode medium" in hint
    hint = slower_profile_hint("medium", frames_total=10, retransmitted=11)
    assert hint is not None and "--mode robust" in hint

    assert slower_profile_hint("fast", frames_total=10, retransmitted=10) is None
    assert slower_profile_hint("fast", frames_total=10, retransmitted=0) is None
    # Маленький файл: пара повторов — ещё не деградация.
    assert slower_profile_hint("fast", frames_total=1, retransmitted=2) is None
    # robust — уже самый живучий: совета нет при любых повторах.
    assert slower_profile_hint("robust", frames_total=10, retransmitted=100) is None


def test_merge_batches_glues_same_profile_batches():
    """Партии одного профиля склеиваются в одну непрерывную передачу:
    в воздухе пауза между send_blocks — риск ложного конца партии."""
    from cli_common import merge_batches

    merged = merge_batches(
        [("robust", [b"h"] * 3), ("robust", [b"d1", b"d2"]), ("medium", [b"x"])]
    )
    assert merged == [("robust", [b"h", b"h", b"h", b"d1", b"d2"]), ("medium", [b"x"])]


class _ScriptedEndpoint:
    """Фейковый эндпоинт: очередь входящих партий + журнал отправленного."""

    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []

    def receive_blocks(self, mode, timeout_s):
        return self.incoming.pop(0) if self.incoming else []

    def send_blocks(self, blocks, profile):
        self.sent.append((profile, blocks))


def _ok_receiver_and_ping():
    """FileReceiver, уже собравший файл, + HEADER-пинг того же трансфера."""
    from protocol.transfer import FileReceiver, FileSender

    sender = FileSender(b"payload", "f.bin", "robust")
    receiver = FileReceiver()
    for _, blocks in sender.outgoing():
        for block in blocks:
            receiver.feed(block)
    assert receiver.response()  # первый DONE ушёл, receiver.ok = True
    assert receiver.ok
    ping = sender.outgoing()  # без NACK'ов outgoing() = пинг HEADER'ом
    return receiver, ping[0][1]


def test_receiver_answers_done_again_on_sender_ping():
    """Тикет 12: отправитель не расслышал DONE и пингует HEADER'ом —
    приёмник обязан ещё раз ответить DONE, а не уйти (гонка «2 с тишины
    против пинга через RESPONSE_WAIT_S», живой прогон 2026-07-25)."""
    from cli_common import serve_done_after_success
    from protocol import frames

    receiver, ping_blocks = _ok_receiver_and_ping()
    endpoint = _ScriptedEndpoint([ping_blocks])  # пинг, дальше тишина

    repeats = serve_done_after_success(
        endpoint, receiver, "robust", ping_wait_s=0.0, max_wait_s=5.0
    )

    assert repeats == 1
    profile, blocks = endpoint.sent[-1]
    assert profile == "robust"
    frame_type, _, _, _ = frames.decode_frame(blocks[0])
    assert frame_type == frames.DONE


def test_receiver_leaves_after_quiet_linger_without_pings():
    from cli_common import serve_done_after_success

    receiver, _ = _ok_receiver_and_ping()
    endpoint = _ScriptedEndpoint([])  # тишина сразу

    repeats = serve_done_after_success(
        endpoint, receiver, "robust", ping_wait_s=0.0, max_wait_s=5.0
    )

    assert repeats == 0
    assert endpoint.sent == []


def test_bad_mode_gives_usage_error_not_traceback(tmp_path):
    src = tmp_path / "x.bin"
    src.write_bytes(b"hi")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "send.py"), str(src), "--mode", "turbo"],
        capture_output=True, encoding="utf-8", env=_ENV, cwd=ROOT, timeout=30,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
