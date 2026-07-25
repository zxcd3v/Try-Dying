"""Тесты протокола на внешнем шве: receive_файл(фейковый_модем(send_файл))).

Спека, Testing Decisions: протокол отлаживается целиком без звука через
фейковый модем (контракт v1.2) с искусственными ошибками. Внутренности
(формат кадров, RS, стейт-машина NACK) напрямую не тестируются.
"""

import hashlib
import random

import pytest

from protocol.fake_modem import FakeLink
from protocol.transfer import transfer


def _payload(n: int, seed: int = 1) -> bytes:
    """Случайные байты — худший случай для zlib (несжимаемые)."""
    return random.Random(seed).randbytes(n)


@pytest.mark.parametrize("mode", ["robust", "medium", "fast"])
def test_clean_roundtrip_sha256_matches(mode):
    data = _payload(5_000)
    result = transfer(data, "demo.bin", mode, FakeLink())
    assert result.ok
    assert result.name == "demo.bin"
    assert hashlib.sha256(result.data).digest() == hashlib.sha256(data).digest()


@pytest.mark.parametrize("mode", ["robust", "medium", "fast"])
def test_bit_errors_within_rs_power_fixed_without_retransmit(mode):
    data = _payload(3_000, seed=2)
    link = FakeLink(bit_flip_prob=0.002, seed=7)
    result = transfer(data, "noisy.bin", mode, link)
    assert result.ok
    assert result.data == data
    assert result.retransmitted == 0
    assert result.rounds == 1


def test_lost_and_unrepairable_blocks_recovered_by_nack_cycle():
    data = _payload(8_000, seed=3)
    # Потери целых блоков + очаги битовых ошибок сверх мощности RS:
    # часть кадров гибнет, NACK-цикл обязан добрать их до DONE.
    link = FakeLink(block_loss_prob=0.15, bit_flip_prob=0.008, seed=11)
    result = transfer(data, "lossy.bin", "medium", link)
    assert result.ok
    assert result.data == data
    assert result.rounds > 1
    assert result.retransmitted > 0


def test_control_frames_always_robust_regardless_of_data_mode():
    link = FakeLink()
    result = transfer(_payload(2_000, seed=4), "ctl.bin", "fast", link)
    assert result.ok
    sender_sends = [(profile, n) for end, profile, n in link.log if end == "a"]
    assert ("robust", 3) in sender_sends  # HEADER ×3 — отдельно, в robust
    assert any(profile == "fast" for profile, _ in sender_sends)  # DATA — в режиме
    # Всё, что шлёт приёмник (NACK/DONE), — только robust
    assert all(profile == "robust" for end, profile, _ in link.log if end == "b")


def test_single_header_copy_loss_not_fatal():
    # Первый блок в эфире — первая копия HEADER; глушим ровно её.
    link = FakeLink(drop_indices={0})
    result = transfer(_payload(1_000, seed=5), "hdr.bin", "robust", link)
    assert result.ok
    assert result.rounds == 1  # реплики ×3 покрыли потерю, NACK-цикл не понадобился
