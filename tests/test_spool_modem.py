"""Спул-модем — контракт v1.2 между двумя процессами через каталог (тикет 05).

Тесты дёргают только внешний шов: два конца на одном каталоге, блоки
send_blocks одного конца выходят из receive_blocks другого.
"""

import numpy as np
import pytest

from modem.profiles import MAX_BLOCK
from protocol.spool_modem import SpoolEndpoint


def test_blocks_cross_spool_in_order_both_directions(tmp_path):
    ep_a = SpoolEndpoint(tmp_path, "a")
    ep_b = SpoolEndpoint(tmp_path, "b")
    blocks = [bytes([i]) * (i + 1) for i in range(5)]

    ep_a.send_blocks(blocks, "robust")
    assert list(ep_b.receive_blocks("robust", timeout_s=0.0)) == blocks

    ep_b.send_blocks([b"resp"], "robust")
    assert list(ep_a.receive_blocks("robust", timeout_s=0.0)) == [b"resp"]


def test_blocks_consumed_once(tmp_path):
    ep_a = SpoolEndpoint(tmp_path, "a")
    ep_b = SpoolEndpoint(tmp_path, "b")

    ep_a.send_blocks([b"one"], "robust")
    assert list(ep_b.receive_blocks("robust", timeout_s=0.0)) == [b"one"]
    assert list(ep_b.receive_blocks("robust", timeout_s=0.0)) == []


def test_oversized_block_rejected(tmp_path):
    ep_a = SpoolEndpoint(tmp_path, "a")
    with pytest.raises(ValueError):
        ep_a.send_blocks([b"x" * (MAX_BLOCK["robust"] + 1)], "robust")


def test_bit_flips_corrupt_but_preserve_length(tmp_path):
    ep_a = SpoolEndpoint(tmp_path, "a", bit_flip_prob=0.05, seed=7)
    ep_b = SpoolEndpoint(tmp_path, "b")
    block = bytes(range(96))

    ep_a.send_blocks([block], "robust")
    (received,) = ep_b.receive_blocks("robust", timeout_s=0.0)
    assert len(received) == len(block)
    assert received != block


def test_block_loss_drops_blocks(tmp_path):
    ep_a = SpoolEndpoint(tmp_path, "a", block_loss_prob=1.0)
    ep_b = SpoolEndpoint(tmp_path, "b")

    ep_a.send_blocks([b"gone"], "robust")
    assert list(ep_b.receive_blocks("robust", timeout_s=0.0)) == []


def test_full_transfer_over_spool_with_errors(tmp_path):
    """NACK-цикл через спул: биты портятся, блоки теряются — хеш сходится."""
    from protocol.transfer import transfer

    class SpoolLink:
        a = SpoolEndpoint(tmp_path, "a", bit_flip_prob=0.002, seed=3)
        b = SpoolEndpoint(tmp_path, "b", block_loss_prob=0.15, seed=4)

    data = bytes(np.random.default_rng(1).integers(0, 256, 3_000, dtype=np.uint8))
    result = transfer(data, "demo.bin", "medium", SpoolLink)
    assert result.ok
    assert result.data == data
