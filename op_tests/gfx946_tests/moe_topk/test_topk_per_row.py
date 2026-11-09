# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""
Minimal perf model (gfx946) correctness check for topk_per_row (k=2048),
derived from op_tests/test_topk_per_row.py.

Each mode runs exactly once on two rows: prefill mixes a short row (identity
indices + -1 padding) with a row just over K; decode uses next_n=2 so the two
rows have different valid lengths. References and checks run on CPU.

Run: GPU_CHIP=sb python op_tests/gfx946_tests/test_topk_per_row.py
"""
import torch
import aiter
from aiter.ops.topk_per_row import TOPK, topk_per_row_decode, topk_per_row_prefill

SEQ_LEN = TOPK + 64


def check_row(logits_row, got, length, row):
    """got must select exactly the top-min(length, K) scores of logits_row[:length]."""
    got = got.long()
    valid = min(length, TOPK)
    selected = got[got >= 0]
    assert int((got == -1).sum()) == TOPK - valid, f"row={row}: padding mismatch"
    assert selected.numel() == valid, f"row={row}: expected {valid} indices, got {selected.numel()}"
    assert selected.unique().numel() == valid, f"row={row}: duplicate indices"
    assert int(selected.min()) >= 0 and int(selected.max()) < length, f"row={row}: index out of range"
    if length <= TOPK:
        assert torch.equal(got[:length], torch.arange(length)), f"row={row}: short row not identity"
        return
    expected = logits_row[:length].topk(TOPK).values
    actual = logits_row[selected].sort(descending=True).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_prefill():
    lengths = [48, SEQ_LEN] * 48
    logits = torch.randn(len(lengths), SEQ_LEN, generator=torch.Generator().manual_seed(0))
    row_starts = torch.zeros(len(lengths), dtype=torch.int32)
    row_ends = torch.tensor(lengths, dtype=torch.int32)

    got, values = topk_per_row_prefill(
        logits.cuda(), row_starts.cuda(), row_ends.cuda(), return_values=True
    )
    got, values = got.cpu(), values.cpu()
    assert got.shape == (len(lengths), TOPK)
    for i, length in enumerate(lengths):
        check_row(logits[i], got[i], length, i)
        valid = min(length, TOPK)
        gathered = logits[i].gather(0, got[i, :valid].long())
        torch.testing.assert_close(values[i, :valid], gathered, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_decode(batch=48, next_n=2):
    num_rows = batch * next_n
    logits = torch.randn(num_rows, SEQ_LEN, generator=torch.Generator().manual_seed(2))
    seq_lens = torch.full((batch,), SEQ_LEN, dtype=torch.int32)

    got = topk_per_row_decode(logits.cuda(), next_n, seq_lens.cuda()).cpu()
    assert got.shape == (num_rows, TOPK)
    for i in range(num_rows):
        row_end = SEQ_LEN - next_n + (i % next_n) + 1
        check_row(logits[i], got[i], row_end, i)


TESTS = {
    "topk_per_row[prefill, return_values]": test_prefill,
    "topk_per_row[decode, next_n=2]": test_decode,
}

failures = []
for name, fn in TESTS.items():
    try:
        fn()
        aiter.logger.info(f"{name} passed~")
    except AssertionError as e:
        aiter.logger.error(f"{name} failed!\n{e}")
        failures.append(name)

if failures:
    raise AssertionError(f"topk_per_row correctness failures: {failures}")
