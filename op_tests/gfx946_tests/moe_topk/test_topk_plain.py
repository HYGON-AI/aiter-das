# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""
Minimal perf model (gfx946) correctness check for topk_plain

Each case calls the operator exactly once on a tiny input. Inputs are built on
CPU and the reference/comparisons run on CPU, so only topk_plain is simulated.

Run: GPU_CHIP=sb python op_tests/gfx946_tests/test_topk_plain.py
"""
import torch
import aiter
from aiter import dtypes
from aiter.ops.topk_plain import topk_plain

torch.manual_seed(0)

BATCH, HIDDEN, TOPK = 960, 512, 64


def run_case(dtype, largest):
    # Distinct values (|k| <= 256 is exact in bf16/fp16), so values and ids are unique.
    x = torch.stack([torch.randperm(HIDDEN) for _ in range(BATCH)]) - HIDDEN // 2
    x = x.to(dtype)

    topk_ids = torch.zeros((BATCH, TOPK), dtype=dtypes.i32, device="cuda")
    topk_value = torch.zeros((BATCH, TOPK), dtype=dtype, device="cuda")
    empty = torch.tensor([], dtype=torch.int32, device="cuda")
    topk_plain(x.cuda(), topk_ids, topk_value, TOPK, largest, empty, empty, -1, 1)
    ids = topk_ids.cpu().long()
    values = topk_value.cpu().float()

    assert ids.ge(0).all() and ids.lt(HIDDEN).all(), f"invalid ids:\n{ids}"
    assert ids.sort(dim=-1).values.diff(dim=-1).ne(0).all(), f"duplicate ids:\n{ids}"

    # Output order is not part of the contract; compare as sorted multisets.
    ref = x.float().topk(TOPK, dim=-1, largest=largest).values.sort(dim=-1).values
    got_values = values.sort(dim=-1).values
    got_from_ids = x.float().gather(1, ids).sort(dim=-1).values
    assert torch.equal(got_values, ref), f"value mismatch:\nout={got_values}\nref={ref}"
    assert torch.equal(got_from_ids, ref), f"id mismatch:\nx[ids]={got_from_ids}\nref={ref}"


CASES = [(dtypes.fp32, True), (dtypes.bf16, False)]

failures = []
for dtype, largest in CASES:
    name = f"topk_plain[{dtype}, largest={largest}, batch={BATCH}, hidden={HIDDEN}, topk={TOPK}]"
    try:
        run_case(dtype, largest)
        aiter.logger.info(f"{name} passed~")
    except AssertionError as e:
        aiter.logger.error(f"{name} failed!\n{e}")
        failures.append(name)

if failures:
    raise AssertionError(f"topk_plain correctness failures: {failures}")
