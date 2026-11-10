# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Single-launch correctness check for kpool_topk.

Inputs and the reference are built on CPU;
kpool_topk itself runs on the device.

Run: GPU_CHIP=sb python op_tests/test_kpool_topk.py
"""
import torch

from aiter import kpool_topk

POOL_SIZE = 16
GROUP_TOPK = 128
TOPK = POOL_SIZE * GROUP_TOPK
ROWS = 96  # 96 blocks -> one per CU on a 96-CU target
NUM_GROUPS = 3000  # > GROUP_TOPK: every row performs a real top-k
TAIL = 3  # token ids appended from seq_lens % POOL_SIZE
OUT_COLS = TOPK + POOL_SIZE - 1


def reference(score: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """CPU reference: top-k groups -> their pool tokens -> tail tokens, + offset."""
    out = torch.full((ROWS, OUT_COLS), -1, dtype=torch.int32)
    token_offsets = torch.arange(POOL_SIZE, dtype=torch.int32)
    for row in range(ROWS):
        groups = score[row].topk(GROUP_TOPK).indices.to(torch.int32)
        raw = (groups[:, None] * POOL_SIZE + token_offsets).reshape(-1)
        raw = torch.cat(
            [raw, NUM_GROUPS * POOL_SIZE + torch.arange(TAIL, dtype=torch.int32)]
        )
        out[row, : raw.numel()] = raw + offsets[row]
    return out


@torch.inference_mode()
def main() -> int:
    torch.manual_seed(42)

    score = torch.randn(ROWS, NUM_GROUPS, dtype=torch.float32)
    lengths = torch.full((ROWS,), NUM_GROUPS, dtype=torch.int32)
    offsets = torch.arange(ROWS, dtype=torch.int32) * 1024
    seq_lens = torch.full((ROWS,), NUM_GROUPS * POOL_SIZE + TAIL, dtype=torch.int32)

    out = kpool_topk(
        score.cuda(),
        lengths.cuda(),
        POOL_SIZE,
        TOPK,
        topk_indices_offset=offsets.cuda(),
        seq_lens=seq_lens.cuda(),
    ).cpu()

    props = torch.cuda.get_device_properties(0)
    print(
        f"device={props.name} cus={props.multi_processor_count} "
        f"grid={ROWS} pool_size={POOL_SIZE} group_topk={GROUP_TOPK} "
        f"topk={TOPK} groups_per_row={NUM_GROUPS} tail={TAIL}",
        flush=True,
    )

    ref = reference(score, offsets)
    if out.shape != ref.shape:
        raise AssertionError(f"shape mismatch: out={tuple(out.shape)} ref={tuple(ref.shape)}")

    # Token order inside a row is not part of the contract, so compare sorted rows.
    out_sorted = out.sort(dim=-1).values
    ref_sorted = ref.sort(dim=-1).values
    if not torch.equal(out_sorted, ref_sorted):
        row, col = (out_sorted != ref_sorted).nonzero()[0].tolist()
        raise AssertionError(
            f"mismatch at row={row} col={col}: "
            f"out={out_sorted[row, col].item()} ref={ref_sorted[row, col].item()}"
        )

    print("PASS kpool_topk", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
