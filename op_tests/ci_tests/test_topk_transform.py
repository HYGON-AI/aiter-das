# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright 2023-2024 SGLang Team
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Original tests derive from sgl-kernel/tests/test_topk.py (SGLang, Apache-2.0).
# Hygon modifications: AITER integration, strict CPU-reference checks, mixed
# score distributions, and torch.compile regression coverage, licensed under MIT.
# See LICENSE.Apache-2.0 and LICENSE for the applicable terms.

"""三个 TopK 接口的正确性回归测试，统一在本文件运行。

测试范围：
    - 原有 112 项常规测试：覆盖不同 batch、序列长度和三个接口的调用模式。
    - test_topk_transform_strict 的 28 项测试：四种模式 × 七类输入，补充
      空行/短行、K 附近边界、长行、同分、窄分布、混合分布及非平凡索引映射。
      该子集在 CPU 上生成输入并计算参考结果，适合 gfx946 PMD；输入传输和
      待测算子在 GPU 上执行，输出取回 CPU 后检查。两个子集均检查精确结果，
      允许同分元素选取不同的合法索引，不要求 TopK 输出有序。

整体用法（在已配置 GPU/PMD、PyTorch 和 AITER 的环境中，从 AITER 仓库根目录运行）：
    # 全量正确性回归，共 141 项（含 torch.compile）；gfx936/gfx938 真机可使用此入口。
    python -m pytest -v op_tests/ci_tests/test_topk_transform.py

    # 仅运行 CPU-reference 严格子集，共 28 项；gfx946 PMD 建议使用此入口。
    python -m pytest -v op_tests/ci_tests/test_topk_transform.py::test_topk_transform_strict

    # 仅复现 K 附近边界的 plain 模式；其他参数组合也可按相同方式选择。
    python -m pytest -v "op_tests/ci_tests/test_topk_transform.py::test_topk_transform_strict[boundary-plain]"

    # 仅运行原有 112 项常规测试，或仅收集用例而不执行。
    python -m pytest -v op_tests/ci_tests/test_topk_transform.py -k "not strict"
    python -m pytest --collect-only -q op_tests/ci_tests/test_topk_transform.py

可增加 -s 查看运行输出，增加 --durations=0 查看每项测试耗时。测试耗时包含
数据准备、参考计算和可能的 JIT 编译，不能用作算子性能数据。
"""

from typing import Optional

import pytest
import torch
from aiter.ops.topk_transform import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)


def _ref_torch_impl(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()
        scores = []
        for i, (start, end) in enumerate(zip(ks, ke)):
            scores.append(score[i, start:end].unsqueeze(0))
        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices


def _ref_torch_transform_ragged_impl(
    score: torch.Tensor,
    seq_len: int,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)

    mask = indices != -1
    topk_indices_offset = topk_indices_offset.unsqueeze(1)
    return torch.where(mask, indices + topk_indices_offset, indices)


MAX_SEQ_LEN = 131072
# CPU-reference 严格子集使用当前接口支持的固定 TopK 大小。
K = 2048


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
):
    # 常规矩阵的有效长度均不小于 K，page table 为恒等映射。
    # 检查索引范围、唯一性及精确选中分数；同分元素允许使用不同索引。
    ours = indices_our.to(torch.int64)
    reference = indices_ref.to(torch.int64)
    if topk_indices_offset is not None:
        ours = ours - topk_indices_offset[:, None]
        reference = reference - topk_indices_offset[:, None]
    assert bool(torch.all((ours >= 0) & (ours < seq_len)))
    sorted_indices = ours.sort(dim=-1).values
    assert bool(torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]))
    if row_starts is not None:
        ours = ours + row_starts[:, None]
        reference = reference + row_starts[:, None]
    actual_values = score.gather(1, ours).sort(dim=-1).values
    expected_values = score.gather(1, reference).sort(dim=-1).values
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # 当前接口仅支持 K=2048。
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_kernel(bs: int, k: int, seq_len: int, has_row_starts: bool) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)
    indices_our = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    # TopK 输出无序，排序后统一校验。
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    assert_equal(score, indices_ref, indices_our, bs, k, seq_len, row_starts=row_starts)


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # 当前接口仅支持 K=2048。
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_topk_transform_kernel(bs: int, k: int, seq_len: int, mode: str) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # decode 每条序列只有一个 query，cu_seqlens_q 为 0 到 bs。
    # 此处 page table 为 arange，映射后的值等于原始 TopK 索引。
    if mode == "decode":
        step = 1
    else:
        step = 4 if bs % 4 == 0 else 1
    num_tokens = bs
    bs = bs // step

    if mode == "extend":
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(
        0, num_tokens + 1, step=step, dtype=torch.int32, device="cuda"
    )
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)

    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=seq_len,
        src_page_table=src_page_table,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )

    # TopK 输出无序，排序后统一校验。
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        row_starts=row_starts,
    )


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # 当前接口仅支持 K=2048。
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_transform_ragged_kernel(
    bs: int, k: int, seq_len: int, has_row_starts: bool
) -> None:
    # ragged 接口用于 prefill。
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # bs 表示 query token 数。
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    # row_starts 指定每行有效 KV 分数的起点。
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )

    dst_page_table_ref = _ref_torch_transform_ragged_impl(
        score=score,
        seq_len=seq_len,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )

    # TopK 输出无序，排序后统一校验。
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        seq_len,
        topk_indices_offset,
        row_starts=row_starts,
    )


def check_indices(
    score, output, lengths, starts, offsets=None, page_table=None, owners=None
):
    # score 及参考计算保留在 CPU；只取回待测算子的输出进行检查。
    output = output.cpu().to(torch.int64)
    assert output.shape == (len(lengths), K)
    for row, length in enumerate(lengths):
        count = min(length, K)
        values = output[row]
        # 短行必须恰好填充 K-count 个 -1，其余索引不能重复。
        assert int((values == -1).sum()) == K - count
        selected = values[values != -1]
        assert selected.unique().numel() == count, (row, "duplicate indices")
        if page_table is not None:
            # 测试页表采用每条序列不同的可逆映射：物理索引 = base + 7 * 局部索引。
            # 先验证映射值，再还原为行内索引，避免恒等页表掩盖 transform 错误。
            owner = owners[row]
            base = int(page_table[owner, 0])
            assert torch.all((selected - base) % 7 == 0)
            selected = (selected - base) // 7
        elif offsets is not None:
            # ragged 输出包含每行的 KV 基址，校验分数前先减去该偏移。
            selected = selected - int(offsets[row])
        assert torch.all((selected >= 0) & (selected < length)), (row, "index range")
        expected = score[row, starts[row] : starts[row] + length].topk(count).values
        actual = score[row, selected + starts[row]].sort(descending=True).values
        # 比较选中分数的多重集：允许同分索引不同，但不允许任何分数误差。
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["plain", "paged_decode", "paged_prefill", "ragged"])
@pytest.mark.parametrize("case", ["short", "boundary", "random", "long", "ties", "narrow", "mixed"])
@torch.inference_mode()
def test_topk_transform_strict(kind, case):
    # 覆盖短行填充、K 边界、一般变长、长行、同分以及候选桶溢出。
    lengths = {
        "short": [0, 1, 37, 2048],
        "boundary": [2047, 2048, 2049, 4096],
        "random": [4096, 8192, 16384, 3073],
        "long": [65536, 131072],
        "ties": [8192, 12288],
        "narrow": [8192, 16384],
        "mixed": [8192, 16384],
    }[case]
    batch = len(lengths)
    # decode 不传 row_starts；其他模式使用不同的非零起点。
    starts = [0 if kind == "paged_decode" else (i * 17 + 3) for i in range(batch)]
    width = max(s + n for s, n in zip(starts, lengths))
    generator = torch.Generator().manual_seed(42)
    storage = torch.randn(batch, width + 13, generator=generator)
    if case == "ties":
        storage.fill_(1.0)
    elif case in ("narrow", "mixed"):
        # 将分数集中到 1 附近，触发粗分桶候选过多时的精确选择路径。
        storage.mul_(1e-5).add_(1.0)
        if case == "mixed":
            # 阈值桶溢出前已有更大分数被选中：切换到 FP32 全行细分时，
            # 必须清空粗分桶的计数，避免重复选中大值并漏掉真正的 TopK。
            for row, start in enumerate(starts):
                storage[row, start : start + 512] = 2.0
    # 每行额外保留 13 个元素，使 stride(0) 大于有效列数；GPU 端保留同样布局。
    score = storage[:, :width]
    device_storage = storage.cuda()
    device_score = device_storage[:, :width]
    lens = torch.tensor(lengths, dtype=torch.int32).cuda()
    rows = None if kind == "paged_decode" else torch.tensor(starts, dtype=torch.int32).cuda()
    offsets = torch.arange(batch, dtype=torch.int32) * (width + 100) + 37
    pages = owners = None
    if kind == "plain":
        output = fast_topk_v2(device_score, lens, K, rows)
    elif kind == "ragged":
        output = fast_topk_transform_ragged_fused(device_score, lens, offsets.cuda(), K, rows)
    else:
        # decode 每行属于独立序列；prefill 每两行属于同一序列，检查 query 归属。
        owners = (
            list(range(batch)) if kind == "paged_decode" else [i // 2 for i in range(batch)]
        )
        nseq = max(owners) + 1
        pages = (
            torch.arange(width, dtype=torch.int32)[None, :] * 7
            + torch.arange(nseq, dtype=torch.int32)[:, None] * (width * 7 + 100)
            + 11
        )
        cu = list(range(batch + 1)) if kind == "paged_decode" else list(range(0, batch + 1, 2))
        output = fast_topk_transform_fused(
            device_score, lens, pages.cuda(), torch.tensor(cu, dtype=torch.int32).cuda(), K, rows
        )
    assert output.dtype == torch.int32
    check_indices(
        score, output, lengths, starts, offsets if kind == "ragged" else None, pages, owners
    )


@torch.inference_mode()
def test_topk_transform_ragged_compile():
    # eager 绕过 torch.library 的额外分发，但 torch.compile 仍须捕获已注册算子。
    # 使用轻量 FX backend 校验完整图和实际输出，不依赖 Inductor 编译环境。
    graphs = []

    def capture(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    generator = torch.Generator().manual_seed(43)
    storage = torch.randn(4, 8220, generator=generator)
    score = storage[:, :8210]
    lengths = [0, 2048, 4096, 8192]
    starts = [3, 5, 9, 13]
    offsets = torch.arange(4, dtype=torch.int32) * 10000 + 37
    compiled = torch.compile(
        fast_topk_transform_ragged_fused, backend=capture, fullgraph=True
    )
    output = compiled(
        storage.cuda()[:, :8210],
        torch.tensor(lengths, dtype=torch.int32).cuda(),
        offsets.cuda(),
        K,
        torch.tensor(starts, dtype=torch.int32).cuda(),
    )
    assert graphs and any(
        "aiter.fast_topk_transform_ragged_interface" in str(node.target)
        for graph in graphs for node in graph.graph.nodes
    )
    check_indices(score, output, lengths, starts, offsets)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
