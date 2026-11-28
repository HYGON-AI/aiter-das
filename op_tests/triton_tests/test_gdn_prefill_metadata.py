# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import itertools

import pytest
import torch

import aiter.ops.triton.fla.sglang.chunk_delta_h as delta


@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('initial', ['indexed', 'dense', 'none'])
def test_prefill_metadata_state_and_graph(monkeypatch, inplace, initial):
    torch.manual_seed(183)
    lengths = [65, 127, 0, 70]
    H, HV, D = 2, 6, 128
    N = len(lengths)
    T = sum(lengths)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device='cuda', dtype=dtype)*.02
    k = rand((1, T, H, D))
    w = rand((1, T, HV, D))
    u = rand((1, T, HV, D))
    g = torch.empty((1, T, HV), device='cuda', dtype=torch.float32)
    start = 0
    chunks = []
    offsets = [0]
    for n, length in enumerate(lengths):
        for t in range(0, length, 64):
            size = min(64, length-t)
            g[:, start+t:start+t+size] = -.01 * \
                torch.arange(1, size+1, device='cuda')[None, :, None]
            chunks.append([n, t//64])
        offsets.append(len(chunks))
        start += length
    cu = torch.tensor(
        [0] +
        list(
            itertools.accumulate(lengths)),
        device='cuda',
        dtype=torch.int64)
    ci = torch.tensor(chunks, device='cuda', dtype=torch.int64)
    co = torch.tensor(offsets, device='cuda', dtype=torch.int64)
    ids = torch.tensor([2, -1, 1, 3], device='cuda',
                       dtype=torch.int32) if initial == 'indexed' else None
    storage = rand((5, 2, HV, D, D), torch.float32)
    pool = storage[:, 0] if initial != 'none' else None
    original = pool.clone() if pool is not None else None
    expected_pool = original.clone() if original is not None else None
    expected_h = torch.empty((1, len(chunks), HV, D, D), device='cuda', dtype=k.dtype)
    expected_v = torch.empty_like(u)
    start = 0
    chunk_index = 0
    for n, length in enumerate(lengths):
        slot = int(ids[n]) if ids is not None else n
        h = original[slot].clone() if original is not None and slot >= 0 else torch.zeros(
            (HV, D, D), device='cuda')
        for t in range(0, length, 64):
            size = min(64, length-t)
            sl = slice(start+t, start+t+size)
            expected_h[0, chunk_index] = h.to(k.dtype)
            wh = w[0, sl].float().transpose(0, 1)
            # The implementation splits K into 64-wide BF16 matrix products.
            rounded = h.to(k.dtype).float()
            pred = torch.matmul(wh[:, :, :64], rounded[:, :, :64].transpose(1, 2))
            pred += torch.matmul(wh[:, :, 64:], rounded[:, :, 64:].transpose(1, 2))
            new_v = u[0, sl].float().transpose(0, 1)-pred
            expected_v[0, sl] = new_v.transpose(0, 1).to(k.dtype)
            gates = g[0, sl].transpose(0, 1)
            last = gates[:, -1]
            weights = (last[:, None]-gates).exp()
            scaled = (new_v*weights[:, :, None]).to(k.dtype).float()
            keys = k[0, sl].float().repeat_interleave(HV//H, 1).permute(1, 0, 2)
            h = h*last.exp()[:, None, None]+torch.matmul(scaled.transpose(1, 2), keys)
            chunk_index += 1
        if inplace and expected_pool is not None and slot >= 0:
            expected_pool[slot] = h
        start += length

    def unexpected(*args, **kwargs):
        raise AssertionError('Precomputed metadata was discarded')
    monkeypatch.setattr(delta, 'prepare_chunk_indices', unexpected)
    monkeypatch.setattr(delta, 'prepare_chunk_offsets', unexpected)

    def run():
        if pool is not None:
            pool.copy_(original)
        return delta.chunk_gated_delta_rule_fwd_h(k, w, u, g=g, initial_state=pool,
                                                  initial_state_indices=ids, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co,
                                                  inplace_update=inplace, kernel_cfg={'BV': 32, 'num_warps': 4, 'num_stages': 1})
    run()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        h, v = run()
    capture.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(h, expected_h, atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(v, expected_v, atol=2e-4, rtol=2e-2)
    if pool is not None:
        torch.testing.assert_close(pool, expected_pool, atol=2e-5, rtol=2e-3)
