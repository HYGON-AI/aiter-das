# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate compact recurrent verify against independent FP32 recurrence."""
import pytest
import torch

from aiter.ops.triton.fla.gdn_replayssm import gdn_replayssm_verify


@pytest.mark.parametrize('history_dot', [False, True])
@pytest.mark.parametrize('exact_raw', [False, True])
@pytest.mark.parametrize('graph', [False, True])
def test_gdn_replayssm_verify(history_dot, exact_raw, graph):
    torch.manual_seed(183)
    device = 'cuda'
    H, HV, D, L = 2, 6, 128, 16
    lengths = [1, 0, 4, 2, 3]
    B = len(lengths)
    M = sum(lengths)
    slots = 12

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device=device, dtype=dtype)*.1
    q, k = rand((M, H, D)), rand((M, H, D))
    v, a, b = rand((M, HV, D)), rand((M, HV)), rand((M, HV))
    al, bias = rand((HV,), torch.float32), rand((HV,), torch.float32)
    # Deliberately use padded slot strides, nonidentity replay mapping and
    # different state/ring indices. A null and an empty sequence are included.
    storage = rand((slots, 2, HV, D, D), torch.float32)
    state = storage[:, 0]
    checkpoint = state.clone()
    state_ids = torch.tensor([2, 4, 1, 0, 3], device=device, dtype=torch.int64)
    replay_ids = torch.tensor([7, 2, 9, 1, 4], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 1, 1, 5, 7, 10], device=device, dtype=torch.int32)
    pos = torch.full((slots,), 8, device=device, dtype=torch.int32)
    base = torch.arange(slots, device=device, dtype=torch.int32)
    flush = torch.zeros(slots, device=device, dtype=torch.int8)
    flush[9] = 1
    dc, kc = rand((slots, HV, L, D)), rand((slots, H, L, D))
    gc = -torch.rand((slots, HV, L), device=device)*.1
    rv, rk = torch.zeros_like(dc), torch.zeros_like(kc)
    bc = torch.zeros_like(gc) if exact_raw else None
    expected_d, expected_k, expected_g = dc.clone(), kc.clone(), gc.clone()
    expected_rv, expected_rk = rv.clone(), rk.clone()
    expected_bc = bc.clone() if bc is not None else None
    expected = torch.zeros_like(v)
    out = torch.full_like(v, float('nan'))
    start = 0
    for n, length in enumerate(lengths):
        si, ri = int(state_ids[n]), int(replay_ids[n])
        offset = int(base[ri])
        if si == 0:
            start += length
            continue
        h = checkpoint[si].clone()
        if not int(flush[ri]):
            for t in range(int(pos[ri])):
                p = (offset+t) % L
                keys = kc[ri, :, p].float().repeat_interleave(HV//H, 0)
                h = h*gc[ri, :, p].exp()[:, None, None] + \
                    dc[ri, :, p].float()[:, :, None]*keys[:, None, :]
        for t in range(length):
            i = start+t
            key = k[i].float()
            query = q[i].float()
            key = key/torch.sqrt(key.square().sum(-1, keepdim=True)+1e-6)
            query = query/torch.sqrt(query.square().sum(-1, keepdim=True)+1e-6)
            key_heads = key.repeat_interleave(HV//H, 0)
            query_heads = query.repeat_interleave(HV//H, 0)*D**-.5
            gate = -al.exp()*torch.nn.functional.softplus(a[i].float()+bias)
            beta = b[i].float().sigmoid()
            h = h*gate.exp()[:, None, None]
            update = (v[i].float()-(h*key_heads[:, None, :]).sum(-1))*beta[:, None]
            h = h+update[:, :, None]*key_heads[:, None, :]
            expected[i] = (h*query_heads[:, None, :]).sum(-1).to(v.dtype)
            p = (offset+int(pos[ri])+t) % L
            expected_d[ri, :, p] = update.to(dc.dtype)
            expected_k[ri, :, p] = key.to(kc.dtype)
            expected_g[ri, :, p] = gate
            if exact_raw:
                expected_rv[ri, :, p] = v[i]
                expected_rk[ri, :, p] = k[i]
                expected_bc[ri, :, p] = beta
            else:
                expected_rv[ri, :, p] = (update -
                                         expected_d[ri, :, p].float()).to(rv.dtype)
                expected_rk[ri, :, p] = (key-expected_k[ri, :, p].float()).to(rk.dtype)
        start += length

    def run():
        for flag in (False, True):
            gdn_replayssm_verify(q, k, v, a, b, al, bias, out, state, dc, kc, gc, rv, rk, bc,
                                 cu, state_ids, replay_ids, pos, base, flush, D**-.5, L, 4,
                                 is_flush_kernel=flag, history_dot=history_dot)
    run()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        capture.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(state, checkpoint, atol=0, rtol=0)
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(gc, expected_g, atol=2e-6, rtol=2e-5)
    if exact_raw:
        torch.testing.assert_close(rv, expected_rv, atol=0, rtol=0)
        torch.testing.assert_close(rk, expected_rk, atol=0, rtol=0)
        torch.testing.assert_close(bc, expected_bc, atol=2e-6, rtol=2e-5)
    else:
        # Test the represented value, rather than an isolated low part at a
        # BF16 rounding boundary where its sign legitimately changes.
        torch.testing.assert_close(
            dc.float() +
            rv.float(),
            expected_d.float() +
            expected_rv.float(),
            atol=2e-5,
            rtol=2e-3)
        torch.testing.assert_close(
            kc.float() +
            rk.float(),
            expected_k.float() +
            expected_rk.float(),
            atol=2e-6,
            rtol=2e-5)
