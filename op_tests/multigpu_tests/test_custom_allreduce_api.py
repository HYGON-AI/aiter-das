"""CAR collectives, fused operators, transport and input reuse contracts at TP2/4/8.

Select one --suite per torchrun invocation. Transport also supports --tp6.
"""
import argparse
from datetime import timedelta
import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.dist.device_communicators import custom_all_reduce as car
from aiter.ops import custom_all_reduce as ops
from aiter.ops import custom_all_reduce as native


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected an input validation error before launching a kernel")


def run_collectives(args):
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    ca = None
    cases = 0
    try:
        for register in (False, True):
            ca = CustomAllreduce(dist.group.WORLD, local, max_size=16 * 1024 * 1024,
                                 enable_register_for_capturing=register)
            assert not ca.disabled and ca.transport == args.transport
            if args.transport == "fabric":
                assert not ca.enable_register_for_capturing
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for shape, dim in [((world * 2, 64), 0), ((3, world * 16), -1),
                                   ((8, world * 3), -1), ((2, world * 3, 16), 1),
                                   ((8, world * 2, 3), 1), ((2, world * 2, 3, 8), 1)]:
                    base = torch.arange(torch.tensor(shape).prod().item(), device=local,
                                        dtype=torch.float32).reshape(shape).remainder(7) * .125
                    inp = (base + rank).to(dtype)
                    outshape = list(shape)
                    outshape[dim] //= world
                    out = torch.empty(outshape, device=local, dtype=dtype)
                    assert ca.should_custom_rs(inp, dim)
                    ca.custom_reduce_scatter(inp, out, dim=dim)
                    reference = sum((base + r).to(dtype).float() for r in range(world)).to(dtype)
                    torch.testing.assert_close(out, reference.chunk(world, dim=dim)[rank], rtol=0, atol=0)
                    with ca.capture():
                        # The warmup result is consumed; returning zeros/None without a write fails.
                        out.fill_(float('nan'))
                        ca.custom_reduce_scatter(inp, out, dim=dim)
                        torch.testing.assert_close(out, reference.chunk(world, dim=dim)[rank], rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            ca.custom_reduce_scatter(inp, out, dim=dim)
                    for i in range(args.repeats):
                        inp.copy_((base + rank + (i % 8) * .125).to(dtype))
                        graph.replay()
                        reference = sum((base + r + (i % 8) * .125).to(dtype).float() for r in range(world)).to(dtype)
                        torch.testing.assert_close(out, reference.chunk(world, dim=dim)[rank], rtol=0, atol=0, msg=f"RS register={register} dtype={dtype} shape={shape} dim={dim} replay={i} rank={rank}")
                    cases += 1
                    del graph
            for dtype in (torch.float32, torch.float16, torch.bfloat16,
                          torch.int16, torch.int32, torch.int64):
                for dim in (0, -1):
                    base = torch.arange(128, device=local).reshape(4, 32)
                    # int32 includes NaN bit patterns when viewed as fp32; int64
                    # includes high words that a numerical float cast would lose.
                    offset = (1 << 40) if dtype == torch.int64 else (0x7fc00000 if dtype == torch.int32 else 0)
                    def rank_input(r, i):
                        return (base + offset + r * 128 + i).to(dtype)
                    inp = rank_input(rank, 0)
                    expected = torch.cat([rank_input(r, 0) for r in range(world)], dim=dim)
                    out = torch.empty_like(expected)
                    assert ca.all_gather_unreg(inp, out=out, dim=dim) is out
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    with ca.capture():
                        torch.testing.assert_close(ca.custom_all_gather(inp, dim), expected, rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            result = ca.custom_all_gather(inp, dim)
                    for i in range(args.repeats):
                        inp.copy_(rank_input(rank, i % 8))
                        graph.replay()
                        expected = torch.cat([rank_input(r, i % 8) for r in range(world)], dim=dim)
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    cases += 1
                    del graph, result
            x = torch.ones((world * 2, 64), device=local)
            y = torch.empty((2, 64), device=local)
            assert not ca.should_custom_rs(x, 5)
            assert not ca.should_custom_rs(x.t(), 0)
            assert not ca.should_custom_rs(x.to(torch.int32), 0)
            assert not ca.should_custom_rs(torch.ones(2, 64), 0)
            expect_value_error(lambda: ca.reduce_scatter(x, torch.empty_like(x), 0))
            expect_value_error(lambda: ca.reduce_scatter(x, x[:2], 0))
            expect_value_error(lambda: ca.all_gather_unreg(x, out=y))
            expect_value_error(lambda: ca.all_gather_unreg(x.unsqueeze(0), dim=1))
            assert ca.custom_all_gather(x.unsqueeze(0), dim=1) is None
            big = torch.empty((world * 4, 64), device=local)
            expect_value_error(lambda: ca.all_gather_unreg(big[:4], out=big))
            # Original low-level ABI remains callable alongside the dimension entry.
            ops.reduce_scatter(ca._ptr, x, y, ca._pool['input'].data_ptr, ca._pool['input'].max_size)
            torch.testing.assert_close(y, torch.full_like(y, world), rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier()
            ca.close()
            ca.close()
            assert ca._ptr == 0 and not ca._pool._buffers and ca.disabled
            del ca
            ca = None
        for _ in range(args.close_repeats):
            ca = CustomAllreduce(dist.group.WORLD, local, max_size=1024 * 1024,
                                 enable_register_for_capturing=False)
            assert not ca.disabled
            x = torch.full((2, 64), rank + 1., device=local)
            out = ca.custom_all_reduce(x)
            assert out is not None
            torch.testing.assert_close(out, torch.full_like(out, world * (world + 1) / 2), rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier()
            ca.close()
            ca = None
        print('COMM_SYNC_CONTRACT_PASS ' + json.dumps(dict(rank=rank, world=world,
              cases=cases, replays=args.repeats, close_repeats=args.close_repeats,
              arch=torch.cuda.get_device_properties(local).gcnArchName,
              transport=args.transport, ag_rs_copy_in=ops.ag_rs_requires_copy_in())), flush=True)
    finally:
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def check(actual, expected, dtype, label):
    assert torch.isfinite(actual.float()).all(), label
    # Predeclared for reductions with production-dtype rounding. Quantization
    # error is tested separately from the unquantized norm and residual.
    tol = 0.025 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol, msg=label)


def expect_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected validation failure before launch")


def run_fusions(args):
    rank, local, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    count = 0
    ca = None
    modes = [False, True] if args.register == 'both' else [args.register == 'on']
    try:
        for register in modes:
            ca = CustomAllreduce(dist.group.WORLD, device, max_size=32 * 1024 * 1024,
                                 enable_register_for_capturing=register)
            assert not ca.disabled
            for dtype in (torch.float16, torch.bfloat16):
                # norm, Gemma, padded storage/output, packed strided input,
                # per-token mirror, per-group mirror and scale strides.
                cases = [('norm', 512, False), ('gemma', 2880, True),
                         ('pad', 2048, False), ('packed', 512, True),
                         ('token', 2048, True), ('token', 8192, False),
                         ('group', 2048, True), ('transpose', 8192, False),
                         ('token_split', 8192, False), ('group_split', 8192, False),
                         ('norm', 16384, False)]
                for kind, width, one_stage in cases:
                    tokens = 64 if kind.endswith("_split") else 4
                    kind = kind.removesuffix("_split")
                    storage_width = width + 16 if kind in ('pad', 'packed') else width
                    storage = torch.empty((tokens, storage_width), device=device, dtype=dtype)
                    inp = storage[:, :width] if kind == 'packed' else storage
                    base = (torch.arange(tokens * width, device=device).reshape(tokens, width) % 17 - 8).float() / 16
                    residual = torch.empty_like(base, dtype=dtype)
                    weight = ((torch.arange(width, device=device) % 7).float() / 8 + .5).to(dtype)
                    eps = 1e-6
                    gemma = kind == 'gemma'

                    def fill(iteration):
                        delta = (iteration % 4) / 16
                        storage.fill_(7)
                        storage[:, :width].copy_((base + rank / 8 + delta).to(dtype))
                        residual.fill_(.125 + (iteration % 3) / 16)
                        summed = sum((base + r / 8 + delta).to(dtype).float() for r in range(world)).to(dtype).float()
                        z = summed + residual.float()
                        norm = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
                        norm *= weight.float() + (1 if gemma else 0)
                        return z, norm

                    def invoke():
                        if kind in ('group', 'transpose'):
                            return ca.custom_fused_ar_rms_per_group_quant(inp, residual, weight, eps,
                                group_size=128, use_1stage=one_stage, emit_bf16=True,
                                transpose_scale=kind == 'transpose')
                        if kind == 'token':
                            return ca.custom_fused_ar_rms_quant(inp, residual, weight, eps,
                                use_1stage=one_stage, gemma_norm=False, emit_bf16=True)
                        if kind == 'packed':
                            return ca.custom_fused_ar_rms_packed_input(inp, residual, weight, eps,
                                use_1stage=one_stage)
                        return ca.custom_fused_ar_rms(inp, residual, weight, eps,
                            use_1stage=one_stage, out_hidden_dim=width + 32 if kind == 'pad' else 0,
                            gemma_norm=gemma)

                    def validate(outputs, z, norm, label):
                        assert outputs is not None, label
                        check(outputs[1], z.to(dtype), dtype, label + '/residual')
                        if kind in ('token', 'group', 'transpose'):
                            quant, _, scales, mirror = outputs
                            assert quant.dtype == torch.float8_e4m3fn and mirror.dtype == dtype
                            check(mirror, norm.to(dtype), dtype, label + '/mirror')
                            groups = 1 if kind == 'token' else width // 128
                            expected_scale = norm.reshape(tokens, groups, -1).abs().amax(-1) / 448
                            expected_scale = torch.where(expected_scale == 0, torch.ones_like(expected_scale), expected_scale)
                            check(scales, expected_scale, dtype, label + '/scale')
                            assert tuple(scales.shape) == (tokens, groups)
                            if kind == 'transpose':
                                assert scales.stride() == (1, tokens)
                            dequant = quant.float().reshape(tokens, groups, -1) * scales.unsqueeze(-1)
                            # One half FP8 step plus production dtype rounding.
                            torch.testing.assert_close(dequant.reshape_as(norm), norm,
                                                       rtol=.10, atol=.03, msg=label + '/dequant')
                        else:
                            check(outputs[0][:, :width], norm.to(dtype), dtype, label + '/norm')
                            if kind == 'pad':
                                assert outputs[0].shape == (tokens, width + 32)
                                assert torch.count_nonzero(outputs[0][:, width:]) == 0

                    z, norm = fill(0)
                    validate(invoke(), z, norm, f'eager/{kind}/{tokens}x{width}/{dtype}/{register}')
                    with ca.capture():
                        invoke()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            outputs = invoke()
                    for i in range(args.repeats):
                        z, norm = fill(i)
                        graph.replay()
                        validate(outputs, z, norm, f'graph/{kind}/{tokens}x{width}/{dtype}/{register}/{i}')
                    del graph, outputs
                    count += 1

                # Exact E2M1 ties, signs, packing order and E8M0 scale. A sum of
                # exactly one with eps=0 makes normalized values equal weight.
                width = 512
                values = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75,
                                       2, 2.5, 3, 3.5, 4, 5, 6, -6], device=device)
                weight = values.repeat(width // 16).to(dtype)
                x = torch.full((4, width), 1 / world, device=device, dtype=dtype)
                residual = torch.zeros_like(x)
                bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], device=device)
                codes = torch.bucketize(weight.float().abs(), bounds, right=True).to(torch.uint8)
                codes |= (weight < 0).to(torch.uint8) << 3
                packed = codes[::2] | (codes[1::2] << 4)
                for one_stage in (False, True):
                    result = ca.custom_fused_ar_rms_mxfp4_quant(x, residual, weight, 0.,
                        use_1stage=one_stage, emit_bf16=True)
                    assert result[0].dtype == result[2].dtype == torch.uint8
                    torch.testing.assert_close(result[0], packed.expand(4, -1), rtol=0, atol=0)
                    torch.testing.assert_close(result[2], torch.full_like(result[2], 127), rtol=0, atol=0)
                    check(result[3], weight.expand_as(x), dtype, 'mxfp4/mirror')
                    with ca.capture():
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            result = ca.custom_fused_ar_rms_mxfp4_quant(x, residual, weight, 0.,
                                use_1stage=one_stage, emit_bf16=True)
                    for i in range(args.repeats):
                        # Powers of two make rsqrt normalization exact at E2M1
                        # ties. For a sum of 3, rsqrtf(9)*3 is slightly below
                        # one, legitimately mapping every tie to the lower bin.
                        x.fill_((2 ** (i % 4)) / world)
                        graph.replay()
                        torch.testing.assert_close(result[0], packed.expand(4, -1), rtol=0, atol=0)
                        torch.testing.assert_close(result[2], torch.full_like(result[2], 127), rtol=0, atol=0)
                        check(result[1], torch.full_like(x, 2 ** (i % 4)), dtype, 'mxfp4/residual')
                        check(result[3], weight.expand_as(x), dtype, 'mxfp4/mirror')
                    del graph, result
                    count += 1
                # Zero input has zero E2M1 payload and the upstream E8M0 zero scale.
                x.zero_()
                zero = ca.custom_fused_ar_rms_mxfp4_quant(x, residual, weight, 1e-6, emit_bf16=True)
                assert all(torch.count_nonzero(t) == 0 for t in zero)
                # Exact E4M3FN encoding (ties-to-even, signs, subnormals, max).
                f8 = torch.tensor([0, 2**-10, 2**-9, 3*2**-10, .25, 1.0625, 1.1875,
                                   1.25, 16, 240, 448, -448, -1.0625, -2**-9, 2, -2], device=device)
                f8w = f8.repeat(width // 16).to(dtype)
                x.fill_(1 / world)
                f8out = ca.custom_fused_ar_rms_quant(x, residual, f8w, 0., use_1stage=True, emit_bf16=True)
                torch.testing.assert_close(f8out[0].view(torch.uint8),
                    f8w.to(torch.float8_e4m3fn).view(torch.uint8).expand_as(x), rtol=0, atol=0)
                torch.testing.assert_close(f8out[2], torch.ones_like(f8out[2]), rtol=0, atol=0)
                # Existing QK uses global statistics, but preserves local Q/K/V.
                qkv = torch.empty((4, 768), dtype=dtype, device=device)
                qw = torch.ones(256, dtype=dtype, device=device)
                kw = torch.full_like(qw, .5)
                qbase = (torch.arange(qkv.numel(), device=device).reshape(qkv.shape) % 13 - 6).float() / 8
                def fill_qk(i):
                    all_inputs = [(qbase + r / 8 + (i % 4) / 16).to(dtype) for r in range(world)]
                    qkv.copy_(all_inputs[rank])
                    ref = []
                    for start, w in ((0, qw), (256, kw)):
                        variance = torch.stack([t[:, start:start+256].float().square().mean(-1, keepdim=True) for t in all_inputs]).mean(0)
                        ref.append(qkv[:, start:start+256].float() * torch.rsqrt(variance + 1e-6) * w.float())
                    return (*ref, qkv[:, 512:])
                expected = fill_qk(0)
                for got, want in zip(ca.custom_fused_qknorm_ar(qkv, qw, kw, 1e-6), expected):
                    check(got, want, dtype, 'qk/eager')
                with ca.capture():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        qko = ca.custom_fused_qknorm_ar(qkv, qw, kw, 1e-6)
                for i in range(args.repeats):
                    expected = fill_qk(i)
                    graph.replay()
                    for got, want in zip(qko, expected):
                        check(got, want, dtype, 'qk/graph')
                del graph, qko
                count += 3
                # The exported native API must reject bad metadata before copy/launch.
                expect_error(lambda: native.fused_allreduce_rmsnorm(0, x, residual,
                    torch.empty_like(x), torch.empty_like(x), weight, 1e-6,
                    ca._pool["input"].data_ptr, ca._pool["input"].max_size, False))
                expect_error(lambda: native.fused_allreduce_rmsnorm(ca._ptr, x, residual,
                    torch.empty_like(x, dtype=torch.float32), torch.empty_like(x), weight, 1e-6,
                    ca._pool["input"].data_ptr, ca._pool["input"].max_size, False))
                expect_error(lambda: ca.fused_ar_rms(x, residual, w=weight, eps=1e-6, out_hidden_dim=32))
                expect_error(lambda: ca.fused_ar_rms_per_group_quant(x, residual, w=weight, eps=1e-6, group_size=31))
            torch.cuda.synchronize()
            dist.barrier()
            ca.close()
            ca = None
        print('COMM_FUSION_CONTRACT_PASS ' + json.dumps(dict(rank=rank, world=world,
            cases=count, replays=args.repeats, fused_copy_in=native.ag_rs_requires_copy_in(), arch=torch.cuda.get_device_properties(local).gcnArchName)), flush=True)
    finally:
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def run_transport(args):
    rank,local,world=[int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')]
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('gloo',timeout=timedelta(seconds=120))
    ca=None
    try:
        os.environ['AITER_AR_TRANSPORT']='auto'
        ca=car.CustomAllreduce(dist.group.WORLD,device,max_size=16*1024*1024,enable_register_for_capturing=False)
        assert not ca.disabled and ca.transport=='ipc'
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            x=torch.empty((4,512),device=device,dtype=dtype)
            x.fill_(rank+1)
            y=ca.custom_all_reduce(x)
            torch.testing.assert_close(y,torch.full_like(y,sum(range(1,world+1))),rtol=0,atol=0)
            with ca.capture():
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=ca.custom_all_reduce(x)
            for step in range(100):
                x.fill_(rank+1+(step%4)/8);graph.replay()
                torch.testing.assert_close(y,torch.full_like(y,sum(range(1,world+1))+world*(step%4)/8),rtol=0,atol=0)
            del graph,y
        torch.cuda.synchronize();dist.barrier();ca.close();ca.close();ca=None
        if args.tp6:
            print('CAR_TP6_PASS',rank,world,flush=True)
            return
        # Inject one failed rank at three native resource phases; all ranks
        # must leave initialization disabled and release their allocations.
        os.environ['AITER_AR_TRANSPORT']='ipc'
        for owner,method in ((car.IPCBufferPool,'create'),(car.IPCBufferPool,'get_local_meta'),(car.ops,'init_custom_ar')):
            original=getattr(owner,method)
            def injected(*a,**kw):
                if rank==0:raise RuntimeError('injected '+method)
                return original(*a,**kw)
            with patch.object(owner,method,injected):
                ca=car.CustomAllreduce(dist.group.WORLD,device,max_size=1024*1024)
            assert ca.disabled and ca._ptr==0
            ca.close();assert not ca._pool._buffers;ca=None
            dist.barrier()
            print('CAR_INIT_FAILURE_PASS',rank,method,flush=True)
        if world>2:
            with patch.object(car,'is_fully_connected',return_value=False):
                ca=car.CustomAllreduce(dist.group.WORLD,device,max_size=1024*1024)
            assert ca.disabled;ca.close();ca=None
        if args.fabric_disabled:
            os.environ['AITER_AR_TRANSPORT']='fabric'
            try:car.CustomAllreduce(dist.group.WORLD,device,max_size=1024*1024)
            except RuntimeError as exc:assert 'not compiled' in str(exc)
            else:raise AssertionError('disabled build admitted Fabric')
        print('TRANSPORT_CONTRACTS_PASS',json.dumps({'rank':rank,'world':world,
              'fabric_build':bool(car.ops.fabric_ar_available()),'copyin':bool(car.ops.ag_rs_requires_copy_in())}),flush=True)
    finally:
        if ca is not None:ca.close()
        dist.destroy_process_group()



def run_reuse_case(ca, args, dtype, numel, mode):
    rank, world = dist.get_rank(), dist.get_world_size()
    inp = torch.empty(numel, device=ca.device, dtype=dtype)
    seed = torch.empty_like(inp)
    # Dyadic fractions plus small integers are exactly representable in BF16.
    base = (torch.arange(numel, device=ca.device) % 8).to(dtype) / 8

    def chain():
        inp.copy_(seed)
        outputs = []
        for _ in range(args.depth):
            out = ca.custom_all_reduce(inp, use_new=not args.legacy)
            if out is None:
                raise RuntimeError("custom allreduce unexpectedly fell back")
            outputs.append(out)
            inp.add_(1)
        return outputs

    seed.copy_(base + rank)
    chain()
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with ca.capture():
            with torch.cuda.graph(graph):
                outputs = chain()
    mismatch = 0
    max_error = 0.0
    first = None
    for replay in range(args.repeats):
        shift = replay % 4
        seed.copy_(base + rank + shift)
        # Synchronize only before the chain, never between AR and input reuse.
        torch.cuda.synchronize()
        dist.barrier()
        if graph is None:
            outputs = chain()
        else:
            graph.replay()
        torch.cuda.synchronize()
        for step, out in enumerate(outputs):
            expected = (base.float() * world + world * (shift + step)
                        + world * (world - 1) / 2).to(dtype)
            count = int(torch.count_nonzero(out != expected))
            if count:
                mismatch += count
                max_error = max(max_error, float((out.float() - expected.float()).abs().max()))
                if first is None:
                    first = {"replay": replay, "step": step,
                             "actual_head": out[:8].float().cpu().tolist(),
                             "expected_head": expected[:8].float().cpu().tolist()}
    result = dict(rank=rank, world=world, dtype=str(dtype), numel=numel,
                  mode=mode, requested_registered=bool(args.registered),
                  registered=ca.enable_register_for_capturing,
                  legacy=args.legacy, repeats=args.repeats, depth=args.depth,
                  mismatches=mismatch, max_error=max_error, first_failure=first)
    print("AR_REUSE_RESULT " + json.dumps(result), flush=True)
    dist.barrier()
    return int(mismatch != 0)



def run_reuse(args):
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    os.environ["AITER_AR_TRANSPORT"] = "ipc"
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    ca = None
    try:
        ca = CustomAllreduce(dist.group.WORLD, local, max_size=8 * 1024 * 1024,
                             enable_register_for_capturing=True)
        assert not ca.disabled and ca.transport == "ipc"
        arch = torch.cuda.get_device_properties(local).gcnArchName.split(":")[0]
        if arch in ("gfx936", "gfx938"):
            assert not ca.enable_register_for_capturing
        # Exact dyadic sums detect peer reads racing the next staging copy.
        case_args = argparse.Namespace(registered=1, legacy=False,
                                       repeats=args.repeats, depth=8)
        failures = 0
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for numel in (2048, 12288, 98304, 1048576):
                for mode in ("eager", "graph"):
                    failures += run_reuse_case(ca, case_args, dtype, numel, mode)
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, failures)
        assert not any(counts), f"input reuse failures per rank: {counts}"
    finally:
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=['collectives', 'fusions', 'transport', 'reuse'])
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--close-repeats', type=int, default=20)
    parser.add_argument('--transport', choices=['ipc', 'fabric'], default='ipc')
    parser.add_argument('--register', choices=['off', 'on', 'both'], default='both')
    parser.add_argument('--tp6', action='store_true')
    parser.add_argument('--fabric-disabled', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2 or args.close_repeats < 1:
        parser.error('repeats must be >= 2 and close-repeats >= 1')
    if (args.tp6 or args.fabric_disabled) and args.suite != 'transport':
        parser.error('--tp6 and --fabric-disabled require --suite transport')
    if args.suite == 'collectives':
        os.environ['AITER_AR_TRANSPORT'] = args.transport
        run_collectives(args)
    elif args.suite == 'fusions':
        run_fusions(args)
    elif args.suite == 'reuse':
        run_reuse(args)
    else:
        run_transport(args)


if __name__ == '__main__':
    main()
