# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""gfx946 correctness tests for the MHC pre stage-2 fused kernel (mhc_pre_big_fuse).

The device under test is the whole aiter.mhc_pre entry point; stage 2 is the part
under pilot. Inputs are generated on CPU, the reference is an independent CPU
float32 implementation (RMSNorm, sigmoid, 20-round Sinkhorn), and every case
calls the operator exactly once: no warmup, no timing loop.

Run inside the perf-model container with the PMD chip selector exported:
    export PMD_PATH=/opt/rocm/pmd GPU_CHIP=sb GPU_ARCHS=gfx946
    python op_tests/gfx946_tests/mhc/test_mhc_pre_big_fuse.py --suite core \
        --expect-xcds 1 --json ../../../test_mhc_pre/mhc_pre_big_fuse_sb_core.json
Suites: core (36 end-to-end), stage2core (36, stage-1 inputs fed from CPU),
stage2splitk (10, n_splits 4 and 32), postonly (4), tf32 (4), stress (1),
invalid (4), pathprobe (1).
--case accepts "group:index" or a comma separated list; --list-cases prints them.
The pilot fixes AITER_MHC_PRE_KERNEL=aiter so that stage 2 never lands on the
experimental tlstyle kernel; --suite pathprobe is the single exception and only
records which host function auto selected.
Numerical verdict: zero out-of-tolerance elements. layer_input is judged on the
bf16 bit pattern (same sign and both non-zero means |int16 delta| <= 1), the two
float32 outputs use an absolute tolerance.
"""

import argparse
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path

import torch

HC_MULT = 4
HC_MULT2 = HC_MULT * HC_MULT
HC_MULT3 = HC_MULT * 2 + HC_MULT2
BATCHES = (1, 32, 64, 128, 256, 512, 1024, 2048, 8192)
HIDDENS = (1280, 2560, 4096, 7168)
FP32_TOL = 1e-4
SUBNORMAL_FLOOR = 2.0 ** -133
PRE_ENV = ("AITER_MHC_PRE_KERNEL", "AITER_MHC_PRE_BIG_FUSE_KERNEL",
           "AITER_MHC_PRE_TILE_N", "AITER_MHC_PRE_GEMM_MMAC_K")


def fail_with_detail(message, record):
    """AssertionError that carries the measured dict, so failed cases keep numbers."""
    error = AssertionError(message)
    error.detail = record
    return error


def bf16_bits(value):
    """Bit pattern of a bf16 tensor as int16, for unit-in-the-last-place checks."""
    return value.view(torch.int16).to(torch.int32)


def compare_layer_input(ref, act):
    """Count elements violating the bf16 bit-pattern rule and report the worst gap.

    Rule, per the pilot plan: same sign and both sides normal means at most one bf16
    step apart; anything else (opposite signs, a zero, or a subnormal on either side)
    must either match bit-for-bit or stay within the subnormal floor 2**-133, so that
    the exponent-boundary case where a 1-step difference straddles the sign bit is not
    reported as a violation.
    """
    ref_bits, act_bits = bf16_bits(ref), bf16_bits(act)
    ref_f, act_f = ref.float(), act.float()
    normal = (ref_bits & 0x7F80) != 0
    normal &= (act_bits & 0x7F80) != 0
    same_sign = (ref_bits < 0) == (act_bits < 0)
    ulp = (ref_bits - act_bits).abs()
    abs_diff = (ref_f - act_f).abs()
    rel = abs_diff / ref_f.abs().clamp(min=1e-30)
    ok = torch.where(same_sign & normal, ulp <= 1,
                     (ref_bits == act_bits) | (abs_diff <= SUBNORMAL_FLOOR))
    bad = ~ok
    return dict(
        violations=int(bad.sum()),
        max_ulp=int(ulp.max()) if ulp.numel() else 0,
        ulp_2_count=int((ulp == 2).sum()),
        ulp_ge_3_count=int((ulp >= 3).sum()),
        max_abs_diff=float(abs_diff.max()),
        max_rel_diff=float(rel.max()),
        sign_flip=int(((ref_f * act_f) < 0).sum()),
        zero_side=int(((~normal) & (ref_bits != act_bits)).sum()),
    )


def compare_fp32(ref, act, tol):
    diff = (ref - act).abs()
    bad = ~(diff <= tol)
    rel = diff / ref.abs().clamp(min=1e-30)
    return dict(violations=int(bad.sum()), max_abs_diff=float(diff.max()),
                max_rel_diff=float(rel.max()))


def sinkhorn_row_deviation(comb_mix):
    """Max |column-sum - 1| over the last axis pair of the Sinkhorn output."""
    if comb_mix is None:
        return None
    return float((comb_mix.sum(dim=-2) - 1.0).abs().max())


def reference(residual, fn, hc_scale, hc_base, args, hc_head_only):
    """Independent CPU float32 implementation of the MHC pre operator."""
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    out = residual_flat @ fn.T
    mixes = out * (sqrsum.unsqueeze(-1) / fn.shape[-1] + args["rms_eps"]).rsqrt()
    if hc_head_only:
        hc_scale_expanded = hc_scale[0].expand(HC_MULT)
        mixes = mixes * hc_scale_expanded + hc_base
        pre_mix = mixes[:, :HC_MULT].sigmoid().unsqueeze(-1) + args["hc_pre_eps"]
        layer_input = (residual * pre_mix).sum(-2).bfloat16()
        return None, None, layer_input
    hc_scale_expanded = torch.cat([
        hc_scale[0].expand(HC_MULT),
        hc_scale[1].expand(HC_MULT),
        hc_scale[2].expand(HC_MULT2),
    ])
    mixes = mixes * hc_scale_expanded + hc_base
    pre_mix = mixes[:, :HC_MULT].sigmoid().unsqueeze(-1) + args["hc_pre_eps"]
    post_mix = mixes[:, HC_MULT:2 * HC_MULT].sigmoid() * args["hc_post_mult_value"]
    res_mix = mixes[:, 2 * HC_MULT:].view(-1, HC_MULT, HC_MULT)
    x = res_mix.softmax(-1) + args["hc_sinkhorn_eps"]
    x = x / (x.sum(-2, keepdim=True) + args["hc_sinkhorn_eps"])
    for _ in range(args["sinkhorn_repeat"] - 1):
        x = x / (x.sum(-1, keepdim=True) + args["hc_sinkhorn_eps"])
        x = x / (x.sum(-2, keepdim=True) + args["hc_sinkhorn_eps"])
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix.unsqueeze(-1), x, layer_input


def make_inputs(m, hidden_size, seed, hc_mult3=HC_MULT3):
    gen = torch.Generator().manual_seed(seed)
    residual = torch.randn(m, HC_MULT, hidden_size, generator=gen).bfloat16()
    fn = torch.randn(hc_mult3, HC_MULT * hidden_size, generator=gen)
    hc_scale = torch.randn((3,), generator=gen) * 0.1
    hc_base = torch.randn((hc_mult3,), generator=gen) * 0.1
    return residual, fn, hc_scale, hc_base


ARGS_FULL = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
                 hc_post_mult_value=1.0, sinkhorn_repeat=20)
ARGS_HEAD = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
                 hc_post_mult_value=1.0, sinkhorn_repeat=0)


def case(m, hidden, kind, **extra):
    c = dict(m=m, hidden_size=hidden, kind=kind)
    c.update(extra)
    return c


def cases(suite):
    if suite == "core":
        return [case(m, h, "core") for m in BATCHES for h in HIDDENS]
    if suite == "postonly":
        return [case(m, h, "postonly") for m, h in
                ((1, 1280), (32, 2560), (256, 4096), (1024, 7168))]
    if suite == "tf32":
        # Restricted to the m128 stage-1 regime (hidden 4096/7168, m > 512 for the
        # narrow hiddens), where the tf32 flag is handed to the kernel instead of
        # being applied on the host.
        return [case(m, h, "tf32") for m, h in
                ((256, 4096), (512, 7168), (1024, 4096), (2048, 7168))]
    if suite == "stress":
        return [case(65536, 1280, "stress")]
    if suite == "pathprobe":
        return [case(1024, 4096, "pathprobe")]
    if suite == "stage2core":
        return [case(m, h, "stage2core") for m in BATCHES for h in HIDDENS]
    if suite == "stage2splitk":
        return [case(m, h, "stage2splitk", n_splits=k)
                for k in (4, 32) for m, h in
                ((32, 4096), (32, 7168), (128, 7168), (1024, 4096), (2048, 7168))]
    if suite == "splitk":
        return [case(m, h, "splitk", n_splits=k) for k, m, h in
                ((4, 32, 4096), (8, 128, 7168), (16, 256, 1280), (32, 1024, 4096))]
    if suite == "invalid":
        return [case(32, 1280, "invalid", name=n) for n in PRE_ENV]
    return []


GROUPS = ("core", "postonly", "tf32", "stress", "invalid", "pathprobe",
          "stage2core", "stage2splitk", "splitk")


def select(all_cases, spec):
    """--case accepts "kind:index" pairs, bare indices, or whole group names."""
    if not spec:
        return all_cases
    wanted = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            group, index = token.split(":", 1)
            wanted.add((group, int(index)))
        elif token.isdigit():
            wanted.update((c["kind"], i) for i, c in enumerate(all_cases)
                          if i == int(token))
        else:
            wanted.update((token, i) for i, c in enumerate(all_cases)
                          if c["kind"] == token)
    return [c for i, c in enumerate(all_cases) if (c["kind"], i) in wanted]


def run_dut(aiter, residual, fn, hc_scale, hc_base, args, use_tf32):
    post, comb, layer = aiter.mhc_pre(
        residual.cuda(), fn.cuda(), hc_scale.cuda(), hc_base.cuda(),
        args["rms_eps"], args["hc_pre_eps"], args["hc_sinkhorn_eps"],
        args["hc_post_mult_value"], args["sinkhorn_repeat"], use_tf32,
    )
    torch.cuda.synchronize()
    return post.cpu(), comb.cpu(), layer.cpu()


def run_case(aiter, c):
    seed = 20261104 + c["m"] + c["hidden_size"]
    hc_head_only = c["kind"] == "postonly"
    mult3 = HC_MULT if hc_head_only else HC_MULT3
    residual, fn, hc_scale, hc_base = make_inputs(c["m"], c["hidden_size"], seed, mult3)
    args = ARGS_HEAD if hc_head_only else ARGS_FULL
    ref_post, ref_comb, ref_layer = reference(
        residual, fn, hc_scale, hc_base, args, hc_head_only)
    use_tf32 = c["kind"] == "tf32"
    out_post, out_comb, out_layer = run_dut(
        aiter, residual, fn, hc_scale, hc_base, args, use_tf32)
    record = dict(layer_input=compare_layer_input(ref_layer, out_layer))
    if not hc_head_only:
        record["post_mix"] = compare_fp32(ref_post, out_post, FP32_TOL)
        record["comb_mix"] = compare_fp32(ref_comb, out_comb, FP32_TOL)
        record["comb_column_sum_dev"] = sinkhorn_row_deviation(out_comb)
    bad = record["layer_input"]["violations"]
    for key in ("post_mix", "comb_mix"):
        if key in record:
            bad += record[key]["violations"]
    if bad:
        raise fail_with_detail(f"{bad} out-of-tolerance elements: {json.dumps(record)}", record)
    return record


def run_tf32_identical(aiter, c):
    """gfx946 has no tf32 stage-1 variant, so the flag must not change any output."""
    seed = 20261104 + c["m"] + c["hidden_size"]
    residual, fn, hc_scale, hc_base = make_inputs(c["m"], c["hidden_size"], seed)
    off = run_dut(aiter, residual, fn, hc_scale, hc_base, ARGS_FULL, False)
    on = run_dut(aiter, residual, fn, hc_scale, hc_base, ARGS_FULL, True)
    identical = all(torch.equal(a, b) for a, b in zip(off, on))
    if not identical:
        raise AssertionError("use_tf32 changed stage-2 outputs on gfx946")
    return dict(bit_identical=True)


def run_pathprobe(aiter, c):
    """Record only which host function auto selects; no numerical verdict."""
    residual, fn, hc_scale, hc_base = make_inputs(c["m"], c["hidden_size"], 7)
    os.environ["AITER_MHC_PRE_KERNEL"] = "auto"
    os.environ.pop("AITER_MHC_PRE_BIG_FUSE_KERNEL", None)
    run_dut(aiter, residual, fn, hc_scale, hc_base, ARGS_FULL, False)
    return dict(env_kernel="auto", note="see the [mhc_pre_trace] line above")


def stage1_reference(residual, fn, n_splits):
    """Split the stage-1 GEMM into n_splits partials on CPU, in float32.

    Returns (gemm_out_mul, gemm_out_sqrsum) exactly as the device-side stage 1
    would hand them to stage 2, so that a stage-2 failure cannot be confused with
    a stage-1 failure.
    """
    residual_flat = residual.flatten(-2, -1).float()
    hc_hidden = residual_flat.shape[-1]
    chunk = hc_hidden // n_splits
    mul = torch.empty(n_splits, residual_flat.shape[0], fn.shape[0])
    sqr = torch.empty(n_splits, residual_flat.shape[0])
    for i in range(n_splits):
        piece = residual_flat[:, i * chunk:(i + 1) * chunk]
        mul[i] = piece @ fn[:, i * chunk:(i + 1) * chunk].T
        sqr[i] = piece.square().sum(-1)
    return mul, sqr


def run_stage2_only(aiter, c):
    """Drive the stage-2 host entry directly; stage-1 inputs come from the CPU.

    The reference is derived from the same stage-1 numbers the kernel receives, so
    the only thing under test is stage 2 itself.
    """
    m, hidden = c["m"], c["hidden_size"]
    n_splits = c.get("n_splits", 1)
    residual, fn, hc_scale, hc_base = make_inputs(m, hidden, 20261104 + m + hidden)
    mul, sqr = stage1_reference(residual, fn, n_splits)
    sqrsum_full = sqr.sum(0)
    mixes = mul.sum(0) * (sqrsum_full.unsqueeze(-1) / (HC_MULT * hidden)
                          + ARGS_FULL["rms_eps"]).rsqrt()
    hc_scale_expanded = torch.cat([
        hc_scale[0].expand(HC_MULT), hc_scale[1].expand(HC_MULT),
        hc_scale[2].expand(HC_MULT2)])
    mixes = mixes * hc_scale_expanded + hc_base
    pre_mix = mixes[:, :HC_MULT].sigmoid().unsqueeze(-1) + ARGS_FULL["hc_pre_eps"]
    ref_post = (mixes[:, HC_MULT:2 * HC_MULT].sigmoid()
                * ARGS_FULL["hc_post_mult_value"]).unsqueeze(-1)
    res_mix = mixes[:, 2 * HC_MULT:].view(-1, HC_MULT, HC_MULT)
    x = res_mix.softmax(-1) + ARGS_FULL["hc_sinkhorn_eps"]
    x = x / (x.sum(-2, keepdim=True) + ARGS_FULL["hc_sinkhorn_eps"])
    for _ in range(ARGS_FULL["sinkhorn_repeat"] - 1):
        x = x / (x.sum(-1, keepdim=True) + ARGS_FULL["hc_sinkhorn_eps"])
        x = x / (x.sum(-2, keepdim=True) + ARGS_FULL["hc_sinkhorn_eps"])
    ref_layer = (residual * pre_mix).sum(-2).bfloat16()

    pad = (HC_MULT3 + 31) // 32 * 32
    mul_pad = torch.zeros(mul.shape[0], mul.shape[1], pad)
    mul_pad[:, :, :HC_MULT3] = mul
    # The stage-1 producer hands stage 2 a padded tensor sliced to hc_mult3 columns,
    # so stride(1) stays `pad`. Passing the packed slice would change that stride.
    d_mul = mul_pad.cuda()[:, :, :HC_MULT3]
    d_sqr = sqr.cuda()
    post_mix = torch.empty(m, HC_MULT, 1, dtype=torch.float32, device="cuda")
    comb_mix = torch.empty(m, HC_MULT, HC_MULT, dtype=torch.float32, device="cuda")
    layer_input = torch.empty(m, hidden, dtype=torch.bfloat16, device="cuda")
    aiter.mhc_pre_big_fuse(
        post_mix, comb_mix, layer_input, d_mul, d_sqr,
        hc_scale.cuda(), hc_base.cuda(), residual.cuda(),
        ARGS_FULL["rms_eps"], ARGS_FULL["hc_pre_eps"], ARGS_FULL["hc_sinkhorn_eps"],
        ARGS_FULL["hc_post_mult_value"], ARGS_FULL["sinkhorn_repeat"])
    torch.cuda.synchronize()
    record = dict(
        n_splits=n_splits,
        layer_input=compare_layer_input(ref_layer, layer_input.cpu()),
        post_mix=compare_fp32(ref_post, post_mix.cpu(), FP32_TOL),
        comb_mix=compare_fp32(x, comb_mix.cpu(), FP32_TOL),
        comb_column_sum_dev=sinkhorn_row_deviation(comb_mix.cpu()))
    bad = sum(record[k]["violations"] for k in ("layer_input", "post_mix", "comb_mix"))
    if bad:
        raise fail_with_detail(f"{bad} out-of-tolerance elements: {json.dumps(record)}", record)
    return record


def compare_reduction(ref, act, rel_tol=1e-5, near_zero_floor=1.0, near_zero_abs_tol=1e-3):
    """Relative check for the split-k reduction outputs, cancellation cases separate.

    Both the GEMM partial sum (order 1e2-1e3) and sqrsum (order 1e3-1e4) are
    reductions whose elements can cancel, so a fixed absolute tolerance is the wrong
    scale. Elements whose reference magnitude is below `near_zero_floor` are
    cancellation results: their relative error is meaningless, so they are judged on
    the absolute error instead, with `near_zero_abs_tol` set to roughly one fp32 ulp
    of the full sum's magnitude.
    """
    ref = ref.flatten()
    act = act.flatten()
    diff = (ref - act).abs()
    nz = ref.abs() >= near_zero_floor
    rel = diff / ref.abs().clamp(min=near_zero_floor)
    bad_nz = (rel > rel_tol) & nz
    bad_z = (diff > near_zero_abs_tol) & (~nz)
    return dict(
        violations=int(bad_nz.sum() + bad_z.sum()),
        max_abs_diff=float(diff.max()),
        max_rel_diff=float(rel[nz].max()) if bool(nz.any()) else 0.0,
        near_zero_count=int((~nz).sum()),
        near_zero_max_abs=float(diff[~nz].max()) if bool((~nz).any()) else 0.0,
        near_zero_max_rel=float((diff / ref.abs().clamp(min=1e-30))[~nz].max())
        if bool((~nz).any()) else 0.0,
    )


def run_reduce_splitk(aiter, c):
    """Smoke test for the standalone split-k reduction kernel against torch.sum.

    Also records the branch change this switch causes: with the reduced input the
    stage-2 kernel receives n_splits == 1 and therefore takes the serial reduction
    branch, which is existing semantics rather than a defect.
    """
    m, hidden = c["m"], c["hidden_size"]
    n_splits = c["n_splits"]
    residual, fn, hc_scale, hc_base = make_inputs(m, hidden, 20261104 + m + hidden)
    mul, sqr = stage1_reference(residual, fn, n_splits)
    pad = (HC_MULT3 + 31) // 32 * 32
    mul_pad = torch.zeros(n_splits, m, pad)
    mul_pad[:, :, :HC_MULT3] = mul
    d_mul = mul_pad.cuda()[:, :, :HC_MULT3]
    d_sqr = sqr.cuda()
    out_red = torch.zeros(1, m, pad, dtype=torch.float32, device="cuda")[:, :, :HC_MULT3]
    sqr_red = torch.zeros(1, m, dtype=torch.float32, device="cuda")
    aiter.mhc_pre_reduce_splitk(out_red, sqr_red, d_mul, d_sqr)
    torch.cuda.synchronize()
    ref_mul = mul.sum(0)
    ref_sqr = sqr.sum(0)
    record = dict(
        n_splits=n_splits,
        # The GEMM partials are of order 1e2-1e3 (fp32 ulp ~3e-5 there) and the device
        # sums the same 32 values in a different order than torch, measured worst case
        # 1.2e-4 over 32 splits; 1e-3 keeps about 8x margin and still flags a wrong partial.
        mul=compare_fp32(ref_mul, out_red.cpu(), 1e-3),
        sqrsum=compare_reduction(ref_sqr, sqr_red.cpu()))
    bad = record["mul"]["violations"] + record["sqrsum"]["violations"]
    if bad:
        raise fail_with_detail(f"{bad} out-of-tolerance elements: {json.dumps(record)}", record)
    return record


def run_invalid(aiter, c):
    """Non-expected values for the four switches: record reject vs silent fallback."""
    residual, fn, hc_scale, hc_base = make_inputs(32, 1280, 11)
    previous = {k: os.environ.get(k) for k in PRE_ENV}
    os.environ["AITER_MHC_PRE_KERNEL"] = "aiter"
    os.environ[c["name"]] = "not-a-valid-value"
    try:
        run_dut(aiter, residual, fn, hc_scale, hc_base, ARGS_FULL, False)
        rejected, error = False, None
    except (RuntimeError, ValueError) as exc:
        rejected, error = True, str(exc)
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return dict(rejected=rejected, error=error)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=GROUPS + ("all",), default="core")
    parser.add_argument("--case", help="select cases as kind:index[,kind:index]")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--require-arch", default="gfx946")
    parser.add_argument("--expect-xcds", type=int,
                        help="assert HSA reports one device with N XCDs")
    parser.add_argument("--json", type=Path,
                        help="write structured results; parent directory must exist")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    suites = GROUPS if args.suite == "all" else (args.suite,)
    if args.list_cases:
        for s in suites:
            for i, c in enumerate(cases(s)):
                print(f"{c['kind']}:{i} {json.dumps(c)}")
        return
    selected = [(s, c) for s in suites for c in select(cases(s), args.case)]
    if not selected:
        parser.error("no case matches --case")

    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == args.require_arch, props.gcnArchName
    topology = None
    if args.expect_xcds is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from query_xcd import query_xcds
        raw = query_xcds()
        topology = [r["xcds"] if isinstance(r, dict) else r for r in raw]
        assert topology == [args.expect_xcds], raw

    # The pilot must never depend on the auto rule for the stage-2 entry, except
    # in the two suites that exist to observe that rule.
    if args.suite not in ("pathprobe", "invalid"):
        preset = os.environ.get("AITER_MHC_PRE_KERNEL")
        if preset not in (None, "", "aiter"):
            raise RuntimeError(
                f"AITER_MHC_PRE_KERNEL must be unset or 'aiter', got {preset!r}")
        os.environ["AITER_MHC_PRE_KERNEL"] = "aiter"
    for name in PRE_ENV[1:]:
        current = os.environ.get(name)
        if current is None:
            continue
        if name == "AITER_MHC_PRE_BIG_FUSE_KERNEL" and current.startswith("b"):
            # A forced stage-2 block shape (b<block_size>,n<num_rows>,k<residual_block>)
            # is how the block-shape comparison selects an instance.
            continue
        if current == "":
            os.environ.pop(name, None)
        else:
            raise RuntimeError(f"{name} must be unset for this suite, got {current!r}")
    os.environ["AITER_MHC_PRE_TRACE"] = "1"

    import aiter
    from aiter.jit import core
    module = None
    if not args.rebuild:
        try:
            module = core.get_module("module_mhc")
        except ModuleNotFoundError:
            pass
    if module is None:
        build = core.get_args_of_build("module_mhc")
        build["md_name"] = "module_mhc"
        core.build_module(**{k: v for k, v in build.items()
                             if k in inspect.signature(core.build_module).parameters})
        module = core.get_module("module_mhc")
    so = Path(module.__file__).resolve()
    identity = dict(
        arch=props.gcnArchName, cu_num=props.multi_processor_count, topology=topology,
        gpu_chip=os.environ.get("GPU_CHIP"), module=str(so),
        module_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        env={k: os.environ.get(k) for k in PRE_ENV},
        torch=torch.__version__, hip=torch.version.hip)
    os.environ.pop("GPU_CHIP", None)
    result = dict(identity=identity, suite=args.suite, cases=[], complete=False)
    print("IDENTITY " + json.dumps(identity), flush=True)

    reused = {}
    if args.resume:
        old = json.loads(args.resume.read_text())
        assert old["identity"] == identity and old["complete"], "resume identity mismatch"
        reused = {r["id"]: r for r in old["cases"] if r["status"] == "PASS"}

    def save(record):
        result["cases"].append(record)
        print(record["status"] + " " + json.dumps(record), flush=True)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2) + "\n")

    failures = 0
    for suite, c in selected:
        key = suite + "_" + hashlib.sha256(
            json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
        if key in reused:
            save(dict(reused[key], reused=True))
            continue
        print("START " + json.dumps(c), flush=True)
        record = dict(id=key, **c)
        try:
            if suite == "tf32":
                detail = run_tf32_identical(aiter, c)
            elif suite == "pathprobe":
                detail = run_pathprobe(aiter, c)
            elif suite == "invalid":
                detail = run_invalid(aiter, c)
            elif suite == "splitk":
                detail = run_reduce_splitk(aiter, c)
            elif suite in ("stage2core", "stage2splitk"):
                detail = run_stage2_only(aiter, c)
            else:
                detail = run_case(aiter, c)
            save(dict(record, status="PASS", **detail))
        except Exception as exc:
            failures += 1
            stack = traceback.extract_tb(exc.__traceback__)
            site = f"{stack[-1].filename.split('/')[-1]}:{stack[-1].lineno}" if stack else ""
            save(dict(record, status="FAIL", error=str(exc), raise_site=site,
                      detail=getattr(exc, "detail", None),
                      error_type=type(exc).__name__))
    if failures:
        result["complete"] = False
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SUMMARY total={len(result['cases'])} failed={failures}", flush=True)
    if failures:
        raise SystemExit(1)
    result["complete"] = True
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT PASS {len(result['cases'])} cases", flush=True)


if __name__ == "__main__":
    main()
