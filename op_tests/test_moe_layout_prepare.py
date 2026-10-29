# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""fused_moe 权重布局决策与一次性准备的测试

覆盖：
  1. 双表方向：收益 ≥5% 选 asm_shuffle，<5% 选 raw
  2. model_dim 不是 64 的倍数：asm 补 K 后执行，数值正确（执行层补 w2_scale）
  3. moe_c 的 w8a8_fp8 + 非对齐 model_dim：降级 raw（宿主侧补 K 数值错误）
  4. 决策不会主动选的布局：用 _shared_shuffle_impl 强制 asm_shuffle 后仍正确
  5. moe_c w8a16：marlin 准备 + 重复准备返回同一对象
  6. 前向零重排：aiter.ops.shuffle 布局函数调用计数 = 0
  7. 五类错配：布局与家族不符 / 属性丢失 / activation、topk、expert_mask
     与准备时不一致

用法（容器内、仓库根）：PYTHONPATH=. python op_tests/test_moe_layout_prepare.py
退出码 0 = 全部通过。
"""
import torch

from aiter import ActivationType, QuantType, dtypes, pertoken_quant
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.fused_moe_layout import (describe_fused_moe_layout,
                                   fused_moe_prepare_weights, get_weight_layout)
from aiter.moe import _shared_shuffle_impl
import aiter.ops.shuffle as SH

torch.set_default_device("cuda")
DT = torch.bfloat16
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' | ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def cos_rel(a, b):
    af, bf = a.float(), b.float()
    cos = float(torch.nn.functional.cosine_similarity(af.flatten(), bf.flatten(), dim=0))
    rel = float((af - bf).norm() / bf.norm().clamp(min=1e-12))
    return cos, rel


def route_of(fn, *a, **k):
    import aiter.fused_moe_router as R
    seen = []
    orig = {n: e.execute for n, e in R.BACKEND_REGISTRY.items()}
    for name, e in R.BACKEND_REGISTRY.items():
        def mk(n, f):
            def w(ctx):
                seen.append(n)
                return f(ctx)
            return w
        R.BACKEND_REGISTRY[name] = R.BackendEntry(e.name, e.match, e.lookup,
                                                 mk(name, e.execute))
    try:
        out = fn(*a, **k)
    finally:
        for name in orig:
            e = R.BACKEND_REGISTRY[name]
            R.BACKEND_REGISTRY[name] = R.BackendEntry(e.name, e.match, e.lookup, orig[name])
    return out, (seen[-1] if seen else "RAISE")


def expect_value_error(name, needle, fn):
    try:
        fn()
    except ValueError as exc:
        ok = needle in str(exc)
        check(name, ok, f"msg={str(exc)[:110]}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exc {type(exc).__name__}: {exc}")


def build(M, E, N, K, topk, quant="fp8", seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn((M, K), dtype=DT) / 10
    w1 = torch.randn((E, 2 * N, K), dtype=DT) / 10
    w2 = torch.randn((E, K, N), dtype=DT) / 10
    if quant == "none":
        w1q, w2q = w1, w2
        s1 = s2 = None
        qt = QuantType.No
    else:
        qdt = dtypes.fp8 if quant == "fp8" else torch.int8
        w1q, s1 = pertoken_quant(w1, quant_dtype=qdt)
        w2q, s2 = pertoken_quant(w2, quant_dtype=qdt)
        qt = QuantType.per_Token
    tw, ti = fused_topk(hidden, torch.randn((M, E), dtype=DT), topk, True)
    ref = torch_moe(hidden, w1, w2, tw, ti)
    return dict(hidden=hidden, w1q=w1q, w2q=w2q, s1=s1, s2=s2, qt=qt,
                tw=tw, ti=ti, ref=ref, M=M, E=E, N=N, K=K, topk=topk)


def run_case(c, layout, w1=None, w2=None, **over):
    kw = dict(activation=ActivationType.Silu, quant_type=c["qt"],
              w1_scale=c["s1"], w2_scale=c["s2"], dtype=DT,
              weight_layout=layout)
    kw.update(over)
    return route_of(fused_moe, c["hidden"], w1 if w1 is not None else c["w1q"],
                    w2 if w2 is not None else c["w2q"], c["tw"], c["ti"], **kw)


# ---------------------------------------------------------------------------
# 1. 双表方向（只查决策，不执行 kernel）
# ---------------------------------------------------------------------------
low = build(8, 256, 256, 2048, 8, quant="none", seed=1)
d_low = describe_fused_moe_layout(low["w1q"], low["w2q"], quant_type=low["qt"],
                                 dtype=DT, topk=8, activation="silu")
print(f"describe low-gain shape: {d_low['layout']} gain={d_low['weighted_gain']} "
      f"reason={d_low['reason']}")
check("dual-table low gain -> raw",
      d_low["layout"] == "raw" and (d_low["weighted_gain"] is None
                                    or d_low["weighted_gain"] < 0.05),
      f"gain={d_low['weighted_gain']}")

high = build(8, 128, 384, 4096, 8, quant="none", seed=2)  # noqa: F841  (shape reserved for manual runs)
w1h = torch.zeros((128, 2 * 384, 4096), dtype=DT)
w2h = torch.zeros((128, 4096, 384), dtype=DT)
d_high = describe_fused_moe_layout(w1h, w2h, quant_type=QuantType.No,
                                  dtype=DT, topk=8, activation="silu")
print(f"describe high-gain shape: {d_high['layout']} gain={d_high['weighted_gain']} "
      f"reason={d_high['reason']}")
check("dual-table high gain -> asm_shuffle",
      d_high["layout"] == "asm_shuffle" and d_high["weighted_gain"] >= 0.05,
      f"gain={d_high['weighted_gain']}")

# ---------------------------------------------------------------------------
# 2. K 非 64 倍数：asm 补 K（fp8 / int8）
# ---------------------------------------------------------------------------
for q in ("fp8", "int8"):
    c = build(16, 256, 128, 7160, 8, quant=q, seed=3)
    d = describe_fused_moe_layout(c["w1q"], c["w2q"], quant_type=c["qt"],
                                 w1_scale=c["s1"], w2_scale=c["s2"],
                                 dtype=DT, topk=8, activation="silu")
    check(f"padk decision {q} -> asm_shuffle/padded 7168",
          d["layout"] == "asm_shuffle" and d["padded_k"] == 7168,
          f"layout={d['layout']} padded_k={d['padded_k']}")
    p1, p2, lay = fused_moe_prepare_weights(
        c["w1q"], c["w2q"], quant_type=c["qt"], w1_scale=c["s1"], w2_scale=c["s2"],
        dtype=DT, topk=8, activation="silu")
    check(f"padk prepared shape {q}", tuple(p1.shape)[2] == 7168 and lay == "asm_shuffle",
          f"w1={tuple(p1.shape)} layout={lay}")
    out, land = run_case(c, lay, w1=p1, w2=p2)
    cos, rel = cos_rel(out, c["ref"])
    check(f"padk numeric {q}", land == "asm" and cos >= 0.995, f"land={land} cos={cos:.5f}")

# 3. moe_c·fp8 非对齐 K -> 降级 raw
c = build(8, 256, 128, 7160, 8, quant="fp8", seed=4)
d = describe_fused_moe_layout(c["w1q"], c["w2q"], quant_type=c["qt"],
                             w1_scale=c["s1"], w2_scale=c["s2"],
                             dtype=DT, topk=8, activation="silu")
check("moe_c fp8 non-aligned K degrades to raw (never marlin)",
      "marlin" not in d["feasible_layouts"], f"feasible={d['feasible_layouts']}")

# --- 后续用 w8a8 fp8 对齐形状做 asm/错配用例 ---
c = build(16, 256, 128, 7168, 8, quant="fp8", seed=5)
d = describe_fused_moe_layout(c["w1q"], c["w2q"], quant_type=c["qt"],
                             w1_scale=c["s1"], w2_scale=c["s2"],
                             dtype=DT, topk=8, activation="silu")
print(f"w8a8 aligned decision: {d['layout']} gain={d['weighted_gain']} reason={d['reason']}")

# 4. 强制 asm_shuffle（决策可能选 raw）
p1, p2, _s1, _s2 = _shared_shuffle_impl(
    c["w1q"], c["w2q"], c["s1"], c["s2"], quant_kind="w8a8_fp8",
    padded_k=None, layout="asm_shuffle")
setattr(p1, "weight_layout", "asm_shuffle")
setattr(p2, "weight_layout", "asm_shuffle")
check("shared core returns shuffled w2_scale positionally",
      _s2 is c["s2"] and _s1 is c["s1"], "asm 未补 K 时 scale 原样返回")
out_f, land_f = run_case(c, "asm_shuffle", w1=p1, w2=p2)
cos_f, _ = cos_rel(out_f, c["ref"])
check("forced asm_shuffle runs on asm", land_f == "asm" and cos_f >= 0.995,
      f"land={land_f} cos={cos_f:.5f}")

# ---------------------------------------------------------------------------
# 5. moe_c w8a16：marlin 准备 + 重复准备返回同一对象
# ---------------------------------------------------------------------------
E2, N2, K2 = 128, 768, 2048
w1i = torch.randint(-127, 128, (E2, 2 * N2, K2), dtype=torch.int8)
w2i = torch.randint(-127, 128, (E2, K2, N2), dtype=torch.int8)
s1 = torch.ones((E2, 2 * N2, 1), dtype=DT)
s2 = torch.ones((E2, K2, 1), dtype=DT)
hidden2 = torch.randn((64, K2), dtype=DT) / 10
tw2, ti2 = fused_topk(hidden2, torch.randn((64, E2), dtype=DT), 8, True)
ref2 = torch_moe(hidden2, w1i.to(DT) * 0.01, w2i.to(DT) * 0.01, tw2, ti2, s1, s2)
m1, m2, lay = fused_moe_prepare_weights(
    w1i, w2i, quant_type=QuantType.No, dtype=DT, topk=8, activation="silu")
check("w8a16 prepare -> marlin", lay == "marlin" and get_weight_layout(m1, m2) == "marlin",
      f"layout={lay}")
m1b, m2b, layb = fused_moe_prepare_weights(
    m1, m2, quant_type=QuantType.No, dtype=DT, topk=8, activation="silu")
check("prepare again returns the same tensors",
      layb == "marlin" and m1b is m1 and m2b is m2)
out_m, land_m = route_of(fused_moe, hidden2, m1, m2, tw2, ti2,
                         activation=ActivationType.Silu, quant_type=QuantType.No,
                         w1_scale=s1, w2_scale=s2, dtype=DT, weight_layout=lay)
cos_m, _ = cos_rel(out_m, ref2)
check("w8a16 marlin numeric", land_m == "moe_c" and cos_m >= 0.99,
      f"land={land_m} cos={cos_m:.5f}")

# ---------------------------------------------------------------------------
# 6. 前向零重排
# ---------------------------------------------------------------------------
names = ("asm_shuffle_weight_b8", "moe_layout_shuffle_gemm2",
         "w8a16_marlin_weight_1", "w8a16_marlin_weight_2",
         "w4a16_marlin_weight_1", "w4a16_marlin_weight_2",
         "w4a8_moe_repack_shuffle", "wfp4a8_moe_layout_shuffle")
counts = {n: 0 for n in names}
orig = {n: getattr(SH, n) for n in names}


def wrap(n):
    f = orig[n]

    def w(*a, **k):
        counts[n] += 1
        return f(*a, **k)
    return w


for n in names:
    setattr(SH, n, wrap(n))
try:
    run_case(c, "asm_shuffle", w1=p1, w2=p2)
    route_of(fused_moe, hidden2, m1, m2, tw2, ti2,
             activation=ActivationType.Silu, quant_type=QuantType.No,
             w1_scale=s1, w2_scale=s2, dtype=DT, weight_layout="marlin")
finally:
    for n in names:
        setattr(SH, n, orig[n])
total = sum(counts.values())
check("zero live shuffle during forward", total == 0, f"counts={counts}")

# ---------------------------------------------------------------------------
# 6b. 非 raw 布局短路直连
# ---------------------------------------------------------------------------
from dataclasses import replace as _replace

from aiter.fused_moe_layout import _build_problem
from aiter.fused_moe_router import (BACKEND_REGISTRY, _LAYOUT_BACKEND_BINDING,
                                    _layout_directly_usable, plan)

p_raw = _build_problem(c["w1q"], c["w2q"], c["qt"], DT, 8, "silu", None,
                       w1_scale=c["s1"], w2_scale=c["s2"])


def _probe(layout):
    return _replace(p_raw, weight_layout=layout,
                    use_shuffle=1 if layout == "asm_shuffle" else 0)


check("binding table is the single source (marlin->moe_c, asm_shuffle->asm)",
      _LAYOUT_BACKEND_BINDING == {"marlin": "moe_c", "asm_shuffle": "asm"},
      f"{_LAYOUT_BACKEND_BINDING}")

for layout, bound in _LAYOUT_BACKEND_BINDING.items():
    prob = _probe(layout)
    others_usable = []
    for name, entry in BACKEND_REGISTRY.items():
        if not entry.match(prob):
            continue
        params = entry.lookup(prob)
        if params is None:
            continue
        if _layout_directly_usable(prob, name, params) and name != bound:
            others_usable.append(name)
    check(f"gate: only {bound} directly consumes {layout}", not others_usable,
          f"others={others_usable}")

    routed = plan(prob)
    landed = routed.stage1_backend or routed.chained_backend
    check(f"short-circuit: {layout} -> {bound}", landed == bound, f"landed={landed}")

# raw 布局仍走路由表 / 候选链 / triton 兜底（不短路），且不抛错
raw_route = plan(_probe("raw"))
check("raw layout still uses the route table / candidate chain / triton fallback",
      (raw_route.stage1_backend or raw_route.chained_backend) in BACKEND_REGISTRY,
      f"landed={raw_route.stage1_backend or raw_route.chained_backend}")

# ---------------------------------------------------------------------------
# 7. 错配用例
# ---------------------------------------------------------------------------
# 7.1 布局与家族不符：无量化权重（bf16）+ marlin
nq = build(8, 64, 128, 1024, 8, quant="none", seed=6)


def _family_mismatch():
    fused_moe(nq["hidden"], nq["w1q"], nq["w2q"], nq["tw"], nq["ti"],
              activation=ActivationType.Silu, quant_type=QuantType.No,
              dtype=DT, weight_layout="marlin")


expect_value_error("mismatch: marlin layout on non-quantized weights",
                   "not directly usable", _family_mismatch)

# 7.2 属性丢失 + 显式 raw：marlin 权重被当 raw
def _attr_lost():
    delattr(m1, "weight_layout")
    delattr(m2, "weight_layout")
    fused_moe(hidden2, m1, m2, tw2, ti2, activation=ActivationType.Silu,
              quant_type=QuantType.No, w1_scale=s1, w2_scale=s2,
              dtype=DT, weight_layout="raw")


expect_value_error("mismatch: prepared marlin weights declared raw",
                   "conflicts with the tensor marker", _attr_lost)
setattr(m1, "weight_layout", "marlin")
setattr(m2, "weight_layout", "marlin")

# 7.3 activation 不一致：准备用 silu，运行用 situ（asm/triton 拒绝 situ）
def _act_mismatch():
    fused_moe(c["hidden"], p1, p2, c["tw"], c["ti"], activation="situ",
              quant_type=c["qt"], w1_scale=c["s1"], w2_scale=c["s2"],
              dtype=DT, weight_layout="asm_shuffle")


expect_value_error("mismatch: activation silu->situ", "feasible layouts", _act_mismatch)

# 7.4 topk 不一致：准备 topk=8，运行 topk=2（该形状无 topk=2 配置）
def _topk_mismatch():
    tw2_, ti2_ = fused_topk(c["hidden"], torch.randn((c["M"], c["E"]), dtype=DT), 2, True)
    fused_moe(c["hidden"], p1, p2, tw2_, ti2_, activation=ActivationType.Silu,
              quant_type=c["qt"], w1_scale=c["s1"], w2_scale=c["s2"],
              dtype=DT, weight_layout="asm_shuffle")


expect_value_error("mismatch: topk 8->2", "feasible layouts", _topk_mismatch)

# 7.5 expert_mask 不一致：准备非 EP，运行 EP（EP 只允许 raw）
def _ep_mismatch():
    em = torch.ones((c["E"],), dtype=torch.int32)
    em[c["E"] // 2:] = 0
    fused_moe(c["hidden"], p1, p2, c["tw"], c["ti"], expert_mask=em,
              activation=ActivationType.Silu, quant_type=c["qt"],
              w1_scale=c["s1"], w2_scale=c["s2"], dtype=DT,
              weight_layout="asm_shuffle")


expect_value_error("mismatch: non-EP prepared, EP runtime", "feasible layouts", _ep_mismatch)

print("MOE_LAYOUT_PREPARE_" + ("PASS" if not FAILS else f"FAIL {FAILS}"))
raise SystemExit(0 if not FAILS else 1)
