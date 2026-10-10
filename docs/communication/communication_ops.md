<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
-->

# 通信算子接口说明与性能结果

本文介绍 Hygon AITER 的通信接口扩展及调用约定。AR、AG、RS 分别表示 AllReduce、AllGather、ReduceScatter；TP 表示参与张量并行的 rank 数。设备代码在 HCU 上仍使用 PyTorch 的 `cuda` 名称。

普通 CAR 的完整示例、Graph 注意事项及当前 gfx936/gfx938 性能见 [CustomAllreduce 用法与性能](custom_allreduce.md)。

## 接口总览

| 接口族 | 新增或扩展能力 | 使用入口 |
|---|---|---|
| Custom AllReduce（CAR） | 保留原有 AR，补充容量、初始化失败和资源释放处理；AG 增加整数位保真，RS 增加任意维度及准入判断 | `CustomAllreduce.should_custom_ag/should_custom_rs`、`custom_all_gather/custom_reduce_scatter` |
| AR + RMSNorm | Gemma `1+weight`、输出补零、有效 hidden 宽度、末维切片输入 | `custom_fused_ar_rms`、`custom_fused_ar_rms_packed_input` |
| AR + RMSNorm + FP8 | per-token 增加 Gemma 和未量化输出；per-group 增加转置 scale 存储；保留原有 per-group 三/四输出接口 | `custom_fused_ar_rms_quant`、`custom_fused_ar_rms_per_group_quant` |
| AR + RMSNorm + MXFP4 | 新增 E2M1 打包及每 32 元素 E8M0 scale | `custom_fused_ar_rms_mxfp4_quant` |
| QK-Norm + AR + RoPE | 在原有分布式 QK 归一化上融合 NeoX RoPE | `custom_fused_qknorm_ar_rope` |
| AR + MHC post | 新增自动、一阶段、split 入口及分配输出的 wrapper | `fused_allreduce_mhc_post_{only,one_stage,split}` |
| HIP Quick AllReduce | 补齐 native QR、IPC 句柄、量化模式及 QR + RMSNorm | [Quick AllReduce](quick_all_reduce.md) |
| 并行组接口 | DP/EP/PP 通信封装、DCP/PCP 分组、维度 RS、量化统一分派；保留 TP 与自定义组 | `aiter.dist.communication_op`、`init_dist_env` |
| Iris/Triton 通信 | 新增 AG、RS、RS + RMSNorm + 可选 FP8 + 可选 AG | [Iris 通信](iris_comms.md) |
| FlyDSL INT4 Quick AllReduce | 新增可选 BF16 INT4 后端及小载荷 FP 策略 | [QuickAllReduceInt4](quick_all_reduce.md#flydsl-int4-接口) |

新增 HIP 融合、AG/RS、QR 和 INT4 的已验证范围是 **gfx936 / gfx938、单节点 IPC、TP2/4/8**。普通 CAR 保留已有的 6/16/32/40-rank 和 supernode 路径；不能据此推定新增融合也支持这些 rank 数。Iris 的限制见其独立文档。

这里的 EP 扩展指专家并行组的 collective 封装，不包含 FlyDSL/Mori 的专家 token dispatch/combine 或 MegaMoEV2。

## 初始化与通用约定

框架已有 AITER 并行组时，优先使用 `aiter.dist.communication_op` 的 TP/DP/EP/PP 函数；直接管理通信对象时使用下面的方式。示例保存为脚本后可用 `torchrun --standalone --nproc_per_node=2 example.py` 运行。

```python
import os
import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("gloo")
device = torch.device("cuda", local_rank)
ca = CustomAllreduce(dist.group.WORLD, device, max_size=64 << 20,
                     enable_register_for_capturing=False)
try:
    if ca.disabled:
        raise RuntimeError("当前设备组不满足 CAR 的构建或拓扑要求")
    x = torch.full((4, 256), dist.get_rank() + 1,
                   dtype=torch.bfloat16, device=device)
    y = ca.custom_all_reduce(x)
    if y is None:
        raise RuntimeError("输入超出 CAR 准入范围")
    torch.cuda.synchronize()
    dist.barrier()
finally:
    ca.close()
    dist.destroy_process_group()
```

- 各 rank 使用不同 GPU，按相同顺序调用 collective，输入形状/dtype 和选项保持一致。直接 CAR 对象使用 CPU/Gloo 组交换元数据。
- `max_size` 为预分配 staging 容量，默认 1 GiB；普通 AR 的默认载荷门限为 256 MiB，可用 `AITER_AR_MAX_SIZE_MB` 调整，但始终受 `max_size/2` 约束。融合和 RS 仍受 `min(64 MiB, max_size/2)` 约束。
- 直接 `custom_*` 的部分入口以 `None` 表示不适用；调用方应检查准入或处理回退。**RS 成功也返回 `None`，实际结果写入指定 output**，应在调用前使用 `should_custom_rs` 判断。
- `AITER_AR_TRANSPORT` 默认 `ipc`，也接受 `auto` / `fabric`。`AITER_ENABLE_SUPERNODE_AR=0` 是编译时 IPC-only 开关；未设置时保留 Fabric 构建。Fabric 需要相应 DTK 头文件与运行时支持。
- 当前新增接口的结果来自 IPC。Fabric 完整释放重建、跨节点新增功能仍未完成验收，已有 supernode 能力不能作为这些新路径的验证结果。
- 结束使用前等待通信及输出消费完成，销毁关联 Graph，再由所有参与 rank 有序关闭对象。不要在仍使用 IPC buffer 时释放通信对象。

## AllGather / ReduceScatter

以下代码在上例的 `try` 内、`ca` 存活期间运行。

```python
world = dist.get_world_size()
assert ca.should_custom_ag(x, dim=-1)
gathered = ca.custom_all_gather(x, dim=-1)  # [4, 256 * world]

rs_input = torch.ones((4, world * 256), dtype=x.dtype, device=device)
rs_output = torch.empty((4, 256), dtype=x.dtype, device=device)
assert ca.should_custom_rs(rs_input, dim=1)
ca.custom_reduce_scatter(rs_input, rs_output, dim=1)
```

| 接口 | 形状与返回 | 约束 |
|---|---|---|
| `custom_all_gather(inp, dim=0)` | 沿 dim 按组内 rank 顺序拼接；返回新 Tensor | 直接 CAR 支持首维/末维；连续 FP32/FP16/BF16/int16/int32/int64 |
| `all_gather_unreg(inp, out=None, dim=0)` | 可提供预分配输出，返回 out | 总输入字节数整除 16；末维 AG 的每行字节数也整除 16；输入 ≤ `max_size/(2*TP)` |
| `all_gather_reg(inp, out=None, dim=0)` | 注册形式，返回 out | HCU 可能根据 native 能力转为 copy-in；Fabric 不支持此注册形式 |
| `custom_reduce_scatter(input, output, dim=0)` | 跨 rank 求和后按 dim 分片，写 output；返回 `None` | 连续 FP32/FP16/BF16；dim 可为首/中间/末维及负索引；该维整除 TP |
| `reduce_scatter(inp, out, dim=0, *, registered=False)` | 同上，输入不适用时抛异常 | 总输入字节数整除 16；dim=0 时每 rank 分片也满足 16 字节对齐 |

所有输入输出必须在对象绑定的设备上、dtype 相同、存储互不重叠；输出连续且形状准确。整数 AG 通过等宽位视图传输，不转换数值。AG/RS 的 capture 外 warmup 会产生真实通信结果。

底层 `aiter.ops.custom_all_reduce.reduce_scatter_dim` 接受展平后的 `(m,n,k,split_dim)`；原 `reduce_scatter(_fa, inp, out, reg_ptr, reg_bytes)` 保留首维语义。通常无需直接构造这些 native 参数。

## AR + residual + RMSNorm / Gemma

设每个 rank 输入为 `x_r`，本地 residual 为 `r`，有效宽度 `H=weight.numel()`：

```text
s = sum_r(x_r[..., :H]) + r
norm = s * rsqrt(mean(s^2, last_dim) + eps) * weight
Gemma 时最后一项为 (1 + weight)
```

返回顺序始终是 **`(norm_output, residual_output)`**，其中 `residual_output=s`；输入 residual 不被修改。归一化分母使用有效宽度 H，填充列不参与。

```python
residual = torch.zeros_like(x)
weight = torch.ones(x.shape[-1], dtype=x.dtype, device=device)
norm, residual_out = ca.custom_fused_ar_rms(
    x, residual, weight, 1e-6, use_1stage=False,
    out_hidden_dim=512, gemma_norm=False,
)
# norm: [4,512]，后256列为0；residual_out: [4,256]

storage = torch.empty((4, 512), dtype=x.dtype, device=device)
view = storage[:, :256]
view.copy_(x)
norm2, residual2 = ca.custom_fused_ar_rms_packed_input(
    view, residual, weight, 1e-6, gemma_norm=True,
)
```

- 支持 FP32/FP16/BF16；input、residual、weight 同 dtype/设备。weight 为连续一维向量；residual 的最后一维为 H，其余维与 input 一致。
- 输入宽度可大于 H；两者都须按 16 字节 pack 对齐。`out_hidden_dim=0` 使用输入宽度；指定值必须 ≥H 且按同一 pack 对齐。
- packed 入口接受二维末维切片：`stride(-1)=1`、`stride(0)>=input.shape[-1]`，内部直接打包到通信 buffer。
- 低层对象方法 `fused_ar_rms(inp, res_inp, *, w, eps, res_out=None, out=None, ...)` 可复用输出。默认 `use_1stage=False` 保持原有两阶段选择。

TP 组级入口为 `tensor_model_parallel_fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps, prefill_support=False, x_pad_to_multiple=0, gemma_norm=False)`。这里的 `x_pad_to_multiple` 是输出宽度的对齐倍数：输出宽度取 `ceil(H/multiple)*multiple`，与 CAR 的直接指定 `out_hidden_dim` 不同。native 对应 `aiter.ops.custom_all_reduce.fused_allreduce_rmsnorm` / `fused_allreduce_rmsnorm_pad`，均写入调用方提供的输出。

## AR + RMSNorm + 量化

```python
# per-token FP8：新增 Gemma 与未量化 mirror
q, residual_out, scale, mirror = ca.custom_fused_ar_rms_quant(
    x, residual, weight, 1e-6, gemma_norm=True, emit_bf16=True,
)
# per-group FP8：逻辑 shape 不变，scale 按列存储
qg, rg, sg = ca.custom_fused_ar_rms_per_group_quant(
    x, residual, weight, 1e-6, group_size=128, transpose_scale=True,
)
# MXFP4：两个 E2M1 编码打包到一个字节
q4, r4, s4 = ca.custom_fused_ar_rms_mxfp4_quant(
    x, residual, weight, 1e-6,
)
```

令输入为 `[M,H]`：

| 模式 | quant output | scale | 限制 |
|---|---|---|---|
| per-token FP8 | `[M,H]`，HCU E4M3FN | FP32 `[M,1]` | FP32/FP16/BF16 输入；可用 Gemma |
| per-group FP8 | `[M,H]`，HCU E4M3FN | FP32 `[M,H/group_size]` | FP16/BF16；group_size 默认128，须整除 H |
| MXFP4 | uint8 `[M,H/2]` | uint8 `[M,H/32]`，E8M0 编码 | FP16/BF16；H 为32的倍数；不支持 Gemma |

默认返回 `(q, residual_out, scale)`；`emit_bf16=True` 增加第四项未量化 norm mirror。**mirror 实际 dtype 跟随输入，FP16 输入返回 FP16**。残差输出也保持输入 dtype。FP8 的反量化为 `q.float()*scale`（per-group 按组广播）。MXFP4 每字节低/高 nibble 对应相邻两个值，需解码 E2M1 和 E8M0 后使用，不能直接把 uint8 scale 当浮点乘数。

量化输入宽度必须等于 weight 长度，`H/(16/element_size) <= 1024`。per-group 的 `group_size/(16/element_size)` 必须是 2 的幂且 ≤64。`transpose_scale=True` 只用于二维 per-group：**逻辑 shape 仍为 `[M,G]`，stride 为 `(1,M)`**，调用 `.contiguous()` 会改变这一存储约定。MXFP4 要求连续输入。

MXFP4 的直接 CAR 默认两阶段路径另要求输入 ≤512 KiB，`H/(16/element_size)` 为 TP 的倍数；请求 mirror 时还须为32的倍数。超过该载荷的直接调用需显式指定 `use_1stage=True`，并满足一阶段的宽度及 communicator 容量约束；默认两阶段不支持时会抛异常。

组级统一入口为：

```python
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm_quant

# 前提：已初始化 AITER TP 组。
outputs = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
    x, residual, weight, 1e-6,
    quant_type="per_group", group_size=128,
    emit_bf16=True, transpose_scale=True,
)
```

`quant_type` 的推荐值为 `per_token`、`per_group`、`mxfp4`；也接受 `QuantType.per_Token/per_1x128/per_1x32`。原 `tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group` 保留，新增 `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` 便捷入口。

三种 native 输出入口分别为 `fused_allreduce_rmsnorm_quant`、`fused_allreduce_rmsnorm_quant_per_group`、`fused_allreduce_rmsnorm_mxfp4_quant`，位于 `aiter.ops.custom_all_reduce`；它们使用 native CAR handle 和预分配输出。普通应用优先使用上面的对象方法或组级封装。

## QK-Norm + 分布式统计 + RoPE

```python
qkv = torch.randn((4, 1024), dtype=x.dtype, device=device)
qw = torch.ones(512, dtype=x.dtype, device=device)
kw = torch.ones(256, dtype=x.dtype, device=device)
angles = torch.randn((32, 32), device=device)
cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(x.dtype)
positions = torch.arange(4, dtype=torch.int64, device=device)
q, k, v = ca.custom_fused_qknorm_ar_rope(
    qkv, qw, kw, cache, positions, head_dim=128, rotary_dim=64, eps=1e-6,
)
# q/k/v: [4,512] / [4,256] / [4,256]
```

输入最后一维按 `[Q,K,V]` 拼接，Q/K 宽度由权重长度给出。每 token 的 Q 归一化统计量为 `sum_r(sum(Q_r**2))/(TP*Hq)`，K 同理；使用各 rank 本地 Q/K 乘全局 rstd 和本地权重，再按 head 应用 RoPE。**这里不对 Q/K/V 数值做 AllReduce 求和，V 原样复制。**未带 RoPE 的原接口 `custom_fused_qknorm_ar` 继续可用。

- 支持连续 FP32/FP16/BF16，权重/cache 与输入同 dtype/设备；`position_ids` 为同设备连续 int64 `[M]`，值须在 cache 行范围内。
- cache 为 `[max_pos, rotary_dim]`，前半 cos、后半 sin，采用 NeoX 前后半配对；每 head 的非 rotary 部分不旋转。
- 令 `P=16/element_size`：Q/K/V 宽度各为 `32*P` 的正倍数，总宽度/P ≤1024；head_dim 为 P 的倍数且整除 `32*P`；Q/K 宽度均为 head_dim 的倍数；`0<rotary_dim<=head_dim` 且 `rotary_dim % (2*P) == 0`。
- TP 封装为 `tensor_model_parallel_fused_qknorm_allreduce_rope(qkv_in, q_w, k_w, cos_sin_cache, position_ids, head_dim, rotary_dim, eps)`，返回 `(q,k,v)`。

## AR + MHC post

仅执行 AllReduce 与 MHC post 混合，不包含 MHC pre 或 RMSNorm。令 `a=sum_r(x_r)`，输出：

```text
next_residual[m,j,h] = a[m,h]*post_layer_mix[m,j]
                       + sum_i(residual_in[m,i,h]*comb_res_mix[m,i,j])
```

```python
from aiter.ops.custom_all_reduce import launch_fused_allreduce_mhc_post_only

mhc_residual = torch.zeros((4, 4, 256), dtype=x.dtype, device=device)
post = torch.ones((4, 4), dtype=torch.float32, device=device)
comb = torch.eye(4, dtype=torch.float32, device=device).repeat(4, 1, 1)
next_residual = launch_fused_allreduce_mhc_post_only(
    ca._ptr, x, mhc_residual, post, comb,
    reg_ptr=ca._pool["input"].data_ptr,
    reg_bytes=ca._pool["input"].max_size,
)
```

这是与 CAR 对象绑定的低层入口：DTK 下必须传入该对象的 copy-in buffer，不能把默认 `reg_ptr=0` 当作可用配置。

| 参数 | 要求 |
|---|---|
| input / residual / output | FP16/BF16；`[M,H]` / `[M,4,H]` / `[M,4,H]`；H 为256的正倍数 |
| post_layer_mix / comb_res_mix | FP32 `[M,4]` / `[M,4,4]`；分配 wrapper 也接受 post `[M,4,1]` |
| 布局 | 全部在同一 GPU、连续；输出不与任何输入重叠 |
| 开关 | `use_new=True`，`open_fp8_quant=False`；当前不支持 FP8 MHC |

`fused_allreduce_mhc_post_only` 自动选择路径，`fused_allreduce_mhc_post_one_stage` 为一阶段入口，`fused_allreduce_mhc_post_split` 为 split 入口；三者的 native 参数顺序都是 `(_fa, inp, next_residual, residual_in, post_layer_mix, comb_res_mix, use_new=True, open_fp8_quant=False, reg_ptr=0, reg_bytes=0)`，原位写 next_residual、返回 `None`。`launch_fused_allreduce_mhc_post_only` 分配并返回输出；`launch_fused_allreduce_mhc_post_split` 还接受关键字 `next_residual` 以复用存储。

## 并行组与框架调用

```python
from aiter.ops.communication import init_dist_env, destroy_dist_env
from aiter.dist.communication_op import tensor_model_parallel_reduce_scatter

# 独立脚本中，在初始化任何其他 process group 之前调用。
# 各进程先根据 LOCAL_RANK 设置 GPU；使用 torchrun 提供的 env://。
init_dist_env(world_size=2, rankID=int(os.environ["RANK"]),
              local_rank=int(os.environ["LOCAL_RANK"]))
try:
    value = torch.ones((4, 256), dtype=torch.bfloat16, device="cuda")
    shard = tensor_model_parallel_reduce_scatter(value, use_custom=True, dim=-1)
finally:
    destroy_dist_env()
```

`init_dist_env` 的 `world_size` 保留 **TP 大小** 含义，可改用 `tensor_model_parallel_size`；两者同时传入时必须相等。可选 `pipeline_model_parallel_size`、`data_parallel_size/data_parallel_rank`、`prefill_context_model_parallel_size`、`decode_context_parallel_size`、`custom_group_config`。DCP 必须整除 TP；启用 DP 时，`rankID` 是 TP×PP×PCP 内部 rank，DP 坐标单独传递。所有进程使用同一 rendezvous。

下列函数从 `aiter.dist.communication_op` 导入；前缀与后缀组合即完整名称。

| 前缀 | 支持的后缀 |
|---|---|
| `tensor_model_parallel_` | `all_reduce`、`all_gather`、`reduce_scatter`、`gather`，以及上文融合入口 |
| `data_parallel_` / `expert_parallel_` | `all_reduce`、`all_gather`、`reduce_scatter`、`gather`、`broadcast`、`broadcast_tensor_dict` |
| `pipeline_model_parallel_` | `all_reduce`、`all_gather`、`broadcast`、`broadcast_tensor_dict`、`send`、`recv` |
| `custom_` | `all_reduce`、`all_gather`、`reduce_scatter`，以 `group="name"` 选择组 |

AG 的常用签名为 `(input_, use_custom=False, dim=-1)`，RS 为 `(input_, use_custom=True, dim=0)`，AR 为 `(input_, use_new=True, open_fp8_quant=False)`。组级 AG 可通过 rank 维移动/reshape 支持中间维；这与直接 CAR 的首/末维限制不同。`use_custom=True` 请求自定义后端，仍可能按设备或输入条件回退，不能单凭此参数认定执行了某个 kernel。

`gather(input_, dst=0, dim=-1)` 只在目标 rank 返回 Tensor，其余返回 `None`；`broadcast(input_, src=0)` 原位修改并返回 input。src/dst 均为组内 rank。PP 的 `send(input_, dst=None)` / `recv(size, dtype, src=None)` 默认选择组内下一/上一 rank，并按组大小环绕。发送与接收必须由对应进程配对。

自定义组配置示例为 `{"pair": [[0,1],[2,3]]}`，由所有 rank 一致传入初始化；`get_tp_group/get_dp_group/get_ep_group/get_pp_group/get_dcp_group/get_pcp_group/get_custom_group` 位于 `aiter.dist.parallel_state`。分组布局为 ExternalDP×DP×PP×PCP×TP，EP 合并 DP/PCP/TP 维，DCP 在 TP 内分组。

## Graph 与输出生命周期

CAR 直接调用需在所有 rank 完成 JIT/预热后，嵌套 `with ca.capture():` 和 `with torch.cuda.graph(graph):` 捕获。使用 TP/DP 等组封装时使用对应 `GroupCoordinator.graph_capture()`。保持输入地址固定，用 `copy_` 更新内容；保留输出和通信对象直到 Graph 使用结束。

gfx936/gfx938 的高层 IPC communicator 即使收到 `enable_register_for_capturing=True`（或框架开启 `AITER_AR_ENABLE_REG_CAPTURE`），也会关闭直接注册并使用 capture 内 copy-in，避免读取历史输入；custom allreduce 和 Graph 仍启用。这两种架构的优化 one-stage kernel 还包含结束同步，防止后续拷贝过早覆盖其他 rank 正在读取的 staging 输入。直接使用低层 registered-input 接口不会自动经过 Python 保护，调用方应使用高层 communicator。Fabric 的原有注册限制保持不变。

普通 AR 与部分 RMS/quant wrapper 在 `ca.capture()` 内但尚未开始实际 Graph capture 的预热阶段返回占位零值，不能当作模型输出消费；AG/RS 和 QK-RoPE 的该阶段执行真实通信。Iris 不支持 Graph。QR 和 INT4 的 Graph 约定见 [Quick AllReduce](quick_all_reduce.md)。

## 性能结果

下列性能表保留其标注日期的测量结果。普通 CAR 在 2026-10-10 合入 gfx936/gfx938 优化 one-stage 结束同步后的最新测量，见 [CustomAllreduce 性能](custom_allreduce.md#典型性能)。不同接口及测量条件的结果不能直接互换。

### AR + RMSNorm，BF16 TP2

测量日期 2026-10-08，DTK 26.04 / HIP 6.3 系列、PyTorch 2.11；gfx936 使用 bmz，gfx938 使用 nmz-2。接口为 `custom_fused_ar_rms`，`eps=1e-6`、默认两阶段、无 Gemma/量化、IPC copy-in。输入是精确可表示的常量：rank r 输入 `(r+1)/8`，residual=0.125，weight=1。

每个 Graph 包含100次操作；每样本先在计时区间外对齐 rank，记录 GPU event 每次均值，取两 rank 较慢值。200个样本全部保留，初始化、编译和 barrier 不计入。这是合成负载的稳态设备耗时，不是模型吞吐或单请求尾延迟。

| 架构 | 形状 | median μs | P95 μs | max μs | CV |
|---|---|---:|---:|---:|---:|
| gfx936 | `[2,7168]` | 16.486 | 16.594 | 16.733 | 0.31% |
| gfx936 | `[128,8192]` | 44.491 | 44.632 | 44.664 | 0.17% |
| gfx938 | `[2,7168]` | 19.305 | 19.680 | 23.366 | 1.82% |
| gfx938 | `[128,8192]` | 42.612 | 42.805 | 42.931 | 0.21% |

上表报告指定 Graph 提交粒度和逐样本对齐条件下的结果。通信 kernel 时间包含等待其他 rank 的时间，主机提交间隙会产生长尾；不能将这些长尾全部归为 BF16 运算成本，也不能将上表推广为请求级 P95。

### 其他接口

- HIP FP / FlyDSL INT4 / RCCL 对比见 [Quick AllReduce 性能](quick_all_reduce.md#性能结果)。
- AG/RS、融合接口与并行组的代表性专项结果见下文；Iris见独立文档。不同输入、精度和计时口径的结果不合并计算加速比。
- 完整模型精度/吞吐、长时间运行与 Fabric/跨节点新路径不在上述性能结论范围内。新增可选后端不会自动替换框架的原有通信选择。

接口实现：[CAR 对象](../../aiter/dist/device_communicators/custom_all_reduce.py)、[native 接口](../../aiter/ops/custom_all_reduce.py)、[组级封装](../../aiter/dist/communication_op.py)。

### AG/RS 与融合接口专项性能

日期2026-10-09，gfx936/bmz25（80 CU）与gfx938/nmz-2（72 CU），均使用物理卡6/7、TP2、PyTorch 2.11.0、HIP 6.3.26113、IPC。每组30个样本全部保留，每个样本取两rank较慢值；†表示CV>10%，不据其中位数承诺稳定收益。HIP测量覆盖FP16/BF16，Iris及组封装覆盖BF16；下表展示代表性结果。这些是代表性输入的接口成本，不是完整模型吞吐；TP4/8性能不由TP2外推。

下表为BF16、Graph内GPU event时间，每个Graph含16次接口调用。CAR使用copy-in，RMS/FP8保持`use_1stage=False`，由native按形状选择两阶段或split；MXFP4显式一阶段。初始化、编译、预热、参考计算和计时外rank对齐不计入，Python分发/分配不计入Graph重放时间；capture中实际需要的copy和kernel均在计时范围内。

输入采用确定性的周期数列加rank偏移，残差0.125；MXFP4使用覆盖E2M1边界的权重与恒定输入。padding实际输入宽度为H+16、输出H+32；packed输入有H+16行stride。QK的Q/K/V宽度固定512/256/256，head_dim=128、rotary_dim=64；MHC大形状宽度为4096。这些数据不用于不同数学功能之间的加速比比较。

| 接口 | 输入形状 | gfx936 median μs（CV） | gfx938 median μs（CV） |
|---|---|---:|---:|
| AG首维 | `4×2048` | 10.56（6.94%） | 11.87（3.17%） |
| AG首维 | `128×8192` | 34.21（1.12%） | 31.88（2.20%） |
| AG末维 | `4×2048` | 10.40（6.64%） | 12.56（7.17%） |
| AG末维 | `128×8192` | 34.34（1.47%） | 32.03（1.24%） |
| RS首维 | `4×2048` | 10.83（5.75%） | 13.43（33.00%）† |
| RS首维 | `128×8192` | 23.99（2.21%） | 23.28（2.85%） |
| RS末维 | `4×2048` | 11.12（7.57%） | 13.56（4.34%） |
| RS末维 | `128×8192` | 24.89（7.90%） | 23.94（2.33%） |
| RS中间维 | `4×2×1024` | 11.03（6.20%） | 12.96（13.10%）† |
| RS中间维 | `128×2×4096` | 24.77（2.04%） | 23.99（3.23%） |
| AR+RMSNorm | `4×2048` | 15.84（3.68%） | 18.94（4.78%） |
| AR+RMSNorm | `128×8192` | 45.40（1.85%） | 43.91（11.72%）† |
| AR+Gemma | `4×2048` | 15.85（4.14%） | 18.25（3.82%） |
| AR+Gemma | `128×8192` | 45.36（54.35%）† | 44.21（73.27%）† |
| AR+RMSNorm padding | `4×2064` | 16.63（29.19%）† | 18.52（5.70%） |
| AR+RMSNorm padding | `128×8208` | 45.19（1.37%） | 44.50（1.42%） |
| AR+RMSNorm末维切片 | `4×2048` | 16.83（3.05%） | 20.18（16.29%）† |
| AR+RMSNorm末维切片 | `128×8192` | 46.63（1.37%） | 44.44（1.73%） |
| per-token FP8+mirror | `4×2048` | 16.06（1.73%） | 17.95（4.50%） |
| per-token FP8+mirror | `128×8192` | 55.24（3.78%） | 50.72（1.72%） |
| per-token FP8+Gemma+mirror | `4×2048` | 16.10（5.86%） | 18.82（13.41%）† |
| per-token FP8+Gemma+mirror | `128×8192` | 55.49（8.77%） | 51.45（131.97%）† |
| per-group FP8+mirror | `4×2048` | 15.73（3.08%） | 18.09（149.41%）† |
| per-group FP8+mirror | `128×8192` | 54.98（0.75%） | 51.59（0.99%） |
| per-group FP8+转置scale+mirror | `4×2048` | 15.65（2.25%） | 17.71（3.45%） |
| per-group FP8+转置scale+mirror | `128×8192` | 54.90（0.95%） | 52.25（8.90%） |
| MXFP4+mirror（一阶段） | `4×2048` | 15.08（2.56%） | 17.69（11.17%）† |
| MXFP4+mirror（一阶段） | `128×8192` | 44.89（5.49%） | 47.66（5.88%） |
| QK-Norm+RoPE | `4×1024` | 15.97（2.45%） | 18.32（2.35%） |
| QK-Norm+RoPE | `128×1024` | 24.00（3.29%） | 27.99（15.23%）† |
| MHC自动 | `4×2048` | 13.52（2.85%） | 15.72（2.64%） |
| MHC自动 | `128×4096` | 29.47（2.06%） | 31.88（3.01%） |
| MHC一阶段 | `4×2048` | 13.36（1.85%） | 15.65（1.00%） |
| MHC一阶段 | `128×4096` | 29.43（1.92%） | 32.52（1.83%） |
| MHC split | `4×2048` | 14.48（1.40%） | 17.22（23.70%）† |
| MHC split | `128×4096` | 35.54（2.66%） | 35.85（2.60%） |

整数AG的专项结果（int64，包含大于2^40的整数，逐位核对）：

| 接口 | 输入形状 | gfx936 median μs（CV） | gfx938 median μs（CV） |
|---|---|---:|---:|
| int64 AG末维 | `4×2048` | 11.90（5.32%） | 14.24（4.78%） |
| int64 AG末维 | `128×8192` | 106.51（0.65%） | 94.15（2.96%） |

### 并行组封装性能

以下为BF16、两rank的完整eager同步调用墙钟时间，包含Python分发、必要分配及最终设备同步；AR和broadcast还包含输入clone以保护后续迭代。AG/RS显式`use_custom=False`走组级RCCL路径，TP AR允许CAR选择；DP/EP/PP和自定义组按各自实际路由执行。每个样本只调用一次，与上面的Graph设备时间分开看。

| 接口 | 输入形状 | gfx936 median μs（CV） | gfx938 median μs（CV） |
|---|---|---:|---:|
| TP AR | `4×2048` | 198.71（32.74%）† | 202.69（8.71%） |
| TP AR | `128×8192` | 195.05（2.68%） | 205.31（3.11%） |
| TP AG | `4×2048` | 182.76（4.18%） | 204.93（6.56%） |
| TP AG | `128×8192` | 180.18（43.72%）† | 201.00（3.14%） |
| TP RS | `4×2048` | 209.96（3.60%） | 233.30（3.15%） |
| TP RS | `128×8192` | 204.86（10.28%）† | 234.41（8.69%） |
| DP AR | `4×2048` | 143.99（34.20%）† | 146.89（36.54%）† |
| DP AR | `128×8192` | 184.08（2.82%） | 184.06（2.37%） |
| DP AG | `4×2048` | 206.49（5.07%） | 209.08（4.26%） |
| DP AG | `128×8192` | 208.08（78.25%）† | 206.37（69.79%）† |
| DP RS | `4×2048` | 234.55（4.54%） | 243.39（93.28%）† |
| DP RS | `128×8192` | 235.31（20.80%）† | 237.33（19.91%）† |
| DP broadcast | `4×2048` | 173.32（3.14%） | 165.95（15.98%）† |
| DP broadcast | `128×8192` | 165.50（4.53%） | 170.44（44.89%）† |
| DP gather | `4×2048` | 220.18（12.15%）† | 215.57（54.58%）† |
| DP gather | `128×8192` | 215.06（54.30%）† | 214.96（5.60%） |
| EP AR | `4×2048` | 150.76（73.04%）† | 157.22（3.73%） |
| EP AR | `128×8192` | 186.42（44.13%）† | 183.32（2.46%） |
| EP AG | `4×2048` | 204.16（5.18%） | 207.34（14.83%）† |
| EP AG | `128×8192` | 204.11（2.35%） | 201.06（1.82%） |
| EP RS | `4×2048` | 237.52（2.49%） | 234.56（2.50%） |
| EP RS | `128×8192` | 233.47（3.34%） | 235.74（63.63%）† |
| PP AR | `4×2048` | 143.47（50.74%）† | 146.07（25.37%）† |
| PP AR | `128×8192` | 178.34（47.16%）† | 172.13（4.29%） |
| PP AG | `4×2048` | 183.23（3.16%） | 184.33（4.89%） |
| PP AG | `128×8192` | 180.94（8.25%） | 180.47（47.78%）† |
| PP broadcast | `4×2048` | 140.95（6.23%） | 140.41（32.64%）† |
| PP broadcast | `128×8192` | 150.11（73.25%）† | 157.77（2.22%） |
| 自定义组 AR | `4×2048` | 141.61（4.31%） | 153.56（9.64%） |
| 自定义组 AR | `128×8192` | 177.52（2.35%） | 180.38（500.93%）† |
| 自定义组 AG | `4×2048` | 184.45（4.90%） | 208.08（4.73%） |
| 自定义组 AG | `128×8192` | 181.29（78.29%）† | 202.69（2.98%） |
| 自定义组 RS | `4×2048` | 209.86（4.47%） | 236.65（3.09%） |
| 自定义组 RS | `128×8192` | 209.58（2.99%） | 232.32（2.69%） |
| PP send/recv配对 | `4×2048` | 110.47（4.68%） | 114.69（4.97%） |
| PP send/recv配对 | `128×8192` | 111.40（3.69%） | 112.73（2.58%） |

EP broadcast/gather也完成了同矩阵测量。对象字典传输和组初始化/销毁属于控制面，没有给出其吞吐数据。当前结果覆盖代表性shape与选项，不代表全部维度、量化组大小、Graph模式和输入分布的性能最优配置。
