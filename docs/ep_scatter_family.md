<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
-->

# ep_scatter 算子族说明

`ep_scatter` 算子族面向 MoE 专家并行（EP）的 dispatch 排列阶段：把 topk 路由后的 token 激活重排成 DeepGEMM 风格的 grouped GEMM 布局（按专家聚簇的行 + `m_indices` 组编号 + `inv_perm` 逆排列），并可在重排中融合 int8 / FP8 / smooth-int8 量化；`ep_gather` 在分专家计算后按 topk 权重把行加权汇聚还原成 token 序。全族共 6 个公开接口，声明于 `aiter/ops/moe_op.py`，经 JIT 模块 `module_moe_utils` 惰性编译加载（`from aiter.ops.moe_op import *` 亦使其可在 `aiter` 顶层直接导入）。

## 算子功能

计算流程：路由（上游 topk）→ 每 expert 计数 → 对齐 padding + exclusive scan 得各 expert 段起始偏移 → 各 (token, k) 以 atomicAdd 在本 expert 段内认领一行，记录 `inv_perm` 与 `m_indices` → 拷贝 / 量化写出激活行；分专家 GEMM 之后由 `ep_gather` 按 `inv_perm` 取回并加权求和。

### 符号表

| 符号 | 含义 |
|---|---|
| `M` | token 数（num_tokens） |
| `H` | hidden size，每个 token 激活行的宽度 |
| `K` | 每个 token 路由的专家数（topk） |
| `E` | local_num_experts，本卡本地专家数，即分组段数 |
| `A` | alignment，每个 expert 段的对齐粒度 |
| `count_e` | 路由到本地专家 `e` 的有效 (token, k) 个数（由调用方统计） |
| `M_sum` | `Σ_e align(count_e)`，输出行缓冲的总行数 |

### 布局定义

```text
padded_e  = (count_e + A - 1) & ~(A - 1)        # 位掩码对齐，A 必须为 2 的幂
start_e   = Σ_{e' < e} padded_e'                 # exclusive scan
M_sum     = Σ_e padded_e
行归属：[start_e, start_e + count_e) 为有效行；[start_e + count_e, start_e + padded_e) 为 padding
inv_perm[t, k] = 本算子为 (t, k) 认领的行号；无效路由写 -1
m_indices[row] = 该行所属的本地专家 id；padding 行的取值随算子变体（见下表）
```

### 算子总表

| 算子 | 激活输入 → 输出 | scale 粒度 | 内部 scan | m_indices 的 padding |
|---|---|---|---|---|
| `ep_scatter` | int8（已量化）→ int8 | 每 token（透传） | 有，原位改写计数 | 预填 -1 |
| `ep_fused_quant_scatter` | fp16/bf16 → int8 | 每 token | 有，原位改写计数 | 不写，保留调用方预填值 |
| `ep_fused_fp8_quant_scatter` | fp16/bf16 → e4m3/e5m2 | 每 token | 有，原位改写计数 | `fill_padded_m_indices=True` 填专家 id，False 填 -1 |
| `ep_fused_smooth_quant_scatter` | fp16/bf16 → int8 | 每 (token, expert) | 无，吃预扫描 start 偏移 | 不写，保留调用方预填值 |
| `ep_build_m_indices` | —（仅 topk_ids） | — | 内部自建直方图 + scan | padding 与尾部均填 -1 |
| `ep_gather` | fp16/bf16/fp32 → 同 dtype | —（加权求和） | 无 | — |

### 量化与汇聚公式

int8（`ep_fused_quant_scatter`，每 token 一次，复制到该 token 认领的每一行）：

```text
smax    = max_j |x[t, j]|                    # fp32
inv_s   = 127 / smax                          # __fdiv_rn 正确舍入除法
scale   = 1 / inv_s                           # 再做一次 __fdiv_rn（两级除法，保证可复现）
q[t, j] = clamp(round_rne(x[t, j] * inv_s), -128, 127)
输出：aq_out[row, :] = q，aq_scale_out[row] = scale
```

FP8（`ep_fused_fp8_quant_scatter`，`fp8type`：0 = e4m3，1 = e5m2）：

```text
fp8_max = 448（e4m3）/ 57344（e5m2）
scale   = max(smax / fp8_max, 1 / (fp8_max * 512))    # 有下限保护，全零行安全
q[t, j] = fp8_rne(x[t, j] / scale)
```

smooth int8（`ep_fused_smooth_quant_scatter`，逐 (token, k) 独立量化）：

```text
s(t,k)_j   = x[t, j] * smooth[e(t,k), j]     # fp32 逐元素，e 为映射后的本地专家
mx         = max_j |s(t,k)_j|，下限 clamp 至 1e-6
scale(t,k) = mx / 127
q(t,k)_j   = clamp(round_rne(s(t,k)_j * (127 / mx)), -128, 127)
```

汇聚（`ep_gather`）：

```text
output[t, j] = Σ_k w[t, k] * a[inv_perm[t, k], j]
求和仅覆盖满足（映射后专家 id ≥ 0 且 inv_perm[t, k] ≥ 0）的 k；fp32 按 k 升序累加，最后一次性 cast 回输入 dtype
输出形状：[M, H]，类型与 a 相同
```

### 边界语义

- 无效路由 `(t, k)` 不认领行，`inv_perm[t, k] = -1`；判定见「expert_map 语义」小节。
- 行认领用 `atomicAdd`，**专家段内的行顺序不确定**（取决于原子操作竞态）；正确性校验必须 order-agnostic（通过 `inv_perm` 取行比对）。
- 全部 (t, k) 无效时 `M_sum = 0`、`inv_perm` 全 -1，各 scatter 变体正常返回（有测试覆盖）。
- 空输入早退：`ep_scatter`（M=0）、`ep_fused_fp8_quant_scatter`（M=0）、`ep_build_m_indices`（任一输入 numel=0）在检查前直接 return；`ep_fused_quant_scatter` 与 `ep_fused_smooth_quant_scatter` 无 M=0 早退，行为未定义且未测试。
- `ep_build_m_indices` 统计时忽略 `topk_ids` 中 `[0, E)` 以外的 id；`m_indices.numel()` 大于 `M_sum` 时尾部一并填 -1。

### 负向范围

> **本族不做 softmax、topk 选择或 topk 权重 renormalize；`topk_ids` / `topk_weights` 由上游路由产生。**

> **除 `ep_build_m_indices` 内部自建直方图外，本族不生成 per-expert 计数：调用方必须预先统计 `expert_num_tokens`（`ep_fused_smooth_quant_scatter` 还须预先扫描出 start 偏移）。**

> **`expert_num_tokens` 是破坏性输入：scan 原位把 counts 改写为段起始偏移，scatter 阶段再用 atomicAdd 继续递增；调用返回后原 counts 丢失，重复调用 / Graph 回放前必须重灌。**

> **全部输出张量（aq_out、aq_scale_out、m_indices、inv_perm、output）由调用方预分配；接口不检查其容量，也不提供 autograd。**

> **本族不检查各张量的设备一致性与 `expert_num_tokens` 内容正确性；counts 与 topk_ids 不一致时结果静默错误。**

### 易错点

> **`ep_fused_smooth_quant_scatter` 的 `expert_offsets` 参数是预扫描的 start 偏移（exclusive scan），不是 counts——传 counts 会静默错位。**

> **`ep_fused_quant_scatter` 与 `ep_fused_smooth_quant_scatter` 不写 m_indices 的 padding 行：需要 -1 填充语义时必须先自行预填。**

> **alignment 必须是 2 的幂：布局用位掩码 `(c + A - 1) & ~(A - 1)` 计算，接口只检查 `alignment > 0`，非 2 的幂会产生错误布局。**

> **int8 融合量化（`ep_fused_quant_scatter`）对全零行无下限保护：smax = 0 时走 0·∞ = NaN 路径，行为未在测试中覆盖，调用方应避免。**

## 接口与参数

6 个接口全部为位置参数、无默认值、无返回值（结果写入传入的输出张量）。`expert_map` 传 `None` 表示无 EP 映射（TP 场景），此时 `topk_ids` 直接作为本地专家 id，要求落在 `[0, E)`。

```python
from aiter.ops.moe_op import (
    ep_scatter,
    ep_gather,
    ep_build_m_indices,
    ep_fused_quant_scatter,
    ep_fused_fp8_quant_scatter,
    ep_fused_smooth_quant_scatter,
)

ep_scatter(
    aq,                    # [M, H] int8
    aq_scale,              # [M] 或 [M, 1] fp32
    topk_ids,              # [M, K] int64
    expert_map,            # Optional[int32]，-1 = 丢弃
    expert_num_tokens,     # [E] int32 counts（输入）/ 段起始偏移（被原位改写）
    aq_out,                # [M_sum, H] int8
    aq_scale_out,          # [M_sum, 1] fp32
    m_indices,             # [M_sum] int32
    inv_perm,              # [M, K] int32
    local_num_experts, alignment,
)

ep_gather(
    a,                     # [M_sum, H] fp16/bf16/fp32
    topk_ids,              # [M, K] int64
    topk_weights,          # [M, K] fp32
    inv_perm,              # [M, K] int32
    expert_map,            # Optional[int32]
    output,                # [M, H]，dtype 与 a 相同
)

ep_build_m_indices(
    topk_ids,              # [M, K] int32 或 int64
    m_indices,             # [M_sum] int32
    local_num_experts, alignment,
)

ep_fused_quant_scatter(
    input,                 # [M, H] fp16/bf16
    topk_ids, expert_map, expert_num_tokens,
    aq_out,                # [M_sum, H] int8
    aq_scale_out,          # [M_sum, 1] fp32
    m_indices, inv_perm,
    local_num_experts, alignment,
)

ep_fused_fp8_quant_scatter(
    input,                 # [M, H] fp16/bf16
    topk_ids, expert_map, expert_num_tokens,
    aq_out,                # [M_sum, H] 单字节（fp8 位模式）
    aq_scale_out, m_indices, inv_perm,
    local_num_experts, alignment,
    fp8type,               # 0 = e4m3，1 = e5m2
    fill_padded_m_indices, # True: padding 填专家 id；False: 填 -1
)

ep_fused_smooth_quant_scatter(
    input,                 # [M, H] fp16/bf16
    topk_ids, expert_map,
    expert_offsets,        # [E] int32，预扫描 start 偏移（非 counts）
    smooth_scale,          # [E, H] fp32
    aq_out,                # [M_sum, H] int8
    aq_scale_out,          # [M_sum, 1] fp32
    m_indices, inv_perm,
    local_num_experts, alignment,
)
```

### 公共参数

| 参数 | 类型 / 形状 | 含义与约束 |
|---|---|---|
| `topk_ids` | `int64 [M, K]`（build 兼容 int32） | 路由结果；scatter 族要求 `K ≤ 8`，smooth 变体进一步要求 `K ∈ {4, 8}`；负值 = 丢弃 |
| `expert_map` | 可选 `int32 [E_global]` | 全局专家 id → 本地 id；`-1` = 本卡不持有（丢弃）；传 `None` 时 `topk_ids` 必须已落在 `[0, E)` |
| `expert_num_tokens` | `int32 [E]` | 输入每 expert 计数；**被原位破坏**（scan 改写 + atomicAdd 递增），长度须 ≥ `E` |
| `m_indices` | `int32 [M_sum]` | 行 → 本地专家 id，供 DeepGEMM 风格 masked grouped GEMM 分组；padding 语义见算子总表 |
| `inv_perm` | `int32 [M, K]` | (t, k) → 认领行号，无效 = -1；供 `ep_gather` 取回 |
| `aq_out` / `aq_scale_out` | `[M_sum, H]` / `[M_sum, 1]` | 量化激活与 scale，行数须 ≥ `M_sum`（由调用方按布局公式分配） |
| `local_num_experts` | `int` | `1 ≤ E ≤ 1024`（EP_SCAN_MAX_EXPERTS） |
| `alignment` | `int` | 段对齐粒度，**必须为 2 的幂**，仅检查 `> 0` |

### 通用约束

- 所有张量须在 GPU 上且设备一致（接口不检查）；激活与 scale 的连续性在 `ep_scatter` / fp8 变体中被显式检查，`ep_gather` 对 `a` 做 `expect_contiguous`，其余张量（含 `topk_ids`、`expert_map`、`output`）的连续性由调用方保证。
- scatter 族要求 `H % 16 == 0`；`ep_gather` 要求 `H % (16 / sizeof(dtype))` 整除（fp16/bf16 为 8，fp32 为 4）。
- 输入 dtype：融合量化变体的 dispatch 覆盖 fp16/bf16/fp32/fp64，但**仅 fp16/bf16 有测试覆盖**；`ep_gather` 的 fp32 路径有测试。
- 推理用途，不提供 autograd；所有 kernel 在当前 stream 上启动，无主机同步、无内部设备分配。

### 检查边界

接口用 TORCH_CHECK 检查：关键 dtype（`topk_ids` int64、index 张量 int32、scale fp32、量化输出单字节、`smooth_scale` fp32）、`ep_scatter`/fp8 的 2D 与连续性、`H % 16`、`K ≤ 8`（smooth 另有 `K ∈ {4, 8}`）、`1 ≤ E ≤ 1024`、`fp8type ∈ {0, 1}`。`ep_gather` 除 `H % VEC` 外几乎没有主机检查，dtype 约束由 `data_ptr<T>` 间接强制。

为避免热路径主机同步，接口**不检查**：`expert_num_tokens` 内容与 `topk_ids` 的一致性、`expert_map` 的覆盖域与值域、输出缓冲容量（`M_sum` 上限）、设备一致性。**调用方必须保证 `expert_map` 的每个取值 ∈ {-1} ∪ [0, E)**——int8 融合与 smooth 的部分 kernel 对映射结果不做范围检查，越界值会导致 atomicAdd 越界写。

### expert_map 语义

安全契约（各变体行为一致）：`topk_ids ∈ [0, expert_map.numel())`，map 取值 ∈ {-1} ∪ [0, E)。在此契约之外，各 kernel 的查找前置条件不同：

| 算子 / kernel | map 查找前置条件 | 映射结果处理 |
|---|---|---|
| `ep_scatter`（fast 与 generic） | `id ≥ 0` 即查 map（id 超出 map 长度为越界读，调用方保证） | 映射值 ∉ [0, E) 丢弃 |
| `ep_fused_quant_scatter`、`ep_fused_fp8_quant_scatter`、smooth 的 fallback kernel | 先要求 `id ∈ [0, E)` 再查 map：**id ≥ E 直接丢弃，不查 map** | int8 变体与 smooth fallback 仅查 `== -1`，映射值 ≥ E 会越界写；fp8 变体检查完整 |
| smooth 的 low_reg / tk kernel | `id ≥ 0` 即查 map | 映射值 == -1 丢弃；映射值 ≥ E 会越界写 |
| `ep_gather` | `id ≥ 0` 即查 map | 映射值 < 0 或 `inv_perm < 0` 跳过该 k |

即：当全局专家数大于本地专家数（map 长度 > E）且高段 id 会映射回有效本地专家时，`ep_scatter` 与 smooth low_reg/tk/gather 会正常翻译，而融合量化变体把它们当无效丢弃——混用这些算子时必须让 `topk_ids` 的取值域与所用变体的前置条件匹配。

### kernel 分派边界

| 算子 | 条件 | kernel |
|---|---|---|
| `ep_scatter` | `M ≤ 4096` 且 `H ≤ 8192` | fast：1024 线程/block，2 token/block，行经 shared memory（2H 字节）中转，float4 直读 |
| | 其他 | generic：1024 线程 = 8 组 × 128，1 token/block，行经 shared memory（H 字节）中转 |
| `ep_gather` | `K ≤ 2` / `K ≤ 4` / 其他 | K_MAX = 2 / 4 / 8 模板实例 |
| | `H = 7168` 且 16-bit dtype | deepseek 分支：128 线程，grid_x = 7 |
| | `M ∈ [256, 512]` 且 `H = 4096` | 128 线程，grid_x = 4 |
| smooth | `H ∈ {1024, 2048, 4096, 7168, 8192, 16384}`（编译期分支，block = H/16 线程） | `M ≤ 128`：tk kernel，grid (K, M)，一个 block 一个 (t, k)；`M > 128`：low_reg kernel，grid (M+1)/2，2 token/block，输入行驻留寄存器跨 k 复用 |
| | 其他 `H`（仍需 % 16） | fallback：256 线程，1 token/block，逐 k 重读输入 |
| `ep_build_m_indices` | — | 单 block 1024 线程：smem 直方图 + BlockScan + 单调游标填充 |

## 使用方式

### 布局准备（各示例公共）

```python
import torch
from aiter.ops.moe_op import ep_scatter, ep_gather

M, H, K, E, align = 64, 2048, 8, 16, 64          # align 必须是 2 的幂
DEV = "cuda"
torch.manual_seed(0)

topk_ids = torch.randint(0, E, (M, K), dtype=torch.int64, device=DEV)
counts = torch.bincount(topk_ids.reshape(-1), minlength=E)   # 本族不代劳：计数由上游完成
padded = (counts + align - 1) // align * align
starts = padded.cumsum(0) - padded                # exclusive scan
M_sum = int(padded.sum())
```

### ep_scatter 与 ep_gather（已量化 int8 路径）

```python
aq = torch.randint(-128, 128, (M, H), dtype=torch.int8, device=DEV)
aq_scale = torch.rand(M, 1, dtype=torch.float32, device=DEV)
aq_out = torch.empty(M_sum, H, dtype=torch.int8, device=DEV)
aq_scale_out = torch.empty(M_sum, 1, dtype=torch.float32, device=DEV)
m_indices = torch.empty(M_sum, dtype=torch.int32, device=DEV)
inv_perm = torch.empty(M, K, dtype=torch.int32, device=DEV)

ep_scatter(aq, aq_scale, topk_ids, None, counts.to(torch.int32),
           aq_out, aq_scale_out, m_indices, inv_perm, E, align)

# 正确性核对（order-agnostic：段内行序是原子竞态结果，须经 inv_perm 取行）
valid = topk_ids >= 0
rows = inv_perm[valid].long()
assert torch.equal(aq_out[rows], aq.unsqueeze(1).expand(M, K, H)[valid])
assert torch.equal(aq_scale_out[rows].reshape(-1), aq_scale.expand(M, K)[valid])

# 分专家计算后加权还原；此处以 aq_out 的 fp16 视图充当分组计算结果做演示
mid = aq_out.to(torch.float16)
topk_weights = torch.rand(M, K, dtype=torch.float32, device=DEV)
out = torch.empty(M, H, dtype=torch.float16, device=DEV)
ep_gather(mid, topk_ids, topk_weights, inv_perm, None, out)
ref = (mid[inv_perm.long()].float() * topk_weights.unsqueeze(-1)).sum(1)
torch.testing.assert_close(out, ref.to(torch.float16), rtol=1e-2, atol=1e-3)
```

示例使用无丢弃的均匀路由，`valid` 恒真，`mid[inv_perm]` 才能直接索引；真实模型的 `topk_ids` 含丢弃项，参考实现须按 k 逐项过滤。断言仅用于正确性展示，不应纳入热路径计时。上例 `counts.to(torch.int32)` 传入后已被破坏，再次调用前需重新统计。

### ep_fused_fp8_quant_scatter（融合 e4m3 量化）

```python
from aiter.ops.moe_op import ep_fused_fp8_quant_scatter

x = (torch.randn(M, H, device=DEV) * 3).to(torch.bfloat16)
aq_out = torch.empty(M_sum, H, dtype=torch.uint8, device=DEV)   # fp8 位模式按单字节存放
aq_scale_out = torch.empty(M_sum, 1, dtype=torch.float32, device=DEV)
m_indices = torch.empty(M_sum, dtype=torch.int32, device=DEV)
inv_perm = torch.empty(M, K, dtype=torch.int32, device=DEV)

ep_fused_fp8_quant_scatter(x, topk_ids, None, counts.to(torch.int32),
                           aq_out, aq_scale_out, m_indices, inv_perm,
                           E, align, 0, True)   # fp8type=0 (e4m3)，padding 填专家 id

# 反量化误差上界：smax × 半个 fp8 ulp（e4m3 相对 ulp 2^-3），参考测试同式
rows = inv_perm[valid].long()
smax = x.float().abs().amax(1).unsqueeze(1).expand(M, K)[valid][:, None]
err = (aq_out[rows].view(torch.float8_e4m3fn).float() * aq_scale_out[rows]
       - x.float().unsqueeze(1).expand(M, K, H)[valid]).abs()
assert torch.all(err <= smax * 0.125 * 1.1)
```

换 e5m2 时 `fp8type=1`、视图换 `torch.float8_e5m2`、ulp 相对值换 0.25。`fill_padded_m_indices=True` 用于 padding 行也携带专家 id 的下游分组语义；需要 -1 跳过语义时传 False。

### ep_fused_smooth_quant_scatter（smooth int8，需预扫描偏移）

```python
from aiter.ops.moe_op import ep_fused_smooth_quant_scatter

smooth = (torch.rand(E, H, device=DEV) + 0.5).float()           # [E, H] fp32
aq_out = torch.empty(M_sum, H, dtype=torch.int8, device=DEV)
aq_scale_out = torch.empty(M_sum, 1, dtype=torch.float32, device=DEV)
m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device=DEV)  # 本算子不写 padding，需预填
inv_perm = torch.empty(M, K, dtype=torch.int32, device=DEV)

ep_fused_smooth_quant_scatter(x, topk_ids, None, starts.to(torch.int32),  # 注意：start 偏移而非 counts
                              smooth, aq_out, aq_scale_out, m_indices,
                              inv_perm, E, align)
```

本算子不做内部 scan：`starts` 是 exclusive scan 结果，`m_indices` 的 padding 保留预填值。`K` 只能取 4 或 8。

### Graph 捕获

```python
pristine_counts = counts.to(torch.int32).clone()    # 破坏性输入的干净母本
expert_num_tokens = torch.empty_like(pristine_counts)

def run():
    expert_num_tokens.copy_(pristine_counts)        # 计数重灌必须在图内完成
    ep_scatter(aq, aq_scale, topk_ids, None, expert_num_tokens,
               aq_out, aq_scale_out, m_indices, inv_perm, E, align)

run()  # 首次 JIT 必须在捕获之外完成
warm_stream = torch.cuda.Stream()
warm_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warm_stream):
    for _ in range(3):
        run()
torch.cuda.current_stream().wait_stream(warm_stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    captured_out = run()
graph.replay()
torch.cuda.synchronize()
```

捕获要求固定 shape 与地址、原地 `copy_` 更新、保持输入与捕获输出存活；**`expert_num_tokens` 的重灌必须作为图内节点**（回放不会重新执行 Python 侧的 clone），smooth 变体重灌的则是 start 偏移母本。首次 JIT 与预热在捕获和计时之外。

### 功能验证入口

从仓库根目录运行：

```bash
python -B -m pytest -q op_tests/test_ep_scatter_family.py            # 全族正确性
python -B -m pytest -q op_tests/test_ep_scatter_family.py -k fp8     # 按算子过滤
python op_tests/test_ep_scatter_family.py --bench                    # 正确性通过后附 CUDA-event 计时表
```

正确性容差：纯搬运算子（`ep_scatter`、`ep_build_m_indices`）与 int8 / smooth 量化（参考实现逐条镜像 kernel 的 fp32 运算序与 RNE 舍入）要求逐位一致；fp8 路径允许 ≤ 2% 的字节失配率（gfx938 硬件 cvt 与 torch cast 的舍入平局）并校验反量化误差 ≤ smax × ulp_rel × 1.1；`ep_gather` 用 rtol 1e-2 / atol 1e-3。

计时口径声明：测试文件的 `--bench` 是 CUDA event 计时（3 轮取最小，扣除计数重灌 clone 开销）的快速表。

## 实现与架构范围

全部实现位于 `csrc/kernels/moe_align_sum_kernels.cu` 的 `aiter::moe_permute` 命名空间（该文件同时承载 `moe_align_block_size` 等非本族算子），host 入口与 TORCH_CHECK 在同文件，pybind 注册经 `MOE_UTILS_PYBIND` 宏。JIT 模块 `module_moe_utils` 由 `moe_utils_pybind.cu` + `topk_softmax_kernels.cu` + `moe_align_sum_kernels.cu` 组成，以 `-DENABLE_FP8` 编译；首次调用触发惰性构建。

- **scan**：单 block 1024 线程，hipcub BlockScan 做 padding 后 exclusive scan，原位写回 `expert_num_tokens`；`ep_scatter` 与 fp8 变体用带填充版本顺带并行预填 `m_indices`（二分查找归属专家，替代逐专家串行填充）。
- **scatter**：行认领全部走 `atomicAdd`；量化行在 shared memory 暂存后按 k 复制写出，int8/fp8 融合变体每 token 只读一遍输入、算一次 max（warp shuffle + 跨 warp 两级归约）。
- **FP8 转换**：gfx938 用硬件指令 `__builtin_hcu_cvt_pk_fp8_f32`（e4m3）/ `__builtin_hcu_cvt_pk_bf8_f32`（e5m2）四值打包；其他架构走软件位级 RNE 转换，两路结果在舍入平局上可有 1 bit 差异（即 2% 字节失配容差的来源）。
- **smooth 三档分派**：小批量（M ≤ 128）用 tk kernel 把所有 (t, k) 暴露为独立 block；大批量用 low_reg kernel 让输入行驻留寄存器跨 k 复用（该算子 HBM 带宽受限，smooth 矩阵的重复读基本不命中 L2）；非编译期 H 走 fallback。

测试覆盖缺口（如实声明）：融合量化变体的 fp32/fp64 输入路径、int8 融合的全零行、`M = 0`、`expert_map` 值 ≥ E 的越界行为均未被测试覆盖，按上文约束规避；fp8 的 e5m2 路径测试少于 e4m3。本容器验证环境为 BW1100（gfx938）。

## gfx938 性能报告

以下对比原实现与本仓库 AITER 实现的 ep_scatter 算子族：**original** 表示作为对照的优化前实现，**aiter** 表示本仓库当前实现。AITER 源码取自 HEAD `08b66ee4`（2026-09-21），性能采样日期 **2026-09-23**。加速比 = original 耗时 / aiter 耗时，大于 1 表示 aiter 更快。本报告共 99 个配置（去重后），其中 86 个双方可比。

### 环境与计时口径

| 项目 | gfx938 |
|---|---|
| 产品型号 / 架构 | BW1100 / gfx938 |
| 计算单元（CU）数 | 64 |
| 测试卡 / 测试日期 | GPU0 / 2026-09-23 |
| 可见显存 | 147440 MiB |
| 软件 | PyTorch 2.10.0，HIP 6.3.26113 |
| 输入 | `topk_ids` int64，激活 bf16/fp16，量化输出 int8 / 单字节 fp8，权重/scale/smooth 表 fp32，计数 int32 |
| 调用方式 | 公开 Python 接口，输出预分配，`expert_map=None`，默认 kernel 分派 |

- 计时使用 `aiter.test_common.run_perftest`（torch.profiler 设备 kernel 时间）：预热 10 次后，profiler 记录 50 次调用，丢弃首轮，其余调用内全部 GPU kernel 的设备时间总和除以调用次数，得到**平均每次调用的设备 kernel 时间**。它不包含主机提交间隙与 Python 分发开销，**不是 CPU 墙钟耗时**。
- 两实现同进程、同卡、使用完全相同的输入张量，逐配置先后测量。
- scatter 族的 `expert_num_tokens` 是破坏性输入：计时闭包每次从预分配的干净副本池取一份新缓冲（池深 = 预热 + 迭代 + 8，覆盖 profiler 的内存探测预跑），计时窗口内不出现恢复 kernel；非破坏性算子直接传入闭包计时。
- JIT 编译、输入生成、正确性校验、H2D/D2H 传输均不进入计时；没有在 API 外预先替 aiter 完成其内部必要的数据转换（计数统计 / scan 由基准脚本显式构造后同等传入两侧）。
- 稳定性：bf16 基准网格整轮复测一遍，逐配置加速比的轮间漂移中位数 **0.09%**、最大 **1.07%**，均远小于下文 2% 的持平阈值。
- ±2% 内的小幅波动视为基本持平，不解读为稳定收益。

### Shape 与输入构造

`M` 为 token 数，`H` 为 hidden size，`K` 为 topk，`E` 为本地专家数，`align` 为段对齐粒度。配置共三个来源，去重后合计 99 个：

1. **基准网格**（脚本 `--all-configs`，bf16，30 个）：scatter `(M,H,K,E,align)` 8 项——(512,2048,8,64,128)、(2048,4096,8,128,128)、(5000,2048,8,64,128)、(1024,7168,8,256,128)、(4096,7168,8,256,128)、(8192,4096,8,128,128)、(16384,2048,4,64,128)、(256,16384,2,8,64)；gather `(M,H,K)` 4 项——(2048,4096,8)、(4096,7168,8)、(300,4096,2)、(16384,2048,8)；build `(M,K,E,align)` 4 项——(2048,8,256,128)、(4096,8,128,64)、(512,4,32,64)、(8192,8,256,128)；fused_quant 4 项——(2048,4096,8,128,128)、(4096,7168,8,256,128)、(512,2048,4,32,64)、(8192,2048,8,64,128)；fp8_quant 4 项——(2048,4096,8,128,128) 的 e4m3 fill=False 与 e5m2 fill=True、(1024,7168,8,256,128) e4m3 fill=True、(4096,7168,8,256,128) e4m3 fill=False；smooth_quant 6 项——(2048,4096,8,128,128)、(1024,7168,8,256,128)、(4096,7168,8,256,128)、(512,2048,4,32,64)、(129,1024,4,16,64)、(97,5120,4,16,64)。
2. **fp16 复测**（18 个）：gather、fused_quant、fp8_quant、smooth_quant 四算子在基准网格上以 fp16 激活重测（scatter 输入为 int8、build 输入为 int64 id，与激活 dtype 无关，不重测）。
3. **扩展 shape**（9 组 shape × 6 算子 = 54 次测量，去重后 51 个，bf16）：(8192,7168,8,256,128)、(16384,4096,8,128,128)、(4096,2048,4,32,64)、(100,4096,8,16,64)、(2048,8192,8,64,128)、(100,2048,8,32,128)、(4096,5120,8,64,128)、(2048,16384,8,64,128)、(2048,4096,8,512,128)。三处重复已去除：gather 的 (2048,4096,8) 与基准网格重复；build 的 (2048,8,64,128) 在 (2048,8192) 与 (2048,16384) 两组 shape 间重复、(8192,8,256,128) 与基准网格重复。

各算子逐 shape 表即上述三个来源的合集（fp16 行以 dtype 列区分）。扩展 shape 的选取目的：小 M（M=100，含 smooth M≤128 的 tk kernel 分支）、deepseek 规格大 M（8192×7168）、超大 M（16384）、`K=4`、`H=5120` fallback 分支、`H=8192/16384` 边界、`E=512` 多专家。

输入构造（两侧完全相同）：`topk_ids` 均匀随机分布于 `[0, E)`（固定种子），无丢弃项；per-expert 计数由 `topk_ids` 统计并按 `align` 向上对齐、exclusive scan 得 start 偏移，`M_sum` 由实际路由决定；融合量化与 gather 的激活 `x = randn×3` 转目标 dtype；`ep_scatter` 的 int8 行均匀分布于 [-128, 128)；scale、gather 权重、smooth 表均取 U[0,1)（smooth 表 +0.5）；fp8 的 e4m3/e5m2 与 fill 变体在 dtype 列标注。基准负载为合成算子测试负载，不是线上请求回放。

### 常规汇总

耗时单位 **μs**（三位小数），加速比保留三位小数；汇总用未舍入值计算：

| 算子 | 可比配置数 | 快/平/慢 | 几何平均加速比 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| `ep_scatter` | 15 | 13/2/0 | 1.088× | 0.985× | 1.253× |
| `ep_gather` | 16 | 0/0/16 | 0.906× | 0.826× | 0.948× |
| `ep_build_m_indices` | 11 | 11/0/0 | 1.448× | 1.136× | 1.899× |
| `ep_fused_quant_scatter` | 14 | 1/1/12 | 0.939× | 0.910× | 1.048× |
| `ep_fused_fp8_quant_scatter` | 16 | 16/0/0 | 1.274× | 1.049× | 1.513× |
| `ep_fused_smooth_quant_scatter` | 14 | 3/0/11 | 1.077× | 0.821× | 2.717× |
| 全族 | 86 | 44/3/39 | 1.095× | 0.821× | 2.717× |

### ep_scatter 逐 shape 结果

| `M` | `H` | `K` | `E` | `align` | original (μs) | aiter (μs) | 加速比 | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 2048 | 8 | 32 | 128 | 12.080 | 9.640 | 1.253× | |
| 100 | 4096 | 8 | 16 | 64 | 13.121 | 12.106 | 1.084× | |
| 512 | 2048 | 8 | 64 | 128 | 22.901 | 19.394 | 1.181× | |
| 1024 | 7168 | 8 | 256 | 128 | 80.945 | 66.339 | 1.220× | |
| 2048 | 4096 | 8 | 128 | 128 | 100.465 | 90.669 | 1.108× | |
| 2048 | 4096 | 8 | 512 | 128 | 224.446 | 196.958 | 1.140× | |
| 2048 | 8192 | 8 | 64 | 128 | 127.495 | 120.541 | 1.058× | |
| 4096 | 2048 | 4 | 32 | 64 | 66.235 | 62.816 | 1.054× | |
| 4096 | 5120 | 8 | 64 | 128 | 162.571 | 150.839 | 1.078× | |
| 4096 | 7168 | 8 | 256 | 128 | 227.126 | 206.280 | 1.101× | |
| 5000 | 2048 | 8 | 64 | 128 | 169.021 | 165.281 | 1.023× | |
| 8192 | 4096 | 8 | 128 | 128 | 369.154 | 362.757 | 1.018× | 持平 |
| 8192 | 7168 | 8 | 256 | 128 | 484.888 | 474.578 | 1.022× | |
| 16384 | 2048 | 4 | 64 | 128 | 433.373 | 439.790 | 0.985× | 持平 |
| 16384 | 4096 | 8 | 128 | 128 | 709.333 | 689.395 | 1.029× | |

### ep_gather 逐 shape 结果

| `M` | `H` | `K` | dtype | original (μs) | aiter (μs) | 加速比 |
|---:|---:|---:|---|---:|---:|---:|
| 100 | 2048 | 8 | bf16 | 6.859 | 8.306 | 0.826× |
| 100 | 4096 | 8 | bf16 | 10.072 | 11.423 | 0.882× |
| 300 | 4096 | 2 | bf16 | 8.959 | 10.112 | 0.886× |
| 300 | 4096 | 2 | fp16 | 8.082 | 9.013 | 0.897× |
| 2048 | 4096 | 8 | bf16 | 98.974 | 104.884 | 0.944× |
| 2048 | 4096 | 8 | fp16 | 98.222 | 103.617 | 0.948× |
| 2048 | 8192 | 8 | bf16 | 175.715 | 193.720 | 0.907× |
| 2048 | 16384 | 8 | bf16 | 334.643 | 372.247 | 0.899× |
| 4096 | 2048 | 4 | bf16 | 60.381 | 63.937 | 0.944× |
| 4096 | 5120 | 8 | bf16 | 224.384 | 262.689 | 0.854× |
| 4096 | 7168 | 8 | bf16 | 314.032 | 341.279 | 0.920× |
| 4096 | 7168 | 8 | fp16 | 307.333 | 329.055 | 0.934× |
| 8192 | 7168 | 8 | bf16 | 622.144 | 673.208 | 0.924× |
| 16384 | 2048 | 8 | bf16 | 350.036 | 386.036 | 0.907× |
| 16384 | 2048 | 8 | fp16 | 346.082 | 372.629 | 0.929× |
| 16384 | 4096 | 8 | bf16 | 671.412 | 739.202 | 0.908× |

### ep_build_m_indices 逐 shape 结果

| `M` | `K` | `E` | `align` | original (μs) | aiter (μs) | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 8 | 16 | 64 | 10.063 | 6.999 | 1.438× |
| 100 | 8 | 32 | 128 | 15.373 | 9.532 | 1.613× |
| 512 | 4 | 32 | 64 | 15.449 | 9.599 | 1.609× |
| 2048 | 8 | 64 | 128 | 32.704 | 23.536 | 1.390× |
| 2048 | 8 | 256 | 128 | 91.878 | 51.292 | 1.791× |
| 2048 | 8 | 512 | 128 | 173.562 | 91.411 | 1.899× |
| 4096 | 4 | 32 | 64 | 23.117 | 19.650 | 1.176× |
| 4096 | 8 | 64 | 128 | 41.145 | 34.021 | 1.209× |
| 4096 | 8 | 128 | 64 | 60.098 | 41.844 | 1.436× |
| 8192 | 8 | 256 | 128 | 116.342 | 81.488 | 1.428× |
| 16384 | 8 | 128 | 128 | 118.039 | 103.878 | 1.136× |

### ep_fused_quant_scatter 逐 shape 结果

| `M` | `H` | `K` | `E` | `align` | dtype | original (μs) | aiter (μs) | 加速比 | 备注 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 100 | 2048 | 8 | 32 | 128 | bf16 | 14.475 | 13.808 | 1.048× | |
| 100 | 4096 | 8 | 16 | 64 | bf16 | 15.898 | 16.726 | 0.950× | |
| 2048 | 4096 | 8 | 128 | 128 | bf16 | 134.038 | 143.771 | 0.932× | |
| 2048 | 4096 | 8 | 128 | 128 | fp16 | 134.592 | 143.805 | 0.936× | |
| 2048 | 4096 | 8 | 512 | 128 | bf16 | 210.897 | 212.561 | 0.992× | 持平 |
| 2048 | 8192 | 8 | 64 | 128 | bf16 | 170.976 | 185.569 | 0.921× | |
| 2048 | 16384 | 8 | 64 | 128 | bf16 | 281.264 | 292.634 | 0.961× | |
| 4096 | 5120 | 8 | 64 | 128 | bf16 | 271.258 | 296.850 | 0.914× | |
| 4096 | 7168 | 8 | 256 | 128 | bf16 | 318.224 | 348.275 | 0.914× | |
| 4096 | 7168 | 8 | 256 | 128 | fp16 | 319.398 | 347.831 | 0.918× | |
| 8192 | 2048 | 8 | 64 | 128 | bf16 | 397.020 | 430.516 | 0.922× | |
| 8192 | 2048 | 8 | 64 | 128 | fp16 | 396.641 | 431.340 | 0.920× | |
| 8192 | 7168 | 8 | 256 | 128 | bf16 | 620.040 | 681.203 | 0.910× | |
| 16384 | 4096 | 8 | 128 | 128 | bf16 | 959.763 | 1048.948 | 0.915× | |

### ep_fused_fp8_quant_scatter 逐 shape 结果

dtype 列标注 fp8 格式（e4m3/e5m2）与 `fill_padded_m_indices` 取值。

| `M` | `H` | `K` | `E` | `align` | dtype | original (μs) | aiter (μs) | 加速比 | 备注 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 100 | 2048 | 8 | 32 | 128 | bf16/e4m3 fill=False | 15.825 | 13.645 | 1.160× | |
| 100 | 4096 | 8 | 16 | 64 | bf16/e4m3 fill=False | 17.228 | 14.666 | 1.175× | |
| 1024 | 7168 | 8 | 256 | 128 | bf16/e4m3 fill=True | 136.935 | 130.045 | 1.053× | |
| 1024 | 7168 | 8 | 256 | 128 | fp16/e4m3 fill=True | 135.928 | 129.608 | 1.049× | |
| 2048 | 4096 | 8 | 128 | 128 | bf16/e4m3 fill=False | 178.493 | 135.116 | 1.321× | |
| 2048 | 4096 | 8 | 128 | 128 | bf16/e5m2 fill=True | 177.983 | 159.168 | 1.118× | |
| 2048 | 4096 | 8 | 128 | 128 | fp16/e4m3 fill=False | 177.986 | 135.603 | 1.313× | |
| 2048 | 4096 | 8 | 128 | 128 | fp16/e5m2 fill=True | 177.846 | 158.878 | 1.119× | |
| 2048 | 4096 | 8 | 512 | 128 | bf16/e4m3 fill=False | 257.924 | 212.239 | 1.215× | |
| 2048 | 8192 | 8 | 64 | 128 | bf16/e4m3 fill=False | 244.292 | 168.325 | 1.451× | |
| 2048 | 16384 | 8 | 64 | 128 | bf16/e4m3 fill=False | 412.407 | 272.498 | 1.513× | |
| 4096 | 5120 | 8 | 64 | 128 | bf16/e4m3 fill=False | 361.598 | 268.501 | 1.347× | |
| 4096 | 7168 | 8 | 256 | 128 | bf16/e4m3 fill=False | 457.823 | 314.810 | 1.454× | |
| 4096 | 7168 | 8 | 256 | 128 | fp16/e4m3 fill=False | 453.056 | 314.568 | 1.440× | |
| 8192 | 7168 | 8 | 256 | 128 | bf16/e4m3 fill=False | 894.175 | 612.020 | 1.461× | |
| 16384 | 4096 | 8 | 128 | 128 | bf16/e4m3 fill=False | 1295.844 | 966.143 | 1.341× | |

### ep_fused_smooth_quant_scatter 逐 shape 结果

| `M` | `H` | `K` | `E` | `align` | dtype | original (μs) | aiter (μs) | 加速比 | 备注 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 100 | 2048 | 8 | 32 | 128 | bf16 | 47.794 | 17.592 | 2.717× | tk kernel |
| 100 | 4096 | 8 | 16 | 64 | bf16 | 54.542 | 20.783 | 2.624× | tk kernel |
| 1024 | 7168 | 8 | 256 | 128 | bf16 | 260.606 | 317.258 | 0.821× | |
| 1024 | 7168 | 8 | 256 | 128 | fp16 | 260.788 | 316.653 | 0.824× | |
| 2048 | 4096 | 8 | 128 | 128 | bf16 | 263.352 | 270.556 | 0.973× | |
| 2048 | 4096 | 8 | 128 | 128 | fp16 | 263.032 | 270.516 | 0.972× | |
| 2048 | 4096 | 8 | 512 | 128 | bf16 | 311.097 | 326.696 | 0.952× | |
| 2048 | 8192 | 8 | 64 | 128 | bf16 | 531.327 | 549.379 | 0.967× | |
| 2048 | 16384 | 8 | 64 | 128 | bf16 | 1118.965 | 1250.037 | 0.895× | H=16384 编译分支 |
| 4096 | 5120 | 8 | 64 | 128 | bf16 | 1892.768 | 1315.417 | 1.439× | fallback kernel |
| 4096 | 7168 | 8 | 256 | 128 | bf16 | 928.679 | 1122.885 | 0.827× | |
| 4096 | 7168 | 8 | 256 | 128 | fp16 | 929.559 | 1122.515 | 0.828× | |
| 8192 | 7168 | 8 | 256 | 128 | bf16 | 1814.841 | 2201.516 | 0.824× | |
| 16384 | 4096 | 8 | 128 | 128 | bf16 | 1876.622 | 2031.465 | 0.924× | |

### 结果分析

`ep_build_m_indices` 几何平均 **1.448×**（11/11 配置领先，`E=512` 时最高 1.899×）；`ep_fused_fp8_quant_scatter` 几何平均 **1.274×**（16/16 领先，`H=16384` 时最高 1.513×）。`ep_scatter` 几何平均 **1.088×**，收益随 `M` 增大收窄（`M≥8192` 后回到持平区附近）。全族最大单项为 smooth 的 tk kernel 小批量分支（M=100 时 **2.717×**）。
`ep_gather` 全部 16 个配置落后（几何平均 **0.906×**，慢约 10%）；`ep_fused_quant_scatter` 14 个可比配置中 12 个落后（几何平均 **0.939×**）。smooth 双极分化：小 M tk kernel 领先 2.6–2.7×、`H=5120` fallback 领先 1.44×，但 `H=7168` 的 low_reg 大 M 路径一致落后（0.82–0.83×），把该算子几何平均压到 1.077×。

## 相关实现与测试入口

| 项目 | 固定版本 / 标识 |
|---|---|
| 算子族引入提交 | `0cb69999`（2026-09-18，feat(moe_utils): add ep_scatter_family） |
| 文档基线（AITER HEAD） | `08b66ee4cd1403fb7160e659bba14193f779d58e`（2026-09-21） |
| 性能采样日期 | 2026-09-23 |

- [Python 接口声明（JIT stub）](../aiter/ops/moe_op.py)
- [pybind 注册](../csrc/pybind/moe_utils_pybind.cu)
- [native 入口与全部 kernel](../csrc/kernels/moe_align_sum_kernels.cu)
- [C++ 声明](../csrc/include/moe_utils.h)
- [功能测试与 --bench 快速计时](../op_tests/test_ep_scatter_family.py)
